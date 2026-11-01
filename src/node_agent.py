"""An outside video processor ("node") of the vb-ai portal, working over the public internet.

The node reports to the portal every ~30 s with its token; a free node may get a
processing job in the answer (the most powerful free node gets it first). The job
runs the same pipeline as the local Celery worker (``run_video_pipeline``):

- the video is downloaded by a URL with a job key (resumed after a dropped connection,
  checked against its size); ``VB_NODE_STREAM=true`` gives that URL straight to OpenCV's
  FFmpeg instead (HTTP range requests, no disk). Streaming reads the video twice over the
  network (detector, then classifier), and a connection lost for good ends the stream as
  the end of the video would: the rest of the match would be silently missing;
- track files go to the portal as rallies become final, so they appear while it runs;
- at the end the predictions, ``ball.csv``, the typed tracks and the detector log are
  uploaded and the portal imports them as it does for a local run.

A job cancelled on the portal (processing started again, given to another node) is
stopped on the next report. A node that stops reporting loses its job to another one.

    VB_PORTAL_URL=https://api.vb-ai.ru VB_NODE_TOKEN=vbn_... uv run python -m src.node_agent

Settings (env): ``VB_PORTAL_URL``, ``VB_NODE_TOKEN`` (from ``scripts/worker_nodes.py create``
of the API), ``VB_NODE_NAME`` (host name), ``VB_NODE_POWER`` (CPU count; raise it for a GPU
machine), ``VB_NODE_WORK_DIR`` (``node-work``), ``VB_NODE_STREAM`` (false),
``VB_NODE_UPLOAD_CHUNK_KB`` (900: under the 1 MB request limit of a default nginx); the detector
and classifier take the ``INFERENCE_*`` settings of the Celery worker.
"""

from __future__ import annotations

import http.client
import json
import logging
import os
import platform
import queue
import shutil
import socket
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

try:
    from .celery_worker import run_video_pipeline
    from .predictions_io import JSONL_NAME, read_meta
except ImportError:
    from celery_worker import run_video_pipeline
    from predictions_io import JSONL_NAME, read_meta

logger = logging.getLogger("node_agent")

# FFmpeg (inside OpenCV) goes on after a dropped connection instead of ending the video.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS", "reconnect;1|reconnect_streamed;1|reconnect_on_network_error;1|reconnect_delay_max;60"
)

# The detector and the classifier run as ``uv run src/...`` from the repository (``/app`` in docker).
os.environ.setdefault("INFERENCE_REPO_DIR", str(Path(__file__).resolve().parents[1]))

USER_AGENT = "vb-ai-node/1"


class JobStopped(Exception):
    """The job is no longer this node's: cancelled on the portal or given to another node."""


class PortalError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status


class Portal:
    def __init__(self, url: str, token: str, retries: int = 5, upload_chunk: int = 900 * 1024) -> None:
        self.url = url.rstrip("/")
        self.token = token
        self.retries = retries
        self.upload_chunk = upload_chunk

    def request(self, method: str, path: str, payload: Any = None, data: bytes | None = None, timeout: float = 60) -> Any:
        """JSON in and out (or raw ``data``); a dropped connection and a 5xx are tried again, a 409 means the job is gone."""
        headers = {"Authorization": f"Bearer {self.token}", "User-Agent": USER_AGENT}
        body = data
        if data is not None:
            headers["Content-Type"] = "application/octet-stream"
        elif payload is not None:
            headers["Content-Type"] = "application/json"
            body = json.dumps(payload).encode("utf-8")
        for attempt in range(1, self.retries + 1):
            try:
                request = urllib.request.Request(f"{self.url}{path}", data=body, method=method, headers=headers)
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    raw = response.read()
                    return json.loads(raw) if raw else None
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                if exc.code == 409:
                    raise JobStopped(detail) from exc
                if exc.code < 500 or attempt == self.retries:
                    raise PortalError(exc.code, detail) from exc
            except (http.client.HTTPException, OSError) as exc:
                if attempt == self.retries:
                    raise
                logger.warning("%s %s failed (%s), retrying", method, path, exc)
            time.sleep(min(2**attempt, 30))
        raise AssertionError("unreachable")

    def upload(self, path: str, source: Path) -> None:
        """``source`` in pieces of ``upload_chunk`` bytes: a proxy may limit the size of a request."""
        size = source.stat().st_size
        with source.open("rb") as handle:
            offset = 0
            while True:
                piece = handle.read(self.upload_chunk)
                done = offset + len(piece) >= size
                self.request("PUT", f"{path}?offset={offset}&done={str(done).lower()}", data=piece, timeout=120)
                offset += len(piece)
                if done:
                    return

    def download(self, path: str, target: Path, should_stop: Callable[[], None]) -> None:
        """The whole video into ``target``, going on from where a dropped download stopped.

        Gives up after ``retries`` failures in a row with nothing new downloaded.
        """
        attempt = 0
        reached = -1
        while attempt < self.retries:
            offset = target.stat().st_size if target.exists() else 0
            attempt = 1 if offset > reached else attempt + 1
            reached = offset
            headers = {"User-Agent": USER_AGENT}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            try:
                request = urllib.request.Request(f"{self.url}{path}", headers=headers)
                with urllib.request.urlopen(request, timeout=120) as response:
                    if offset and response.status != 206:
                        offset = 0  # the whole file came instead of the rest
                    # "bytes 100-199/200" for the rest, Content-Length for the whole file
                    content_range = response.headers.get("Content-Range", "")
                    total = int(content_range.rsplit("/", 1)[1]) if "/" in content_range else int(
                        response.headers.get("Content-Length") or -1
                    )
                    with target.open("r+b" if offset else "wb") as out:
                        out.seek(offset)
                        out.truncate()
                        while chunk := response.read(1 << 20):
                            out.write(chunk)
                            should_stop()
                if total < 0 or target.stat().st_size == total:
                    return
                logger.warning("Download stopped at %s of %s bytes", target.stat().st_size, total)
            except urllib.error.HTTPError as exc:
                if exc.code == 416 and offset:
                    return  # already whole
                if exc.code == 404:
                    raise JobStopped("video is no longer served") from exc
                if attempt == self.retries:
                    raise
            except (http.client.HTTPException, OSError) as exc:
                if attempt == self.retries:
                    raise
                logger.warning("Download failed (%s), resuming", exc)
            time.sleep(min(2**attempt, 30))
        raise RuntimeError(f"Video download did not complete: {target}")


class NodeAgent:
    def __init__(self, portal: Portal, name: str, power: float, work_dir: Path, stream: bool) -> None:
        self.portal = portal
        self.name = name
        self.power = power
        self.work_dir = work_dir
        self.stream = stream
        self.jobs: queue.Queue[dict[str, Any]] = queue.Queue()
        self.heartbeat_seconds = 30.0
        self._lock = threading.Lock()
        self._job: dict[str, Any] | None = None
        self._stage: str | None = None
        self._progress: float | None = None
        self._stop_job = threading.Event()

    # --- reports ---

    def _info(self) -> dict[str, Any]:
        return {
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "cpus": os.cpu_count(),
            "device": os.getenv("INFERENCE_DETECTOR_DEVICE", os.getenv("INFERENCE_DEVICE", "CPU")),
            "detector": Path(os.getenv("INFERENCE_MODEL_XML", "RAVEL-VB-011-9f.xml")).name,
        }

    def heartbeat(self) -> None:
        with self._lock:
            job_id = self._job["id"] if self._job else None
            payload = {
                "name": self.name,
                "power": self.power,
                "info": self._info(),
                "ready": job_id is None,
                "jobId": job_id,
                "stage": self._stage,
                "progress": self._progress,
            }
        answer = self.portal.request("POST", "/api/nodes/heartbeat", payload, timeout=30)
        self.heartbeat_seconds = float(answer.get("heartbeatSeconds") or self.heartbeat_seconds)
        with self._lock:
            if answer.get("cancel") and job_id and self._job and self._job["id"] == job_id:
                logger.warning("Job %s was taken back by the portal", job_id)
                self._stop_job.set()
            job = answer.get("job")
            if job and self._job is None:
                self._job = job
                self._stage, self._progress = "starting", None
                self._stop_job.clear()
                self.jobs.put(job)

    def _heartbeat_loop(self) -> None:
        while True:
            try:
                self.heartbeat()
            except Exception as exc:
                logger.warning("Heartbeat failed: %s", exc)
            time.sleep(self.heartbeat_seconds)

    def _set_stage(self, stage: str, progress: float | None = None) -> None:
        with self._lock:
            self._stage, self._progress = stage, progress

    # --- jobs ---

    def run_forever(self) -> None:
        threading.Thread(target=self._heartbeat_loop, name="heartbeat", daemon=True).start()
        logger.info("Node %s (power %s) reports to %s", self.name, self.power, self.portal.url)
        while True:
            job = self.jobs.get()
            try:
                self.run_job(job)
            finally:
                with self._lock:
                    self._job = None
                    self._stage = self._progress = None

    def _check_stop(self) -> None:
        if self._stop_job.is_set():
            raise JobStopped("cancelled on the portal")

    def run_job(self, job: dict[str, Any]) -> None:
        job_id = job["id"]
        base = f"/api/nodes/jobs/{job_id}"
        # Per project: a job given back to this node after a restart resumes the detection.
        target_dir = self.work_dir / job["projectId"]
        logger.info("Job %s: project %s", job_id, job["projectId"])
        # Committed track files the portal has not got yet (a window's failure is only logged by the
        # pipeline): sent with the next ones; the final call raises.
        unsent: list[str] = []

        def upload(path: Path, name: str) -> None:
            self.portal.upload(f"{base}/files/{name}", path)

        def on_rallies(file_names: list[str], reset: bool) -> None:
            self._check_stop()
            unsent.extend(file_names)
            for name in unsent:
                upload(target_dir / "tracks" / name, f"tracks/{name}")
            self.portal.request("POST", f"{base}/rallies", {"files": unsent, "reset": reset})
            unsent.clear()

        def on_progress() -> None:
            self._check_stop()
            predictions = target_dir / JSONL_NAME
            if predictions.is_file():
                try:
                    self._set_stage("detection", float(read_meta(predictions)["last_frame"]))
                except (OSError, ValueError, KeyError):
                    pass

        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "court.json").write_text(json.dumps(job["court"], ensure_ascii=False), encoding="utf-8")
            if self.stream:
                video = f"{self.portal.url}{job['videoUrl']}"
            else:
                self._set_stage("download")
                local_video = target_dir / f"video{Path(job['videoName']).suffix or '.mp4'}"
                self.portal.download(job["videoUrl"], local_video, self._check_stop)
                video = str(local_video)
            self._set_stage("detection")
            result = run_video_pipeline(video, target_dir, job.get("actionModel"), on_rallies, on_progress)
            self._check_stop()
            self._set_stage("upload")
            # The classifier wrote the action types into the track files: they go again.
            for path in sorted(result["tracks_dir"].glob("track_*.json")):
                upload(path, f"tracks/{path.name}")
            upload(result["ball_csv"], "ball.csv")
            if result["predictions"].name == "predictions.jsonl.gz":
                upload(result["predictions"], "predictions.jsonl.gz")
            self._upload_log(target_dir, base)
            self.portal.request("POST", f"{base}/complete", {"error": result["error"]})
            logger.info("Job %s done%s", job_id, f" (classifier: {result['error']})" if result["error"] else "")
        except JobStopped as exc:
            logger.warning("Job %s stopped: %s", job_id, exc)
        except Exception as exc:
            logger.exception("Job %s failed", job_id)
            self._upload_log(target_dir, base)
            try:
                self.portal.request("POST", f"{base}/complete", {"failed": True, "error": str(exc)[-2000:]})
            except Exception:
                logger.exception("Failed to report job %s", job_id)
        shutil.rmtree(target_dir, ignore_errors=True)

    def _upload_log(self, target_dir: Path, base: str) -> None:
        log = target_dir / "detector.log"
        if log.is_file():
            try:
                self.portal.upload(f"{base}/files/detector.log", log)
            except Exception as exc:
                logger.warning("Failed to upload the detector log: %s", exc)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    url = os.getenv("VB_PORTAL_URL", "").strip()
    token = os.getenv("VB_NODE_TOKEN", "").strip()
    if not url or not token:
        raise SystemExit("VB_PORTAL_URL and VB_NODE_TOKEN are required")
    agent = NodeAgent(
        Portal(url, token, upload_chunk=int(float(os.getenv("VB_NODE_UPLOAD_CHUNK_KB", "900")) * 1024)),
        name=os.getenv("VB_NODE_NAME") or socket.gethostname(),
        power=float(os.getenv("VB_NODE_POWER") or os.cpu_count() or 1),
        work_dir=Path(os.getenv("VB_NODE_WORK_DIR", "node-work")).resolve(),
        stream=os.getenv("VB_NODE_STREAM", "false").lower() in {"1", "true", "yes", "on"},
    )
    agent.run_forever()


if __name__ == "__main__":
    main()
