import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

import node_agent

VIDEO = bytes(range(256)) * 40


class FakePortal:
    """The node API of the portal, as much of it as the agent uses."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, object]] = []
        self.files: dict[str, bytes] = {}
        self.cut_video_at: int | None = None
        self.stopped = False
        portal = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def _answer(self, status: int, payload: object = None, body: bytes | None = None, headers=None) -> None:
                data = body if body is not None else json.dumps(payload).encode()
                self.send_response(status)
                for key, value in (headers or {}).items():
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                start = int(self.headers.get("Range", "bytes=0-")[6:].rstrip("-"))
                portal.calls.append(("GET", self.path, start))
                self.send_response(206 if start else 200)
                if start:
                    self.send_header("Content-Range", f"bytes {start}-{len(VIDEO) - 1}/{len(VIDEO)}")
                self.send_header("Content-Length", str(len(VIDEO) - start))
                self.end_headers()
                # A dropped connection: the body ends before its declared length.
                end = len(VIDEO) if portal.cut_video_at is None else portal.cut_video_at
                portal.cut_video_at = None
                self.wfile.write(VIDEO[start:end])
                self.close_connection = True

            def do_PUT(self) -> None:
                url = urlparse(self.path)
                query = parse_qs(url.query)
                body = self.rfile.read(int(self.headers["Content-Length"]))
                name = url.path.split("/files/", 1)[1]
                offset = int(query["offset"][0])
                portal.files[name] = portal.files.get(name, b"")[:offset] + body
                portal.calls.append(("PUT", name, (offset, query["done"][0])))
                self._answer(200, {"size": offset + len(body)})

            def do_POST(self) -> None:
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"null")
                portal.calls.append(("POST", self.path, payload))
                if portal.stopped:
                    self._answer(409, {"detail": "Job is cancelled"})
                else:
                    self._answer(200, {"success": True})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


@pytest.fixture
def portal():
    fake = FakePortal()
    yield fake
    fake.server.shutdown()


def test_upload_goes_in_pieces_and_download_resumes(portal, tmp_path: Path, monkeypatch):
    monkeypatch.setattr(node_agent.time, "sleep", lambda _: None)
    client = node_agent.Portal(portal.url, "t", upload_chunk=4000)
    source = tmp_path / "big.bin"
    source.write_bytes(VIDEO)
    client.upload("/api/nodes/jobs/j/files/predictions.jsonl.gz", source)
    assert portal.files["predictions.jsonl.gz"] == VIDEO
    assert [c[2] for c in portal.calls] == [(0, "false"), (4000, "false"), (8000, "true")]

    portal.calls.clear()
    portal.cut_video_at = 5000
    target = tmp_path / "video.mp4"
    client.download("/api/nodes/jobs/j/video?key=k", target, lambda: None)
    assert target.read_bytes() == VIDEO
    assert [c[2] for c in portal.calls] == [0, 5000]


def test_job_sends_rallies_as_found_then_results_and_completes(portal, tmp_path: Path, monkeypatch):
    seen = {}

    def pipeline(video, target_dir, action_model, on_rallies, on_progress):
        seen["video"] = Path(video).read_bytes()
        seen["court"] = json.loads((target_dir / "court.json").read_text())
        tracks = target_dir / "tracks"
        tracks.mkdir()
        on_rallies([], True)
        (tracks / "track_0000.json").write_text('{"n": 0}')
        on_rallies(["track_0000.json"], False)
        (tracks / "track_0000.json").write_text('{"n": 0, "typed": true}')
        (target_dir / "ball.csv").write_text("Frame\n")
        (target_dir / "predictions.jsonl.gz").write_bytes(b"gz")
        return {
            "predictions": target_dir / "predictions.jsonl.gz",
            "ball_csv": target_dir / "ball.csv",
            "tracks_dir": tracks,
            "error": None,
        }

    monkeypatch.setattr(node_agent, "run_video_pipeline", pipeline)
    agent = node_agent.NodeAgent(node_agent.Portal(portal.url, "t"), "n", 4, tmp_path / "work", stream=False)
    job = {"id": "j", "projectId": "p", "videoUrl": "/api/nodes/jobs/j/video?key=k", "videoName": "p.mp4", "court": {"c": 1}}
    agent.run_job(job)

    assert seen == {"video": VIDEO, "court": {"c": 1}}
    posts = [(c[1], c[2]) for c in portal.calls if c[0] == "POST"]
    assert posts == [
        ("/api/nodes/jobs/j/rallies", {"files": [], "reset": True}),
        ("/api/nodes/jobs/j/rallies", {"files": ["track_0000.json"], "reset": False}),
        ("/api/nodes/jobs/j/complete", {"error": None}),
    ]
    assert portal.files["tracks/track_0000.json"] == b'{"n": 0, "typed": true}'
    assert portal.files["predictions.jsonl.gz"] == b"gz"
    assert not (tmp_path / "work" / "p").exists()


def test_failed_job_is_reported_and_a_cancelled_one_is_not(portal, tmp_path: Path, monkeypatch):
    def failing(video, target_dir, action_model, on_rallies, on_progress):
        raise RuntimeError("detector died")

    monkeypatch.setattr(node_agent, "run_video_pipeline", failing)
    agent = node_agent.NodeAgent(node_agent.Portal(portal.url, "t"), "n", 4, tmp_path / "work", stream=True)
    job = {"id": "j", "projectId": "p", "videoUrl": "/v", "videoName": "p.mp4", "court": {}}
    agent.run_job(job)
    assert portal.calls[-1] == ("POST", "/api/nodes/jobs/j/complete", {"failed": True, "error": "detector died"})

    def cancelled(video, target_dir, action_model, on_rallies, on_progress):
        portal.stopped = True
        on_rallies([], True)

    portal.calls.clear()
    monkeypatch.setattr(node_agent, "run_video_pipeline", cancelled)
    agent.run_job(job)
    assert [c[1] for c in portal.calls] == ["/api/nodes/jobs/j/rallies"]
