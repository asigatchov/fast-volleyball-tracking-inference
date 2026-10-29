import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse
import json
from typing import Any, Callable

from celery import Celery
import cv2

try:
    from .hot_tracks import HotTracks
    from .jersey_numbers import NumberReader
    from .player_reid import VECTORS_FILE, ReidEncoder, assign_players
    from .make_reels import crop_and_save_track_payload, crop_and_save_track_payloads
    from .predictions_io import JSONL_NAME, find_predictions, iter_frames, pack, read_meta
except ImportError:
    from hot_tracks import HotTracks
    from jersey_numbers import NumberReader
    from player_reid import VECTORS_FILE, ReidEncoder, assign_players
    from make_reels import crop_and_save_track_payload, crop_and_save_track_payloads
    from predictions_io import JSONL_NAME, find_predictions, iter_frames, pack, read_meta

logger = logging.getLogger(__name__)

DEFAULT_REDIS_URL = "redis://redis:6379/0"
BROKER_URL = os.getenv("CELERY_BROKER_URL", os.getenv("REDIS_URL", DEFAULT_REDIS_URL))
RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", os.getenv("REDIS_URL", DEFAULT_REDIS_URL))

celery_app = Celery("vb_ai_inference_worker", broker=BROKER_URL, backend=RESULT_BACKEND)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)
API_IMPORT_TASK_NAME = os.getenv("API_IMPORT_TASK_NAME", "api.import_rallies_from_tracks")
API_SET_STATUS_TASK_NAME = os.getenv("API_SET_STATUS_TASK_NAME", "api.set_project_status")
API_UPDATE_REEL_TASK_NAME = os.getenv("API_UPDATE_REEL_TASK_NAME", "api.update_reel_result")
API_REFRESH_ACTIONS_TASK_NAME = os.getenv("API_REFRESH_ACTIONS_TASK_NAME", "api.refresh_actions_from_tracks")
API_ADD_RALLIES_TASK_NAME = os.getenv("API_ADD_RALLIES_TASK_NAME", "api.add_rallies_from_tracks")
API_IMPORT_PLAYERS_TASK_NAME = os.getenv("API_IMPORT_PLAYERS_TASK_NAME", "api.import_player_assignment")
# Rallies are added while the detector runs (see hot_tracks.py); off: once at the end.
INFERENCE_HOT_TRACKS = os.getenv("INFERENCE_HOT_TRACKS", "true").lower() in {"1", "true", "yes", "on"}
# A detector killed (out of memory, a restart) goes on from its last checkpoint this many times.
INFERENCE_MAX_ATTEMPTS = max(int(os.getenv("INFERENCE_MAX_ATTEMPTS", "3")), 1)
# How often the growing predictions are looked at.
HOT_POLL_SECONDS = float(os.getenv("INFERENCE_HOT_POLL_SECONDS", "15"))
API_IMPORT_TASK_QUEUE = os.getenv("API_IMPORT_TASK_QUEUE", "api-import")
API_IMPORT_REPLACE_EXISTING = os.getenv("API_IMPORT_REPLACE_EXISTING", "true").lower() in {"1", "true", "yes", "on"}
# Action classifiers a task may ask for by name (see _resolve_action_model_xml); the API offers the same list.
ACTION_MODELS = {"action_clf_slowfast_r50_verified", "action_clf_r2plus1d18_verified"}

celery = celery_app
app = celery_app


def _resolve_video_fps(video_path: str | Path) -> float:
    default_fps = float(os.getenv("TRACK_VIDEO_FPS", "30"))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return default_fps
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    return fps if fps > 0 else default_fps


def _resolve_video_size(video_path: str | Path) -> tuple[int, int]:
    default_width = int(os.getenv("TRACK_VIDEO_WIDTH", "1920"))
    default_height = int(os.getenv("TRACK_VIDEO_HEIGHT", "1080"))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return default_width, default_height

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if width <= 0 or height <= 0:
        return default_width, default_height
    return width, height


def _predictions_json_to_ball_csv(predictions_path: Path, csv_path: Path) -> None:
    """Convert the RAVEL-VB predictions (JSONL or JSON) into the ``Frame,Visibility,X,Y,Radius`` CSV.

    With ``--frame-step N`` the network only sees every N-th frame. Skipped frames
    are filled by linear interpolation when both neighbouring shown frames have a
    ball, otherwise they are written as invisible.
    """
    meta = read_meta(predictions_path)
    frame_step = max(int(meta.get("frame_step") or 1), 1)
    total_frames = int(meta.get("processed_frames") or 0)

    balls: dict[int, tuple[float, float, float]] = {}
    for frame, items in iter_frames(predictions_path):
        best_score = -1.0
        for item in items:
            if item.get("class_name") != "ball":
                continue
            score = float(item.get("score", 0.0))
            if score <= best_score:
                continue
            x1, y1, x2, y2 = (float(v) for v in item["bbox_xyxy"])
            radius = max(x2 - x1, y2 - y1) / 2
            balls[frame] = ((x1 + x2) / 2, (y1 + y2) / 2, radius)
            best_score = score

    if not total_frames and balls:
        total_frames = max(balls) + 1

    rows = ["Frame,Visibility,X,Y,Radius"]
    for frame in range(total_frames):
        ball = balls.get(frame)
        offset = frame % frame_step
        if ball is None and offset:
            prev_ball = balls.get(frame - offset)
            next_ball = balls.get(frame - offset + frame_step)
            if prev_ball is not None and next_ball is not None:
                t = offset / frame_step
                ball = tuple(p + (n - p) * t for p, n in zip(prev_ball, next_ball))
        if ball is None:
            rows.append(f"{frame},0,-1,-1,0")
        else:
            x, y, radius = ball
            rows.append(f"{frame},1,{int(round(x))},{int(round(y))},{int(round(radius))}")

    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _queue_status_update(project_id: str, user_id: str, status: str) -> None:
    try:
        celery_app.send_task(
            API_SET_STATUS_TASK_NAME,
            args=[project_id, user_id, status],
            queue=API_IMPORT_TASK_QUEUE,
        )
    except Exception:
        logger.exception("Failed to enqueue status update for project=%s user=%s status=%s", project_id, user_id, status)


def _resolve_action_model_xml(repo_dir: Path, action_model: str | None) -> Path:
    """The classifier .xml for a run: the requested model, or the env/default one when none is given."""
    if action_model:
        if action_model not in ACTION_MODELS:
            raise ValueError(f"Unknown action model: {action_model}")
        model_xml = (repo_dir / "ov" / f"{action_model}.xml").resolve()
        ov_dir = (repo_dir / "ov").resolve()
        if model_xml.parent != ov_dir:
            raise ValueError(f"Action model escapes the ov directory: {action_model}")
        return model_xml
    return Path(
        os.getenv("INFERENCE_ACTION_MODEL_XML", str(repo_dir / "ov" / "action_clf_r2plus1d18_verified.xml"))
    ).resolve()


def _run_contact_classification(
    repo_dir: Path,
    video_path: str | Path,
    predictions_json: Path,
    tracks_dir: Path,
    device: str,
    action_model: str | None = None,
) -> None:
    """Label the detected contacts (Serve / Receive / Set / Attack) in the track files."""
    model_xml = _resolve_action_model_xml(repo_dir, action_model)
    if not model_xml.exists():
        raise FileNotFoundError(f"Action classifier not found: {model_xml}")
    classify_cmd = [
        "uv",
        "run",
        "src/classify_contacts.py",
        str(video_path),
        "--players_json_path",
        str(predictions_json),
        "--tracks_dir",
        str(tracks_dir),
        "--model",
        str(model_xml),
        "--device",
        device,
    ]
    logger.info("Running contact classification command: %s", " ".join(classify_cmd))
    try:
        subprocess.run(classify_cmd, cwd=repo_dir, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Contact classification failed: {exc.stderr[-2000:] if exc.stderr else str(exc)}"
        ) from exc


@celery_app.task(name="inference.classify_actions")
def classify_actions(project_id: str, user_id: str, file_path: str, action_model: str | None = None) -> dict[str, str | None]:
    """Run the action classifier alone over the contacts of the last detection.

    The detector output (predictions.json, tracks) is reused; the API then re-types the
    actions of the existing rallies. It is told about a failure too: it has to release the project.
    """
    video_path = Path(file_path).resolve()
    target_video_dir = video_path.parent / video_path.stem
    error: str | None = None
    try:
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        predictions_json = find_predictions(target_video_dir)
        tracks_dir = target_video_dir / "tracks"
        if predictions_json is None or not tracks_dir.is_dir():
            raise FileNotFoundError(f"Detection results not found in {target_video_dir}")
        repo_dir = Path(os.getenv("INFERENCE_REPO_DIR", "/app")).resolve()
        _run_contact_classification(repo_dir, video_path, predictions_json, tracks_dir, os.getenv("INFERENCE_DEVICE", "CPU"), action_model)
    except Exception as exc:
        logger.exception("Actions classification failed for project=%s user=%s", project_id, user_id)
        error = str(exc)

    celery_app.send_task(
        API_REFRESH_ACTIONS_TASK_NAME,
        args=[project_id, user_id, error],
        kwargs={"context": "classification"},
        queue=API_IMPORT_TASK_QUEUE,
    )
    return {"project_id": project_id, "user_id": user_id, "error": error}


@celery_app.task(name="inference.assign_players")
def assign_players_task(
    project_id: str,
    user_id: str,
    file_path: str,
    actions: list[dict[str, Any]],
    k: int = 12,
    side: str = "near",
    run_id: str = "latest",
    read_numbers: bool = True,
    candidates: list[str] | None = None,
) -> dict[str, Any]:
    """Sort the actions of one half into players (src/player_reid.py).

    With ``read_numbers`` the jersey numbers are read (src/jersey_numbers.py, pose + PP-OCRv6)
    and become the players, limited to ``candidates`` when given. The actions are also grouped
    by appearance (re-id) into ``k`` groups, which only order the unsorted ones for the user:
    without numbers read no action gets a player.

    ``actions`` come from the API (``frame``, ``track_id``, ``ball``, ``side``); the result is
    ``<project>/players/<run_id>/players.json`` with a crop per action (a directory per run, so
    the browser never shows the crops of an earlier run), and the API is told to read it (or
    about the error).
    """
    video_path = Path(file_path).resolve()
    target_video_dir = video_path.parent / video_path.stem
    error: str | None = None
    try:
        predictions = find_predictions(target_video_dir)
        court_path = target_video_dir / "court.json"
        if predictions is None or not court_path.is_file():
            raise FileNotFoundError(f"Detection results or court markup not found in {target_video_dir}")
        repo_dir = Path(os.getenv("INFERENCE_REPO_DIR", "/app")).resolve()
        model_xml = Path(os.getenv("INFERENCE_REID_MODEL_XML", str(repo_dir / "ov" / "person-reidentification-retail-0287.xml")))
        result = assign_players(
            str(video_path),
            str(predictions),
            json.loads(court_path.read_text(encoding="utf-8")),
            actions,
            target_video_dir / "players" / Path(run_id).name,
            ReidEncoder(model_xml, os.getenv("INFERENCE_DEVICE", "CPU")),
            k=k,
            side=side,
            numbers=NumberReader(device=os.getenv("INFERENCE_DEVICE", "CPU")) if read_numbers else None,
            candidates=candidates or None,
        )
        logger.info(
            "Players assigned project=%s (%s): %d actions, %d skipped",
            project_id, result["mode"], len(result["assignments"]), len(result["skipped"]),
        )
    except Exception as exc:
        logger.exception("Player assignment failed for project=%s user=%s", project_id, user_id)
        error = str(exc)

    celery_app.send_task(
        API_IMPORT_PLAYERS_TASK_NAME,
        args=[project_id, user_id, run_id, error],
        queue=API_IMPORT_TASK_QUEUE,
    )
    return {"project_id": project_id, "user_id": user_id, "error": error}


@celery_app.task(name="inference.player_vectors")
def player_vectors_task(
    project_id: str,
    user_id: str,
    file_path: str,
    actions: list[dict[str, Any]],
    side: str = "near",
    run_id: str = "latest",
) -> dict[str, Any]:
    """Re-id vectors of the actors of an earlier run (made before runs kept them), no number reading.

    Writes ``<project>/players/<run_id>/vectors.npz`` (or ``vectors_error.txt``) and removes the
    API's ``vectors_pending`` marker; the API ranks actions by likeness from it.
    """
    video_path = Path(file_path).resolve()
    target_video_dir = video_path.parent / video_path.stem
    run_dir = target_video_dir / "players" / Path(run_id).name
    scratch = run_dir / "_vectors"
    error: str | None = None
    try:
        predictions = find_predictions(target_video_dir)
        court_path = target_video_dir / "court.json"
        if predictions is None or not court_path.is_file():
            raise FileNotFoundError(f"Detection results or court markup not found in {target_video_dir}")
        repo_dir = Path(os.getenv("INFERENCE_REPO_DIR", "/app")).resolve()
        model_xml = Path(os.getenv("INFERENCE_REID_MODEL_XML", str(repo_dir / "ov" / "person-reidentification-retail-0287.xml")))
        assign_players(
            str(video_path),
            str(predictions),
            json.loads(court_path.read_text(encoding="utf-8")),
            actions,
            scratch,
            ReidEncoder(model_xml, os.getenv("INFERENCE_DEVICE", "CPU")),
            side=side,
        )
        (scratch / VECTORS_FILE).replace(run_dir / VECTORS_FILE)
        (run_dir / "vectors_error.txt").unlink(missing_ok=True)
    except Exception as exc:
        logger.exception("Player vectors failed for project=%s user=%s", project_id, user_id)
        error = str(exc)
        if run_dir.is_dir():
            (run_dir / "vectors_error.txt").write_text(error[:2000], encoding="utf-8")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        (run_dir / "vectors_pending").unlink(missing_ok=True)
    return {"project_id": project_id, "user_id": user_id, "error": error}


def _queue_rallies(project_id: str, user_id: str, file_names: list[str], reset: bool = False) -> None:
    """Hand committed track files to the API; ``reset`` first drops the rallies of an earlier run."""
    if not file_names and not reset:
        return
    celery_app.send_task(
        API_ADD_RALLIES_TASK_NAME,
        args=[project_id, user_id, file_names, reset],
        queue=API_IMPORT_TASK_QUEUE,
    )


def _completed_detection(target_dir: Path, model_xml: Path, frame_step: str) -> Path | None:
    """Predictions of a finished detection with this model and sampling, to be used again."""
    path = find_predictions(target_dir)
    if path is None:
        return None
    try:
        meta = read_meta(path)
    except (OSError, ValueError):
        return None
    same = meta.get("model") == str(model_xml) and str(meta.get("frame_step")) == str(frame_step)
    return path if same and meta.get("complete") else None


def _resumable_detection(predictions: Path, model_xml: Path, video_path: str | Path, frame_step: str) -> bool:
    """An unfinished JSONL of the same run: the detector goes on from its last checkpoint."""
    if not predictions.is_file():
        return False
    try:
        meta = read_meta(predictions)
    except (OSError, ValueError):
        return False
    return (
        not meta.get("complete")
        and meta.get("model") == str(model_xml)
        and meta.get("video") == str(video_path)
        and str(meta.get("frame_step")) == str(frame_step)
    )


def _run_detector(
    detector_cmd: list[str], repo_dir: Path, log_path: Path, on_progress, poll_seconds: float
) -> None:
    """Runs the detector, resuming it after a crash; ``on_progress`` is called while it works.

    What ``on_progress`` raises kills the detector and ends the run.
    """
    for attempt in range(1, INFERENCE_MAX_ATTEMPTS + 1):
        with log_path.open("ab") as log:
            process = subprocess.Popen(detector_cmd, cwd=repo_dir, stdout=log, stderr=log)
            try:
                while process.poll() is None:
                    time.sleep(poll_seconds)
                    if process.poll() is None:
                        on_progress()
            except BaseException:
                # The run is stopped (cancelled, the worker shuts down): the detector goes with it.
                process.kill()
                process.wait()
                raise
        if process.returncode == 0:
            return
        logger.warning(
            "Detector exited with %s (attempt %d of %d)", process.returncode, attempt, INFERENCE_MAX_ATTEMPTS
        )
    tail = log_path.read_bytes()[-2000:].decode("utf-8", errors="replace") if log_path.exists() else ""
    raise RuntimeError(f"OpenVINO inference failed with exit code {process.returncode}: {tail}")


def is_url(video: str) -> bool:
    return urlparse(video).scheme in {"http", "https"}


def run_video_pipeline(
    video: str,
    target_video_dir: Path,
    action_model: str | None,
    on_rallies: Callable[[list[str], bool], None],
    on_progress: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Detection, then rallies with their touches, then the types of the actions.

    ``video`` is a local file or an http(s) URL (an outside node reads the portal's
    video over the network); everything goes to ``target_video_dir``, which holds
    ``court.json``. The detector streams its predictions into ``predictions.jsonl``.
    While it runs, the rallies that are already final are found window by window and
    handed to ``on_rallies(file_names, reset)``; the end line of the JSONL is the flag
    that the last frame is done. A run that died goes on from where it stopped; a
    finished detection with the same model is not repeated, only the rallies are found
    again. ``on_progress`` is called while the detector works; what it raises stops
    the detector.

    Raises when the detection fails; a failed classifier is only reported in ``error``.
    """
    if not is_url(video):
        video = str(Path(video).resolve())
        if not Path(video).exists():
            raise FileNotFoundError(f"Video file not found: {video}")

    repo_dir = Path(os.getenv("INFERENCE_REPO_DIR", "/app")).resolve()
    if not repo_dir.exists():
        raise FileNotFoundError(f"Inference repo not found: {repo_dir}")

    model_xml = Path(
        os.getenv(
            "INFERENCE_MODEL_XML",
            str(repo_dir / "ov" / "RAVEL-VB-011-9f.xml"),
        )
    ).resolve()
    if not model_xml.exists():
        raise FileNotFoundError(f"Detector model not found: {model_xml}")
    # An .onnx release runs on ONNX Runtime (CUDA in the GPU image), an .xml on OpenVINO.
    detector_script = (
        "src/inference_player_ball_onnx.py"
        if model_xml.suffix == ".onnx"
        else "src/inference_player_ball_openvino.py"
    )

    device = os.getenv("INFERENCE_DEVICE", "CPU")
    # The action classifier stays on OpenVINO (CPU only); the detector may run elsewhere.
    detector_device = os.getenv("INFERENCE_DETECTOR_DEVICE", device)
    frame_step = os.getenv("INFERENCE_FRAME_STEP", "2")
    target_video_dir.mkdir(parents=True, exist_ok=True)

    # Rally detection needs the court homography: do not start without markup.
    court_json = target_video_dir / "court.json"
    if not court_json.exists():
        raise FileNotFoundError(f"Court markup is required before processing: {court_json}")

    video_width, video_height = _resolve_video_size(video)
    fps = _resolve_video_fps(video)
    beach = os.getenv("INFERENCE_BEACH", "false").lower() in {"1", "true", "yes", "on"}
    tracks_dir = target_video_dir / "tracks"
    predictions = target_video_dir / JSONL_NAME

    def run_window(frame_start: int, frame_end: int, window_dir: Path) -> None:
        track_cmd = [
            "uv",
            "run",
            "src/track_calculator_with_court.py",
            "--players_json_path",
            str(predictions_in_use),
            "--court_json_path",
            str(court_json),
            "--output_dir",
            str(target_video_dir.parent),
            "--tracks_dir",
            str(window_dir),
            "--frame_start",
            str(frame_start),
            "--frame_end",
            str(frame_end),
            "--video_width",
            str(video_width),
            "--video_height",
            str(video_height),
            "--fps",
            str(fps),
        ]
        if beach:
            track_cmd.append("--beach")
        logger.info("Running track command: %s", " ".join(track_cmd))
        try:
            subprocess.run(track_cmd, cwd=repo_dir, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Track calculator failed: {exc.stderr[-2000:] if exc.stderr else str(exc)}"
            ) from exc

    hot = HotTracks(target_video_dir, fps, run_window)
    reused = _completed_detection(target_video_dir, model_xml, frame_step)
    resumed = reused is None and _resumable_detection(predictions, model_xml, video, frame_step)
    predictions_in_use = reused or predictions
    if not resumed:
        # A new run: the rallies of the earlier one go as soon as it starts.
        hot.reset()
        on_rallies([], True)
    if reused is None:
        # The detection is redone: earlier predictions would be taken for its result.
        for stale in (target_video_dir / "predictions.jsonl.gz", target_video_dir / "predictions.json"):
            stale.unlink(missing_ok=True)
    logger.info(
        "Processing %s: detection %s",
        target_video_dir,
        "reused" if reused else "resumed" if resumed else "started",
    )

    def commit_ready() -> None:
        """Called while the detector runs: adds the rallies that are already final."""
        if on_progress is not None:
            on_progress()
        if not INFERENCE_HOT_TRACKS or not predictions.is_file():
            return
        try:
            last_frame = read_meta(predictions)["last_frame"]
            if hot.due(last_frame):
                on_rallies(hot.advance(last_frame), False)
        except Exception:
            # The final pass at the end covers whatever a window missed.
            logger.exception("Hot track window failed for %s", target_video_dir)

    if reused is None:
        detector_cmd = [
            "uv",
            "run",
            detector_script,
            video,
            "--model",
            str(model_xml),
            "--output",
            str(predictions),
            "--resume",
            "--frame-step",
            frame_step,
            "--device",
            detector_device,
        ]
        logger.info("Running inference command: %s", " ".join(detector_cmd))
        _run_detector(detector_cmd, repo_dir, target_video_dir / "detector.log", commit_ready, HOT_POLL_SECONDS)
    meta = read_meta(predictions_in_use)
    if not meta.get("complete"):
        raise RuntimeError(f"Predictions are not complete: {predictions_in_use}")
    normalized_csv = target_video_dir / "ball.csv"
    _predictions_json_to_ball_csv(predictions_in_use, normalized_csv)
    on_rallies(hot.advance(int(meta["last_frame"]), final=True), False)

    # The actions are typed by the classifier over all the rallies at once.
    error: str | None = None
    if os.getenv("INFERENCE_CLASSIFY_ACTIONS", "true").lower() in {"1", "true", "yes", "on"}:
        try:
            _run_contact_classification(repo_dir, video, predictions_in_use, tracks_dir, device, action_model)
        except Exception as exc:
            logger.exception("Actions classification failed for %s", target_video_dir)
            error = str(exc)
    if predictions_in_use == predictions:
        # Nothing reads the plain file any more: about a third of it gzipped.
        try:
            predictions_in_use = pack(predictions)
        except OSError:
            logger.exception("Failed to pack %s", predictions)
    return {
        "predictions": predictions_in_use,
        "ball_csv": normalized_csv,
        "tracks_dir": tracks_dir,
        "error": error,
    }


@celery_app.task(name="inference.process_uploaded_video")
def process_uploaded_video(
    project_id: str, user_id: str, file_path: str, file_url: str, action_model: str | None = None
) -> dict[str, str | bool]:
    """The pipeline (``run_video_pipeline``) over a video of the shared uploads directory.

    The rallies go to the API as they are found; at the end the API re-types the
    touches it added and releases the project.
    """
    parsed = urlparse(file_url)
    if parsed.scheme not in {"http", "https"}:
        _queue_status_update(project_id, user_id, "new")
        raise ValueError("file_url must start with http:// or https://")

    video_path = Path(file_path).resolve()
    try:
        result = run_video_pipeline(
            str(video_path),
            video_path.parent / video_path.stem,
            action_model,
            lambda file_names, reset: _queue_rallies(project_id, user_id, file_names, reset),
        )
    except Exception:
        _queue_status_update(project_id, user_id, "new")
        raise

    celery_app.send_task(
        API_REFRESH_ACTIONS_TASK_NAME,
        args=[project_id, user_id, result["error"]],
        kwargs={"context": "detection"},
        queue=API_IMPORT_TASK_QUEUE,
    )
    return {
        "project_id": project_id,
        "user_id": user_id,
        "file_path": str(video_path),
        "file_url": file_url,
        "exists": video_path.exists(),
        "predictions": str(result["predictions"]),
        "ball_csv": str(result["ball_csv"]),
        "tracks_dir": str(result["tracks_dir"]),
        "tracks_exists": result["tracks_dir"].exists(),
    }


@celery_app.task(name="inference.make_reels")
def make_reels_task(
    project_id: str,
    user_id: str,
    file_path: str,
    file_url: str,
    reels: list[dict[str, Any]],
    uploads_dir: str,
) -> dict[str, int | str]:
    parsed = urlparse(file_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("file_url must start with http:// or https://")

    video_path = Path(file_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    uploads_root = Path(uploads_dir).resolve()
    reels_dir = uploads_root / user_id / project_id / "reels"
    reels_dir.mkdir(parents=True, exist_ok=True)

    done = 0
    failed = 0
    for item in reels:
        reel_id = str(item.get("reel_id", ""))
        rally_id = str(item.get("rally_id", ""))
        if not rally_id:
            rally_ids = item.get("rally_ids")
            if isinstance(rally_ids, list) and rally_ids:
                rally_id = str(rally_ids[0])
        title = str(item.get("title", "reel"))
        track_json_raw = item.get("track_json")
        track_json_path = item.get("track_json_path")
        try:
            if not reel_id:
                raise ValueError("Missing reel_id")
            track_payloads: list[dict] = []
            track_jsons_raw = item.get("track_jsons")
            track_json_paths = item.get("track_json_paths")

            if isinstance(track_json_paths, list) and track_json_paths:
                for track_path_raw in track_json_paths:
                    if not isinstance(track_path_raw, str) or not track_path_raw:
                        continue
                    path = Path(track_path_raw).resolve()
                    if not path.exists() or not path.is_file():
                        raise FileNotFoundError(f"track_json_path not found: {path}")
                    with path.open("r", encoding="utf-8") as f:
                        payload = json.load(f)
                    if not isinstance(payload, dict):
                        raise ValueError("track payload must be object")
                    track_payloads.append(payload)
            elif isinstance(track_jsons_raw, list) and track_jsons_raw:
                for track_json_item in track_jsons_raw:
                    if isinstance(track_json_item, dict):
                        payload = track_json_item
                    elif isinstance(track_json_item, str):
                        payload = json.loads(track_json_item)
                    else:
                        continue
                    if not isinstance(payload, dict):
                        raise ValueError("track payload must be object")
                    track_payloads.append(payload)
            elif isinstance(track_json_path, str) and track_json_path:
                path = Path(track_json_path).resolve()
                if not path.exists() or not path.is_file():
                    raise FileNotFoundError(f"track_json_path not found: {path}")
                with path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
                if not isinstance(payload, dict):
                    raise ValueError("track payload must be object")
                track_payloads.append(payload)
            else:
                if not isinstance(track_json_raw, str):
                    raise ValueError("Missing track_json/track_json_path/track_jsons/track_json_paths")
                payload = json.loads(track_json_raw)
                if not isinstance(payload, dict):
                    raise ValueError("track payload must be object")
                track_payloads.append(payload)

            if not track_payloads:
                raise ValueError("No valid track payloads found")

            # "full": the whole frame with the ball arcs of the sets, else the vertical crop.
            full_frame = item.get("format") == "full"
            set_trails = [
                (int(trail[0]), int(trail[1]))
                for trail in item.get("set_trails") or []
                if isinstance(trail, (list, tuple)) and len(trail) == 2
            ] if full_frame else None

            file_name = f"{reel_id}.mp4"
            output_path = reels_dir / file_name
            if len(track_payloads) == 1:
                crop_and_save_track_payload(
                    video_path=str(video_path),
                    track_payload=track_payloads[0],
                    output_path=str(output_path),
                    predictions_path=item.get("predictions_path"),
                    full_frame=full_frame,
                    set_trails=set_trails,
                )
            else:
                crop_and_save_track_payloads(
                    video_path=str(video_path),
                    track_payloads=track_payloads,
                    output_path=str(output_path),
                    fade_transition=bool(item.get("fade_transition")),
                    predictions_path=item.get("predictions_path"),
                    full_frame=full_frame,
                    set_trails=set_trails,
                )
            web_output_path = reels_dir / f"{reel_id}_web.mp4"
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(output_path),
                "-c:v",
                "libx264",
                "-crf",
                "23",
                "-preset",
                "medium",
                "-profile:v",
                "main",
                "-level",
                "4.0",
                "-vf",
                "format=yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-movflags",
                "+faststart",
                "-threads",
                "0",
                str(web_output_path),
            ]
            try:
                subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"ffmpeg web conversion failed: {exc.stderr[-2000:] if exc.stderr else str(exc)}"
                ) from exc
            web_output_path.replace(output_path)
            relative_url = f"/uploads/{user_id}/{project_id}/reels/{file_name}"
            celery_app.send_task(
                API_UPDATE_REEL_TASK_NAME,
                kwargs={
                    "reel_id": reel_id,
                    "project_id": project_id,
                    "user_id": user_id,
                    "status": "ready",
                    "url": relative_url,
                    "title": title,
                    "rally_id": rally_id or None,
                },
                queue=API_IMPORT_TASK_QUEUE,
            )
            done += 1
            logger.info("Reel generated project=%s reel=%s title=%s path=%s", project_id, reel_id, title, output_path)
        except Exception:
            failed += 1
            logger.exception("Failed to generate reel project=%s reel=%s title=%s", project_id, reel_id, title)
            try:
                celery_app.send_task(
                    API_UPDATE_REEL_TASK_NAME,
                    kwargs={
                        "reel_id": reel_id,
                        "project_id": project_id,
                        "user_id": user_id,
                        "status": "failed",
                        "url": None,
                        "title": title,
                        "rally_id": rally_id or None,
                    },
                    queue=API_IMPORT_TASK_QUEUE,
                )
            except Exception:
                logger.exception("Failed to enqueue reel failure status project=%s reel=%s", project_id, reel_id)

    return {
        "project_id": project_id,
        "user_id": user_id,
        "generated": done,
        "failed": failed,
        "reels_dir": str(reels_dir),
    }
