import logging
import os
import subprocess
from pathlib import Path
from urllib.parse import urlparse
import json
from typing import Any

from celery import Celery
import cv2

try:
    from .make_reels import crop_and_save_track_payload, crop_and_save_track_payloads
except ImportError:
    from make_reels import crop_and_save_track_payload, crop_and_save_track_payloads

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
API_IMPORT_TASK_QUEUE = os.getenv("API_IMPORT_TASK_QUEUE", "api-import")
API_IMPORT_REPLACE_EXISTING = os.getenv("API_IMPORT_REPLACE_EXISTING", "true").lower() in {"1", "true", "yes", "on"}

celery = celery_app
app = celery_app


def _resolve_video_fps(video_path: Path) -> float:
    default_fps = float(os.getenv("TRACK_VIDEO_FPS", "30"))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return default_fps
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    return fps if fps > 0 else default_fps


def _resolve_video_size(video_path: Path) -> tuple[int, int]:
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


def _predictions_json_to_ball_csv(predictions_json: Path, csv_path: Path) -> None:
    """Convert RAVEL-VB predictions JSON into the ``Frame,Visibility,X,Y,Radius`` CSV.

    With ``--frame-step N`` the network only sees every N-th frame. Skipped frames
    are filled by linear interpolation when both neighbouring shown frames have a
    ball, otherwise they are written as invisible.
    """
    with predictions_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    frame_step = max(int(payload.get("frame_step", 1)), 1)
    total_frames = int(payload.get("benchmark", {}).get("processed_frames", 0))

    balls: dict[int, tuple[float, float, float]] = {}
    best_score: dict[int, float] = {}
    for item in payload.get("predictions", []):
        if item.get("class_name") != "ball":
            continue
        frame = int(item["frame_index"])
        score = float(item.get("score", 0.0))
        if score <= best_score.get(frame, -1.0):
            continue
        x1, y1, x2, y2 = (float(v) for v in item["bbox_xyxy"])
        radius = max(x2 - x1, y2 - y1) / 2
        balls[frame] = ((x1 + x2) / 2, (y1 + y2) / 2, radius)
        best_score[frame] = score

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


@celery_app.task(name="inference.process_uploaded_video")
def process_uploaded_video(project_id: str, user_id: str, file_path: str, file_url: str) -> dict[str, str | bool]:
    parsed = urlparse(file_url)
    if parsed.scheme not in {"http", "https"}:
        _queue_status_update(project_id, user_id, "new")
        raise ValueError("file_url must start with http:// or https://")

    video_path = Path(file_path).resolve()
    if not video_path.exists():
        _queue_status_update(project_id, user_id, "new")
        raise FileNotFoundError(f"Video file not found: {video_path}")

    repo_dir = Path(os.getenv("INFERENCE_REPO_DIR", "/app")).resolve()
    if not repo_dir.exists():
        _queue_status_update(project_id, user_id, "new")
        raise FileNotFoundError(f"Inference repo not found: {repo_dir}")

    model_xml = Path(
        os.getenv(
            "INFERENCE_MODEL_XML",
            str(repo_dir / "ov" / "RAVEL-VB-011-9f.xml"),
        )
    ).resolve()
    if not model_xml.exists():
        _queue_status_update(project_id, user_id, "new")
        raise FileNotFoundError(f"OpenVINO model not found: {model_xml}")

    device = os.getenv("INFERENCE_DEVICE", "CPU")
    frame_step = os.getenv("INFERENCE_FRAME_STEP", "2")
    video_uuid = video_path.stem
    output_dir = video_path.parent
    target_video_dir = output_dir / video_uuid
    target_video_dir.mkdir(parents=True, exist_ok=True)
    predictions_json = target_video_dir / "predictions.json"

    # Rally detection needs the court homography: do not start without markup.
    court_json = target_video_dir / "court.json"
    if not court_json.exists():
        _queue_status_update(project_id, user_id, "new")
        raise FileNotFoundError(f"Court markup is required before processing: {court_json}")

    inference_cmd = [
        "uv",
        "run",
        "src/inference_player_ball_openvino.py",
        str(video_path),
        "--model",
        str(model_xml),
        "--output",
        str(predictions_json),
        "--frame-step",
        frame_step,
        "--device",
        device,
    ]
    logger.info("Running inference command: %s", " ".join(inference_cmd))
    try:
        subprocess.run(inference_cmd, cwd=repo_dir, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        _queue_status_update(project_id, user_id, "new")
        raise RuntimeError(
            f"OpenVINO inference failed: {exc.stderr[-2000:] if exc.stderr else str(exc)}"
        ) from exc

    normalized_csv = target_video_dir / "ball.csv"
    if not predictions_json.exists():
        _queue_status_update(project_id, user_id, "new")
        raise FileNotFoundError(f"predictions.json not found after inference: {predictions_json}")
    try:
        _predictions_json_to_ball_csv(predictions_json, normalized_csv)
    except Exception:
        _queue_status_update(project_id, user_id, "new")
        raise

    video_width, video_height = _resolve_video_size(video_path)
    track_cmd = [
        "uv",
        "run",
        "src/track_calculator_with_court.py",
        "--players_json_path",
        str(predictions_json),
        "--court_json_path",
        str(court_json),
        "--output_dir",
        str(output_dir),
        "--video_width",
        str(video_width),
        "--video_height",
        str(video_height),
        "--fps",
        str(_resolve_video_fps(video_path)),
    ]
    if os.getenv("INFERENCE_BEACH", "false").lower() in {"1", "true", "yes", "on"}:
        track_cmd.append("--beach")
    logger.info("Running track command: %s", " ".join(track_cmd))
    try:
        subprocess.run(track_cmd, cwd=repo_dir, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        _queue_status_update(project_id, user_id, "new")
        raise RuntimeError(
            f"Track calculator failed: {exc.stderr[-2000:] if exc.stderr else str(exc)}"
        ) from exc

    tracks_dir = target_video_dir / "tracks"
    celery_app.send_task(
        API_IMPORT_TASK_NAME,
        args=[project_id, user_id, API_IMPORT_REPLACE_EXISTING],
        queue=API_IMPORT_TASK_QUEUE,
    )
    return {
        "project_id": project_id,
        "user_id": user_id,
        "file_path": str(video_path),
        "file_url": file_url,
        "exists": video_path.exists(),
        "ball_csv": str(normalized_csv),
        "tracks_dir": str(tracks_dir),
        "tracks_exists": tracks_dir.exists(),
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

            file_name = f"{reel_id}.mp4"
            output_path = reels_dir / file_name
            if len(track_payloads) == 1:
                crop_and_save_track_payload(
                    video_path=str(video_path),
                    track_payload=track_payloads[0],
                    output_path=str(output_path),
                )
            else:
                crop_and_save_track_payloads(
                    video_path=str(video_path),
                    track_payloads=track_payloads,
                    output_path=str(output_path),
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
