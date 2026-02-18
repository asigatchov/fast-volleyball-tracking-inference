import logging
import os
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from celery import Celery

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
API_IMPORT_TASK_QUEUE = os.getenv("API_IMPORT_TASK_QUEUE", "api-import")
API_IMPORT_REPLACE_EXISTING = os.getenv("API_IMPORT_REPLACE_EXISTING", "true").lower() in {"1", "true", "yes", "on"}

celery = celery_app
app = celery_app


def _resolve_model_arg(repo_dir: Path) -> str:
    script = repo_dir / "src" / "inference_openvino_seq_gray_v2.py"
    if not script.exists():
        return "--model_xml"
    content = script.read_text(encoding="utf-8", errors="ignore")
    if "--model_path" in content:
        return "--model_path"
    return "--model_xml"


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
            str(repo_dir / "ov" / "VballNetV2_seq9_grayscale_ov.xml"),
        )
    ).resolve()
    if not model_xml.exists():
        _queue_status_update(project_id, user_id, "new")
        raise FileNotFoundError(f"OpenVINO model not found: {model_xml}")

    device = os.getenv("INFERENCE_DEVICE", "CPU")
    video_uuid = video_path.stem
    output_dir = video_path.parent
    target_video_dir = output_dir / video_uuid
    target_video_dir.mkdir(parents=True, exist_ok=True)
    model_arg = _resolve_model_arg(repo_dir)

    inference_cmd = [
        "uv",
        "run",
        "src/inference_openvino_seq_gray_v2.py",
        "--video_path",
        str(video_path),
        model_arg,
        str(model_xml),
        "--output_dir",
        str(output_dir),
        "--only_csv",
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

    expected_csv = output_dir / f"{video_uuid}_predict_ball.csv"
    nested_csv = output_dir / video_uuid / "ball.csv"
    fallback_csv = output_dir / f"{video_uuid}_ball.csv"
    normalized_csv = target_video_dir / "ball.csv"

    source_csv: Path | None = None
    for candidate in (nested_csv, expected_csv, fallback_csv):
        if candidate.exists():
            source_csv = candidate
            break
    if source_csv is None:
        _queue_status_update(project_id, user_id, "new")
        raise FileNotFoundError(
            f"ball.csv not found after inference. Tried: {nested_csv}, {expected_csv}, {fallback_csv}"
        )

    if source_csv.resolve() != normalized_csv.resolve():
        shutil.copy2(source_csv, normalized_csv)

    track_cmd = [
        "uv",
        "run",
        "src/track_calculator.py",
        "--csv_path",
        str(normalized_csv),
        "--output_dir",
        str(output_dir),
    ]
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
