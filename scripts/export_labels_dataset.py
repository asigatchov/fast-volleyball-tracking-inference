#!/usr/bin/env python3

import argparse
import json
import re
import shutil
import subprocess
import sys
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference_openvino_seq_gray_v2 import (
    BALL_RAW_SIZE_HISTORY,
    BALL_SIZE_HISTORY,
    estimate_ball_radius,
)


MODEL_XML = REPO_ROOT / "ov/VballNetGridV2b_seq9_grayscale_20260909_001145.xml"
PLAYER_LABELS_DIR = Path(
    "/home/ubuntu/projects/vb-action/prepared/volleyball_tape_vb_detection_rfdetr_medium/labels"
)
OUTPUT_BASE = Path("/mnt/data-ml/datashared/beach_vollyeball")
WORK_ROOT = Path("/tmp") / "export_labels_dataset_work"
TRAILING_INDEX_RE = re.compile(r"^(?P<base>.+)_(?P<index>\d{3,4})$")


@dataclass
class ClipSelection:
    start_frame: int
    end_frame: int
    source_track: str
    track_label: str
    clip_suffix: str


@dataclass
class LabelJob:
    label_path: Path
    video_stem: str
    dataset_name: str
    video_path: Path
    frame_width: int
    frame_height: int
    frame_count: int


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def prepare_work_dir() -> None:
    if WORK_ROOT.exists():
        shutil.rmtree(WORK_ROOT)
    WORK_ROOT.mkdir(parents=True, exist_ok=True)


def prepare_dataset_dir(dataset_name: str) -> Path:
    output_root = OUTPUT_BASE / dataset_name
    if output_root.exists():
        shutil.rmtree(output_root)
    for dirname in ("video", "csv", "jsons"):
        (output_root / dirname).mkdir(parents=True, exist_ok=True)
    return output_root


def infer_candidate_dataset_name(video_stem: str) -> str:
    match = TRAILING_INDEX_RE.match(video_stem)
    if match:
        return match.group("base")
    return video_stem


def collect_jobs(label_glob: str) -> list[LabelJob]:
    label_paths = sorted(PLAYER_LABELS_DIR.glob(label_glob))
    if not label_paths:
        raise FileNotFoundError(f"No label files matched: {label_glob}")

    stems = [path.name[: -len("_players.json")] for path in label_paths]
    candidate_counts = Counter(infer_candidate_dataset_name(stem) for stem in stems)

    jobs: list[LabelJob] = []
    for label_path, video_stem in zip(label_paths, stems, strict=True):
        source_json = load_json(label_path)
        video_path = Path(source_json["meta"]["video_path"])
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found for {label_path.name}: {video_path}")

        candidate = infer_candidate_dataset_name(video_stem)
        dataset_name = candidate if candidate_counts[candidate] > 1 else video_stem
        jobs.append(
            LabelJob(
                label_path=label_path,
                video_stem=video_stem,
                dataset_name=dataset_name,
                video_path=video_path,
                frame_width=int(source_json["meta"]["frame_width"]),
                frame_height=int(source_json["meta"]["frame_height"]),
                frame_count=int(source_json["meta"]["frame_count"]),
            )
        )

    return jobs


def run_inference(video_path: Path) -> Path:
    out_dir = WORK_ROOT / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            "uv",
            "run",
            "src/inference_openvino_seq_gray_v2.py",
            "--video_path",
            str(video_path),
            "--model_xml",
            str(MODEL_XML),
            "--output_dir",
            str(out_dir),
            "--only_csv",
            "--device",
            "CPU",
        ]
    )
    return out_dir / f"{video_path.stem}_predict_ball.csv"


def run_track_calculator(csv_path: Path, video_width: int, video_height: int) -> Path:
    out_dir = WORK_ROOT / "tracks"
    out_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            "uv",
            "run",
            "src/track_calculator.py",
            "--csv_path",
            str(csv_path),
            "--output_dir",
            str(out_dir),
            "--video_width",
            str(video_width),
            "--video_height",
            str(video_height),
            "--beach",
        ]
    )
    return out_dir / csv_path.stem.replace("_predict_ball", "") / "tracks"


def collect_clip_selections(tracks_dir: Path, frame_count: int) -> list[ClipSelection]:
    candidates: list[ClipSelection] = []
    for track_path in sorted(tracks_dir.glob("track_*.json")):
        payload = load_json(track_path)
        positions = payload.get("positions", [])
        label = payload.get("rally_classification", {}).get("label", "unknown")
        if positions:
            pos_frames = [int(item[1]) for item in positions]
            start_frame = max(0, min(int(payload.get("start_frame", pos_frames[0])), pos_frames[0]))
            end_frame = min(
                frame_count - 1,
                max(int(payload.get("last_frame", pos_frames[-1])), pos_frames[-1]),
            )
        else:
            start_frame = 0
            end_frame = frame_count - 1

        candidates.append(
            ClipSelection(
                start_frame=start_frame,
                end_frame=end_frame,
                source_track=track_path.name,
                track_label=label,
                clip_suffix="",
            )
        )

    if not candidates:
        return [
            ClipSelection(
                start_frame=0,
                end_frame=frame_count - 1,
                source_track="full_video_fallback",
                track_label="fallback",
                clip_suffix="",
            )
        ]

    candidates.sort(key=lambda item: (item.start_frame, item.end_frame, item.source_track))
    suffix_width = max(4, len(str(len(candidates))))
    for index, candidate in enumerate(candidates, start=1):
        candidate.clip_suffix = f"_{index:0{suffix_width}d}"
    return candidates


def build_ball_boxes(video_path: Path, csv_df: pd.DataFrame) -> dict[int, list[int] | None]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    boxes: dict[int, list[int] | None] = {}
    prev_gray = None
    size_state = {
        "filtered_history": deque(maxlen=BALL_SIZE_HISTORY),
        "raw_history": deque(maxlen=BALL_RAW_SIZE_HISTORY),
        "smoothed_radius": 0.0,
    }

    try:
        for _, row in csv_df.sort_values("Frame").iterrows():
            frame_idx = int(row["Frame"])
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Video ended before frame {frame_idx}: {video_path}")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            visibility = int(row["Visibility"])
            x = int(row["X"])
            y = int(row["Y"])
            radius_csv = int(row["Radius"])
            bbox = None

            if visibility > 0 and x >= 0 and y >= 0:
                radius_est, contour = estimate_ball_radius(prev_gray, gray, x, y, size_state)
                if contour is not None and len(contour) > 0:
                    bx, by, bw, bh = cv2.boundingRect(contour)
                    bbox = [int(bx), int(by), int(bx + bw), int(by + bh)]
                else:
                    radius = radius_csv if radius_csv > 0 else radius_est
                    if radius > 0:
                        bbox = [
                            max(0, x - radius),
                            max(0, y - radius),
                            min(frame.shape[1] - 1, x + radius),
                            min(frame.shape[0] - 1, y + radius),
                        ]

            boxes[frame_idx] = bbox
            prev_gray = gray
    finally:
        cap.release()

    return boxes


def export_video_clip(video_path: Path, output_path: Path, start_frame: int, end_frame: int) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to create video writer: {output_path}")

    try:
        for _ in range(start_frame, end_frame + 1):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
    finally:
        writer.release()
        cap.release()


def export_ball_csv(
    csv_df: pd.DataFrame, output_path: Path, start_frame: int, end_frame: int
) -> pd.DataFrame:
    clip_df = csv_df[(csv_df["Frame"] >= start_frame) & (csv_df["Frame"] <= end_frame)].copy()
    clip_df["Frame"] = clip_df["Frame"] - start_frame
    clip_df.to_csv(output_path, index=False)
    return clip_df


def export_labels_json(
    source_json: dict[str, Any],
    output_video_path: Path,
    output_json_path: Path,
    start_frame: int,
    end_frame: int,
    local_csv_df: pd.DataFrame,
    ball_boxes: dict[int, list[int] | None],
) -> None:
    frames_payload: dict[str, list[dict[str, Any]]] = {}

    for _, row in local_csv_df.iterrows():
        local_frame = int(row["Frame"])
        source_frame = local_frame + start_frame
        detections = list(source_json.get("frames", {}).get(str(source_frame), []))
        ball_bbox = ball_boxes.get(source_frame)
        if ball_bbox is not None:
            detections.append(
                {
                    "class_name": "ball",
                    "score": float(row["Visibility"]),
                    "bbox_xyxy": [float(value) for value in ball_bbox],
                }
            )
        frames_payload[str(local_frame)] = detections

    meta = dict(source_json.get("meta", {}))
    meta.update(
        {
            "video_path": str(output_video_path),
            "frame_count": end_frame - start_frame + 1,
            "class_name": "player+ball",
        }
    )

    save_json(output_json_path, {"meta": meta, "frames": frames_payload})


def process_job(job: LabelJob, output_root: Path) -> None:
    source_json = load_json(job.label_path)
    csv_path = run_inference(job.video_path)
    csv_df = pd.read_csv(csv_path)
    tracks_dir = run_track_calculator(csv_path, job.frame_width, job.frame_height)
    selections = collect_clip_selections(tracks_dir, frame_count=job.frame_count)
    ball_boxes = build_ball_boxes(job.video_path, csv_df)

    for selection in selections:
        clip_stem = f"{job.video_stem}{selection.clip_suffix}" if selection.clip_suffix else job.video_stem
        output_video_path = output_root / "video" / f"{clip_stem}.mp4"
        output_csv_path = output_root / "csv" / f"{clip_stem}_ball.csv"
        output_json_path = output_root / "jsons" / f"{clip_stem}.json"

        export_video_clip(
            video_path=job.video_path,
            output_path=output_video_path,
            start_frame=selection.start_frame,
            end_frame=selection.end_frame,
        )
        local_csv_df = export_ball_csv(
            csv_df=csv_df,
            output_path=output_csv_path,
            start_frame=selection.start_frame,
            end_frame=selection.end_frame,
        )
        export_labels_json(
            source_json=source_json,
            output_video_path=output_video_path,
            output_json_path=output_json_path,
            start_frame=selection.start_frame,
            end_frame=selection.end_frame,
            local_csv_df=local_csv_df,
            ball_boxes=ball_boxes,
        )

        print(
            f"{job.dataset_name}/{clip_stem}: {selection.start_frame}-{selection.end_frame} "
            f"track={selection.source_track} label={selection.track_label}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export datasets from *_players.json labels")
    parser.add_argument(
        "--label-glob",
        default="*_players.json",
        help="Glob relative to labels dir, e.g. 'g_beach_mix_20260507_*_players.json'",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    jobs = collect_jobs(args.label_glob)
    prepare_work_dir()

    roots: dict[str, Path] = {}
    for job in jobs:
        if job.dataset_name not in roots:
            roots[job.dataset_name] = prepare_dataset_dir(job.dataset_name)

    for job in jobs:
        process_job(job, roots[job.dataset_name])

    print(f"Exported {len(jobs)} videos into {len(roots)} dataset directories under {OUTPUT_BASE}")


if __name__ == "__main__":
    main()
