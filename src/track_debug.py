#!/usr/bin/env python3
import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


TRACK_COLOR = (0, 255, 255)
DETECTION_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)
SHADOW_COLOR = (0, 0, 0)
WINDOW_NAME = "Track Debug"


def require_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "cv2 is not installed. Install GUI OpenCV package, for example project extra `dev`."
        ) from exc
    return cv2


@dataclass
class TrackInfo:
    track_id: int
    start_frame: int
    last_frame: int
    reason: str
    positions_by_frame: Dict[int, Tuple[int, int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive debug viewer for track_*.json and ball CSV detections."
    )
    parser.add_argument("--video_path", required=True, help="Path to source video")
    parser.add_argument("--csv_path", required=True, help="Path to predict_ball.csv")
    parser.add_argument("--json_dir", required=True, help="Directory with track_*.json")
    return parser.parse_args()


def load_tracks(json_dir: str) -> Tuple[List[TrackInfo], Dict[int, List[TrackInfo]]]:
    json_path = Path(json_dir)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON directory not found: {json_dir}")

    tracks: List[TrackInfo] = []
    frame_to_tracks: Dict[int, List[TrackInfo]] = {}

    for file_path in sorted(json_path.glob("track_*.json")):
        with file_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        positions_by_frame: Dict[int, Tuple[int, int]] = {}
        for item in data.get("positions", []):
            if len(item) != 2:
                continue
            position, frame = item
            if not isinstance(position, (list, tuple)) or len(position) < 2:
                continue
            positions_by_frame[int(frame)] = (int(round(position[0])), int(round(position[1])))

        track = TrackInfo(
            track_id=int(data.get("track_id", 0)),
            start_frame=int(data.get("start_frame", 0)),
            last_frame=int(data.get("last_frame", 0)),
            reason=str(data.get("reason", "Unknown")),
            positions_by_frame=positions_by_frame,
        )
        tracks.append(track)

        for frame in positions_by_frame:
            frame_to_tracks.setdefault(frame, []).append(track)

    tracks.sort(key=lambda item: (item.start_frame, item.track_id))
    for frame_tracks in frame_to_tracks.values():
        frame_tracks.sort(key=lambda item: item.track_id)

    return tracks, frame_to_tracks


def load_detections(csv_path: str) -> Dict[int, Tuple[int, int, int]]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    detections: Dict[int, Tuple[int, int, int]] = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"Frame", "X", "Y", "Radius"}
        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                f"CSV must contain columns: {', '.join(sorted(required_columns))}"
            )

        for row in reader:
            try:
                frame = int(float(row["Frame"]))
                x = float(row["X"])
                y = float(row["Y"])
                radius = int(round(float(row.get("Radius", 0) or 0)))
                visibility = float(row.get("Visibility", 0) or 0)
            except (TypeError, ValueError):
                continue

            if visibility <= 0 or x < 0 or y < 0:
                continue
            detections[frame] = (int(round(x)), int(round(y)), max(radius, 6))

    return detections


def draw_text(frame, text: str, origin: Tuple[int, int], scale: float = 0.6) -> None:
    cv2 = require_cv2()
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        SHADOW_COLOR,
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )


def draw_marker(
    frame,
    center: Tuple[int, int],
    color: Tuple[int, int, int],
    radius: int,
    label: Optional[str] = None,
) -> None:
    cv2 = require_cv2()
    cv2.circle(frame, center, radius, color, 2, cv2.LINE_AA)
    cv2.drawMarker(
        frame,
        center,
        color,
        markerType=cv2.MARKER_CROSS,
        markerSize=max(radius * 2, 16),
        thickness=2,
        line_type=cv2.LINE_AA,
    )
    if label:
        draw_text(frame, label, (center[0] + 10, center[1] - 10), scale=0.5)


def find_last_ended_track(frame_idx: int, tracks: List[TrackInfo]) -> Optional[TrackInfo]:
    ended = [track for track in tracks if track.last_frame < frame_idx]
    if not ended:
        return None
    return max(ended, key=lambda track: track.last_frame)


def render_frame(
    frame,
    frame_idx: int,
    total_frames: int,
    fps: float,
    active_tracks: List[TrackInfo],
    detections: Dict[int, Tuple[int, int, int]],
    all_tracks: List[TrackInfo],
    paused: bool,
) -> None:
    cv2 = require_cv2()
    detection = detections.get(frame_idx)
    has_track = bool(active_tracks)

    if has_track:
        for track in active_tracks:
            pos = track.positions_by_frame.get(frame_idx)
            if not pos:
                continue
            draw_marker(frame, pos, TRACK_COLOR, 12, f"track {track.track_id}")
    elif detection:
        x, y, radius = detection
        draw_marker(frame, (x, y), DETECTION_COLOR, radius, "csv detect")

    status = "pause" if paused else "play"
    frame_time = frame_idx / fps if fps > 0 else 0.0
    draw_text(frame, f"frame {frame_idx}/{max(total_frames - 1, 0)}  {frame_time:.2f}s  {status}", (12, 28))

    if has_track:
        labels = ", ".join(
            f"{track.track_id}[{track.start_frame}-{track.last_frame}]"
            for track in active_tracks
        )
        draw_text(frame, f"active track: {labels}", (12, 56))
    elif detection:
        x, y, radius = detection
        draw_text(frame, f"no track, csv detect: ({x}, {y}) r={radius}", (12, 56))
    else:
        draw_text(frame, "no track, no csv detect", (12, 56))

    last_ended = find_last_ended_track(frame_idx, all_tracks)
    if last_ended:
        draw_text(
            frame,
            f"last ended: track {last_ended.track_id} @ {last_ended.last_frame}  reason: {last_ended.reason}",
            (12, 84),
            scale=0.55,
        )

    draw_text(frame, "space pause/play  a -1  f +1  s -15  w +15  q exit", (12, 112), scale=0.55)


def clamp_frame(frame_idx: int, total_frames: int) -> int:
    if total_frames <= 0:
        return 0
    return max(0, min(frame_idx, total_frames - 1))


def show_debug_view(
    video_path: str,
    tracks: List[TrackInfo],
    frame_to_tracks: Dict[int, List[TrackInfo]],
    detections: Dict[int, Tuple[int, int, int]],
) -> None:
    cv2 = require_cv2()
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    frame_idx = 0
    paused = True

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                break

            active_tracks = frame_to_tracks.get(frame_idx, [])
            render_frame(
                frame=frame,
                frame_idx=frame_idx,
                total_frames=total_frames,
                fps=fps,
                active_tracks=active_tracks,
                detections=detections,
                all_tracks=tracks,
                paused=paused,
            )

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(0 if paused else 30) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = not paused
                continue
            if key == ord("a"):
                frame_idx = clamp_frame(frame_idx - 1, total_frames)
                paused = True
                continue
            if key == ord("f"):
                frame_idx = clamp_frame(frame_idx + 1, total_frames)
                paused = True
                continue
            if key == ord("s"):
                frame_idx = clamp_frame(frame_idx - 15, total_frames)
                paused = True
                continue
            if key == ord("w"):
                frame_idx = clamp_frame(frame_idx + 15, total_frames)
                paused = True
                continue

            if not paused:
                frame_idx = clamp_frame(frame_idx + 1, total_frames)
                if frame_idx >= max(total_frames - 1, 0):
                    paused = True
            else:
                continue
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    tracks, frame_to_tracks = load_tracks(args.json_dir)
    detections = load_detections(args.csv_path)
    show_debug_view(args.video_path, tracks, frame_to_tracks, detections)


if __name__ == "__main__":
    main()
