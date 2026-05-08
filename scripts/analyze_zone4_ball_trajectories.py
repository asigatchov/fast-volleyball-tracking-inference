#!/usr/bin/env python3
"""Build ball trajectories and evaluate zone-4 attack corridor near antennas."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import pandas as pd


COURT_LENGTH_M = 18.0
COURT_WIDTH_M = 9.0
GREEN_CORRIDOR_M = 0.8
YELLOW_CORRIDOR_M = 1.6
RED_CORRIDOR_M = 2.0

QUALITY_COLORS_BGR = {
    "green_excellent_0_80cm": (0, 220, 0),
    "yellow_ok_80_160cm": (0, 220, 255),
    "red_uncomfortable_160_200cm": (0, 0, 255),
    "outside_over_200cm": (160, 160, 160),
}


@dataclass
class Detection:
    frame: int
    x: float
    y: float
    radius: float
    confidence: float


@dataclass
class Track:
    track_id: int
    detections: list[Detection] = field(default_factory=list)
    missed: int = 0

    @property
    def start_frame(self) -> int:
        return self.detections[0].frame

    @property
    def last_frame(self) -> int:
        return self.detections[-1].frame

    def predict(self, frame: int) -> np.ndarray:
        if len(self.detections) < 2:
            last = self.detections[-1]
            return np.array([last.x, last.y], dtype=np.float32)

        prev = self.detections[-2]
        last = self.detections[-1]
        dt = max(1, last.frame - prev.frame)
        velocity = np.array([(last.x - prev.x) / dt, (last.y - prev.y) / dt], dtype=np.float32)
        return np.array([last.x, last.y], dtype=np.float32) + velocity * (frame - last.frame)


def load_detections(csv_path: str) -> dict[int, list[Detection]]:
    df = pd.read_csv(csv_path)
    required = {"Frame", "Visibility", "X", "Y", "Radius"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    detections_by_frame: dict[int, list[Detection]] = {}
    visible = df[(df["Visibility"] > 0) & (df["X"] >= 0) & (df["Y"] >= 0)]
    for row in visible.itertuples(index=False):
        det = Detection(
            frame=int(row.Frame),
            x=float(row.X),
            y=float(row.Y),
            radius=float(row.Radius),
            confidence=float(row.Visibility),
        )
        detections_by_frame.setdefault(det.frame, []).append(det)
    return detections_by_frame


def build_tracks(
    detections_by_frame: dict[int, list[Detection]],
    max_distance_px: float,
    max_missing_frames: int,
    min_points: int,
) -> list[Track]:
    active: list[Track] = []
    finished: list[Track] = []
    next_id = 0

    for frame in range(min(detections_by_frame), max(detections_by_frame) + 1):
        detections = detections_by_frame.get(frame, [])
        for track in active:
            track.missed += 1

        candidates: list[tuple[float, int, int]] = []
        for ti, track in enumerate(active):
            pred = track.predict(frame)
            for di, det in enumerate(detections):
                dist = float(np.linalg.norm(pred - np.array([det.x, det.y], dtype=np.float32)))
                if dist <= max_distance_px:
                    candidates.append((dist, ti, di))

        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        for _, ti, di in sorted(candidates, key=lambda item: item[0]):
            if ti in used_tracks or di in used_detections:
                continue
            active[ti].detections.append(detections[di])
            active[ti].missed = 0
            used_tracks.add(ti)
            used_detections.add(di)

        for di, det in enumerate(detections):
            if di in used_detections:
                continue
            active.append(Track(track_id=next_id, detections=[det]))
            next_id += 1

        still_active: list[Track] = []
        for track in active:
            if track.missed > max_missing_frames:
                finished.append(track)
            else:
                still_active.append(track)
        active = still_active

    finished.extend(active)
    return [track for track in finished if len(track.detections) >= min_points]


def load_court(court_json_path: str, net_height_m: float) -> dict[str, Any]:
    with open(court_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    keypoints_raw = data["annotations"][0]["keypoints"]
    names = data.get("categories", [{}])[0].get("keypoints", [])
    points = []
    for idx in range(0, len(keypoints_raw), 3):
        x, y, visibility = keypoints_raw[idx : idx + 3]
        if visibility <= 0:
            raise ValueError("All 8 court keypoints must be visible for this analysis")
        name = names[idx // 3] if idx // 3 < len(names) else str(idx // 3 + 1)
        points.append({"name": name, "x": float(x), "y": float(y)})

    if len(points) < 8:
        raise ValueError("Court JSON must contain 8 keypoints")

    floor_img = np.array(
        [[points[0]["x"], points[0]["y"]], [points[1]["x"], points[1]["y"]],
         [points[2]["x"], points[2]["y"]], [points[3]["x"], points[3]["y"]]],
        dtype=np.float32,
    )
    floor_court = np.array(
        [[-9.0, -4.5], [9.0, -4.5], [9.0, 4.5], [-9.0, 4.5]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(floor_img, floor_court)

    return {
        "points": points,
        "source_json_path": court_json_path,
        "image_file_name": data.get("images", [{}])[0].get("file_name"),
        "homography": homography,
        "center_floor_line": np.array([[points[4]["x"], points[4]["y"]], [points[5]["x"], points[5]["y"]]], dtype=np.float32),
        "net_top_line": np.array([[points[6]["x"], points[6]["y"]], [points[7]["x"], points[7]["y"]]], dtype=np.float32),
        "net_height_m": net_height_m,
    }


def image_to_court(homography: np.ndarray, x: float, y: float) -> tuple[float, float]:
    point = np.array([[[x, y]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(point, homography)[0][0]
    return float(transformed[0]), float(transformed[1])


def interpolate_line(line: np.ndarray, t: float) -> np.ndarray:
    return line[0] + (line[1] - line[0]) * t


def project_to_line_t(point: np.ndarray, line: np.ndarray) -> float:
    vec = line[1] - line[0]
    denom = float(np.dot(vec, vec))
    if denom <= 1e-9:
        return 0.0
    return float(np.dot(point - line[0], vec) / denom)


def estimate_net_height_m(point: np.ndarray, t: float, court: dict[str, Any]) -> float:
    floor_point = interpolate_line(court["center_floor_line"], t)
    top_point = interpolate_line(court["net_top_line"], t)
    floor_y = float(floor_point[1])
    top_y = float(top_point[1])
    if abs(floor_y - top_y) < 1e-6:
        return 0.0
    height_ratio = (floor_y - float(point[1])) / (floor_y - top_y)
    return max(0.0, height_ratio * float(court["net_height_m"]))


def side_from_court_x(court_x: float) -> str:
    return "near_left_zone4" if court_x < 0 else "far_right_zone4"


def classify_corridor(distance_m: float) -> str:
    if distance_m <= GREEN_CORRIDOR_M:
        return "green_excellent_0_80cm"
    if distance_m <= YELLOW_CORRIDOR_M:
        return "yellow_ok_80_160cm"
    if distance_m <= RED_CORRIDOR_M:
        return "red_uncomfortable_160_200cm"
    return "outside_over_200cm"


def is_beyond_zone4_antenna(side: str, net_t: float) -> bool:
    if side == "near_left_zone4":
        return net_t < 0.0
    return net_t > 1.0


def classify_attack_corridor(distance_m: float, beyond_antenna: bool) -> str:
    if beyond_antenna:
        return "red_uncomfortable_160_200cm"
    return classify_corridor(distance_m)


def smooth_values(values: list[float], window: int = 3) -> list[float]:
    if len(values) < window:
        return values
    pad = window // 2
    smoothed = []
    for idx in range(len(values)):
        start = max(0, idx - pad)
        end = min(len(values), idx + pad + 1)
        smoothed.append(float(np.mean(values[start:end])))
    return smoothed


def find_apex_indices(samples: list[dict[str, Any]], min_delta_px: float = 0.75) -> list[int]:
    if len(samples) < 3:
        return []

    ys = smooth_values([float(sample["image_y"]) for sample in samples], window=3)
    deltas = [ys[idx] - ys[idx - 1] for idx in range(1, len(ys))]
    apex_indices = []
    for idx in range(1, len(deltas)):
        was_rising = deltas[idx - 1] < -min_delta_px
        now_falling = deltas[idx] > min_delta_px
        if was_rising and now_falling:
            apex_indices.append(idx)
    return apex_indices


def select_attack_sample(samples: list[dict[str, Any]], court: dict[str, Any]) -> tuple[dict[str, Any], str]:
    if not samples:
        raise ValueError("Track has no samples")

    apex_indices = find_apex_indices(samples)
    if apex_indices:
        apex_idx = apex_indices[1] if len(apex_indices) > 1 else apex_indices[-1]
        descending_indices = []
        ys = smooth_values([float(sample["image_y"]) for sample in samples], window=3)
        for idx in range(apex_idx + 1, len(samples)):
            if ys[idx] - ys[idx - 1] > 0.75:
                descending_indices.append(idx)
        if len(descending_indices) >= 2:
            return samples[descending_indices[1]], "second_descending_point_after_set_apex"
        if descending_indices:
            return samples[descending_indices[0]], "first_descending_point_after_set_apex"

    net_candidates = [
        sample
        for sample in samples
        if -0.2 <= sample["net_t"] <= 1.2
        and sample["height_m_at_net_plane_approx"] >= court["net_height_m"] * 0.75
    ]
    if net_candidates:
        return max(net_candidates, key=lambda sample: sample["height_m_at_net_plane_approx"]), "fallback_highest_near_net"

    return max(samples, key=lambda sample: sample["height_m_at_net_plane_approx"]), "fallback_highest_track_point"


def evaluate_track(track: Track, court: dict[str, Any], fps: float) -> dict[str, Any]:
    net_top_line = court["net_top_line"]
    samples = []
    for sample_idx, det in enumerate(track.detections):
        img_point = np.array([det.x, det.y], dtype=np.float32)
        court_x, court_y = image_to_court(court["homography"], det.x, det.y)
        t = project_to_line_t(img_point, net_top_line)
        t_clamped = min(1.0, max(0.0, t))
        height_m = estimate_net_height_m(img_point, t_clamped, court)
        side = side_from_court_x(court_x)
        target_antenna_t = 0.0 if side == "near_left_zone4" else 1.0
        distance_to_zone4_antenna_m = abs(t - target_antenna_t) * COURT_WIDTH_M
        beyond_antenna = is_beyond_zone4_antenna(side, t)
        samples.append(
            {
                "sample_idx": sample_idx,
                "frame": det.frame,
                "image_x": det.x,
                "image_y": det.y,
                "court_x_m": court_x,
                "court_y_m": court_y,
                "net_t": t,
                "height_m_at_net_plane_approx": height_m,
                "side": side,
                "distance_to_zone4_antenna_m": distance_to_zone4_antenna_m,
                "is_beyond_antenna": beyond_antenna,
            }
        )

    best, selection_reason = select_attack_sample(samples, court)
    corridor = classify_attack_corridor(best["distance_to_zone4_antenna_m"], best["is_beyond_antenna"])

    return {
        "track_id": track.track_id,
        "start_frame": track.start_frame,
        "end_frame": track.last_frame,
        "duration_sec": (track.last_frame - track.start_frame + 1) / fps,
        "points": len(track.detections),
        "zone4_side": best["side"],
        "attack_frame": best["frame"],
        "net_position_t_0_left_1_right": best["net_t"],
        "height_m_at_net_plane_approx": best["height_m_at_net_plane_approx"],
        "distance_to_zone4_antenna_m": best["distance_to_zone4_antenna_m"],
        "is_beyond_antenna": best["is_beyond_antenna"],
        "attack_selection_reason": selection_reason,
        "corridor": corridor,
        "samples": samples,
    }


def write_outputs(output_dir: str, analyses: list[dict[str, Any]], court: dict[str, Any]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "zone4_trajectories.json")
    csv_path = os.path.join(output_dir, "zone4_summary.csv")

    serializable = {
        "court": {
            "net_height_m": court["net_height_m"],
            "keypoints": court["points"],
            "corridor_criteria_m": {
                "green_excellent_0_80cm": [0.0, GREEN_CORRIDOR_M],
                "yellow_ok_80_160cm": [GREEN_CORRIDOR_M, YELLOW_CORRIDOR_M],
                "red_uncomfortable_160_200cm": [YELLOW_CORRIDOR_M, RED_CORRIDOR_M],
            },
        },
        "tracks": analyses,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    summary_fields = [
        "track_id",
        "start_frame",
        "end_frame",
        "duration_sec",
        "points",
        "zone4_side",
        "attack_frame",
        "net_position_t_0_left_1_right",
        "height_m_at_net_plane_approx",
        "distance_to_zone4_antenna_m",
        "is_beyond_antenna",
        "attack_selection_reason",
        "corridor",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for analysis in analyses:
            writer.writerow({field: analysis[field] for field in summary_fields})

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


def resolve_video_path(video_path: str | None, court: dict[str, Any], csv_path: str) -> str:
    if video_path:
        return video_path

    candidates: list[str] = []
    image_file_name = court.get("image_file_name")
    if image_file_name:
        json_dir = os.path.dirname(str(court.get("source_json_path") or ""))
        csv_dir = os.path.dirname(csv_path)
        candidates.extend(
            [
                os.path.join(json_dir, image_file_name),
                os.path.join(csv_dir, image_file_name),
                image_file_name,
            ]
        )

    csv_stem = os.path.splitext(os.path.basename(csv_path))[0].replace("_predict_ball", "")
    candidates.append(os.path.join(os.path.dirname(csv_path), f"{csv_stem}.mp4"))

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate

    raise FileNotFoundError("Video file was not found. Pass it explicitly with --video-path.")


def draw_court_overlay(frame: np.ndarray, court: dict[str, Any]) -> None:
    points = court["points"]
    court_poly = np.array([[int(p["x"]), int(p["y"])] for p in points[:4]], dtype=np.int32)
    cv2.polylines(frame, [court_poly], isClosed=True, color=(255, 255, 255), thickness=2)

    center_floor_line = court["center_floor_line"]
    net_top_line = court["net_top_line"]
    cv2.line(frame, tuple(center_floor_line[0].astype(int)), tuple(center_floor_line[1].astype(int)), (255, 255, 255), 2)
    cv2.line(frame, tuple(net_top_line[0].astype(int)), tuple(net_top_line[1].astype(int)), (255, 255, 255), 2)
    cv2.line(frame, tuple(center_floor_line[0].astype(int)), tuple(net_top_line[0].astype(int)), (255, 255, 255), 2)
    cv2.line(frame, tuple(center_floor_line[1].astype(int)), tuple(net_top_line[1].astype(int)), (255, 255, 255), 2)

    corridor_lines = [
        (0.0, QUALITY_COLORS_BGR["red_uncomfortable_160_200cm"]),
        (GREEN_CORRIDOR_M, QUALITY_COLORS_BGR["green_excellent_0_80cm"]),
        (YELLOW_CORRIDOR_M, QUALITY_COLORS_BGR["yellow_ok_80_160cm"]),
        (RED_CORRIDOR_M, QUALITY_COLORS_BGR["red_uncomfortable_160_200cm"]),
    ]
    for distance_m, color in corridor_lines:
        offset_t = distance_m / COURT_WIDTH_M
        for t in (offset_t, 1.0 - offset_t):
            floor_point = interpolate_line(center_floor_line, t).astype(int)
            top_point = interpolate_line(net_top_line, t).astype(int)
            draw_dashed_line(frame, floor_point, top_point, color, thickness=2, dash_px=16, gap_px=12)


def draw_dashed_line(
    frame: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
    dash_px: int,
    gap_px: int,
) -> None:
    vector = end.astype(np.float32) - start.astype(np.float32)
    length = float(np.linalg.norm(vector))
    if length < 1.0:
        return

    direction = vector / length
    current = 0.0
    while current < length:
        segment_start = start.astype(np.float32) + direction * current
        segment_end = start.astype(np.float32) + direction * min(length, current + dash_px)
        cv2.line(
            frame,
            tuple(np.round(segment_start).astype(int)),
            tuple(np.round(segment_end).astype(int)),
            color,
            thickness,
            cv2.LINE_AA,
        )
        current += dash_px + gap_px


def build_visualization_index(analyses: list[dict[str, Any]]) -> tuple[dict[int, list[dict[str, Any]]], dict[int, dict[str, Any]]]:
    samples_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    analyses_by_id = {int(item["track_id"]): item for item in analyses}
    for analysis in analyses:
        track_id = int(analysis["track_id"])
        for sample in analysis["samples"]:
            indexed_sample = dict(sample)
            indexed_sample["track_id"] = track_id
            indexed_sample["corridor"] = analysis["corridor"]
            indexed_sample["is_attack_point"] = int(sample["frame"]) == int(analysis["attack_frame"])
            samples_by_frame[int(sample["frame"])].append(indexed_sample)
    return samples_by_frame, analyses_by_id


def draw_trails(
    frame: np.ndarray,
    current_frame: int,
    samples_by_frame: dict[int, list[dict[str, Any]]],
    trail_frames: int,
) -> None:
    start = max(0, current_frame - trail_frames)
    for sample_frame in range(start, current_frame + 1):
        alpha = 0.25 + 0.75 * ((sample_frame - start + 1) / max(1, trail_frames + 1))
        for sample in samples_by_frame.get(sample_frame, []):
            color = QUALITY_COLORS_BGR.get(str(sample["corridor"]), (220, 220, 220))
            faded = tuple(int(channel * alpha) for channel in color)
            radius = 4 if sample_frame < current_frame else 8
            center = (int(round(sample["image_x"])), int(round(sample["image_y"])))
            cv2.circle(frame, center, radius, faded, -1, cv2.LINE_AA)
            if sample.get("is_attack_point"):
                cv2.drawMarker(
                    frame,
                    center,
                    QUALITY_COLORS_BGR.get(str(sample["corridor"]), (255, 255, 255)),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=24,
                    thickness=3,
                    line_type=cv2.LINE_AA,
                )
            if sample_frame == current_frame:
                cv2.circle(frame, center, radius + 2, (255, 255, 255), 2, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, current_frame: int, current_samples: list[dict[str, Any]], analyses_by_id: dict[int, dict[str, Any]]) -> None:
    cv2.rectangle(frame, (12, 12), (620, 112), (0, 0, 0), -1)
    cv2.putText(frame, f"frame {current_frame} | space pause | esc/q quit", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "green <=0.8m  yellow <=1.6m  red <=2.0m", (24, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    if current_samples:
        sample = current_samples[0]
        analysis = analyses_by_id[int(sample["track_id"])]
        text = (
            f"track {sample['track_id']} {analysis['zone4_side']} "
            f"{analysis['distance_to_zone4_antenna_m']:.2f}m {analysis['corridor']}"
        )
        cv2.putText(frame, text, (24, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.62, QUALITY_COLORS_BGR.get(analysis["corridor"], (255, 255, 255)), 2, cv2.LINE_AA)


def render_visualization_frame(
    raw_frame: np.ndarray,
    current_frame: int,
    samples_by_frame: dict[int, list[dict[str, Any]]],
    analyses_by_id: dict[int, dict[str, Any]],
    court: dict[str, Any],
    trail_frames: int,
) -> np.ndarray:
    frame = raw_frame.copy()
    draw_court_overlay(frame, court)
    draw_trails(frame, current_frame, samples_by_frame, trail_frames)
    draw_hud(frame, current_frame, samples_by_frame.get(current_frame, []), analyses_by_id)
    return frame


def open_video_writer(output_file: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    ext = os.path.splitext(output_file)[1].lower()
    fourcc_name = "mp4v" if ext in {".mp4", ".m4v", ".mov"} else "XVID"
    writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*fourcc_name), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer: {output_file}")
    return writer


def interpolate_track_x(samples: list[dict[str, Any]], start_frame: int, end_frame: int) -> np.ndarray:
    frames = np.arange(start_frame, end_frame + 1)
    if not samples:
        return np.zeros(len(frames), dtype=np.float32)

    known_frames = np.array([int(sample["frame"]) for sample in samples], dtype=np.int32)
    known_x = np.array([float(sample["image_x"]) for sample in samples], dtype=np.float32)
    order = np.argsort(known_frames)
    known_frames = known_frames[order]
    known_x = known_x[order]
    return np.interp(frames, known_frames, known_x).astype(np.float32)


def smooth_track_x(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < 3:
        return values
    window = max(3, min(window, len(values)))
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, np.ones(window, dtype=np.float32) / window, mode="valid")


def crop_vertical_9_16(frame: np.ndarray, center_x: float) -> np.ndarray:
    frame_height, frame_width = frame.shape[:2]
    crop_height = frame_height
    crop_width = min(frame_width, int(round(crop_height * 9 / 16)))
    left = int(round(center_x - crop_width / 2))
    right = left + crop_width

    pad_left = max(0, -left)
    pad_right = max(0, right - frame_width)
    if pad_left or pad_right:
        frame = cv2.copyMakeBorder(frame, 0, 0, pad_left, pad_right, cv2.BORDER_REFLECT_101)
        left += pad_left
        right += pad_left

    left = max(0, min(left, frame.shape[1] - crop_width))
    right = left + crop_width
    return frame[:, left:right]


def make_green_reels(
    video_path: str,
    analyses: list[dict[str, Any]],
    output_dir: str,
    padding_sec: float,
    smooth_window: int,
) -> None:
    green_tracks = [analysis for analysis in analyses if str(analysis["corridor"]).startswith("green_")]
    os.makedirs(output_dir, exist_ok=True)
    if not green_tracks:
        print("No green tracks found for reels.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    crop_width = min(frame_width, int(round(frame_height * 9 / 16)))
    crop_height = frame_height
    padding_frames = int(round(max(0.0, padding_sec) * fps))
    video_stem = os.path.splitext(os.path.basename(video_path))[0]

    print(f"Writing {len(green_tracks)} green reel(s) to: {output_dir}")
    for analysis in green_tracks:
        start_frame = max(0, int(analysis["start_frame"]) - padding_frames)
        end_frame = min(total_video_frames - 1, int(analysis["end_frame"]) + padding_frames)
        x_values = interpolate_track_x(analysis["samples"], start_frame, end_frame)
        x_values = smooth_track_x(x_values, smooth_window)

        track_id = int(analysis["track_id"])
        output_path = os.path.join(output_dir, f"reel_{video_stem}_track_{track_id:04d}_green.mp4")
        writer = open_video_writer(output_path, fps, crop_width, crop_height)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        written = 0
        for idx, frame_number in enumerate(range(start_frame, end_frame + 1)):
            ok, frame = cap.read()
            if not ok:
                break
            cropped = crop_vertical_9_16(frame, x_values[idx])
            if cropped.shape[:2] != (crop_height, crop_width):
                cropped = cv2.resize(cropped, (crop_width, crop_height), interpolation=cv2.INTER_AREA)
            writer.write(cropped)
            written += 1

        writer.release()
        print(f"Wrote reel: {output_path} ({written} frames)")

    cap.release()


def visualize(
    video_path: str,
    analyses: list[dict[str, Any]],
    court: dict[str, Any],
    delay_ms: int,
    trail_frames: int,
    scale: float,
    output_file: str | None,
    show_window: bool,
) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    samples_by_frame, analyses_by_id = build_visualization_index(analyses)
    window_name = "Zone 4 ball trajectories"
    paused = False
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = open_video_writer(output_file, fps, width, height) if output_file else None
    if show_window:
        print("Visualization controls: space = pause/resume, q or esc = quit")
    if output_file:
        print(f"Writing visualization video: {output_file}")

    raw_frame = None
    current_frame = 0
    while True:
        if not paused:
            ok, raw_frame = cap.read()
            if not ok:
                break
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

        frame = render_visualization_frame(
            raw_frame=raw_frame,
            current_frame=current_frame,
            samples_by_frame=samples_by_frame,
            analyses_by_id=analyses_by_id,
            court=court,
            trail_frames=trail_frames,
        )

        if writer is not None and not paused:
            writer.write(frame)

        if show_window:
            display = frame
            if scale != 1.0:
                display = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cv2.imshow(window_name, display)

            key = cv2.waitKey(0 if paused else delay_ms) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = not paused

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Wrote visualization video: {output_file}")
    if show_window:
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--court-json-path", required=True)
    parser.add_argument("--output-dir", default="output/zone4_trajectories")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--net-height-m", type=float, default=2.24)
    parser.add_argument("--max-distance-px", type=float, default=180.0)
    parser.add_argument("--max-missing-frames", type=int, default=8)
    parser.add_argument("--min-points", type=int, default=8)
    parser.add_argument("--min-duration-sec", type=float, default=0.0)
    parser.add_argument("--video-path", help="Video path for --visualize. Defaults to file_name from court JSON.")
    parser.add_argument("--visualize", action="store_true", help="Show OpenCV visualization with trajectory trails.")
    parser.add_argument("--output-file", help="Path to save visualization video with overlays.")
    parser.add_argument("--reels", action="store_true", help="Create 9:16 reels for green tracks.")
    parser.add_argument("--reels-output-dir", help="Directory for green 9:16 reels. Defaults to <output-dir>/reels.")
    parser.add_argument("--reel-padding-sec", type=float, default=0.5)
    parser.add_argument("--reel-smooth-window", type=int, default=15)
    parser.add_argument("--visualize-delay-ms", type=int, default=1)
    parser.add_argument("--visualize-trail-frames", type=int, default=15)
    parser.add_argument("--visualize-scale", type=float, default=0.75)
    parser.add_argument(
        "--min-attack-height-m",
        type=float,
        default=0.0,
        help="Filter summaries by approximate ball height near the net plane.",
    )
    args = parser.parse_args()

    detections_by_frame = load_detections(args.csv_path)
    if not detections_by_frame:
        raise ValueError("No visible ball detections found")

    tracks = build_tracks(
        detections_by_frame,
        max_distance_px=args.max_distance_px,
        max_missing_frames=args.max_missing_frames,
        min_points=args.min_points,
    )
    court = load_court(args.court_json_path, net_height_m=args.net_height_m)
    analyses = [evaluate_track(track, court, fps=args.fps) for track in tracks]
    analyses = [
        item
        for item in analyses
        if item["duration_sec"] >= args.min_duration_sec
        and item["height_m_at_net_plane_approx"] >= args.min_attack_height_m
    ]
    analyses.sort(key=lambda item: (item["start_frame"], item["track_id"]))

    write_outputs(args.output_dir, analyses, court)
    print(f"Tracks: {len(analyses)}")
    for item in analyses:
        print(
            f"track={item['track_id']} frames={item['start_frame']}-{item['end_frame']} "
            f"side={item['zone4_side']} dist={item['distance_to_zone4_antenna_m']:.2f}m "
            f"corridor={item['corridor']}"
        )

    if args.visualize or args.output_file:
        video_path = resolve_video_path(args.video_path, court, args.csv_path)
        visualize(
            video_path=video_path,
            analyses=analyses,
            court=court,
            delay_ms=args.visualize_delay_ms,
            trail_frames=args.visualize_trail_frames,
            scale=args.visualize_scale,
            output_file=args.output_file,
            show_window=args.visualize,
        )

    if args.reels:
        video_path = resolve_video_path(args.video_path, court, args.csv_path)
        reels_output_dir = args.reels_output_dir or os.path.join(args.output_dir, "reels")
        make_green_reels(
            video_path=video_path,
            analyses=analyses,
            output_dir=reels_output_dir,
            padding_sec=args.reel_padding_sec,
            smooth_window=args.reel_smooth_window,
        )


if __name__ == "__main__":
    main()
