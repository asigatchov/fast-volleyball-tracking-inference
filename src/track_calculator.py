#!/usr/bin/env python3
import argparse
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ball_tracker import BallTracker, Track
from constants import (
    DEFAULT_BOUNCE_FRAMES,
    DEFAULT_DETECTION_BOX_RADIUS,
    DEFAULT_EXTEND_SECONDS,
    DEFAULT_FPS,
    DEFAULT_MAX_DISTANCE,
    DEFAULT_MAX_X_DISPLACEMENT,
    DEFAULT_MIN_DURATION_SEC,
    DEFAULT_MIN_Y_DISPLACEMENT,
    DEFAULT_NET_Y_THRESHOLD,
)
from court_transformer import CoordinateTransformer, CourtTransformer
from models import CourtGeometry
from track_utils import find_cyclic_sequences, find_rolling_sequences

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrackCalculatorConfig:
    csv_path: str
    output_dir: str
    fps: float
    max_distance: float
    min_duration_sec: float
    max_x_displacement: float
    min_y_displacement: float
    bounce_frames: int
    court_json_path: Optional[str]
    video_width: Optional[int]
    video_height: Optional[int]


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def resolve_video_basename(csv_path: str) -> str:
    csv_name = os.path.splitext(os.path.basename(csv_path))[0]
    parent = os.path.basename(os.path.dirname(csv_path))

    if csv_name == "ball" and parent:
        return parent
    if csv_name.endswith("_predict_ball"):
        return csv_name[: -len("_predict_ball")]
    if csv_name.endswith("_ball"):
        return csv_name[: -len("_ball")]
    return csv_name


class TrackCalculator:
    def __init__(self, config: TrackCalculatorConfig) -> None:
        self.config = config
        self.tracks: List[Track] = []
        self._camera_position = "unknown"

        transformer = CourtTransformer(config.court_json_path)
        result = transformer.load()
        self._court_geometry = result.geometry
        self._court_matrix = result.matrix
        self._coordinate_transformer = CoordinateTransformer(
            self._court_geometry, self._court_matrix
        )
        self._court_enabled = config.court_json_path is not None and self._court_geometry

        if config.court_json_path and not self._court_geometry:
            LOG.warning("Court JSON provided but could not be loaded, using image coordinates")
        else:
            self._camera_position = self._classify_camera_position()

    def _validate_csv(self) -> None:
        if not os.path.exists(self.config.csv_path):
            raise FileNotFoundError(f"CSV not found: {self.config.csv_path}")

    def _load_and_process_csv(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.csv_path)
        df["Frame"] = pd.to_numeric(df["Frame"], errors="coerce")
        df["Visibility"] = pd.to_numeric(df["Visibility"], errors="coerce")
        df["X"] = pd.to_numeric(df["X"], errors="coerce")
        df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
        df.loc[(df["X"] == -1) | (df["Visibility"] == 0), ["X", "Y"]] = np.nan
        self._maybe_rescale_court_geometry(df)
        return df

    def _maybe_rescale_court_geometry(self, df: pd.DataFrame) -> None:
        if not self._court_enabled or not self._court_geometry:
            return

        target_w = self.config.video_width
        target_h = self.config.video_height

        if target_w is None or target_h is None:
            max_x = df["X"].max(skipna=True)
            max_y = df["Y"].max(skipna=True)
            if (
                pd.notna(max_x)
                and pd.notna(max_y)
                and (
                    max_x > self._court_geometry.image_width
                    or max_y > self._court_geometry.image_height
                )
            ):
                target_w = max(int(max_x) + 1, self._court_geometry.image_width)
                target_h = max(int(max_y) + 1, self._court_geometry.image_height)
                LOG.warning(
                    "Detected coordinates exceed court.json image size (%sx%s). "
                    "Applied best-effort scaling to %sx%s; prefer explicit --video_width/--video_height.",
                    self._court_geometry.image_width,
                    self._court_geometry.image_height,
                    target_w,
                    target_h,
                )

        if target_w is None or target_h is None:
            return
        if target_w <= 0 or target_h <= 0:
            return
        if (
            target_w == self._court_geometry.image_width
            and target_h == self._court_geometry.image_height
        ):
            return

        scale_x = target_w / self._court_geometry.image_width
        scale_y = target_h / self._court_geometry.image_height
        scaled_keypoints = tuple((x * scale_x, y * scale_y) for x, y in self._court_geometry.keypoints)

        self._court_geometry = CourtGeometry(
            length_m=self._court_geometry.length_m,
            width_m=self._court_geometry.width_m,
            net_height_m=self._court_geometry.net_height_m,
            image_width=target_w,
            image_height=target_h,
            keypoints=scaled_keypoints,
        )
        self._court_matrix = CourtTransformer._calculate_transform(scaled_keypoints)
        self._coordinate_transformer = CoordinateTransformer(self._court_geometry, self._court_matrix)
        self._camera_position = self._classify_camera_position()
        LOG.info("Scaled court keypoints to %sx%s", target_w, target_h)

    def _classify_camera_position(self) -> str:
        if not self._court_geometry or len(self._court_geometry.keypoints) < 8:
            return "unknown"

        p1, p2, p3, p4 = self._court_geometry.keypoints[:4]
        p7, p8 = self._court_geometry.keypoints[6], self._court_geometry.keypoints[7]

        dx = p8[0] - p7[0]
        dy = p8[1] - p7[1]
        court_span = max(np.hypot(p4[0] - p1[0], p4[1] - p1[1]), np.hypot(p3[0] - p2[0], p3[1] - p2[1]), 1.0)
        net_span = np.hypot(dx, dy)
        net_span_ratio = net_span / court_span

        if abs(dx) < 1.0 or abs(dy) / (abs(dx) + 1e-6) > 0.7 or net_span_ratio < 0.28:
            return "sideline"

        left_depth = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
        right_depth = np.hypot(p4[0] - p3[0], p4[1] - p3[1])
        depth_ratio = max(left_depth, right_depth) / max(1.0, min(left_depth, right_depth))
        net_mid_x = (p7[0] + p8[0]) / 2.0
        center_offset = abs(net_mid_x - self._court_geometry.image_width / 2.0) / max(
            1.0, self._court_geometry.image_width
        )

        if depth_ratio <= 1.35 and center_offset <= 0.12:
            return "backline"
        return "diagonal"

    @staticmethod
    def _is_overlapping(track1: Track, track2: Track) -> bool:
        return track1.start_frame <= track2.last_frame and track2.start_frame <= track1.last_frame

    def _trim_bounce_start(self, track: Track) -> Track:
        if not track.positions:
            return track

        original_start = track.start_frame
        original_end = track.last_frame

        sequences = find_cyclic_sequences(track.positions)
        if sequences:
            for _, end in sequences:
                track.start_frame = end
                break

        sequences = find_rolling_sequences(track.positions)
        if sequences:
            for start, _ in sequences:
                track.last_frame = start
                break

        if track.start_frame != original_start or track.last_frame != original_end:
            track.positions = [
                pos for pos in track.positions if track.start_frame <= pos[1] <= track.last_frame
            ]
        return track

    def _net_y_at_x(self, x: float) -> float:
        if not self._court_geometry or len(self._court_geometry.keypoints) < 8:
            return DEFAULT_NET_Y_THRESHOLD

        net_left = self._court_geometry.keypoints[6]
        net_right = self._court_geometry.keypoints[7]
        dx = net_right[0] - net_left[0]
        if abs(dx) < 1e-6:
            return float(min(net_left[1], net_right[1]))
        t = (x - net_left[0]) / dx
        return float(net_left[1] + t * (net_right[1] - net_left[1]))

    def _is_above_net(self, x: float, y: float) -> bool:
        net_y = self._net_y_at_x(x)
        image_h = self._court_geometry.image_height if self._court_geometry else 720
        clearance = max(6.0, image_h * 0.01)
        return y < (net_y - clearance)

    def _find_rolling_start_frame(
        self, positions: List[Any], start_index: int = 0
    ) -> Optional[int]:
        if not positions or start_index >= len(positions):
            return None

        tail = positions[start_index:]
        if len(tail) < 8:
            return None

        xs = np.array([p[0][0] for p in tail], dtype=np.float64)
        ys = np.array([p[0][1] for p in tail], dtype=np.float64)
        frames = np.array([p[1] for p in tail], dtype=np.int64)

        image_w = self._court_geometry.image_width if self._court_geometry else 1280
        image_h = self._court_geometry.image_height if self._court_geometry else 720

        window = 8
        y_range_threshold = max(18.0, image_h * 0.025)
        x_range_threshold = max(30.0, image_w * 0.03)
        floor_y = float(np.percentile(ys, 85))
        floor_tolerance = max(22.0, image_h * 0.05)

        for i in range(0, len(tail) - window + 1):
            xw = xs[i : i + window]
            yw = ys[i : i + window]

            y_range = float(np.max(yw) - np.min(yw))
            x_range = float(np.max(xw) - np.min(xw))
            is_near_floor = float(np.mean(yw)) >= (floor_y - floor_tolerance)
            if y_range <= y_range_threshold and x_range >= x_range_threshold and is_near_floor:
                return int(frames[i])

        return None

    def _analyze_track_trajectory(self, track: Track) -> Dict[str, Any]:
        positions = sorted(track.positions, key=lambda p: p[1])
        if not positions:
            return {
                "camera_position": self._camera_position,
                "last_above_net_frame": None,
                "stop_rising_above_net_frame": None,
                "rolling_start_frame": None,
                "game_pause_frame": None,
                "stop_rising_above_net_sec": None,
            }

        above_flags = [self._is_above_net(pos[0][0], pos[0][1]) for pos in positions]
        above_indices = [idx for idx, flag in enumerate(above_flags) if flag]
        last_above_idx = above_indices[-1] if above_indices else None
        last_above_frame = positions[last_above_idx][1] if last_above_idx is not None else None

        stop_rising_frame = None
        search_start_idx = 0
        if last_above_idx is not None and (last_above_idx + 1) < len(positions):
            stop_rising_frame = positions[last_above_idx + 1][1]
            search_start_idx = last_above_idx + 1

        rolling_start_frame = self._find_rolling_start_frame(positions, start_index=search_start_idx)
        game_pause_frame = rolling_start_frame

        if rolling_start_frame is not None:
            stop_rising_frame = rolling_start_frame
        elif stop_rising_frame is not None:
            game_pause_frame = stop_rising_frame

        stop_rising_sec = (
            float(stop_rising_frame) / self.config.fps
            if stop_rising_frame is not None and self.config.fps > 0
            else None
        )

        return {
            "camera_position": self._camera_position,
            "last_above_net_frame": int(last_above_frame) if last_above_frame is not None else None,
            "stop_rising_above_net_frame": int(stop_rising_frame) if stop_rising_frame is not None else None,
            "rolling_start_frame": int(rolling_start_frame) if rolling_start_frame is not None else None,
            "game_pause_frame": int(game_pause_frame) if game_pause_frame is not None else None,
            "stop_rising_above_net_sec": stop_rising_sec,
        }

    def _trim_after_game_pause(self, track: Track) -> Track:
        if not track.positions:
            return track

        analysis = self._analyze_track_trajectory(track)
        pause_frame = analysis["game_pause_frame"]
        if pause_frame is None:
            return track
        if pause_frame <= track.start_frame or pause_frame >= track.last_frame:
            return track

        track.last_frame = int(pause_frame)
        track.positions = [pos for pos in track.positions if track.start_frame <= pos[1] <= track.last_frame]
        return track

    def _filter_by_min_duration(self, tracks: List[Track]) -> List[Track]:
        return [track for track in tracks if track.duration_sec() >= self.config.min_duration_sec]

    def _remove_overlapping(self, tracks: List[Track]) -> List[Track]:
        sorted_tracks = sorted(tracks, key=lambda x: x.duration_sec(), reverse=True)
        filtered = []
        used = set()
        for i, track1 in enumerate(sorted_tracks):
            if i in used:
                continue
            filtered.append(track1)
            used.add(i)
            for j, track2 in enumerate(sorted_tracks):
                if j <= i or j in used:
                    continue
                if self._is_overlapping(track1, track2):
                    used.add(j)
        return filtered

    def _extend_tracks(self, tracks: List[Track], seconds: float = DEFAULT_EXTEND_SECONDS) -> List[Track]:
        frames_to_extend = int(self.config.fps * seconds)
        extended = []
        for track in tracks:
            track.start_frame = max(0, track.start_frame - frames_to_extend)
            track.last_frame = track.last_frame + frames_to_extend
            extended.append(track)
        return extended

    def _merge_overlapping(self, tracks: List[Track]) -> List[Track]:
        merged = []
        used = set()
        sorted_ext = sorted(tracks, key=lambda x: x.start_frame)
        for i, track1 in enumerate(sorted_ext):
            if i in used:
                continue
            merged_track = track1
            merged_positions = list(merged_track.positions)
            used.add(i)
            for j, track2 in enumerate(sorted_ext):
                if j <= i or j in used:
                    continue
                if self._is_overlapping(merged_track, track2):
                    merged_track.start_frame = min(merged_track.start_frame, track2.start_frame)
                    merged_track.last_frame = max(merged_track.last_frame, track2.last_frame)
                    merged_positions.extend(track2.positions)
                    used.add(j)
            merged_track.positions = sorted(merged_positions, key=lambda x: x[1])
            merged.append(merged_track)
        return merged

    def _over_net_level(self, track: Track) -> bool:
        if not track.positions:
            return False
        return any(self._is_above_net(pos[0][0], pos[0][1]) for pos in track.positions)

    def _filter_by_net(self, tracks: List[Track]) -> List[Track]:
        return [track for track in tracks if self._over_net_level(track)]

    def _filter_short_tracks(self, episodes: List[Track]) -> List[Track]:
        episodes = [self._trim_bounce_start(ep) for ep in episodes]
        episodes = [self._trim_after_game_pause(ep) for ep in episodes]
        episodes = self._filter_by_min_duration(episodes)
        episodes = self._remove_overlapping(episodes)
        episodes = self._extend_tracks(episodes)
        episodes = self._merge_overlapping(episodes)
        episodes = [self._trim_bounce_start(ep) for ep in episodes]
        episodes = [self._trim_after_game_pause(ep) for ep in episodes]
        if self._court_enabled:
            episodes = self._filter_by_net(episodes)
        return sorted(episodes, key=lambda x: x.start_frame)

    def _process_detections(self, df: pd.DataFrame) -> None:
        tracker = BallTracker(
            buffer_size=2500,
            max_disappeared=40,
            max_distance=self.config.max_distance,
            fps=self.config.fps,
        )
        close_tracks: List[Track] = []

        all_frames = sorted(df["Frame"].dropna().astype(int).unique())
        for frame_num in all_frames:
            frame_rows = df[df["Frame"] == frame_num]
            detections = []
            for _, row in frame_rows.iterrows():
                if not np.isnan(row["X"]) and not np.isnan(row["Y"]):
                    detections.append(
                        {
                            "x1": row["X"] - DEFAULT_DETECTION_BOX_RADIUS,
                            "y1": row["Y"] - DEFAULT_DETECTION_BOX_RADIUS,
                            "x2": row["X"] + DEFAULT_DETECTION_BOX_RADIUS,
                            "y2": row["Y"] + DEFAULT_DETECTION_BOX_RADIUS,
                            "confidence": row["Visibility"],
                            "cls_id": 0,
                        }
                    )
            _, _, closed = tracker.update(detections, frame_num)
            close_tracks.extend(closed)

        for track_id in list(tracker.tracks.keys()):
            close_tracks.append(tracker.tracks[track_id])
            del tracker.tracks[track_id]

        episodes = [track for track in close_tracks if track.positions]
        self.tracks = self._filter_short_tracks(episodes)

    def _save_tracks_to_json(self) -> None:
        csv_name = os.path.splitext(os.path.basename(self.config.csv_path))[0]
        video_basename = resolve_video_basename(self.config.csv_path)
        tracks_dir = os.path.join(self.config.output_dir, video_basename, "tracks")
        os.makedirs(tracks_dir, exist_ok=True)

        for track in self.tracks:
            track_dict = track.to_dict()
            trajectory_analysis = self._analyze_track_trajectory(track)
            track_dict["trajectory_analysis"] = trajectory_analysis
            if self._court_enabled:
                court_positions = []
                for pos in track_dict["positions"]:
                    img_x, img_y = pos[0]
                    court_x, court_y = self._coordinate_transformer.to_court(img_x, img_y)
                    court_positions.append([[court_x, court_y], pos[1]])
                track_dict["court_positions"] = court_positions
                track_dict["court_info"] = {
                    "image_width": self._court_geometry.image_width,
                    "image_height": self._court_geometry.image_height,
                    "court_points_count": len(self._court_geometry.keypoints),
                    "has_court_transform": self._court_matrix is not None,
                    "camera_position": self._camera_position,
                }

            file_path = os.path.join(tracks_dir, f"track_{track.track_id:04d}.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(track_dict, f, indent=2, ensure_ascii=False)

        LOG.info("Saved %s tracks to %s", len(self.tracks), tracks_dir)
        LOG.debug("CSV source name: %s", csv_name)

    def run(self) -> None:
        self._validate_csv()
        df = self._load_and_process_csv()
        self._process_detections(df)
        self._save_tracks_to_json()
        LOG.info("Done. Found %s tracks.", len(self.tracks))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calculate tracks from CSV to JSON")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to ball.csv")
    parser.add_argument("--court_json_path", type=str, help="Path to court coordinates JSON file")
    parser.add_argument(
        "--video_width",
        type=int,
        default=1920,
        help="Source video width for court scaling (default: 1920)",
    )
    parser.add_argument(
        "--video_height",
        type=int,
        default=1080,
        help="Source video height for court scaling (default: 1080)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Root output directory for JSON"
    )
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="Frames per second")
    parser.add_argument(
        "--max_distance", type=float, default=DEFAULT_MAX_DISTANCE, help="Max tracking distance"
    )
    parser.add_argument(
        "--min_duration_sec",
        type=float,
        default=DEFAULT_MIN_DURATION_SEC,
        help="Minimum track duration",
    )
    parser.add_argument(
        "--max_x_displacement",
        type=float,
        default=DEFAULT_MAX_X_DISPLACEMENT,
        help="Max X displacement",
    )
    parser.add_argument(
        "--min_y_displacement",
        type=float,
        default=DEFAULT_MIN_Y_DISPLACEMENT,
        help="Min Y displacement",
    )
    parser.add_argument(
        "--bounce_frames",
        type=int,
        default=DEFAULT_BOUNCE_FRAMES,
        help="Frames to analyze bounce",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.verbose)

    config = TrackCalculatorConfig(
        csv_path=args.csv_path,
        court_json_path=args.court_json_path,
        output_dir=args.output_dir,
        fps=args.fps,
        max_distance=args.max_distance,
        min_duration_sec=args.min_duration_sec,
        max_x_displacement=args.max_x_displacement,
        min_y_displacement=args.min_y_displacement,
        bounce_frames=args.bounce_frames,
        video_width=args.video_width,
        video_height=args.video_height,
    )

    calculator = TrackCalculator(config)
    calculator.run()


if __name__ == "__main__":
    main()
