#!/usr/bin/env python3
import argparse
import json
import logging
import os
from dataclasses import dataclass
from typing import List, Optional

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
        return df

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

        net_y = DEFAULT_NET_Y_THRESHOLD
        if self._court_geometry and len(self._court_geometry.keypoints) >= 8:
            net_left_y = self._court_geometry.keypoints[6][1]
            net_right_y = self._court_geometry.keypoints[7][1]
            net_y = min(net_left_y, net_right_y)

        min_ball_y = min(pos[0][1] for pos in track.positions)
        return min_ball_y < net_y

    def _filter_by_net(self, tracks: List[Track]) -> List[Track]:
        return [track for track in tracks if self._over_net_level(track)]

    def _filter_short_tracks(self, episodes: List[Track]) -> List[Track]:
        episodes = [self._trim_bounce_start(ep) for ep in episodes]
        episodes = self._filter_by_min_duration(episodes)
        episodes = self._remove_overlapping(episodes)
        episodes = self._extend_tracks(episodes)
        episodes = self._merge_overlapping(episodes)
        episodes = [self._trim_bounce_start(ep) for ep in episodes]
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
    )

    calculator = TrackCalculator(config)
    calculator.run()


if __name__ == "__main__":
    main()
