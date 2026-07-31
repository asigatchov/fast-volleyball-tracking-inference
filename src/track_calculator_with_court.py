#!/usr/bin/env python3
"""Track calculation with court-aware rally vs technical-return analysis."""

from __future__ import annotations

import argparse
import cv2
import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from ball_tracker import BallTracker, Track
from constants import (
    BEACH_COURT_LENGTH_M,
    BEACH_COURT_WIDTH_M,
    COURT_LENGTH_M,
    COURT_WIDTH_M,
    DEFAULT_BOUNCE_FRAMES,
    DEFAULT_DETECTION_BOX_RADIUS,
    DEFAULT_FPS,
    DEFAULT_MAX_DISTANCE,
    DEFAULT_MAX_X_DISPLACEMENT,
    DEFAULT_MIN_DURATION_SEC,
    DEFAULT_MIN_Y_DISPLACEMENT,
    DEFAULT_NET_Y_THRESHOLD,
)
from court_transformer import CoordinateTransformer, CourtTransformer
from models import CourtGeometry

LOG = logging.getLogger(__name__)

REFERENCE_VIDEO_WIDTH = 1920.0
NET_HEIGHT_CM = 243.0
BALL_DIAMETER_CM = 21.0
BALL_SIZE_WINDOW = 6
MAX_MERGE_GAP_FRAMES = 40
MAX_AIRBORNE_REENTRY_GAP_SECONDS = 3.0
POST_PAUSE_TAIL_SECONDS = 0.5
MIN_EFFECTIVE_SCOPE_M = 0.45
MIN_EFFECTIVE_SCOPE_PX = 14.0


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
    beach: bool


@dataclass(frozen=True)
class FrameObservation:
    frame: int
    x: float
    y: float
    radius_px: Optional[float]
    smoothed_radius_px: Optional[float]
    visibility: float


@dataclass(frozen=True)
class BallSizeFeatures:
    radius_med_px: Optional[float]
    diameter_med_px: Optional[float]
    ball_cm_per_px: Optional[float]
    effective_scope_px: float
    effective_scope_m: float


@dataclass(frozen=True)
class CsvSizeCalibration:
    radius_lower_px: Optional[float]
    radius_upper_px: Optional[float]
    radius_median_px: Optional[float]
    radius_near_far_threshold_px: Optional[float]
    filtered_count: int


@dataclass
class TrackAnalysisRecord:
    track: Track
    observations: list[FrameObservation]
    trajectory_analysis: dict[str, Any]
    rally_features: dict[str, Any]
    rally_classification: dict[str, Any]
    state_before: dict[str, Any]
    state_after: dict[str, Any]
    score_event: Optional[dict[str, Any]] = None


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


class CourtContext:
    """Stores court geometry, camera classification and coordinate transforms."""

    def __init__(self, config: TrackCalculatorConfig) -> None:
        self._court_length_m = BEACH_COURT_LENGTH_M if config.beach else COURT_LENGTH_M
        self._court_width_m = BEACH_COURT_WIDTH_M if config.beach else COURT_WIDTH_M
        self._geometry: Optional[CourtGeometry] = None
        self._matrix = None
        self._transformer = CoordinateTransformer(None, None)
        self._camera_position = "unknown"
        self._cm_per_px_scale: Optional[float] = None

        loader = CourtTransformer(
            config.court_json_path,
            court_length_m=self._court_length_m,
            court_width_m=self._court_width_m,
        )
        result = loader.load()
        self._geometry = result.geometry
        self._matrix = result.matrix
        self._transformer = CoordinateTransformer(self._geometry, self._matrix)
        if self._geometry:
            self._camera_position = self._classify_camera_position()
            self._cm_per_px_scale = self._calculate_cm_per_px_scale()

    @property
    def enabled(self) -> bool:
        return self._geometry is not None

    @property
    def geometry(self) -> Optional[CourtGeometry]:
        return self._geometry

    @property
    def camera_position(self) -> str:
        return self._camera_position

    @property
    def court_length_m(self) -> float:
        return self._court_length_m

    @property
    def court_width_m(self) -> float:
        return self._court_width_m

    @property
    def cm_per_px_scale(self) -> Optional[float]:
        return self._cm_per_px_scale

    @property
    def coordinate_transformer(self) -> CoordinateTransformer:
        return self._transformer

    def maybe_rescale(self, frame_df: pd.DataFrame, target_w: Optional[int], target_h: Optional[int]) -> None:
        if not self._geometry:
            return

        new_w = target_w
        new_h = target_h
        if new_w is None or new_h is None:
            max_x = frame_df["X"].max(skipna=True)
            max_y = frame_df["Y"].max(skipna=True)
            if pd.notna(max_x) and pd.notna(max_y):
                new_w = max(int(max_x) + 1, self._geometry.image_width)
                new_h = max(int(max_y) + 1, self._geometry.image_height)

        if new_w is None or new_h is None or new_w <= 0 or new_h <= 0:
            return
        if new_w == self._geometry.image_width and new_h == self._geometry.image_height:
            return

        scale_x = new_w / self._geometry.image_width
        scale_y = new_h / self._geometry.image_height
        scaled_keypoints = tuple((x * scale_x, y * scale_y) for x, y in self._geometry.keypoints)

        self._geometry = CourtGeometry(
            length_m=self._geometry.length_m,
            width_m=self._geometry.width_m,
            net_height_m=self._geometry.net_height_m,
            image_width=int(new_w),
            image_height=int(new_h),
            keypoints=scaled_keypoints,
        )
        transformer = CourtTransformer(
            None,
            court_length_m=self._court_length_m,
            court_width_m=self._court_width_m,
        )
        self._matrix = transformer._calculate_transform(scaled_keypoints)
        self._transformer = CoordinateTransformer(self._geometry, self._matrix)
        self._camera_position = self._classify_camera_position()
        self._cm_per_px_scale = self._calculate_cm_per_px_scale()

    def to_court(self, x: float, y: float) -> tuple[float, float]:
        return self._transformer.to_court(x, y)

    def net_y_at_x(self, x: float) -> float:
        if not self._geometry or len(self._geometry.keypoints) < 8:
            return DEFAULT_NET_Y_THRESHOLD
        net_left = self._geometry.keypoints[6]
        net_right = self._geometry.keypoints[7]
        dx = net_right[0] - net_left[0]
        if abs(dx) < 1e-6:
            return float(min(net_left[1], net_right[1]))
        t = (x - net_left[0]) / dx
        return float(net_left[1] + t * (net_right[1] - net_left[1]))

    def evaluate_backline_start(
        self,
        x: float,
        y: float,
        scope_px: float,
        scope_m: float,
    ) -> dict[str, Any]:
        if self._camera_position != "backline" or not self._geometry or len(self._geometry.keypoints) < 4:
            return {
                "is_strongly_outside_start": False,
                "start_side_line_distance_px": None,
                "start_court_x_m": None,
                "start_court_y_m": None,
                "start_outside_reason": None,
            }

        p1, p2, p3, p4 = self._geometry.keypoints[:4]
        near_left = p1
        far_left = p2
        far_right = p3
        near_right = p4

        side_distance_px = min(
            self._point_to_line_distance(x, y, near_left, far_left),
            self._point_to_line_distance(x, y, near_right, far_right),
        )
        near_corner_distance_px = min(
            float(np.hypot(x - near_left[0], y - near_left[1])),
            float(np.hypot(x - near_right[0], y - near_right[1])),
        )
        polygon = np.array([near_left, far_left, far_right, near_right], dtype=np.float32)
        signed_distance = float(cv2.pointPolygonTest(polygon, (float(x), float(y)), True))
        court_x, court_y = self.to_court(x, y)
        lateral_limit_m = self._court_length_m / 2.0 + max(scope_m * 1.6, 0.8)
        width_limit_m = self._court_width_m / 2.0 + max(scope_m * 1.3, 0.6)
        outside_court = signed_distance < -max(scope_px * 1.2, 24.0)
        far_from_sideline_extension = side_distance_px > max(scope_px * 3.2, 70.0)
        far_from_near_corner = near_corner_distance_px > max(scope_px * 2.8, 50.0)

        reason = None
        if outside_court and far_from_sideline_extension and far_from_near_corner:
            reason = "outside_court_far_from_sideline_extensions"
        elif outside_court and abs(court_y) > width_limit_m * 1.6:
            reason = "too_wide_for_backline_start"

        return {
            "is_strongly_outside_start": reason is not None,
            "start_side_line_distance_px": float(side_distance_px),
            "start_polygon_signed_distance_px": signed_distance,
            "start_court_x_m": float(court_x),
            "start_court_y_m": float(court_y),
            "start_outside_reason": reason,
        }

    @staticmethod
    def _point_to_line_distance(
        x: float,
        y: float,
        p1: tuple[float, float],
        p2: tuple[float, float],
    ) -> float:
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        denom = math.hypot(dx, dy)
        if denom <= 1e-6:
            return float(np.hypot(x - x1, y - y1))
        return abs(dy * x - dx * y + x2 * y1 - y2 * x1) / denom

    def _classify_camera_position(self) -> str:
        if not self._geometry or len(self._geometry.keypoints) < 8:
            return "unknown"

        p1, p2, p3, p4 = self._geometry.keypoints[:4]
        p7, p8 = self._geometry.keypoints[6], self._geometry.keypoints[7]
        dx = p8[0] - p7[0]
        dy = p8[1] - p7[1]
        court_span = max(
            np.hypot(p4[0] - p1[0], p4[1] - p1[1]),
            np.hypot(p3[0] - p2[0], p3[1] - p2[1]),
            1.0,
        )
        net_span = np.hypot(dx, dy)
        net_span_ratio = net_span / court_span

        if abs(dx) < 1.0 or abs(dy) / (abs(dx) + 1e-6) > 0.7 or net_span_ratio < 0.28:
            return "sideline"

        left_depth = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
        right_depth = np.hypot(p4[0] - p3[0], p4[1] - p3[1])
        depth_ratio = max(left_depth, right_depth) / max(1.0, min(left_depth, right_depth))
        net_mid_x = (p7[0] + p8[0]) / 2.0
        center_offset = abs(net_mid_x - self._geometry.image_width / 2.0) / max(
            1.0, self._geometry.image_width
        )
        if depth_ratio <= 1.35 and center_offset <= 0.12:
            return "backline"
        return "diagonal"

    def _calculate_cm_per_px_scale(self) -> Optional[float]:
        if not self._geometry or len(self._geometry.keypoints) < 8:
            return None

        keypoints = self._geometry.keypoints
        candidates: list[float] = []

        p1, p3, p4 = keypoints[0], keypoints[2], keypoints[3]
        span_px = 0.0
        span_cm = 0.0
        if self._camera_position == "backline":
            span_px = float(np.hypot(p1[0] - p4[0], p1[1] - p4[1]))
            span_cm = self._court_width_m * 100.0
        elif self._camera_position == "sideline":
            span_px = float(np.hypot(p3[0] - p4[0], p3[1] - p4[1]))
            span_cm = self._court_length_m * 100.0
        if span_px > 1e-6 and span_cm > 0.0:
            candidates.append(span_cm / span_px)

        p5, p6, p7, p8 = keypoints[4], keypoints[5], keypoints[6], keypoints[7]
        net_right_px = float(np.hypot(p8[0] - p6[0], p8[1] - p6[1]))
        net_left_px = float(np.hypot(p7[0] - p5[0], p7[1] - p5[1]))
        valid_net = [v for v in (net_right_px, net_left_px) if v > 1e-6]
        if valid_net:
            candidates.append(NET_HEIGHT_CM / float(np.mean(valid_net)))

        if not candidates:
            return None
        return float(np.mean(candidates))


class TrackFeatureExtractor:
    """Computes size-aware, court-aware features for a single track."""

    def __init__(self, court: CourtContext, fps: float, calibration: CsvSizeCalibration) -> None:
        self._court = court
        self._fps = fps
        self._calibration = calibration

    def build_frame_observations(
        self,
        track: Track,
        observations_by_frame: Dict[int, FrameObservation],
    ) -> list[FrameObservation]:
        result: list[FrameObservation] = []
        for pos, frame in sorted(track.positions, key=lambda item: item[1]):
            source = observations_by_frame.get(int(frame))
            if source is not None:
                result.append(source)
                continue
            result.append(
                FrameObservation(
                    frame=int(frame),
                    x=float(pos[0]),
                    y=float(pos[1]),
                    radius_px=None,
                    smoothed_radius_px=None,
                    visibility=1.0,
                )
            )
        return result

    def compute_ball_size_features(self, observations: Sequence[FrameObservation]) -> BallSizeFeatures:
        radii = [obs.smoothed_radius_px for obs in observations if obs.smoothed_radius_px and obs.smoothed_radius_px > 0]
        radius_med_px = float(np.median(radii)) if radii else None
        diameter_med_px = (radius_med_px * 2.0) if radius_med_px is not None else None

        ball_cm_per_px = None
        if diameter_med_px and diameter_med_px > 1e-6:
            ball_cm_per_px = BALL_DIAMETER_CM / diameter_med_px

        effective_scope_px = max(
            MIN_EFFECTIVE_SCOPE_PX,
            (diameter_med_px * 1.75) if diameter_med_px is not None else 0.0,
        )

        court_scope_m = None
        if self._court.cm_per_px_scale is not None:
            court_scope_m = effective_scope_px * self._court.cm_per_px_scale / 100.0
        ball_scope_m = None
        if ball_cm_per_px is not None:
            ball_scope_m = effective_scope_px * ball_cm_per_px / 100.0

        candidates = [v for v in (court_scope_m, ball_scope_m) if v is not None and v > 0.0]
        effective_scope_m = max(MIN_EFFECTIVE_SCOPE_M, float(np.mean(candidates)) if candidates else MIN_EFFECTIVE_SCOPE_M)
        return BallSizeFeatures(
            radius_med_px=radius_med_px,
            diameter_med_px=diameter_med_px,
            ball_cm_per_px=ball_cm_per_px,
            effective_scope_px=float(effective_scope_px),
            effective_scope_m=float(effective_scope_m),
        )

    def is_above_net(self, observation: FrameObservation, size_features: BallSizeFeatures) -> bool:
        net_y = self._court.net_y_at_x(observation.x)
        top_y = observation.y - (observation.smoothed_radius_px or size_features.radius_med_px or 0.0)
        clearance_px = max(6.0, size_features.effective_scope_px * 0.35)

        if self._court.cm_per_px_scale is None:
            return top_y < (net_y - clearance_px)

        height_delta_cm = (net_y - top_y) * self._court.cm_per_px_scale
        return height_delta_cm > max(6.0, size_features.effective_scope_m * 100.0 * 0.3)

    def analyze_post_net_phase(
        self,
        observations: Sequence[FrameObservation],
        last_above_idx: Optional[int],
        size_features: BallSizeFeatures,
    ) -> dict[str, Any]:
        if last_above_idx is None or last_above_idx + 1 >= len(observations):
            return {
                "post_net_frames": 0,
                "post_net_duration_sec": 0.0,
                "post_net_path_len_px": 0.0,
                "post_net_path_len_m": 0.0,
                "post_net_x_range_px": 0.0,
                "post_net_x_range_m": 0.0,
                "post_net_vy_sign_changes": 0,
                "post_net_has_second_phase": False,
            }

        tail = list(observations[last_above_idx + 1 :])
        frames = np.array([obs.frame for obs in tail], dtype=np.float64)
        xs = np.array([obs.x for obs in tail], dtype=np.float64)
        ys = np.array([obs.y for obs in tail], dtype=np.float64)

        if len(tail) < 2:
            return {
                "post_net_frames": len(tail),
                "post_net_duration_sec": 0.0,
                "post_net_path_len_px": 0.0,
                "post_net_path_len_m": 0.0,
                "post_net_x_range_px": 0.0,
                "post_net_x_range_m": 0.0,
                "post_net_vy_sign_changes": 0,
                "post_net_has_second_phase": False,
            }

        frame_diffs = np.maximum(np.diff(frames), 1.0)
        dy = np.diff(ys)
        dx = np.diff(xs)
        step_dist_px = np.sqrt(dx * dx + dy * dy)
        vy = dy / frame_diffs
        signs = np.sign(vy)
        signs[np.abs(vy) < max(1.0, size_features.effective_scope_px * 0.15)] = 0
        signs = signs[signs != 0]
        vy_sign_changes = int(np.sum(signs[1:] * signs[:-1] < 0)) if len(signs) >= 2 else 0

        path_len_px = float(step_dist_px.sum())
        path_len_m = path_len_px * size_features.effective_scope_m / max(size_features.effective_scope_px, 1e-6)
        x_range_px = float(xs.max() - xs.min())
        x_range_m = x_range_px * size_features.effective_scope_m / max(size_features.effective_scope_px, 1e-6)
        duration_sec = max(0.0, float(frames[-1] - frames[0]) / self._fps) if self._fps > 0 else 0.0

        has_second_phase = (
            len(tail) >= 8
            and duration_sec >= 0.8
            and path_len_m >= size_features.effective_scope_m * 6.0
            and vy_sign_changes >= 2
        )
        return {
            "post_net_frames": int(len(tail)),
            "post_net_duration_sec": float(duration_sec),
            "post_net_path_len_px": path_len_px,
            "post_net_path_len_m": float(path_len_m),
            "post_net_x_range_px": x_range_px,
            "post_net_x_range_m": float(x_range_m),
            "post_net_vy_sign_changes": int(vy_sign_changes),
            "post_net_has_second_phase": bool(has_second_phase),
        }

    def infer_serve_side(self, observations: Sequence[FrameObservation]) -> str:
        if not observations:
            return "unknown"

        seed = observations[: min(6, len(observations))]
        x = float(np.median([obs.x for obs in seed]))
        y = float(np.median([obs.y for obs in seed]))
        start_radii = [
            obs.smoothed_radius_px for obs in seed if obs.smoothed_radius_px is not None and obs.smoothed_radius_px > 0
        ]
        start_radius_med = float(np.median(start_radii)) if start_radii else None

        if self._court.camera_position == "backline":
            size_features = self.compute_ball_size_features(seed)
            start_validation = self._court.evaluate_backline_start(
                x,
                y,
                size_features.effective_scope_px,
                size_features.effective_scope_m,
            )
            side_distance = start_validation.get("start_side_line_distance_px")
            image_width = self._court.geometry.image_width if self._court.geometry is not None else 0
            side_distance_threshold = max(70.0, image_width * 0.04)
            if side_distance is not None:
                if side_distance >= side_distance_threshold:
                    return "near"
                if side_distance <= side_distance_threshold * 0.82:
                    return "far"

            threshold = self._calibration.radius_near_far_threshold_px
            if start_radius_med is not None and threshold is not None:
                deadband = max(1.5, threshold * 0.08)
                if start_radius_med >= threshold + deadband * 2.0:
                    return "near"
                if start_radius_med <= threshold + deadband:
                    return "far"
            return "near" if y > self._court.net_y_at_x(x) else "far"

        if self._court.camera_position == "sideline":
            court_x, _ = self._court.to_court(x, y)
            return "left" if court_x < 0 else "right"

        return "unknown"

    def extract_features(
        self,
        track: Track,
        observations: Sequence[FrameObservation],
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        if not observations:
            empty_analysis = {
                "camera_position": self._court.camera_position,
                "serve_side": "unknown",
                "last_above_net_frame": None,
                "stop_rising_above_net_frame": None,
                "rolling_start_frame": None,
                "game_pause_frame": None,
                "stop_rising_above_net_sec": None,
                "is_technical_return": False,
                "technical_return_confidence": 0.0,
                "technical_return_side": "unknown",
                "post_net_frames": 0,
                "post_net_duration_sec": 0.0,
                "post_net_path_len_m": 0.0,
                "post_net_has_second_phase": False,
                "ball_diameter_px_med6_summary": None,
                "effective_scope_px_summary": None,
                "effective_scope_m_summary": None,
            }
            empty_features = {
                "track_id": int(track.track_id),
                "frame_start": int(track.start_frame),
                "frame_end": int(track.last_frame),
                "duration_sec": 0.0,
                "points_count": 0,
                "coverage": 0.0,
                "measurement_unit": "m" if self._court.cm_per_px_scale is not None else "px",
            }
            empty_classification = {
                "label": "not_rally",
                "is_rally": False,
                "not_rally_reason": "empty_track",
                "rally_confidence": 0.0,
                "not_rally_confidence": 1.0,
                "technical_return_confidence": 0.0,
                "score": -999,
            }
            return empty_analysis, empty_features, empty_classification

        size_features = self.compute_ball_size_features(observations)
        serve_side = self.infer_serve_side(observations)
        above_flags = [self.is_above_net(obs, size_features) for obs in observations]
        above_indices = [idx for idx, flag in enumerate(above_flags) if flag]
        last_above_idx = above_indices[-1] if above_indices else None
        last_above_frame = observations[last_above_idx].frame if last_above_idx is not None else None
        stop_rising_frame = observations[last_above_idx + 1].frame if last_above_idx is not None and last_above_idx + 1 < len(observations) else None

        post_net = self.analyze_post_net_phase(observations, last_above_idx, size_features)
        feature_payload = self._base_feature_payload(track, observations, serve_side, size_features, post_net)
        technical_return = self._classify_technical_return(feature_payload, size_features, serve_side)
        rally_classification = self._classify_rally(feature_payload, technical_return)

        confirmed_pause = technical_return["is_technical_return"]
        game_pause_frame = stop_rising_frame if confirmed_pause else None
        stop_rising_sec = (
            float(stop_rising_frame) / self._fps if stop_rising_frame is not None and self._fps > 0 else None
        )
        trajectory_analysis = {
            "camera_position": self._court.camera_position,
            "serve_side": serve_side,
            "last_above_net_frame": int(last_above_frame) if last_above_frame is not None else None,
            "stop_rising_above_net_frame": int(stop_rising_frame) if stop_rising_frame is not None else None,
            "rolling_start_frame": None,
            "game_pause_frame": int(game_pause_frame) if game_pause_frame is not None else None,
            "stop_rising_above_net_sec": stop_rising_sec,
            "is_technical_return": technical_return["is_technical_return"],
            "technical_return_confidence": technical_return["technical_return_confidence"],
            "technical_return_side": serve_side,
            "post_net_frames": post_net["post_net_frames"],
            "post_net_duration_sec": post_net["post_net_duration_sec"],
            "post_net_path_len_m": post_net["post_net_path_len_m"],
            "post_net_has_second_phase": post_net["post_net_has_second_phase"],
            "ball_diameter_px_med6_summary": size_features.diameter_med_px,
            "start_radius_med_px": feature_payload.get("start_radius_med_px"),
            "start_radius_near_far_threshold_px": feature_payload.get("start_radius_near_far_threshold_px"),
            "start_is_strongly_outside_backline": feature_payload.get("start_is_strongly_outside_backline", False),
            "start_outside_reason": feature_payload.get("start_outside_reason"),
            "start_side_line_distance_px": feature_payload.get("start_side_line_distance_px"),
            "start_polygon_signed_distance_px": feature_payload.get("start_polygon_signed_distance_px"),
            "start_court_x_m": feature_payload.get("start_court_x_m"),
            "start_court_y_m": feature_payload.get("start_court_y_m"),
            "effective_scope_px_summary": size_features.effective_scope_px,
            "effective_scope_m_summary": size_features.effective_scope_m,
        }
        feature_payload.update(
            {
                "has_game_pause": confirmed_pause,
                "has_rolling": False,
                "camera_position": self._court.camera_position,
                "serve_side": serve_side,
                "technical_return_score": technical_return["score"],
            }
        )
        return trajectory_analysis, feature_payload, rally_classification

    def _base_feature_payload(
        self,
        track: Track,
        observations: Sequence[FrameObservation],
        serve_side: str,
        size_features: BallSizeFeatures,
        post_net: dict[str, Any],
    ) -> dict[str, Any]:
        frames = np.array([obs.frame for obs in observations], dtype=np.float64)
        xs = np.array([obs.x for obs in observations], dtype=np.float64)
        ys = np.array([obs.y for obs in observations], dtype=np.float64)

        duration_frames = max(int(frames[-1] - frames[0] + 1), 1)
        duration_sec = duration_frames / self._fps if self._fps > 0 else 0.0
        coverage = float(len(observations) / duration_frames)

        if len(observations) == 1:
            path_len_px = 0.0
            median_speed_px_s = 0.0
            p90_speed_px_s = 0.0
            max_speed_px_s = 0.0
            vy_sign_changes = 0
            vx_sign_changes = 0
            max_gap_frames = 0
            gap_count_gt5 = 0
        else:
            frame_diffs = np.maximum(np.diff(frames), 1.0)
            dx = np.diff(xs)
            dy = np.diff(ys)
            step_dist_px = np.sqrt(dx * dx + dy * dy)
            dt = np.maximum(frame_diffs / max(self._fps, 1e-6), 1.0 / max(self._fps, 1.0))
            speed = step_dist_px / dt
            path_len_px = float(step_dist_px.sum())
            median_speed_px_s = float(np.median(speed))
            p90_speed_px_s = float(np.percentile(speed, 90))
            max_speed_px_s = float(np.max(speed))
            vx_sign_changes = self._count_sign_changes(dx)
            vy_sign_changes = self._count_sign_changes(dy)
            max_gap_frames = int(np.max(frame_diffs))
            gap_count_gt5 = int(np.sum(frame_diffs > 5))

        scale_m_per_px = self._resolve_metric_scale(size_features)
        x_range_px = float(xs.max() - xs.min())
        y_range_px = float(ys.max() - ys.min())
        path_len_m = path_len_px * scale_m_per_px if scale_m_per_px is not None else None
        x_range_m = x_range_px * scale_m_per_px if scale_m_per_px is not None else None
        y_range_m = y_range_px * scale_m_per_px if scale_m_per_px is not None else None

        start_zone = self._point_zone(observations[: min(6, len(observations))], size_features)
        end_zone = self._point_zone(observations[-min(6, len(observations)) :], size_features)
        start_x = float(np.median([obs.x for obs in observations[: min(6, len(observations))]]))
        start_y = float(np.median([obs.y for obs in observations[: min(6, len(observations))]]))
        start_validation = self._court.evaluate_backline_start(
            start_x,
            start_y,
            size_features.effective_scope_px,
            size_features.effective_scope_m,
        )
        start_radii = [
            obs.smoothed_radius_px
            for obs in observations[: min(6, len(observations))]
            if obs.smoothed_radius_px is not None and obs.smoothed_radius_px > 0
        ]
        start_radius_med_px = float(np.median(start_radii)) if start_radii else None

        return {
            "track_id": int(track.track_id),
            "frame_start": int(frames[0]),
            "frame_end": int(frames[-1]),
            "duration_sec": float(duration_sec),
            "points_count": int(len(observations)),
            "coverage": coverage,
            "x_range_px": x_range_px,
            "y_range_px": y_range_px,
            "path_len_px": path_len_px,
            "median_speed_px_s": median_speed_px_s,
            "p90_speed_px_s": p90_speed_px_s,
            "max_speed_px_s": max_speed_px_s,
            "vy_sign_changes": int(vy_sign_changes),
            "vx_sign_changes": int(vx_sign_changes),
            "gap_count_gt5": int(gap_count_gt5),
            "max_gap_frames": int(max_gap_frames),
            "measurement_unit": "m" if scale_m_per_px is not None else "px",
            "x_range_m": float(x_range_m) if x_range_m is not None else None,
            "y_range_m": float(y_range_m) if y_range_m is not None else None,
            "path_len_m": float(path_len_m) if path_len_m is not None else None,
            "post_net_path_len_m": float(post_net["post_net_path_len_m"]),
            "post_net_duration_sec": float(post_net["post_net_duration_sec"]),
            "post_net_has_second_phase": bool(post_net["post_net_has_second_phase"]),
            "start_zone": start_zone,
            "end_zone": end_zone,
            "serve_side": serve_side,
            "start_radius_med_px": start_radius_med_px,
            "start_radius_near_far_threshold_px": self._calibration.radius_near_far_threshold_px,
            "start_is_strongly_outside_backline": start_validation["is_strongly_outside_start"],
            "start_outside_reason": start_validation["start_outside_reason"],
            "start_side_line_distance_px": start_validation["start_side_line_distance_px"],
            "start_polygon_signed_distance_px": start_validation["start_polygon_signed_distance_px"],
            "start_court_x_m": start_validation["start_court_x_m"],
            "start_court_y_m": start_validation["start_court_y_m"],
            "crosses_net": bool(post_net["post_net_frames"] > 0),
            "ball_diameter_px_med6": size_features.diameter_med_px,
            "effective_scope_px": size_features.effective_scope_px,
            "effective_scope_m": size_features.effective_scope_m,
        }

    @staticmethod
    def _count_sign_changes(values: np.ndarray, eps: float = 2.0) -> int:
        if values.size < 2:
            return 0
        signs = np.sign(values)
        signs[np.abs(values) < eps] = 0
        signs = signs[signs != 0]
        if signs.size < 2:
            return 0
        return int(np.sum(signs[1:] * signs[:-1] < 0))

    def _resolve_metric_scale(self, size_features: BallSizeFeatures) -> Optional[float]:
        candidates = []
        if self._court.cm_per_px_scale is not None:
            candidates.append(self._court.cm_per_px_scale / 100.0)
        if size_features.ball_cm_per_px is not None:
            candidates.append(size_features.ball_cm_per_px / 100.0)
        if not candidates:
            return None
        return float(np.mean(candidates))

    def _point_zone(self, observations: Sequence[FrameObservation], size_features: BallSizeFeatures) -> str:
        if not observations:
            return "unknown"
        x = float(np.median([obs.x for obs in observations]))
        y = float(np.median([obs.y for obs in observations]))
        if self._court.camera_position == "backline":
            return "near" if y > self._court.net_y_at_x(x) else "far"
        if self._court.camera_position == "sideline":
            court_x, _ = self._court.to_court(x, y)
            threshold = size_features.effective_scope_m
            if court_x < -threshold:
                return "left"
            if court_x > threshold:
                return "right"
            return "center"
        return "unknown"

    def _classify_technical_return(
        self,
        features: dict[str, Any],
        size_features: BallSizeFeatures,
        serve_side: str,
    ) -> dict[str, Any]:
        scope_m = max(size_features.effective_scope_m, 1e-6)
        score = 0
        reasons: list[str] = []

        duration_sec = features["duration_sec"]
        path_len_m = features.get("path_len_m")
        x_range_m = features.get("x_range_m")
        post_net_path_len_m = features.get("post_net_path_len_m", 0.0)

        if duration_sec <= 2.6:
            score += 3
            reasons.append("short_duration")
        if path_len_m is not None and path_len_m <= scope_m * 12.0:
            score += 3
            reasons.append("short_path")
        if x_range_m is not None and x_range_m <= scope_m * 3.5:
            score += 2
            reasons.append("tight_lateral_range")
        if features["vy_sign_changes"] <= 5:
            score += 1
            reasons.append("simple_vertical_shape")
        if not features["post_net_has_second_phase"]:
            score += 2
            reasons.append("no_second_phase")
        if post_net_path_len_m <= scope_m * 6.0:
            score += 1
            reasons.append("short_post_net_tail")
        if serve_side in {"near", "far", "left", "right"} and features["end_zone"] == serve_side:
            score += 1
            reasons.append("returns_to_serve_side")
        if (
            features.get("crosses_net")
            and path_len_m is not None
            and path_len_m >= scope_m * 13.0
            and duration_sec >= 1.8
        ):
            score -= 5
            reasons.append("crosses_net_with_game_length")

        confidence = 1.0 / (1.0 + math.exp(-(float(score) - 5.5)))
        return {
            "is_technical_return": confidence >= 0.5,
            "technical_return_confidence": float(confidence),
            "score": int(score),
            "reasons": reasons,
        }

    def _classify_rally(self, features: dict[str, Any], technical_return: dict[str, Any]) -> dict[str, Any]:
        if technical_return["is_technical_return"]:
            return {
                "label": "not_rally",
                "is_rally": False,
                "not_rally_reason": "technical_return",
                "rally_confidence": float(1.0 - technical_return["technical_return_confidence"]),
                "not_rally_confidence": float(technical_return["technical_return_confidence"]),
                "technical_return_confidence": technical_return["technical_return_confidence"],
                "score": int(-technical_return["score"]),
                "positive_flags": [],
                "penalty_flags": technical_return["reasons"],
            }

        score = 0
        positive_flags: list[str] = []
        penalty_flags: list[str] = []
        scope_m = max(features.get("effective_scope_m", MIN_EFFECTIVE_SCOPE_M), 1e-6)

        if features["duration_sec"] >= 4.0:
            score += 2
            positive_flags.append("long_duration")
        if features.get("path_len_m") is not None and features["path_len_m"] >= scope_m * 18.0:
            score += 3
            positive_flags.append("long_path")
        if features.get("x_range_m") is not None and features["x_range_m"] >= scope_m * 4.5:
            score += 1
            positive_flags.append("wide_x_range")
        if features.get("y_range_m") is not None and features["y_range_m"] >= scope_m * 5.0:
            score += 1
            positive_flags.append("wide_y_range")
        if features["vy_sign_changes"] >= 7:
            score += 1
            positive_flags.append("many_vertical_phases")
        if features["post_net_has_second_phase"]:
            score += 2
            positive_flags.append("post_net_continuation")
        elif features["post_net_duration_sec"] < 0.6:
            score -= 1
            penalty_flags.append("short_post_net_tail")
        if features.get("start_is_strongly_outside_backline"):
            score -= 1
            penalty_flags.append(features.get("start_outside_reason", "outside_backline_start"))

        rally_confidence = 1.0 / (1.0 + math.exp(-(float(score) - 4.0)))
        return {
            "label": "rally" if rally_confidence >= 0.5 else "not_rally",
            "is_rally": rally_confidence >= 0.5,
            "not_rally_reason": None if rally_confidence >= 0.5 else "insufficient_rally_evidence",
            "rally_confidence": float(rally_confidence),
            "not_rally_confidence": float(1.0 - rally_confidence),
            "technical_return_confidence": technical_return["technical_return_confidence"],
            "score": int(score),
            "positive_flags": positive_flags,
            "penalty_flags": penalty_flags,
        }


class MatchStateMachine:
    """Tracks serve order and estimated score across a sequence of tracks."""

    def __init__(self, camera_position: str) -> None:
        self._camera_position = camera_position
        self._score = self._init_score(camera_position)
        self._state = "awaiting_technical_return"
        self._expected_server = "unknown"
        self._service_turn_index = 0
        self._rally_index = 0
        self._pending_rally_track_id: Optional[int] = None

    def apply(self, records: Sequence[TrackAnalysisRecord]) -> list[TrackAnalysisRecord]:
        updated: list[TrackAnalysisRecord] = []
        for record in sorted(records, key=lambda item: item.track.start_frame):
            record.state_before = self._snapshot()
            self._apply_contextual_adjustments(record)
            score_event = self._transition(record)
            record.score_event = score_event
            record.state_after = self._snapshot()
            updated.append(record)
        return updated

    def _apply_contextual_adjustments(self, record: TrackAnalysisRecord) -> None:
        serve_side = record.trajectory_analysis.get("serve_side", "unknown")
        duration_sec = record.rally_features.get("duration_sec", 0.0)
        path_len_m = record.rally_features.get("path_len_m")
        technical_conf = float(record.rally_classification.get("technical_return_confidence", 0.0))
        rally_conf = float(record.rally_classification.get("rally_confidence", 0.0))

        if (
            record.rally_classification.get("label") == "not_rally"
            and record.rally_classification.get("not_rally_reason") != "technical_return"
            and self._expected_server != "unknown"
            and serve_side == self._expected_server
            and duration_sec >= 3.0
            and path_len_m is not None
            and path_len_m >= record.rally_features.get("effective_scope_m", MIN_EFFECTIVE_SCOPE_M) * 14.0
        ):
            boosted_conf = max(rally_conf, 0.72)
            record.rally_classification.update(
                {
                    "label": "rally",
                    "is_rally": True,
                    "not_rally_reason": None,
                    "rally_confidence": boosted_conf,
                    "not_rally_confidence": 1.0 - boosted_conf,
                    "state_machine_adjustment": "serve_queue_promoted_to_rally",
                }
            )
            positives = list(record.rally_classification.get("positive_flags", []))
            positives.append("state_machine_expected_server_match")
            record.rally_classification["positive_flags"] = positives

        if (
            record.rally_classification.get("not_rally_reason") == "technical_return"
            and self._expected_server != "unknown"
            and serve_side == self._expected_server
        ):
            boosted_conf = min(0.99, max(technical_conf, 0.75) + 0.1)
            record.rally_classification["technical_return_confidence"] = boosted_conf
            record.rally_classification["not_rally_confidence"] = boosted_conf
            record.rally_classification["rally_confidence"] = 1.0 - boosted_conf
            penalties = list(record.rally_classification.get("penalty_flags", []))
            penalties.append("state_machine_expected_server_match")
            record.rally_classification["penalty_flags"] = penalties
            record.trajectory_analysis["technical_return_confidence"] = boosted_conf

    def _transition(self, record: TrackAnalysisRecord) -> Optional[dict[str, Any]]:
        label = record.rally_classification.get("label")
        reason = record.rally_classification.get("not_rally_reason")
        serve_side = record.trajectory_analysis.get("serve_side", "unknown")
        score_event = None

        if label == "rally":
            self._state = "rally_in_progress"
            self._pending_rally_track_id = int(record.track.track_id)
            self._rally_index += 1
            record.rally_features["serve_queue_position"] = self._service_turn_index
            record.rally_features["expected_server_side"] = self._expected_server
            return None

        if reason == "technical_return":
            if self._pending_rally_track_id is not None and serve_side in self._score:
                self._score[serve_side] += 1
                score_event = {
                    "winner_side": serve_side,
                    "score_after": dict(self._score),
                    "resolved_rally_track_id": self._pending_rally_track_id,
                }
            self._pending_rally_track_id = None
            if serve_side in self._score:
                self._expected_server = serve_side
            self._service_turn_index += 1
            self._state = "ready_for_serve"
            record.rally_features["serve_queue_position"] = self._service_turn_index
            record.rally_features["expected_server_side"] = self._expected_server
            return score_event

        record.rally_features["serve_queue_position"] = self._service_turn_index
        record.rally_features["expected_server_side"] = self._expected_server
        return None

    def _snapshot(self) -> dict[str, Any]:
        return {
            "state": self._state,
            "expected_server_side": self._expected_server,
            "score": dict(self._score),
            "service_turn_index": self._service_turn_index,
            "rally_index": self._rally_index,
            "pending_rally_track_id": self._pending_rally_track_id,
        }

    @staticmethod
    def _init_score(camera_position: str) -> dict[str, int]:
        if camera_position == "sideline":
            return {"left": 0, "right": 0}
        return {"near": 0, "far": 0}


class TrackCalculatorWithCourt:
    """Structured pipeline for track generation and court-aware analysis."""

    def __init__(self, config: TrackCalculatorConfig) -> None:
        self.config = config
        self.tracks: list[Track] = []
        self._court = CourtContext(config)
        self._csv_size_calibration = CsvSizeCalibration(
            radius_lower_px=None,
            radius_upper_px=None,
            radius_median_px=None,
            radius_near_far_threshold_px=None,
            filtered_count=0,
        )
        self._frame_width_scale = self._compute_frame_width_scale()
        self._scaled_max_distance = self.config.max_distance * self._frame_width_scale
        self._feature_extractor = TrackFeatureExtractor(
            self._court,
            self.config.fps,
            self._csv_size_calibration,
        )
        self._observations_by_frame: dict[int, FrameObservation] = {}

    def run(self) -> None:
        df = self._load_csv()
        self._process_detections(df)
        self._save_tracks_to_json()
        LOG.info("Done. Found %s tracks.", len(self.tracks))

    def _compute_frame_width_scale(self) -> float:
        width = self._resolved_video_width()
        if width is None or width <= 0:
            return 1.0
        return width / REFERENCE_VIDEO_WIDTH

    def _resolved_video_width(self) -> Optional[int]:
        if self.config.video_width is not None and self.config.video_width > 0:
            return self.config.video_width
        if self._court.geometry is not None:
            return self._court.geometry.image_width
        return None

    def _resolved_video_height(self) -> Optional[int]:
        if self.config.video_height is not None and self.config.video_height > 0:
            return self.config.video_height
        if self._court.geometry is not None:
            return self._court.geometry.image_height
        return None

    def _load_csv(self) -> pd.DataFrame:
        if not os.path.exists(self.config.csv_path):
            raise FileNotFoundError(f"CSV not found: {self.config.csv_path}")

        df = pd.read_csv(self.config.csv_path)
        for column in ("Frame", "Visibility", "X", "Y", "Radius"):
            if column not in df.columns:
                df[column] = np.nan
            df[column] = pd.to_numeric(df[column], errors="coerce")

        df.loc[(df["Visibility"] <= 0) | (df["X"] == -1), ["X", "Y"]] = np.nan
        valid_radius = df["Radius"].where(df["Radius"] > 0)
        self._csv_size_calibration = self._build_csv_size_calibration(valid_radius)
        df["RadiusMed6"] = (
            valid_radius.rolling(window=BALL_SIZE_WINDOW, min_periods=1, center=True).median()
        )
        global_radius = (
            self._csv_size_calibration.radius_median_px
            if self._csv_size_calibration.radius_median_px is not None
            else float(valid_radius.median()) if valid_radius.notna().any() else np.nan
        )
        df["RadiusMed6"] = df["RadiusMed6"].fillna(global_radius)

        self._court.maybe_rescale(df, self.config.video_width, self.config.video_height)
        self._feature_extractor = TrackFeatureExtractor(
            self._court,
            self.config.fps,
            self._csv_size_calibration,
        )
        self._observations_by_frame = self._build_observation_index(df)
        return df

    @staticmethod
    def _build_csv_size_calibration(valid_radius: pd.Series) -> CsvSizeCalibration:
        radii = [float(v) for v in valid_radius.dropna().tolist() if float(v) > 0]
        if not radii:
            return CsvSizeCalibration(None, None, None, None, 0)

        q1 = float(np.percentile(radii, 25))
        q3 = float(np.percentile(radii, 75))
        iqr = q3 - q1
        lower = max(0.0, q1 - 1.5 * iqr)
        upper = q3 + 1.5 * iqr
        filtered = [value for value in radii if lower <= value <= upper]
        if not filtered:
            filtered = radii

        median = float(np.median(filtered))
        return CsvSizeCalibration(
            radius_lower_px=float(lower),
            radius_upper_px=float(upper),
            radius_median_px=median,
            radius_near_far_threshold_px=median,
            filtered_count=len(filtered),
        )

    @staticmethod
    def _build_observation_index(df: pd.DataFrame) -> dict[int, FrameObservation]:
        result: dict[int, FrameObservation] = {}
        for row in df.itertuples(index=False):
            if pd.isna(row.Frame):
                continue
            frame = int(row.Frame)
            if pd.isna(row.X) or pd.isna(row.Y):
                continue
            radius = float(row.Radius) if pd.notna(row.Radius) and row.Radius > 0 else None
            smoothed_radius = (
                float(row.RadiusMed6)
                if pd.notna(row.RadiusMed6) and row.RadiusMed6 > 0
                else radius
            )
            result[frame] = FrameObservation(
                frame=frame,
                x=float(row.X),
                y=float(row.Y),
                radius_px=radius,
                smoothed_radius_px=smoothed_radius,
                visibility=float(row.Visibility) if pd.notna(row.Visibility) else 1.0,
            )
        return result

    def _process_detections(self, df: pd.DataFrame) -> None:
        tracker = BallTracker(
            buffer_size=2500,
            max_disappeared=40,
            max_distance=self._scaled_max_distance,
            fps=self.config.fps,
        )
        closed_tracks: list[Track] = []
        all_frames = sorted(df["Frame"].dropna().astype(int).unique())
        for frame_num in all_frames:
            frame_rows = df[df["Frame"] == frame_num]
            detections = [self._row_to_detection(row) for row in frame_rows.itertuples(index=False)]
            detections = [det for det in detections if det is not None]
            _, _, closed = tracker.update(detections, frame_num)
            closed_tracks.extend(closed)

        for track_id in list(tracker.tracks.keys()):
            closed_tracks.append(tracker.tracks[track_id])
            del tracker.tracks[track_id]

        episodes = [track for track in closed_tracks if track.positions]
        self.tracks = self._post_process_tracks(episodes)

    def _row_to_detection(self, row: Any) -> Optional[dict[str, float]]:
        if pd.isna(row.X) or pd.isna(row.Y):
            return None
        radius = None
        if pd.notna(row.RadiusMed6) and row.RadiusMed6 > 0:
            radius = float(row.RadiusMed6)
        elif pd.notna(row.Radius) and row.Radius > 0:
            radius = float(row.Radius)
        half_size = max(float(DEFAULT_DETECTION_BOX_RADIUS), radius if radius is not None else 0.0)
        return {
            "x1": float(row.X) - half_size,
            "y1": float(row.Y) - half_size,
            "x2": float(row.X) + half_size,
            "y2": float(row.Y) + half_size,
            "confidence": float(row.Visibility) if pd.notna(row.Visibility) else 1.0,
            "cls_id": 0.0,
        }

    def _post_process_tracks(self, tracks: Sequence[Track]) -> list[Track]:
        filtered = [track for track in tracks if track.duration_sec() >= self.config.min_duration_sec]
        filtered = self._remove_overlapping(filtered)
        filtered = self._extend_tracks(filtered)
        filtered = self._merge_overlapping(filtered)
        filtered = self._split_discontinuous_tracks(filtered)
        filtered = self._merge_airborne_reentry_tracks(filtered)
        if self._court.enabled:
            filtered = [track for track in filtered if self._track_crosses_net(track)]
        return sorted(filtered, key=lambda item: item.start_frame)

    def _remove_overlapping(self, tracks: Sequence[Track]) -> list[Track]:
        sorted_tracks = sorted(tracks, key=lambda item: item.duration_sec(), reverse=True)
        chosen: list[Track] = []
        taken: set[int] = set()
        for i, track in enumerate(sorted_tracks):
            if i in taken:
                continue
            chosen.append(track)
            for j, other in enumerate(sorted_tracks):
                if j <= i or j in taken:
                    continue
                if track.start_frame <= other.last_frame and other.start_frame <= track.last_frame:
                    taken.add(j)
        return chosen

    def _extend_tracks(self, tracks: Sequence[Track]) -> list[Track]:
        # This pipeline classifies event boundaries. Artificial extension smears
        # short technical-return episodes into neighboring rallies and hurts the
        # state machine, so we keep native tracker boundaries here.
        return list(tracks)

    def _merge_overlapping(self, tracks: Sequence[Track]) -> list[Track]:
        merged: list[Track] = []
        used: set[int] = set()
        items = sorted(tracks, key=lambda item: item.start_frame)
        for i, track in enumerate(items):
            if i in used:
                continue
            base = track
            positions = list(base.positions)
            used.add(i)
            for j, other in enumerate(items):
                if j <= i or j in used:
                    continue
                gap = max(0, max(other.start_frame - base.last_frame, base.start_frame - other.last_frame))
                if gap <= MAX_MERGE_GAP_FRAMES and not (
                    other.last_frame < base.start_frame or other.start_frame > base.last_frame
                ):
                    base.start_frame = min(base.start_frame, other.start_frame)
                    base.last_frame = max(base.last_frame, other.last_frame)
                    positions.extend(other.positions)
                    used.add(j)
            base.positions = type(base.positions)(sorted(positions, key=lambda item: item[1]), maxlen=base.positions.maxlen)
            merged.append(base)
        return merged

    def _split_discontinuous_tracks(self, tracks: Sequence[Track]) -> list[Track]:
        result: list[Track] = []
        for track in tracks:
            positions = sorted(track.positions, key=lambda item: item[1])
            if not positions:
                continue
            chunks: list[list[Any]] = [[positions[0]]]
            for pos in positions[1:]:
                prev_frame = int(chunks[-1][-1][1])
                frame = int(pos[1])
                if frame - prev_frame > MAX_MERGE_GAP_FRAMES:
                    chunks.append([pos])
                else:
                    chunks[-1].append(pos)

            for idx, chunk in enumerate(chunks):
                child = Track()
                child.track_id = track.track_id * 1000 + idx if len(chunks) > 1 else track.track_id
                child.reason = track.reason
                child.fps = track.fps
                child.positions = type(track.positions)(chunk, maxlen=track.positions.maxlen)
                child.start_frame = int(chunk[0][1])
                child.last_frame = int(chunk[-1][1])
                child.ball_sizes = track.ball_sizes
                child.prediction = track.prediction
                result.append(child)
        return result

    def _merge_airborne_reentry_tracks(self, tracks: Sequence[Track]) -> list[Track]:
        items = sorted(tracks, key=lambda item: item.start_frame)
        if not items:
            return []

        merged: list[Track] = []
        current = items[0]
        for candidate in items[1:]:
            if self._should_merge_airborne_reentry(current, candidate):
                current = self._combine_tracks(current, candidate)
            else:
                merged.append(current)
                current = candidate
        merged.append(current)
        return merged

    def _should_merge_airborne_reentry(self, first: Track, second: Track) -> bool:
        gap_frames = second.start_frame - first.last_frame
        if gap_frames <= 0:
            return False

        max_gap_frames = max(1, int(round(self.config.fps * MAX_AIRBORNE_REENTRY_GAP_SECONDS)))
        if gap_frames > max_gap_frames:
            return False

        first_points = self._sorted_track_points(first)
        second_points = self._sorted_track_points(second)
        if len(first_points) < 2 or len(second_points) < 2:
            return False

        if not self._track_exits_top(first_points) or not self._track_enters_from_top(second_points):
            return False

        width = self._resolved_video_width()
        max_x_gap = max(140.0, (width * 0.18) if width is not None else 220.0)
        end_x = float(np.median([point[0] for point in first_points[-min(3, len(first_points)) :]]))
        start_x = float(np.median([point[0] for point in second_points[: min(3, len(second_points))]]))
        return abs(end_x - start_x) <= max_x_gap

    def _track_exits_top(self, points: Sequence[tuple[float, float, int]]) -> bool:
        top_margin = self._top_reentry_margin()
        tail = points[-min(5, len(points)) :]
        if min(point[1] for point in tail) > top_margin:
            return False

        tail_vy = self._median_vertical_speed(tail)
        if tail_vy >= -4.0:
            return False

        return (tail[0][1] - tail[-1][1]) >= 18.0

    def _track_enters_from_top(self, points: Sequence[tuple[float, float, int]]) -> bool:
        top_margin = self._top_reentry_margin()
        head = points[: min(5, len(points))]
        if min(point[1] for point in head) > top_margin:
            return False

        head_vy = self._median_vertical_speed(head)
        if head_vy <= 4.0:
            return False

        return (head[-1][1] - head[0][1]) >= 18.0

    def _top_reentry_margin(self) -> float:
        height = self._resolved_video_height()
        if height is None or height <= 0:
            return 120.0
        return max(80.0, height * 0.16)

    @staticmethod
    def _sorted_track_points(track: Track) -> list[tuple[float, float, int]]:
        points: list[tuple[float, float, int]] = []
        for pos, frame in sorted(track.positions, key=lambda item: item[1]):
            points.append((float(pos[0]), float(pos[1]), int(frame)))
        return points

    @staticmethod
    def _median_vertical_speed(points: Sequence[tuple[float, float, int]]) -> float:
        if len(points) < 2:
            return 0.0

        speeds: list[float] = []
        for (_, y1, frame1), (_, y2, frame2) in zip(points, points[1:]):
            dt = max(frame2 - frame1, 1)
            speeds.append((y2 - y1) / dt)
        if not speeds:
            return 0.0
        return float(np.median(speeds))

    @staticmethod
    def _combine_tracks(base: Track, other: Track) -> Track:
        base.start_frame = min(base.start_frame, other.start_frame)
        base.last_frame = max(base.last_frame, other.last_frame)
        base.positions = type(base.positions)(
            sorted([*base.positions, *other.positions], key=lambda item: item[1]),
            maxlen=base.positions.maxlen,
        )
        base.ball_sizes = type(base.ball_sizes)([*base.ball_sizes, *other.ball_sizes], maxlen=base.ball_sizes.maxlen)
        base.prediction = other.prediction if other.prediction else base.prediction
        return base

    def _track_crosses_net(self, track: Track) -> bool:
        observations = self._feature_extractor.build_frame_observations(track, self._observations_by_frame)
        size_features = self._feature_extractor.compute_ball_size_features(observations)
        return any(self._feature_extractor.is_above_net(obs, size_features) for obs in observations)

    def _save_tracks_to_json(self) -> None:
        video_basename = resolve_video_basename(self.config.csv_path)
        tracks_dir = os.path.join(self.config.output_dir, video_basename, "tracks")
        os.makedirs(tracks_dir, exist_ok=True)
        for file_name in os.listdir(tracks_dir):
            if file_name.startswith("track_") and file_name.endswith(".json"):
                os.remove(os.path.join(tracks_dir, file_name))

        records = self._build_analysis_records()
        state_machine = MatchStateMachine(self._court.camera_position)
        records = state_machine.apply(records)

        for record in records:
            track = record.track
            observations = record.observations
            track_dict = track.to_dict()
            track_dict["trajectory_analysis"] = record.trajectory_analysis
            track_dict["rally_features"] = record.rally_features
            track_dict["rally_classification"] = record.rally_classification
            track_dict["match_state_before"] = record.state_before
            track_dict["match_state_after"] = record.state_after
            track_dict["score_event"] = record.score_event
            track_dict["frame_observations"] = [
                {
                    "frame": obs.frame,
                    "x": obs.x,
                    "y": obs.y,
                    "radius_px": obs.radius_px,
                    "radius_med6_px": obs.smoothed_radius_px,
                }
                for obs in observations
            ]
            if self._court.enabled and self._court.geometry is not None:
                track_dict["court_positions"] = [
                    [list(self._court.to_court(obs.x, obs.y)), obs.frame] for obs in observations
                ]
                track_dict["court_info"] = {
                    "image_width": self._court.geometry.image_width,
                    "image_height": self._court.geometry.image_height,
                    "court_points_count": len(self._court.geometry.keypoints),
                    "has_court_transform": self._court.geometry is not None,
                    "camera_position": self._court.camera_position,
                    "court_mode": "beach" if self.config.beach else "classic",
                    "court_length_m": self._court.court_length_m,
                    "court_width_m": self._court.court_width_m,
                    "cm_per_px": self._court.cm_per_px_scale,
                    "net_height_cm": NET_HEIGHT_CM,
                    "ball_diameter_cm": BALL_DIAMETER_CM,
                    "csv_radius_calibration": {
                        "radius_lower_px": self._csv_size_calibration.radius_lower_px,
                        "radius_upper_px": self._csv_size_calibration.radius_upper_px,
                        "radius_median_px": self._csv_size_calibration.radius_median_px,
                        "radius_near_far_threshold_px": self._csv_size_calibration.radius_near_far_threshold_px,
                        "filtered_count": self._csv_size_calibration.filtered_count,
                    },
                }
            track_dict["tracking_scale"] = {
                "reference_width_px": int(REFERENCE_VIDEO_WIDTH),
                "frame_width_px": self._resolved_video_width(),
                "frame_height_px": self._resolved_video_height(),
                "frame_size_source": (
                    "cli_override"
                    if self.config.video_width is not None or self.config.video_height is not None
                    else "court_json"
                    if self._court.geometry is not None
                    else "unknown"
                ),
                "frame_width_coeff": self._frame_width_scale,
                "base_max_distance": self.config.max_distance,
                "scaled_max_distance": self._scaled_max_distance,
            }
            file_path = os.path.join(tracks_dir, f"track_{track.track_id:04d}.json")
            with open(file_path, "w", encoding="utf-8") as handle:
                json.dump(track_dict, handle, indent=2, ensure_ascii=False)

    def _build_analysis_records(self) -> list[TrackAnalysisRecord]:
        records: list[TrackAnalysisRecord] = []
        for track in self.tracks:
            observations = self._feature_extractor.build_frame_observations(track, self._observations_by_frame)
            trajectory_analysis, rally_features, rally_classification = self._feature_extractor.extract_features(
                track, observations
            )
            records.append(
                TrackAnalysisRecord(
                    track=track,
                    observations=observations,
                    trajectory_analysis=trajectory_analysis,
                    rally_features=rally_features,
                    rally_classification=rally_classification,
                    state_before={},
                    state_after={},
                )
            )
        return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calculate tracks from CSV with court-aware analysis")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to ball.csv")
    parser.add_argument("--court_json_path", type=str, help="Path to court coordinates JSON file")
    parser.add_argument(
        "--video_width",
        type=int,
        default=None,
        help="Source video width override. If omitted, uses size from court JSON first.",
    )
    parser.add_argument(
        "--video_height",
        type=int,
        default=None,
        help="Source video height override. If omitted, uses size from court JSON first.",
    )
    parser.add_argument("--beach", action="store_true", help="Use beach volleyball court dimensions")
    parser.add_argument("--output_dir", type=str, default="output", help="Root output directory for JSON")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="Frames per second")
    parser.add_argument("--max_distance", type=float, default=DEFAULT_MAX_DISTANCE, help="Max tracking distance")
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
        help="Reserved for compatibility",
    )
    parser.add_argument(
        "--min_y_displacement",
        type=float,
        default=DEFAULT_MIN_Y_DISPLACEMENT,
        help="Reserved for compatibility",
    )
    parser.add_argument(
        "--bounce_frames",
        type=int,
        default=DEFAULT_BOUNCE_FRAMES,
        help="Reserved for compatibility",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.verbose)
    config = TrackCalculatorConfig(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        fps=args.fps,
        max_distance=args.max_distance,
        min_duration_sec=args.min_duration_sec,
        max_x_displacement=args.max_x_displacement,
        min_y_displacement=args.min_y_displacement,
        bounce_frames=args.bounce_frames,
        court_json_path=args.court_json_path,
        video_width=args.video_width,
        video_height=args.video_height,
        beach=args.beach,
    )
    calculator = TrackCalculatorWithCourt(config)
    calculator.run()


if __name__ == "__main__":
    main()
