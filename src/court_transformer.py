"""Court geometry loading and coordinate transforms."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from constants import (
    COURT_LENGTH_M,
    COURT_WIDTH_M,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    NET_HEIGHT_M,
)
from models import CourtGeometry, Point2D


@dataclass
class CourtTransformResult:
    geometry: Optional[CourtGeometry]
    matrix: Optional[np.ndarray]


class CourtTransformer:
    """Loads court geometry and provides coordinate transforms."""

    def __init__(self, court_json_path: Optional[str]) -> None:
        self._court_json_path = court_json_path

    def load(self) -> CourtTransformResult:
        if not self._court_json_path:
            return CourtTransformResult(None, None)

        if not os.path.exists(self._court_json_path):
            return CourtTransformResult(None, None)

        try:
            with open(self._court_json_path, "r", encoding="utf-8") as f:
                court_data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return CourtTransformResult(None, None)

        images = court_data.get("images", [])
        image_width = DEFAULT_IMAGE_WIDTH
        image_height = DEFAULT_IMAGE_HEIGHT
        if images:
            image_width = images[0].get("width", image_width)
            image_height = images[0].get("height", image_height)

        annotations = court_data.get("annotations", [])
        if not annotations:
            return CourtTransformResult(None, None)

        keypoints_raw = annotations[0].get("keypoints", [])
        if len(keypoints_raw) < 24:
            return CourtTransformResult(None, None)

        keypoints: list[Point2D] = []
        for i in range(0, len(keypoints_raw), 3):
            if i + 2 >= len(keypoints_raw):
                break
            x, y, visibility = keypoints_raw[i], keypoints_raw[i + 1], keypoints_raw[i + 2]
            if visibility > 0:
                keypoints.append((float(x), float(y)))

        if len(keypoints) < 4:
            return CourtTransformResult(None, None)

        geometry = CourtGeometry(
            length_m=COURT_LENGTH_M,
            width_m=COURT_WIDTH_M,
            net_height_m=NET_HEIGHT_M,
            image_width=int(image_width),
            image_height=int(image_height),
            keypoints=tuple(keypoints),
        )

        matrix = self._calculate_transform(keypoints)
        return CourtTransformResult(geometry, matrix)

    @staticmethod
    def _calculate_transform(keypoints: Sequence[Point2D]) -> Optional[np.ndarray]:
        if len(keypoints) < 4:
            return None

        img_points = np.array(
            [
                keypoints[0],
                keypoints[1],
                keypoints[2],
                keypoints[3],
            ],
            dtype=np.float32,
        )
        court_points = np.array(
            [
                [-COURT_LENGTH_M / 2, -COURT_WIDTH_M / 2],
                [COURT_LENGTH_M / 2, -COURT_WIDTH_M / 2],
                [COURT_LENGTH_M / 2, COURT_WIDTH_M / 2],
                [-COURT_LENGTH_M / 2, COURT_WIDTH_M / 2],
            ],
            dtype=np.float32,
        )

        try:
            return cv2.getPerspectiveTransform(img_points, court_points)
        except cv2.error:
            return None


class CoordinateTransformer:
    """Transforms image points to court coordinates."""

    def __init__(self, geometry: Optional[CourtGeometry], matrix: Optional[np.ndarray]) -> None:
        self._geometry = geometry
        self._matrix = matrix

    def to_court(self, x: float, y: float) -> Tuple[float, float]:
        if not self._geometry:
            return float(x), float(y)

        if self._matrix is None:
            norm_x = x / max(1, self._geometry.image_width)
            norm_y = y / max(1, self._geometry.image_height)
            return (
                norm_x * COURT_LENGTH_M - COURT_LENGTH_M / 2,
                norm_y * COURT_WIDTH_M - COURT_WIDTH_M / 2,
            )

        point = np.array([[x, y]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point.reshape(-1, 1, 2), self._matrix)
        return (float(transformed[0][0][0]), float(transformed[0][0][1]))
