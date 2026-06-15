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

    def __init__(
        self,
        court_json_path: Optional[str],
        court_length_m: float = COURT_LENGTH_M,
        court_width_m: float = COURT_WIDTH_M,
    ) -> None:
        self._court_json_path = court_json_path
        self._court_length_m = court_length_m
        self._court_width_m = court_width_m

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

        image_width, image_height, keypoints = self._parse_court_data(court_data)

        if len(keypoints) < 4:
            return CourtTransformResult(None, None)

        geometry = CourtGeometry(
            length_m=self._court_length_m,
            width_m=self._court_width_m,
            net_height_m=NET_HEIGHT_M,
            image_width=int(image_width),
            image_height=int(image_height),
            keypoints=tuple(keypoints),
        )

        matrix = self._calculate_transform(keypoints)
        return CourtTransformResult(geometry, matrix)

    @staticmethod
    def _parse_court_data(court_data: dict) -> tuple[int, int, list[Point2D]]:
        image_width = DEFAULT_IMAGE_WIDTH
        image_height = DEFAULT_IMAGE_HEIGHT

        images = court_data.get("images", [])
        if images:
            image_width = int(images[0].get("width", image_width))
            image_height = int(images[0].get("height", image_height))

        annotations = court_data.get("annotations", [])
        if annotations:
            keypoints_raw = annotations[0].get("keypoints", [])
            keypoints: list[Point2D] = []
            for i in range(0, len(keypoints_raw), 3):
                if i + 2 >= len(keypoints_raw):
                    break
                x, y, visibility = keypoints_raw[i], keypoints_raw[i + 1], keypoints_raw[i + 2]
                if visibility > 0:
                    keypoints.append((float(x), float(y)))
            return image_width, image_height, keypoints

        image_width = int(court_data.get("frame_width", image_width))
        image_height = int(court_data.get("frame_height", image_height))
        keypoints = []
        for point in court_data.get("keypoints", []):
            if not point.get("visible", False):
                continue
            x = point.get("x")
            y = point.get("y")
            if x is None or y is None:
                continue
            keypoints.append((float(x), float(y)))
        return image_width, image_height, keypoints

    def _calculate_transform(self, keypoints: Sequence[Point2D]) -> Optional[np.ndarray]:
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
                [-self._court_length_m / 2, -self._court_width_m / 2],
                [self._court_length_m / 2, -self._court_width_m / 2],
                [self._court_length_m / 2, self._court_width_m / 2],
                [-self._court_length_m / 2, self._court_width_m / 2],
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
                norm_x * self._geometry.length_m - self._geometry.length_m / 2,
                norm_y * self._geometry.width_m - self._geometry.width_m / 2,
            )

        point = np.array([[x, y]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point.reshape(-1, 1, 2), self._matrix)
        return (float(transformed[0][0][0]), float(transformed[0][0][1]))
