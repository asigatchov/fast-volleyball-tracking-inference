"""Domain models for tracking and court geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


Point2D = Tuple[float, float]


@dataclass(frozen=True)
class CourtGeometry:
    """Volleyball court dimensions and image references."""

    length_m: float
    width_m: float
    net_height_m: float
    image_width: int
    image_height: int
    keypoints: Tuple[Point2D, ...]


@dataclass(frozen=True)
class BallDetection:
    """Single frame detection in image coordinates."""

    frame_index: int
    visible: bool
    x: float
    y: float


@dataclass
class BallTrack:
    """Maintains recent detections and visibility for tracking."""

    maxlen: int
    _points: list[Optional[Point2D]]

    def __init__(self, maxlen: int) -> None:
        self.maxlen = maxlen
        self._points = []

    def update(self, point: Optional[Point2D]) -> None:
        if len(self._points) >= self.maxlen:
            self._points.pop(0)
        self._points.append(point)

    def reset(self) -> None:
        self._points.clear()

    def points(self) -> Sequence[Point2D]:
        return [p for p in self._points if p is not None]

    def tail(self) -> Optional[Point2D]:
        for item in reversed(self._points):
            if item is not None:
                return item
        return None


@dataclass(frozen=True)
class VideoClip:
    """Represents a video clip in frame indices."""

    start_frame: int
    end_frame: int

    @property
    def length(self) -> int:
        return max(0, self.end_frame - self.start_frame + 1)
