#!/usr/bin/env python3
"""Interactive viewer for ball tracks, player detections and contact labels.

    uv run src/show_rally.py ../uploads/mix/beach-mixt/tracks /path/to/video.mp4 \
        --players_json_path ../uploads/mix/beach-mixt_predictions.json

Keys:
    space  play / pause          a / d   step one frame back / forward
    w / s  jump 15 frames        n / p   next / previous track
    v      save selected track video to --output_dir
    t      ball path on/off      h       help on/off
    q, ESC quit
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import cv2
import numpy as np

from court_transformer import CourtTransformer
from player_contacts import PlayerStore

LOG = logging.getLogger(__name__)

WINDOW_NAME = "Volleyball tracks"
CONTEXT_SECONDS = 2.0
CONTACT_HIGHLIGHT_FRAMES = 10
JUMP_FRAMES = 15
TIMELINE_HEIGHT = 46
HEADER_HEIGHT = 96
FOOTER_HEIGHT = 44

WHITE = (255, 255, 255)
GREY = (150, 150, 150)
DARK = (40, 40, 40)
YELLOW = (0, 255, 255)
BALL_COLOR = (60, 60, 255)
PATH_PAST = (255, 190, 60)
PATH_FUTURE = (110, 90, 40)
START_COLOR = (255, 120, 255)
END_COLOR = (120, 255, 255)
RALLY_START_COLOR = (120, 255, 120)

CONTACT_COLORS = {
    "serve": (255, 0, 255),
    "attack": (60, 60, 255),
    "block": (0, 140, 255),
    "overhead_pass": (80, 230, 80),
    "dig": (255, 220, 0),
    "low_touch": (160, 160, 160),
}

COURT_COLOR = (90, 200, 90)
NET_COLOR = (200, 200, 90)
COURT_POINT_COLOR = (0, 0, 255)
PLAYER_BOX_COLOR = (255, 210, 80)
BALL_DIAMETER_M = 0.21

HELP_LINES = [
    "space play/pause   a/d frame -1/+1   w/s +15/-15",
    "n/p next/prev track   v save track video   b box view   t path   h help   q quit",
]


@dataclass
class TrackView:
    """One track_*.json prepared for drawing."""

    path: str
    data: dict[str, Any]
    observations: dict[int, dict[str, Any]] = field(default_factory=dict)
    contacts: list[dict[str, Any]] = field(default_factory=list)
    ball_start: Optional[dict[str, Any]] = None
    ball_end: Optional[dict[str, Any]] = None
    court_view_frames: dict[int, dict[str, Any]] = field(default_factory=dict)
    court_length_m: float = 0.0
    court_width_m: float = 0.0
    first_frame: int = 0
    last_frame: int = 0

    @classmethod
    def load(cls, path: str) -> Optional["TrackView"]:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            LOG.warning("Skipping %s: %s", os.path.basename(path), exc)
            return None

        observations = {
            int(item["frame"]): item
            for item in data.get("frame_observations", [])
            if item.get("frame") is not None
        }
        if not observations:
            LOG.warning("Skipping %s: no frame observations", os.path.basename(path))
            return None

        interaction = data.get("player_interaction") or {}
        court_view = data.get("court_view") or {}
        court_view_frames = {
            int(item["frame"]): item
            for item in court_view.get("frames", [])
            if item.get("frame") is not None
        }
        frames = sorted(observations)
        return cls(
            path=path,
            data=data,
            observations=observations,
            contacts=list(interaction.get("contacts") or []),
            ball_start=interaction.get("ball_start"),
            ball_end=interaction.get("ball_end"),
            court_view_frames=court_view_frames,
            court_length_m=float(court_view.get("court_length_m", 0.0)),
            court_width_m=float(court_view.get("court_width_m", 0.0)),
            first_frame=frames[0],
            last_frame=frames[-1],
        )

    @property
    def track_id(self) -> int:
        return int(self.data.get("track_id", 0))

    @property
    def label(self) -> str:
        return str(self.data.get("rally_classification", {}).get("label", "unknown"))

    @property
    def is_rally(self) -> bool:
        return bool(self.data.get("rally_classification", {}).get("is_rally"))

    def source_size(self) -> Optional[tuple[int, int]]:
        court = self.data.get("court_info") or {}
        width = court.get("image_width")
        height = court.get("image_height")
        if width and height:
            return int(width), int(height)

        scale = self.data.get("tracking_scale") or {}
        width = scale.get("frame_width_px")
        height = scale.get("frame_height_px")
        if width and height:
            return int(width), int(height)
        return None

    def summary_lines(self) -> list[str]:
        classification = self.data.get("rally_classification", {})
        trajectory = self.data.get("trajectory_analysis", {})
        summary = (self.data.get("player_interaction") or {}).get("summary", {})
        features = self.data.get("rally_features", {})

        reason = classification.get("not_rally_reason")
        first = "%s%s  conf %.2f  %.1fs" % (
            self.label,
            f" ({reason})" if reason else "",
            float(classification.get("rally_confidence", 0.0)),
            float(features.get("duration_sec", 0.0)),
        )
        second = "serve %s (%s)   %s -> %s" % (
            trajectory.get("serve_side", "unknown"),
            trajectory.get("serve_side_source", "n/a"),
            summary.get("start_origin", "n/a"),
            summary.get("end_origin", "n/a"),
        )
        if summary.get("rally_start_frame"):
            second += "   rally from f%d (prep %.1fs)" % (
                int(summary["rally_start_frame"]),
                float(summary.get("preparation_sec", 0.0)),
            )
        if summary.get("possessions"):
            second += "   touches " + "-".join(
                "%s%d" % (item["side"][:1], item["touches"]) for item in summary["possessions"][:12]
            )
        if summary.get("contact_count"):
            second += "   contacts %d: dig %d, over %d, att %d, block %d, serve %d, switches %d" % (
                summary.get("contact_count", 0),
                summary.get("dig_count", 0),
                summary.get("overhead_count", 0),
                summary.get("attack_count", 0),
                summary.get("block_count", 0),
                summary.get("serve_count", 0),
                summary.get("side_switch_count", 0),
            )
        return [first, second]


class RallyViewer:
    """Frame-by-frame playback of tracks over the source video."""

    def __init__(
        self,
        tracks: Sequence[TrackView],
        video_path: str,
        players: PlayerStore,
        fps: Optional[float] = None,
        court_keypoints: Sequence[Optional[tuple[float, float]]] = (),
        output_dir: Optional[str] = None,
    ) -> None:
        self._tracks = list(tracks)
        self._players = players
        self._court_keypoints = tuple(court_keypoints)
        self._output_dir = output_dir
        self._capture = cv2.VideoCapture(video_path)
        if not self._capture.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        self._video_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._video_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = fps or self._capture.get(cv2.CAP_PROP_FPS) or 30.0
        self._context_frames = max(1, int(round(self._fps * CONTEXT_SECONDS)))

        self._index = 0
        self._frame = self._tracks[0].first_frame
        self._playing = False
        self._show_box_view = True
        self._show_path = True
        self._show_help = True
        self._schematic = False
        self._cached_frame: Optional[np.ndarray] = None
        self._cached_frame_index: Optional[int] = None

    def close(self) -> None:
        self._capture.release()

    @property
    def track(self) -> TrackView:
        return self._tracks[self._index]

    def run(self) -> None:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            WINDOW_NAME,
            min(1800, self._video_width * (2 if self._show_box_view else 1)),
            min(1000, self._video_height + 160),
        )
        LOG.info("%s", " | ".join(HELP_LINES))

        while True:
            canvas = self.render()
            cv2.imshow(WINDOW_NAME, canvas)
            delay = max(1, int(1000 / self._fps)) if self._playing else 30
            if not self._handle_key(cv2.waitKey(delay) & 0xFF):
                break
            if self._playing:
                if self._frame >= self._last_visible_frame():
                    self._playing = False
                else:
                    self._frame += 1

        cv2.destroyAllWindows()

    def render(self) -> np.ndarray:
        if self._schematic:
            frame = np.zeros((self._video_height, self._video_width, 3), dtype=np.uint8)
        else:
            frame = self._read_frame(self._frame)
            if frame is None:
                frame = np.zeros((self._video_height, self._video_width, 3), dtype=np.uint8)
                cv2.putText(
                    frame, f"no frame {self._frame}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, GREY, 2
                )
            else:
                frame = frame.copy()

        scale_x, scale_y = self._scale()
        self._draw_court(frame, scale_x, scale_y)
        self._draw_players(frame, scale_x, scale_y, schematic=self._schematic)
        if self._show_path:
            self._draw_path(frame, scale_x, scale_y)
        self._draw_ball(frame, scale_x, scale_y)
        self._draw_edges(frame, scale_x, scale_y)
        self._draw_active_contact(frame, scale_x, scale_y)

        if self._show_box_view:
            box_view = np.zeros_like(frame)
            self._draw_top_down_view(box_view)
            frame = np.hstack((box_view, frame))
        return self._compose(frame)

    def _court_to_top_down(
        self,
        court_x: float,
        court_y: float,
        frame_shape: tuple[int, ...],
    ) -> tuple[int, int]:
        """Court metres (length, width) to a bird's-eye panel pixel."""
        height, width = frame_shape[:2]
        court_length = max(self.track.court_length_m, 1.0)
        court_width = max(self.track.court_width_m, 1.0)
        free_zone_m = 2.0
        usable_height = height - 70
        usable_width = width - 80
        scale = min(
            usable_height / (court_length + 2.0 * free_zone_m),
            usable_width / (court_width + 2.0 * free_zone_m),
        )
        center_x = width / 2.0
        center_y = height / 2.0 + 8.0
        pixel_x = center_x + float(court_y) * scale
        pixel_y = center_y - float(court_x) * scale
        return (
            int(np.clip(round(pixel_x), 3, width - 4)),
            int(np.clip(round(pixel_y), 36, height - 4)),
        )

    def _draw_top_down_view(self, frame: np.ndarray) -> None:
        """Court, players, ball and court-plane ball direction from calculator."""
        cv2.putText(frame, "COURT TOP VIEW", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 1)
        if not self.track.court_view_frames:
            cv2.putText(
                frame,
                "no court_view: rerun track_calculator_with_court.py",
                (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                GREY,
                1,
            )
            return

        half_length = self.track.court_length_m / 2.0
        half_width = self.track.court_width_m / 2.0
        corners = np.asarray(
            [
                self._court_to_top_down(-half_length, -half_width, frame.shape),
                self._court_to_top_down(half_length, -half_width, frame.shape),
                self._court_to_top_down(half_length, half_width, frame.shape),
                self._court_to_top_down(-half_length, half_width, frame.shape),
            ],
            dtype=np.int32,
        )
        cv2.polylines(frame, [corners], True, COURT_COLOR, 2)
        net_left = self._court_to_top_down(0.0, -half_width, frame.shape)
        net_right = self._court_to_top_down(0.0, half_width, frame.shape)
        cv2.line(frame, net_left, net_right, NET_COLOR, 3)
        cv2.putText(frame, "FAR", (corners[1][0] - 42, corners[1][1] + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1)
        cv2.putText(frame, "NEAR", (corners[0][0] - 48, corners[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1)

        payload = self.track.court_view_frames.get(self._frame)
        if payload is None:
            return
        for player in payload.get("players") or []:
            center = self._court_to_top_down(
                float(player["court_x"]), float(player["court_y"]), frame.shape
            )
            color = PLAYER_BOX_COLOR
            cv2.circle(frame, center, 8, color, -1)
            cv2.putText(
                frame,
                str(player["track_id"]),
                (center[0] + 11, center[1] - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        ball = payload.get("ball")
        if not ball:
            return
        ball_center = self._court_to_top_down(
            float(ball["court_x"]), float(ball["court_y"]), frame.shape
        )
        panel_scale = min(
            (frame.shape[0] - 70) / (max(self.track.court_length_m, 1.0) + 4.0),
            (frame.shape[1] - 80) / (max(self.track.court_width_m, 1.0) + 4.0),
        )
        ball_radius = max(2, int(round(BALL_DIAMETER_M * panel_scale / 2.0)))
        cv2.circle(frame, ball_center, ball_radius, BALL_COLOR, -1)
        cv2.circle(frame, ball_center, ball_radius + 2, WHITE, 1)

        direction = payload.get("ball_direction")
        if not direction:
            return
        # Court x is the length axis (screen vertical), court y is width
        # (screen horizontal). Normalize because only direction is meaningful;
        # an airborne ball projected onto the ground plane has unstable speed.
        dx = float(direction.get("dx", 0.0))
        dy = float(direction.get("dy", 0.0))
        norm = float(np.hypot(dx, dy))
        if norm <= 1e-6:
            return
        arrow_length = 42.0
        arrow_end = (
            int(round(ball_center[0] + arrow_length * dy / norm)),
            int(round(ball_center[1] - arrow_length * dx / norm)),
        )
        cv2.arrowedLine(frame, ball_center, arrow_end, BALL_COLOR, 3, cv2.LINE_AA, tipLength=0.28)

    def _scale(self) -> tuple[float, float]:
        source = self.track.source_size()
        if not source:
            return 1.0, 1.0
        return self._video_width / float(source[0]), self._video_height / float(source[1])

    def _read_frame(self, index: int) -> Optional[np.ndarray]:
        if self._cached_frame_index == index and self._cached_frame is not None:
            return self._cached_frame

        if self._cached_frame_index is None or index != self._cached_frame_index + 1:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self._capture.read()
        if not ok:
            self._cached_frame_index = None
            self._cached_frame = None
            return None

        self._cached_frame_index = index
        self._cached_frame = frame
        return frame

    def _draw_court(self, frame: np.ndarray, scale_x: float, scale_y: float) -> None:
        """Court and net as the analysis sees them, over a blank frame."""
        points = self._court_keypoints
        visible = {index: point for index, point in enumerate(points) if point is not None}
        if len(visible) < 4:
            cv2.putText(
                frame,
                "no court json: pass --court_json_path to draw the court",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                GREY,
                1,
            )
            return

        def pixel(index: int) -> tuple[int, int]:
            x, y = visible[index]
            return int(x * scale_x), int(y * scale_y)

        # Do not connect across an invisible point (point 3 is absent in the
        # beach-mixt annotation); drawing that diagonal would mislead review.
        for first, second in ((0, 1), (1, 2), (2, 3), (3, 0)):
            if first in visible and second in visible:
                cv2.line(frame, pixel(first), pixel(second), COURT_COLOR, 2)
        if all(index in visible for index in (4, 5, 6, 7)):
            ground = [pixel(index) for index in (4, 5)]
            top = [pixel(index) for index in (6, 7)]
            cv2.line(frame, ground[0], ground[1], NET_COLOR, 1)
            cv2.line(frame, top[0], top[1], NET_COLOR, 2)
            cv2.line(frame, ground[0], top[0], NET_COLOR, 1)
            cv2.line(frame, ground[1], top[1], NET_COLOR, 1)

        for number, (x, y) in ((index + 1, point) for index, point in visible.items()):
            center = (int(x * scale_x), int(y * scale_y))
            cv2.circle(frame, center, 4, COURT_POINT_COLOR, -1)
            self._put_label(
                frame,
                str(number),
                (center[0] + 7, center[1] - 7),
                COURT_POINT_COLOR,
                scale=0.55,
                thickness=2,
            )

    def _draw_players(
        self,
        frame: np.ndarray,
        scale_x: float,
        scale_y: float,
        schematic: bool = False,
    ) -> None:
        for box in self._players.boxes_at(self._frame):
            # In tracking/follow mode all players must have equal styling;
            # action highlighting is reserved for the ball/contact markers.
            color = PLAYER_BOX_COLOR
            p1 = (int(box.x1 * scale_x), int(box.y1 * scale_y))
            p2 = (int(box.x2 * scale_x), int(box.y2 * scale_y))
            self._dashed_rectangle(frame, p1, p2, color, 2)
            if schematic:
                cv2.drawMarker(
                    frame,
                    (int(box.foot_x * scale_x), int(box.foot_y * scale_y)),
                    color,
                    cv2.MARKER_TRIANGLE_UP,
                    8,
                    1,
                )
            cv2.putText(
                frame,
                str(box.track_id),
                (p1[0], max(12, p1[1] - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
            )

    @staticmethod
    def _dashed_rectangle(
        frame: np.ndarray,
        p1: tuple[int, int],
        p2: tuple[int, int],
        color: tuple[int, int, int],
        thickness: int = 1,
        dash: int = 8,
    ) -> None:
        x1, y1 = p1
        x2, y2 = p2
        for start, end in (
            ((x1, y1), (x2, y1)),
            ((x2, y1), (x2, y2)),
            ((x2, y2), (x1, y2)),
            ((x1, y2), (x1, y1)),
        ):
            length = max(abs(end[0] - start[0]), abs(end[1] - start[1]))
            if length == 0:
                continue
            for offset in range(0, length, dash * 2):
                a, b = offset / length, min(offset + dash, length) / length
                point_a = (round(start[0] + (end[0] - start[0]) * a), round(start[1] + (end[1] - start[1]) * a))
                point_b = (round(start[0] + (end[0] - start[0]) * b), round(start[1] + (end[1] - start[1]) * b))
                cv2.line(frame, point_a, point_b, color, thickness, cv2.LINE_AA)

    def _draw_path(self, frame: np.ndarray, scale_x: float, scale_y: float) -> None:
        track = self.track
        previous: Optional[tuple[int, int]] = None
        previous_frame: Optional[int] = None
        for number in sorted(track.observations):
            observation = track.observations[number]
            point = (int(observation["x"] * scale_x), int(observation["y"] * scale_y))
            if previous is not None and number - (previous_frame or number) <= 8:
                color = PATH_PAST if number <= self._frame else PATH_FUTURE
                cv2.line(frame, previous, point, color, 2 if number <= self._frame else 1)
            previous = point
            previous_frame = number

        for contact in track.contacts:
            observation = track.observations.get(int(contact["frame"]))
            if observation is None:
                continue
            center = (int(observation["x"] * scale_x), int(observation["y"] * scale_y))
            cv2.drawMarker(
                frame,
                center,
                CONTACT_COLORS.get(contact.get("contact_type"), WHITE),
                cv2.MARKER_TILTED_CROSS,
                14,
                2,
            )

    def _draw_ball(self, frame: np.ndarray, scale_x: float, scale_y: float) -> None:
        observation = self.track.observations.get(self._frame)
        if observation is None:
            return
        center = (int(observation["x"] * scale_x), int(observation["y"] * scale_y))
        radius = observation.get("radius_med6_px") or observation.get("radius_px") or 8.0
        cv2.circle(frame, center, max(6, int(float(radius) * scale_x)), BALL_COLOR, 2)
        cv2.circle(frame, center, 2, BALL_COLOR, -1)

    def _draw_edges(self, frame: np.ndarray, scale_x: float, scale_y: float) -> None:
        for name, edge, color in (
            ("start", self.track.ball_start, START_COLOR),
            ("end", self.track.ball_end, END_COLOR),
        ):
            if not edge:
                continue
            center = (int(edge["ball_x"] * scale_x), int(edge["ball_y"] * scale_y))
            cv2.circle(frame, center, 16, color, 2)
            if abs(int(edge["frame"]) - self._frame) > self._fps:
                continue
            self._put_label(
                frame,
                "%s: %s%s" % (
                    name,
                    edge.get("origin", "?"),
                    f" p{edge['player_track_id']}" if edge.get("player_track_id") is not None else "",
                ),
                (center[0] + 20, center[1] - 22),
                color,
            )

    def _draw_active_contact(self, frame: np.ndarray, scale_x: float, scale_y: float) -> None:
        contact = self._active_contact()
        if contact is None:
            return
        center = (int(contact["ball_x"] * scale_x), int(contact["ball_y"] * scale_y))
        color = CONTACT_COLORS.get(contact.get("contact_type"), WHITE)
        cv2.circle(frame, center, 22, color, 2)
        text = "%s  h=%.2f  conf %.2f  p%d  %s  f%d" % (
            contact.get("contact_type", "?"),
            float(contact.get("depth_ratio", 0.0)),
            float(contact.get("confidence", 0.0)),
            int(contact.get("player_track_id", -1)),
            contact.get("side", "?"),
            int(contact["frame"]),
        )
        self._put_label(frame, text, (center[0] + 26, center[1] + 8), color, scale=0.6, thickness=2)

    @staticmethod
    def _put_label(
        frame: np.ndarray,
        text: str,
        anchor: tuple[int, int],
        color: tuple[int, int, int],
        scale: float = 0.5,
        thickness: int = 1,
    ) -> None:
        """Draws text on a dark plate, kept inside the frame."""
        (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        x = int(np.clip(anchor[0], 2, max(2, frame.shape[1] - width - 4)))
        y = int(np.clip(anchor[1], height + 4, frame.shape[0] - 4))
        cv2.rectangle(frame, (x - 2, y - height - 4), (x + width + 2, y + 4), (0, 0, 0), -1)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

    def _active_contact(self) -> Optional[dict[str, Any]]:
        best = None
        for contact in self.track.contacts:
            distance = abs(int(contact["frame"]) - self._frame)
            if distance <= CONTACT_HIGHLIGHT_FRAMES and (best is None or distance < best[0]):
                best = (distance, contact)
        return best[1] if best else None

    def _compose(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        header = np.zeros((HEADER_HEIGHT, width, 3), dtype=np.uint8)
        track = self.track

        title = "[%d/%d] track %04d   frame %d (%.2fs)" % (
            self._index + 1,
            len(self._tracks),
            track.track_id,
            self._frame,
            self._frame / self._fps if self._fps else 0.0,
        )
        if not track.first_frame <= self._frame <= track.last_frame:
            title += "   <outside track>"
        if self._playing:
            title += "   PLAY"
        if self._schematic:
            title += "   SCHEMATIC"
        if self._show_box_view:
            title += "   BOX VIEW"

        color = (120, 255, 120) if track.is_rally else (120, 120, 255)
        cv2.putText(header, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)
        for offset, line in enumerate(track.summary_lines()):
            cv2.putText(header, line, (10, 46 + offset * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

        parts = [header, frame, self._timeline(width)]
        if self._show_help:
            footer = np.zeros((FOOTER_HEIGHT, width, 3), dtype=np.uint8)
            for offset, line in enumerate(HELP_LINES):
                cv2.putText(
                    footer, line, (10, 18 + offset * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1
                )
            parts.append(footer)
        return np.vstack(parts)

    def _timeline(self, width: int) -> np.ndarray:
        strip = np.zeros((TIMELINE_HEIGHT, width, 3), dtype=np.uint8)
        track = self.track
        left, right = 10, width - 10
        span = max(1, self._last_visible_frame() - self._first_visible_frame())

        def position(frame_index: int) -> int:
            ratio = (frame_index - self._first_visible_frame()) / span
            return int(left + ratio * (right - left))

        cv2.line(strip, (left, 24), (right, 24), DARK, 6)
        cv2.line(strip, (position(track.first_frame), 24), (position(track.last_frame), 24), GREY, 6)

        for edge, color in ((track.ball_start, START_COLOR), (track.ball_end, END_COLOR)):
            if edge:
                x = position(int(edge["frame"]))
                cv2.line(strip, (x, 12), (x, 36), color, 2)
        for contact in track.contacts:
            x = position(int(contact["frame"]))
            cv2.line(strip, (x, 14), (x, 34), CONTACT_COLORS.get(contact.get("contact_type"), WHITE), 2)

        rally_start = (self.data_summary() or {}).get("rally_start_frame")
        if rally_start:
            x = position(int(rally_start))
            cv2.line(strip, (x, 8), (x, 40), RALLY_START_COLOR, 2)

        cursor = position(self._frame)
        cv2.line(strip, (cursor, 6), (cursor, 42), YELLOW, 2)
        cv2.putText(strip, str(track.first_frame), (left, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, GREY, 1)
        cv2.putText(strip, str(track.last_frame), (right - 44, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, GREY, 1)
        return strip

    def data_summary(self) -> dict[str, Any]:
        return (self.track.data.get("player_interaction") or {}).get("summary", {})

    def _first_visible_frame(self) -> int:
        return max(0, self.track.first_frame - self._context_frames)

    def _last_visible_frame(self) -> int:
        return self.track.last_frame + self._context_frames

    def _handle_key(self, key: int) -> bool:
        if key in (ord("q"), 27):
            return False
        if key == ord(" "):
            self._playing = not self._playing
        elif key == ord("a"):
            self._seek(self._frame - 1)
        elif key == ord("d"):
            self._seek(self._frame + 1)
        elif key == ord("s"):
            self._seek(self._frame - JUMP_FRAMES)
        elif key == ord("w"):
            self._seek(self._frame + JUMP_FRAMES)
        elif key == ord("n"):
            self.select(self._index + 1)
        elif key == ord("p"):
            self.select(self._index - 1)
        elif key == ord("v"):
            self.save_selected_rally()
        elif key == ord("t"):
            self._show_path = not self._show_path
        elif key == ord("b"):
            self._show_box_view = not self._show_box_view
        elif key == ord("h"):
            self._show_help = not self._show_help
        return True

    def save_selected_rally(self) -> Optional[str]:
        """Write the selected track with overlays to ``--output_dir``."""
        if not self._output_dir:
            LOG.warning("Cannot save rally: --output_dir was not provided")
            return None

        os.makedirs(self._output_dir, exist_ok=True)
        output_path = os.path.join(self._output_dir, f"track_{self.track.track_id:04d}.mp4")
        current_frame = self._frame
        was_playing = self._playing
        self._playing = False
        self._cached_frame = None
        self._cached_frame_index = None
        writer: Optional[cv2.VideoWriter] = None
        try:
            for frame_index in range(self.track.first_frame, self.track.last_frame + 1):
                self._frame = frame_index
                rendered = self.render()
                if writer is None:
                    height, width = rendered.shape[:2]
                    writer = cv2.VideoWriter(
                        output_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        float(self._fps),
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise OSError(f"Cannot open video writer: {output_path}")
                writer.write(rendered)
        finally:
            if writer is not None:
                writer.release()
            self._frame = current_frame
            self._playing = was_playing
            self._cached_frame = None
            self._cached_frame_index = None

        LOG.info("Rally video written to %s", output_path)
        return output_path

    def _seek(self, frame_index: int) -> None:
        self._playing = False
        self._frame = int(np.clip(frame_index, self._first_visible_frame(), self._last_visible_frame()))

    def select(self, index: int) -> None:
        self._index = int(np.clip(index, 0, len(self._tracks) - 1))
        self._playing = False
        self._frame = self.track.first_frame
        LOG.info("track %04d  %s", self.track.track_id, " | ".join(self.track.summary_lines()))


def load_tracks(tracks_dir: str) -> list[TrackView]:
    paths = sorted(glob.glob(os.path.join(tracks_dir, "track_*.json")))
    if not paths:
        raise FileNotFoundError(f"No track_*.json in {tracks_dir}")
    tracks = [view for view in (TrackView.load(path) for path in paths) if view is not None]
    if not tracks:
        raise ValueError(f"No usable tracks in {tracks_dir}")
    return sorted(tracks, key=lambda item: item.first_frame)


def discover_auxiliary_json(tracks_dir: str) -> tuple[Optional[str], Optional[str]]:
    """Find prediction/court JSON files produced next to a clip's tracks folder."""
    clip_dir = os.path.dirname(os.path.abspath(os.path.normpath(tracks_dir)))
    output_dir = os.path.dirname(clip_dir)
    stem = os.path.basename(clip_dir)
    # Prediction exports commonly put tracks under ``<clip>_predict/tracks``
    # while the sidecar files remain ``<clip>_predict.json`` and
    # ``<clip>_court.json``.
    stems = [stem]
    if stem.endswith("_predict"):
        stems.insert(0, stem[: -len("_predict")])

    def first_existing(suffixes: Sequence[str]) -> Optional[str]:
        for candidate_stem in stems:
            for suffix in suffixes:
                candidate = os.path.join(output_dir, candidate_stem + suffix)
                if os.path.isfile(candidate):
                    return candidate
        return None

    return first_existing(("_predict.json", "_predictions.json")), first_existing(
        ("_court.json", "_coort.json")
    )


def load_court_keypoints(court_path: Optional[str]) -> tuple[Optional[tuple[float, float]], ...]:
    """Load court points without renumbering invisible JSON keypoints."""
    if not court_path:
        return ()
    try:
        with open(court_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return ()

    raw_points = data.get("keypoints") or []
    if raw_points:
        points: list[Optional[tuple[float, float]]] = []
        for point in raw_points:
            if not point.get("visible", False) or point.get("x") is None or point.get("y") is None:
                points.append(None)
            else:
                points.append((float(point["x"]), float(point["y"])))
        return tuple(points)

    annotations = data.get("annotations") or []
    if annotations:
        values = annotations[0].get("keypoints") or []
        points = []
        for index in range(0, len(values), 3):
            if index + 2 >= len(values) or values[index + 2] <= 0:
                points.append(None)
            else:
                points.append((float(values[index]), float(values[index + 1])))
        return tuple(points)
    return ()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize ball tracks, players and contacts")
    parser.add_argument("tracks_dir", type=str, help="Directory with track_*.json")
    parser.add_argument("video_file", type=str, help="Source video")
    parser.add_argument(
        "--players_json_path",
        type=str,
        default=None,
        help="Player detections JSON (auto-detected next to the clip when omitted)",
    )
    parser.add_argument(
        "--court_json_path",
        type=str,
        default=None,
        help="Court annotation JSON (auto-detected next to the clip when omitted)",
    )
    parser.add_argument("--fps", type=float, default=None, help="Override video FPS")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for v-key annotated rally videos",
    )
    parser.add_argument("--track", type=int, default=0, help="Index of the track to start from")
    parser.add_argument(
        "--snapshot",
        type=str,
        default=None,
        help="Render the first frame of the selected track to this file and exit (no GUI)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    tracks = load_tracks(args.tracks_dir)
    LOG.info("Loaded %d tracks from %s", len(tracks), args.tracks_dir)
    discovered_players, discovered_court = discover_auxiliary_json(args.tracks_dir)
    players_path = args.players_json_path or discovered_players
    court_path = args.court_json_path or discovered_court
    if args.players_json_path is None and players_path:
        LOG.info("Auto-detected player boxes: %s", players_path)
    if args.court_json_path is None and court_path:
        LOG.info("Auto-detected court: %s", court_path)

    players = PlayerStore(players_path)
    court = CourtTransformer(court_path).load()
    keypoints = load_court_keypoints(court_path)
    if not keypoints and court.geometry is not None:
        keypoints = court.geometry.keypoints

    viewer = RallyViewer(tracks, args.video_file, players, args.fps, keypoints, args.output_dir)
    try:
        viewer.select(args.track)
        if args.snapshot:
            cv2.imwrite(args.snapshot, viewer.render())
            LOG.info("Snapshot written to %s", args.snapshot)
            return
        viewer.run()
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
