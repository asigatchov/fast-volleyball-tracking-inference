"""Ball-player contact detection from player detections and ball trajectory.

A contact is a point where the ball trajectory breaks beyond what gravity
explains, close enough to a player to be reachable. The ball height at that
point, measured against the player's standing height, tells the technique apart:

* above the head       -> attack / serve / block
* head level           -> overhead pass ("pass sverkhu")
* chest down to knees  -> underhand dig ("priyom snizu")
* below the feet       -> ball at ground level
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

LOG = logging.getLogger(__name__)

PLAYER_CLASS_IDS = (0,)
PLAYER_CLASS_NAMES = ("player", "person")
BALL_CLASS_IDS = (1,)
BALL_CLASS_NAMES = ("ball",)

DEFAULT_PLAYER_SCORE_THRESHOLD = 0.30
DEFAULT_BALL_SCORE_THRESHOLD = 0.35
MIN_PLAYER_BOX_HEIGHT_PX = 14.0
MIN_PLAYER_BOX_WIDTH_PX = 6.0

# Court-space tolerance for keeping a detection as an on-court participant.
ON_COURT_LENGTH_MARGIN_M = 3.5
ON_COURT_WIDTH_MARGIN_M = 1.5
# Distance from an end line that still counts as the serving zone.
SERVE_ZONE_MARGIN_M = 1.2

# Trajectory break detection. A touch is a velocity jump that free flight
# cannot explain, so the expected gravity change is subtracted first.
VELOCITY_WINDOW = 5
MIN_VELOCITY_SAMPLES = 2
MAX_SEGMENT_GAP_FRAMES = 4
MIN_SPEED_PX_PER_FRAME = 1.0
MIN_VELOCITY_RESIDUAL_PX_FRAME = 6.0
RELATIVE_VELOCITY_RESIDUAL = 0.30
# A player reaching for the ball moves towards it; this confirms a touch that
# the trajectory alone leaves ambiguous.
APPROACH_WINDOW_FRAMES = 5
APPROACH_MIN_PX_PER_FRAME = 1.0
GRAVITY_CM_PER_S2 = 981.0
DEFAULT_GRAVITY_PX_PER_FRAME2 = 1.0
CONTACT_MERGE_SECONDS = 0.30

# Reach model around a player box: an ellipse covering arms up, dive and
# sideways reach. Candidates are ranked by the normalized distance inside it,
# so the physically closest player wins instead of the one with the biggest box.
REACH_TOP_RATIO = 0.45
REACH_BOTTOM_RATIO = 0.15
REACH_SIDE_HEIGHT_RATIO = 0.42
REACH_SIDE_WIDTH_RATIO = 0.70
MAX_REACH_RATIO = 1.0
REACH_RATIO_COST_WEIGHT = 0.25
START_ATTACH_SECONDS = 0.5
# A serve is struck right after the trajectory starts at the serving player.
SERVE_WINDOW_SECONDS = 1.0
SERVE_MIN_SPEED_GAIN = 1.2
# The rally starts when the ball leaves the hands it was handled in: it must
# then stay out of everybody's reach for a while, and the handling before it
# must have lasted long enough to be a preparation rather than a normal start.
RALLY_START_FLIGHT_SECONDS = 0.3
RALLY_START_MIN_PREPARATION_SECONDS = 0.5
# Being reachable is not the same as being handled: a ball in flight passes over
# players all the time, so the preparation phase uses a much tighter zone.
RALLY_START_HOLD_REACH_RATIO = 0.6

# Perspective consistency between ball and player. A volleyball diameter is
# about 21 cm (its circumference is about 65 cm), while an adult player is
# normally 170-185 cm tall. Since the detector stores ball radius, a matching
# projected radius is roughly 5.7-6.2% of the standing player-box height.
# Bboxes and ball masks are noisy, so the accepted band is expanded before an
# additive ranking penalty is applied.
BALL_DIAMETER_CM = 21.0
# Use one average adult height for perspective/depth scaling.  The ball
# radius is measured by the detector on each frame, so its pixel size carries
# the per-frame distance information.
PLAYER_HEIGHT_CM = 185.0
PLAYER_HEIGHT_MIN_CM = PLAYER_HEIGHT_CM
PLAYER_HEIGHT_MAX_CM = PLAYER_HEIGHT_CM
SIZE_RATIO_TOLERANCE = 0.25
# Perspective must be strong enough to beat a large foreground box when a
# smaller, farther player overlaps it (for example player 73 at frames
# 5545-5550 in beach-mixt).
SIZE_MISMATCH_COST_WEIGHT = 1.25

# A crouching player has a short box, so the raw box top is a bad reference:
# a chest-high dig would look like a ball at head level. Depth is therefore
# measured against the player's standing height, tracked over a moving window.
STANDING_HEIGHT_WINDOW = 61
STANDING_HEIGHT_QUANTILE = 0.8

# Technique boundaries as ball height inside the standing body
# (0.0 = top of the head, 1.0 = feet).
OVERHEAD_TOP_DEPTH = -0.05
OVERHEAD_DEPTH_LIMIT = 0.25
DIG_DEPTH_LIMIT = 0.95

# Posture tells the technique apart as well: a box wider than this relative to
# its height means a lunge or a bend, so the ball is played from below unless
# it is above the head. An upright player sits around 0.35-0.5.
WIDE_POSTURE_RATIO = 0.65
WIDE_POSTURE_OVERHEAD_LIMIT = 0.0

# A team plays at most three touches before the ball must cross the net. When
# two players on opposite sides are almost equally plausible for a touch, that
# rule decides which of them it was.
MAX_TOUCHES_PER_SIDE = 3
AMBIGUITY_COST_FACTOR = 1.6
MAX_CONTACT_ALTERNATIVES = 4

# Rally context for ambiguous projected boxes. A small temporal vote is much
# more stable than classifying one ball center: low-confidence detections and
# radii that jump away from the local median contribute less to the vote.
BALL_SIDE_WINDOW_SECONDS = 0.25
BALL_SIDE_MIN_PROBABILITY = 0.60
BALL_SIDE_MIN_EVIDENCE_WEIGHT = 1.0
BALL_SIDE_NET_DEAD_ZONE_M = 0.35
RECEIVER_CONTEXT_TOUCHES = 2

CONTACT_ATTACK = "attack"
CONTACT_BLOCK = "block"
CONTACT_SERVE = "serve"
CONTACT_OVERHEAD = "overhead_pass"
CONTACT_DIG = "dig"
CONTACT_LOW = "low_touch"


class CourtLike(Protocol):
    """Subset of the court context needed to interpret contacts."""

    @property
    def camera_position(self) -> str: ...

    @property
    def court_length_m(self) -> float: ...

    @property
    def court_width_m(self) -> float: ...

    def to_court(self, x: float, y: float) -> Tuple[float, float]: ...

    def net_y_at_x(self, x: float) -> float: ...


@dataclass(frozen=True)
class PlayerBox:
    frame: int
    track_id: int
    score: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def foot_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def foot_y(self) -> float:
        return self.y2


@dataclass(frozen=True)
class BallSample:
    frame: int
    x: float
    y: float
    radius_px: Optional[float] = None
    confidence: float = 1.0


@dataclass(frozen=True)
class Contact:
    frame: int
    player_track_id: int
    contact_type: str
    depth_ratio: float
    posture_ratio: float
    reach_ratio: float
    approach_px_frame: float
    direction_change_deg: float
    velocity_residual_px_frame: float
    speed_before_px_frame: float
    speed_after_px_frame: float
    confidence: float
    side: str
    player_court_x: Optional[float]
    player_court_y: Optional[float]
    in_serve_zone: bool
    ball_x: float
    ball_y: float
    ball_above_net: bool
    gap_frames: int
    ball_player_radius_ratio: Optional[float] = None
    inferred_player_height_cm: Optional[float] = None
    size_mismatch: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": int(self.frame),
            "player_track_id": int(self.player_track_id),
            "contact_type": self.contact_type,
            "depth_ratio": round(float(self.depth_ratio), 4),
            "posture_ratio": round(float(self.posture_ratio), 4),
            "reach_ratio": round(float(self.reach_ratio), 4),
            "approach_px_frame": round(float(self.approach_px_frame), 3),
            "direction_change_deg": round(float(self.direction_change_deg), 2),
            "velocity_residual_px_frame": round(float(self.velocity_residual_px_frame), 3),
            "speed_before_px_frame": round(float(self.speed_before_px_frame), 3),
            "speed_after_px_frame": round(float(self.speed_after_px_frame), 3),
            "confidence": round(float(self.confidence), 4),
            "side": self.side,
            "player_court_x": (
                round(float(self.player_court_x), 3) if self.player_court_x is not None else None
            ),
            "player_court_y": (
                round(float(self.player_court_y), 3) if self.player_court_y is not None else None
            ),
            "in_serve_zone": bool(self.in_serve_zone),
            "ball_x": round(float(self.ball_x), 2),
            "ball_y": round(float(self.ball_y), 2),
            "ball_above_net": bool(self.ball_above_net),
            "gap_frames": int(self.gap_frames),
            "ball_player_radius_ratio": (
                round(float(self.ball_player_radius_ratio), 5)
                if self.ball_player_radius_ratio is not None
                else None
            ),
            "inferred_player_height_cm": (
                round(float(self.inferred_player_height_cm), 1)
                if self.inferred_player_height_cm is not None
                else None
            ),
            "size_mismatch": round(float(self.size_mismatch), 4),
        }


@dataclass(frozen=True)
class BallEdge:
    """Where a ball trajectory begins or ends, and next to whom."""

    frame: int
    origin: str  # "serve_zone" | "in_court" | "off_court" | "unattached"
    player_track_id: Optional[int]
    side: str
    player_court_x: Optional[float]
    player_court_y: Optional[float]
    in_serve_zone: bool
    reach_ratio: Optional[float]
    depth_ratio: Optional[float]
    posture_ratio: Optional[float]
    ball_x: float
    ball_y: float

    @property
    def at_serve_zone(self) -> bool:
        return self.origin == "serve_zone"

    @property
    def at_player_in_court(self) -> bool:
        return self.origin == "in_court"

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": int(self.frame),
            "origin": self.origin,
            "player_track_id": (
                int(self.player_track_id) if self.player_track_id is not None else None
            ),
            "side": self.side,
            "player_court_x": (
                round(float(self.player_court_x), 3) if self.player_court_x is not None else None
            ),
            "player_court_y": (
                round(float(self.player_court_y), 3) if self.player_court_y is not None else None
            ),
            "in_serve_zone": bool(self.in_serve_zone),
            "reach_ratio": (
                round(float(self.reach_ratio), 4)
                if self.reach_ratio is not None
                else None
            ),
            "depth_ratio": (
                round(float(self.depth_ratio), 4) if self.depth_ratio is not None else None
            ),
            "posture_ratio": (
                round(float(self.posture_ratio), 4) if self.posture_ratio is not None else None
            ),
            "ball_x": round(float(self.ball_x), 2),
            "ball_y": round(float(self.ball_y), 2),
        }


@dataclass(frozen=True)
class ContactChoice:
    """The touch as attributed, plus the players it could also belong to."""

    best: Contact
    alternatives: tuple[Contact, ...]


@dataclass(frozen=True)
class TrackContactAnalysis:
    contacts: tuple[Contact, ...]
    start: Optional[BallEdge]
    end: Optional[BallEdge]
    rally_start_frame: Optional[int]
    rally_state: dict[str, Any] = field(default_factory=dict)
    player_proximity: dict[str, Any] = field(default_factory=dict)


class RallyContactStateMachine:
    """Resolves ambiguous touches using serve, possession and stable ball side.

    Image-space player boxes from opposite halves often overlap. After a serve,
    the first two touches are therefore expected on the receiving half unless
    there is no reachable candidate there. Ball-side evidence is accumulated
    over a short window, weighted by detector confidence and radius stability,
    so one position/radius jump cannot flip the state by itself.
    """

    def __init__(
        self,
        court: CourtLike,
        fps: float,
        samples: Sequence[BallSample],
        start: Optional[BallEdge],
    ) -> None:
        self._court = court
        self._fps = fps if fps and fps > 0 else 30.0
        self._samples = tuple(sorted(samples, key=lambda item: item.frame))
        self._start = start
        self.phase = "pre_serve"
        self.serving_side: Optional[str] = None
        if start is not None and start.at_serve_zone and start.side != "unknown":
            self.serving_side = start.side
        self.possession_side: Optional[str] = None
        self.possession_touches = 0
        self.reception_touches = 0
        self.ball_side: Optional[str] = None
        self.ball_side_probability = 0.0
        self.transitions: list[dict[str, Any]] = []

    def resolve(
        self,
        choices: Sequence[ContactChoice],
        typed: Optional[Sequence[Contact]] = None,
    ) -> list[Contact]:
        resolved: list[Contact] = []
        for index, choice in enumerate(choices):
            contact_type = (
                typed[index].contact_type
                if typed is not None and index < len(typed)
                else choice.best.contact_type
            )
            options = tuple(
                replace(option, contact_type=contact_type)
                for option in (choice.best, *choice.alternatives)
            )
            phase_before = self.phase
            self.ball_side, self.ball_side_probability = self._estimate_ball_side(
                choice.best.frame
            )
            selected, reason = self._select(options, contact_type)
            self._advance(selected)
            resolved.append(selected)
            self.transitions.append(
                {
                    "frame": int(selected.frame),
                    "phase_before": phase_before,
                    "phase_after": self.phase,
                    "contact_type": selected.contact_type,
                    "selected_side": selected.side,
                    "selection_reason": reason,
                    "ball_side": self.ball_side or "unknown",
                    "ball_side_probability": round(float(self.ball_side_probability), 4),
                    "serving_side": self.serving_side or "unknown",
                    "possession_side": self.possession_side or "unknown",
                    "possession_touches": int(self.possession_touches),
                    "reception_touches": int(self.reception_touches),
                }
            )
        return resolved

    def snapshot(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "serving_side": self.serving_side or "unknown",
            "possession_side": self.possession_side or "unknown",
            "possession_touches": int(self.possession_touches),
            "reception_touches": int(self.reception_touches),
            "ball_side": self.ball_side or "unknown",
            "ball_side_probability": round(float(self.ball_side_probability), 4),
            "transitions": self.transitions,
        }

    def _select(
        self,
        options: Sequence[Contact],
        contact_type: str,
    ) -> tuple[Contact, str]:
        best = options[0]
        if contact_type == CONTACT_SERVE:
            if self._start is not None and self._start.player_track_id is not None:
                serving_player = next(
                    (
                        option
                        for option in options
                        if option.player_track_id == self._start.player_track_id
                    ),
                    None,
                )
                if serving_player is not None:
                    return serving_player, "serve_player"
            serving = self._option_on_side(options, self.serving_side)
            return (serving, "serve_side") if serving is not None else (best, "geometry")

        receiving_side = self._opposite(self.serving_side)
        if (
            self.phase in ("serve_flight", "reception")
            and self.reception_touches < RECEIVER_CONTEXT_TOUCHES
        ):
            receiver = self._option_on_side(options, receiving_side)
            if receiver is not None:
                return receiver, "post_serve_receiver"

        if self.ball_side_probability >= BALL_SIDE_MIN_PROBABILITY:
            on_ball_side = self._option_on_side(options, self.ball_side)
            if on_ball_side is not None:
                return on_ball_side, "stable_ball_side"

        if self.possession_side is not None:
            expected = (
                self._opposite(self.possession_side)
                if self.possession_touches >= MAX_TOUCHES_PER_SIDE
                else self.possession_side
            )
            possession = self._option_on_side(options, expected)
            if possession is not None:
                return possession, "possession"
        return best, "geometry"

    def _advance(self, contact: Contact) -> None:
        if contact.contact_type == CONTACT_SERVE:
            if contact.side != "unknown":
                self.serving_side = contact.side
            self.phase = "serve_flight"
            self.possession_side = self.serving_side
            self.possession_touches = 1
            self.reception_touches = 0
            return

        receiving_side = self._opposite(self.serving_side)
        if self.phase in ("serve_flight", "reception") and contact.side == receiving_side:
            self.reception_touches += 1
            self.phase = (
                "reception"
                if self.reception_touches < RECEIVER_CONTEXT_TOUCHES
                else "possession"
            )
        elif self.phase == "pre_serve":
            self.phase = "possession"

        if contact.side == "unknown":
            return
        if contact.side == self.possession_side:
            self.possession_touches += 1
        else:
            self.possession_side = contact.side
            self.possession_touches = 1

    def _estimate_ball_side(self, frame: int) -> tuple[Optional[str], float]:
        window = max(2, int(round(self._fps * BALL_SIDE_WINDOW_SECONDS)))
        local = [sample for sample in self._samples if abs(sample.frame - frame) <= window]
        radii = [
            float(sample.radius_px)
            for sample in local
            if sample.radius_px and sample.radius_px > 0
        ]
        median_radius = float(np.median(radii)) if radii else None
        votes: dict[str, float] = defaultdict(float)
        for sample in local:
            try:
                court_x, _ = self._court.to_court(sample.x, sample.y)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(court_x) or abs(court_x) <= BALL_SIDE_NET_DEAD_ZONE_M:
                continue
            side = self._side_from_court_x(court_x)
            raw_confidence = float(sample.confidence)
            confidence = (
                float(np.clip(raw_confidence, 0.0, 1.0))
                if math.isfinite(raw_confidence)
                else 0.0
            )
            radius_weight = 1.0
            if median_radius and sample.radius_px and sample.radius_px > 0:
                radius_weight = 1.0 / (
                    1.0 + abs(math.log2(float(sample.radius_px) / median_radius))
                )
            distance_weight = min(1.0, abs(court_x) / 1.5)
            votes[side] += max(0.05, confidence) * radius_weight * distance_weight

        total = sum(votes.values())
        if total < BALL_SIDE_MIN_EVIDENCE_WEIGHT:
            return None, 0.0
        side, weight = max(votes.items(), key=lambda item: item[1])
        return side, float(weight / total)

    def _side_from_court_x(self, court_x: float) -> str:
        if self._court.camera_position == "sideline":
            return "left" if court_x < 0 else "right"
        return "near" if court_x < 0 else "far"

    @staticmethod
    def _option_on_side(
        options: Sequence[Contact], side: Optional[str]
    ) -> Optional[Contact]:
        if side is None:
            return None
        return next((option for option in options if option.side == side), None)

    @staticmethod
    def _opposite(side: Optional[str]) -> Optional[str]:
        return {
            "near": "far",
            "far": "near",
            "left": "right",
            "right": "left",
        }.get(side)


class PlayerStore:
    """Frame-indexed player detections filtered to on-court participants."""

    def __init__(
        self,
        players_json_path: Optional[str],
        court: Optional[CourtLike] = None,
        score_threshold: float = DEFAULT_PLAYER_SCORE_THRESHOLD,
        ball_score_threshold: float = DEFAULT_BALL_SCORE_THRESHOLD,
        source_width: Optional[int] = None,
        source_height: Optional[int] = None,
    ) -> None:
        self._court = court
        self._score_threshold = score_threshold
        self._ball_score_threshold = ball_score_threshold
        self._by_frame: Dict[int, Tuple[PlayerBox, ...]] = {}
        self._by_key: Dict[Tuple[int, int], PlayerBox] = {}
        self._ball_by_frame: Dict[int, BallSample] = {}
        self._court_position: Dict[Tuple[int, int], Tuple[float, float]] = {}
        self._stable_side: Dict[int, str] = {}
        self._standing_height: Dict[Tuple[int, int], float] = {}
        self._source_size: Optional[Tuple[int, int]] = None
        self._fps: Optional[float] = None
        self._path = players_json_path
        self._loaded = False
        self._dropped_off_court = 0
        self._kept = 0

        if players_json_path:
            self._load(players_json_path, source_width, source_height)

    @property
    def enabled(self) -> bool:
        return self._loaded and bool(self._by_frame)

    @property
    def fps(self) -> Optional[float]:
        return self._fps

    @property
    def has_ball_detections(self) -> bool:
        return bool(self._ball_by_frame)

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "players_json_path": self._path,
            "loaded": self._loaded,
            "frames_with_players": len(self._by_frame),
            "frames_with_ball": len(self._ball_by_frame),
            "boxes_kept": self._kept,
            "boxes_dropped_off_court": self._dropped_off_court,
            "score_threshold": self._score_threshold,
            "ball_score_threshold": self._ball_score_threshold,
            "source_size": list(self._source_size) if self._source_size else None,
            "source_fps": self._fps,
        }

    def ball_at(self, frame: int) -> Optional[BallSample]:
        return self._ball_by_frame.get(int(frame))

    def ball_samples(self) -> Tuple[BallSample, ...]:
        """Ball detections ordered by frame."""
        return tuple(self._ball_by_frame[frame] for frame in sorted(self._ball_by_frame))

    def frame_range(self) -> Optional[Tuple[int, int]]:
        """First and last frame with any detection, players or ball."""
        frames = [*self._by_frame, *self._ball_by_frame]
        if not frames:
            return None
        return min(frames), max(frames)

    def boxes_at(self, frame: int) -> Tuple[PlayerBox, ...]:
        return self._by_frame.get(int(frame), ())

    def nearest_frame_boxes(self, frame: int, max_gap: int = 2) -> Tuple[PlayerBox, ...]:
        """Boxes at `frame`, falling back to the closest indexed frame."""
        boxes = self.boxes_at(frame)
        if boxes:
            return boxes
        for offset in range(1, max_gap + 1):
            for candidate in (frame - offset, frame + offset):
                boxes = self.boxes_at(candidate)
                if boxes:
                    return boxes
        return ()

    def boxes_near_frame(self, frame: int, max_gap: int = 2) -> Tuple[PlayerBox, ...]:
        """Boxes at a frame plus missing player tracks from adjacent frames.

        A player's confidence often drops exactly when the ball overlaps their
        body. Keep the exact-frame box for every visible track, but recover a
        track that disappeared under the score threshold from a nearby frame.
        """
        result = list(self.boxes_at(frame))
        seen_track_ids = {box.track_id for box in result}
        for offset in range(1, max_gap + 1):
            for candidate in (frame - offset, frame + offset):
                for box in self.boxes_at(candidate):
                    if box.track_id in seen_track_ids:
                        continue
                    result.append(box)
                    seen_track_ids.add(box.track_id)
        return tuple(result)

    def box_at(self, frame: int, track_id: int) -> Optional[PlayerBox]:
        return self._by_key.get((int(frame), int(track_id)))

    def court_position(self, box: PlayerBox) -> Tuple[Optional[float], Optional[float]]:
        return self._court_position.get((box.frame, box.track_id), (None, None))

    def standing_height(self, box: PlayerBox) -> float:
        """Height of the player when upright, robust to crouching and diving."""
        return self._standing_height.get((box.frame, box.track_id), box.height)

    def stable_side(self, box: PlayerBox) -> str:
        """Return a track-stable court side, robust to jumps at the net.

        A jumping player's box bottom moves upward in the image.  A single
        homography projection can therefore land on the opposite side of the
        net for a frame even though the player has not crossed it.  The
        median side of the player's complete track rejects that transient.
        """
        side = self._stable_side.get(int(box.track_id))
        if side is not None:
            return side
        court_x, court_y = self.court_position(box)
        if self._court is None:
            return "unknown"
        if self._court.camera_position == "sideline":
            if court_x is not None:
                return "left" if court_x < 0 else "right"
            return "unknown"
        if court_x is not None:
            return "near" if court_x < 0 else "far"
        return "near" if box.foot_y > self._court.net_y_at_x(box.foot_x) else "far"

    @staticmethod
    def _compute_standing_heights(
        by_track: Dict[int, List[PlayerBox]],
    ) -> Dict[Tuple[int, int], float]:
        result: Dict[Tuple[int, int], float] = {}
        half = STANDING_HEIGHT_WINDOW // 2
        for track_id, boxes in by_track.items():
            ordered = sorted(boxes, key=lambda item: item.frame)
            heights = np.asarray([box.height for box in ordered], dtype=np.float64)
            for index, box in enumerate(ordered):
                window = heights[max(0, index - half) : index + half + 1]
                result[(box.frame, track_id)] = float(
                    np.quantile(window, STANDING_HEIGHT_QUANTILE)
                )
        return result

    def _load(
        self,
        path: str,
        source_width: Optional[int],
        source_height: Optional[int],
    ) -> None:
        if not os.path.exists(path):
            LOG.warning("Players JSON not found: %s", path)
            return

        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            LOG.warning("Cannot read players JSON %s: %s", path, exc)
            return

        predictions, meta = self._extract_predictions(payload)
        if not predictions:
            LOG.warning("Players JSON %s has no predictions", path)
            return

        self._fps = self._as_float(meta.get("fps"))
        scale_x, scale_y = self._resolve_scale(meta, source_width, source_height)

        grouped: Dict[int, List[PlayerBox]] = defaultdict(list)
        by_track: Dict[int, List[PlayerBox]] = defaultdict(list)
        ball_scores: Dict[int, float] = {}
        for item in predictions:
            parsed_ball = self._parse_ball(item, scale_x, scale_y)
            if parsed_ball is not None:
                ball, score = parsed_ball
                if score >= ball_scores.get(ball.frame, -1.0):
                    self._ball_by_frame[ball.frame] = ball
                    ball_scores[ball.frame] = score
                continue

            box = self._parse_box(item, scale_x, scale_y)
            if box is None:
                continue
            court_x, court_y = self._to_court_position(box)
            if not self._is_on_court(court_x, court_y):
                self._dropped_off_court += 1
                continue
            grouped[box.frame].append(box)
            by_track[box.track_id].append(box)
            self._by_key[(box.frame, box.track_id)] = box
            self._court_position[(box.frame, box.track_id)] = (court_x, court_y)
            self._kept += 1

        self._by_frame = {
            frame: tuple(sorted(boxes, key=lambda item: -item.score))
            for frame, boxes in grouped.items()
        }
        self._standing_height = self._compute_standing_heights(by_track)
        self._stable_side = self._compute_stable_sides(by_track)
        self._loaded = True
        LOG.info(
            "Loaded %s player boxes over %s frames from %s (dropped %s off-court)",
            self._kept,
            len(self._by_frame),
            os.path.basename(path),
            self._dropped_off_court,
        )

    def _compute_stable_sides(self, by_track: Dict[int, List[PlayerBox]]) -> Dict[int, str]:
        if self._court is None:
            return {}
        result: Dict[int, str] = {}
        for track_id, boxes in by_track.items():
            coordinates = [
                self._court_position.get((box.frame, track_id), (None, None))[0]
                for box in boxes
            ]
            values = [float(value) for value in coordinates if value is not None and math.isfinite(value)]
            if not values:
                continue
            median = float(np.median(values))
            if self._court.camera_position == "sideline":
                result[track_id] = "left" if median < 0.0 else "right"
            else:
                result[track_id] = "near" if median < 0.0 else "far"
        return result

    @staticmethod
    def _extract_predictions(payload: Any) -> tuple[Sequence[dict], dict]:
        if isinstance(payload, dict):
            for key in ("predictions", "detections", "results"):
                value = payload.get(key)
                if isinstance(value, list):
                    return value, payload
            return (), payload
        if isinstance(payload, list):
            return payload, {}
        return (), {}

    def _resolve_scale(
        self,
        meta: dict,
        source_width: Optional[int],
        source_height: Optional[int],
    ) -> Tuple[float, float]:
        size = meta.get("source_size") or meta.get("frame_size") or {}
        width = self._as_float(size.get("width")) if isinstance(size, dict) else None
        height = self._as_float(size.get("height")) if isinstance(size, dict) else None
        if width and height:
            self._source_size = (int(width), int(height))
        if not width or not height or not source_width or not source_height:
            return 1.0, 1.0
        if int(width) == int(source_width) and int(height) == int(source_height):
            return 1.0, 1.0
        return float(source_width) / float(width), float(source_height) / float(height)

    def _parse_ball(
        self,
        item: Any,
        scale_x: float,
        scale_y: float,
    ) -> Optional[tuple[BallSample, float]]:
        if not isinstance(item, dict):
            return None

        class_id = item.get("class_id")
        class_name = str(item.get("class_name", "")).lower()
        if class_id is not None:
            if int(class_id) not in BALL_CLASS_IDS:
                return None
        elif class_name not in BALL_CLASS_NAMES:
            return None

        score = self._as_float(item.get("score", item.get("confidence", 1.0))) or 0.0
        if score < self._ball_score_threshold:
            return None

        bbox = item.get("bbox_xyxy") or item.get("bbox") or item.get("box")
        frame = item.get("frame_index", item.get("frame"))
        if frame is None or not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return None

        x1 = float(bbox[0]) * scale_x
        y1 = float(bbox[1]) * scale_y
        x2 = float(bbox[2]) * scale_x
        y2 = float(bbox[3]) * scale_y
        radius = (abs(x2 - x1) + abs(y2 - y1)) / 4.0
        if radius <= 0.0:
            return None

        sample = BallSample(
            frame=int(frame),
            x=(x1 + x2) / 2.0,
            y=(y1 + y2) / 2.0,
            radius_px=radius,
            confidence=float(score),
        )
        return sample, float(score)

    def _parse_box(self, item: Any, scale_x: float, scale_y: float) -> Optional[PlayerBox]:
        if not isinstance(item, dict):
            return None

        class_id = item.get("class_id")
        class_name = str(item.get("class_name", "")).lower()
        if class_id is not None:
            if int(class_id) not in PLAYER_CLASS_IDS:
                return None
        elif class_name and class_name not in PLAYER_CLASS_NAMES:
            return None

        score = self._as_float(item.get("score", item.get("confidence", 1.0))) or 0.0
        if score < self._score_threshold:
            return None

        bbox = item.get("bbox_xyxy") or item.get("bbox") or item.get("box")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return None

        frame = item.get("frame_index", item.get("frame"))
        if frame is None:
            return None

        x1 = float(bbox[0]) * scale_x
        y1 = float(bbox[1]) * scale_y
        x2 = float(bbox[2]) * scale_x
        y2 = float(bbox[3]) * scale_y
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        if (y2 - y1) < MIN_PLAYER_BOX_HEIGHT_PX or (x2 - x1) < MIN_PLAYER_BOX_WIDTH_PX:
            return None

        track_id = item.get("track_id")
        return PlayerBox(
            frame=int(frame),
            track_id=int(track_id) if track_id is not None else -1,
            score=float(score),
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
        )

    def _to_court_position(self, box: PlayerBox) -> Tuple[Optional[float], Optional[float]]:
        if self._court is None:
            return None, None
        court_x, court_y = self._court.to_court(box.foot_x, box.foot_y)
        return float(court_x), float(court_y)

    def _is_on_court(self, court_x: Optional[float], court_y: Optional[float]) -> bool:
        if self._court is None or court_x is None or court_y is None:
            return True
        if not (math.isfinite(court_x) and math.isfinite(court_y)):
            return False
        length_limit = self._court.court_length_m / 2.0 + ON_COURT_LENGTH_MARGIN_M
        width_limit = self._court.court_width_m / 2.0 + ON_COURT_WIDTH_MARGIN_M
        return abs(court_x) <= length_limit and abs(court_y) <= width_limit

    @staticmethod
    def _as_float(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None


class ContactDetector:
    """Finds ball-player contacts along a ball trajectory."""

    def __init__(
        self,
        store: PlayerStore,
        court: CourtLike,
        fps: float,
        frame_step: int = 1,
    ) -> None:
        self._store = store
        self._court = court
        self._fps = fps if fps and fps > 0 else 30.0
        # Ball and players sampled every `frame_step` frames: the windows below
        # are tuned in frames of a fully sampled video.
        self._frame_step = max(1, int(frame_step))
        self._velocity_window = max(
            MIN_VELOCITY_SAMPLES + 1, math.ceil(VELOCITY_WINDOW / self._frame_step)
        )
        self._max_segment_gap = MAX_SEGMENT_GAP_FRAMES * self._frame_step
        self._gravity_px_per_frame2 = self._resolve_gravity()

    @property
    def enabled(self) -> bool:
        return self._store.enabled

    @property
    def gravity_px_per_frame2(self) -> float:
        return self._gravity_px_per_frame2

    def analyze(self, samples: Sequence[BallSample]) -> TrackContactAnalysis:
        if not self.enabled or len(samples) < 3:
            return TrackContactAnalysis(contacts=(), start=None, end=None, rally_start_frame=None)

        ordered = sorted(samples, key=lambda item: item.frame)
        start = self.detect_start(ordered)
        choices = self._detect_choices(ordered)

        # Serves are marked before the sides are resolved: a serve tells the
        # possession pass that the ball has to change sides next.
        typed = self._refine_serves([choice.best for choice in choices], start)
        rally_start = self.find_rally_start(ordered, typed)
        typed = self._mark_serve_at_rally_start(typed, rally_start)
        state_machine = RallyContactStateMachine(self._court, self._fps, ordered, start)
        contacts = state_machine.resolve(choices, typed)

        return TrackContactAnalysis(
            contacts=tuple(contacts),
            start=start,
            end=self.detect_end(ordered),
            rally_start_frame=rally_start,
            rally_state=state_machine.snapshot(),
            player_proximity=self.player_proximity(ordered),
        )

    def find_rally_start(
        self,
        samples: Sequence[BallSample],
        contacts: Sequence[Contact],
    ) -> Optional[int]:
        """Frame where the ball leaves the hands it was being handled in.

        Before a serve the ball is passed over, held and tossed - all of it
        within somebody's reach. The rally starts when the ball leaves that
        reach for a flight, which is the serve strike.
        """
        if not self.enabled or not samples:
            return None

        ordered = sorted(samples, key=lambda item: item.frame)
        flight_frames = max(2, int(round(self._fps * RALLY_START_FLIGHT_SECONDS)))
        last_in_reach: Optional[int] = None
        away_since: Optional[int] = None
        for sample in ordered:
            radius = float(sample.radius_px or 0.0)
            held = self.player_within_reach(
                sample.frame, sample.x, sample.y, radius, RALLY_START_HOLD_REACH_RATIO
            )
            if held is not None:
                last_in_reach = sample.frame
                away_since = None
                continue
            if away_since is None:
                away_since = sample.frame
            elif sample.frame - away_since >= flight_frames:
                break

        if last_in_reach is None:
            return None
        if last_in_reach - ordered[0].frame < self._fps * RALLY_START_MIN_PREPARATION_SECONDS:
            return None

        snapped = min(
            (contact for contact in contacts if abs(contact.frame - last_in_reach) <= flight_frames),
            key=lambda contact: abs(contact.frame - last_in_reach),
            default=None,
        )
        return int(snapped.frame) if snapped is not None else int(last_in_reach)

    def _mark_serve_at_rally_start(
        self,
        contacts: Sequence[Contact],
        rally_start: Optional[int],
    ) -> list[Contact]:
        """The strike right after the preparation phase is the serve.

        The rally starts when the ball goes up off the toss; the strike follows
        it, so the serve is the hardest hit inside that opening second.
        """
        if rally_start is None or any(item.contact_type == CONTACT_SERVE for item in contacts):
            return list(contacts)

        window = self._fps * SERVE_WINDOW_SECONDS
        struck: Optional[tuple[float, int]] = None
        for index, contact in enumerate(contacts):
            if not rally_start <= contact.frame <= rally_start + window:
                continue
            gain = contact.speed_after_px_frame / max(contact.speed_before_px_frame, 0.1)
            if gain >= SERVE_MIN_SPEED_GAIN and (struck is None or gain > struck[0]):
                struck = (gain, index)

        if struck is None:
            return list(contacts)
        return [
            replace(contact, contact_type=CONTACT_SERVE) if index == struck[1] else contact
            for index, contact in enumerate(contacts)
        ]

    def _refine_serves(
        self,
        contacts: Sequence[Contact],
        start: Optional[BallEdge],
    ) -> list[Contact]:
        """Marks the serve by context rather than by how high the ball was hit.

        A serve is struck by the very player the trajectory departs from, in the
        serving zone, right after the ball leaves their hands - it can be a jump
        serve above the head or a float served off a toss at head height, so the
        ball height says nothing. Standing in the serving zone is not enough
        either: a player receiving deep in their own court stands there too.
        """
        if not contacts:
            return []

        served = self._find_serve(contacts, start)
        return [
            replace(contact, contact_type=CONTACT_SERVE)
            if index == served
            else contact
            for index, contact in enumerate(contacts)
        ]

    def _find_serve(
        self,
        contacts: Sequence[Contact],
        start: Optional[BallEdge],
    ) -> Optional[int]:
        if start is None or not start.at_serve_zone:
            return None

        opening_window = self._fps * SERVE_WINDOW_SECONDS
        best: Optional[tuple[float, int]] = None
        for index, contact in enumerate(contacts):
            if contact.frame - start.frame > opening_window:
                break
            if not contact.in_serve_zone:
                continue
            if start.player_track_id is not None and contact.player_track_id != start.player_track_id:
                continue
            # The serve is the strike, not the toss that precedes it, and it
            # sends the ball away faster than it arrived. A ball that is already
            # flying fast belongs to a rally, not to a serve.
            gain = contact.speed_after_px_frame / max(contact.speed_before_px_frame, 0.1)
            if gain < SERVE_MIN_SPEED_GAIN:
                continue
            if best is None or gain > best[0]:
                best = (gain, index)
        return best[1] if best else None

    def detect(self, samples: Sequence[BallSample]) -> list[Contact]:
        choices = self._detect_choices(samples)
        return self._resolve_possession(choices, samples=samples)

    def _detect_choices(self, samples: Sequence[BallSample]) -> list[ContactChoice]:
        if not self.enabled or len(samples) < 3:
            return []

        ordered = sorted(samples, key=lambda item: item.frame)
        breaks = self._find_trajectory_breaks(ordered)
        merge_window = max(2, int(round(self._fps * CONTACT_MERGE_SECONDS)))

        choices: list[ContactChoice] = []
        for candidate in breaks:
            choice = self._attach_player(ordered, candidate)
            if choice is None:
                continue
            if choices and choice.best.frame - choices[-1].best.frame <= merge_window:
                if choice.best.confidence > choices[-1].best.confidence:
                    choices[-1] = choice
                continue
            choices.append(choice)
        return choices

    def _resolve_possession(
        self,
        choices: Sequence[ContactChoice],
        typed: Optional[Sequence[Contact]] = None,
        samples: Sequence[BallSample] = (),
        start: Optional[BallEdge] = None,
    ) -> list[Contact]:
        """Resolve ambiguous boxes through the rally context state machine."""
        return RallyContactStateMachine(self._court, self._fps, samples, start).resolve(
            choices, typed
        )

    def detect_start(self, samples: Sequence[BallSample]) -> Optional[BallEdge]:
        """Attaches the first ball samples to the player the ball departs from."""
        return self._detect_edge(samples, from_end=False)

    def detect_end(self, samples: Sequence[BallSample]) -> Optional[BallEdge]:
        """Attaches the last ball samples to the player the ball ends up with."""
        return self._detect_edge(samples, from_end=True)

    def _detect_edge(
        self,
        samples: Sequence[BallSample],
        from_end: bool,
    ) -> Optional[BallEdge]:
        if not self.enabled or not samples:
            return None

        ordered = sorted(samples, key=lambda item: item.frame, reverse=from_end)
        first = ordered[0]
        window = max(1, int(round(self._fps * START_ATTACH_SECONDS)))
        horizon = first.frame - window if from_end else first.frame + window

        # The ball departs from (or arrives at) the player it is next to on the
        # very first sample; later samples are already in flight and can pass
        # anyone, so the earliest attachment wins over a closer but later one.
        best: Optional[tuple[float, float, BallSample, PlayerBox, float]] = None
        for sample in ordered:
            if (sample.frame < horizon) if from_end else (sample.frame > horizon):
                break
            radius = float(sample.radius_px or 0.0)
            for box in self._store.nearest_frame_boxes(sample.frame):
                reach_ratio, cost, depth = self._reach_metrics(box, sample.x, sample.y, radius)
                if reach_ratio > MAX_REACH_RATIO:
                    continue
                if best is None or cost < best[0]:
                    best = (cost, reach_ratio, sample, box, depth)
            if best is not None:
                break

        if best is None:
            return BallEdge(
                frame=int(first.frame),
                origin="unattached",
                player_track_id=None,
                side="unknown",
                player_court_x=None,
                player_court_y=None,
                in_serve_zone=False,
                reach_ratio=None,
                depth_ratio=None,
                posture_ratio=None,
                ball_x=float(first.x),
                ball_y=float(first.y),
            )

        _, reach_ratio, sample, box, depth = best
        court_x, court_y = self._store.court_position(box)
        in_serve_zone = self._in_serve_zone(court_x)
        if in_serve_zone:
            origin = "serve_zone"
        elif court_x is None:
            origin = "unattached"
        elif abs(court_x) > self._court.court_length_m / 2.0:
            origin = "off_court"
        else:
            origin = "in_court"

        return BallEdge(
            frame=int(sample.frame),
            origin=origin,
            player_track_id=int(box.track_id),
            side=self._side_of(box, court_x),
            player_court_x=court_x,
            player_court_y=court_y,
            in_serve_zone=in_serve_zone,
            reach_ratio=float(reach_ratio),
            depth_ratio=float(depth),
            posture_ratio=self._posture_ratio(box),
            ball_x=float(sample.x),
            ball_y=float(sample.y),
        )

    def player_within_reach(
        self,
        frame: int,
        x: float,
        y: float,
        radius: float = 0.0,
        max_reach_ratio: float = MAX_REACH_RATIO,
    ) -> Optional[PlayerBox]:
        """Closest player that could have touched the ball at this point."""
        best: Optional[tuple[float, PlayerBox]] = None
        for box in self._store.nearest_frame_boxes(int(frame)):
            reach_ratio, cost, _ = self._reach_metrics(box, x, y, radius)
            if reach_ratio > max_reach_ratio:
                continue
            if best is None or cost < best[0]:
                best = (cost, box)
        return best[1] if best else None

    def player_proximity(self, samples: Sequence[BallSample]) -> dict[str, Any]:
        """How often the trajectory approaches any detected main-court player."""
        nearest_ratios: list[float] = []
        for sample in samples:
            boxes = self._store.nearest_frame_boxes(int(sample.frame))
            if not boxes:
                continue
            radius = float(sample.radius_px or 0.0)
            ratios = [
                self._reach_metrics(box, sample.x, sample.y, radius)[0]
                for box in boxes
            ]
            nearest_ratios.append(min(ratios))

        sample_count = len(samples)
        frames_with_boxes = len(nearest_ratios)
        return {
            "player_box_coverage": (
                float(frames_with_boxes / sample_count) if sample_count else 0.0
            ),
            "near_player_frame_ratio": (
                float(sum(ratio <= 1.5 for ratio in nearest_ratios) / frames_with_boxes)
                if frames_with_boxes
                else 0.0
            ),
            "within_reach_frame_ratio": (
                float(sum(ratio <= MAX_REACH_RATIO for ratio in nearest_ratios) / frames_with_boxes)
                if frames_with_boxes
                else 0.0
            ),
            "min_player_reach_ratio": (
                float(min(nearest_ratios)) if nearest_ratios else None
            ),
            "median_player_reach_ratio": (
                float(np.median(nearest_ratios)) if nearest_ratios else None
            ),
        }

    def _find_trajectory_breaks(self, samples: Sequence[BallSample]) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for idx in range(len(samples)):
            before = self._segment_velocity(samples, idx, -1)
            after = self._segment_velocity(samples, idx, +1)
            if before is None or after is None:
                continue

            (vx_before, vy_before), frame_before = before
            (vx_after, vy_after), frame_after = after
            speed_before = float(math.hypot(vx_before, vy_before))
            speed_after = float(math.hypot(vx_after, vy_after))
            if max(speed_before, speed_after) < MIN_SPEED_PX_PER_FRAME:
                continue

            # Free flight only adds gravity between the two fit centroids.
            expected_dvy = self._gravity_px_per_frame2 * max(frame_after - frame_before, 0.0)
            residual = float(
                math.hypot(vx_after - vx_before, vy_after - vy_before - expected_dvy)
            )
            threshold = max(
                MIN_VELOCITY_RESIDUAL_PX_FRAME,
                RELATIVE_VELOCITY_RESIDUAL * max(speed_before, speed_after),
            )
            if residual < threshold:
                continue

            candidates.append(
                {
                    "index": idx,
                    "frame": samples[idx].frame,
                    "angle": self._angle_between((vx_before, vy_before), (vx_after, vy_after)),
                    "residual": residual,
                    "residual_ratio": residual / max(threshold, 1e-6),
                    "speed_before": speed_before,
                    "speed_after": speed_after,
                    "gap_frames": self._local_gap(samples, idx),
                    "strength": residual,
                }
            )

        return self._suppress_neighbors(candidates)

    def _suppress_neighbors(self, candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keeps the strongest break in every window, non-maximum suppression.

        Chaining candidate to candidate instead would let one long noisy stretch
        grow into a single group and swallow the real touches inside it.
        """
        window = max(2, int(round(self._fps * CONTACT_MERGE_SECONDS)))
        chosen: list[dict[str, Any]] = []
        for candidate in sorted(candidates, key=lambda item: -item["strength"]):
            if all(abs(candidate["frame"] - other["frame"]) > window for other in chosen):
                chosen.append(candidate)
        return sorted(chosen, key=lambda item: item["frame"])

    def _resolve_gravity(self) -> float:
        cm_per_px = getattr(self._court, "cm_per_px_scale", None)
        if not cm_per_px or cm_per_px <= 0:
            return DEFAULT_GRAVITY_PX_PER_FRAME2
        return float(GRAVITY_CM_PER_S2 / cm_per_px / (self._fps * self._fps))

    def _segment_velocity(
        self,
        samples: Sequence[BallSample],
        idx: int,
        direction: int,
    ) -> Optional[Tuple[Tuple[float, float], float]]:
        frames: list[float] = []
        xs: list[float] = []
        ys: list[float] = []
        previous_frame = samples[idx].frame
        step = 0
        cursor = idx
        while step < self._velocity_window:
            if cursor < 0 or cursor >= len(samples):
                break
            sample = samples[cursor]
            if abs(sample.frame - previous_frame) > self._max_segment_gap:
                break
            frames.append(float(sample.frame))
            xs.append(float(sample.x))
            ys.append(float(sample.y))
            previous_frame = sample.frame
            cursor += direction
            step += 1

        if len(frames) < MIN_VELOCITY_SAMPLES:
            return None
        span = max(frames) - min(frames)
        if span < 1.0:
            return None

        frame_array = np.asarray(frames, dtype=np.float64)
        vx = float(np.polyfit(frame_array, np.asarray(xs, dtype=np.float64), 1)[0])
        vy = float(np.polyfit(frame_array, np.asarray(ys, dtype=np.float64), 1)[0])
        return (vx, vy), float(frame_array.mean())

    @staticmethod
    def _angle_between(first: Tuple[float, float], second: Tuple[float, float]) -> float:
        norm_first = math.hypot(*first)
        norm_second = math.hypot(*second)
        if norm_first < 1e-6 or norm_second < 1e-6:
            return 0.0
        cosine = (first[0] * second[0] + first[1] * second[1]) / (norm_first * norm_second)
        return float(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))

    @staticmethod
    def _local_gap(samples: Sequence[BallSample], idx: int) -> int:
        previous_gap = samples[idx].frame - samples[idx - 1].frame if idx > 0 else 0
        next_gap = samples[idx + 1].frame - samples[idx].frame if idx + 1 < len(samples) else 0
        return int(max(previous_gap, next_gap, 0))

    def _attach_player(
        self,
        samples: Sequence[BallSample],
        candidate: dict[str, Any],
    ) -> Optional[ContactChoice]:
        """Builds the contact for the closest player, keeping close runners-up.

        Near and far players project onto the same image region, so a touch can
        be genuinely ambiguous; the runners-up let the possession pass pick the
        side that fits the rally.
        """
        sample = samples[candidate["index"]]
        boxes = self._store.boxes_near_frame(sample.frame)
        if not boxes:
            return None

        radius = float(sample.radius_px or 0.0)
        scored: list[tuple[float, float, PlayerBox, float]] = []
        for box in boxes:
            reach_ratio, cost, depth = self._reach_metrics(box, sample.x, sample.y, radius)
            if reach_ratio > MAX_REACH_RATIO:
                continue
            scored.append((cost, reach_ratio, box, depth))

        if not scored:
            return None

        scored.sort(key=lambda item: item[0])
        best_cost = scored[0][0]
        options = [
            self._build_contact(sample, candidate, reach_ratio, box, depth)
            for cost, reach_ratio, box, depth in scored
            if cost <= max(best_cost, 1e-6) * AMBIGUITY_COST_FACTOR
        ][:MAX_CONTACT_ALTERNATIVES]
        return ContactChoice(best=options[0], alternatives=tuple(options[1:]))

    def _build_contact(
        self,
        sample: BallSample,
        candidate: dict[str, Any],
        reach_ratio: float,
        box: PlayerBox,
        depth: float,
    ) -> Contact:
        court_x, court_y = self._store.court_position(box)
        posture = self._posture_ratio(box)
        approach = self._approach_speed(box, sample)
        contact_type = self._classify(depth, posture, court_x)
        confidence = self._confidence(candidate, reach_ratio, depth, approach)
        standing_height = max(self._store.standing_height(box), box.height, 1.0)
        radius = float(sample.radius_px or 0.0)
        size_ratio = radius / standing_height if radius > 0.0 else None
        inferred_height = self._inferred_player_height_cm(radius, standing_height)
        size_mismatch = self._size_mismatch(radius, standing_height)
        return Contact(
            frame=int(sample.frame),
            player_track_id=int(box.track_id),
            contact_type=contact_type,
            depth_ratio=float(depth),
            posture_ratio=float(posture),
            reach_ratio=float(reach_ratio),
            approach_px_frame=float(approach),
            direction_change_deg=float(candidate["angle"]),
            velocity_residual_px_frame=float(candidate["residual"]),
            speed_before_px_frame=float(candidate["speed_before"]),
            speed_after_px_frame=float(candidate["speed_after"]),
            confidence=float(confidence),
            side=self._side_of(box, court_x),
            player_court_x=court_x,
            player_court_y=court_y,
            in_serve_zone=self._in_serve_zone(court_x),
            ball_x=float(sample.x),
            ball_y=float(sample.y),
            ball_above_net=self._ball_above_net(sample, radius),
            gap_frames=int(candidate["gap_frames"]),
            ball_player_radius_ratio=size_ratio,
            inferred_player_height_cm=inferred_height,
            size_mismatch=size_mismatch,
        )

    def _reach_metrics(
        self,
        box: PlayerBox,
        ball_x: float,
        ball_y: float,
        radius: float,
    ) -> tuple[float, float, float]:
        """Returns (reach ratio, ranking cost, ball height inside the body)."""
        height = max(self._store.standing_height(box), box.height, 1.0)
        width = max(box.width, 1.0)
        head_y = box.y2 - height
        reach_x = max(width * REACH_SIDE_WIDTH_RATIO, height * REACH_SIDE_HEIGHT_RATIO) + radius
        reach_top = head_y - (height * REACH_TOP_RATIO + radius)
        reach_bottom = box.y2 + (height * REACH_BOTTOM_RATIO + radius)
        reach_center_y = (reach_top + reach_bottom) / 2.0
        reach_y = max((reach_bottom - reach_top) / 2.0, 1.0)

        reach_ratio = math.hypot(
            (ball_x - box.center_x) / reach_x,
            (ball_y - reach_center_y) / reach_y,
        )
        # Rank mainly by distance to the body itself: a player whose torso the
        # ball flies over sits far from it, while the true toucher does not.
        body_distance = math.hypot(
            max(0.0, abs(ball_x - box.center_x) - width / 2.0),
            max(0.0, min(box.y1, head_y) - ball_y, ball_y - box.y2),
        ) / height
        size_mismatch = self._size_mismatch(radius, height)
        cost = (
            body_distance
            + REACH_RATIO_COST_WEIGHT * reach_ratio
            + SIZE_MISMATCH_COST_WEIGHT * size_mismatch
        )
        # A box may be shortened by a crouch or a lunge.  For depth, derive
        # the player's pixel scale from the measured ball radius and the
        # physical dimensions (21 cm diameter, 185 cm average player height).
        # This keeps the depth estimate tied to the current frame's ball size.
        depth_height = self._depth_height_px(radius, height)
        depth_head_y = box.y2 - depth_height
        depth = (ball_y - depth_head_y) / depth_height
        return reach_ratio, cost, depth

    @staticmethod
    def _depth_height_px(radius: float, fallback_height: float) -> float:
        if radius > 0.0:
            ball_radius_cm = BALL_DIAMETER_CM / 2.0
            return max(1.0, radius * PLAYER_HEIGHT_CM / ball_radius_cm)
        return max(fallback_height, 1.0)

    @staticmethod
    def _posture_ratio(box: PlayerBox) -> float:
        """Width over height of the box: how far the player is from upright."""
        return float(box.width / max(box.height, 1.0))

    @staticmethod
    def _size_mismatch(radius: float, height: float) -> float:
        """Log-distance outside the plausible ball/player perspective ratio."""
        if radius <= 0.0:
            return 0.0
        ratio = radius / max(height, 1.0)
        ball_radius_cm = BALL_DIAMETER_CM / 2.0
        physical_min = ball_radius_cm / PLAYER_HEIGHT_MAX_CM
        physical_max = ball_radius_cm / PLAYER_HEIGHT_MIN_CM
        accepted_min = physical_min * (1.0 - SIZE_RATIO_TOLERANCE)
        accepted_max = physical_max * (1.0 + SIZE_RATIO_TOLERANCE)
        if accepted_min <= ratio <= accepted_max:
            return 0.0
        boundary = accepted_min if ratio < accepted_min else accepted_max
        return float(abs(math.log2(ratio / boundary)))

    @staticmethod
    def _inferred_player_height_cm(radius: float, height: float) -> Optional[float]:
        """Player height implied by a 21 cm ball diameter at this scale."""
        if radius <= 0.0:
            return None
        return float((BALL_DIAMETER_CM / 2.0) * height / radius)

    def _classify(self, depth: float, posture: float, court_x: Optional[float]) -> str:
        """Technique from ball height and posture; serves are marked later."""
        if depth < OVERHEAD_TOP_DEPTH:
            if court_x is not None and abs(court_x) <= 1.6:
                return CONTACT_BLOCK
            return CONTACT_ATTACK
        return describe_technique(depth, posture)

    def _approach_speed(self, box: PlayerBox, sample: BallSample) -> float:
        """How fast the player closes on the ball, in px per frame.

        Measured against the ball position at the contact, so only the player's
        own movement counts.
        """
        earlier = None
        window = APPROACH_WINDOW_FRAMES
        # A sparse store has no box exactly APPROACH_WINDOW_FRAMES back.
        for window in range(APPROACH_WINDOW_FRAMES, APPROACH_WINDOW_FRAMES + self._frame_step):
            earlier = self._store.box_at(box.frame - window, box.track_id)
            if earlier is not None:
                break
        if earlier is None:
            return 0.0
        before = math.hypot(sample.x - earlier.center_x, sample.y - earlier.foot_y)
        now = math.hypot(sample.x - box.center_x, sample.y - box.foot_y)
        return float((before - now) / window)

    def _confidence(
        self,
        candidate: dict[str, Any],
        reach_ratio: float,
        depth: float,
        approach: float = 0.0,
    ) -> float:
        score = 0.25
        score += min(float(candidate["residual_ratio"]) / 3.0, 1.0) * 0.35
        score += max(0.0, MAX_REACH_RATIO - reach_ratio) / MAX_REACH_RATIO * 0.25
        if 0.0 <= depth <= 1.0:
            score += 0.10
        if candidate["gap_frames"] >= 3:
            score += 0.05
        if approach >= APPROACH_MIN_PX_PER_FRAME:
            score += 0.10
        return float(min(0.99, score))

    def _in_serve_zone(self, court_x: Optional[float]) -> bool:
        if court_x is None:
            return False
        return abs(court_x) >= self._court.court_length_m / 2.0 - SERVE_ZONE_MARGIN_M

    def _side_of(self, box: PlayerBox, court_x: Optional[float]) -> str:
        stable = self._store.stable_side(box)
        if stable != "unknown":
            return stable
        if self._court.camera_position == "sideline":
            if court_x is None:
                return "unknown"
            return "left" if court_x < 0 else "right"
        if court_x is not None:
            return "near" if court_x < 0 else "far"
        return "near" if box.foot_y > self._court.net_y_at_x(box.foot_x) else "far"

    def _ball_above_net(self, sample: BallSample, radius: float) -> bool:
        net_y = self._court.net_y_at_x(sample.x)
        return (sample.y - radius) < net_y


EMPTY_CONTACT_SUMMARY: dict[str, Any] = {
    "contact_count": 0,
    "contact_players_count": 0,
    "side_switch_count": 0,
    "dig_count": 0,
    "overhead_count": 0,
    "attack_count": 0,
    "block_count": 0,
    "serve_count": 0,
    "low_touch_count": 0,
    "first_contact_frame": None,
    "first_contact_side": "unknown",
    "first_contact_type": None,
    "first_contact_in_serve_zone": False,
    "first_contact_court_x": None,
    "last_contact_frame": None,
    "last_contact_side": "unknown",
    "has_serve_contact": False,
    "serve_side": "unknown",
    "serve_frame": None,
    "mean_contact_confidence": 0.0,
    "max_same_side_run": 0,
    "contacts_per_second": 0.0,
    "start_origin": "unknown",
    "start_side": "unknown",
    "start_player_track_id": None,
    "start_court_x": None,
    "starts_from_serve_zone": False,
    "starts_inside_court": False,
    "start_technique": None,
    "end_origin": "unknown",
    "end_side": "unknown",
    "end_player_track_id": None,
    "end_court_x": None,
    "ends_at_serve_zone": False,
    "ends_in_court": False,
    "end_technique": None,
    "rally_start_frame": None,
    "preparation_sec": 0.0,
    "possessions": [],
    "max_touches_in_possession": 0,
    "three_touch_possessions": 0,
    "over_three_touch_possessions": 0,
    "touches_by_player": {},
}


def summarize_contacts(
    contacts: Sequence[Contact],
    fps: float,
    start: Optional[BallEdge] = None,
    end: Optional[BallEdge] = None,
    rally_start_frame: Optional[int] = None,
) -> dict[str, Any]:
    """Aggregates contacts and both trajectory ends into rally-level evidence."""
    summary = dict(EMPTY_CONTACT_SUMMARY)
    summary.update(_summarize_start(start))
    summary.update(_summarize_end(end))
    if rally_start_frame is not None:
        summary["rally_start_frame"] = int(rally_start_frame)
        if start is not None and fps > 0:
            summary["preparation_sec"] = round(max(0.0, (rally_start_frame - start.frame) / fps), 3)
    if not contacts:
        return summary

    ordered = sorted(contacts, key=lambda item: item.frame)
    playable = {"near", "far", "left", "right"}
    sides = [contact.side for contact in ordered]
    switches = sum(
        1
        for previous, current in zip(sides, sides[1:])
        if previous in playable and current in playable and previous != current
    )

    max_run = 1
    run = 1
    for previous, current in zip(sides, sides[1:]):
        run = run + 1 if previous == current else 1
        max_run = max(max_run, run)

    first = ordered[0]
    serve = next((contact for contact in ordered if contact.contact_type == CONTACT_SERVE), None)
    span_seconds = max(1, ordered[-1].frame - ordered[0].frame) / fps if fps > 0 else 0.0

    # Where the ball departs from beats any later contact: a player receiving
    # a serve deep in their own court also stands in the serving zone.
    if start is not None and start.at_serve_zone:
        serve_side = start.side
        serve_frame: Optional[int] = int(start.frame)
    elif serve is not None:
        serve_side = serve.side
        serve_frame = int(serve.frame)
    else:
        serve_side = "unknown"
        serve_frame = None

    # Touches made while the ball was still being handed around before the
    # serve are preparation, not rally touches.
    play_from = serve.frame if serve is not None else rally_start_frame
    summary.update(
        _summarize_possessions(
            [item for item in ordered if play_from is None or item.frame >= play_from]
        )
    )
    summary.update(
        {
            "contact_count": len(ordered),
            "contact_players_count": len({contact.player_track_id for contact in ordered}),
            "side_switch_count": int(switches),
            "dig_count": sum(1 for item in ordered if item.contact_type == CONTACT_DIG),
            "overhead_count": sum(1 for item in ordered if item.contact_type == CONTACT_OVERHEAD),
            "attack_count": sum(1 for item in ordered if item.contact_type == CONTACT_ATTACK),
            "block_count": sum(1 for item in ordered if item.contact_type == CONTACT_BLOCK),
            "serve_count": sum(1 for item in ordered if item.contact_type == CONTACT_SERVE),
            "low_touch_count": sum(1 for item in ordered if item.contact_type == CONTACT_LOW),
            "first_contact_frame": int(first.frame),
            "first_contact_side": first.side,
            "first_contact_type": first.contact_type,
            "first_contact_in_serve_zone": bool(first.in_serve_zone),
            "first_contact_court_x": first.player_court_x,
            "last_contact_frame": int(ordered[-1].frame),
            "last_contact_side": ordered[-1].side,
            "has_serve_contact": serve is not None,
            "serve_side": serve_side,
            "serve_frame": serve_frame,
            "mean_contact_confidence": float(np.mean([item.confidence for item in ordered])),
            "max_same_side_run": int(max_run),
            "contacts_per_second": float(len(ordered) / span_seconds) if span_seconds > 0 else 0.0,
        }
    )
    return summary


def _summarize_start(start: Optional[BallEdge]) -> dict[str, Any]:
    if start is None:
        return {}
    return {
        "start_origin": start.origin,
        "start_side": start.side,
        "start_player_track_id": start.player_track_id,
        "start_court_x": start.player_court_x,
        "starts_from_serve_zone": bool(start.at_serve_zone),
        "starts_inside_court": bool(start.at_player_in_court),
        "start_technique": describe_technique(start.depth_ratio, start.posture_ratio),
    }


def _summarize_possessions(contacts: Sequence[Contact]) -> dict[str, Any]:
    """Splits the touches into possessions - runs of touches by one side."""
    possessions: list[dict[str, Any]] = []
    for contact in contacts:
        if possessions and possessions[-1]["side"] == contact.side:
            current = possessions[-1]
            current["touches"] += 1
            current["end_frame"] = int(contact.frame)
            current["players"].append(int(contact.player_track_id))
            current["types"].append(contact.contact_type)
            continue
        possessions.append(
            {
                "side": contact.side,
                "touches": 1,
                "start_frame": int(contact.frame),
                "end_frame": int(contact.frame),
                "players": [int(contact.player_track_id)],
                "types": [contact.contact_type],
            }
        )

    touches_by_player: Dict[str, int] = defaultdict(int)
    for contact in contacts:
        touches_by_player[str(contact.player_track_id)] += 1

    counts = [item["touches"] for item in possessions]
    return {
        "possessions": possessions,
        "max_touches_in_possession": max(counts) if counts else 0,
        "three_touch_possessions": sum(1 for value in counts if value == MAX_TOUCHES_PER_SIDE),
        "over_three_touch_possessions": sum(1 for value in counts if value > MAX_TOUCHES_PER_SIDE),
        "touches_by_player": dict(touches_by_player),
    }


def _summarize_end(end: Optional[BallEdge]) -> dict[str, Any]:
    if end is None:
        return {}
    return {
        "end_origin": end.origin,
        "end_side": end.side,
        "end_player_track_id": end.player_track_id,
        "end_court_x": end.player_court_x,
        "ends_at_serve_zone": bool(end.at_serve_zone),
        "ends_in_court": bool(end.at_player_in_court),
        "end_technique": describe_technique(end.depth_ratio, end.posture_ratio),
    }


def describe_technique(
    depth_ratio: Optional[float],
    posture_ratio: Optional[float] = None,
) -> Optional[str]:
    """Maps ball height and player posture to a playing technique."""
    if depth_ratio is None:
        return None
    if depth_ratio < OVERHEAD_TOP_DEPTH:
        return CONTACT_ATTACK
    overhead_limit = (
        WIDE_POSTURE_OVERHEAD_LIMIT
        if posture_ratio is not None and posture_ratio >= WIDE_POSTURE_RATIO
        else OVERHEAD_DEPTH_LIMIT
    )
    if depth_ratio < overhead_limit:
        return CONTACT_OVERHEAD
    if depth_ratio <= DIG_DEPTH_LIMIT:
        return CONTACT_DIG
    return CONTACT_LOW
