import argparse
import gzip
import json
import logging
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm
try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover - optional dependency
    Image = None
    ImageDraw = None
    ImageFont = None

try:
    from .constants import DEFAULT_CROP_ASPECT_RATIO, DEFAULT_FPS, DEFAULT_SMOOTH_WINDOW
    from .predictions_io import iter_frames
except ImportError:
    from constants import DEFAULT_CROP_ASPECT_RATIO, DEFAULT_FPS, DEFAULT_SMOOTH_WINDOW
    from predictions_io import iter_frames

LOG = logging.getLogger(__name__)
WATERMARK_TEXT = "vb-ai.ru"
WATERMARK_FONT_PATH = Path(__file__).resolve().parent / "fonts" / "PlayfairDisplay-MediumItalic.ttf"
BLACK_TRANSITION_SECONDS = 0.3
DEFAULT_PADDING_SECONDS = 0.33
# The player of a highlighted action: green corners around the box, blinking 5 frames on / 5 off,
# from a moment before the contact to a moment after it.
PLAYER_MARK_COLOR = (0, 255, 0)
PLAYER_MARK_BLINK_FRAMES = 5
PLAYER_MARK_BEFORE_SECONDS = 1.5
PLAYER_MARK_AFTER_SECONDS = 0.5
# The detector writes every other frame or so: a box is held this many frames over a gap.
PLAYER_MARK_HOLD_FRAMES = 4
# Where the detector gave the actor another track id: the box overlapping the last one this much.
PLAYER_MARK_MIN_OVERLAP = 0.3
Box = Tuple[float, float, float, float]
# The full-frame reel draws the ball arcs of the sets: the last SET_TRAIL_POOL started, the newest
# yellow with a glow, the older ones grey and fainter with age (as the annotate page shows them).
SET_TRAIL_POOL = 10
SET_TRAIL_MAX_GAP_FRAMES = 10
SET_TRAIL_NEWEST_COLOR = (21, 204, 250)
SET_TRAIL_OLD_COLOR = (240, 232, 226)
# A set: (its frame, the ball points (frame, x, y) up to the attack).
SetTrail = Tuple[int, List[Tuple[int, float, float]]]


try:
    from scipy.signal import savgol_filter
except Exception:  # pragma: no cover - optional dependency
    savgol_filter = None


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def ensure_reels_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_single_track(track_json_path: str) -> Dict:
    """Load a single track JSON and normalize it for processing."""
    with open(track_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return load_track_from_payload(data)


def _positive_dimension(value: object) -> Optional[int]:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        return None
    return int(round(float(value)))


def tracking_frame_size(track_payload: Dict) -> Optional[Tuple[int, int]]:
    """Resolution in which track coordinates were produced.

    New track files carry both dimensions in ``tracking_scale``. Older files
    may only have width there, so ``court_info`` is used as a compatible
    fallback. A missing height is handled later with uniform width scaling.
    """
    tracking_scale = track_payload.get("tracking_scale") or {}
    court_info = track_payload.get("court_info") or {}
    width = _positive_dimension(tracking_scale.get("frame_width_px"))
    height = _positive_dimension(tracking_scale.get("frame_height_px"))
    width = width or _positive_dimension(court_info.get("image_width"))
    height = height or _positive_dimension(court_info.get("image_height"))
    if width is None:
        return None
    return width, height or 0


def load_track_from_payload(track_payload: Dict) -> Dict:
    positions = []
    for item in track_payload.get("positions", []):
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        xy, frame = item[0], item[1]
        if not isinstance(xy, (list, tuple)) or len(xy) < 2:
            continue
        x, y = xy[0], xy[1]
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            continue
        if not isinstance(frame, (int, float)):
            continue
        positions.append((float(x), float(y), int(frame)))

    start_frame = track_payload.get("start_frame")
    last_frame = track_payload.get("last_frame")
    if not isinstance(start_frame, int) or not isinstance(last_frame, int):
        raise ValueError("track_json must include integer start_frame and last_frame")

    return {
        "start_frame": start_frame,
        "last_frame": last_frame,
        "positions": positions,
        "tracking_frame_size": tracking_frame_size(track_payload),
    }


def scale_track_to_video(track: Dict, video_width: int, video_height: int) -> Dict:
    """Convert proxy-space positions to the opened video's pixel space."""
    source_size = track.get("tracking_frame_size")
    if not source_size:
        return track

    source_width, source_height = source_size
    if source_width <= 0 or video_width <= 0 or video_height <= 0:
        return track

    scale_x = float(video_width) / float(source_width)
    # Old track JSONs only stored width. Their proxy/full videos have the same
    # aspect ratio, so uniform width scaling is the safest compatible fallback.
    scale_y = (
        float(video_height) / float(source_height)
        if source_height and source_height > 0
        else scale_x
    )
    if math.isclose(scale_x, 1.0) and math.isclose(scale_y, 1.0):
        return track

    LOG.info(
        "Scaling track coordinates: %sx%s -> %sx%s (x=%.4f, y=%.4f)",
        source_width,
        source_height or "?",
        video_width,
        video_height,
        scale_x,
        scale_y,
    )
    scaled = dict(track)
    scaled["positions"] = [
        (float(x) * scale_x, float(y) * scale_y, int(frame))
        for x, y, frame in track["positions"]
    ]
    scaled["tracking_frame_size"] = (video_width, video_height)
    return scaled


def interpolate_positions(
    frame_to_pos: Dict[int, Tuple[float, float]],
    start_frame: int,
    end_frame: int,
    mode: str,
) -> np.ndarray:
    frames = np.arange(start_frame, end_frame + 1)
    if not frame_to_pos:
        return np.zeros_like(frames, dtype=float)

    known_frames = np.array(sorted(frame_to_pos.keys()))
    known_x = np.array([frame_to_pos[f][0] for f in known_frames], dtype=float)

    if mode == "linear":
        return np.interp(frames, known_frames, known_x)

    x_values = np.empty_like(frames, dtype=float)
    last_x = known_x[0]
    idx = 0
    for i, frame_idx in enumerate(frames):
        while idx < len(known_frames) and known_frames[idx] <= frame_idx:
            last_x = known_x[idx]
            idx += 1
        x_values[i] = last_x
    return x_values


def smooth_values(values: np.ndarray, method: str, window: int, polyorder: int) -> np.ndarray:
    if method == "none":
        return values
    if method == "moving_avg":
        if len(values) < 2:
            return values
        window = max(3, min(window, len(values)))
        if window % 2 == 0:
            window += 1
        pad = window // 2
        padded = np.pad(values, (pad, pad), mode="edge")
        return np.convolve(padded, np.ones(window) / window, mode="valid")
    if method == "savitzky_golay":
        if savgol_filter is None:
            raise RuntimeError("savitzky_golay smoothing requires scipy")
        window = max(5, min(window, len(values)))
        if window % 2 == 0:
            window += 1
        polyorder = min(polyorder, window - 1)
        return savgol_filter(values, window_length=window, polyorder=polyorder, mode="interp")
    if method == "kalman":
        return kalman_smooth(values)
    raise ValueError(f"Unknown smoothing method: {method}")


def kalman_smooth(values: np.ndarray, process_var: float = 1e-3, meas_var: float = 1e-1) -> np.ndarray:
    if len(values) == 0:
        return values
    x_est = values[0]
    p = 1.0
    smoothed = []
    for z in values:
        p = p + process_var
        k = p / (p + meas_var)
        x_est = x_est + k * (z - x_est)
        p = (1 - k) * p
        smoothed.append(x_est)
    return np.array(smoothed)


def crop_frame(frame: np.ndarray, center_x: int, crop_width: int) -> np.ndarray:
    """Crop around ``center_x``, shifting the window to stay inside the frame."""
    frame_width = frame.shape[1]
    crop_width = min(crop_width, frame_width)
    left = max(0, min(center_x - crop_width // 2, frame_width - crop_width))
    return frame[:, left:left + crop_width]


def clip_frame_range(
    start_frame: int,
    end_frame: int,
    total_video_frames: int,
    fps: float,
    padding: float = DEFAULT_PADDING_SECONDS,
) -> Tuple[int, int]:
    """Add ``padding`` seconds on both sides without leaving the video bounds."""
    padding_frames = max(0, round(padding * fps))
    clip_start = max(0, int(start_frame) - padding_frames)
    clip_end = int(end_frame) + padding_frames
    if total_video_frames > 0:
        clip_end = min(clip_end, total_video_frames - 1)
    return clip_start, clip_end


def add_watermark_top_right(frame: np.ndarray, text: str = WATERMARK_TEXT) -> np.ndarray:
    """Draw a compact watermark in the top-right corner."""
    h, w = frame.shape[:2]
    alpha = 0.2
    scale = max(0.6, min(1.0, w / 1200.0)) * 3.0
    margin = max(10, int(round(scale * 16)))

    if Image is not None and ImageDraw is not None and ImageFont is not None and WATERMARK_FONT_PATH.exists():
        try:
            font_size = max(16, int(round(scale * 24)))
            font = ImageFont.truetype(str(WATERMARK_FONT_PATH), size=font_size)
            rgba = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
            overlay = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
            text_w = right - left
            text_h = bottom - top
            x = max(margin, w - text_w - margin)
            y = margin
            text_alpha = int(round(255 * alpha))

            draw.text((x + 2, y + 2), text, font=font, fill=(0, 0, 0, text_alpha))
            draw.text((x, y), text, font=font, fill=(255, 255, 255, text_alpha))

            composited = Image.alpha_composite(rgba, overlay).convert("RGB")
            return cv2.cvtColor(np.array(composited), cv2.COLOR_RGB2BGR)
        except Exception:
            LOG.exception("Failed to render watermark with TTF font, fallback to OpenCV font")

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(round(scale * 2)))
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    x = max(margin, w - text_w - margin)
    y = margin + text_h

    fallback_overlay = frame.copy()
    cv2.putText(
        fallback_overlay,
        text,
        (x + 2, y + 2),
        font,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(fallback_overlay, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.addWeighted(fallback_overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame


def black_fade_alpha(
    frame_index: int,
    total_frames: int,
    fps: float,
    fade_in: bool,
    fade_out: bool,
    duration_sec: float = BLACK_TRANSITION_SECONDS,
) -> float:
    """Brightness for a fade from/to black at a segment boundary."""
    if total_frames <= 1 or fps <= 0 or duration_sec <= 0:
        return 1.0
    fade_frames = min(
        max(2, int(round(fps * duration_sec))),
        max(2, total_frames // 2),
    )
    alpha = 1.0
    if fade_in and frame_index < fade_frames:
        alpha = min(alpha, frame_index / float(fade_frames - 1))
    fade_out_start = total_frames - fade_frames
    if fade_out and frame_index >= fade_out_start:
        alpha = min(alpha, (total_frames - 1 - frame_index) / float(fade_frames - 1))
    return float(max(0.0, min(1.0, alpha)))


def apply_black_fade(
    frame: np.ndarray,
    frame_index: int,
    total_frames: int,
    fps: float,
    fade_in: bool,
    fade_out: bool,
    duration_sec: float = BLACK_TRANSITION_SECONDS,
) -> np.ndarray:
    alpha = black_fade_alpha(
        frame_index,
        total_frames,
        fps,
        fade_in=fade_in,
        fade_out=fade_out,
        duration_sec=duration_sec,
    )
    if alpha >= 1.0:
        return frame
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=0)


def _predictions_frame_size(predictions_path: str) -> Optional[Tuple[int, int]]:
    """``source_size`` of the predictions header: the frame size of the boxes."""
    opener = gzip.open if str(predictions_path).endswith(".gz") else open
    try:
        with opener(predictions_path, "rt", encoding="utf-8") as handle:
            header = json.loads(handle.readline())
        size = header.get("source_size") or {}
        return int(size["width"]), int(size["height"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _actor_track(players: List[Tuple[Optional[int], Box]], track_id: Optional[int], ball: Optional[list]) -> Optional[int]:
    """Index of the acting player: by the action's track id, else the box nearest to the ball."""
    if track_id is not None:
        for index, (track, _) in enumerate(players):
            if track == track_id:
                return index
    if not ball or not players:
        return None
    bx, by = float(ball[0]), float(ball[1])

    def distance(box: Box) -> float:
        return float(np.hypot(max(box[0] - bx, 0.0, bx - box[2]), max(box[1] - by, 0.0, by - box[3])))

    return min(range(len(players)), key=lambda i: distance(players[i][1]))


def _overlap(a: Box, b: Box) -> float:
    """Intersection over union of two ``x1, y1, x2, y2`` boxes."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def _follow_actor(
    players_by_frame: Dict[int, List[Tuple[Optional[int], Box]]],
    frames: List[int],
    track: Optional[int],
    box: Box,
) -> Dict[int, Box]:
    """The actor's box on ``frames`` (in walking order from the contact): by its track id, and where
    the detector gave the person another id, by the box that overlaps the last one most."""
    boxes: Dict[int, Box] = {}
    for frame in frames:
        players = players_by_frame.get(frame)
        if not players:
            continue
        same = next((b for t, b in players if track is not None and t == track), None)
        if same is None:
            best = max(players, key=lambda p: _overlap(p[1], box))
            if _overlap(best[1], box) < PLAYER_MARK_MIN_OVERLAP:
                continue
            track, same = best
        box = same
        boxes[frame] = box
    return boxes


def player_mark_boxes(
    predictions_path: str,
    highlights: List[Dict],
    fps: float,
    video_size: Tuple[int, int],
) -> Dict[int, Box]:
    """``frame -> box`` (video pixels) of the player to mark, on the frames the blinking mark is on.

    ``highlights``: ``{"frame", "track_id"?, "ball"?: [x, y]}`` of each action. The actor is found at
    the contact frame and followed from there both ways; where the ranges of two actions meet, a
    frame belongs to the nearer contact.
    """
    if not highlights:
        return {}
    before = int(round(PLAYER_MARK_BEFORE_SECONDS * fps))
    after = int(round(PLAYER_MARK_AFTER_SECONDS * fps))
    frames = sorted(int(h["frame"]) for h in highlights)
    players_by_frame: Dict[int, List[Tuple[Optional[int], Box]]] = {}
    first = max(frames[0] - before - PLAYER_MARK_HOLD_FRAMES, 0)
    for frame, items in iter_frames(predictions_path, first, frames[-1] + after + PLAYER_MARK_HOLD_FRAMES):
        players_by_frame[frame] = [
            (item.get("track_id"), tuple(float(v) for v in item["bbox_xyxy"]))
            for item in items
            if item.get("class_name") == "player"
        ]
    if not players_by_frame:
        return {}

    source = _predictions_frame_size(predictions_path)
    scale_x = video_size[0] / source[0] if source else 1.0
    scale_y = video_size[1] / source[1] if source else 1.0
    known = sorted(players_by_frame)

    marks: Dict[int, Tuple[int, Box]] = {}
    for highlight in highlights:
        contact = int(highlight["frame"])
        # The contact frame itself may be one the detector skipped: the nearest written one.
        nearest = min(known, key=lambda f: abs(f - contact))
        if abs(nearest - contact) > PLAYER_MARK_HOLD_FRAMES:
            continue
        players = players_by_frame[nearest]
        index = _actor_track(players, highlight.get("track_id"), highlight.get("ball"))
        if index is None:
            continue
        track, contact_box = players[index]
        start = contact - before
        actor = _follow_actor(players_by_frame, list(range(nearest, contact + after + 1)), track, contact_box)
        actor.update(_follow_actor(players_by_frame, list(range(nearest, start - PLAYER_MARK_HOLD_FRAMES - 1, -1)), track, contact_box))

        held: Optional[Box] = None
        held_at = start - PLAYER_MARK_HOLD_FRAMES - 1
        for frame in range(start - PLAYER_MARK_HOLD_FRAMES, contact + after + 1):
            if frame in actor:
                held, held_at = actor[frame], frame
            if frame < start or held is None or frame - held_at > PLAYER_MARK_HOLD_FRAMES:
                continue
            if (frame - start) // PLAYER_MARK_BLINK_FRAMES % 2:
                continue
            distance = abs(frame - contact)
            if frame in marks and marks[frame][0] <= distance:
                continue
            x1, y1, x2, y2 = held
            marks[frame] = (distance, (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y))
    return {frame: box for frame, (_, box) in marks.items()}


def draw_player_mark(frame: np.ndarray, box: Box, color: Tuple[int, int, int] = PLAYER_MARK_COLOR) -> np.ndarray:
    """Sight-like corners around ``box``: four L-shaped strokes, a quarter of the shorter side long."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    pad = 0.08 * max(x2 - x1, y2 - y1)
    x1, y1 = int(max(0, x1 - pad)), int(max(0, y1 - pad))
    x2, y2 = int(min(width - 1, x2 + pad)), int(min(height - 1, y2 + pad))
    if x2 <= x1 or y2 <= y1:
        return frame
    arm = max(4, int(0.25 * min(x2 - x1, y2 - y1)))
    thickness = max(2, int(round(height / 250)))
    for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(frame, (cx, cy), (cx + dx * arm, cy), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * arm), color, thickness, cv2.LINE_AA)
    return frame


def set_trail_points(
    tracks: List[Dict],
    windows: List[Tuple[int, int]],
    frame_size: Tuple[int, int],
) -> List[SetTrail]:
    """The ball points of each set window, in the pixels of the video.

    The points come from the ball tracks of the reel's pieces; a set outside of every piece is
    left out, so the pool counts only the sets the reel shows.
    """
    positions: Dict[int, Tuple[float, float]] = {}
    for track in tracks:
        for x, y, frame in scale_track_to_video(track, *frame_size)["positions"]:
            if x >= 0 and y >= 0:
                positions[frame] = (x, y)
    frames = sorted(positions)
    trails: List[SetTrail] = []
    for start, end in sorted(windows):
        if not any(track["start_frame"] <= start <= track["last_frame"] for track in tracks):
            continue
        trails.append((start, [(f, *positions[f]) for f in frames if start <= f <= end]))
    return trails


def draw_set_trails(frame: np.ndarray, trails: List[SetTrail], frame_idx: int) -> np.ndarray:
    """The arcs of the last sets up to ``frame_idx``; the arc of a set in flight ends at it."""
    started = [trail for trail in trails if trail[0] <= frame_idx][-SET_TRAIL_POOL:]
    height = frame.shape[0]
    for index, (_, points) in enumerate(started):
        # A gap in the track breaks the line.
        lines: List[List[Tuple[int, int]]] = []
        last_frame: Optional[int] = None
        for f, x, y in points:
            if f > frame_idx:
                break
            if last_frame is None or f - last_frame > SET_TRAIL_MAX_GAP_FRAMES:
                lines.append([])
            lines[-1].append((int(round(x)), int(round(y))))
            last_frame = f
        polylines = [np.array(line, dtype=np.int32) for line in lines if len(line) > 1]
        if not polylines:
            continue
        age = len(started) - 1 - index
        newest = age == 0
        width = max(2, int(round(height / (220 if newest else 360))))
        layers = (
            [(SET_TRAIL_NEWEST_COLOR, width * 3, 0.35), (SET_TRAIL_NEWEST_COLOR, width, 1.0)]
            if newest
            else [(SET_TRAIL_OLD_COLOR, width, max(0.2, 0.7 - age * 0.06))]
        )
        # Blend inside the bounding box of the arc only: a full-frame blend per arc is slow.
        all_points = np.concatenate(polylines)
        pad = width * 3 + 2
        x1, y1 = np.maximum(all_points.min(axis=0) - pad, 0)
        x2, y2 = np.minimum(all_points.max(axis=0) + pad, (frame.shape[1] - 1, height - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        roi = frame[y1:y2 + 1, x1:x2 + 1]
        shifted = [line - (x1, y1) for line in polylines]
        for color, thickness, alpha in layers:
            if alpha >= 1.0:
                cv2.polylines(roi, shifted, False, color, thickness, cv2.LINE_AA)
                continue
            overlay = roi.copy()
            cv2.polylines(overlay, shifted, False, color, thickness, cv2.LINE_AA)
            cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, dst=roi)
    return frame


def crop_and_save_track(
    video_path: str,
    track: Dict,
    output_path: str,
    visualize: bool = False,
    smoothing: str = "moving_avg",
    interpolation: str = "hold",
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
    smooth_polyorder: int = 2,
    margin: float = 0.0,
    padding: float = DEFAULT_PADDING_SECONDS,
    fade_in: bool = False,
    fade_out: bool = False,
    fade_duration: float = BLACK_TRANSITION_SECONDS,
    highlights: Optional[List[Dict]] = None,
    predictions_path: Optional[str] = None,
    full_frame: bool = False,
    set_trails: Optional[List[Tuple[int, int]]] = None,
    trail_tracks: Optional[List[Dict]] = None,
) -> None:
    """``full_frame``: the whole frame instead of the vertical crop that follows the ball.

    ``set_trails`` are the (set frame, attack frame) windows whose ball arcs are drawn, with the
    points from ``trail_tracks`` (the tracks of all pieces of the reel, so the pool of arcs goes
    on from one piece to the next) or from ``track``.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = DEFAULT_FPS
    delay = int(1000 / fps)

    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError("Failed to read first video frame")

    frame_height, frame_width = frame.shape[:2]
    trails = (
        set_trail_points(trail_tracks or [track], set_trails, (frame_width, frame_height)) if set_trails else []
    )
    track = scale_track_to_video(track, frame_width, frame_height)
    crop_width = frame_width if full_frame else min(int(frame_height * DEFAULT_CROP_ASPECT_RATIO), frame_width)
    crop_height = frame_height

    out = None
    if not visualize:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (crop_width, crop_height))

    frame_to_pos = {int(p[2]): (p[0], p[1]) for p in track["positions"]}
    if not frame_to_pos:
        raise ValueError("Track contains no ball positions")

    # Before and after the track the crop holds the edge ball position.
    start_frame, end_frame = clip_frame_range(
        track["start_frame"],
        track["last_frame"],
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        fps,
        padding,
    )

    x_values = interpolate_positions(frame_to_pos, start_frame, end_frame, interpolation)
    x_smooth = smooth_values(x_values, smoothing, smooth_window, smooth_polyorder)

    total_frames = end_frame - start_frame + 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    player_marks = (
        player_mark_boxes(predictions_path, highlights, fps, (frame_width, frame_height))
        if highlights and predictions_path
        else {}
    )

    for i, frame_idx in enumerate(
        tqdm(
            range(start_frame, end_frame + 1),
            desc="Saving frames",
            total=total_frames,
            unit="frame",
        )
    ):
        ret, frame = cap.read()
        if not ret:
            LOG.warning("Frame %s not read - video ended early.", frame_idx)
            break

        if frame_idx in player_marks:
            draw_player_mark(frame, player_marks[frame_idx])
        if trails:
            draw_set_trails(frame, trails, frame_idx)

        center_x = int(x_smooth[i])
        if margin:
            dx = x_smooth[i] - x_smooth[i - 1] if i > 0 else 0.0
            lead = margin * np.sign(dx) if dx != 0 else 0.0
            center_x = int(center_x + lead)

        cropped = crop_frame(frame, center_x, crop_width)

        if out is not None:
            if cropped.shape[1] != crop_width:
                cropped = cv2.resize(cropped, (crop_width, crop_height))
            cropped = add_watermark_top_right(cropped)
        cropped = apply_black_fade(
            cropped,
            i,
            total_frames,
            fps,
            fade_in=fade_in,
            fade_out=fade_out,
            duration_sec=fade_duration,
        )

        if out is not None:
            out.write(cropped)

        if visualize:
            cv2.imshow("Cropped", cropped)
            if cv2.waitKey(delay) & 0xFF == ord("q"):
                break

    cap.release()
    if out is not None:
        out.release()
    if visualize:
        cv2.destroyAllWindows()


def crop_and_save_track_payload(
    video_path: str,
    track_payload: Dict,
    output_path: str,
    visualize: bool = False,
    smoothing: str = "moving_avg",
    interpolation: str = "hold",
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
    smooth_polyorder: int = 2,
    margin: float = 0.0,
    padding: float = DEFAULT_PADDING_SECONDS,
    fade_in: bool = False,
    fade_out: bool = False,
    fade_duration: float = BLACK_TRANSITION_SECONDS,
    predictions_path: Optional[str] = None,
    full_frame: bool = False,
    set_trails: Optional[List[Tuple[int, int]]] = None,
    trail_tracks: Optional[List[Dict]] = None,
) -> None:
    track = load_track_from_payload(track_payload)
    crop_and_save_track(
        video_path=video_path,
        track=track,
        output_path=output_path,
        visualize=visualize,
        smoothing=smoothing,
        interpolation=interpolation,
        smooth_window=smooth_window,
        smooth_polyorder=smooth_polyorder,
        margin=margin,
        padding=padding,
        fade_in=fade_in,
        fade_out=fade_out,
        fade_duration=fade_duration,
        highlights=track_payload.get("highlights"),
        predictions_path=predictions_path,
        full_frame=full_frame,
        set_trails=set_trails,
        trail_tracks=trail_tracks,
    )


def crop_and_save_track_payloads(
    video_path: str,
    track_payloads: List[Dict],
    output_path: str,
    visualize: bool = False,
    smoothing: str = "moving_avg",
    interpolation: str = "hold",
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
    smooth_polyorder: int = 2,
    margin: float = 0.0,
    padding: float = DEFAULT_PADDING_SECONDS,
    fade_transition: bool = False,
    fade_duration: float = BLACK_TRANSITION_SECONDS,
    predictions_path: Optional[str] = None,
    full_frame: bool = False,
    set_trails: Optional[List[Tuple[int, int]]] = None,
) -> None:
    if not track_payloads:
        raise ValueError("track_payloads must not be empty")

    if len(track_payloads) == 1:
        crop_and_save_track_payload(
            video_path=video_path,
            track_payload=track_payloads[0],
            output_path=output_path,
            visualize=visualize,
            smoothing=smoothing,
            interpolation=interpolation,
            smooth_window=smooth_window,
            smooth_polyorder=smooth_polyorder,
            margin=margin,
            padding=padding,
            predictions_path=predictions_path,
            full_frame=full_frame,
            set_trails=set_trails,
        )
        return

    trail_tracks = [load_track_from_payload(payload) for payload in track_payloads] if set_trails else None

    output_target = Path(output_path).resolve()
    output_target.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="reel_segments_") as tmp_dir_raw:
        tmp_dir = Path(tmp_dir_raw)
        segment_files: List[Path] = []
        for idx, payload in enumerate(track_payloads, start=1):
            segment_path = tmp_dir / f"segment_{idx:04d}.mp4"
            crop_and_save_track_payload(
                video_path=video_path,
                track_payload=payload,
                output_path=str(segment_path),
                visualize=visualize,
                smoothing=smoothing,
                interpolation=interpolation,
                smooth_window=smooth_window,
                smooth_polyorder=smooth_polyorder,
                margin=margin,
                padding=padding,
                fade_in=fade_transition and idx > 1,
                fade_out=fade_transition and idx < len(track_payloads),
                fade_duration=fade_duration,
                predictions_path=predictions_path,
                full_frame=full_frame,
                set_trails=set_trails,
                trail_tracks=trail_tracks,
            )
            segment_files.append(segment_path)

        concat_list = tmp_dir / "concat_list.txt"
        concat_list.write_text(
            "\n".join([f"file '{segment_path.name}'" for segment_path in segment_files]),
            encoding="utf-8",
        )

        concat_cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-c",
            "copy",
            str(output_target),
        ]
        try:
            subprocess.run(concat_cmd, cwd=tmp_dir, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"ffmpeg concat failed: {exc.stderr[-2000:] if exc.stderr else str(exc)}"
            ) from exc


class FadeTransitionAction(argparse.Action):
    """Enable the fade transition and optionally set its duration in seconds."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, True)
        if values is not None:
            namespace.fade_duration = values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize rally clips with ball-centered cropping."
    )
    parser.add_argument("--video_path", required=True, help="Path to video file")
    parser.add_argument("--track_json", help="Path to a single track JSON file")
    parser.add_argument("--track_jsons", nargs="+", help="Paths to multiple track JSON files")
    parser.add_argument(
        "--fade-transition",
        "--black-transition",
        dest="fade_transition",
        action=FadeTransitionAction,
        nargs="?",
        type=float,
        default=False,
        metavar="SECONDS",
        help=(
            "Join --track_jsons with a fade to black followed by a fade from "
            "black; optional duration of each fade in seconds (default: 0.3)"
        ),
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help="Output file for the combined reel (only with --track_jsons)",
    )
    parser.set_defaults(fade_duration=BLACK_TRANSITION_SECONDS)
    parser.add_argument("--json_dir", help="Directory with track_*.json files")
    parser.add_argument("--output_dir", default=None, help="Root output directory")
    parser.add_argument("--visualize", action="store_true", help="Show real-time cropped video")
    parser.add_argument(
        "--smoothing",
        choices=["none", "moving_avg", "savitzky_golay", "kalman"],
        default="moving_avg",
        help="Smoothing method for x trajectory",
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=DEFAULT_SMOOTH_WINDOW,
        help="Window size for smoothing",
    )
    parser.add_argument(
        "--smooth_polyorder",
        type=int,
        default=2,
        help="Polynomial order for Savitzky-Golay",
    )
    parser.add_argument(
        "--interpolation",
        choices=["hold", "linear"],
        default="hold",
        help="Interpolation method for missing detections",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="Lead offset in pixels based on movement direction",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=DEFAULT_PADDING_SECONDS,
        help="Seconds of video to add before the start and after the end of each rally",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.padding < 0:
        parser.error("--padding must be greater than or equal to 0")
    if args.fade_duration <= 0:
        parser.error("--fade-transition duration must be greater than 0")
    if args.output_path and not args.track_jsons:
        parser.error("--output_path requires --track_jsons")
    setup_logging(args.verbose)

    base_name = os.path.splitext(os.path.basename(args.video_path))[0]

    if args.track_jsons:
        json_paths = args.track_jsons
    elif args.track_json:
        json_paths = [args.track_json]
    elif args.json_dir:
        json_paths = sorted(
            [
                os.path.join(args.json_dir, f)
                for f in os.listdir(args.json_dir)
                if f.startswith("track_") and f.endswith(".json")
            ]
        )
    else:
        raise ValueError("Specify --track_json, --track_jsons, or --json_dir")

    reels_dir = os.path.join(args.output_dir, base_name, "reels") if args.output_dir else "reels"
    ensure_reels_dir(reels_dir)

    if args.fade_transition and (not args.track_jsons or len(args.track_jsons) < 2):
        parser.error("--fade-transition requires at least two files in --track_jsons")

    if args.track_jsons:
        track_payloads = []
        for track_json_path in args.track_jsons:
            with open(track_json_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict):
                raise ValueError(f"Track JSON must contain an object: {track_json_path}")
            track_payloads.append(payload)

        output_path = args.output_path or os.path.join(
            reels_dir, f"reel_{base_name}_combined.mp4"
        )
        crop_and_save_track_payloads(
            video_path=args.video_path,
            track_payloads=track_payloads,
            output_path=output_path,
            visualize=args.visualize,
            smoothing=args.smoothing,
            interpolation=args.interpolation,
            smooth_window=args.smooth_window,
            smooth_polyorder=args.smooth_polyorder,
            margin=args.margin,
            padding=args.padding,
            fade_transition=args.fade_transition,
            fade_duration=args.fade_duration,
        )
        if not args.visualize:
            LOG.info("Saved combined reel: %s", output_path)
        return

    for track_json_path in json_paths:
        track = load_single_track(track_json_path)

        track_filename = os.path.basename(track_json_path)
        track_basename = os.path.splitext(track_filename)[0]
        try:
            track_number = track_basename.split("_")[-1]
            int(track_number)
        except (ValueError, IndexError):
            track_number = track_basename

        output_path = os.path.join(reels_dir, f"reel_{base_name}_{track_number}.mp4")

        crop_and_save_track(
            args.video_path,
            track,
            output_path,
            visualize=args.visualize,
            smoothing=args.smoothing,
            interpolation=args.interpolation,
            smooth_window=args.smooth_window,
            smooth_polyorder=args.smooth_polyorder,
            margin=args.margin,
            padding=args.padding,
        )

        if not args.visualize:
            LOG.info("Saved: %s", output_path)


if __name__ == "__main__":
    main()
