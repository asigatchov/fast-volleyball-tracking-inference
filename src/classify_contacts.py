#!/usr/bin/env python3
"""Classify detected ball contacts as Serve / Receive / Set / Attack / NoAction.

    uv run src/classify_contacts.py video.mp4 \
        --players_json_path ../uploads/mix/beach-mixt/predictions.json \
        --tracks_dir ../uploads/mix/beach-mixt/tracks

The contacts come from ``player_interaction.contacts`` of every ``track_*.json``
(written by track_calculator_with_court, drawn by show_rally). Around each
contact frame a window of +-N frames is cropped at the player closest to the
ball and given to the clip classifier of ``vb-action/infer_action_clips.py``,
exported to OpenVINO by ``scripts/export_action_classifier.py``. The result is
written back into the contact as ``action``: ``{type, score, probabilities}``.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np
import openvino as ov

LOG = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "ov" / "action_clf_r2plus1d18_pm6_player_beach_hall.xml"

Box = tuple[float, float, float, float]  # cx, cy, w, h (normalized)
# frame -> "player" / "ball" -> boxes; "player_tracks" holds the track id of every player box
Boxes = dict[int, dict[str, list[Any]]]
# Size of the box built for a ball known only by its centre (a contact's ball_x / ball_y).
BALL_POINT_SIZE_PX = 20.0


def load_boxes(path: Path, width: int, height: int) -> Boxes:
    """Player and ball boxes of a detector predictions JSON, normalized to the frame size."""
    data = json.loads(path.read_text(encoding="utf-8"))
    boxes: Boxes = defaultdict(lambda: {"player": [], "ball": [], "player_tracks": []})
    for item in data["predictions"]:
        name = item.get("class_name")
        if name not in ("player", "ball") or "bbox_xyxy" not in item:
            continue
        x1, y1, x2, y2 = item["bbox_xyxy"]
        box = ((x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height)
        boxes[int(item["frame_index"])][name].append(box)
        if name == "player":
            boxes[int(item["frame_index"])]["player_tracks"].append(item.get("track_id"))
    return boxes


def _class_boxes(boxes: Boxes, frame: int, name: str) -> list[Box]:
    return boxes[frame][name] if frame in boxes else []


def _nearest_box(candidates: Sequence[Box], reference: Box) -> Box:
    return min(candidates, key=lambda box: (box[0] - reference[0]) ** 2 + (box[1] - reference[1]) ** 2)


# The crop geometry below repeats vb-action (action_clf/clips.py, infer_action_clips.crop_plan):
# the classifier was trained on these crops, so the window must be built the same way.


def player_ball_boxes(
    boxes: Boxes, center: int, half_window: int, player: Box, width: int, height: int, ball_reach: float
) -> list[Box]:
    """One player followed over the window plus the ball while it is close to him."""
    found = [player]
    for direction in (-1, 1):
        reference = player
        for step in range(1, half_window + 1):
            candidates = _class_boxes(boxes, center + direction * step, "player")
            if candidates:
                reference = _nearest_box(candidates, reference)
                found.append(reference)
    reach = ball_reach * player[3] * height
    for frame in range(center - half_window, center + half_window + 1):
        for ball in _class_boxes(boxes, frame, "ball"):
            if np.hypot((ball[0] - player[0]) * width, (ball[1] - player[1]) * height) < reach:
                found.append(ball)
    return found


def square_crop_window(
    boxes: Sequence[Box], width: int, height: int, scale: float, min_side: int = 64
) -> tuple[int, int, int]:
    """One square pixel window ``(x0, y0, side)`` covering all boxes; it may stick out of the frame."""
    x1 = min((cx - w / 2) * width for cx, _, w, _ in boxes)
    x2 = max((cx + w / 2) * width for cx, _, w, _ in boxes)
    y1 = min((cy - h / 2) * height for _, cy, _, h in boxes)
    y2 = max((cy + h / 2) * height for _, cy, _, h in boxes)
    side = max(int(round(max(x2 - x1, y2 - y1) * scale)), min_side)
    return int(round((x1 + x2) / 2 - side / 2)), int(round((y1 + y2) / 2 - side / 2)), side


def player_crop_window(
    player: Box, width: int, height: int, scale: float, lift: float, view: float, min_side: int = 64
) -> tuple[int, int, int]:
    """Square window sized by the player's height: its middle ``view`` share is ``scale`` heights wide.

    The centre is lifted by ``lift`` heights so raised arms and the ball above the head fit.
    """
    player_height = player[3] * height
    side = max(int(round(scale * player_height / view)), min_side)
    x0 = int(round(player[0] * width - side / 2))
    y0 = int(round(player[1] * height - lift * player_height - side / 2))
    return x0, y0, side


def contact_player(
    boxes: Boxes, contact: dict[str, Any], ball: Optional[Box], width: int, height: int
) -> Optional[Box]:
    """Box of the player who touches the ball at the contact.

    The contact detector names him (``player_track_id``): on annotated clips that is the
    acting player in 99 of 107 contacts. The player whose box centre is closest to the ball
    is right in 78 only: at a serve the ball above the server's head is closer, in the
    picture, to the small players at the far end. Without the track the ball decides, with
    the distance to the box measured in player heights so a far player does not win.
    """
    frame = int(contact["frame"])
    players = _class_boxes(boxes, frame, "player")
    if not players:
        return None
    track_id = contact.get("player_track_id")
    if track_id is not None:
        for box, box_track in zip(players, boxes[frame]["player_tracks"]):
            if box_track == track_id:
                return box
    if ball is None:
        return None

    def reach(box: Box) -> float:
        dx = max(abs(box[0] - ball[0]) * width - box[2] * width / 2, 0.0)
        dy = max(abs(box[1] - ball[1]) * height - box[3] * height / 2, 0.0)
        return float(np.hypot(dx, dy)) / max(box[3] * height, 1e-9)

    # Inside several boxes at once the ball is with the nearest, hence the biggest, player.
    return min(players, key=lambda box: (round(reach(box), 2), -box[3]))


def crop_square(frame: np.ndarray, x0: int, y0: int, side: int, size: int) -> np.ndarray:
    """Crop a square window (zero-padded outside the frame) and resize to ``size``."""
    height, width = frame.shape[:2]
    pad = [max(-y0, 0), max(y0 + side - height, 0), max(-x0, 0), max(x0 + side - width, 0)]
    if any(pad):
        frame = cv2.copyMakeBorder(frame, *pad, cv2.BORDER_CONSTANT, value=0)
        x0, y0 = x0 + pad[2], y0 + pad[0]
    crop = frame[y0 : y0 + side, x0 : x0 + side]
    interpolation = cv2.INTER_AREA if side > size else cv2.INTER_LINEAR
    return cv2.resize(crop, (size, size), interpolation=interpolation)


def crop_plan(
    boxes: Boxes,
    contact: dict[str, Any],
    half_window: int,
    width: int,
    height: int,
    crop_scale: float,
    ball_search: int = 3,
    ball_reach: float = 2.0,
    crop: Optional[dict[str, Any]] = None,
) -> Optional[tuple[int, int, int]]:
    """Crop window around the player closest to the ball at the contact, None without players.

    ``crop`` is the rule the classifier was trained with: ``{"mode": "player", ...}`` sizes the
    window by the player's height; without it the window covers the player and the ball boxes.
    """
    center = int(contact["frame"])
    players = _class_boxes(boxes, center, "player")
    if not players:
        return None
    ball: Optional[Box] = None
    for offset in sorted(range(-ball_search, ball_search + 1), key=abs):
        balls = _class_boxes(boxes, center + offset, "ball")
        if balls:
            ball = balls[0]
            break
    if ball is None:
        # The tracker interpolates the ball through detector gaps, so a contact can have a position without a box.
        ball_x, ball_y = contact.get("ball_x"), contact.get("ball_y")
        if isinstance(ball_x, (int, float)) and isinstance(ball_y, (int, float)):
            ball = (ball_x / width, ball_y / height, BALL_POINT_SIZE_PX / width, BALL_POINT_SIZE_PX / height)
    if crop and crop.get("mode") == "player":
        player = contact_player(boxes, contact, ball, width, height)
        if player is None:
            return None
        return player_crop_window(player, width, height, crop["scale"], crop["lift"], crop["view"])
    if ball is None:
        return None
    player = min(players, key=lambda box: ((box[0] - ball[0]) * width) ** 2 + ((box[1] - ball[1]) * height) ** 2)
    window_boxes = player_ball_boxes(boxes, center, half_window, player, width, height, ball_reach)
    return square_crop_window(window_boxes, width, height, crop_scale)


def read_clips(
    video_path: str, plans: dict[int, tuple[int, int, int]], half_window: int, stored_size: int
) -> dict[int, np.ndarray]:
    """RGB clips ``[T, S, S, 3]`` for every centre frame of ``plans``, in one sequential pass.

    Frames are decoded in order (seeking is not frame-accurate); a window that
    sticks out of the video repeats the edge frame, as in training.
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    last = max(total - 1, 0)
    length = 2 * half_window + 1
    # frame -> (centre, position in the clip) for every clip that needs the frame
    wanted: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for center in plans:
        for position, frame in enumerate(range(center - half_window, center + half_window + 1)):
            wanted[min(max(frame, 0), last)].append((center, position))
    crops: dict[int, list[Optional[np.ndarray]]] = {center: [None] * length for center in plans}

    index = 0
    final = max(wanted, default=-1)
    while index <= final and cap.grab():
        if index in wanted:
            ok, image = cap.retrieve()
            if not ok:
                break
            for center, position in wanted[index]:
                x0, y0, side = plans[center]
                crops[center][position] = crop_square(image, x0, y0, side, stored_size)[:, :, ::-1]
        index += 1
    cap.release()
    return {center: np.stack(clip) for center, clip in crops.items() if all(crop is not None for crop in clip)}


class ActionClassifier:
    """OpenVINO clip classifier: ``[B, T, S, S, 3]`` RGB crops -> ``[B, classes]`` probabilities."""

    def __init__(self, model_xml: Path, device: str = "CPU") -> None:
        meta = json.loads(model_xml.with_suffix(".json").read_text(encoding="utf-8"))
        self.class_names: list[str] = meta["class_names"]
        self.half_window: int = meta["half_window"]
        self.stored_size: int = meta["stored_size"]
        self.crop_scale: float = meta["crop_scale"]
        self.crop: Optional[dict[str, Any]] = meta.get("crop")
        self._model = ov.Core().compile_model(str(model_xml), device)

    def __call__(self, clips: np.ndarray) -> np.ndarray:
        clips = np.ascontiguousarray(clips, dtype=np.float32)
        # Same test-time augmentation as infer_action_clips.classify: the clip and its mirror.
        mirrored = np.ascontiguousarray(clips[:, :, :, ::-1])
        return (self._model(clips)[0] + self._model(mirrored)[0]) / 2


def load_rally_tracks(tracks_dir: str) -> dict[str, dict[str, Any]]:
    """``track_*.json`` that are rallies and have contacts, by path."""
    tracks: dict[str, dict[str, Any]] = {}
    for path in sorted(glob.glob(os.path.join(tracks_dir, "track_*.json"))):
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        classification = data.get("rally_classification")
        if isinstance(classification, dict) and classification.get("is_rally") is False:
            continue
        if _contacts(data):
            tracks[path] = data
    return tracks


def _contacts(track: dict[str, Any]) -> list[dict[str, Any]]:
    interaction = track.get("player_interaction")
    contacts = interaction.get("contacts") if isinstance(interaction, dict) else None
    if not isinstance(contacts, list):
        return []
    return [c for c in contacts if isinstance(c, dict) and isinstance(c.get("frame"), (int, float))]


def classify_tracks(
    video_path: str,
    players_json_path: str,
    tracks_dir: str,
    classifier: ActionClassifier,
    batch_size: int = 8,
) -> dict[str, int]:
    """Write ``action`` into every contact of the rally tracks; returns counts per action type."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    tracks = load_rally_tracks(tracks_dir)
    boxes = load_boxes(Path(players_json_path), width, height)
    plans: dict[int, tuple[int, int, int]] = {}
    for track in tracks.values():
        for contact in _contacts(track):
            plan = crop_plan(
                boxes, contact, classifier.half_window, width, height, classifier.crop_scale, crop=classifier.crop
            )
            if plan is not None:
                plans[int(contact["frame"])] = plan

    clips = read_clips(video_path, plans, classifier.half_window, classifier.stored_size)
    centers = sorted(clips)
    probabilities: dict[int, np.ndarray] = {}
    for start in range(0, len(centers), batch_size):
        batch = centers[start : start + batch_size]
        for center, row in zip(batch, classifier(np.stack([clips[center] for center in batch]))):
            probabilities[center] = row

    counts: dict[str, int] = defaultdict(int)
    for path, track in tracks.items():
        for contact in _contacts(track):
            row = probabilities.get(int(contact["frame"]))
            if row is None:
                contact.pop("action", None)
                counts["unclassified"] += 1
                continue
            best = int(np.argmax(row))
            contact["action"] = {
                "type": classifier.class_names[best],
                "score": round(float(row[best]), 4),
                "probabilities": {name: round(float(p), 4) for name, p in zip(classifier.class_names, row)},
            }
            counts[classifier.class_names[best]] += 1
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(track, handle, indent=2, ensure_ascii=False)
    return dict(counts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify detected ball contacts with the action clip classifier")
    parser.add_argument("video")
    parser.add_argument("--players_json_path", required=True, help="detector predictions JSON (players and ball)")
    parser.add_argument("--tracks_dir", required=True, help="directory with track_*.json")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="OpenVINO .xml of the classifier")
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--batch_size", type=int, default=8)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args()
    classifier = ActionClassifier(Path(args.model), args.device)
    counts = classify_tracks(args.video, args.players_json_path, args.tracks_dir, classifier, args.batch_size)
    LOG.info("Classified contacts: %s", ", ".join(f"{name} {count}" for name, count in sorted(counts.items())) or "none")


if __name__ == "__main__":
    main()
