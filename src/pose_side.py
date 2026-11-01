#!/usr/bin/env python3
"""Who touched the ball and on which half he stands, from the pose (RTMO) around each contact.

    ~/projects/pose/.venv/bin/python src/pose_side.py video.mp4 \
        --players_json_path <project>/predictions.jsonl.gz \
        --tracks_dir <project>/tracks --court_json <project>/court.json

The contact detector names the touching player by the boxes (``player_track_id``) and takes
his side from the median of his whole track. Both go wrong at the net: the ball passes a
blocker's box, and a track that jumps from one player to another across the net gets the
other team's side. Here, for every contact of the rally tracks:

1. RTMO finds the people in one square window around the ball, on the frames
   ``contact - back .. contact + 2``.
2. The toucher is the person whose forearm (elbow -> wrist, extended) passes closest to the
   ball at ``contact +-2``, in torso units, with a penalty when his size does not fit the
   ball's (a far player whose hand overlaps the ball in projection). He replaces the
   detector's player only when clearly better (the gate of the 2026-10-24 evaluation).
3. That person is followed back over the window, frame to frame by box overlap.
4. His side comes from the feet: the ankles of the frame where they are lowest in the image.
   A jump lifts the feet in the image, so at the net their projection falls on the other
   side; ``back`` frames before the hit (0.5 s at 30 fps) the attacker is still on the
   ground. The point goes through the court homography: the net is ``court_x = 0``.

The result goes into the contact: ``side`` and ``player_track_id`` are replaced (the
detector's values are kept as ``detector_side`` / ``detector_player_track_id``, so a rerun
starts from them), ``pose`` holds the evidence.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Sequence

import cv2
import numpy as np

try:
    from .court_transformer import CoordinateTransformer, CourtTransformer
    from .frame_seek import SeekingReader, seek_enabled
    from .models import CourtGeometry
    from .predictions_io import iter_frames
except ImportError:
    from court_transformer import CoordinateTransformer, CourtTransformer
    from frame_seek import SeekingReader, seek_enabled
    from models import CourtGeometry
    from predictions_io import iter_frames

LOG = logging.getLogger(__name__)

DEFAULT_POSE_ROOT = Path.home() / "projects" / "pose"
DEFAULT_CHECKPOINT = "/mnt/data-ml/datashared/vitpose_store/work_dirs/rtmo_t_vaa_ft/best_AP.pth"
DEFAULT_RTMPOSE_CHECKPOINT = "/mnt/data-ml/datashared/vitpose_store/work_dirs/rtmpose_s_vaa_ft/best_AP.pth"

# COCO keypoints
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_ANKLE, R_ANKLE = 11, 12, 15, 16

# Window around the ball: a share of the frame height, lowered so the feet of a jumping
# attacker under a high ball still fit.
WINDOW_SHARE = 0.75
WINDOW_LOWER = 0.1
# Frames around the contact where the hand is looked for.
TOUCH_OFFSETS = (-2, 0, 2)
# Hand score: forearm extended past the wrist, penalties per frame off the contact and for a
# person whose torso / ball radius is off the usual 5.
FOREARM_EXTENSION = 0.35
FRAME_PENALTY = 0.06
DEPTH_RATIO = 5.0
DEPTH_PENALTY = 1.0
# The pose replaces the detector's player when its toucher scores at most this, and the
# detector's player scores at least GAIN worse.
MAX_HAND_SCORE = 0.8
MIN_GAIN = 0.6
KEYPOINT_THR = 0.2
ANKLE_THR = 0.3
MATCH_IOU = 0.3
# Feet closer to the net than this decide nothing.
NET_DEAD_ZONE_M = 0.15

Pose = dict[str, np.ndarray]  # bboxes [N, 4] xyxy, keypoints [N, 17, 2], keypoint_scores [N, 17]


@dataclass
class Person:
    frame: int
    box: np.ndarray  # xyxy, frame pixels
    keypoints: np.ndarray  # [17, 2], frame pixels
    scores: np.ndarray  # [17]


@dataclass
class PoseJob:
    """One frame of one contact: the window around the ball and the detector's player boxes in it."""

    frame: int
    image: np.ndarray
    window: tuple[int, int, int]  # x0, y0, side
    boxes: list[np.ndarray]  # xyxy, frame pixels


# The people of every job, in frame pixels.
Predictor = Callable[[list[PoseJob]], list[list["Person"]]]


def _add_pose_paths(pose_root: Path) -> None:
    sys.path.insert(0, str(pose_root))
    sys.path.insert(0, str(pose_root / "src"))


def build_rtmo_predictor(pose_root: Path, checkpoint: str, size: int, device: str) -> Predictor:
    """RTMO (bottom-up, torch) of the pose project on the window around the ball; its venv runs this script."""
    _add_pose_paths(pose_root)
    from infer_rtmo import build_predictor  # type: ignore[import-not-found]

    predictor = build_predictor(checkpoint, size, device, fp16=False, score_thr=0.1)

    def predict(jobs: list[PoseJob]) -> list[list[Person]]:
        crops = []
        for job in jobs:
            x0, y0, side = job.window
            crop = job.image[y0 : y0 + side, x0 : x0 + side]
            crops.append(cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA) if side > size else crop)
        results = predictor.predict_batch(crops)
        return [
            _people({key: np.asarray(value) for key, value in result.items()}, job.frame, job.window[0], job.window[1], job.window[2] / crop.shape[0])
            for job, crop, result in zip(jobs, crops, results)
        ]

    return predict


def build_rtmpose_predictor(pose_root: Path, checkpoint: str, device: str, flip_test: bool = True) -> Predictor:
    """RTMPose (top-down, torch) on the detector's player boxes in the window."""
    _add_pose_paths(pose_root)
    import torch  # type: ignore[import-not-found]
    from vitpose.config import load_config  # type: ignore[import-not-found]
    from vitpose.rtmpose.inference import RTMPosePredictor  # type: ignore[import-not-found]

    config_json = Path(checkpoint).with_name("config.json")
    cfg = json.loads(config_json.read_text()) if config_json.exists() else load_config("rtmpose/rtmpose_s")
    predictor = RTMPosePredictor(cfg, checkpoint, device=device, dtype=torch.float32, flip_test=flip_test)

    def predict(jobs: list[PoseJob]) -> list[list[Person]]:
        found = []
        for job in jobs:
            result = predictor(job.image, np.asarray(job.boxes).reshape(-1, 4)) if job.boxes else np.zeros((0, 17, 3))
            found.append([
                Person(frame=job.frame, box=np.asarray(box, dtype=np.float64), keypoints=keypoints[:, :2].astype(np.float64), scores=keypoints[:, 2].astype(np.float64))
                for box, keypoints in zip(job.boxes, result)
            ])
        return found

    return predict


class CentredCourt:
    """Court metres measured from the marked centre line.

    The homography comes from the four corners only. With a wide lens (and an approximate
    markup) the marked centre line (keypoints 5 and 6) lands about a metre off ``court_x = 0``
    in it, which decides the half of every player at the net. ``court_x`` is shifted by the
    centre line's own ``court_x`` at the point's ``court_y`` (linear between the two marks).
    """

    def __init__(self, transformer: CoordinateTransformer, centre: Optional[tuple[tuple[float, float], tuple[float, float]]]):
        self._transformer = transformer
        self._centre = centre

    def to_court(self, x: float, y: float) -> tuple[float, float]:
        court_x, court_y = self._transformer.to_court(x, y)
        if self._centre is not None:
            (x1, y1), (x2, y2) = self._centre
            t = (court_y - y1) / (y2 - y1) if abs(y2 - y1) > 1e-6 else 0.0
            court_x -= x1 + t * (x2 - x1)
        return court_x, court_y


def load_court(court_json: str, width: int, height: int) -> Optional[CentredCourt]:
    """Image -> court metres for the video's size (the markup may be made on another size)."""
    result = CourtTransformer(court_json).load()
    geometry = result.geometry
    if geometry is None or result.matrix is None:
        return None
    matrix = result.matrix
    if (geometry.image_width, geometry.image_height) != (width, height):
        sx, sy = width / geometry.image_width, height / geometry.image_height
        keypoints = tuple((x * sx, y * sy) for x, y in geometry.keypoints)
        geometry = CourtGeometry(
            length_m=geometry.length_m,
            width_m=geometry.width_m,
            net_height_m=geometry.net_height_m,
            image_width=width,
            image_height=height,
            keypoints=keypoints,
        )
        matrix = CourtTransformer(None)._calculate_transform(keypoints)
    transformer = CoordinateTransformer(geometry, matrix)
    # The centre line marks (5_center_left, 6_center_right) follow the four corners.
    centre = None
    if len(geometry.keypoints) >= 6:
        centre = (transformer.to_court(*geometry.keypoints[4]), transformer.to_court(*geometry.keypoints[5]))
    return CentredCourt(transformer, centre)


def load_detections(path: str) -> tuple[dict[int, list[tuple[int, np.ndarray]]], dict[int, list[np.ndarray]]]:
    """Player boxes ``frame -> [(track id, xyxy)]`` and ball boxes ``frame -> [xyxy]``, frame pixels."""
    players: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    balls: dict[int, list[np.ndarray]] = defaultdict(list)
    for frame, items in iter_frames(path):
        for item in items:
            box = item.get("bbox_xyxy")
            if box is None:
                continue
            if item.get("class_name") == "player" and isinstance(item.get("track_id"), int):
                players[frame].append((item["track_id"], np.asarray(box, dtype=np.float64)))
            elif item.get("class_name") == "ball":
                balls[frame].append(np.asarray(box, dtype=np.float64))
    return players, balls


def _near_frames(frame: int, reach: int = 2) -> list[int]:
    return sorted(range(frame - reach, frame + reach + 1), key=lambda f: abs(f - frame))


def ball_radius(balls: dict[int, list[np.ndarray]], frame: int, x: float, y: float) -> Optional[float]:
    for f in _near_frames(frame):
        candidates = [b for b in balls.get(f, []) if abs((b[0] + b[2]) / 2 - x) < 60 and abs((b[1] + b[3]) / 2 - y) < 60]
        if candidates:
            box = min(candidates, key=lambda b: ((b[0] + b[2]) / 2 - x) ** 2 + ((b[1] + b[3]) / 2 - y) ** 2)
            return float(((box[2] - box[0]) + (box[3] - box[1])) / 4)
    return None


def track_box(players: dict[int, list[tuple[int, np.ndarray]]], frame: int, track_id: Any) -> Optional[np.ndarray]:
    if not isinstance(track_id, int):
        return None
    for f in _near_frames(frame):
        for tid, box in players.get(f, []):
            if tid == track_id:
                return box
    return None


def iou(a: np.ndarray, b: np.ndarray) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return float(inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter))


def contact_window(x: float, y: float, width: int, height: int) -> tuple[int, int, int]:
    """Square ``(x0, y0, side)`` around the ball, shifted into the frame."""
    side = min(int(round(WINDOW_SHARE * height)), width, height)
    x0 = int(round(x - side / 2))
    y0 = int(round(y - side / 2 + WINDOW_LOWER * side))
    return min(max(x0, 0), width - side), min(max(y0, 0), height - side), side


def _point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(float(np.dot(ab, ab)), 1e-9), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def torso_length(person: Person) -> Optional[float]:
    s, k = person.scores, person.keypoints
    if min(s[L_SHOULDER], s[R_SHOULDER], s[L_HIP], s[R_HIP]) < KEYPOINT_THR:
        return None
    shoulders = (k[L_SHOULDER] + k[R_SHOULDER]) / 2
    hips = (k[L_HIP] + k[R_HIP]) / 2
    length = float(np.linalg.norm(shoulders - hips))
    return length if length >= 4 else None


def hand_score(person: Person, ball: np.ndarray, radius: Optional[float], contact_frame: int) -> Optional[float]:
    """How well the person's forearm meets the ball, lower is better; None without arms."""
    torso = torso_length(person)
    if torso is None:
        return None
    best: Optional[float] = None
    for elbow, wrist in ((L_ELBOW, L_WRIST), (R_ELBOW, R_WRIST)):
        if min(person.scores[elbow], person.scores[wrist]) < KEYPOINT_THR:
            continue
        a, b = person.keypoints[elbow], person.keypoints[wrist]
        tip = b + FOREARM_EXTENSION * (b - a)
        distance = max(_point_segment_distance(ball, a, tip) - (radius or 0.0), 0.0) / torso
        best = distance if best is None else min(best, distance)
    if best is None:
        return None
    score = best + FRAME_PENALTY * abs(person.frame - contact_frame)
    if radius:
        score += DEPTH_PENALTY * abs(math.log(torso / radius / DEPTH_RATIO))
    return score


def feet_point(person: Person) -> Optional[tuple[float, float]]:
    """Between the ankles, at the lower one: where the player stands."""
    visible = [index for index in (L_ANKLE, R_ANKLE) if person.scores[index] >= ANKLE_THR]
    if not visible:
        return None
    points = person.keypoints[visible]
    return float(points[:, 0].mean()), float(points[:, 1].max())


def follow(people: dict[int, list[Person]], start: Person, frames: Sequence[int]) -> list[Person]:
    """The start person over ``frames`` (walked away from his frame), matched by box overlap."""
    chain = [start]
    for direction in (-1, 1):
        reference = start
        ordered = sorted((f for f in frames if (f - start.frame) * direction > 0), key=lambda f: abs(f - start.frame))
        for frame in ordered:
            candidates = people.get(frame, [])
            if not candidates:
                continue
            best = max(candidates, key=lambda person: iou(person.box, reference.box))
            if iou(best.box, reference.box) < MATCH_IOU:
                continue
            chain.append(best)
            reference = best
    return sorted(chain, key=lambda person: person.frame)


def _people(pose: Pose, frame: int, x0: int, y0: int, scale: float) -> list[Person]:
    found = []
    for box, keypoints, scores in zip(pose["bboxes"], pose["keypoints"], pose["keypoint_scores"]):
        offset = np.array([x0, y0], dtype=np.float64)
        found.append(
            Person(
                frame=frame,
                box=np.asarray(box, dtype=np.float64) * scale + np.tile(offset, 2),
                keypoints=np.asarray(keypoints, dtype=np.float64) * scale + offset,
                scores=np.asarray(scores, dtype=np.float64),
            )
        )
    return found


def window_frames(contact: int, back: int, step: int) -> list[int]:
    """The frames posed for a contact: every ``step``-th one from ``back`` before it, and the touch."""
    frames = set(range(contact - back, contact, max(step, 1))) | {contact + offset for offset in TOUCH_OFFSETS}
    return sorted(frame for frame in frames if frame >= 0)


def window_boxes(players: dict[int, list[tuple[int, np.ndarray]]], frame: int, window: tuple[int, int, int]) -> list[np.ndarray]:
    """Detector player boxes of the frame (the detector runs on every second one: else the nearest)
    with at least half of their area inside the window."""
    x0, y0, side = window
    for f in _near_frames(frame, 1):
        if f in players:
            break
    inside = []
    for _, box in players.get(f, []):
        w = min(box[2], x0 + side) - max(box[0], x0)
        h = min(box[3], y0 + side) - max(box[1], y0)
        area = (box[2] - box[0]) * (box[3] - box[1])
        if w > 0 and h > 0 and area > 0 and w * h >= 0.5 * area:
            inside.append(box)
    return inside


def read_windows(
    video_path: str, plans: dict[int, tuple[int, int, int]], back: int, step: int
) -> Iterator[tuple[int, int, np.ndarray]]:
    """``(contact frame, frame, image)`` for every frame of every contact's window, in video order."""
    wanted: dict[int, list[int]] = defaultdict(list)
    for contact in plans:
        for frame in window_frames(contact, back, step):
            wanted[frame].append(contact)

    def frame_source() -> Iterator[tuple[int, np.ndarray]]:
        if seek_enabled():
            yield from SeekingReader(video_path).frames(wanted)
            return
        cap = cv2.VideoCapture(video_path)
        index = 0
        final = max(wanted, default=-1)
        try:
            while index <= final and cap.grab():
                if index in wanted:
                    ok, image = cap.retrieve()
                    if not ok:
                        break
                    yield index, image
                index += 1
        finally:
            cap.release()

    for frame, image in frame_source():
        for contact in wanted[frame]:
            yield contact, frame, image


def load_rally_tracks(tracks_dir: str) -> dict[str, dict[str, Any]]:
    tracks: dict[str, dict[str, Any]] = {}
    for path in sorted(glob.glob(os.path.join(tracks_dir, "track_*.json"))):
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        classification = data.get("rally_classification")
        if isinstance(classification, dict) and classification.get("is_rally") is False:
            continue
        if contacts_of(data):
            tracks[path] = data
    return tracks


def contacts_of(track: dict[str, Any]) -> list[dict[str, Any]]:
    interaction = track.get("player_interaction")
    contacts = interaction.get("contacts") if isinstance(interaction, dict) else None
    if not isinstance(contacts, list):
        return []
    return [c for c in contacts if isinstance(c, dict) and isinstance(c.get("frame"), (int, float))]


def side_names(tracks: dict[str, dict[str, Any]]) -> tuple[str, str]:
    """(court_x < 0, court_x > 0): near/far for a camera behind the end line, left/right beside it."""
    for track in tracks.values():
        for contact in contacts_of(track):
            side = contact.get("detector_side", contact.get("side"))
            if side in ("left", "right"):
                return "left", "right"
    return "near", "far"


def resolve_contact(
    contact: dict[str, Any],
    people: dict[int, list[Person]],
    players: dict[int, list[tuple[int, np.ndarray]]],
    balls: dict[int, list[np.ndarray]],
    court: CentredCourt,
    names: tuple[str, str],
    frames: Sequence[int],
) -> Optional[dict[str, Any]]:
    """The pose evidence of one contact: toucher, his feet and side; None when nothing was found."""
    frame = int(contact["frame"])
    ball = np.array([float(contact["ball_x"]), float(contact["ball_y"])])
    radius = ball_radius(balls, frame, ball[0], ball[1])
    detector_track = contact.get("detector_player_track_id", contact.get("player_track_id"))

    best: Optional[tuple[float, Person]] = None
    for offset in TOUCH_OFFSETS:
        for person in people.get(frame + offset, []):
            score = hand_score(person, ball, radius, frame)
            if score is not None and (best is None or score < best[0]):
                best = (score, person)

    # The detector's player in the pose: the person over his box, nearest the contact frame.
    detector_box = track_box(players, frame, detector_track)
    detector: Optional[tuple[Optional[float], Person]] = None
    if detector_box is not None:
        for offset in sorted(TOUCH_OFFSETS, key=abs):
            candidates = [p for p in people.get(frame + offset, []) if iou(p.box, detector_box) >= MATCH_IOU]
            if candidates:
                person = max(candidates, key=lambda p: iou(p.box, detector_box))
                detector = (hand_score(person, ball, radius, frame), person)
                break

    toucher, source = None, None
    if best is not None and best[0] <= MAX_HAND_SCORE and abs(best[1].frame - frame) <= 2:
        detector_score = detector[0] if detector is not None else None
        if detector is None or detector_score is None or detector_score - best[0] >= MIN_GAIN:
            toucher, source = best[1], "pose"
    if toucher is None and detector is not None:
        toucher, source = detector[1], "detector"
    if toucher is None:
        return None

    chain = follow(people, toucher, frames)
    feet = [(person, point) for person in chain if (point := feet_point(person)) is not None]
    if not feet:
        return None
    # Lowest feet in the image: the frame he stands on the floor, not in a jump.
    grounded, (fx, fy) = max(feet, key=lambda item: item[1][1])
    court_x, court_y = court.to_court(fx, fy)
    sides = [court.to_court(*point)[0] for _, point in feet]
    evidence: dict[str, Any] = {
        "toucher": source,
        "hand_score": round(best[0], 3) if best is not None else None,
        "detector_hand_score": round(detector[0], 3) if detector is not None and detector[0] is not None else None,
        "feet_frame": grounded.frame,
        "feet_xy": [round(fx, 1), round(fy, 1)],
        "court_x": round(court_x, 3),
        "court_y": round(court_y, 3),
        "frames": len(feet),
        # Share of the followed frames whose feet fall on the same half: low on a jump at the net.
        "agreement": round(sum((x < 0) == (court_x < 0) for x in sides) / len(sides), 2),
    }
    if math.isfinite(court_x) and abs(court_x) > NET_DEAD_ZONE_M:
        evidence["side"] = names[0] if court_x < 0 else names[1]
    if source == "pose":
        # The detector's track under the pose toucher, to keep the player link (jersey numbers).
        tracked = [
            (iou(box, toucher.box), tid)
            for f in _near_frames(toucher.frame)
            for tid, box in players.get(f, [])
        ]
        overlap, track_id = max(tracked, default=(0.0, None))
        if track_id is not None and overlap >= MATCH_IOU:
            evidence["track_id"] = track_id
    return evidence


def refine_tracks(
    video_path: str,
    players_json_path: str,
    tracks_dir: str,
    court_json: str,
    predictor: Predictor,
    back: int = 15,
    step: int = 3,
    batch_size: int = 8,
    dry_run: bool = False,
    only_frames: Optional[set[int]] = None,
) -> dict[str, int]:
    """Write the pose side and toucher into the contacts of the rally tracks; returns counts."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    court = load_court(court_json, width, height)
    if court is None:
        raise ValueError(f"No court markup in {court_json}")
    tracks = load_rally_tracks(tracks_dir)
    names = side_names(tracks)
    players, balls = load_detections(players_json_path)

    plans: dict[int, tuple[int, int, int]] = {}
    for track in tracks.values():
        for contact in contacts_of(track):
            if only_frames is not None and int(contact["frame"]) not in only_frames:
                continue
            if isinstance(contact.get("ball_x"), (int, float)) and isinstance(contact.get("ball_y"), (int, float)):
                plans[int(contact["frame"])] = contact_window(contact["ball_x"], contact["ball_y"], width, height)

    people: dict[int, dict[int, list[Person]]] = defaultdict(dict)
    batch: list[tuple[int, int, np.ndarray]] = []

    pose_seconds = 0.0
    pose_boxes = 0

    def run_batch() -> None:
        nonlocal pose_seconds, pose_boxes
        jobs = [PoseJob(frame, image, plans[contact], window_boxes(players, frame, plans[contact])) for contact, frame, image in batch]
        started = time.perf_counter()
        found = predictor(jobs)
        pose_seconds += time.perf_counter() - started
        pose_boxes += sum(len(job.boxes) for job in jobs)
        for (contact, frame, _), persons in zip(batch, found):
            people[contact][frame] = persons
        batch.clear()

    for item in read_windows(video_path, plans, back, step):
        batch.append(item)
        if len(batch) >= batch_size:
            run_batch()
    if batch:
        run_batch()

    jobs_total = sum(len(frames) for frames in people.values())
    LOG.info(
        "Pose: %d contacts, %d frames, %d detector boxes in the windows, %.1f s (%.1f ms a frame)",
        len(people), jobs_total, pose_boxes, pose_seconds, 1000 * pose_seconds / max(jobs_total, 1),
    )
    counts: dict[str, int] = defaultdict(int)
    for path, track in tracks.items():
        for contact in contacts_of(track):
            frame = int(contact["frame"])
            if only_frames is not None and frame not in only_frames:
                continue
            if frame not in plans:
                counts["no ball"] += 1
                continue
            evidence = resolve_contact(contact, people[frame], players, balls, court, names, window_frames(frame, back, step))
            detector_side = contact.get("detector_side", contact.get("side"))
            detector_track = contact.get("detector_player_track_id", contact.get("player_track_id"))
            if evidence is None:
                counts["no pose"] += 1
                contact.pop("pose", None)
                continue
            contact["pose"] = evidence
            contact["detector_side"] = detector_side
            contact["detector_player_track_id"] = detector_track
            contact["side"] = evidence.get("side", detector_side)
            contact["player_track_id"] = evidence.get("track_id", detector_track)
            counts[f"toucher {evidence['toucher']}"] += 1
            if contact["player_track_id"] != detector_track:
                counts["player changed"] += 1
            if contact["side"] != detector_side:
                counts["side changed"] += 1
                LOG.info(
                    "frame %s: %s -> %s (toucher %s, court_x %.2f, feet at %s)",
                    frame, detector_side, contact["side"], evidence["toucher"], evidence["court_x"], evidence["feet_frame"],
                )
        if not dry_run:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(track, handle, indent=2, ensure_ascii=False)
    return dict(counts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refine the toucher and his court side of each contact by pose")
    parser.add_argument("video")
    parser.add_argument("--players_json_path", required=True, help="detector predictions (players and ball)")
    parser.add_argument("--tracks_dir", required=True, help="directory with track_*.json")
    parser.add_argument("--court_json", required=True, help="court markup (COCO keypoints)")
    parser.add_argument("--pose_root", default=str(DEFAULT_POSE_ROOT), help="the pose project (infer_rtmo.py)")
    parser.add_argument("--model", choices=("rtmo", "rtmpose"), default="rtmo",
                        help="rtmo: bottom-up on the window; rtmpose: top-down on the detector's boxes")
    parser.add_argument("--checkpoint", default=None, help="default: rtmo_t_vaa_ft / rtmpose_s_vaa_ft best_AP.pth")
    parser.add_argument("--no_flip", action="store_true", help="rtmpose without the flip test (about 2x faster)")
    parser.add_argument("--input_size", type=int, default=960, help="rtmo input size")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--back", type=int, default=15, help="frames before the contact to look for the feet")
    parser.add_argument("--step", type=int, default=3, help="pose every step-th frame of those before the contact")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--frames", type=int, nargs="*", help="only these contact frames (debugging)")
    parser.add_argument("--dry_run", action="store_true", help="report only, do not write the tracks")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args()
    if args.model == "rtmpose":
        predictor = build_rtmpose_predictor(Path(args.pose_root), args.checkpoint or DEFAULT_RTMPOSE_CHECKPOINT, args.device, not args.no_flip)
    else:
        predictor = build_rtmo_predictor(Path(args.pose_root), args.checkpoint or DEFAULT_CHECKPOINT, args.input_size, args.device)
    counts = refine_tracks(
        args.video, args.players_json_path, args.tracks_dir, args.court_json, predictor,
        args.back, args.step, args.batch_size, args.dry_run,
        set(args.frames) if args.frames else None,
    )
    LOG.info("Pose refinement: %s", ", ".join(f"{name} {count}" for name, count in sorted(counts.items())) or "none")


if __name__ == "__main__":
    main()
