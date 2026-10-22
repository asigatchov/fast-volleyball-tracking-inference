"""Read the jersey number of a tracked player: pose -> torso crop -> PP-OCRv6 text recognition.

Measured on a match in one kit colour (vb-action ``tools/action_eval/jersey_number_probe.py``,
first set of project 30a64638, 70 actions whose number a person can read): every 4th frame of
the track +-60 frames around the contact, the torso cut by the pose keypoints (a fixed part of
the box reads 7 of 70 instead of 38), reads voted by confidence with single digits giving their
weight to the two-digit number they are part of -> 38 right, 3 wrong.

Both models run on OpenVINO from ``ov/``:
- ``yolo26m_pose.xml`` -- end-to-end YOLO pose, [N, 300, 57]: box, score, class, 17 keypoints;
- ``ppocrv6_medium_rec.xml`` + ``.json`` -- PP-OCRv6 text-line recognizer and its characters.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

OV_DIR = Path(__file__).resolve().parents[1] / "ov"
POSE_XML = OV_DIR / "yolo26m_pose.xml"
OCR_XML = OV_DIR / "ppocrv6_medium_rec.xml"
DIGITS = re.compile(r"^\d{1,2}$")
# COCO keypoints: 5/6 shoulders, 11/12 hips.
TORSO_POINTS = [5, 6, 11, 12]
Box = tuple[float, float, float, float]


def person_crop(image: np.ndarray, box: Box) -> Optional[np.ndarray]:
    """The tracked box with a margin (10% across, 5% up and down), as the probe cut it."""
    x0, y0, x1, y1 = box
    px, py = 0.1 * (x1 - x0), 0.05 * (y1 - y0)
    crop = image[int(max(y0 - py, 0)) : int(min(y1 + py, image.shape[0])), int(max(x0 - px, 0)) : int(min(x1 + px, image.shape[1]))]
    return crop if crop.size and crop.shape[0] >= 40 else None


def torso_crop(person: np.ndarray, keypoints: np.ndarray, min_conf: float = 0.5) -> Optional[np.ndarray]:
    """Shoulders-to-hips box of one person crop (``keypoints`` [17, 3] in crop pixels), 15% wider each side."""
    points = keypoints[TORSO_POINTS]
    if (points[:, 2] < min_conf).any():
        return None
    x0, x1 = points[:, 0].min(), points[:, 0].max()
    y0, y1 = points[:2, 1].min(), points[2:, 1].max()
    width, height = x1 - x0, y1 - y0
    if width < 8 or height < 12:
        return None
    h, w = person.shape[:2]
    crop = person[int(max(y0 - 0.05 * height, 0)) : int(min(y1, h)), int(max(x0 - 0.15 * width, 0)) : int(min(x1 + 0.15 * width, w))]
    return crop if crop.size else None


class PoseModel:
    """YOLO pose on one person crop: the keypoints of the biggest person found, or None."""

    def __init__(self, model_xml: Path = POSE_XML, device: str = "CPU", size: int = 320, min_score: float = 0.25) -> None:
        import openvino as ov

        self._model = ov.Core().compile_model(str(model_xml), device)
        self.size = size
        self.min_score = min_score

    def __call__(self, person_bgr: np.ndarray) -> Optional[np.ndarray]:
        h, w = person_bgr.shape[:2]
        scale = self.size / max(h, w)
        nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
        ph, pw = math.ceil(nh / 32) * 32, math.ceil(nw / 32) * 32
        canvas = np.full((ph, pw, 3), 114, np.uint8)
        top, left = (ph - nh) // 2, (pw - nw) // 2
        canvas[top : top + nh, left : left + nw] = cv2.resize(person_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        detections = self._model(np.ascontiguousarray(x))[0][0]  # [300, 57]
        detections = detections[detections[:, 4] >= self.min_score]
        if not len(detections):
            return None
        areas = (detections[:, 2] - detections[:, 0]) * (detections[:, 3] - detections[:, 1])
        keypoints = detections[int(areas.argmax()), 6:].reshape(17, 3).copy()
        keypoints[:, 0] = (keypoints[:, 0] - left) / scale
        keypoints[:, 1] = (keypoints[:, 1] - top) / scale
        return keypoints


class TextReader:
    """PP-OCRv6 text-line recognizer: BGR crops -> (text, score), CTC-decoded as PaddleOCR does."""

    def __init__(self, model_xml: Path = OCR_XML, device: str = "CPU") -> None:
        import openvino as ov

        self._model = ov.Core().compile_model(str(model_xml), device)
        self.characters: list[str] = json.loads(model_xml.with_suffix(".json").read_text(encoding="utf-8"))["characters"]

    @staticmethod
    def _batch(crops: list[np.ndarray], height: int = 48, min_width: int = 320) -> np.ndarray:
        ratios = [c.shape[1] / max(c.shape[0], 1) for c in crops]
        width = max(min_width, int(height * max(ratios)))
        batch = np.zeros((len(crops), 3, height, width), np.float32)
        for i, (crop, ratio) in enumerate(zip(crops, ratios)):
            w = min(width, max(1, math.ceil(height * ratio)))
            image = cv2.resize(crop, (w, height)).astype(np.float32) / 255.0
            batch[i, :, :, :w] = ((image - 0.5) / 0.5).transpose(2, 0, 1)
        return batch

    def __call__(self, crops: list[np.ndarray], batch: int = 32) -> list[tuple[str, float]]:
        # Similar aspect ratios together: less padding in a batch.
        order = sorted(range(len(crops)), key=lambda i: crops[i].shape[1] / max(crops[i].shape[0], 1))
        results: dict[int, tuple[str, float]] = {}
        for start in range(0, len(order), batch):
            chunk = order[start : start + batch]
            probs = self._model(self._batch([crops[i] for i in chunk]))[0]  # [B, T, classes]
            ids, best = probs.argmax(-1), probs.max(-1)
            for i, row, row_p in zip(chunk, ids, best):
                text, scores, previous = [], [], -1
                for token, p in zip(row, row_p):
                    if token != previous and token != 0:
                        text.append(self.characters[token - 1])
                        scores.append(float(p))
                    previous = token
                results[i] = ("".join(text), float(np.mean(scores)) if scores else 0.0)
        return [results[i] for i in range(len(crops))]


def decide_number(
    reads: Iterable[tuple[str, float]],
    threshold: float = 0.6,
    candidates: Optional[set[str]] = None,
    min_share: float = 0.5,
) -> tuple[Optional[str], float]:
    """Number of one action from its reads, and its share of the vote.

    Reads of 1-2 digits at ``threshold`` or above vote with their confidence; a single digit gives
    its weight to the one two-digit number of the action that contains it ("1", "7" -> "17"), and
    without any two-digit read the action gets no number (a lone "1" is as often half of "17").
    ``candidates`` (the team's numbers) keeps only reads among them; then a single digit that is
    itself a candidate stands on its own.
    """
    tally: dict[str, float] = defaultdict(float)
    for text, conf in reads:
        text = text.strip()
        if DIGITS.match(text) and text != "0" and conf >= threshold:
            tally[text.lstrip("0") or "0"] += conf
    if candidates:
        tally = defaultdict(float, {t: w for t, w in tally.items() if t in candidates or any(t in c for c in candidates if len(c) == 2)})
    twos = [t for t in tally if len(t) == 2]
    for single in [t for t in tally if len(t) == 1]:
        hosts = [h for h in twos if single in h]
        if len(hosts) == 1:
            tally[hosts[0]] += tally.pop(single)
    if candidates:
        tally = defaultdict(float, {t: w for t, w in tally.items() if t in candidates})
    elif not twos:
        return None, 0.0
    if not tally:
        return None, 0.0
    number, weight = max(tally.items(), key=lambda t: t[1])
    share = weight / sum(tally.values())
    return (number, share) if share >= min_share else (None, share)


class NumberReader:
    """Pose + OCR together: the torso crop of a tracked box, and the reads of many torsos."""

    def __init__(self, pose_xml: Path = POSE_XML, ocr_xml: Path = OCR_XML, device: str = "CPU") -> None:
        self.pose = PoseModel(pose_xml, device)
        self.reader = TextReader(ocr_xml, device)

    def torso(self, image: np.ndarray, box: Box) -> Optional[np.ndarray]:
        person = person_crop(image, box)
        if person is None:
            return None
        keypoints = self.pose(person)
        torso = torso_crop(person, keypoints) if keypoints is not None else None
        # A copy: the view would keep the whole frame alive while the torsos of all actions are gathered.
        return torso.copy() if torso is not None else None

    def read(self, torsos: list[np.ndarray]) -> list[tuple[str, float]]:
        return self.reader(torsos)
