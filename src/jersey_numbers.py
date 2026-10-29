"""Read the jersey number of a tracked player: pose -> torso crop -> PP-OCRv6 text recognition.

Measured on a match in one kit colour (vb-action ``tools/action_eval/jersey_number_probe.py``,
first set of project 30a64638, 70 actions whose number a person can read): every 4th frame of
the track +-60 frames around the contact, the torso cut by the pose keypoints (a fixed part of
the box reads 7 of 70 instead of 38), reads voted by confidence with single digits giving their
weight to the two-digit number they are part of -> 38 right, 3 wrong.

Both models run on OpenVINO from ``ov/``:
- ``rtmpose_s_vaa_ft.xml`` -- RTMPose-s top-down (``~/projects/pose``, fine-tuned on players): the
  person crop warped to 192x256 -> SimCC logits of the 17 COCO keypoints;
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
POSE_XML = OV_DIR / "rtmpose_s_vaa_ft.xml"
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


def torso_crop(person: np.ndarray, keypoints: np.ndarray, min_conf: float = 0.3) -> Optional[np.ndarray]:
    """The number's patch of one person crop (``keypoints`` [17, 3] in crop pixels): from the shoulders
    down 80% of the way to the hips, 15% wider each side.

    The cut is tight on purpose: the OCR resizes the patch to a 48 px line, so every strip of shorts
    or space above the shoulders shrinks the digits. On 239 legible crops of four matches the old
    cut (5% above the shoulders, down to the hips) read digits in 59%, this one in 81%.
    """
    points = keypoints[TORSO_POINTS]
    if (points[:, 2] < min_conf).any():
        return None
    x0, x1 = points[:, 0].min(), points[:, 0].max()
    y0, y1 = points[:2, 1].min(), points[2:, 1].max()
    width, height = x1 - x0, y1 - y0
    if width < 8 or height < 12:
        return None
    h, w = person.shape[:2]
    crop = person[int(max(y0, 0)) : int(min(y1 - 0.2 * height, h)), int(max(x0 - 0.15 * width, 0)) : int(min(x1 + 0.15 * width, w))]
    return crop if crop.size else None


class PoseModel:
    """RTMPose on one person crop: the keypoints [17, 3] (x, y in crop pixels, SimCC score) of the person it frames.

    Top-down: the whole crop is the person's box (``person_crop`` already adds the margins), padded
    by ``padding`` and warped to the model input as mmpose does. SimCC scores run lower than YOLO's
    keypoint confidences (a clear torso point sits at 0.3-0.6), hence the lower ``torso_crop`` threshold.
    """

    # ImageNet statistics on RGB, as the model was trained (mmpose data preprocessor).
    MEAN = np.array([123.675, 116.28, 103.53], np.float32)
    STD = np.array([58.395, 57.12, 57.375], np.float32)

    def __init__(self, model_xml: Path = POSE_XML, device: str = "CPU", padding: float = 1.25) -> None:
        import openvino as ov

        self._model = ov.Core().compile_model(str(model_xml), device)
        shape = self._model.input(0).shape
        self.input_size = (int(shape[3]), int(shape[2]))  # (w, h)
        self.split_ratio = self._model.output(0).shape[2] / self.input_size[0]
        self.padding = padding

    def _warp(self, h: float, w: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Affine matrix of the box (0, 0, w, h) onto the input, with the box's centre and padded size."""
        in_w, in_h = self.input_size
        center = np.array([w / 2, h / 2], np.float32)
        box_w, box_h = w * self.padding, h * self.padding
        # The same aspect ratio as the input: the box is widened or heightened, never squeezed.
        if box_w > box_h * in_w / in_h:
            box_h = box_w * in_h / in_w
        else:
            box_w = box_h * in_w / in_h
        scale = np.array([box_w, box_h], np.float32)
        src = np.float32([center, center + [-box_w / 2, 0], center + [0, -box_w / 2]])
        dst = np.float32([[in_w / 2, in_h / 2], [0, in_h / 2], [in_w / 2, in_h / 2 - in_w / 2]])
        return cv2.getAffineTransform(src, dst), center, scale

    def __call__(self, person_bgr: np.ndarray) -> Optional[np.ndarray]:
        h, w = person_bgr.shape[:2]
        if h < 2 or w < 2:
            return None
        matrix, center, scale = self._warp(h, w)
        crop = cv2.warpAffine(person_bgr, matrix, self.input_size, flags=cv2.INTER_LINEAR)
        x = ((crop[:, :, ::-1].astype(np.float32) - self.MEAN) / self.STD).transpose(2, 0, 1)[None]
        simcc_x, simcc_y = (out[0] for out in self._model(np.ascontiguousarray(x)).values())  # [17, w*r], [17, h*r]
        px, py = simcc_x.argmax(1) / self.split_ratio, simcc_y.argmax(1) / self.split_ratio
        scores = np.minimum(simcc_x.max(1), simcc_y.max(1))
        # Input pixels back to crop pixels: the input covers the padded box around the centre.
        in_w, in_h = self.input_size
        keypoints = np.stack(
            [px / in_w * scale[0] + center[0] - scale[0] / 2, py / in_h * scale[1] + center[1] - scale[1] / 2, scores], 1
        ).astype(np.float32)
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
