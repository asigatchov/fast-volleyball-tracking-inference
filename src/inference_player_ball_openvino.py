#!/usr/bin/env python3
"""Standalone OpenVINO video inference for RAVEL-VB.

Копия ``vb-action/RAVEL-VB/infer_openvino.py`` без ветки YOLO. Декодер читает
любой релиз из ``ov/``: тензоры, которых у релиза нет (``ball_grid``,
``ball_size``, ``temporal_association``, ``head_point_norm``,
``body_point_*``), просто пропускаются, а размер кадра, число каналов (серый у
v45) и длина клипа берутся из sidecar ``.json``.

    uv run python src/inference_player_ball_openvino.py video.mp4 \
        --model ov/RAVEL-VB-012-9f.xml --output ../uploads/mix/
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import openvino as ov
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "ov" / "RAVEL-VB-012-9f.xml"

# Точки голов -- как в ``infer_tape_vb6_v31``: пики карты ``body_point_logits``
# (канал 0 -- голова) после подавления 3x3, со сдвигом внутри клетки. Порог
# 0.30 -- тот же, что ``VB6_HEAD_THRESHOLD`` по умолчанию у торчёвого пути.
HEAD_CLASS = "head"
HEAD_CHANNEL = 0
HEAD_THRESHOLD = 0.30
HEAD_NMS_KERNEL = 3
# Половина крошечной рамки, в которой точка едет через слияние и трекер:
# около 2.5 px на 640x360.
POINT_HALF_WIDTH = 0.004
POINT_HALF_HEIGHT = 0.007
# Зона головы в рамке, доли её высоты: наклонённый игрок держит голову ниже
# верхнего края, поэтому зона доходит почти до середины.
ZONE_ABOVE = 0.08
ZONE_BELOW = 0.45
MATCHED_COLOUR = (0, 220, 0)
LOOSE_COLOUR = (0, 0, 255)


def _box_iou(left: list[float], right: list[float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(left_area + right_area - intersection, 1e-9)


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    order = scores.argsort()[::-1]
    kept: list[int] = []
    while order.size:
        current = int(order[0])
        kept.append(current)
        if order.size == 1:
            break
        remaining = order[1:]
        left = boxes[current]
        right = boxes[remaining]
        x1 = np.maximum(left[0], right[:, 0])
        y1 = np.maximum(left[1], right[:, 1])
        x2 = np.minimum(left[2], right[:, 2])
        y2 = np.minimum(left[3], right[:, 3])
        intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        left_area = max(0.0, float(left[2] - left[0])) * max(
            0.0, float(left[3] - left[1])
        )
        right_area = np.maximum(0.0, right[:, 2] - right[:, 0]) * np.maximum(
            0.0, right[:, 3] - right[:, 1]
        )
        overlap = intersection / np.maximum(
            left_area + right_area - intersection, 1e-9
        )
        order = remaining[overlap <= threshold]
    return np.asarray(kept, dtype=np.int64)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=-1, keepdims=True)


def head_points(
    outputs: dict[str, np.ndarray], threshold: float
) -> list[list[dict]]:
    """Все пики карты голов выше ``threshold``, по клипам и кадрам.

    Numpy-повтор ``infer_tape_vb6_v31.head_points``: максимум 3x3 оставляет
    только локальные пики, сдвиг внутри клетки уводит точку с её центра.
    """
    logits = outputs["body_point_logits"][:, :, HEAD_CHANNEL]
    offsets = outputs["body_point_offset"][:, :, HEAD_CHANNEL]
    scores = 1.0 / (1.0 + np.exp(-logits.astype(np.float32)))
    batch, frames, grid_height, grid_width = scores.shape
    pad = HEAD_NMS_KERNEL // 2
    padded = np.pad(
        scores, ((0, 0), (0, 0), (pad, pad), (pad, pad)), constant_values=-np.inf
    )
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, (HEAD_NMS_KERNEL, HEAD_NMS_KERNEL), axis=(2, 3)
    )
    peaks = np.where(windows.max(axis=(-2, -1)) == scores, scores, 0.0)

    found: list[list[dict]] = [[] for _ in range(batch)]
    for index, frame, row, column in np.argwhere(peaks >= threshold).tolist():
        x = (column + float(offsets[index, frame, 0, row, column])) / grid_width
        y = (row + float(offsets[index, frame, 1, row, column])) / grid_height
        found[index].append(
            {
                "frame_slot": frame,
                "class_id": 2,
                "class_name": HEAD_CLASS,
                "score": round(float(peaks[index, frame, row, column]), 4),
                "bbox_xyxy_norm": [
                    x - POINT_HALF_WIDTH,
                    y - POINT_HALF_HEIGHT,
                    x + POINT_HALF_WIDTH,
                    y + POINT_HALF_HEIGHT,
                ],
                "point_xy_norm": [round(x, 5), round(y, 5)],
                "query_index": None,
            }
        )
    return found


def match_heads_to_boxes(
    boxes: list[list[float]], points: list[tuple[int, int]]
) -> list[int | None]:
    """Одна голова на рамку, одна рамка на голову; ``None`` -- точка без рамки.

    Точка может лежать в зоне головы сразу нескольких рамок -- так бывает с
    высокой рамкой переднего плана, -- поэтому пары выбираются глобально, по
    сумме расстояний до центра верхнего края, а не жадно. Как в
    ``infer_tape_vb6_v31.match_heads_to_boxes``.
    """
    owner: list[int | None] = [None] * len(points)
    if not boxes or not points:
        return owner
    cost = np.full((len(boxes), len(points)), 1e6, dtype=np.float32)
    for row, (x1, y1, x2, y2) in enumerate(boxes):
        height = max(1.0, y2 - y1)
        anchor = ((x1 + x2) / 2, y1)
        for column, (x, y) in enumerate(points):
            if not x1 <= x <= x2:
                continue
            if not y1 - ZONE_ABOVE * height <= y <= y1 + ZONE_BELOW * height:
                continue
            cost[row, column] = float(np.hypot(x - anchor[0], y - anchor[1]))
    rows, columns = linear_sum_assignment(cost)
    for row, column in zip(rows, columns):
        if cost[row, column] < 1e6:
            owner[column] = int(row)
    return owner


def decode_batch(
    outputs: dict[str, np.ndarray],
    score_threshold: float,
    ball_width: float,
    ball_height: float,
    include_features: bool = False,
    iou_threshold: float = 0.5,
    max_players: int = 16,
    ball_threshold: float | None = None,
    head_threshold: float = HEAD_THRESHOLD,
) -> list[list[dict]]:
    """Decode the tensor ABI exported with the RAVEL-VB OpenVINO model.

    ``ball_threshold`` -- порог мяча; ``None`` оставляет прежнее поведение,
    когда мяч отсекается тем же ``score_threshold``, что и игроки.
    ``head_threshold`` -- порог пиков карты голов; работает, только если
    модель отдаёт ``body_point_logits``.
    """
    if ball_threshold is None:
        ball_threshold = score_threshold
    results: list[list[dict]] = []
    batch_size, frame_count, query_count = outputs["logits"].shape[:3]
    for batch_index in range(batch_size):
        sample: list[dict] = []
        track_ids = np.arange(query_count, dtype=np.int64)
        next_track_id = query_count
        for frame_slot in range(frame_count):
            if frame_slot and "temporal_forward_association" in outputs:
                links = outputs["temporal_forward_association"][
                    batch_index, frame_slot - 1
                ]
                previous_scores = links[:, :-1]
                previous_queries = previous_scores.argmax(axis=-1)
                previous_probability = previous_scores[
                    np.arange(query_count), previous_queries
                ]
                dustbin_probability = links[:, -1]
                linked = previous_probability > dustbin_probability
                current_track_ids = np.empty(query_count, dtype=np.int64)
                current_track_ids[linked] = track_ids[previous_queries[linked]]
                new_count = int((~linked).sum())
                current_track_ids[~linked] = np.arange(
                    next_track_id,
                    next_track_id + new_count,
                    dtype=np.int64,
                )
                next_track_id += new_count
                track_ids = current_track_ids
            elif frame_slot and "temporal_association" in outputs:
                links = outputs["temporal_association"][
                    batch_index, frame_slot - 1
                ]
                track_ids = track_ids[links.argmax(axis=-1)]

            probabilities = _softmax(
                outputs["logits"][batch_index, frame_slot]
            )
            scores = probabilities[:, 1]
            boxes_cxcywh = outputs["boxes"][batch_index, frame_slot]
            boxes = np.empty_like(boxes_cxcywh)
            boxes[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
            boxes[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
            boxes[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
            boxes[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2
            boxes = boxes.clip(0.0, 1.0)
            selected = np.flatnonzero(scores >= score_threshold)
            if selected.size:
                selected = selected[
                    _nms(boxes[selected], scores[selected], iou_threshold)
                ]
                selected = selected[np.argsort(scores[selected])[::-1]][:max_players]
                for query_index in selected:
                    query_index = int(query_index)
                    record = {
                        "frame_slot": frame_slot,
                        "class_id": 0,
                        "class_name": "player",
                        "score": float(scores[query_index]),
                        "bbox_xyxy_norm": boxes[query_index].tolist(),
                        "query_index": query_index,
                        "track_id": int(track_ids[query_index]),
                    }
                    # Голова на запрос (v41+): где голова игрока этой рамки.
                    if "head_point_norm" in outputs:
                        record["head_point_norm"] = outputs["head_point_norm"][
                            batch_index, frame_slot, query_index
                        ].tolist()
                    if include_features:
                        for key in (
                            "head_points",
                            "foot_points",
                            "query_vectors",
                        ):
                            if key in outputs:
                                record[key] = outputs[key][
                                    batch_index, frame_slot, query_index
                                ].tolist()
                    sample.append(record)

            if "ball_grid" in outputs:
                grid = outputs["ball_grid"][batch_index, frame_slot]
                confidence = grid[0]
                flat = int(confidence.argmax())
                score = float(confidence.flat[flat])
                if score >= ball_threshold:
                    row, column = np.unravel_index(flat, confidence.shape)
                    center_x = (column + float(grid[1, row, column])) / confidence.shape[1]
                    center_y = (row + float(grid[2, row, column])) / confidence.shape[0]
                    # Модель, которая сама меряет мяч (VB6 v24 и позже), отдаёт
                    # ``ball_size`` [B, T, 2, H, W]; тогда рамка -- её, а не
                    # медиана разметки, одинаковая у сетки и у дальней линии.
                    box_width, box_height = ball_width, ball_height
                    if "ball_size" in outputs:
                        size = outputs["ball_size"][batch_index, frame_slot]
                        box_width = float(size[0, row, column])
                        box_height = float(size[1, row, column])
                    sample.append(
                        {
                            "frame_slot": frame_slot,
                            "class_id": 1,
                            "class_name": "ball",
                            "score": score,
                            "bbox_xyxy_norm": [
                                max(0.0, center_x - box_width / 2),
                                max(0.0, center_y - box_height / 2),
                                min(1.0, center_x + box_width / 2),
                                min(1.0, center_y + box_height / 2),
                            ],
                            "query_index": None,
                        }
                    )
        results.append(sample)
    if "body_point_logits" in outputs and "body_point_offset" in outputs:
        for sample, points in zip(results, head_points(outputs, head_threshold)):
            sample.extend(points)
    return results


def merge_frame_predictions(
    predictions: list[dict], iou_threshold: float = 0.5
) -> list[dict]:
    """Merge duplicate detections emitted by overlapping clips."""
    result: list[dict] = []
    for class_name in sorted({item["class_name"] for item in predictions}):
        candidates = sorted(
            (item for item in predictions if item["class_name"] == class_name),
            key=lambda item: float(item["score"]),
            reverse=True,
        )
        if class_name == "ball":
            if candidates:
                result.append(dict(candidates[0]))
            continue
        while candidates:
            selected = candidates.pop(0)
            result.append(dict(selected))
            candidates = [
                item
                for item in candidates
                if _box_iou(
                    selected["bbox_xyxy_norm"], item["bbox_xyxy_norm"]
                )
                <= iou_threshold
            ]
    return result


class PlayerHysteresis:
    """Assign stable IDs and bridge short player-detection gaps."""

    def __init__(
        self,
        open_threshold: float,
        close_threshold: float,
        hold_frames: int,
        match_iou: float = 0.05,
        max_center_distance: float = 0.08,
    ) -> None:
        self.open_threshold = open_threshold
        self.close_threshold = close_threshold
        self.hold_frames = hold_frames
        self.match_iou = match_iou
        self.max_center_distance = max_center_distance
        self.tracks: dict[int, dict] = {}
        self.next_track_id = 0

    @staticmethod
    def _center_distance(left: list[float], right: list[float]) -> float:
        left_x, left_y = (left[0] + left[2]) / 2, (left[1] + left[3]) / 2
        right_x, right_y = (right[0] + right[2]) / 2, (right[1] + right[3]) / 2
        return ((left_x - right_x) ** 2 + (left_y - right_y) ** 2) ** 0.5

    def update(self, frame_index: int, predictions: list[dict]) -> list[dict]:
        players = [
            dict(item)
            for item in predictions
            if item["class_name"] == "player"
            and float(item["score"]) >= self.close_threshold
        ]
        output = [
            dict(item) for item in predictions if item["class_name"] != "player"
        ]
        pairs: list[tuple[float, int, int]] = []
        for track_id, state in self.tracks.items():
            previous_box = state["prediction"]["bbox_xyxy_norm"]
            for candidate_index, candidate in enumerate(players):
                candidate_box = candidate["bbox_xyxy_norm"]
                overlap = _box_iou(previous_box, candidate_box)
                distance = self._center_distance(previous_box, candidate_box)
                if overlap >= self.match_iou or distance <= self.max_center_distance:
                    pairs.append((overlap - distance, track_id, candidate_index))
        pairs.sort(reverse=True)
        matched_tracks: set[int] = set()
        matched_candidates: set[int] = set()
        for _, track_id, candidate_index in pairs:
            if track_id in matched_tracks or candidate_index in matched_candidates:
                continue
            candidate = players[candidate_index]
            candidate["track_id"] = track_id
            candidate["interpolated"] = False
            self.tracks[track_id] = {
                "prediction": candidate,
                "missing": 0,
            }
            output.append(dict(candidate))
            matched_tracks.add(track_id)
            matched_candidates.add(candidate_index)

        for track_id in list(self.tracks):
            if track_id in matched_tracks:
                continue
            state = self.tracks[track_id]
            state["missing"] += 1
            if state["missing"] > self.hold_frames:
                del self.tracks[track_id]
                continue
            prediction = dict(state["prediction"])
            prediction["frame_index"] = frame_index
            prediction["track_id"] = track_id
            prediction["interpolated"] = True
            prediction["score"] = float(prediction["score"]) * (
                0.9 ** state["missing"]
            )
            output.append(prediction)

        for candidate_index, candidate in enumerate(players):
            if (
                candidate_index in matched_candidates
                or float(candidate["score"]) < self.open_threshold
            ):
                continue
            track_id = self.next_track_id
            self.next_track_id += 1
            candidate["track_id"] = track_id
            candidate["interpolated"] = False
            self.tracks[track_id] = {"prediction": candidate, "missing": 0}
            output.append(dict(candidate))
        return output


class RavelVBOpenVINO:
    def __init__(
        self,
        model_path: Path,
        num_threads: int,
        performance_hint: str,
        device: str,
    ) -> None:
        metadata_path = model_path.with_suffix(".json")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"model metadata not found: {metadata_path}")
        self.metadata: dict[str, Any] = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
        self.config = self.metadata["config"]
        required_config = {
            "image_width",
            "image_height",
            "clip_length",
            "input_channels",
        }
        missing_config = sorted(required_config - self.config.keys())
        if missing_config:
            raise ValueError(
                f"model metadata is missing config fields: {missing_config}"
            )
        core = ov.Core()
        compile_config: dict[str, Any] = {"PERFORMANCE_HINT": performance_hint}
        if num_threads > 0:
            compile_config["INFERENCE_NUM_THREADS"] = num_threads
        compiled = core.compile_model(
            core.read_model(str(model_path)), device, compile_config
        )
        self.device = device
        self.compiled = compiled
        self.request = compiled.create_infer_request()
        self.input = compiled.input(self.metadata.get("input_name", "frames"))
        output_names = self.metadata.get("output_names")
        if not isinstance(output_names, list):
            raise ValueError("model metadata must contain an output_names list")
        required_outputs = {"logits", "boxes"}
        missing_outputs = sorted(required_outputs - set(output_names))
        if missing_outputs:
            raise ValueError(
                f"OpenVINO model is missing required outputs: {missing_outputs}"
            )
        # A ball-less release (VB7 onward: the ball is a separate network) must
        # not be handed ball priors it would silently use; a ball release must
        # have them.  ``decode_batch`` already skips the ball when the tensor is
        # absent, so this is the only place that needs to know.
        self.has_ball = "ball_grid" in output_names
        if self.has_ball:
            ball_missing = sorted(
                {"ball_width_prior", "ball_height_prior"} - self.config.keys()
            )
            if ball_missing:
                raise ValueError(
                    f"ball release is missing config fields: {ball_missing}"
                )
            self.ball_width = float(self.config["ball_width_prior"])
            self.ball_height = float(self.config["ball_height_prior"])
        else:
            self.ball_width = self.ball_height = 0.0
        self.outputs = {name: compiled.output(name) for name in output_names}

    def preprocess(self, bgr: np.ndarray) -> np.ndarray:
        """Кадр -> тензор модели: размер, число каналов и шкала [0, 1].

        Уменьшение идёт через ``INTER_AREA`` -- тем же способом, каким нарезаны
        кадры обучающего датасета; при увеличении area не годится, и тогда
        берётся ``INTER_LINEAR``. Серые релизы (v45/v46 -- один канал на кадр)
        получают одноканальный кадр, цветные -- RGB.
        """
        width = int(self.config["image_width"])
        height = int(self.config["image_height"])
        shrinking = width <= bgr.shape[1] and height <= bgr.shape[0]
        resized = cv2.resize(
            bgr,
            (width, height),
            interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR,
        )
        if int(self.config["input_channels"]) == 1:
            grey = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            return np.ascontiguousarray(
                grey[None].astype(np.float32) / 255.0
            )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(
            rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        )

    def infer(self, clip: np.ndarray) -> tuple[dict[str, np.ndarray], float]:
        started = time.perf_counter()
        result = self.request.infer({self.input: clip})
        elapsed = time.perf_counter() - started
        return (
            {
                name: np.asarray(result[port])
                for name, port in self.outputs.items()
            },
            elapsed,
        )


def _draw_predictions(
    frame: np.ndarray,
    frame_index: int,
    fps: float,
    predictions: list[dict],
) -> np.ndarray:
    canvas = frame.copy()
    players = balls = 0
    for item in predictions:
        class_name = item["class_name"]
        if class_name == HEAD_CLASS:
            continue
        x1, y1, x2, y2 = [int(round(value)) for value in item["bbox_xyxy"]]
        if class_name == "player":
            color = (0, 180, 255)
            players += 1
        else:
            color = (255, 0, 255)
            balls += int(class_name == "ball")
        track_id = item.get("track_id")
        label = class_name if track_id is None else f"{class_name}#{track_id}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            canvas,
            f"{label} {float(item['score']):.2f}",
            (x1, max(54, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    header = (
        f"RAVEL-VB | frame {frame_index} | {frame_index / max(fps, 1e-6):.2f}s | "
        f"players {players} | balls {balls}"
    )
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (20, 20, 20), -1)
    cv2.putText(
        canvas,
        header,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )
    _draw_heads(canvas, predictions)
    return canvas


def _draw_heads(canvas: np.ndarray, predictions: list[dict]) -> None:
    """Точки голов, как у ``infer_tape_vb6_v31``.

        зелёный O      голова, которую взяла рамка, одна на рамку
        красный X      голова, которую не взяла ни одна рамка
        красный верх   рамка, которой не досталось головы
    """
    height, width = canvas.shape[:2]
    centres: list[tuple[int, int]] = []
    for item in predictions:
        if item["class_name"] != HEAD_CLASS:
            continue
        point = item.get("point_xy_norm")
        if point is None:
            x1, y1, x2, y2 = item["bbox_xyxy"]
            centres.append((int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))))
        else:
            centres.append(
                (int(round(point[0] * width)), int(round(point[1] * height)))
            )
    if not centres:
        return

    box_pixels = [
        [float(value) for value in item["bbox_xyxy"]]
        for item in predictions
        if item["class_name"] == "player"
    ]
    owner = match_heads_to_boxes(box_pixels, centres)
    for centre, box_index in zip(centres, owner):
        if box_index is None:
            cv2.drawMarker(
                canvas, centre, LOOSE_COLOUR, cv2.MARKER_TILTED_CROSS, 14, 2,
                cv2.LINE_AA,
            )
        else:
            cv2.circle(canvas, centre, 7, MATCHED_COLOUR, 2, cv2.LINE_AA)
    taken = {index for index in owner if index is not None}
    for index, (x1, y1, x2, _y2) in enumerate(box_pixels):
        if index not in taken:
            cv2.line(
                canvas,
                (int(round(x1)), int(round(y1))),
                (int(round(x2)), int(round(y1))),
                LOOSE_COLOUR,
                2,
                cv2.LINE_AA,
            )
    matched = len(taken)
    cv2.putText(
        canvas,
        f"heads {matched} matched / {len(centres) - matched} loose | "
        f"boxes without a head {len(box_pixels) - matched}",
        (10, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect and track volleyball players and the ball with RAVEL-VB."
    )
    parser.add_argument("video", help="input video")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument(
        "--output",
        help="write predictions to this JSON file or directory",
    )
    parser.add_argument("--output-video", help="write annotated video to this file")
    parser.add_argument(
        "--show",
        action="store_true",
        help="show annotated video in an OpenCV window (Esc or q to stop)",
    )
    parser.add_argument("--stride", type=int, default=9)
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help=(
            "подавать в сеть каждый N-й кадр: 2 -- каждый второй, 3 -- каждый "
            "третий. Клип из тех же девяти слотов охватывает вдвое-втрое "
            "больше времени, а декодирование и проходы сети во столько же раз "
            "дешевле. Предсказания появляются только у показанных кадров; на "
            "видео пропущенный кадр донашивает рамки предыдущего показанного"
        ),
    )
    parser.add_argument("--score-threshold", type=float, default=0.35)
    parser.add_argument(
        "--ball-threshold",
        type=float,
        default=0.35,
        help=(
            "порог мяча, отдельный от --score-threshold: мяч -- своя голова со своей "
            "шкалой уверенности, и порог игроков ему не подходит"
        ),
    )
    parser.add_argument(
        "--head-threshold",
        type=float,
        default=HEAD_THRESHOLD,
        help=(
            "порог пиков карты голов (body_point_logits); у моделей без неё "
            "ни на что не влияет"
        ),
    )
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--close-threshold", type=float, default=0.20)
    parser.add_argument("--hysteresis-frames", type=int, default=2)
    parser.add_argument("--no-player-hysteresis", action="store_true")
    parser.add_argument(
        "--include-features",
        action="store_true",
        help="include head/foot points and query vectors in JSON",
    )
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("CPU", "GPU", "AUTO"),
        default="CPU",
        help="OpenVINO device used for inference",
    )
    parser.add_argument(
        "--performance-hint",
        choices=("LATENCY", "THROUGHPUT"),
        default="LATENCY",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    if args.frame_step < 1:
        raise ValueError("--frame-step must be >= 1")
    if not 0 <= args.score_threshold <= 1:
        raise ValueError("--score-threshold must be in [0, 1]")
    if not 0 <= args.ball_threshold <= 1:
        raise ValueError("--ball-threshold must be in [0, 1]")
    if not 0 <= args.head_threshold <= 1:
        raise ValueError("--head-threshold must be in [0, 1]")
    if not 0 <= args.close_threshold <= 1:
        raise ValueError("--close-threshold must be in [0, 1]")
    if (
        not args.no_player_hysteresis
        and args.close_threshold > args.score_threshold
    ):
        raise ValueError("--close-threshold must not exceed --score-threshold")
    if args.hysteresis_frames < 0 or args.warmup_runs < 0 or args.num_threads < 0:
        raise ValueError("frame counts and thread count must be non-negative")

    model_path = Path(args.model).expanduser().resolve()
    video_path = Path(args.video).expanduser().resolve()
    output_path = None
    if args.output:
        output_arg = args.output
        output_path = Path(output_arg).expanduser().resolve()
        # Accept the directory form used by the surrounding video pipeline,
        # e.g. ``--output ../uploads/mix/``.
        if output_arg.endswith(("/", "\\")) or output_path.is_dir():
            output_path = output_path / f"{video_path.stem}_predictions.json"
    output_video_path = (
        Path(args.output_video).expanduser().resolve()
        if args.output_video
        else None
    )
    if not video_path.is_file():
        raise FileNotFoundError(f"video not found: {video_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"OpenVINO model not found: {model_path}")
    model = RavelVBOpenVINO(
        model_path, args.num_threads, args.performance_hint, args.device
    )
    config = model.config
    clip_length = int(config["clip_length"])
    if args.stride > clip_length:
        raise ValueError("--stride must not exceed the model clip length")
    warmup = np.zeros(
        (
            1,
            clip_length,
            int(config["input_channels"]),
            int(config["image_height"]),
            int(config["image_width"]),
        ),
        dtype=np.float32,
    )
    for _ in range(args.warmup_runs):
        model.infer(warmup)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames: deque[np.ndarray] = deque(maxlen=clip_length)
    indices: deque[int] = deque(maxlen=clip_length)
    raw_by_frame: dict[int, list[dict]] = defaultdict(list)
    predictions_by_frame: dict[int, list[dict]] = defaultdict(list)
    predictions: list[dict] = []
    hysteresis = (
        None
        if args.no_player_hysteresis
        else PlayerHysteresis(
            args.score_threshold,
            args.close_threshold,
            args.hysteresis_frames,
        )
    )
    show_frames: dict[int, np.ndarray] = {}
    show_active = args.show
    show_window_open = False
    next_finalize_frame = 0
    frame_index = 0
    shown = 0
    last_clip_start: int | None = None
    inference_runs = 0
    model_elapsed = 0.0
    candidate_threshold = (
        args.score_threshold
        if args.no_player_hysteresis
        else args.close_threshold
    )

    def infer_clip(
        clip_frames: list[np.ndarray],
        clip_indices: list[int],
        real_frame_count: int,
    ) -> None:
        nonlocal inference_runs, model_elapsed
        clip = np.stack(clip_frames, axis=0)[None]
        outputs, elapsed = model.infer(clip)
        inference_runs += 1
        model_elapsed += elapsed
        decoded = decode_batch(
            outputs,
            candidate_threshold,
            model.ball_width,
            model.ball_height,
            args.include_features,
            ball_threshold=args.ball_threshold,
            head_threshold=args.head_threshold,
        )[0]
        for item in decoded:
            slot = int(item.pop("frame_slot"))
            if slot >= real_frame_count:
                continue
            if (
                item["class_name"] == "ball"
                and float(item["score"]) < args.ball_threshold
            ):
                continue
            item["frame_index"] = clip_indices[slot]
            raw_by_frame[item["frame_index"]].append(item)

    def _show_frame(current_frame: int) -> None:
        """Нарисовать кадр в окне ``--show``, если оно открыто."""
        nonlocal show_active, show_window_open
        show_frame = show_frames.pop(current_frame, None)
        if not show_active or show_frame is None:
            return
        if not show_window_open:
            cv2.namedWindow("RAVEL-VB", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(
                "RAVEL-VB", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
            show_window_open = True
        rendered = _draw_predictions(
            show_frame, current_frame, fps, records_for(current_frame)
        )
        cv2.imshow("RAVEL-VB", rendered)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            show_active = False
            show_window_open = False
            show_frames.clear()
            cv2.destroyAllWindows()

    carried: list[dict] = []

    def records_for(frame: int) -> list[dict]:
        """Что рисовать на кадре.

        У показанного сети кадра -- его собственные записи, у пропущенного --
        записи последнего показанного, иначе при ``--frame-step`` рамки мигали
        бы через кадр. В json донесённые записи не попадают: там есть только
        то, что сеть действительно видела.
        """
        nonlocal carried
        if frame % args.frame_step == 0:
            carried = predictions_by_frame.get(frame, [])
        return carried

    def finalize_frames(end_frame: int) -> None:
        """Finalize frames that cannot be affected by a later inference clip."""
        nonlocal next_finalize_frame, show_active, show_window_open
        while next_finalize_frame < end_frame:
            current_frame = next_finalize_frame
            if current_frame % args.frame_step:
                # Кадр сети не показывали: пустой список здесь состарил бы
                # треки гистерезиса так, будто игроки исчезли.
                _show_frame(current_frame)
                next_finalize_frame += 1
                continue
            merged = merge_frame_predictions(raw_by_frame.pop(current_frame, []))
            if hysteresis is not None:
                filtered = hysteresis.update(current_frame, merged)
            else:
                filtered = [
                    item
                    for item in merged
                    if item["class_name"] != "player"
                    or float(item["score"]) >= args.score_threshold
                ]
            for filtered_item in filtered:
                item = dict(filtered_item)
                x1, y1, x2, y2 = item.pop("bbox_xyxy_norm")
                item["frame_index"] = current_frame
                item["time_sec"] = round(current_frame / fps, 4)
                item["bbox_xyxy"] = [
                    round(x1 * width, 2),
                    round(y1 * height, 2),
                    round(x2 * width, 2),
                    round(y2 * height, 2),
                ]
                predictions.append(item)
                predictions_by_frame[current_frame].append(item)

            _show_frame(current_frame)
            next_finalize_frame += 1

    pipeline_started = time.perf_counter()
    with tqdm(
        total=reported_total or None, desc="RAVEL-VB", unit="frame"
    ) as progress:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            if show_active:
                show_frames[frame_index] = bgr.copy()
            # ``--frame-step`` выбрасывает кадры до сборки клипа: сеть видит те
            # же девять слотов, растянутые на большее время, а у невыбранного
            # кадра предсказаний нет вовсе.
            if frame_index % args.frame_step:
                frame_index += 1
                progress.update(1)
                continue
            frames.append(model.preprocess(bgr))
            indices.append(frame_index)
            shown += 1
            if len(frames) == clip_length and (shown - clip_length) % args.stride == 0:
                current_indices = list(indices)
                infer_clip(list(frames), current_indices, clip_length)
                last_clip_start = current_indices[0]
                finalize_frames(last_clip_start + args.stride * args.frame_step)
            frame_index += 1
            progress.update(1)
    total = frame_index
    next_clip_start = (
        0
        if last_clip_start is None
        else last_clip_start + args.stride * args.frame_step
    )
    if next_clip_start < total and indices:
        tail = [
            (index, frame)
            for index, frame in zip(indices, frames)
            if index >= next_clip_start
        ]
        if tail:
            tail_indices = [item[0] for item in tail]
            tail_frames = [item[1] for item in tail]
            real_frame_count = len(tail_frames)
            while len(tail_frames) < clip_length:
                tail_frames.append(tail_frames[-1])
                tail_indices.append(tail_indices[-1])
            infer_clip(tail_frames, tail_indices, real_frame_count)
    capture.release()
    finalize_frames(total)
    if args.show:
        cv2.destroyAllWindows()
    pipeline_elapsed = time.perf_counter() - pipeline_started
    benchmark = {
        "processed_frames": total,
        "inference_runs": inference_runs,
        "model_seconds": round(model_elapsed, 6),
        "model_ms_per_clip": round(
            1000 * model_elapsed / max(inference_runs, 1), 3
        ),
        "effective_model_frames_per_second": round(
            inference_runs * args.stride / max(model_elapsed, 1e-9), 3
        ),
        "pipeline_seconds": round(pipeline_elapsed, 6),
        "pipeline_frames_per_second": round(
            total / max(pipeline_elapsed, 1e-9), 3
        ),
        "device": model.device,
    }

    if output_video_path is not None:
        render_capture = cv2.VideoCapture(str(video_path))
        if not render_capture.isOpened():
            raise ValueError("cannot open the input video for rendering")
        writer = None
        if output_video_path is not None:
            output_video_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(output_video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )
            if not writer.isOpened():
                render_capture.release()
                raise ValueError("cannot open the output video for rendering")
        try:
            render_index = 0
            while True:
                ok, bgr = render_capture.read()
                if not ok:
                    break
                rendered = _draw_predictions(
                    bgr, render_index, fps, records_for(render_index)
                )
                if writer is not None:
                    writer.write(rendered)
                render_index += 1
        finally:
            if writer is not None:
                writer.release()
            render_capture.release()

    payload = {
        "format": "ravel-vb-predictions-v1",
        "model": str(model_path),
        "video": str(video_path),
        "source_size": {"width": width, "height": height},
        "fps": fps,
        "clip_length": clip_length,
        "stride": args.stride,
        "frame_step": args.frame_step,
        "score_threshold": args.score_threshold,
        "ball_threshold": args.ball_threshold,
        "head_threshold": args.head_threshold,
        "close_threshold": (
            None if args.no_player_hysteresis else args.close_threshold
        ),
        "hysteresis_frames": (
            0 if args.no_player_hysteresis else args.hysteresis_frames
        ),
        "benchmark": benchmark,
        "predictions": predictions,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "output": str(output_path) if output_path else None,
                "output_video": (
                    str(output_video_path) if output_video_path else None
                ),
                "predictions": len(predictions),
                "benchmark": benchmark,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
