import argparse
import logging
import os
import queue
import threading
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
from tqdm import tqdm

from constants import (
    DEFAULT_HEATMAP_THRESHOLD,
    DEFAULT_INPUT_HEIGHT,
    DEFAULT_INPUT_WIDTH,
)
from models import BallTrack

os.environ["LD_LIBRARY_PATH"] = "./.venv/lib/python3.12/site-packages/nvidia/cublas/lib:" + os.environ.get(
    "LD_LIBRARY_PATH", ""
)
ort.preload_dlls()

LOG = logging.getLogger(__name__)

GRID_INPUT_WIDTH = 768
GRID_INPUT_HEIGHT = 432
GRID_COLS = 48
GRID_ROWS = 27
BALL_SIZE_HISTORY = 12
BALL_RAW_SIZE_HISTORY = 5
BALL_TREND_FRAMES = 3
BALL_RADIUS_MIN = 3
BALL_RADIUS_MAX = 40
BALL_ROI_HALF_SIZE = 48


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Volleyball ball detection and tracking with ONNX"
    )
    parser.add_argument(
        "--video_path", type=str, required=True, help="Path to input video file"
    )
    parser.add_argument(
        "--track_length", type=int, default=8, help="Length of the ball track"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save output video and CSV",
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to ONNX model file"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="Enable visualization on display",
    )
    parser.add_argument(
        "--only_csv",
        action="store_true",
        default=False,
        help="Save only CSV, skip video output",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=DEFAULT_HEATMAP_THRESHOLD,
        help="Heatmap confidence threshold",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def infer_model_params(model_path: str) -> dict:
    model_name = Path(model_path).name.lower()
    if "vballnetgrid" in model_name:
        return {
            "family": "grid",
            "seq": 9,
            "input_seq": 9,
            "input_width": GRID_INPUT_WIDTH,
            "input_height": GRID_INPUT_HEIGHT,
            "grid_cols": GRID_COLS,
            "grid_rows": GRID_ROWS,
        }
    return {
        "family": "heatmap",
        "seq": 15 if "seq15" in model_name else 9 if "seq9" in model_name else 3,
        "input_seq": 15 if "seq15" in model_name else 9,
        "input_width": DEFAULT_INPUT_WIDTH,
        "input_height": DEFAULT_INPUT_HEIGHT,
        "grid_cols": None,
        "grid_rows": None,
    }


def resolve_dim(dim, fallback: int) -> int:
    if isinstance(dim, int):
        return dim
    return fallback


def load_onnx_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    model_params = infer_model_params(model_path)
    session = ort.InferenceSession(
        model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]
    input_shape = session.get_inputs()[0].shape
    output_shape = session.get_outputs()[0].shape

    input_seq = resolve_dim(input_shape[1], model_params["input_seq"])
    if model_params["family"] == "grid":
        output_channels = resolve_dim(output_shape[1], input_seq * 3)
        out_dim = output_channels // 3
    else:
        out_dim = resolve_dim(output_shape[1], model_params["seq"])

    model_params["input_seq"] = input_seq
    model_params["seq"] = out_dim

    has_gru = "h0" in input_names
    h0_shape = None
    if has_gru:
        for inp in session.get_inputs():
            if inp.name == "h0":
                h0_shape = inp.shape
                break
        if h0_shape is None:
            raise ValueError("Could not determine h0 shape for GRU model.")
        resolved_shape = []
        for dim in h0_shape:
            if isinstance(dim, str) or dim is None:
                if dim in ["batch", "batch_size", None]:
                    resolved_shape.append(1)
                elif "hidden" in str(dim).lower():
                    resolved_shape.append(512)
                else:
                    raise ValueError(
                        f"Unknown symbolic dimension '{dim}' in h0_shape: {h0_shape}"
                    )
            else:
                resolved_shape.append(dim)
        h0_shape = tuple(resolved_shape)

    batch_size = model_params["input_seq"]

    LOG.info("Model loaded: %s", model_path)
    LOG.info(
        "Family: %s | GRU: %s | Input sequence: %s | Output sequence: %s | h0 shape: %s",
        model_params["family"],
        has_gru,
        batch_size,
        out_dim,
        h0_shape if has_gru else "N/A",
    )
    return (
        session,
        has_gru,
        out_dim,
        h0_shape,
        batch_size,
        input_names,
        output_names,
        model_params,
    )


def initialize_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, frame_width, frame_height, fps, total_frames


def setup_output_writer(
    video_basename, output_dir, frame_width, frame_height, fps, only_csv
):
    if output_dir is None or only_csv:
        return None, None
    video_dir = os.path.join(output_dir, video_basename)
    os.makedirs(video_dir, exist_ok=True)
    output_path = os.path.join(video_dir, "predict.mp4")
    out_writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height)
    )
    return out_writer, output_path


def setup_csv_file(video_basename, output_dir):
    if output_dir is None:
        return None
    video_dir = os.path.join(output_dir, video_basename)
    os.makedirs(video_dir, exist_ok=True)
    csv_path = os.path.join(video_dir, "ball.csv")
    pd.DataFrame(columns=["Frame", "Visibility", "X", "Y", "Radius"]).to_csv(
        csv_path, index=False
    )
    return csv_path


def append_to_csv(result, csv_path):
    if csv_path is None:
        return
    pd.DataFrame([result]).to_csv(csv_path, mode="a", header=False, index=False)


def preprocess_frames(
    frames, input_height=DEFAULT_INPUT_HEIGHT, input_width=DEFAULT_INPUT_WIDTH
):
    processed = []
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (input_width, input_height))
        frame = frame.astype(np.float32) / 255.0
        processed.append(frame)
    return processed


def postprocess_heatmap_output(
    output,
    threshold=DEFAULT_HEATMAP_THRESHOLD,
    input_height=DEFAULT_INPUT_HEIGHT,
    input_width=DEFAULT_INPUT_WIDTH,
    out_dim=9,
):
    results = []
    for frame_idx in range(out_dim):
        heatmap = output[0, frame_idx, :, :]
        _, binary = cv2.threshold(heatmap, threshold, 1.0, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            (binary * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            moments = cv2.moments(largest_contour)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                results.append((1, cx, cy))
            else:
                results.append((0, 0, 0))
        else:
            results.append((0, 0, 0))
    return results


def postprocess_grid_output(
    output, threshold, seq, input_height, input_width, grid_rows, grid_cols
):
    output = output[0].reshape(seq, 3, grid_rows, grid_cols)
    results = []
    for frame_idx in range(seq):
        conf = output[frame_idx, 0]
        x_offset = output[frame_idx, 1]
        y_offset = output[frame_idx, 2]
        max_index = int(np.argmax(conf))
        row = max_index // grid_cols
        col = max_index % grid_cols
        conf_score = float(conf[row, col])
        if conf_score < threshold:
            results.append((0, 0, 0))
            continue
        x = (col + float(x_offset[row, col])) * (input_width / grid_cols)
        y = (row + float(y_offset[row, col])) * (input_height / grid_rows)
        x = int(np.clip(x, 0, input_width - 1))
        y = int(np.clip(y, 0, input_height - 1))
        results.append((1, x, y))
    return results


def decode_predictions(output, model_params, threshold):
    if model_params["family"] == "grid":
        return postprocess_grid_output(
            output=output,
            threshold=threshold,
            seq=model_params["seq"],
            input_height=model_params["input_height"],
            input_width=model_params["input_width"],
            grid_rows=model_params["grid_rows"],
            grid_cols=model_params["grid_cols"],
        )
    return postprocess_heatmap_output(
        output=output,
        threshold=threshold,
        input_height=model_params["input_height"],
        input_width=model_params["input_width"],
        out_dim=model_params["seq"],
    )


def draw_track(
    frame,
    points: List[Tuple[int, int]],
    current_color=(0, 0, 255),
    history_color=(255, 0, 0),
):
    for point in points[:-1]:
        cv2.circle(frame, point, 5, history_color, -1)
    if points:
        cv2.circle(frame, points[-1], 5, current_color, -1)
    return frame


def build_motion_mask(prev_gray, gray):
    diff = cv2.absdiff(prev_gray, gray)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    _, motion_mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    motion_mask = cv2.morphologyEx(
        motion_mask, cv2.MORPH_OPEN, kernel, iterations=1
    )
    motion_mask = cv2.dilate(motion_mask, kernel, iterations=2)
    return motion_mask


def contour_narrow_radius(contour):
    if len(contour) < 5:
        (_, _), radius = cv2.minEnclosingCircle(contour)
        return float(radius)

    (_, _), (width, height), _ = cv2.minAreaRect(contour)
    narrow_diameter = min(width, height)
    if narrow_diameter <= 0:
        (_, _), radius = cv2.minEnclosingCircle(contour)
        return float(radius)
    return float(narrow_diameter) / 2.0


def fallback_radius(size_state):
    smoothed_radius = size_state["smoothed_radius"]
    if smoothed_radius > 0:
        return int(round(smoothed_radius))
    filtered_history = size_state["filtered_history"]
    if not filtered_history:
        return 0
    return int(round(float(np.median(filtered_history))))


def filter_ball_radius(radius, size_state):
    if radius <= 0:
        return fallback_radius(size_state)

    raw_history = size_state["raw_history"]
    filtered_history = size_state["filtered_history"]
    raw_history.append(radius)

    if not filtered_history:
        filtered_history.append(radius)
        size_state["smoothed_radius"] = float(radius)
        return radius

    baseline = size_state["smoothed_radius"]
    if baseline <= 0:
        baseline = float(np.median(filtered_history))

    trend_window = list(raw_history)[-BALL_TREND_FRAMES:]
    trend_confirmed = False
    if len(trend_window) == BALL_TREND_FRAMES:
        upper_shift = [value > baseline for value in trend_window]
        lower_shift = [value < baseline for value in trend_window]
        trend_confirmed = all(upper_shift) or all(lower_shift)

    target_radius = float(radius)
    if trend_confirmed:
        target_radius = float(np.median(trend_window))
    else:
        max_deviation = max(3.0, baseline * 0.55)
        target_radius = float(
            np.clip(radius, baseline - max_deviation, baseline + max_deviation)
        )

    alpha = 0.6 if trend_confirmed else 0.3
    smoothed_radius = baseline * (1.0 - alpha) + target_radius * alpha
    smoothed_radius = float(np.clip(smoothed_radius, BALL_RADIUS_MIN, BALL_RADIUS_MAX))

    accepted_radius = int(round(smoothed_radius))
    filtered_history.append(accepted_radius)
    size_state["smoothed_radius"] = smoothed_radius
    return accepted_radius


def estimate_ball_radius(prev_gray, gray, x_orig, y_orig, size_state):
    if prev_gray is None or x_orig < 0 or y_orig < 0:
        return 0, None

    motion_mask = build_motion_mask(prev_gray, gray)
    x1 = max(0, x_orig - BALL_ROI_HALF_SIZE)
    y1 = max(0, y_orig - BALL_ROI_HALF_SIZE)
    x2 = min(gray.shape[1], x_orig + BALL_ROI_HALF_SIZE)
    y2 = min(gray.shape[0], y_orig + BALL_ROI_HALF_SIZE)
    roi = motion_mask[y1:y2, x1:x2]
    if roi.size == 0:
        return fallback_radius(size_state), None

    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return fallback_radius(size_state), None

    center = np.array([x_orig - x1, y_orig - y1], dtype=np.float32)
    best_contour = None
    best_score = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 8:
            continue
        (cx, cy), _ = cv2.minEnclosingCircle(contour)
        radius = contour_narrow_radius(contour)
        if radius < BALL_RADIUS_MIN or radius > BALL_RADIUS_MAX:
            continue
        distance = np.linalg.norm(np.array([cx, cy], dtype=np.float32) - center)
        score = distance - area * 0.02
        if best_score is None or score < best_score:
            best_score = score
            best_contour = contour

    if best_contour is None:
        return fallback_radius(size_state), None

    radius = contour_narrow_radius(best_contour)
    filtered_radius = filter_ball_radius(int(round(radius)), size_state)
    contour_global = best_contour + np.array([[[x1, y1]]], dtype=np.int32)
    return filtered_radius, contour_global


def render_prediction(frame, points, visibility, x_orig, y_orig, radius, contour):
    vis_frame = draw_track(frame.copy(), points)
    if visibility:
        if contour is not None:
            cv2.drawContours(vis_frame, [contour], -1, (0, 255, 255), 1)
        draw_radius = radius if radius > 0 else 8
        cv2.circle(vis_frame, (x_orig, y_orig), draw_radius, (0, 255, 0), 2)
        cv2.putText(
            vis_frame,
            f"R:{radius}" if radius > 0 else "R:n/a",
            (x_orig + draw_radius + 4, y_orig - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return vis_frame


def run_inference(session, input_tensor, has_gru, h0, input_names, output_names):
    inputs = {input_names[0]: input_tensor}
    if has_gru:
        if len(input_names) < 2:
            raise ValueError("GRU model expects at least 2 inputs: images and h0")
        inputs[input_names[1]] = h0

    outputs = session.run(output_names, inputs)

    heatmaps = outputs[0]
    new_h0 = None
    if has_gru:
        if len(outputs) < 2:
            raise ValueError("GRU model should output at least 2 values: heatmaps and hn")
        new_h0 = outputs[1]
    return heatmaps, new_h0


class BallTrackState:
    def __init__(self, maxlen: int, max_missing: int) -> None:
        self._track = BallTrack(maxlen)
        self._missing = 0
        self._max_missing = max_missing

    def update(self, point: Optional[Tuple[int, int]]) -> None:
        self._track.update(point)
        if point is None:
            self._missing += 1
        else:
            self._missing = 0

    def is_lost(self) -> bool:
        return self._missing >= self._max_missing

    def reset(self) -> None:
        self._track.reset()
        self._missing = 0

    def points(self) -> List[Tuple[int, int]]:
        return list(self._track.points())


def frame_reader(cap, frame_queue, batch_size, stop_event, error_queue):
    try:
        while not stop_event.is_set():
            frames = []
            for _ in range(batch_size):
                if stop_event.is_set():
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            if frames:
                frame_queue.put(frames, timeout=1.0)
            else:
                frame_queue.put(None, timeout=1.0)
                break
    except Exception as exc:  # pragma: no cover - thread safety
        error_queue.put(exc)
        stop_event.set()


def initialize_visualization(enabled: bool) -> bool:
    if not enabled:
        return False
    try:
        cv2.namedWindow("Ball Tracking", cv2.WINDOW_NORMAL)
        return True
    except Exception as exc:
        LOG.warning("Visualization disabled: failed to initialize OpenCV window: %s", exc)
        return False


def show_visualization(enabled: bool, frame) -> Tuple[bool, bool]:
    if not enabled:
        return False, False
    try:
        cv2.imshow("Ball Tracking", frame)
        should_exit = cv2.waitKey(1) & 0xFF == ord("q")
        return True, should_exit
    except Exception as exc:
        LOG.warning("Visualization disabled while rendering frame: %s", exc)
        return False, False


def main():
    args = parse_args()
    setup_logging(args.verbose)

    (
        model_session,
        has_gru,
        out_dim,
        h0_shape,
        batch_size,
        input_names,
        output_names,
        model_params,
    ) = load_onnx_model(args.model_path)
    input_width = model_params["input_width"]
    input_height = model_params["input_height"]

    cap, frame_width, frame_height, fps, total_frames = initialize_video(
        args.video_path
    )
    video_basename = os.path.splitext(os.path.basename(args.video_path))[0]
    out_writer, _ = setup_output_writer(
        video_basename, args.output_dir, frame_width, frame_height, fps, args.only_csv
    )
    csv_path = setup_csv_file(video_basename, args.output_dir)

    frame_buffer = []
    track_state = BallTrackState(maxlen=args.track_length, max_missing=args.track_length)
    frame_index = 0
    frame_queue = queue.Queue(maxsize=2)
    error_queue = queue.Queue()
    stop_event = threading.Event()
    visualization_enabled = initialize_visualization(args.visualize)
    size_state = {
        "filtered_history": deque(maxlen=BALL_SIZE_HISTORY),
        "raw_history": deque(maxlen=BALL_RAW_SIZE_HISTORY),
        "smoothed_radius": 0.0,
    }
    prev_gray = None

    h0 = np.zeros(h0_shape, dtype=np.float32) if has_gru and h0_shape else None

    reader_thread = threading.Thread(
        target=frame_reader,
        args=(cap, frame_queue, batch_size, stop_event, error_queue),
        daemon=True,
    )
    reader_thread.start()

    pbar = tqdm(total=total_frames, desc="Processing video", unit="frame")

    exit_flag = False
    try:
        while not stop_event.is_set():
            if not error_queue.empty():
                raise error_queue.get()

            try:
                frames = frame_queue.get(timeout=0.5)
            except queue.Empty:
                if not reader_thread.is_alive():
                    break
                continue

            if frames is None:
                break

            processed_frames = preprocess_frames(frames, input_height, input_width)

            while len(frame_buffer) < batch_size:
                frame_buffer.append(
                    processed_frames[0]
                    if processed_frames
                    else np.zeros((input_height, input_width), dtype=np.float32)
                )

            for pf in processed_frames:
                frame_buffer.append(pf)
            frame_buffer = frame_buffer[-batch_size:]

            input_tensor = np.stack(frame_buffer, axis=2)
            input_tensor = np.expand_dims(input_tensor, axis=0)
            input_tensor = np.transpose(input_tensor, (0, 3, 1, 2))

            output, new_h0 = run_inference(
                model_session, input_tensor, has_gru, h0, input_names, output_names
            )
            if has_gru and new_h0 is not None:
                h0 = new_h0

            predictions = decode_predictions(output, model_params, args.confidence_threshold)

            for i, (visibility, x, y) in enumerate(predictions[: len(frames)]):
                frame_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
                x_orig = x * frame_width / input_width if visibility else -1
                y_orig = y * frame_height / input_height if visibility else -1

                if visibility:
                    point = (int(x_orig), int(y_orig))
                    track_state.update(point)
                    radius, contour = estimate_ball_radius(
                        prev_gray, frame_gray, point[0], point[1], size_state
                    )
                else:
                    track_state.update(None)
                    radius, contour = 0, None

                if track_state.is_lost():
                    track_state.reset()

                result = {
                    "Frame": frame_index + i,
                    "Visibility": visibility,
                    "X": int(x_orig),
                    "Y": int(y_orig),
                    "Radius": radius,
                }
                append_to_csv(result, csv_path)

                if visualization_enabled or out_writer is not None:
                    vis_frame = render_prediction(
                        frames[i],
                        track_state.points(),
                        visibility,
                        int(x_orig),
                        int(y_orig),
                        radius,
                        contour,
                    )
                    if visualization_enabled:
                        visualization_enabled, should_exit = show_visualization(
                            visualization_enabled, vis_frame
                        )
                        if should_exit:
                            exit_flag = True
                            stop_event.set()
                            break
                    if out_writer is not None:
                        out_writer.write(vis_frame)
                prev_gray = frame_gray

            if exit_flag:
                break

            pbar.update(len(frames))
            frame_index += len(frames)
    finally:
        stop_event.set()
        reader_thread.join(timeout=2.0)
        pbar.close()
        cap.release()
        if out_writer is not None:
            out_writer.release()
        if visualization_enabled:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
