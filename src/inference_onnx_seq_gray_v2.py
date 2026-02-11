import argparse
import logging
import os
import queue
import threading
import time
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


def load_onnx_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    session = ort.InferenceSession(
        model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]

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

    if "seq15" in model_path.lower():
        out_dim = 15
        batch_size = 15
    elif "seq9" in model_path.lower():
        out_dim = 9
        batch_size = 9
    else:
        out_dim = 3
        batch_size = 3

    LOG.info("Model loaded: %s", model_path)
    LOG.info(
        "GRU: %s | Sequence length: %s | Output heatmaps: %s | h0 shape: %s",
        has_gru,
        batch_size,
        out_dim,
        h0_shape if has_gru else "N/A",
    )
    return session, has_gru, out_dim, h0_shape, batch_size, input_names, output_names


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
    pd.DataFrame(columns=["Frame", "Visibility", "X", "Y"]).to_csv(
        csv_path, index=False
    )
    return csv_path


def append_to_csv(result, csv_path):
    if csv_path is None:
        return
    pd.DataFrame([result]).to_csv(csv_path, mode="a", header=False, index=False)


def preprocess_frames(frames, input_height=DEFAULT_INPUT_HEIGHT, input_width=DEFAULT_INPUT_WIDTH):
    processed = []
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (input_width, input_height))
        frame = frame.astype(np.float32) / 255.0
        processed.append(frame)
    return processed


def postprocess_output(
    output, threshold=DEFAULT_HEATMAP_THRESHOLD, input_height=DEFAULT_INPUT_HEIGHT, input_width=DEFAULT_INPUT_WIDTH, out_dim=9
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


def draw_track(
    frame, points: List[Tuple[int, int]], current_color=(0, 0, 255), history_color=(255, 0, 0)
):
    for point in points[:-1]:
        cv2.circle(frame, point, 5, history_color, -1)
    if points:
        cv2.circle(frame, points[-1], 5, current_color, -1)
    return frame


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


def main():
    args = parse_args()
    setup_logging(args.verbose)

    input_width, input_height = DEFAULT_INPUT_WIDTH, DEFAULT_INPUT_HEIGHT

    (
        model_session,
        has_gru,
        out_dim,
        h0_shape,
        batch_size,
        input_names,
        output_names,
    ) = load_onnx_model(args.model_path)

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

            predictions = postprocess_output(
                output,
                threshold=args.confidence_threshold,
                input_height=input_height,
                input_width=input_width,
                out_dim=out_dim,
            )

            for i, (visibility, x, y) in enumerate(predictions[: len(frames)]):
                x_orig = x * frame_width / input_width if visibility else -1
                y_orig = y * frame_height / input_height if visibility else -1

                if visibility:
                    point = (int(x_orig), int(y_orig))
                    track_state.update(point)
                else:
                    track_state.update(None)

                if track_state.is_lost():
                    track_state.reset()

                result = {
                    "Frame": frame_index + i,
                    "Visibility": visibility,
                    "X": int(x_orig),
                    "Y": int(y_orig),
                }
                append_to_csv(result, csv_path)

                if args.visualize or out_writer is not None:
                    vis_frame = frames[i].copy()
                    vis_frame = draw_track(vis_frame, track_state.points())
                    if args.visualize:
                        cv2.namedWindow("Ball Tracking", cv2.WINDOW_NORMAL)
                        cv2.imshow("Ball Tracking", vis_frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            exit_flag = True
                            stop_event.set()
                            break
                    if out_writer is not None:
                        out_writer.write(vis_frame)

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
        if args.visualize:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
