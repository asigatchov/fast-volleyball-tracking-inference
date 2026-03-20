#!/usr/bin/env python3

import argparse
import os
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from openvino.runtime import Core
from tqdm import tqdm


DEFAULT_INPUT_WIDTH = 512
DEFAULT_INPUT_HEIGHT = 288
DEFAULT_SEQ = 9
GRID_INPUT_WIDTH = 768
GRID_INPUT_HEIGHT = 432
GRID_COLS = 48
GRID_ROWS = 27


def parse_args():
    parser = argparse.ArgumentParser(
        description="Volleyball ball detection with OpenVINO (heatmap and grid grayscale models)"
    )
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--model_xml", type=str, required=True, help="Path to .xml or .onnx model")
    parser.add_argument("--track_length", type=int, default=8, help="Track length")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--visualize", action="store_true", help="Show visualization")
    parser.add_argument("--only_csv", action="store_true", help="Save only CSV")
    parser.add_argument("--device", type=str, default="GPU", help="CPU, GPU, AUTO")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold")
    return parser.parse_args()


def infer_model_params(model_path):
    model_name = Path(model_path).name.lower()
    if "vballnetgrid" in model_name:
        return {
            "family": "grid",
            "seq": 9,
            "grayscale": True,
            "input_width": GRID_INPUT_WIDTH,
            "input_height": GRID_INPUT_HEIGHT,
            "grid_cols": GRID_COLS,
            "grid_rows": GRID_ROWS,
        }
    return {
        "family": "heatmap",
        "seq": DEFAULT_SEQ,
        "grayscale": True,
        "input_width": DEFAULT_INPUT_WIDTH,
        "input_height": DEFAULT_INPUT_HEIGHT,
        "grid_cols": None,
        "grid_rows": None,
    }


def load_model(model_path, device="CPU"):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Модель не найдена: {path}")
    if path.suffix.lower() == ".xml":
        model_bin = path.with_suffix(".bin")
        if not model_bin.exists():
            raise FileNotFoundError(f"BIN не найден: {model_bin}")

    model_params = infer_model_params(path)
    core = Core()
    model = core.read_model(model=str(path))

    input_layer = model.input(0)
    pshape = input_layer.partial_shape
    expected_shape = [
        1,
        model_params["seq"],
        model_params["input_height"],
        model_params["input_width"],
    ]

    print(f"Исходная форма входа: {pshape}")
    if pshape.is_dynamic:
        print(f"Динамическая форма — фиксируем на {expected_shape}")
        model.reshape({input_layer.any_name: expected_shape})

    compiled_model = core.compile_model(model=model, device_name=device)
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    print(f"Модель загружена на: {device}")
    print(f"  Вход: {input_layer.any_name} {input_layer.shape}")
    print(f"  Выход: {output_layer.any_name} {output_layer.shape}")
    print(f"  Семейство: {model_params['family']}")

    return compiled_model, input_layer, output_layer, model_params


def initialize_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Не открыть видео: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, width, height, fps, total


def setup_output_writer(basename, out_dir, width, height, fps, only_csv):
    if out_dir is None or only_csv:
        return None, None
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{basename}_predict.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    return writer, path


def setup_csv_file(basename, out_dir):
    if out_dir is None:
        return None
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{basename}_predict_ball.csv")
    pd.DataFrame(columns=["Frame", "Visibility", "X", "Y"]).to_csv(path, index=False)
    return path


def append_to_csv(result, csv_path):
    if csv_path:
        pd.DataFrame([result]).to_csv(csv_path, mode="a", header=False, index=False)


def preprocess_frames(frames, input_height, input_width):
    processed = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (input_width, input_height), interpolation=cv2.INTER_AREA)
        processed.append(resized.astype(np.float32) / 255.0)
    return processed


def postprocess_heatmap(output, threshold, input_height, input_width, out_dim):
    results = []
    for i in range(out_dim):
        heatmap = output[i]
        _, binary = cv2.threshold(heatmap, threshold, 1.0, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            (binary * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            results.append((0, -1, -1))
            continue
        contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(contour)
        if moments["m00"] <= 0:
            results.append((0, -1, -1))
            continue
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        results.append((1, cx, cy))
    return results


def postprocess_grid(output, threshold, seq, input_height, input_width, grid_rows, grid_cols):
    output = output.reshape(seq, 3, grid_rows, grid_cols)
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
            results.append((0, -1, -1))
            continue
        x = (col + float(x_offset[row, col])) * (input_width / grid_cols)
        y = (row + float(y_offset[row, col])) * (input_height / grid_rows)
        x = int(np.clip(x, 0, input_width - 1))
        y = int(np.clip(y, 0, input_height - 1))
        results.append((1, x, y))
    return results


def decode_predictions(output, model_params, threshold):
    if model_params["family"] == "grid":
        return postprocess_grid(
            output=output,
            threshold=threshold,
            seq=model_params["seq"],
            input_height=model_params["input_height"],
            input_width=model_params["input_width"],
            grid_rows=model_params["grid_rows"],
            grid_cols=model_params["grid_cols"],
        )
    return postprocess_heatmap(
        output=output,
        threshold=threshold,
        input_height=model_params["input_height"],
        input_width=model_params["input_width"],
        out_dim=model_params["seq"],
    )


def draw_track(frame, track, cur_color=(0, 0, 255), hist_color=(255, 0, 0)):
    for point in list(track)[:-1]:
        if point:
            cv2.circle(frame, point, 5, hist_color, -1)
    if track and track[-1]:
        cv2.circle(frame, track[-1], 5, cur_color, -1)
    return frame


def render_prediction(frame, visibility, x_orig, y_orig, writer, visualize, track):
    if visibility:
        track.append((x_orig, y_orig))
    else:
        if track:
            track.popleft()

    if writer or visualize:
        vis_frame = draw_track(frame.copy(), track)
        if visibility:
            cv2.circle(vis_frame, (x_orig, y_orig), 12, (0, 255, 0), 2)
        if writer:
            writer.write(vis_frame)
        if visualize:
            win_name = 'Ball Tracking'
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_name, 1920, 1080)  
            cv2.imshow(win_name, vis_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt


def main():
    args = parse_args()
    compiled_model, _, output_layer, model_params = load_model(args.model_xml, device=args.device)

    cap, frame_width, frame_height, fps, total = initialize_video(args.video_path)
    basename = os.path.splitext(os.path.basename(args.video_path))[0]
    writer, _ = setup_output_writer(
        basename, args.output_dir, frame_width, frame_height, fps, args.only_csv
    )
    csv_path = setup_csv_file(basename, args.output_dir)

    seq = model_params["seq"]
    current_frames = []
    track = deque(maxlen=args.track_length)
    frame_index = 0
    pbar = tqdm(total=total, desc="Обработка", unit="кадр")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_frames.append(frame.copy())
            if len(current_frames) != seq:
                frame_index += 1
                pbar.update(1)
                continue

            processed_frames = preprocess_frames(
                current_frames,
                input_height=model_params["input_height"],
                input_width=model_params["input_width"],
            )
            input_tensor = np.asarray([processed_frames], dtype=np.float32)
            output = compiled_model(input_tensor)[output_layer][0]
            predictions = decode_predictions(output, model_params, args.threshold)

            start_frame_index = frame_index - seq + 1
            for local_index, (frame_item, prediction) in enumerate(
                zip(current_frames, predictions, strict=True)
            ):
                visibility, x_resized, y_resized = prediction
                if visibility:
                    x_orig = int(x_resized * frame_width / model_params["input_width"])
                    y_orig = int(y_resized * frame_height / model_params["input_height"])
                else:
                    x_orig, y_orig = -1, -1

                result = {
                    "Frame": start_frame_index + local_index,
                    "Visibility": visibility,
                    "X": x_orig,
                    "Y": y_orig,
                }
                append_to_csv(result, csv_path)
                render_prediction(
                    frame_item, visibility, x_orig, y_orig, writer, args.visualize, track
                )

            current_frames = []
            frame_index += 1
            pbar.update(1)

        if current_frames:
            start_frame_index = frame_index - len(current_frames)
            for local_index, frame_item in enumerate(current_frames):
                result = {
                    "Frame": start_frame_index + local_index,
                    "Visibility": 0,
                    "X": -1,
                    "Y": -1,
                }
                append_to_csv(result, csv_path)
                render_prediction(frame_item, 0, -1, -1, writer, args.visualize, track)
    finally:
        pbar.close()
        cap.release()
        if writer:
            writer.release()
        if args.visualize:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
