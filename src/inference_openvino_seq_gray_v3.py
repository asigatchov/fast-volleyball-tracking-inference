#!/usr/bin/env python3
import argparse
import os
import time
from collections import deque

import cv2
import ffmpegcv
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from openvino import Core
except ImportError:
    from openvino.runtime import Core


def parse_args():
    parser = argparse.ArgumentParser(
        description="Volleyball ball detection with OpenVINO + ffmpegcv Y-plane input"
    )
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--model_xml", type=str, required=True, help="Path to .xml")
    parser.add_argument("--track_length", type=int, default=8, help="Track length")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--visualize", action="store_true", help="Show visualization")
    parser.add_argument("--only_csv", action="store_true", help="Save only CSV")
    parser.add_argument("--device", type=str, default="GPU", help="CPU, GPU, AUTO")
    return parser.parse_args()


def load_model(model_xml, device="CPU"):
    model_bin = model_xml.replace(".xml", ".bin")
    if not os.path.exists(model_xml):
        raise FileNotFoundError(f"XML не найден: {model_xml}")
    if not os.path.exists(model_bin):
        raise FileNotFoundError(f"BIN не найден: {model_bin}")

    core = Core()
    model = core.read_model(model=model_xml)
    input_layer = model.input(0)
    pshape = input_layer.partial_shape

    print(f"Исходная форма входа: {pshape}")
    if pshape.is_dynamic:
        print("Динамическая форма — фиксируем на [1,9,288,512]")
        model.reshape({input_layer.any_name: [1, 9, 288, 512]})

    compiled_model = core.compile_model(model=model, device_name=device)
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    input_shape = input_layer.shape
    out_dim = input_shape[1]

    print(f"Модель загружена на: {device}")
    print(f"  Вход: {input_layer.any_name} {input_shape}")
    print(f"  Выход: {output_layer.any_name} {output_layer.shape}")
    print(f"  out_dim = {out_dim}")

    return compiled_model, input_layer, output_layer, out_dim, input_shape


def open_capture(video_path):
    try:
        cap = ffmpegcv.VideoCaptureNV(video_path, pix_fmt="yuv420p")
        backend = "ffmpegcv.VideoCaptureNV"
    except Exception as exc:
        cap = ffmpegcv.VideoCapture(video_path, pix_fmt="yuv420p")
        backend = f"ffmpegcv.VideoCapture ({type(exc).__name__}: {exc})"
    return cap, backend


def initialize_video(video_path):
    cap, backend = open_capture(video_path)
    w = int(cap.width)
    h = int(cap.height)
    fps = float(cap.fps)
    total = int(getattr(cap, "count", 0) or 0)
    print(f"Видео открыто через: {backend}")
    print(f"  Размер: {w}x{h}, fps={fps:.3f}, frames={total}")
    return cap, w, h, fps, total, backend


def setup_output_writer(basename, out_dir, w, h, fps, only_csv):
    if out_dir is None or only_csv:
        return None, None
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{basename}_predict.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
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


def preprocess_y_frames(frames, src_h, dst_h=288, dst_w=512):
    processed = []
    display_frames = []
    for frame in frames:
        y_plane = frame[:src_h, :]
        resized = cv2.resize(y_plane, (dst_w, dst_h))
        normalized = resized.astype(np.float32) / 255.0
        processed.append(normalized)
        display_frames.append(y_plane)
    return processed, display_frames


def postprocess_output(output, threshold=0.5, out_dim=9):
    results = []
    for i in range(out_dim):
        heatmap = output[i]
        _, binary = cv2.threshold(heatmap, threshold, 1.0, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            (binary * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            contour = max(contours, key=cv2.contourArea)
            moments = cv2.moments(contour)
            if moments["m00"] > 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                results.append((1, cx, cy))
                continue
        results.append((0, 0, 0))
    return results


def draw_track(frame, track, cur_color=(0, 0, 255), hist_color=(255, 0, 0)):
    for point in list(track)[:-1]:
        if point:
            cv2.circle(frame, point, 5, hist_color, -1)
    if track and track[-1]:
        cv2.circle(frame, track[-1], 5, cur_color, -1)
    return frame


def initialize_visualization(enabled):
    if not enabled:
        return False
    try:
        win_name = "Ball Tracking"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 1920, 1080)
        return True
    except Exception as exc:
        print(f"Visualization disabled: failed to initialize OpenCV window: {exc}")
        return False


def show_visualization(enabled, frame):
    if not enabled:
        return False, False
    try:
        cv2.imshow("Ball Tracking", frame)
        should_exit = cv2.waitKey(1) & 0xFF == ord("q")
        return True, should_exit
    except Exception as exc:
        print(f"Visualization disabled while rendering frame: {exc}")
        return False, False


def read_frames(cap, q, max_n=9):
    frames = []
    while len(frames) < max_n:
        ret, frame = safe_read(cap)
        if not ret:
            break
        if frame is not None:
            frames.append(frame)
    q.put(frames if frames else None)


def build_input_tensor(buffer):
    stacked = np.stack(buffer, axis=2)
    return np.expand_dims(stacked, axis=0).transpose(0, 3, 1, 2)


def safe_read(cap):
    try:
        return cap.read()
    except ValueError as exc:
        if "closed file" in str(exc):
            return False, None
        raise


def main():
    args = parse_args()
    in_w, in_h = 512, 288
    batch_size = 9
    run_start = time.perf_counter()

    model_start = time.perf_counter()
    compiled_model, _, output_layer, out_dim, _ = load_model(args.model_xml, device=args.device)
    model_load_s = time.perf_counter() - model_start

    video_start = time.perf_counter()
    cap, fw, fh, fps, total, _ = initialize_video(args.video_path)
    video_open_s = time.perf_counter() - video_start

    base = os.path.splitext(os.path.basename(args.video_path))[0]
    writer, _ = setup_output_writer(base, args.output_dir, fw, fh, fps, args.only_csv)
    csv_path = setup_csv_file(base, args.output_dir)

    buffer = deque(maxlen=batch_size)
    track = deque(maxlen=args.track_length)
    frame_idx = 0
    processed_frames = 0
    preprocess_s = 0.0
    infer_s = 0.0
    postprocess_s = 0.0
    first_result_s = None
    visualization_enabled = initialize_visualization(args.visualize)

    pbar = tqdm(total=total or None, desc="Обработка", unit="кадр")
    exit_flag = False

    while True:
        batch = []
        while len(batch) < batch_size:
            ret, frame = safe_read(cap)
            if not ret:
                break
            if frame is not None:
                batch.append(frame)
        if not batch:
            break

        prep_start = time.perf_counter()
        proc, display_frames = preprocess_y_frames(batch, fh, in_h, in_w)
        preprocess_s += time.perf_counter() - prep_start

        while len(buffer) < batch_size:
            buffer.append(proc[0] if proc else np.zeros((in_h, in_w), np.float32))
        for frame in proc:
            buffer.append(frame)

        input_tensor = build_input_tensor(buffer)

        infer_start = time.perf_counter()
        result = compiled_model(input_tensor)
        output = result[output_layer]
        infer_s += time.perf_counter() - infer_start

        post_start = time.perf_counter()
        preds = postprocess_output(output[0], out_dim=out_dim)
        postprocess_s += time.perf_counter() - post_start

        if first_result_s is None:
            first_result_s = time.perf_counter() - run_start

        for i, (vis, x, y) in enumerate(preds[: len(batch)]):
            x_orig = x * fw / in_w if vis else -1
            y_orig = y * fh / in_h if vis else -1

            if vis:
                track.append((int(x_orig), int(y_orig)))
            elif track:
                track.popleft()

            res = {"Frame": frame_idx + i, "Visibility": vis, "X": int(x_orig), "Y": int(y_orig)}
            append_to_csv(res, csv_path)

            if visualization_enabled or writer:
                vis_frame = cv2.cvtColor(display_frames[i], cv2.COLOR_GRAY2BGR)
                vis_frame = draw_track(vis_frame, track)
                if visualization_enabled:
                    visualization_enabled, should_exit = show_visualization(
                        visualization_enabled, vis_frame
                    )
                    if should_exit:
                        exit_flag = True
                        break
                if writer:
                    writer.write(vis_frame)

        processed_frames += len(batch)
        pbar.update(len(batch))
        frame_idx += len(batch)

        if exit_flag:
            break

    total_s = time.perf_counter() - run_start
    pbar.close()
    cap.release()
    if writer:
        writer.release()
    if visualization_enabled:
        cv2.destroyAllWindows()

    effective_fps = processed_frames / total_s if total_s > 0 else 0.0
    infer_fps = processed_frames / infer_s if infer_s > 0 else 0.0
    print("=== Benchmark ===")
    print(f"startup_to_first_result_s={first_result_s or 0.0:.3f}")
    print(f"model_load_s={model_load_s:.3f}")
    print(f"video_open_s={video_open_s:.3f}")
    print(f"preprocess_s={preprocess_s:.3f}")
    print(f"infer_s={infer_s:.3f}")
    print(f"postprocess_s={postprocess_s:.3f}")
    print(f"total_s={total_s:.3f}")
    print(f"processed_frames={processed_frames}")
    print(f"effective_fps={effective_fps:.3f}")
    print(f"infer_only_fps={infer_fps:.3f}")


if __name__ == "__main__":
    main()
