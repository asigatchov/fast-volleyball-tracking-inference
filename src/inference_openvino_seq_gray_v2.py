#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
import pandas as pd
from openvino.runtime import Core
from collections import deque
import os
import time
from tqdm import tqdm
import threading
import queue


def parse_args():
    parser = argparse.ArgumentParser(
        description="Volleyball ball detection with OpenVINO 2025+ (Intel GPU, grayscale)"
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

    # === КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: обработка динамических входов ===
    input_layer = model.input(0)
    pshape = input_layer.partial_shape

    print(f"Исходная форма входа: {pshape}")

    if pshape.is_dynamic:
        print("Динамическая форма — фиксируем на [1,9,288,512]")
        model.reshape({input_layer.any_name: [1, 9, 288, 512]})

    # Теперь компилируем
    compiled_model = core.compile_model(model=model, device_name=device)

    # Получаем финальные слои
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    input_shape = input_layer.shape
    out_dim = input_shape[1]  # seq length = 9

    print(f"Модель загружена на: {device}")
    print(f"  Вход: {input_layer.any_name} {input_shape}")
    print(f"  Выход: {output_layer.any_name} {output_layer.shape}")
    print(f"  out_dim = {out_dim}")

    return compiled_model, input_layer, output_layer, out_dim, input_shape


def initialize_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не открыть видео: {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, w, h, fps, total


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


def preprocess_frames(frames, h=288, w=512):
    processed = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (w, h))
        normalized = resized.astype(np.float32) / 255.0
        processed.append(normalized)
    return processed


def _postprocess_output(output, threshold=0.5, in_h=288, in_w=512, out_dim=9):
    results = []
    for i in range(out_dim):
        heatmap = output[i, 0, :, :]
        _, binary = cv2.threshold(heatmap, threshold, 1.0, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            (binary * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                results.append((1, cx, cy))
            else:
                results.append((0, 0, 0))
        else:
            results.append((0, 0, 0))
    return results

def postprocess_output(output, threshold=0.5, in_h=288, in_w=512, out_dim=9):
    results = []
    for i in range(out_dim):
        heatmap = output[i]  # (288, 512)
        _, binary = cv2.threshold(heatmap, threshold, 1.0, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            (binary * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                results.append((1, cx, cy))
            else:
                results.append((0, 0, 0))
        else:
            results.append((0, 0, 0))
    return results

def draw_track(frame, track, cur_color=(0, 0, 255), hist_color=(255, 0, 0)):
    for p in list(track)[:-1]:
        if p: cv2.circle(frame, p, 5, hist_color, -1)
    if track and track[-1]: cv2.circle(frame, track[-1], 5, cur_color, -1)
    return frame


def read_frames(cap, q, max_n=9):
    frames = []
    while len(frames) < max_n:
        ret, f = cap.read()
        if not ret: break
        frames.append(f)
    q.put(frames if frames else None)


def main():
    args = parse_args()
    in_w, in_h = 512, 288
    batch_size = 9

    # Загрузка модели с исправлением динамической формы
    compiled_model, input_layer, output_layer, out_dim, input_shape = load_model(
        args.model_xml, device=args.device
    )

    cap, fw, fh, fps, total = initialize_video(args.video_path)
    base = os.path.splitext(os.path.basename(args.video_path))[0]
    writer, _ = setup_output_writer(base, args.output_dir, fw, fh, fps, args.only_csv)
    csv_path = setup_csv_file(base, args.output_dir)

    buffer = deque(maxlen=batch_size)
    track = deque(maxlen=args.track_length)
    frame_idx = 0
    q = queue.Queue(maxsize=2)

    def reader():
        while cap.isOpened():
            read_frames(cap, q, batch_size)
    threading.Thread(target=reader, daemon=True).start()

    pbar = tqdm(total=total, desc="Обработка", unit="кадр")
    exit_flag = False

    while True:
        start = time.time()
        batch = q.get()
        if batch is None: break

        proc = preprocess_frames(batch, in_h, in_w)

        while len(buffer) < batch_size:
            buffer.append(proc[0] if proc else np.zeros((in_h, in_w), np.float32))
        for f in proc:
            buffer.append(f)

        # Вход: (1,9,288,512) — гарантировано после reshape
        stacked = np.stack(buffer, axis=2)
        input_tensor = np.expand_dims(stacked, axis=0).transpose(0, 3, 1, 2)

        # Инференс
        result = compiled_model(input_tensor)
        output = result[output_layer]  # (1,9,288,512)

        preds = postprocess_output(output[0], in_h=in_h, in_w=in_w, out_dim=out_dim)

        for i, (vis, x, y) in enumerate(preds[:len(batch)]):
            x_orig = x * fw / in_w if vis else -1
            y_orig = y * fh / in_h if vis else -1

            if vis:
                track.append((int(x_orig), int(y_orig)))
            else:
                if track: track.popleft()

            res = {"Frame": frame_idx + i, "Visibility": vis, "X": int(x_orig), "Y": int(y_orig)}
            append_to_csv(res, csv_path)

            if args.visualize or writer:
                vis_frame = batch[i].copy()
                vis_frame = draw_track(vis_frame, track)
                if args.visualize:
                    cv2.imshow("Ball Tracking", vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        exit_flag = True
                        break
                if writer:
                    writer.write(vis_frame)
        if exit_flag: break

        pbar.update(len(batch))
        frame_idx += len(batch)

    pbar.close()
    cap.release()
    if writer: writer.release()
    if args.visualize: cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
