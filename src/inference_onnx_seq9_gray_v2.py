import argparse
import cv2
import numpy as np
import pandas as pd
import onnxruntime as ort
from collections import deque
import os
import time
from tqdm import tqdm
import threading
import queue


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
    return parser.parse_args()


def load_model(model_path, input_height=288, input_width=512):
    if not os.path.exists(model_path):
        raise ValueError(f"Model file not found: {model_path}")
    if not model_path.endswith(".onnx"):
        raise ValueError(
            f"Expected ONNX model file with .onnx extension, got: {model_path}"
        )
    session = ort.InferenceSession(
        model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    out_dim = 9 if "seq9_grayscale" in model_path else 3
    return session, out_dim


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
    output_path = os.path.join(output_dir, f"{video_basename}_predict.mp4")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    out_writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height)
    )
    return out_writer, output_path


def setup_csv_file(video_basename, output_dir):
    if output_dir is None:
        return None
    csv_path = os.path.join(output_dir, f"{video_basename}_predict_ball.csv")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pd.DataFrame(columns=["Frame", "Visibility", "X", "Y"]).to_csv(
        csv_path, index=False
    )
    return csv_path


def append_to_csv(result, csv_path):
    if csv_path is None:
        return
    pd.DataFrame([result]).to_csv(csv_path, mode="a", header=False, index=False)


def preprocess_frames(frames, input_height=288, input_width=512):
    processed = []
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (input_width, input_height))
        frame = frame.astype(np.float32) / 255.0
        processed.append(frame)
    return processed


def postprocess_output(
    output, threshold=0.5, input_height=288, input_width=512, out_dim=9
):
    results = []
    for frame_idx in range(out_dim):  # Process all 9 heatmaps
        heatmap = output[0, frame_idx, :, :]
        _, binary = cv2.threshold(heatmap, threshold, 1.0, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            (binary * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                results.append((1, cx, cy))
            else:
                results.append((0, 0, 0))
        else:
            results.append((0, 0, 0))
    return results


def draw_track(
    frame, track_points, current_color=(0, 0, 255), history_color=(255, 0, 0)
):
    for point in list(track_points)[:-1]:
        if point is not None:
            cv2.circle(frame, point, 5, history_color, -1)
    if track_points and track_points[-1] is not None:
        cv2.circle(frame, track_points[-1], 5, current_color, -1)
    return frame


def read_frames(cap, frame_queue, max_frames=9):
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    if frames:
        frame_queue.put(frames)
    else:
        frame_queue.put(None)


def main():
    args = parse_args()
    input_width, input_height = 512, 288
    batch_size = 9

    model_session, out_dim = load_model(args.model_path, input_height, input_width)
    cap, frame_width, frame_height, fps, total_frames = initialize_video(
        args.video_path
    )
    video_basename = os.path.splitext(os.path.basename(args.video_path))[0]
    out_writer, _ = setup_output_writer(
        video_basename, args.output_dir, frame_width, frame_height, fps, args.only_csv
    )
    csv_path = setup_csv_file(video_basename, args.output_dir)

    frame_buffer = deque(maxlen=batch_size)
    track_points = deque(maxlen=args.track_length)
    frame_index = 0
    frame_queue = queue.Queue(maxsize=2)

    # Start frame reading thread
    def frame_reader():
        while cap.isOpened():
            read_frames(cap, frame_queue, batch_size)

    reader_thread = threading.Thread(target=frame_reader, daemon=True)
    reader_thread.start()

    pbar = tqdm(total=total_frames, desc="Processing video", unit="frame")
    exit_flag = False
    while True:
        start_time = time.time()

        # Get batch of frames
        frames = frame_queue.get()
        if frames is None:
            break

        # Preprocess frames in batch
        processed_frames = preprocess_frames(frames, input_height, input_width)

        # Fill buffer if not enough frames
        while len(frame_buffer) < batch_size:
            frame_buffer.append(
                processed_frames[0]
                if processed_frames
                else np.zeros((input_height, input_width), dtype=np.float32)
            )

        # Update buffer with new frames
        for pf in processed_frames:
            frame_buffer.append(pf)

        # Prepare input tensor
        input_tensor = np.stack(frame_buffer, axis=2)  # (height, width, 9)
        input_tensor = np.expand_dims(input_tensor, axis=0)  # (1, height, width, 9)
        input_tensor = np.transpose(input_tensor, (0, 3, 1, 2))  # (1, 9, 288, 512)

        # Run inference
        inputs = {model_session.get_inputs()[0].name: input_tensor}
        output = model_session.run(None, inputs)[0]

        # Process predictions for all 9 frames
        predictions = postprocess_output(
            output, input_height=input_height, input_width=input_width, out_dim=out_dim
        )

        # Save results and visualize for each frame in the batch
        for i, (visibility, x, y) in enumerate(predictions[: len(frames)]):
            x_orig = x * frame_width / input_width if visibility else -1
            y_orig = y * frame_height / input_height if visibility else -1

            if visibility:
                track_points.append((int(x_orig), int(y_orig)))
            else:
                if track_points:
                    track_points.popleft()

            result = {
                "Frame": frame_index + i,
                "Visibility": visibility,
                "X": int(x_orig),
                "Y": int(y_orig),
            }
            append_to_csv(result, csv_path)

            if args.visualize or out_writer is not None:
                vis_frame = frames[i].copy()
                vis_frame = draw_track(vis_frame, track_points)
                if args.visualize:
                    cv2.namedWindow("Ball Tracking", cv2.WINDOW_NORMAL)
                    cv2.imshow("Ball Tracking", vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        exit_flag = True  # Set flag to exit
                        break
                if out_writer is not None:
                    out_writer.write(vis_frame)
        if exit_flag:
            break

        end_time = time.time()
        batch_time = end_time - start_time
        batch_fps = len(frames) / batch_time if batch_time > 0 else 0
        pbar.update(len(frames))
        frame_index += len(frames)

    pbar.close()
    cap.release()
    if out_writer is not None:
        out_writer.release()
    if args.visualize:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
