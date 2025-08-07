import argparse
import json
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
import onnxruntime as ort
from collections import deque
import os
from pathlib import Path
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# --- Глобальные параметры ---
DETECTION_THRESHOLD = 0.5
DISTANCE_TOLERANCE = 9
DEFAULT_OUTPUT_DIR = "test_results"
DEFAULT_REPORT_LEVEL = "detailed"
FIGURE_DPI = 150
REPORT_BACKGROUND_COLOR = "#f8f9fa"


def parse_args():
    parser = argparse.ArgumentParser(
        description="ONNX Volleyball Ball Detection - Multi-Model Evaluation"
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        required=True,
        help="Root directory with match folders (e.g., data/test)",
    )
    parser.add_argument(
        "--model_paths",
        type=str,
        required=True,
        nargs="+",
        help="Paths to ONNX model files (multiple allowed)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for results (default: test_results)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable real-time visualization (only for the last model)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DETECTION_THRESHOLD,
        help="Detection threshold for heatmap (default: 0.5)",
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=DISTANCE_TOLERANCE,
        help="Distance tolerance in pixels (default: 9)",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=DEFAULT_REPORT_LEVEL,
        choices=["summary", "detailed"],
        help="Report detail level (default: detailed)",
    )
    return parser.parse_args()


def load_onnx_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    session = ort.InferenceSession(
        model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    input_names = [inp.name for inp in session.get_inputs()]
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
    out_dim = 9 if "seq9" in model_path.lower() else 3
    print(f"✅ Model loaded: {model_path}")
    print(
        f"   Has GRU state: {has_gru}, Output heatmaps: {out_dim}, h0 shape: {h0_shape if has_gru else 'N/A'}"
    )
    return session, has_gru, out_dim, h0_shape


def preprocess_frame(frame, height=288, width=512):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (width, height))
    normalized = resized.astype(np.float32) / 255.0
    return normalized


def postprocess_heatmap(
    heatmap,
    threshold=0.5,
    frame_width=1920,
    frame_height=1080,
    input_width=512,
    input_height=288,
):
    _, binary = cv2.threshold(heatmap, threshold, 1.0, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        (binary * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            x_orig = int(cx * frame_width / input_width)
            y_orig = int(cy * frame_height / input_height)
            return 1, x_orig, y_orig
    return 0, -1, -1


def load_ground_truth(video_path, test_dir):
    video_path = Path(video_path)
    video_name = video_path.stem
    match_folder = video_path.parent.parent.name
    gt_path = Path(test_dir) / match_folder / "csv" / f"{video_name}_ball.csv"
    if not gt_path.exists():
        print(f"⚠️ Ground truth CSV not found: {gt_path}")
        return None
    try:
        gt_df = pd.read_csv(gt_path)
        gt_dict = {
            row["Frame"]: {
                "visibility": row["Visibility"],
                "x": row["X"],
                "y": row["Y"],
            }
            for _, row in gt_df.iterrows()
        }
        return gt_dict
    except Exception as e:
        print(f"❌ Error loading ground truth CSV {gt_path}: {e}")
        return None


def classify_prediction(pred_coord, gt_coord, tolerance):
    has_pred = pred_coord is not None
    has_gt = gt_coord is not None
    if not has_pred and not has_gt:
        return "tn"
    elif not has_pred and has_gt:
        return "fn"
    elif has_pred and not has_gt:
        return "fp2"
    else:
        distance = np.sqrt(
            (pred_coord[0] - gt_coord[0]) ** 2 + (pred_coord[1] - gt_coord[1]) ** 2
        )
        if distance <= tolerance:
            return "tp"
        else:
            return "fp1"


def process_video(
    session,
    video_path,
    test_dir,
    has_gru,
    out_dim,
    h0_shape=None,
    threshold=0.5,
    input_height=288,
    input_width=512,
    visualize=False,
    tolerance=4,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return None, None
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    gt_dict = load_ground_truth(video_path, test_dir)
    frame_buffer = deque(maxlen=out_dim)
    h0 = np.zeros(h0_shape, dtype=np.float32) if has_gru else None
    track_points = deque(maxlen=8)
    results = {
        "tp": 0,
        "tn": 0,
        "fp1": 0,
        "fp2": 0,
        "fn": 0,
        "total_frames": 0,
        "detected_frames": 0,
    }
    center_idx = out_dim // 2  # центральный фрейм в последовательности

    pbar = tqdm(
        total=total_frames, desc=f"Processing {Path(video_path).name}", unit="frame"
    )
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed = preprocess_frame(frame, input_height, input_width)
        frame_buffer.append(processed)
        if len(frame_buffer) == out_dim:
            input_tensor = np.stack(frame_buffer, axis=0)
            input_tensor = np.expand_dims(input_tensor, axis=0)
            inputs = {"input": input_tensor}
            if has_gru:
                inputs["h0"] = h0
            try:
                outputs = session.run(None, inputs)
                output = outputs[0]
                if has_gru:
                    h0 = outputs[1]
            except Exception as e:
                print(f"❌ ONNX inference error: {e}")
                break
            center_frame_idx = frame_idx - (out_dim - 1) + center_idx
            heatmap = output[0, center_idx, :, :]
            visibility, x, y = postprocess_heatmap(
                heatmap, threshold, frame_width, frame_height, input_width, input_height
            )
            pred_coord = (x, y) if visibility else None
            if visibility:
                track_points.append((x, y))
                results["detected_frames"] += 1
            elif track_points:
                track_points.popleft()
            if gt_dict is not None and center_frame_idx in gt_dict:
                gt = gt_dict[center_frame_idx]
                gt_coord = (gt["x"], gt["y"]) if gt["visibility"] else None
                classification = classify_prediction(pred_coord, gt_coord, tolerance)
                results[classification] += 1
            else:
                if pred_coord is not None:
                    results["fp2"] += 1
                elif gt_dict is not None and center_frame_idx in gt_dict:
                    results["fn"] += 1
            results["total_frames"] += 1
        if visualize:
            vis_frame = frame.copy()
            if track_points:
                for pt in list(track_points)[:-1]:
                    cv2.circle(vis_frame, pt, 5, (255, 0, 0), -1)
                cv2.circle(vis_frame, track_points[-1], 6, (0, 0, 255), -1)
            if gt_dict is not None:
                current_gt_idx = frame_idx - (out_dim - 1) + center_idx
                if current_gt_idx in gt_dict:
                    gt = gt_dict[current_gt_idx]
                    if gt["visibility"]:
                        gt_x, gt_y = int(gt["x"]), int(gt["y"])
                        cv2.circle(vis_frame, (gt_x, gt_y), 4, (0, 255, 0), -1)
            cv2.imshow("Ball Tracking", vis_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        pbar.update(1)
        frame_idx += 1
    pbar.close()
    cap.release()
    if visualize:
        cv2.destroyAllWindows()
    return None, results


def calculate_metrics(results):
    tp, tn, fp1, fp2, fn = (
        results["tp"],
        results["tn"],
        results["fp1"],
        results["fp2"],
        results["fn"],
    )
    total_fp = fp1 + fp2
    total_predictions = tp + total_fp
    total_positives = tp + fn
    total_samples = tp + tn + total_fp + fn
    accuracy = (tp + tn) / total_samples if total_samples > 0 else 0
    precision = tp / total_predictions if total_predictions > 0 else 0
    recall = tp / total_positives if total_positives > 0 else 0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    detection_rate = (
        results["detected_frames"] / results["total_frames"]
        if results["total_frames"] > 0
        else 0
    )
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "detection_rate": detection_rate,
    }


def generate_summary_table(model_results, output_dir):
    df = pd.DataFrame(model_results)
    df["f1_score"] = df["f1_score"].round(3)
    df["accuracy"] = df["accuracy"].round(3)
    df["precision"] = df["precision"].round(3)
    df["recall"] = df["recall"].round(3)
    df["detection_rate"] = df["detection_rate"].round(3)
    csv_path = output_dir / "summary_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"📊 Summary table saved to: {csv_path}")
    return df


def print_summary_table(df):
    print("\n" + "=" * 80)
    print("SUMMARY: Model Comparison")
    print("=" * 80)
    print(
        f"{'Model':<40} {'F1':<8} {'Prec':<8} {'Recall':<8} {'Acc':<8} {'DetRate':<10}"
    )
    print("-" * 80)
    for _, row in df.iterrows():
        print(
            f"{Path(row['model']).name:<40} {row['f1_score']:<8.3f} {row['precision']:<8.3f} "
            f"{row['recall']:<8.3f} {row['accuracy']:<8.3f} {row['detection_rate']:<10.3f}"
        )
    print("=" * 80)


def generate_summary_plot(df, output_dir):
    plt.figure(figsize=(10, 6))
    models = df["model"].apply(lambda x: Path(x).name)
    metrics = ["f1_score", "precision", "recall", "accuracy"]
    x = np.arange(len(models))
    width = 0.2
    for i, metric in enumerate(metrics):
        plt.bar(
            x + i * width,
            df[metric],
            width,
            label=metric.capitalize().replace("_", " "),
        )
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.title("Model Comparison - Key Metrics")
    plt.xticks(x + width * 1.5, models, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "summary_plot.png", dpi=FIGURE_DPI)
    plt.close()
    print(f"📈 Summary plot saved to: {output_dir / 'summary_plot.png'}")


def main():
    print("\n" + "=" * 50)
    print("ONNX Volleyball Tracking - Multi-Model Evaluation")
    print("=" * 50)
    print(f"DETECTION_THRESHOLD:   {DETECTION_THRESHOLD}")
    print(f"DISTANCE_TOLERANCE:    {DISTANCE_TOLERANCE}")
    print(f"EVALUATION_METHOD:     Center Frame Only")
    print("=" * 50 + "\n")

    args = parse_args()
    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"multi_model_test_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Сохраняем конфиг
    with open(output_dir / "test_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    video_paths = list(test_dir.rglob("video/*.mp4"))
    if not video_paths:
        print(f"⚠️  No videos found in {test_dir}/video/")
        return

    print(f"🚀 Found {len(video_paths)} videos.")
    print(f"🧪 Testing {len(args.model_paths)} models...")

    # Список для хранения результатов по каждой модели
    model_results = []

    for i, model_path in enumerate(args.model_paths):
        print(f"\n🔄 TESTING MODEL [{i+1}/{len(args.model_paths)}]: {model_path}")

        # Включаем визуализацию только для последней модели, если указано
        visualize = args.visualize and (i == len(args.model_paths) - 1)

        session, has_gru, out_dim, h0_shape = load_onnx_model(model_path)

        all_results = {
            "tp": 0,
            "tn": 0,
            "fp1": 0,
            "fp2": 0,
            "fn": 0,
            "total_frames": 0,
            "detected_frames": 0,
        }
        start_time = time.time()

        for video_path in video_paths:
            print(f"  🎬 {video_path.name}")
            _, video_results = process_video(
                session=session,
                video_path=str(video_path),
                test_dir=test_dir,
                has_gru=has_gru,
                out_dim=out_dim,
                h0_shape=h0_shape,
                threshold=args.threshold,
                tolerance=args.tolerance,
                visualize=visualize,
            )
            if video_results:
                for key in all_results:
                    all_results[key] += video_results[key]

        test_time = time.time() - start_time
        metrics = calculate_metrics(all_results)

        # Сохраняем результаты модели
        model_results.append(
            {
                "model": model_path,
                "total_frames": all_results["total_frames"],
                "detected_frames": all_results["detected_frames"],
                **metrics,
                "test_time": test_time,
            }
        )

        # Выводим результаты текущей модели
        print_results(metrics, all_results, args, test_time)

    # Генерация сводной таблицы
    df_summary = generate_summary_table(model_results, output_dir)
    print_summary_table(df_summary)
    generate_summary_plot(df_summary, output_dir)

    print(f"\n🎉 All models tested! Summary saved to: {output_dir}")


def print_results(metrics, results, args, test_time):
    if args.report == "detailed":
        print(f"  Total Frames: {results['total_frames']}")
        print(
            f"  TP: {results['tp']}, FP1: {results['fp1']}, FP2: {results['fp2']}, FN: {results['fn']}, TN: {results['tn']}"
        )
    print(
        f"  F1: {metrics['f1_score']:.3f}, Prec: {metrics['precision']:.3f}, Rec: {metrics['recall']:.3f}, "
        f"Acc: {metrics['accuracy']:.3f}, DetRate: {metrics['detection_rate']:.3f}"
    )


if __name__ == "__main__":
    main()
