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

DETECTION_THRESHOLD = 0.5
DISTANCE_TOLERANCE = 4
DEFAULT_OUTPUT_DIR = 'test_results'
DEFAULT_REPORT_LEVEL = 'detailed'
FIGURE_DPI = 150
REPORT_BACKGROUND_COLOR = '#f8f9fa'


def parse_args():
    parser = argparse.ArgumentParser(description="ONNX Volleyball Ball Detection and Evaluation")
    parser.add_argument("--test_dir", type=str, required=True, help="Root directory with match folders (e.g., data/test)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to ONNX model file")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for results (default: test_results)")
    parser.add_argument("--visualize", action="store_true",
                        help="Enable real-time visualization with predictions and ground truth")
    parser.add_argument("--threshold", type=float, default=DETECTION_THRESHOLD,
                        help="Detection threshold for heatmap (default: 0.5)")
    parser.add_argument("--tolerance", type=int, default=DISTANCE_TOLERANCE,
                        help="Distance tolerance in pixels (default: 4)")
    parser.add_argument("--report", type=str, default=DEFAULT_REPORT_LEVEL, choices=['summary', 'detailed'],
                        help="Report detail level (default: detailed)")
    return parser.parse_args()


def load_onnx_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    session = ort.InferenceSession(
        model_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
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
                if dim in ['batch', 'batch_size', None]:
                    resolved_shape.append(1)
                elif 'hidden' in str(dim).lower():
                    resolved_shape.append(512)
                else:
                    raise ValueError(f"Unknown symbolic dimension '{dim}' in h0_shape: {h0_shape}")
            else:
                resolved_shape.append(dim)
        h0_shape = tuple(resolved_shape)
    out_dim = 9 if "seq9" in model_path.lower() else 3
    if "grayscale" not in model_path.lower():
        print("⚠️  Model does not contain 'grayscale' in name — assuming RGB mode, but ONNX may still expect 9-channel input.")
    print(f"✅ Model loaded: {model_path}")
    print(f"   Has GRU state: {has_gru}, Output heatmaps: {out_dim}, h0 shape: {h0_shape if has_gru else 'N/A'}")
    return session, has_gru, out_dim, h0_shape


def preprocess_frame(frame, height=288, width=512):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (width, height))
    normalized = resized.astype(np.float32) / 255.0
    return normalized


def postprocess_heatmap(heatmap, threshold=0.5, frame_width=1920, frame_height=1080, input_width=512, input_height=288):
    _, binary = cv2.threshold(heatmap, threshold, 1.0, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours((binary * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
        gt_dict = {row['Frame']: {'visibility': row['Visibility'], 'x': row['X'], 'y': row['Y']} for _, row in
                   gt_df.iterrows()}
        return gt_dict
    except Exception as e:
        print(f"❌ Error loading ground truth CSV {gt_path}: {e}")
        return None


def classify_prediction(pred_coord, gt_coord, tolerance):
    has_pred = pred_coord is not None
    has_gt = gt_coord is not None
    if not has_pred and not has_gt:
        return 'tn'
    elif not has_pred and has_gt:
        return 'fn'
    elif has_pred and not has_gt:
        return 'fp2'
    else:
        distance = np.sqrt((pred_coord[0] - gt_coord[0]) ** 2 + (pred_coord[1] - gt_coord[1]) ** 2)
        if distance <= tolerance:
            return 'tp'
        else:
            return 'fp1'


def process_video(session, video_path, test_dir, has_gru, out_dim, h0_shape=None, threshold=0.5, input_height=288,
                  input_width=512, visualize=False, tolerance=4):
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
    results = {'tp': 0, 'tn': 0, 'fp1': 0, 'fp2': 0, 'fn': 0, 'total_frames': 0, 'detected_frames': 0}

    center_idx = 1 if out_dim == 3 else 4

    pbar = tqdm(total=total_frames, desc=f"Processing {Path(video_path).name}", unit="frame")
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
                heatmap, threshold,
                frame_width, frame_height, input_width, input_height
            )
            pred_coord = (x, y) if visibility else None

            if visibility:
                track_points.append((x, y))
                results['detected_frames'] += 1
            elif track_points:
                track_points.popleft()

            if gt_dict is not None and center_frame_idx in gt_dict:
                gt = gt_dict[center_frame_idx]
                gt_coord = (gt['x'], gt['y']) if gt['visibility'] else None
                classification = classify_prediction(pred_coord, gt_coord, tolerance)
                results[classification] += 1
            else:
                if pred_coord is not None:
                    results['fp2'] += 1
                elif gt_dict is not None and center_frame_idx in gt_dict:
                    results['fn'] += 1

            results['total_frames'] += 1

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
                    if gt['visibility']:
                        gt_x, gt_y = int(gt['x']), int(gt['y'])
                        cv2.circle(vis_frame, (gt_x, gt_y), 4, (0, 255, 0), -1)
            cv2.imshow("Ball Tracking", vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        pbar.update(1)
        frame_idx += 1

    pbar.close()
    cap.release()
    if visualize:
        cv2.destroyAllWindows()

    return None, results


def calculate_metrics(results):
    tp, tn, fp1, fp2, fn = results['tp'], results['tn'], results['fp1'], results['fp2'], results['fn']
    total_fp = fp1 + fp2
    total_predictions = tp + total_fp
    total_positives = tp + fn
    total_samples = tp + tn + total_fp + fn
    accuracy = (tp + tn) / total_samples if total_samples > 0 else 0
    precision = tp / total_predictions if total_predictions > 0 else 0
    recall = tp / total_positives if total_positives > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    detection_rate = results['detected_frames'] / results['total_frames'] if results['total_frames'] > 0 else 0
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'detection_rate': detection_rate
    }


def generate_visualizations(save_dir, results, metrics, args):
    print("Generating visualizations...")
    fig, ax = plt.subplots(figsize=(8, 6))
    cm_data = np.array([[results['tp'], results['fp1'] + results['fp2']],
                        [results['fn'], results['tn']]])
    sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Positive', 'Predicted Negative'],
                yticklabels=['Actual Positive', 'Actual Negative'], ax=ax)
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_dir / 'confusion_matrix.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()

    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor(REPORT_BACKGROUND_COLOR)
    fig.suptitle('ONNX Volleyball Tracking Results - Center Frame Evaluation', fontsize=24, fontweight='bold', y=0.95,
                 color='#2c3e50')

    ax1 = plt.subplot2grid((3, 4), (0, 0), colspan=2, rowspan=1)
    ax1.axis('off')
    config_text = f"""Dataset: {args.test_dir}
Model: {args.model_path}
Threshold: {args.threshold} | Tolerance: {args.tolerance}px
Evaluation: Center Frame Only"""
    ax1.text(0.05, 0.5, config_text, fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='#e8f4fd', alpha=0.8))

    ax2 = plt.subplot2grid((3, 4), (0, 2), colspan=2, rowspan=1)
    ax2.axis('off')
    cm_text = f"""True Positives: {results['tp']:,}
True Negatives: {results['tn']:,}
False Positives (Wrong): {results['fp1']:,}
False Positives (False): {results['fp2']:,}
False Negatives: {results['fn']:,}"""
    ax2.text(0.05, 0.5, cm_text, fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='#fff2e8', alpha=0.8))

    ax3 = plt.subplot2grid((3, 4), (1, 0), colspan=4, rowspan=1)
    ax3.axis('off')
    metrics_data = [
        ('Accuracy', metrics['accuracy'], '#27ae60'),
        ('Precision', metrics['precision'], '#3498db'),
        ('Recall', metrics['recall'], '#e74c3c'),
        ('F1-Score', metrics['f1_score'], '#9b59b6'),
        ('Detection Rate', metrics['detection_rate'], '#f39c12')
    ]
    y_pos = 0.8
    for name, value, color in metrics_data:
        ax3.barh(y_pos, 1, height=0.08, color='#ecf0f1', alpha=0.5)
        ax3.barh(y_pos, value, height=0.08, color=color, alpha=0.8)
        ax3.text(0.02, y_pos, name, fontsize=11, fontweight='bold', va='center')
        ax3.text(0.98, y_pos, f'{value:.3f} ({value * 100:.1f}%)', fontsize=11, fontweight='bold', va='center',
                 ha='right')
        y_pos -= 0.15
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)

    ax4 = plt.subplot2grid((3, 4), (2, 0), colspan=4, rowspan=1)
    ax4.axis('off')
    total_frames = results['total_frames']
    summary_text = f"""Total Center Frames Processed: {total_frames:,}
Test Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
    ax4.text(0.5, 0.5, summary_text, fontsize=14, fontweight='bold', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.8", facecolor='#d5f4e6', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_dir / 'test_report.png', dpi=FIGURE_DPI, bbox_inches='tight',
                facecolor=REPORT_BACKGROUND_COLOR, edgecolor='none')
    plt.close()
    print(f"Visualizations saved to {save_dir}")


def print_results(metrics, results, args, test_time):
    print("\n" + "=" * 60)
    print("ONNX Volleyball Tracking Results - Center Frame Evaluation")
    print("=" * 60)
    if args.report == 'detailed':
        print(f"Test Dataset: {args.test_dir}")
        print(f"Model: {args.model_path}")
        print(f"Detection Threshold: {args.threshold}")
        print(f"Distance Tolerance: {args.tolerance} pixels")
        print(f"Evaluation Method: Center Frame Only")
        print("-" * 60)
        print("Confusion Matrix:")
        print(f"  True Positives (TP): {results['tp']}")
        print(f"  True Negatives (TN): {results['tn']}")
        print(f"  False Positives (FP1 - wrong position): {results['fp1']}")
        print(f"  False Positives (FP2 - false detection): {results['fp2']}")
        print(f"  False Negatives (FN): {results['fn']}")
        print("-" * 60)
    print("Performance Metrics:")
    print(f"  Accuracy:       {metrics['accuracy']:.3f} ({metrics['accuracy'] * 100:.1f}%)")
    print(f"  Precision:      {metrics['precision']:.3f} ({metrics['precision'] * 100:.1f}%)")
    print(f"  Recall:         {metrics['recall']:.3f} ({metrics['recall'] * 100:.1f}%)")
    print(f"  F1-Score:       {metrics['f1_score']:.3f}")
    print(f"  Detection Rate: {metrics['detection_rate']:.3f} ({metrics['detection_rate'] * 100:.1f}%)")
    print("-" * 60)
    print(f"Total Center Frames Evaluated: {results['total_frames']}")
    print(f"Testing completed in {test_time:.1f}s")
    print(f"Processing speed: {results['total_frames'] / test_time:.1f} FPS")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)


def main():
    print("\n" + "=" * 50)
    print("ONNX Volleyball Tracking Configuration")
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
    output_dir = Path(args.output_dir) / f"test_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "test_config.json", 'w') as f:
        json.dump(vars(args), f, indent=2)

    session, has_gru, out_dim, h0_shape = load_onnx_model(args.model_path)
    video_paths = list(test_dir.rglob("video/*.mp4"))
    if not video_paths:
        print(f"⚠️  No videos found in {test_dir}/video/")
        return

    print(f"🚀 Found {len(video_paths)} videos. Starting batch processing...")
    all_results = {'tp': 0, 'tn': 0, 'fp1': 0, 'fp2': 0, 'fn': 0, 'total_frames': 0, 'detected_frames': 0}
    start_time = time.time()

    for video_path in video_paths:
        print(f"\n🎬 Processing: {video_path}")
        _, video_results = process_video(
            session=session,
            video_path=str(video_path),
            test_dir=test_dir,
            has_gru=has_gru,
            out_dim=out_dim,
            h0_shape=h0_shape,
            threshold=args.threshold,
            tolerance=args.tolerance,
            visualize=args.visualize
        )
        if video_results:
            for key in all_results:
                all_results[key] += video_results[key]

    test_time = time.time() - start_time
    metrics = calculate_metrics(all_results)
    generate_visualizations(output_dir, all_results, metrics, args)
    print_results(metrics, all_results, args, test_time)
    print(f"\n🎉 Batch processing complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()