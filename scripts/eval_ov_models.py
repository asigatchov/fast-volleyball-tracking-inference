#!/usr/bin/env python3
"""Метрики OpenVINO-моделей из ov/ на тестовом наборе.

Прогоняет каждую модель по всем видео теста ровно тем же конвейером, что и
``src/inference_openvino_seq_gray_v2.py`` (те же load_model/preprocess_frames/
decode_predictions, те же непересекающиеся блоки по seq кадров), и сравнивает
предсказания с разметкой из ``<матч>/csv/<видео>_ball.csv``.

Классификация кадра повторяет src/test_models.py:
  tp  — мяч найден и попал в допуск
  fp1 — мяч найден, но дальше допуска от разметки
  fp2 — мяч найден там, где в разметке его нет
  fn  — мяч в разметке есть, модель его не нашла
  tn  — мяча нет ни там, ни там

Допуск задаётся в пикселях для эталонной ширины кадра (1920) и масштабируется
под фактическое разрешение: в тесте есть видео 2688x1512, и фиксированный
допуск в пикселях наказывал бы модель за то, что кадр крупнее.
"""

import argparse
import importlib.util
import json
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEST_DIR = "/home/ubuntu/datasets/volleyball-split/test"
DEFAULT_TOLERANCE = 9.0
REFERENCE_WIDTH = 1920.0


def load_inference_module():
    """Загрузить v2 как модуль, не таща за собой его argparse."""
    path = REPO_ROOT / "src" / "inference_openvino_seq_gray_v2.py"
    spec = importlib.util.spec_from_file_location("inference_v2", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["inference_v2"] = module
    spec.loader.exec_module(module)
    return module


V2 = load_inference_module()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test_dir", type=str, default=DEFAULT_TEST_DIR,
                        help="Корень теста: <матч>/video/*.mp4 и <матч>/csv/*_ball.csv")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Пути к моделям (по умолчанию все ov/*.xml)")
    parser.add_argument("--output_dir", type=str, default="eval_results",
                        help="Куда сложить отчёты")
    parser.add_argument("--device", type=str, default="CPU", help="CPU, GPU, AUTO")
    parser.add_argument("--threshold", type=float, default=0.5, help="Порог уверенности")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                        help=f"Допуск в px при ширине кадра {REFERENCE_WIDTH:.0f}")
    parser.add_argument("--skip_radius", action="store_true",
                        help="Не считать радиус по контурам движения (быстрее)")
    parser.add_argument("--limit_videos", type=int, default=None,
                        help="Взять только первые N видео (для быстрой проверки)")
    parser.add_argument("--sort_by", type=str, default="f1",
                        choices=["f1", "infer_fps", "precision", "recall", "accuracy",
                                 "mean_err_px"],
                        help="По какой колонке сортировать итоговую таблицу")
    return parser.parse_args()


def check_compatible(compiled_model, model_params):
    """Отсеять модели, которые v2 не умеет кормить.

    load_model проверяет только выход, а он у ballnet_v24_tula четырёхмерный и
    проверку проходит — падает уже на первом infer, потому что вход у неё
    [1,9,3,360,640]: пять осей, RGB. v2 подаёт [1, seq, H, W], так что рангом
    входа модель и отсекается — до инференса, а не посреди прогона.
    """
    input_shape = [int(dim) for dim in compiled_model.input(0).shape]
    if len(input_shape) != 4:
        raise ValueError(
            f"вход ранга {len(input_shape)} {input_shape}, v2 подаёт [1, seq, H, W]"
        )
    if input_shape[1] != model_params["seq"]:
        raise ValueError(f"ось seq входа {input_shape} не равна seq={model_params['seq']}")


def discover_models(models_arg):
    if models_arg:
        return [Path(p) for p in models_arg]
    return sorted((REPO_ROOT / "ov").glob("*.xml"))


def discover_videos(test_dir, limit=None):
    """Оба раскладки набора: <корень>/<матч>/video и плоская <корень>/video.

    volleyball-split разложен по матчам, beach-test-raw — одной папкой. Разметка
    в обоих случаях лежит в csv/ рядом с video/, так что load_ground_truth
    (parent.parent/"csv") работает одинаково; отличается только глубина поиска.
    """
    root = Path(test_dir)
    videos = sorted(root.glob("*/video/*.mp4")) or sorted(root.glob("video/*.mp4"))
    return videos[:limit] if limit else videos


def load_ground_truth(video_path):
    gt_path = video_path.parent.parent / "csv" / f"{video_path.stem}_ball.csv"
    if not gt_path.exists():
        return None
    df = pd.read_csv(gt_path)
    return {
        int(row.Frame): (
            (float(row.X), float(row.Y), float(getattr(row, "Radius", 0) or 0))
            if int(row.Visibility) else None
        )
        for row in df.itertuples(index=False)
    }


def classify(pred, gt, tolerance):
    """-> (класс, дистанция или None)"""
    if pred is None and gt is None:
        return "tn", None
    if pred is None:
        return "fn", None
    if gt is None:
        return "fp2", None
    distance = float(np.hypot(pred[0] - gt[0], pred[1] - gt[1]))
    return ("tp" if distance <= tolerance else "fp1"), distance


def new_counters():
    return {"tp": 0, "fp1": 0, "fp2": 0, "fn": 0, "tn": 0,
            "frames": 0, "detected": 0, "gt_visible": 0}


def evaluate_video(compiled_model, output_layer, model_params, video_path,
                   threshold, tolerance_px_at_ref, skip_radius):
    gt_dict = load_ground_truth(video_path)
    if gt_dict is None:
        return None

    cap, frame_width, frame_height, _, total = V2.initialize_video(video_path)
    tolerance = tolerance_px_at_ref * frame_width / REFERENCE_WIDTH
    seq = model_params["seq"]
    model_radius = model_params["planes"] in (2, 4)
    need_radius = not skip_radius

    counters = new_counters()
    distances = []
    radius_abs_errors = []
    infer_seconds = 0.0
    infer_calls = 0

    size_state = {
        "filtered_history": deque(maxlen=V2.BALL_SIZE_HISTORY),
        "raw_history": deque(maxlen=V2.BALL_RAW_SIZE_HISTORY),
        "smoothed_radius": 0.0,
    }
    prev_gray = None
    current_frames = []
    frame_index = 0

    pbar = tqdm(total=total, desc=video_path.stem[:34], unit="кадр", leave=False)
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            current_frames.append(frame)
            if len(current_frames) != seq:
                frame_index += 1
                pbar.update(1)
                continue

            processed = V2.preprocess_frames(
                current_frames,
                input_height=model_params["input_height"],
                input_width=model_params["input_width"],
            )
            tensor = np.asarray([processed], dtype=np.float32)
            started = time.perf_counter()
            output = compiled_model(tensor)[output_layer][0]
            infer_seconds += time.perf_counter() - started
            infer_calls += 1
            predictions = V2.decode_predictions(output, model_params, threshold)

            start_frame_index = frame_index - seq + 1
            for local_index, (frame_item, prediction) in enumerate(
                zip(current_frames, predictions, strict=True)
            ):
                visibility, x_resized, y_resized, radius_norm = prediction
                gray = None
                if visibility:
                    x_orig = int(x_resized * frame_width / model_params["input_width"])
                    y_orig = int(y_resized * frame_height / model_params["input_height"])
                    if model_radius:
                        radius = V2.model_radius_to_pixels(radius_norm, frame_width)
                    elif need_radius:
                        gray = cv2.cvtColor(frame_item, cv2.COLOR_BGR2GRAY)
                        radius, _ = V2.estimate_ball_radius(
                            prev_gray, gray, x_orig, y_orig, size_state
                        )
                    else:
                        radius = 0
                    pred = (x_orig, y_orig, radius)
                else:
                    pred = None
                if need_radius and not model_radius and gray is None:
                    gray = cv2.cvtColor(frame_item, cv2.COLOR_BGR2GRAY)
                prev_gray = gray if gray is not None else prev_gray

                key = start_frame_index + local_index
                if key not in gt_dict:
                    continue  # кадра нет в разметке — не штрафуем и не засчитываем
                gt = gt_dict[key]

                label, distance = classify(pred, gt, tolerance)
                counters[label] += 1
                counters["frames"] += 1
                counters["detected"] += int(pred is not None)
                counters["gt_visible"] += int(gt is not None)
                if distance is not None:
                    distances.append(distance)
                if (label == "tp" and gt[2] > 0 and pred[2] > 0):
                    radius_abs_errors.append(abs(pred[2] - gt[2]))

            current_frames = []
            frame_index += 1
            pbar.update(1)

        # Хвост короче seq модель не видит вовсе — это пропуски, а не отказ.
        if current_frames:
            start_frame_index = frame_index - len(current_frames)
            for local_index in range(len(current_frames)):
                key = start_frame_index + local_index
                if key not in gt_dict:
                    continue
                label, _ = classify(None, gt_dict[key], 0)
                counters[label] += 1
                counters["frames"] += 1
                counters["gt_visible"] += int(gt_dict[key] is not None)
    finally:
        pbar.close()
        cap.release()

    counters["distances"] = distances
    counters["radius_abs_errors"] = radius_abs_errors
    counters["infer_seconds"] = infer_seconds
    counters["infer_calls"] = infer_calls
    counters["tolerance_px"] = tolerance
    counters["resolution"] = f"{frame_width}x{frame_height}"
    return counters


def summarize(counters, name):
    tp, fp1, fp2, fn, tn = (counters["tp"], counters["fp1"], counters["fp2"],
                            counters["fn"], counters["tn"])
    fp = fp1 + fp2
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    distances = counters["distances"]
    radius_errors = counters["radius_abs_errors"]
    infer_seconds = counters["infer_seconds"]
    frames_per_second = counters["frames"] / infer_seconds if infer_seconds else 0.0
    return {
        "name": name,
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "accuracy": round((tp + tn) / total, 4) if total else 0.0,
        "detection_rate": round(counters["detected"] / counters["frames"], 4)
        if counters["frames"] else 0.0,
        "mean_err_px": round(float(np.mean(distances)), 2) if distances else None,
        "median_err_px": round(float(np.median(distances)), 2) if distances else None,
        "p95_err_px": round(float(np.percentile(distances, 95)), 2) if distances else None,
        "radius_mae_px": round(float(np.mean(radius_errors)), 2) if radius_errors else None,
        "tp": tp, "fp1": fp1, "fp2": fp2, "fn": fn, "tn": tn,
        "frames": counters["frames"],
        "gt_visible": counters["gt_visible"],
        "infer_fps": round(frames_per_second, 1),
    }


def merge(counters_list):
    total = new_counters()
    total["distances"] = []
    total["radius_abs_errors"] = []
    total["infer_seconds"] = 0.0
    total["infer_calls"] = 0
    for counters in counters_list:
        for key in ("tp", "fp1", "fp2", "fn", "tn", "frames", "detected", "gt_visible",
                    "infer_calls"):
            total[key] += counters[key]
        total["infer_seconds"] += counters["infer_seconds"]
        total["distances"].extend(counters["distances"])
        total["radius_abs_errors"].extend(counters["radius_abs_errors"])
    return total


def print_table(rows, title):
    columns = [("name", 52), ("f1", 7), ("precision", 10), ("recall", 7),
               ("accuracy", 9), ("detection_rate", 15), ("mean_err_px", 12),
               ("median_err_px", 14), ("radius_mae_px", 14), ("infer_fps", 10)]
    print(f"\n{title}")
    print("-" * sum(width + 1 for _, width in columns))
    print(" ".join(name[:width].ljust(width) for name, width in columns))
    print("-" * sum(width + 1 for _, width in columns))
    for row in rows:
        cells = []
        for key, width in columns:
            value = row.get(key)
            text = "—" if value is None else (f"{value:.3f}" if isinstance(value, float)
                                              else str(value))
            cells.append(text[:width].ljust(width))
        print(" ".join(cells))
    print("-" * sum(width + 1 for _, width in columns))


def main():
    args = parse_args()
    models = discover_models(args.models)
    videos = discover_videos(args.test_dir, args.limit_videos)
    if not videos:
        raise SystemExit(f"Видео не найдены в {args.test_dir}")
    print(f"Моделей: {len(models)}, видео: {len(videos)}, устройство: {args.device}, "
          f"порог: {args.threshold}, допуск: {args.tolerance} px @ {REFERENCE_WIDTH:.0f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    per_video_rows = []
    skipped = []

    for model_path in models:
        print(f"\n=== {model_path.name}", flush=True)
        try:
            compiled_model, _, output_layer, model_params = V2.load_model(
                str(model_path), device=args.device
            )
            check_compatible(compiled_model, model_params)
        except Exception as exc:
            print(f"  ПРОПУСК: несовместима с v2 — {exc}", flush=True)
            skipped.append({"model": model_path.name, "reason": str(exc)})
            continue

        started = time.perf_counter()
        per_video = []
        try:
            for video in videos:
                counters = evaluate_video(
                    compiled_model, output_layer, model_params, video,
                    args.threshold, args.tolerance, args.skip_radius,
                )
                if counters is None:
                    print(f"  нет разметки для {video.name}", flush=True)
                    continue
                per_video.append(counters)
                row = summarize(counters, video.stem)
                row["model"] = model_path.name
                row["resolution"] = counters["resolution"]
                row["tolerance_px"] = round(counters["tolerance_px"], 2)
                per_video_rows.append(row)
        except Exception as exc:
            # Одна сбойная модель не должна уносить с собой уже посчитанные.
            print(f"  ОШИБКА на прогоне: {exc}", flush=True)
            skipped.append({"model": model_path.name, "reason": str(exc)})
            continue

        if not per_video:
            continue
        row = summarize(merge(per_video), model_path.name)
        row["family"] = model_params["family"]
        row["seq"] = model_params["seq"]
        row["input"] = f"{model_params['input_width']}x{model_params['input_height']}"
        row["radius_from_model"] = model_params["planes"] in (2, 4)
        row["wall_seconds"] = round(time.perf_counter() - started, 1)
        summary_rows.append(row)
        print(f"  F1 {row['f1']:.3f}  P {row['precision']:.3f}  R {row['recall']:.3f}  "
              f"acc {row['accuracy']:.3f}  err {row['mean_err_px']} px  "
              f"{row['infer_fps']} кадр/с  за {row['wall_seconds']} с", flush=True)

    # mean_err_px — единственная колонка, где меньше значит лучше.
    ascending = args.sort_by == "mean_err_px"
    summary_rows.sort(key=lambda r: (r[args.sort_by] is None,
                                     r[args.sort_by] if r[args.sort_by] is not None else 0),
                      reverse=not ascending)
    print_table(summary_rows, f"ИТОГО по {len(videos)} видео "
                              f"(допуск {args.tolerance} px @ {REFERENCE_WIDTH:.0f}, "
                              f"сортировка по {args.sort_by})")
    if skipped:
        print("\nПропущены (не тот формат входа/выхода для v2):")
        for item in skipped:
            print(f"  {item['model']}: {item['reason'].splitlines()[0][:120]}")

    pd.DataFrame(summary_rows).to_csv(output_dir / "summary.csv", index=False)
    pd.DataFrame(per_video_rows).to_csv(output_dir / "per_video.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps({"config": vars(args), "summary": summary_rows,
                    "per_video": per_video_rows, "skipped": skipped},
                   ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nОтчёты: {output_dir}/summary.csv, per_video.csv, summary.json")


if __name__ == "__main__":
    main()
