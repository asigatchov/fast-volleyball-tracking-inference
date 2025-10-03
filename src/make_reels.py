import json
import cv2
import numpy as np
import argparse
import os
from tqdm import tqdm  # <-- добавлен импорт


def ensure_reels_dir():
    os.makedirs("reels", exist_ok=True)


def load_single_track(track_json_path):
    """Загружает один трек из JSON-файла в формате из примера."""
    with open(track_json_path, "r") as f:
        data = json.load(f)

    positions = []
    for item in data["positions"]:
        xy, frame = item
        x, y = xy
        positions.append((float(x), float(y), int(frame)))

    return {
        "start_frame": data["start_frame"],
        "last_frame": data["last_frame"],
        "positions": positions,
    }


def crop_and_save_track(video_path, track, output_path, visualize=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # fallback
    delay = int(1000 / fps)

    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError("Не удалось прочитать первый кадр видео")

    frame_height, frame_width = frame.shape[:2]
    aspect_ratio = 9 / 16
    crop_width = int(frame_height * aspect_ratio)
    crop_height = frame_height

    # Инициализация VideoWriter
    out = None
    if not visualize:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (crop_width, crop_height))

    # Словарь: frame → (x, y)
    frame_to_pos = {int(p[2]): (p[0], p[1]) for p in track["positions"]}
    all_frames = sorted(frame_to_pos.keys())
    start_frame = track["start_frame"]
    end_frame = track["last_frame"]

    if not all_frames:
        raise ValueError("Трек не содержит позиций мяча")

    first_known_x = frame_to_pos[all_frames[0]][0]

    # Построим x-координату центра для каждого кадра
    x_centers = []
    for frame_idx in range(start_frame, end_frame + 1):
        if frame_idx in frame_to_pos:
            x_centers.append(frame_to_pos[frame_idx][0])
        else:
            prev_frames = [f for f in all_frames if f <= frame_idx]
            if prev_frames:
                nearest = max(prev_frames)
                x_centers.append(frame_to_pos[nearest][0])
            else:
                x_centers.append(first_known_x)

    # Сглаживание
    x_values = np.array(x_centers)
    window = min(15, len(x_values))
    if window > 1:
        pad = window // 2
        x_padded = np.pad(x_values, (pad, pad), mode="edge")
        x_smooth = np.convolve(x_padded, np.ones(window) / window, mode="valid")
    else:
        x_smooth = x_values

    # === Начало обработки кадров с прогресс-баром ===
    total_frames = end_frame - start_frame + 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # Перемотка к началу трека

    for i, frame_idx in enumerate(tqdm(range(start_frame, end_frame + 1),
                                       desc="Сохранение кадров",
                                       total=total_frames,
                                       unit="кадр")):
        ret, frame = cap.read()
        if not ret:
            print(f"\n⚠️ Кадр {frame_idx} не прочитан — видео закончилось раньше времени.")
            break

        # Определяем центр кропа по сглаженной траектории
        smooth_x = x_smooth[i]
        center_x = int(smooth_x)
        left = max(0, center_x - crop_width // 2)
        right = min(frame_width, left + crop_width)
        actual_crop_width = right - left

        # Если кроп выходит за границы — подправим
        if actual_crop_width < crop_width:
            left = max(0, right - crop_width)
            actual_crop_width = crop_width

        cropped = frame[:, left:left + crop_width]

        # Добавляем кадр в видео
        if out is not None:
            # Убедимся, что размер совпадает
            if cropped.shape[1] != crop_width:
                cropped = cv2.resize(cropped, (crop_width, crop_height))
            out.write(cropped)

        # Визуализация
        if visualize:
            cv2.imshow("Cropped", cropped)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break

    # Завершение
    cap.release()
    if out is not None:
        out.release()
    if visualize:
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Визуализация одного розыгрыша с кадрированием по мячу."
    )
    parser.add_argument("--video_path", required=True, help="Путь к видеофайлу")
    parser.add_argument(
        "--track_json", required=True, help="Путь к JSON-файлу с одним треком"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Показывать видео в реальном времени с кадрированием",
    )
    args = parser.parse_args()

    track = load_single_track(args.track_json)
    base_name = os.path.splitext(os.path.basename(args.video_path))[0]
    ensure_reels_dir()
    output_path = os.path.join("reels", f"reel_{base_name}.mp4")

    crop_and_save_track(args.video_path, track, output_path, visualize=args.visualize)

    if not args.visualize:
        print(f"✅ Сохранено: {output_path}")


if __name__ == "__main__":
    main()
