import cv2
import json
import argparse
import numpy as np
from scipy.signal import find_peaks
import matplotlib

matplotlib.use("Agg")  # Используем бэкенд без GUI
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize ball track with graphs and detect serves"
    )
    parser.add_argument(
        "--video_path", type=str, required=True, help="Path to the video file"
    )
    parser.add_argument(
        "--track_path", type=str, required=True, help="Path to the track JSON file"
    )
    return parser.parse_args()


def load_track(track_path: str):
    with open(track_path, "r") as f:
        track = json.load(f)
    positions = np.array(
        [[pos[0][0], pos[0][1], pos[1]] for pos in track["positions"]]
    )  # x, y, frame
    return positions, track["start_frame"], track["last_frame"], track["fps"]


def detect_serves(
    positions: np.ndarray, min_rise_speed: float = 1.5, min_peak_height: float = 300
):
    frames = positions[:, 2].astype(int)
    y = positions[:, 1]
    dy = np.gradient(y)
    dy_smooth = np.convolve(dy, np.ones(3) / 3, mode="same")

    rising_mask = dy_smooth > min_rise_speed
    rising_frames = frames[rising_mask]

    serves = []
    for f in rising_frames:
        idx = np.where(frames == f)[0][0]
        if idx == 0:
            continue
        prev_idx = idx - 1
        if y[prev_idx] > min_peak_height:
            continue
        future = y[idx : idx + 30]
        if len(future) < 2:
            continue
        if np.std(future) < 5:
            continue
        if np.max(future) - np.min(future) < 50:
            continue
        if not serves or (f - serves[-1] > 30):
            serves.append(f)
    return sorted(serves)


def create_graphs(
    positions: np.ndarray, current_frame: int, serves: list[int], frame_width: int
):
    x, y, frames = positions[:, 0], positions[:, 1], positions[:, 2].astype(int)
    speed = np.sqrt(np.gradient(x) ** 2 + np.gradient(y) ** 2)
    speed_smooth = np.convolve(speed, np.ones(5) / 5, mode="same")

    dpi = 100
    fig_width_px = frame_width  # Ширина графика совпадает с шириной кадра
    fig_height_px = 180  # Фиксированная высота 180px
    fig_width_inch = fig_width_px / dpi
    fig_height_inch = fig_height_px / dpi

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(fig_width_inch, fig_height_inch), dpi=dpi
    )

    ax1.plot(frames, x, "b-", linewidth=1)
    ax1.set_title("X Position")
    ax1.set_ylabel("X (px)")
    ax1.grid(True)
    ax1.invert_yaxis()
    if current_frame in frames:
        idx = np.where(frames == current_frame)[0][0]
        ax1.plot(current_frame, x[idx], "go", markersize=6)
    for serve in serves:
        ax1.axvline(serve, color="red", linestyle="--", alpha=0.7)

    ax2.plot(frames, y, "r-", linewidth=1)
    ax2.set_title("Y Position")
    ax2.set_ylabel("Y (px)")
    ax2.grid(True)
    ax2.invert_yaxis()
    if current_frame in frames:
        idx = np.where(frames == current_frame)[0][0]
        ax2.plot(current_frame, y[idx], "go", markersize=6)
    for serve in serves:
        ax2.axvline(serve, color="red", linestyle="--", alpha=0.7)

    ax3.plot(frames, speed_smooth, "purple", linewidth=1)
    ax3.set_title("Speed (smoothed)")
    ax3.set_ylabel("Speed (px/frame)")
    ax3.grid(True)
    if current_frame in frames:
        idx = np.where(frames == current_frame)[0][0]
        ax3.plot(current_frame, speed_smooth[idx], "go", markersize=6)
    for serve in serves:
        ax3.axvline(serve, color="red", linestyle="--", alpha=0.7)

    plt.tight_layout(pad=2.0)

    canvas = FigureCanvas(fig)
    canvas.draw()

    graph_image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    graph_image = graph_image.reshape((fig_height_px, fig_width_px, 4))[
        :, :, :3
    ]  # Убираем альфа-канал
    plt.close(fig)
    return graph_image


def main():
    args = parse_args()

    positions, start_frame, last_frame, fps = load_track(args.track_path)
    frames_arr = positions[:, 2].astype(int)
    pos_dict = {int(f): (x, y) for x, y, f in positions}

    serves = detect_serves(positions)
    print(f"Обнаружено подач: {len(serves)} на кадрах: {serves}")

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть видео {args.video_path}")
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_count = start_frame

    paused = False
    show_graphs = True

    print("Управление:")
    print("  Пробел — пауза/воспроизведение")
    print("  q — выход")
    print("  g — скрыть/показать графики")

    while frame_count <= last_frame:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Видео закончилось.")
                break
        else:
            ret, frame = cap.read()
            if not ret:
                break
            frame = frame.copy()

        # Создаём графики
        if show_graphs:
            graph_img = create_graphs(positions, frame_count, serves, frame.shape[1])
            # Накладываем графики внизу кадра
            h, w = frame.shape[:2]
            graph_h = graph_img.shape[0]  # 180px
            # Создаём новый кадр с дополнительным пространством снизу
            display_frame = np.zeros((h + graph_h, w, 3), dtype=np.uint8)
            display_frame[:h, :] = frame  # Видео в верхней части
            display_frame[h : h + graph_h, :] = graph_img  # Графики внизу
        else:
            display_frame = frame.copy()

        # Рисуем мяч на видео
        if frame_count in pos_dict:
            x, y = pos_dict[frame_count]
            x, y = int(x), int(y)
            cv2.circle(display_frame, (x, y), 8, (0, 0, 255), -1)
            cv2.circle(display_frame, (x, y), 10, (255, 255, 255), 2)
            cv2.putText(
                display_frame,
                f"Frame: {frame_count}",
                (x + 15, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        cv2.imshow("Ball Tracking with Graphs", display_frame)

        key = cv2.waitKey(1 if not paused else 0)
        if key == ord("q"):
            break
        elif key == 32:  # Пробел
            paused = not paused
        elif key == ord("g"):
            show_graphs = not show_graphs

        if not paused:
            frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
