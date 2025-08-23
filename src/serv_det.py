import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from typing import List, Tuple
import os


def load_track_data(file_path: str) -> List[List]:
    """Загружает данные траектории мяча из JSON-файла."""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        return data["positions"]
    except FileNotFoundError:
        print(f"Файл {file_path} не найден.")
        return []
    except KeyError:
        print(f"Ключ 'positions' не найден в файле {file_path}.")
        return []


def merge_cyclic_sequences(
    sequences: List[Tuple[int, int]], max_frame_gap: int = 10
) -> List[Tuple[int, int]]:
    """Объединяет циклические участки, если расстояние между ними не превышает max_frame_gap кадров."""
    if not sequences:
        return []

    sequences = sorted(sequences, key=lambda x: x[0])
    merged = []
    current_start, current_end = sequences[0]

    for start, end in sequences[1:]:
        if start <= current_end + max_frame_gap:
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end

    merged.append((current_start, current_end))
    return merged


def find_cyclic_sequences(
    positions: List[List],
    min_cycle_amplitude: float = 50.0,
    max_amplitude_variation: float = 40.0,
    min_num_amplitudes: int = 3,
    max_frame_gap: int = 10,
) -> List[Tuple[int, int]]:
    """Находит участки с регулярными циклическими движениями мяча (≥2 цикла)."""
    if not positions or len(positions) < 10:
        return []

    pos_array = np.array(
        [(pos[0][0], pos[0][1], pos[1]) for pos in positions], dtype=np.float64
    )
    x_values = pos_array[:, 0]
    y_values = pos_array[:, 1]
    frames = pos_array[:, 2].astype(int)

    sequences = []
    i = 0
    n = len(pos_array)

    while i < n - 10:
        start_idx = i
        j = i + 1

        while j < n:
            x_range = np.max(x_values[i : j + 1]) - np.min(x_values[i : j + 1])
            if x_range > 150:
                break
            j += 1

        if j - i < 100:
            i = j
            continue

        y_segment = y_values[i:j]
        total_y_range = np.max(y_segment) - np.min(y_segment)
        if total_y_range < min_cycle_amplitude:
            i = j
            continue

        peaks, _ = find_peaks(y_segment, prominence=10)
        troughs, _ = find_peaks(-y_segment, prominence=10)

        if len(peaks) < 2 or len(troughs) < 2:
            i = j
            continue

        events = sorted(
            [(p, "peak") for p in peaks] + [(t, "trough") for t in troughs],
            key=lambda x: x[0],
        )

        amplitudes = []
        for k in range(1, len(events)):
            prev_idx, _ = events[k - 1]
            curr_idx, _ = events[k]
            amplitude = abs(y_segment[curr_idx] - y_segment[prev_idx])
            amplitudes.append(amplitude)

        if len(amplitudes) < min_num_amplitudes:
            i = j
            continue

        good_segments = []
        amp_idx = 0
        while amp_idx < len(amplitudes):
            if amplitudes[amp_idx] < min_cycle_amplitude:
                amp_idx += 1
                continue
            amp_j = amp_idx
            while amp_j < len(amplitudes) and amplitudes[amp_j] >= min_cycle_amplitude:
                amp_j += 1
            if amp_j - amp_idx >= min_num_amplitudes:
                good_segments.append((amp_idx, amp_j))
            amp_idx = amp_j

        for amp_start, amp_end in good_segments:
            left = amp_start
            for right in range(amp_start, amp_end):
                sub = amplitudes[left : right + 1]
                sub_min = min(sub)
                sub_max = max(sub)
                while (sub_max - sub_min > max_amplitude_variation) and left <= right:
                    left += 1
                    sub = amplitudes[left : right + 1]
                    if sub:
                        sub_min = min(sub)
                        sub_max = max(sub)
                if right - left + 1 >= min_num_amplitudes:
                    event_left = events[left][0]
                    event_right = events[right + 1][0]
                    f_start = int(frames[i + event_left])
                    f_end = int(frames[i + event_right])
                    sequences.append((f_start, f_end))
                    left = right + 1

        i = j

    sequences = merge_cyclic_sequences(sequences, max_frame_gap=max_frame_gap)
    return sequences


def find_rolling_sequences(
    positions: List[List],
    max_y_range: float = 40.0,  # Уменьшено для трека 0005
    min_x_range: float = 50.0,
    min_length: int = 60,  # Уменьшено для коротких участков
) -> List[Tuple[int, int]]:
    """Находит участки, где мяч катится по полу (малый размах Y, большой размах X)."""
    if not positions or len(positions) < min_length:
        return []

    pos_array = np.array(
        [(pos[0][0], pos[0][1], pos[1]) for pos in positions], dtype=np.float64
    )
    x_values = pos_array[:, 0]
    y_values = pos_array[:, 1]
    frames = pos_array[:, 2].astype(int)

    sequences = []
    i = 0
    n = len(pos_array)

    while i < n - min_length + 1:
        j = i + min_length - 1
        while j < n:
            y_range = np.max(y_values[i : j + 1]) - np.min(y_values[i : j + 1])
            x_range = np.max(x_values[i : j + 1]) - np.min(x_values[i : j + 1])
            if y_range <= max_y_range and x_range >= min_x_range:
                j += 1
            else:
                break
        if j - i >= min_length:
            sequences.append((int(frames[i]), int(frames[j - 1])))
        i += 1


    sequences = merge_cyclic_sequences(sequences, max_frame_gap=30)

    return sequences


def plot_2d_trajectory(
    positions: List[List],
    file_name: str,
    cyclic_sequences: List[Tuple[int, int]],
    rolling_sequences: List[Tuple[int, int]],
):
    """Строит три 2D-графика: X от frame_num, Y от frame_num и траекторию X-Y,
    выделяя циклические участки красным и участки качения зеленым.
    """
    if not positions:
        print(f"Нет данных для построения графиков для файла {file_name}.")
        return

    x = np.array([pos[0][0] for pos in positions])
    y = np.array([pos[0][1] for pos in positions])
    frames = np.array([pos[1] for pos in positions])

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=False)

    # График X(frame)
    ax1.plot(frames, x, c="#1E90FF", linewidth=1, label="X (обычный)")
    for start, end in cyclic_sequences:
        mask = (frames >= start) & (frames <= end)
        ax1.plot(frames[mask], x[mask], c="red", linewidth=2, label="X (циклический)")
        ax1.axvline(start, color="gray", linestyle="--")
        ax1.axvline(end, color="gray", linestyle="--")
    for start, end in rolling_sequences:
        mask = (frames >= start) & (frames <= end)
        ax1.plot(frames[mask], x[mask], c="green", linewidth=2, label="X (качение)")
        ax1.axvline(start, color="darkgreen", linestyle="-.")
        ax1.axvline(end, color="darkgreen", linestyle="-.")
    ax1.set_xlabel("Номер кадра")
    ax1.set_ylabel("X (пиксели)")
    ax1.set_title(f"Координата X по кадрам ({file_name})")
    ax1.legend()
    ax1.grid(True)

    # График Y(frame)
    ax2.plot(frames, y, c="#1E90FF", linewidth=1, label="Y (обычный)")
    for start, end in cyclic_sequences:
        mask = (frames >= start) & (frames <= end)
        ax2.plot(frames[mask], y[mask], c="red", linewidth=2, label="Y (циклический)")
        ax2.axvline(start, color="gray", linestyle="--")
        ax2.axvline(end, color="gray", linestyle="--")
    for start, end in rolling_sequences:
        mask = (frames >= start) & (frames <= end)
        ax2.plot(frames[mask], y[mask], c="green", linewidth=2, label="Y (качение)")
        ax2.axvline(start, color="darkgreen", linestyle="-.")
        ax2.axvline(end, color="darkgreen", linestyle="-.")
    ax2.set_xlabel("Номер кадра")
    ax2.set_ylabel("Y (пиксели)")
    ax2.set_title(f"Координата Y по кадрам ({file_name})")
    ax2.invert_yaxis()
    ax2.legend()
    ax2.grid(True)

    # График X-Y
    ax3.plot(x, y, c="#1E90FF", linewidth=1, label="Траектория (обычная)")
    for start, end in cyclic_sequences:
        mask = (frames >= start) & (frames <= end)
        ax3.plot(
            x[mask], y[mask], c="red", linewidth=2, label="Траектория (циклическая)"
        )
    for start, end in rolling_sequences:
        mask = (frames >= start) & (frames <= end)
        ax3.plot(x[mask], y[mask], c="green", linewidth=2, label="Траектория (качение)")
    ax3.set_xlabel("X (пиксели)")
    ax3.set_ylabel("Y (пиксели)")
    ax3.set_title(f"Траектория X-Y ({file_name})")
    ax3.invert_yaxis()
    ax3.legend()
    ax3.grid(True)

    # Выводим найденные участки
    if cyclic_sequences or rolling_sequences:
        print(f"Участки для {file_name}:")
        start_frame = int(frames[0]) if len(frames) > 0 else None
        end_frame = int(frames[-1]) if len(frames) > 0 else None
        try:
            track_id = int(file_name.split("_")[1].split(".")[0])
        except Exception:
            track_id = "unknown"
        print(
            f"track_id: {track_id}, start_frame: {start_frame}, end_frame: {end_frame}"
        )
        for start, end in cyclic_sequences:
            print(f"Циклический участок: Начало: кадр {start}, Конец: кадр {end}")
        for start, end in rolling_sequences:
            print(f"Участок качения: Начало: кадр {start}, Конец: кадр {end}")

    plt.tight_layout()
    plt.show()


def main():
    """Основная функция для обработки всех файлов из каталога track_json и построения графиков."""
    base_path = "track_json"
    track_files = [f for f in os.listdir(base_path) if f.endswith(".json")]
    track_files.sort()
    for track_file in track_files:
        file_path = os.path.join(base_path, track_file)
        print(f"\nОбработка файла: {track_file}")
        positions = load_track_data(file_path)
        cyclic_sequences = find_cyclic_sequences(positions, max_frame_gap=10)
        rolling_sequences = find_rolling_sequences(positions)
        plot_2d_trajectory(positions, track_file, cyclic_sequences, rolling_sequences)


if __name__ == "__main__":
    main()
