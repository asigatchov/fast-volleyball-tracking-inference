from scipy.signal import find_peaks
import numpy as np


def merge_sequences(
    sequences, max_frame_gap=10
):
    """Объединяет циклические участки, если расстояние между ними не превышает max_frame_gap кадров.

    Args:
        sequences: Список кортежей (start_frame, end_frame) для циклических участков.
        max_frame_gap: Максимальное расстояние между участками (в кадрах) для их объединения.

    Returns:
        Список объединенных кортежей (start_frame, end_frame).
    """
    if not sequences:
        return []

    # Сортируем по началу
    sequences = sorted(sequences, key=lambda x: x[0])
    merged = []
    current_start, current_end = sequences[0]

    for start, end in sequences[1:]:
        if start <= current_end + max_frame_gap:
            # Участки пересекаются или находятся в пределах max_frame_gap, обновляем конец
            current_end = max(current_end, end)
        else:
            # Новый участок, добавляем предыдущий и начинаем новый
            merged.append((current_start, current_end))
            current_start, current_end = start, end

    # Добавляем последний объединенный участок
    merged.append((current_start, current_end))
    return merged


def find_cyclic_sequences(
    positions,
    min_cycle_amplitude=30.0,  # Минимальная амплитуда одного цикла (размах)
    max_amplitude_variation=50.0,  # Макс. отличие амплитуд между циклами
    min_num_amplitudes=4,  # Мин. число амплитуд для последовательности (~2 цикла)
):
    """Находит участки с регулярными циклическими движениями мяча (≥2 цикла),
    где амплитуды колебаний отличаются не более чем на max_amplitude_variation.
    Доработано для детекции локальных стабильных циклов (например, набивка мяча перед подачей),
    даже если общая вариация амплитуд большая — ищем подпоследовательности.

    Args:
        positions: Список позиций в формате [[x, y], frame].
        min_cycle_amplitude: Минимальный размах Y для признания цикла значимым.
        max_amplitude_variation: Максимальное различие между амплитудами циклов.
        min_num_amplitudes: Минимальное количество consecutive амплитуд для последовательности.

    Returns:
        Список кортежей (start_frame, end_frame) для стабильных циклических участков.
    """
    if not positions or len(positions) < 10:
        return []

    # Преобразуем в массив
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

        # Ищем участок с малым изменением X
        while j < n:
            x_range = np.max(x_values[i : j + 1]) - np.min(x_values[i : j + 1])
            if x_range > 150:
                break
            j += 1

        if j - i < 100:  # слишком короткий участок
            i = j
            continue

        y_segment = y_values[i:j]
        total_y_range = np.max(y_segment) - np.min(y_segment)
        if total_y_range < min_cycle_amplitude:
            i = j
            continue

        # Находим пики и впадины
        peaks, _ = find_peaks(y_segment, prominence=10)
        troughs, _ = find_peaks(-y_segment, prominence=10)

        if len(peaks) < 2 or len(troughs) < 2:
            i = j
            continue

        # Сортируем события по индексу
        events = sorted(
            [(p, "peak") for p in peaks] + [(t, "trough") for t in troughs],
            key=lambda x: x[0],
        )

        # Извлекаем амплитуды (все, без фильтра пока)
        amplitudes = []
        for k in range(1, len(events)):
            prev_idx, _ = events[k - 1]
            curr_idx, _ = events[k]
            amplitude = abs(y_segment[curr_idx] - y_segment[prev_idx])
            amplitudes.append(amplitude)

        if len(amplitudes) < min_num_amplitudes:
            i = j
            continue

        # Шаг 1: Находим "хорошие" сегменты амплитуд, где все >= min_cycle_amplitude (без малых переходов)
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

        # Шаг 2: Для каждого хорошего сегмента ищем подпоследовательности с похожими амплитудами (range <= var)
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
                    # Добавляем участок (от события left до события right+1)
                    event_left = events[left][0]
                    event_right = events[right + 1][0]
                    f_start = int(frames[i + event_left])
                    f_end = int(frames[i + event_right])
                    sequences.append((f_start, f_end))
                    # Переходим к следующему непересекающемуся
                    left = right + 1

        i = j  # переходим к следующему сегменту
    sequences = merge_sequences(sequences)

    return sequences


def find_rolling_sequences(
    positions,
    max_y_range=40.0,  # Уменьшено для трека 0005
    min_x_range=50.0,
    min_length=70,  # Уменьшено для коротких участков
):
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

    sequences = merge_sequences(sequences, max_frame_gap=30)

    return sequences
