import pandas as pd
import numpy as np
import os
import argparse
import cv2
from typing import List, Dict, Optional, Tuple
from ball_tracker import BallTracker
from scipy.signal import find_peaks


def merge_cyclic_sequences(
    sequences: List[Tuple[int, int]], max_frame_gap: int = 10
) -> List[Tuple[int, int]]:
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
    positions: List[List],
    min_cycle_amplitude: float = 30.0,  # Минимальная амплитуда одного цикла (размах)
    max_amplitude_variation: float = 50.0,  # Макс. отличие амплитуд между циклами
    min_num_amplitudes: int = 4,  # Мин. число амплитуд для последовательности (~2 цикла)
) -> List[Tuple[int, int]]:
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
    sequences = merge_cyclic_sequences(sequences)

    return sequences


def find_rolling_sequences(
    positions: List[List],
    max_y_range: float = 40.0,  # Уменьшено для трека 0005
    min_x_range: float = 50.0,
    min_length: int = 70,  # Уменьшено для коротких участков
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


class TrackAnalyzer:
    """Класс для анализа треков из CSV и визуализации на видео."""

    def __init__(
        self,
        csv_path: str,
        video_path: str,
        output_path: Optional[str] = None,  # Путь для сохранения видео (AVI)
        fps: float = 30.0,
        max_distance: float = 200,
        min_duration_sec: float = 1.0,
        max_x_displacement: float = 20.0,  # Порог перемещения по X для навеса
        min_y_displacement: float = 50.0,  # Порог перемещения по Y для навеса
        bounce_frames: int = 10,  # Количество кадров для анализа навеса
    ):
        self.csv_path = csv_path
        self.video_path = video_path
        self.output_path = output_path
        self.fps = fps
        self.max_distance = max_distance
        self.min_duration_sec = min_duration_sec
        self.max_x_displacement = max_x_displacement
        self.min_y_displacement = min_y_displacement
        self.bounce_frames = bounce_frames
        self.tracks: List[Dict] = []
        self.track_durations: Dict[int, float] = {}
        self.track_distances: Dict[int, str] = {}
        self.positions: Dict[int, List[Dict]] = {}

    def _validate_files(self) -> None:
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV не найден: {self.csv_path}")
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Видео не найдено: {self.video_path}")

    def _load_and_process_csv(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        df["Frame"] = pd.to_numeric(df["Frame"], errors="coerce")
        df["Visibility"] = pd.to_numeric(df["Visibility"], errors="coerce")
        df["X"] = pd.to_numeric(df["X"], errors="coerce")
        df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
        df.loc[(df["X"] == -1) | (df["Visibility"] == 0), ["X", "Y"]] = np.nan
        return df

    def _is_overlapping(self, track1: Dict, track2: Dict) -> bool:
        start1, end1 = track1.start_frame, track1.last_frame
        start2, end2 = track2.start_frame, track2.last_frame
        return start1 <= end2 and start2 <= end1

    def _trim_bounce_start(self, track: Dict) -> Dict:
        """Подрезает начало трека, убирая кадры с движением мяча вверх-вниз."""
        if not track.positions:
            return track

        # Сортируем позиции по кадрам
        positions = sorted(track.positions, key=lambda x: x[1])
        new_positions = []
        trim_frame = track.start_frame

        # Проверяем последовательные группы из bounce_frames кадров
        for i in range(len(positions) - self.bounce_frames + 1):
            window = positions[i : i + self.bounce_frames]
            x_coords = [pos[0][0] for pos in window]
            y_coords = [pos[0][1] for pos in window]
            frames = [pos[1] for pos in window]

            # Проверяем, что окно покрывает последовательные кадры
            if max(frames) - min(frames) + 1 > self.bounce_frames:
                continue

            # Вычисляем перемещения по X и Y
            x_displacement = max(x_coords) - min(x_coords)
            y_displacement = max(y_coords) - min(y_coords)

            # Если мяч движется вверх-вниз (большое Y, малое X), продолжаем искать
            if (
                x_displacement <= self.max_x_displacement
                and y_displacement >= self.min_y_displacement
            ):
                continue
            else:
                # Нашли момент, где мяч начинает двигаться по X
                trim_frame = frames[0]
                new_positions = positions[i:]
                break

        if not new_positions:
            new_positions = (
                positions  # Если не нашли подходящее окно, оставляем как есть
            )

        # Обновляем трек
        track.positions = new_positions
        track.start_frame = (
            min([pos[1] for pos in new_positions])
            if new_positions
            else track.start_frame
        )
        track.last_frame = (
            max([pos[1] for pos in new_positions])
            if new_positions
            else track.last_frame
        )
        duration_frames = track.last_frame - track.start_frame + 1
        track.duration_sec = lambda: duration_frames / self.fps
        if new_positions != positions:
            self.track_distances[track.track_id] = (
                f"Trimmed bounce until frame {trim_frame}"
            )

        return track

    def _filter_short_tracks(self, episodes: List[Dict]) -> List[Dict]:

        # Шаг 2: Фильтрация коротких и пересекающихся треков
        filtered_episodes = []
        used_indices = set()

        # Фильтруем треки по минимальной длительности
        long_tracks = [
            ep for ep in episodes if ep.duration_sec() >= self.min_duration_sec
        ]
        # Сортируем треки по длительности (от большего к меньшему) для фильтрации пересечений
        sorted_tracks = sorted(
            long_tracks, key=lambda x: x.duration_sec(), reverse=True
        )

        for i, track1 in enumerate(sorted_tracks):
            if i in used_indices:
                continue
            filtered_episodes.append(track1)
            used_indices.add(i)
            for j, track2 in enumerate(sorted_tracks):
                if j <= i or j in used_indices:
                    continue
                if self._is_overlapping(track1, track2):
                    used_indices.add(j)
                    print(
                        f"Удалён трек {track2.track_id} (пересекается с треком {track1.track_id})"
                    )

        # Шаг 3: Расширение треков на 1 секунду в обе стороны
        frames_to_extend = int(self.fps)  # Количество кадров за 1 секунду
        extended_episodes = []
        for ep in filtered_episodes:
            ep.start_frame = max(0, ep.start_frame - frames_to_extend)
            ep.last_frame = ep.last_frame + frames_to_extend
            extended_episodes.append(ep)

        # Шаг 4: Объединение пересекающихся треков после расширения
        merged_episodes = []
        used_indices = set()
        sorted_extended = sorted(extended_episodes, key=lambda x: x.start_frame)

        for i, track1 in enumerate(sorted_extended):
            if i in used_indices:
                continue
            merged_track = track1
            merged_positions = list(merged_track.positions)
            merged_track_ids = [merged_track.track_id]
            used_indices.add(i)

            for j, track2 in enumerate(sorted_extended):
                if j <= i or j in used_indices:
                    continue
                if self._is_overlapping(merged_track, track2):
                    merged_track.start_frame = min(
                        merged_track.start_frame, track2.start_frame
                    )
                    merged_track.last_frame = max(
                        merged_track.last_frame, track2.last_frame
                    )
                    merged_positions.extend(track2.positions)
                    merged_track_ids.append(track2.track_id)
                    used_indices.add(j)
                    print(
                        f"Объединён трек {track2.track_id} с треком {merged_track.track_id}"
                    )

            # Обновляем позиции и длительность объединённого трека
            merged_track.positions = sorted(
                merged_positions, key=lambda x: x[1]
            )  # Сортировка по кадрам
            duration_frames = merged_track.last_frame - merged_track.start_frame + 1
            merged_track.duration_sec = (
                lambda: duration_frames / self.fps
            )  # Обновляем метод duration_sec
            self.track_distances[merged_track.track_id] = (
                f"Merged tracks: {', '.join(map(str, merged_track_ids))}"
                if len(merged_track_ids) > 1
                else self.track_distances.get(merged_track.track_id, "Unknown")
            )
            merged_episodes.append(merged_track)

        # Шаг 1: Подрезка треков с навесом мяча
        #merged_episodes = [self._trim_bounce_start(ep) for ep in merged_episodes]

        # Сортируем по начальным кадрам для корректной визуализации
        return sorted(merged_episodes, key=lambda x: x.start_frame)

    def _process_detections(self, df: pd.DataFrame) -> None:
        tracker = BallTracker(
            buffer_size=2500, max_disappeared=40, max_distance=self.max_distance, fps=self.fps
        )
        close_tracks = []
        for frame_num in sorted(df["Frame"].dropna().astype(int).unique()):
            frame_rows = df[df["Frame"] == frame_num]
            detections = []
            for _, row in frame_rows.iterrows():
                if not np.isnan(row["X"]) and not np.isnan(row["Y"]):
                    detections.append(
                        {
                            "x1": row["X"] - 20,
                            "y1": row["Y"] - 20,
                            "x2": row["X"] + 20,
                            "y2": row["Y"] + 20,
                            "confidence": row["Visibility"],
                            "cls_id": 0,
                        }
                    )
            _, tracks_dict, close_track = tracker.update(detections, frame_num)
            close_tracks.extend(close_track)

        episodes = []
        for track in close_tracks:
            frames = [f.item() for _, f in track.positions]
            if not frames:
                continue
            self.track_distances[track.track_id] = (
                f"Distance to nearest track > {self.max_distance} pixels"
                if track.reason == "Unknown"
                else track.reason
            )
            start_frame = track.start_frame.item()
            end_frame = track.last_frame.item()
            duration_frames = end_frame - start_frame + 1
            duration_sec = duration_frames / self.fps
            episodes.append(track)
        self.tracks = self._filter_short_tracks(episodes)

    def _print_track_info(self) -> None:
        print("\n=== Список треков ===")
        print(f"Минимальная длительность трека: {self.min_duration_sec} сек")
        print(f"Найдено треков: {len(self.tracks)}")
        print("ID\tStart Frame\tLast Frame\tDuration (sec)\tLength (frames)")
        print("-" * 60)
        for track in self.tracks:
            distance_reason = self.track_distances.get(track.track_id, "Unknown")
            print(
                f"{track.track_id}\t{track.start_frame}\t\t{track.last_frame}\t\t"
                f"{track.duration_sec():.2f}\t\t{track.size()}\t({distance_reason})"
            )
        if not self.tracks:
            print("Нет треков, удовлетворяющих критериям.")

    def _save_tacks_to_file(self, track) -> None:
        import json
        file_path =  f'track_json/track_{track.track_id:04d}.json'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            f.write(json.dumps(track.to_dict()) + "\n")

    def visualize_tracks(self) -> None:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {self.video_path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or self.fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Инициализация VideoWriter для AVI, если output_path задан
        out = None
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

        # Для каждого трека показываем только нужный отрывок
        for track in self.tracks:
            track_id = track.track_id
            start_frame = track.start_frame
            end_frame = track.last_frame
            self._save_tacks_to_file(track)
            sequences = find_cyclic_sequences(track.positions)
            if sequences:
                print("Найдены последовательности набивания мяча:")
                for start, end in sequences:
                    print(f"Начало: кадр {start}, Конец: кадр {end}")

                    track.start_frame = end
                    duration_frames = track.last_frame - track.start_frame + 1
                    track.duration_sec = lambda: duration_frames / self.fps
                    start_frame = track.start_frame

            else:
                print("Последовательности набивания мяча не найдены.")

            sequences = find_rolling_sequences(track.positions)
            if sequences:
                print("Найдены последовательности качения мяча:")
                for start, end in sequences:
                    print(f"Начало: кадр {start}, Конец: кадр {end}")
                    track.last_frame = start
                    duration_frames = track.last_frame - track.start_frame + 1
                    track.duration_sec = lambda: duration_frames / self.fps
                    end_frame = track.last_frame
                    break

            # Перемотать к start_frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_num = start_frame

            while frame_num <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                clean_frame = (
                    frame.copy()
                )  # Копия кадра без отладочной информации для сохранения
                for _pos in track.positions:
                    pos = {"x": int(_pos[0][0]), "y": int(_pos[0][1]), "frame_num": _pos[1]}
                    if pos["frame_num"] == frame_num:
                        x, y = int(pos["x"]), int(pos["y"])
                        # Рисуем на обоих кадрах
                        cv2.circle(frame, (x, y), 10, (0, 255, 255), -1)
                        # cv2.circle(clean_frame, (x, y), 10, (0, 255, 255), -1)
                        time_from_start = (frame_num - start_frame) / fps
                        text = f"ID: {track_id}, {pos['x']},{pos['y']} Time: {time_from_start:.2f}s"
                        cv2.putText(
                            frame,
                            text,
                            (x + 15, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )
                        # --- NEW: draw PRESERV if frame_num in any sequence ---
                        if any(start <= frame_num <= end for start, end in sequences):
                            cv2.putText(
                                frame,
                                "PRESERV",
                                (x, y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 255, 0),
                                2,
                                cv2.LINE_AA,
                            )
                # Добавляем отладочную информацию только для отображения
                if not self.output_path:
                    debug_info = (
                        f"Frame: {frame_num}/{total_frames}, Track ID: {track_id}"
                    )
                    cv2.putText(
                        frame,
                        debug_info,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("Track Visualization", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                # Сохраняем кадр без отладочной информации, если output_path задан
                if out:
                    out.write(clean_frame)
                frame_num += 1

        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        if self.output_path:
            print(f"Видео сохранено: {self.output_path}")
        else:
            print("Визуализация завершена")

    def run(self) -> None:
        self._validate_files()
        df = self._load_and_process_csv()
        self._process_detections(df)
        self._print_track_info()
        self.visualize_tracks()


def main():
    parser = argparse.ArgumentParser(
        description="Анализ треков из CSV и визуализация на видео."
    )
    parser.add_argument("--csv_path", type=str, required=True, help="Путь к ball.csv")
    parser.add_argument("--video_path", type=str, required=True, help="Путь к видео")
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Путь для сохранения выходного видео в формате AVI",
    )
    parser.add_argument(
        "--fps", type=float, default=30, help="FPS видео (по умолчанию 30)"
    )
    parser.add_argument(
        "--max_distance",
        type=float,
        default=200,
        help="Максимальное расстояние для трекинга",
    )
    parser.add_argument(
        "--min_duration_sec",
        type=float,
        default=1.0,
        help="Минимальная длительность трека (сек)",
    )
    parser.add_argument(
        "--max_x_displacement",
        type=float,
        default=20.0,
        help="Максимальное перемещение по X для определения навеса (пиксели)",
    )
    parser.add_argument(
        "--min_y_displacement",
        type=float,
        default=50.0,
        help="Минимальное перемещение по Y для определения навеса (пиксели)",
    )
    parser.add_argument(
        "--bounce_frames",
        type=int,
        default=10,
        help="Количество кадров для анализа навеса",
    )
    args = parser.parse_args()
    analyzer = TrackAnalyzer(
        csv_path=args.csv_path,
        video_path=args.video_path,
        output_path=args.output_path,
        fps=args.fps,
        max_distance=args.max_distance,
        min_duration_sec=args.min_duration_sec,
        max_x_displacement=args.max_x_displacement,
        min_y_displacement=args.min_y_displacement,
        bounce_frames=args.bounce_frames,
    )
    analyzer.run()


if __name__ == "__main__":
    main()
