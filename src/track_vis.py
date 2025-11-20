import pandas as pd
import numpy as np
import os
import argparse
import cv2
import json
import hashlib
import pickle
from typing import List, Dict, Optional, Tuple
from ball_tracker import BallTracker, Track
from track_utils import find_cyclic_sequences, find_rolling_sequences

class TrackAnalyzer:
    """Класс для анализа треков из CSV и визуализации на видео."""
    def __init__(
        self,
        csv_path: str,
        video_path: str,
        output_path: Optional[str] = None,
        fps: float = 30.0,
        max_distance: float = 200,
        min_duration_sec: float = 1.0,
        max_x_displacement: float = 20.0,
        min_y_displacement: float = 50.0,
        bounce_frames: int = 10,
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
        self.cache_file = "track_cache.pkl"
    
    def _validate_files(self) -> None:
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV не найден: {self.csv_path}")
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Видео не найдено: {self.video_path}")
    
    def _get_cache_key(self) -> str:
        """Создает хэш на основе параметров и содержимого CSV-файла."""
        params = {
            "csv_path": self.csv_path,
            "video_path": self.video_path,
            "fps": self.fps,
            "max_distance": self.max_distance,
            "min_duration_sec": self.min_duration_sec,
            "max_x_displacement": self.max_x_displacement,
            "min_y_displacement": self.min_y_displacement,
            "bounce_frames": self.bounce_frames,
        }
        params_str = json.dumps(params, sort_keys=True)
        
        # Хэшируем содержимое CSV-файла
        with open(self.csv_path, "rb") as f:
            csv_hash = hashlib.md5(f.read()).hexdigest()
        
        # Комбинируем параметры и хэш CSV
        combined = params_str + csv_hash
        return hashlib.md5(combined.encode()).hexdigest()

    def _load_cache(self) -> Optional[Dict]:
        """Загружает данные из кэша, если они существуют и валидны."""
        if not os.path.exists(self.cache_file):
            return None
    
        try:
            with open(self.cache_file, "rb") as f:
                cache = pickle.load(f)
                cache_key = self._get_cache_key()
                if cache_key in cache:
                    return cache[cache_key]
        except (pickle.UnpicklingError, EOFError, AttributeError, KeyError, ImportError) as e:
            print(f"⚠️  Ошибка при загрузке кэша: {e}. Кэш будет пересоздан.")
            # Не возвращаем данные — заставим пересчитать
        return None
    
    def _save_cache(self, data: Dict) -> None:
        """Сохраняет данные в кэш."""
        cache = {}

        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "rb") as f:
                    loaded = pickle.load(f)
                    if isinstance(loaded, dict):
                        cache = loaded
                    else:
                        print("⚠️  Кэш имеет неверный формат, будет пересоздан.")
            except (EOFError, pickle.UnpicklingError, AttributeError, ValueError) as e:
                print(f"⚠️  Ошибка при чтении кэша для записи: {e}. Кэш будет пересоздан.")
                # Оставляем cache = {} — начнём с чистого листа

        cache[self._get_cache_key()] = data

        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(cache, f)
        except Exception as e:
            print(f"❌ Не удалось сохранить кэш: {e}")
    
    
    def _load_and_process_csv(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        df["Frame"] = pd.to_numeric(df["Frame"], errors="coerce")
        df["Visibility"] = pd.to_numeric(df["Visibility"], errors="coerce")
        df["X"] = pd.to_numeric(df["X"], errors="coerce")
        df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
        df.loc[(df["X"] == -1) | (df["Visibility"] == 0), ["X", "Y"]] = np.nan
        return df
    
    def _is_overlapping(self, track1: Track, track2: Dict) -> bool:
        start1, end1 = track1.start_frame, track1.last_frame
        start2, end2 = track2.start_frame, track2.last_frame
        return start1 <= end2 and start2 <= end1
    
    def _trim_bounce_start(self, track: Dict) -> Dict:
        """Подрезает начало трека, убирая кадры с движением мяча вверх-вниз."""
        if not track.positions:
            return track
        orig_start = track.start_frame
        orig_end = track.last_frame
        sequences = find_cyclic_sequences(track.positions)
        if sequences:
            print("Найдены последовательности набивания мяча:")
            for start, end in sequences:
                print(f"Начало: кадр {start}, Конец: кадр {end}")
                track.start_frame = end
                duration_frames = track.last_frame - track.start_frame + 1
                track.duration_sec = lambda: duration_frames / self.fps
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
                break
        if track.start_frame != orig_start or track.last_frame != orig_end:
            track.positions = [
                pos for pos in track.positions
                if track.start_frame <= pos[1] <= track.last_frame
            ]
        return track
    
    def _filter_short_tracks(self, episodes: List[Dict]) -> List[Dict]:
        filtered_episodes = []
        used_indices = set()
        episodes = [
            ep for ep in episodes if ep.get_y_range() >= 1080 / 4.0
        ]
        episodes = [self._trim_bounce_start(ep) for ep in episodes]
        long_tracks = [
            ep for ep in episodes if ep.duration_sec() >= self.min_duration_sec
        ]
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
        frames_to_extend = int(self.fps)
        extended_episodes = []
        for ep in filtered_episodes:
            ep.start_frame = max(0, ep.start_frame - frames_to_extend)
            ep.last_frame = ep.last_frame + frames_to_extend
            extended_episodes.append(ep)
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
            merged_track.positions = sorted(
                merged_positions, key=lambda x: x[1]
            )
            duration_frames = merged_track.last_frame - merged_track.start_frame + 1
            merged_track.duration_sec = (
                lambda: duration_frames / self.fps
            )
            self.track_distances[merged_track.track_id] = (
                f"Merged tracks: {', '.join(map(str, merged_track_ids))}"
                if len(merged_track_ids) > 1
                else self.track_distances.get(merged_track.track_id, "Unknown")
            )
            merged_episodes.append(merged_track)
        merged_episodes = [self._trim_bounce_start(ep) for ep in merged_episodes]
        return sorted(merged_episodes, key=lambda x: x.start_frame)
    
    def _process_detections(self, df: pd.DataFrame) -> None:
        # Проверяем кэш
        cached_data = self._load_cache()
        if cached_data:
            self.tracks = cached_data["tracks"]
            self.track_distances = cached_data["track_distances"]
            print("Загружены данные из кэша.")
            return
        
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
        
        # Сохраняем в кэш
        cache_data = {
            "tracks": self.tracks,
            "track_distances": self.track_distances
        }
        self._save_cache(cache_data)
    
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
        """Сохраняет трек в JSON-файл в подкаталог, соответствующий имени видеофайла."""
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        file_path = f'track_json/{video_name}/track_{track.track_id:04d}.json'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(track.to_dict(), f, indent=2)
    
    def visualize_tracks(self) -> None:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {self.video_path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or self.fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out = None
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
        for track in self.tracks:
            track_id = track.track_id
            start_frame = track.start_frame
            end_frame = track.last_frame
            self._save_tacks_to_file(track)
            fade_duration_frames = int(fps * 0.5)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_num = start_frame
            while frame_num <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                clean_frame = frame.copy()
                for _pos in track.positions:
                    pos = {"x": int(_pos[0][0]), "y": int(_pos[0][1]), "frame_num": _pos[1]}
                    if pos["frame_num"] == frame_num:
                        x, y = int(pos["x"]), int(pos["y"])
                        cv2.circle(frame, (x, y), 10, (0, 255, 255), -1)
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
                if out:
                    out.write(clean_frame)
                frame_num += 1
            if out and fade_duration_frames > 0:
                last_frame = clean_frame.copy()
                for fade_step in range(fade_duration_frames):
                    alpha = 1.0 - (fade_step / fade_duration_frames)
                    faded_frame = cv2.convertScaleAbs(last_frame, alpha=alpha, beta=0)
                    out.write(faded_frame)
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
