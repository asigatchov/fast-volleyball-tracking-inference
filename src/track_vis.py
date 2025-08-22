import pandas as pd
import numpy as np
import os
import argparse
import cv2
from typing import List, Dict, Optional
from ball_tracker import BallTracker


class TrackAnalyzer:
    """Класс для анализа треков из CSV и визуализации на видео."""

    def __init__(
        self,
        csv_path: str,
        video_path: str,
        output_video_path: str = "output_video.mp4",
        fps: float = 30.0,
        max_distance: float = 200,
        min_duration_sec: float = 1.0,
    ):
        self.csv_path = csv_path
        self.video_path = video_path
        self.output_video_path = output_video_path
        self.fps = fps
        self.max_distance = max_distance
        self.min_duration_sec = min_duration_sec
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

    def _filter_short_tracks(self, episodes: List[Dict]) -> List[Dict]:
        # Фильтруем треки по минимальной длительности
        long_tracks = [
            ep for ep in episodes if ep.duration_sec() >= self.min_duration_sec
        ]

        # Список для хранения треков, которые нужно оставить
        filtered_episodes = []
        used_indices = set()

        # Сортируем треки по длительности (от большего к меньшему)
        sorted_tracks = sorted(long_tracks, key=lambda x: x.duration_sec(), reverse=True)

        for i, track1 in enumerate(sorted_tracks):
            if i in used_indices:
                continue

            # Добавляем текущий трек в результат
            filtered_episodes.append(track1)
            used_indices.add(i)

            # Проверяем пересечения с другими треками
            for j, track2 in enumerate(sorted_tracks):
                if j <= i or j in used_indices:
                    continue

                # Проверяем, пересекаются ли треки по кадрам
                if self._is_overlapping(track1, track2):
                    # Если треки пересекаются, исключаем более короткий (track2)
                    used_indices.add(j)
                    print(f"Удалён трек {track2.track_id} (пересекается с треком {track1.track_id})")

        # Сортируем по начальным кадрам для корректной визуализации
        return sorted(filtered_episodes, key=lambda x: x.start_frame)

    def _process_detections(self, df: pd.DataFrame) -> None:
        tracker = BallTracker(
            buffer_size=1500, max_disappeared=40, max_distance=self.max_distance
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
            episodes.append(
              track
            )
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

    def visualize_tracks(self) -> None:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {self.video_path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or self.fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        #out = cv2.VideoWriter(self.output_video_path, fourcc, fps, (width, height))

        # Для каждого трека показываем только нужный отрывок
        for track in self.tracks:
            track_id = track.track_id
            start_frame = track.start_frame
            end_frame = track.last_frame

            # Перемотать к start_frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_num = start_frame

            while frame_num <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                # Нарисовать только для текущего трека
                for _pos in track.positions:
                    pos = {
                        'x': _pos[0][0],
                        'y': _pos[0][1],
                        'frame_num': _pos[1]
                    }
                    if pos["frame_num"] == frame_num:
                        x, y = int(pos["x"]), int(pos["y"])
                        cv2.circle(frame, (x, y), 10, (0, 255, 255), -1)
                        time_from_start = (frame_num - start_frame) / fps
                        text = f"ID: {track_id}, Time: {time_from_start:.2f}s"
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
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    print(f"Видео сохранено: {self.output_video_path}")
                    return
                #out.write(frame)
                frame_num += 1

        cap.release()
        #out.release()
        cv2.destroyAllWindows()
        print(f"Видео сохранено: {self.output_video_path}")

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
        "--output_video",
        type=str,
        default="output_video.mp4",
        help="Путь к выходному видео",
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
    args = parser.parse_args()
    analyzer = TrackAnalyzer(
        csv_path=args.csv_path,
        video_path=args.video_path,
        output_video_path=args.output_video,
        fps=args.fps,
        max_distance=args.max_distance,
        min_duration_sec=args.min_duration_sec,
    )
    analyzer.run()


if __name__ == "__main__":
    main()
