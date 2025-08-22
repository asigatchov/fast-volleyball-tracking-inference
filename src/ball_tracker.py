import numpy as np
from collections import deque
from scipy.spatial import distance
from dataclasses import dataclass, field
import json
from typing import List, Tuple, Dict, Optional, Any
import dataclasses


@dataclass
class Track:
    positions: deque = field(default_factory=lambda: deque(maxlen=1500))
    prediction: List[float] = field(default_factory=list)
    last_frame: int = 0
    start_frame: int = 0
    ball_sizes: deque = field(default_factory=lambda: deque(maxlen=1500))
    track_id: int = 0
    reason: str = "Unknown"
    fps: float = 30.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "positions": list(self.positions),
            "prediction": self.prediction,
            "last_frame": self.last_frame,
            "start_frame": self.start_frame,
            "ball_sizes": list(self.ball_sizes),
            "track_id": self.track_id,
        }

    def size(self) -> int:
        # Возвращает разницу между last_frame и start_frame
        return self.last_frame - self.start_frame

    def duration_sec(self) -> float:
        # Возвращает длительность трека в секундах
        sz = self.size()
        return sz / self.fps if self.fps > 0 else 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any], buffer_size: int = 1500) -> "Track":
        track = cls()
        track.positions = deque(data["positions"], maxlen=buffer_size)
        track.prediction = data["prediction"]
        track.last_frame = data["last_frame"]
        track.start_frame = data["start_frame"]
        track.ball_sizes = deque(data.get("ball_sizes", []), maxlen=buffer_size)
        track.track_id = data.get("track_id", 0)
        return track


class BallTracker:
    def __init__(
        self,
        buffer_size=1500,
        max_disappeared=40,
        max_distance=200,
        ball_diameter_cm=21.0,
    ):
        self.next_id = 0
        self.tracks: Dict[int, Track] = {}
        self.buffer_size = buffer_size
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.ball_diameter_cm = ball_diameter_cm

    def box_to_position(self, box):
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        diameter = max(x2 - x1, y2 - y1)
        return center_x, center_y, diameter

    def update(self, detections, frame_number):
        deleted_tracks = []
        for track_id in list(self.tracks.keys()):
            last_frame = self.tracks[track_id].last_frame
            if (frame_number - last_frame) > self.max_disappeared:
                deleted_tracks.append(self.tracks[track_id])
                del self.tracks[track_id]

        active_tracks = list(self.tracks.items())
        unused_detections = list(detections)

        distance_matrix = np.zeros((len(active_tracks), len(unused_detections)))
        for i, (track_id, track) in enumerate(active_tracks):
            if len(track.positions) > 0:
                last_pos = track.positions[-1][0:2]
                last_pos = track.prediction
                for j, det in enumerate(unused_detections):
                    center_x, center_y, diameter = self.box_to_position(det)
                    det_pos = [center_x, center_y]
                    distance_matrix[i, j] = distance.euclidean(
                        last_pos[0:2], det_pos[0:2]
                    )

        matched_pairs = []
        used_detection_indices = set()
        while True:
            if distance_matrix.size == 0 or np.all(np.isinf(distance_matrix)):
                break

            min_val = np.min(distance_matrix)
            if min_val > self.max_distance:
                break

            i, j = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
            track_id, _ = active_tracks[i]
            det = unused_detections[j]

            self._update_track(track_id, det, frame_number)
            matched_pairs.append((track_id, j))
            used_detection_indices.add(j)

            distance_matrix[i, :] = np.inf
            distance_matrix[:, j] = np.inf

        for j, det in enumerate(unused_detections):
            if j not in used_detection_indices:
                reason = (
                    "No active tracks"
                    if not active_tracks
                    else f"Distance to nearest track > {self.max_distance} pixels; min_val = {min_val:.2f}"
                )
                self._add_track(det, frame_number, reason)

        return self._get_main_ball(deleted_tracks)

    def _add_track(self, detection, frame_number, reason="Unknown"):
        track = Track()
        track.track_id = self.next_id
        center_x, center_y, diameter = self.box_to_position(detection)
        position = [center_x, center_y]

        track.positions = deque([(position, frame_number)], maxlen=self.buffer_size)
        track.prediction = position
        track.last_frame = frame_number
        track.start_frame = frame_number
        track.ball_sizes = deque([diameter], maxlen=self.buffer_size)
        track.reason = reason
        self.tracks[self.next_id] = track
        print(
            f"New track {self.next_id} created at frame {frame_number}, position ({center_x:.1f}, {center_y:.1f}), reason: {reason}"
        )
        self.next_id += 1

    def _update_track(self, track_id, detection, frame_number):
        center_x, center_y, diameter = self.box_to_position(detection)
        position = [center_x, center_y]

        self.tracks[track_id].positions.append((position, frame_number))
        self.tracks[track_id].last_frame = frame_number
        self.tracks[track_id].ball_sizes.append(diameter)

        if len(self.tracks[track_id].positions) > 1:
            prev_pos, prev_frame = self.tracks[track_id].positions[-2]
            dt = frame_number - prev_frame
            if dt == 0:
                dx = dy = 0
            else:
                dx = (position[0] - prev_pos[0]) / dt
                dy = (position[1] - prev_pos[1]) / dt
            self.tracks[track_id].prediction = [position[0] + dx, position[1] + dy]
        else:
            self.tracks[track_id].prediction = position

    def _get_main_ball(self, deleted_tracks):
        main_ball = None
        max_score = -1

        for track_id, track in self.tracks.items():
            positions = [p for p, _ in track.positions]
            if len(positions) < 3:
                continue

            time_steps = [f for _, f in track.positions]
            velocities = []
            for i in range(1, len(positions)):
                dt = time_steps[i] - time_steps[i - 1]
                dx = positions[i][0] - positions[i - 1][0]
                dy = positions[i][1] - positions[i - 1][1]
                velocities.append((dx / dt, dy / dt))

            var = np.var(velocities, axis=0)
            stability = 1 / (np.sum(var) + 1e-5)
            length_weight = np.log(len(positions) + 1)
            total_score = stability * length_weight

            if total_score > max_score:
                max_score = total_score
                main_ball = track_id

        tracks_dict = {
            track_id: track.to_dict() for track_id, track in self.tracks.items()
        }
        return main_ball, tracks_dict, deleted_tracks

    def to_json(self) -> str:
        data = {
            "next_id": self.next_id,
            "tracks": {
                str(track_id): track.to_dict()
                for track_id, track in self.tracks.items()
            },
            "buffer_size": self.buffer_size,
            "max_disappeared": self.max_disappeared,
            "max_distance": self.max_distance,
        }
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> "BallTracker":
        data = json.loads(json_str)
        tracker = cls(
            buffer_size=data["buffer_size"],
            max_disappeared=data["max_disappeared"],
            max_distance=data["max_distance"],
        )
        tracker.next_id = data["next_id"]

        for track_id_str, track_data in data["tracks"].items():
            track_id = int(track_id_str)
            tracker.tracks[track_id] = Track.from_dict(
                track_data, buffer_size=tracker.buffer_size
            )

        return tracker
