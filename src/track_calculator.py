#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import argparse
import json
import cv2
from typing import List, Optional, Tuple, Dict
from ball_tracker import BallTracker, Track
from track_utils import find_cyclic_sequences, find_rolling_sequences


class TrackCalculator:
    def __init__(
        self,
        csv_path: str,
        court_json_path: Optional[str] = None,
        output_dir: str = "output",
        fps: float = 30.0,
        max_distance: float = 200.0,
        min_duration_sec: float = 1.0,
        max_x_displacement: float = 20.0,
        min_y_displacement: float = 50.0,
        bounce_frames: int = 10,
    ):
        self.court_json_path = court_json_path
        self.court_points = None
        self.image_width = None
        self.image_height = None
        self.court_transform_matrix = None
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.fps = fps
        self.max_distance = max_distance
        self.min_duration_sec = min_duration_sec
        self.max_x_displacement = max_x_displacement
        self.min_y_displacement = min_y_displacement
        self.bounce_frames = bounce_frames
        self.tracks: List[Track] = []
        self.track_distances: dict[int, str] = {}
        
        # Load court coordinates if provided
        if self.court_json_path:
            self._load_court_coordinates()

    def _validate_csv(self) -> None:
        """Check if the CSV file exists."""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
    
    def _load_court_coordinates(self) -> None:
        """Load volleyball court coordinates from JSON file and calculate transformation matrix."""
        if not self.court_json_path or not os.path.exists(self.court_json_path):
            print("Warning: Court JSON file not provided or not found. Using pixel coordinates.")
            return
        
        try:
            with open(self.court_json_path, 'r') as f:
                court_data = json.load(f)
            
            # Extract image dimensions
            images = court_data.get('images', [])
            if images:
                self.image_width = images[0].get('width', 1280)
                self.image_height = images[0].get('height', 720)
            
            # Extract court keypoints
            annotations = court_data.get('annotations', [])
            if not annotations:
                print("Warning: No annotations found in court JSON")
                return
            
            keypoints = annotations[0].get('keypoints', [])
            if len(keypoints) < 24:  # 8 points × 3 (x,y,visibility)
                print(f"Warning: Insufficient keypoints found ({len(keypoints)}), need at least 24")
                return
            
            # Parse keypoints: [x1,y1,v1,x2,y2,v2,...]
            self.court_points = []
            for i in range(0, len(keypoints), 3):
                if i + 2 < len(keypoints):
                    x, y, visibility = keypoints[i], keypoints[i+1], keypoints[i+2]
                    if visibility > 0:  # Only use visible points
                        self.court_points.append([x, y])
            
            print(f"Loaded {len(self.court_points)} court keypoints")
            
            # Calculate court coordinate system
            # Points mapping:
            # 0,1: back corners (left, right)
            # 4,5: center line points
            # 6,7: net top edge points
            
            if len(self.court_points) >= 8:
                self._calculate_court_transform()
            
        except Exception as e:
            print(f"Warning: Failed to load court coordinates: {e}")
    
    def _calculate_court_transform(self) -> None:
        """Calculate transformation matrix from image coordinates to court coordinates."""
        if not self.court_points or len(self.court_points) < 8:
            return
        
        # Define court dimensions (standard volleyball court)
        # Court is 18m long × 9m wide
        COURT_LENGTH = 18.0  # meters
        COURT_WIDTH = 9.0    # meters
        NET_HEIGHT = 2.43    # meters (men's)
        
        # Image points (from court keypoints)
        img_points = np.array([
            self.court_points[0],  # back left
            self.court_points[1],  # back right
            self.court_points[2],  # front right
            self.court_points[3],  # front left
        ], dtype=np.float32)
        
        # Court points in meters (assuming origin at center back line)
        court_points = np.array([
            [-COURT_LENGTH/2, -COURT_WIDTH/2],  # back left
            [COURT_LENGTH/2, -COURT_WIDTH/2],   # back right
            [COURT_LENGTH/2, COURT_WIDTH/2],    # front right
            [-COURT_LENGTH/2, COURT_WIDTH/2],   # front left
        ], dtype=np.float32)
        
        # Calculate perspective transform
        try:
            self.court_transform_matrix = cv2.getPerspectiveTransform(img_points, court_points)
            print("Court coordinate transformation calculated successfully")
        except Exception as e:
            print(f"Warning: Could not calculate court transform: {e}")
    
    def _transform_point_to_court(self, x: float, y: float) -> Tuple[float, float]:
        """Transform image coordinates to court coordinates (meters)."""
        if self.court_transform_matrix is None:
            # Return normalized coordinates if no transform available
            norm_x = x / (self.image_width or 1280)
            norm_y = y / (self.image_height or 720)
            return (norm_x * 18.0 - 9.0, norm_y * 9.0 - 4.5)  # Scale to court dimensions
        
        # Apply perspective transform
        point = np.array([[x, y]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point.reshape(-1, 1, 2), self.court_transform_matrix)
        return (float(transformed[0][0][0]), float(transformed[0][0][1]))

    def _load_and_process_csv(self) -> pd.DataFrame:
        """Load CSV and preprocess columns."""
        df = pd.read_csv(self.csv_path)
        df["Frame"] = pd.to_numeric(df["Frame"], errors="coerce")
        df["Visibility"] = pd.to_numeric(df["Visibility"], errors="coerce")
        df["X"] = pd.to_numeric(df["X"], errors="coerce")
        df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
        df.loc[(df["X"] == -1) | (df["Visibility"] == 0), ["X", "Y"]] = np.nan
        return df

    def _is_overlapping(self, track1: Track, track2: Track) -> bool:
        """Check if two tracks overlap in time."""
        return (
            track1.start_frame <= track2.last_frame
            and track2.start_frame <= track1.last_frame
        )

    def _trim_bounce_start(self, track: Track) -> Track:
        """Remove cyclic start and rolling end segments from the track."""
        if not track.positions:
            return track

        # Remove cyclic sequences at the beginning
        sequences = find_cyclic_sequences(track.positions)
        if sequences:
            for start, end in sequences:
                track.start_frame = end
                break

        # Remove rolling sequences at the end
        sequences = find_rolling_sequences(track.positions)
        if sequences:
            for start, end in sequences:
                track.last_frame = start
                break

        # Trim positions if start/end frames were updated
        if (
            track.start_frame != track.start_frame
            or track.last_frame != track.last_frame
        ):
            track.positions = [
                pos
                for pos in track.positions
                if track.start_frame <= pos[1] <= track.last_frame
            ]
        return track


    def _over_net_level(self, track: Track) -> bool:
        """Check if the ball rises above the net level during the track.
        
        Args:
            track: Track object containing ball positions
            
        Returns:
            bool: True if ball reaches above net level, False otherwise
        """
        if not self.court_points or len(self.court_points) < 8:
            # If no court points available, use a default threshold
            # Net is typically around y=160-162 in the image coordinates
            y_positions = [pos[0][1] for pos in track.positions]
            min_y = min(y_positions)  # Lower y = higher position
            return min_y < 200  # Threshold for tracks that go reasonably high
        
        # Get net top points (points 6 and 7)
        net_left_y = self.court_points[6][1]   # [313, 162]
        net_right_y = self.court_points[7][1]  # [861, 160]
        net_top_y = min(net_left_y, net_right_y)  # Use the higher point (lower y-value)
        
        # Check if any ball position is above the net
        y_positions = [pos[0][1] for pos in track.positions]
        min_ball_y = min(y_positions)  # Minimum y = highest position
        
        # Return True if ball goes above net level
        return min_ball_y < net_top_y
    def _filter_short_tracks(self, episodes: List[Track]) -> List[Track]:
        """Filter, extend, merge, and clean up tracks."""
        # 1. Filter by vertical range
        #episodes = [ep for ep in episodes if ep.get_y_range() >= 1080 / 4.0]
        episodes = [self._trim_bounce_start(ep) for ep in episodes]

       
        # 3. Keep only tracks longer than minimum duration
        long_tracks = [
            ep for ep in episodes if ep.duration_sec() >= self.min_duration_sec
        ]
       


        sorted_tracks = sorted(
            long_tracks, key=lambda x: x.duration_sec(), reverse=True
        )

        # 4. Remove overlapping tracks (keep longest)
        filtered = []
        used = set()
        for i, track1 in enumerate(sorted_tracks):
            if i in used:
                continue
            filtered.append(track1)
            used.add(i)
            for j, track2 in enumerate(sorted_tracks):
                if j <= i or j in used:
                    continue
                if self._is_overlapping(track1, track2):
                    used.add(j)

        # 5. Extend each track by 1 second on both sides
        frames_to_extend = int(self.fps)
        extended = []
        for ep in filtered:
            ep.start_frame = max(0, ep.start_frame - frames_to_extend)
            ep.last_frame = ep.last_frame + frames_to_extend
            extended.append(ep)

        # 6. Merge overlapping extended tracks
        merged = []
        used = set()
        sorted_ext = sorted(extended, key=lambda x: x.start_frame)
        for i, track1 in enumerate(sorted_ext):
            if i in used:
                continue
            merged_track = track1
            merged_positions = list(merged_track.positions)
            merged_ids = [merged_track.track_id]
            used.add(i)
            for j, track2 in enumerate(sorted_ext):
                if j <= i or j in used:
                    continue
                if self._is_overlapping(merged_track, track2):
                    merged_track.start_frame = min(
                        merged_track.start_frame, track2.start_frame
                    )
                    merged_track.last_frame = max(
                        merged_track.last_frame, track2.last_frame
                    )
                    merged_positions.extend(track2.positions)
                    merged_ids.append(track2.track_id)
                    used.add(j)
            merged_track.positions = sorted(merged_positions, key=lambda x: x[1])
            duration_frames = merged_track.last_frame - merged_track.start_frame + 1
            merged_track.duration_sec = lambda: duration_frames / self.fps
            self.track_distances[merged_track.track_id] = (
                f"Merged: {', '.join(map(str, merged_ids))}"
                if len(merged_ids) > 1
                else self.track_distances.get(merged_track.track_id, "Unknown")
            )
            merged.append(merged_track)

        # 7. Final trim after merging
        merged = [self._trim_bounce_start(ep) for ep in merged]
        # merged = [ep for ep in merged if self._over_net_level(ep)]

        return sorted(merged, key=lambda x: x.start_frame)

    def _process_detections(self, df: pd.DataFrame) -> None:
        """Run tracker and collect completed tracks."""
        tracker = BallTracker(
            buffer_size=2500,
            max_disappeared=40,
            max_distance=self.max_distance,
            fps=self.fps,
        )
        close_tracks = []

        all_frames = sorted(df["Frame"].dropna().astype(int).unique())
        for frame_num in all_frames:
            frame_rows = df[df["Frame"] == frame_num]
            detections = []
            for _, row in frame_rows.iterrows():
                if not np.isnan(row["X"]) and not np.isnan(row["Y"]):
                    detections.append(
                        {
                            "x1": row["X"] - 10,
                            "y1": row["Y"] - 10,
                            "x2": row["X"] + 10,
                            "y2": row["Y"] + 10,
                            "confidence": row["Visibility"],
                            "cls_id": 0,
                        }
                    )
            _, _, close_track = tracker.update(detections, frame_num)
            close_tracks.extend(close_track)

        # Force closure of remaining active tracks at the end
        final_frame = max(all_frames) if all_frames else 0
        for track_id in list(tracker.tracks.keys()):
            # Move all remaining tracks to close_tracks regardless of disappearance
            close_tracks.append(tracker.tracks[track_id])
            del tracker.tracks[track_id]

        episodes = []
        for track in close_tracks:
            if not track.positions:
                continue
            self.track_distances[track.track_id] = (
                f"Distance > {self.max_distance}px"
                if track.reason == "Unknown"
                else track.reason
            )
            episodes.append(track)

        self.tracks = self._filter_short_tracks(episodes)

    def _save_tracks_to_json(self) -> None:
        """Save each track to a separate JSON file with court coordinates."""
        csv_name = os.path.splitext(os.path.basename(self.csv_path))[0]
        video_basename = os.path.basename(os.path.dirname(self.csv_path)) or csv_name.replace("_predict_ball", "")
        tracks_dir = os.path.join(self.output_dir, video_basename, "tracks")
        os.makedirs(tracks_dir, exist_ok=True)

        for track in self.tracks:
            # Add court coordinates to track data
            track_dict = track.to_dict()
            
            # Transform positions to court coordinates
            court_positions = []
            for pos in track_dict['positions']:
                img_x, img_y = pos[0]
                court_x, court_y = self._transform_point_to_court(img_x, img_y)
                court_positions.append([[court_x, court_y], pos[1]])
            
            track_dict['court_positions'] = court_positions
            track_dict['court_info'] = {
                'image_width': self.image_width,
                'image_height': self.image_height,
                'court_points_count': len(self.court_points) if self.court_points else 0,
                'has_court_transform': self.court_transform_matrix is not None
            }
            
            file_path = os.path.join(tracks_dir, f"track_{track.track_id:04d}.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(track_dict, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(self.tracks)} tracks to: {tracks_dir}")

    def run(self) -> None:
        """Main execution flow."""
        self._validate_csv()
        df = self._load_and_process_csv()
        self._process_detections(df)
        self._save_tracks_to_json()
        print(f"Done. Found {len(self.tracks)} tracks.")


def main():
    parser = argparse.ArgumentParser(description="Calculate tracks from CSV to JSON with court coordinates")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to ball.csv")
    parser.add_argument("--court_json_path", type=str, help="Path to court coordinates JSON file")
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Root output directory for JSON"
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Frames per second")
    parser.add_argument(
        "--max_distance", type=float, default=200.0, help="Max tracking distance"
    )
    parser.add_argument(
        "--min_duration_sec", type=float, default=1.0, help="Minimum track duration"
    )
    parser.add_argument(
        "--max_x_displacement", type=float, default=20.0, help="Max X displacement"
    )
    parser.add_argument(
        "--min_y_displacement", type=float, default=50.0, help="Min Y displacement"
    )
    parser.add_argument(
        "--bounce_frames", type=int, default=10, help="Frames to analyze bounce"
    )
    args = parser.parse_args()

    calculator = TrackCalculator(
        csv_path=args.csv_path,
        court_json_path=args.court_json_path,
        output_dir=args.output_dir,
        fps=args.fps,
        max_distance=args.max_distance,
        min_duration_sec=args.min_duration_sec,
        max_x_displacement=args.max_x_displacement,
        min_y_displacement=args.min_y_displacement,
        bounce_frames=args.bounce_frames,
    )
    calculator.run()


if __name__ == "__main__":
    main()
