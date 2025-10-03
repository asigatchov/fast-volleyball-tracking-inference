"""
Pose detection module using MediaPipe for volleyball player pose analysis.
This module detects player poses when the ball is near them.
"""

import cv2
import numpy as np
import mediapipe as mp
import json
import os
import argparse
from typing import List, Tuple, Dict, Optional, Union
from scipy.spatial import distance


class PoseDetector:
    """Class for detecting player poses using MediaPipe when ball is near."""
    
    def __init__(self):
        """Initialize MediaPipe pose detection components."""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize pose detector with default parameters
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def is_ball_near_player(self, ball_pos: Tuple[float, float], player_bbox: Tuple[int, int, int, int], 
                           threshold: float = 100) -> bool:
        """
        Check if ball is near a player's bounding box.
        
        Args:
            ball_pos: Ball position (x, y)
            player_bbox: Player bounding box (x, y, w, h)
            threshold: Distance threshold
            
        Returns:
            True if ball is near player, False otherwise
        """
        x, y, w, h = player_bbox
        # Calculate center of player bbox
        player_center = (x + w/2, y + h/2)
        
        # Calculate distance between ball and player center
        dist = self.calculate_distance(ball_pos, player_center)
        
        return dist <= threshold
    
    def find_closest_player(self, ball_pos: Tuple[float, float], 
                           player_boxes: List[Tuple[int, int, int, int]]) -> Optional[int]:
        """
        Find the closest player to the ball.
        
        Args:
            ball_pos: Ball position (x, y)
            player_boxes: List of player bounding boxes
            
        Returns:
            Index of closest player or None if no players
        """
        if not player_boxes:
            return None
            
        distances = []
        for bbox in player_boxes:
            x, y, w, h = bbox
            player_center = (x + w/2, y + h/2)
            dist = self.calculate_distance(ball_pos, player_center)
            distances.append(dist)
            
        closest_idx = np.argmin(distances)
        return int(closest_idx)
    
    def detect_pose_in_roi(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict:
        """
        Detect pose in a region of interest (player bounding box).
        
        Args:
            frame: Input frame
            bbox: Bounding box (x, y, w, h)
            
        Returns:
            Dictionary with pose detection results
        """
        x, y, w, h = bbox
        
        # Add some padding around the bounding box
        padding = 20
        x1 = max(0, int(x - padding))
        y1 = max(0, int(y - padding))
        x2 = min(frame.shape[1], int(x + w + padding))
        y2 = min(frame.shape[0], int(y + h + padding))
        
        # Extract ROI
        roi = frame[y1:y2, x1:x2]
        
        # Convert BGR to RGB for MediaPipe
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.pose.process(rgb_roi)
        
        pose_data = {
            "landmarks": None,
            "bbox": [x, y, w, h],
            "roi_coords": [x1, y1, x2, y2]
        }
        
        # Extract landmarks if detected
        if results.pose_landmarks:
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                # Convert normalized coordinates back to image coordinates
                landmarks.append({
                    "x": landmark.x * (x2 - x1) + x1,
                    "y": landmark.y * (y2 - y1) + y1,
                    "z": landmark.z,
                    "visibility": landmark.visibility
                })
            pose_data["landmarks"] = landmarks
            
        return pose_data
    
    def draw_pose_landmarks(self, frame: np.ndarray, pose_data: Dict) -> np.ndarray:
        """
        Draw pose landmarks on frame.
        
        Args:
            frame: Input frame
            pose_data: Pose detection results
            
        Returns:
            Frame with pose landmarks drawn
        """
        output_frame = frame.copy()
        
        if pose_data["landmarks"]:
            # Draw bounding box
            x, y, w, h = pose_data["bbox"]
            cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw landmarks
            landmarks = pose_data["landmarks"]
            for i, landmark in enumerate(landmarks):
                if landmark["visibility"] > 0.5:  # Only draw visible landmarks
                    x_coord = int(landmark["x"])
                    y_coord = int(landmark["y"])
                    cv2.circle(output_frame, (x_coord, y_coord), 3, (0, 0, 255), -1)
                    
        return output_frame

    def draw_ball_position(self, frame: np.ndarray, ball_pos: Tuple[int, int]) -> np.ndarray:
        """
        Draw ball position on frame.
        
        Args:
            frame: Input frame
            ball_pos: Ball position (x, y)
            
        Returns:
            Frame with ball position drawn
        """
        output_frame = frame.copy()
        if ball_pos and ball_pos[0] >= 0 and ball_pos[1] >= 0:
            cv2.circle(output_frame, ball_pos, 8, (255, 0, 0), -1)  # Blue circle for ball
            cv2.putText(output_frame, "Ball", (ball_pos[0] + 10, ball_pos[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        return output_frame


def add_pose_to_track_json(track_file: str, video_path: str, output_dir: str = "track_json_with_pose", visualize: bool = False):
    """
    Add pose detection to existing track JSON files.
    
    Args:
        track_file: Path to track JSON file
        video_path: Path to video file
        output_dir: Directory to save updated JSON files
        visualize: Whether to visualize frames with cv2
    """
    # Load track data
    with open(track_file, 'r') as f:
        track_data = json.load(f)
    
    # Initialize pose detector
    pose_detector = PoseDetector()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Process each position in the track
    for i, position_data in enumerate(track_data["positions"]):
        # Handle both old and new format
        if isinstance(position_data, list) and len(position_data) == 2:
            # Old format: [position, frame_num]
            ball_position = position_data[0]
            frame_num = position_data[1]
        else:
            # New format: {"ball_position": [...], "frame_num": ..., ...}
            ball_position = position_data.get("ball_position", [0, 0]) if isinstance(position_data, dict) else [0, 0]
            frame_num = position_data.get("frame_num", 0) if isinstance(position_data, dict) else 0
        
        ball_x, ball_y = ball_position
        
        # Seek to the correct frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue
            
        # For demonstration, we'll simulate player detection
        # In a real implementation, you would detect players in the frame
        # and check if the ball is near any of them
        
        # Simulate player detection (in practice, use YOLO or similar)
        player_boxes = []  # This would come from player detection
        
        # Check if ball is near any player
        nearby_player_idx = None
        if player_boxes:
            nearby_player_idx = pose_detector.find_closest_player((ball_x, ball_y), player_boxes)
        
        # If ball is near a player, detect pose
        if nearby_player_idx is not None:
            player_bbox = player_boxes[nearby_player_idx]
            pose_data = pose_detector.detect_pose_in_roi(frame, player_bbox)
            
            # Add pose data to track
            track_data["positions"][i] = {
                "ball_position": ball_position,
                "frame_num": frame_num,
                "pose_data": pose_data
            }
            
            # Visualize if requested
            if visualize:
                # Draw ball position
                vis_frame = pose_detector.draw_ball_position(frame, (int(ball_x), int(ball_y)))
                
                # Draw pose if detected
                vis_frame = pose_detector.draw_pose_landmarks(vis_frame, pose_data)
                
                # Show frame
                cv2.imshow("Pose Detection", vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            # No nearby player, just store ball position
            track_data["positions"][i] = {
                "ball_position": ball_position,
                "frame_num": frame_num,
                "pose_data": None
            }
            
            # Visualize ball position only
            if visualize:
                vis_frame = pose_detector.draw_ball_position(frame, (int(ball_x), int(ball_y)))
                cv2.imshow("Pose Detection", vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    # Save updated track data
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, os.path.basename(track_file))
    
    with open(output_file, 'w') as f:
        json.dump(track_data, f, indent=2)
    
    if visualize:
        cv2.destroyAllWindows()
        
    cap.release()
    print(f"Saved updated track with pose data to: {output_file}")


def main():
    """Example usage of the PoseDetector."""
    parser = argparse.ArgumentParser(description="Volleyball Pose Detection")
    parser.add_argument("--track_file", type=str, help="Path to track JSON file")
    parser.add_argument("--video_path", type=str, help="Path to video file")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization")
    
    args = parser.parse_args()
    
    # Example usage
    pose_detector = PoseDetector()
    
    # Process a track file if provided
    if args.track_file and args.video_path:
        if os.path.exists(args.track_file):
            add_pose_to_track_json(args.track_file, args.video_path, visualize=args.visualize)
        else:
            print(f"Track file not found: {args.track_file}")
    else:
        # Process a track file
        track_file = "track_json/track_0229.json"
        video_path = "path/to/video.mp4"  # You would provide the actual video path
        
        if os.path.exists(track_file):
            add_pose_to_track_json(track_file, video_path)
        else:
            print(f"Track file not found: {track_file}")


if __name__ == "__main__":
    main()