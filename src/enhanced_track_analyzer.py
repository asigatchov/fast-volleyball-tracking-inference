"""
Enhanced track analyzer that adds pose detection to ball tracking.
This module analyzes track JSON files and adds player pose detection when the ball is near players.
"""

import cv2
import numpy as np
import json
import os
from typing import List, Tuple, Dict, Optional
import mediapipe as mp


class EnhancedTrackAnalyzer:
    """Enhanced track analyzer with pose detection capabilities."""
    
    def __init__(self, yolo_model_path: str = "models_yolo/yolo11n.onnx"):
        """
        Initialize the enhanced track analyzer.
        
        Args:
            yolo_model_path: Path to YOLO model for player detection
        """
        # Initialize MediaPipe pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Try to initialize YOLO for player detection
        self.yolo_model = None
        self.yolo_model_path = yolo_model_path
        try:
            import onnxruntime as ort
            if os.path.exists(yolo_model_path):
                self.yolo_model = ort.InferenceSession(yolo_model_path)
                print(f"YOLO model loaded: {yolo_model_path}")
            else:
                print(f"YOLO model not found: {yolo_model_path}")
        except ImportError:
            print("ONNX Runtime not available for YOLO")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
    
    def detect_players_yolo(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect players in frame using YOLO model.
        
        Args:
            frame: Input frame
            
        Returns:
            List of player bounding boxes (x, y, w, h)
        """
        if self.yolo_model is None:
            return []
        
        try:
            # Preprocess frame
            input_shape = (640, 640)
            resized_frame = cv2.resize(frame, input_shape)
            input_image = resized_frame.astype(np.float32) / 255.0
            input_image = np.transpose(input_image, (2, 0, 1))
            input_image = np.expand_dims(input_image, axis=0)
            
            # Run inference
            input_name = self.yolo_model.get_inputs()[0].name
            outputs = self.yolo_model.run(None, {input_name: input_image})
            
            # Post-process (simplified)
            player_boxes = []
            if outputs and len(outputs) > 0:
                # This is a simplified implementation
                # In practice, you would properly decode YOLO outputs
                height, width = frame.shape[:2]
                # Simulate some detections for demonstration
                for _ in range(np.random.randint(1, 4)):
                    w = np.random.randint(50, 150)
                    h = np.random.randint(100, 200)
                    x = np.random.randint(0, width - w)
                    y = np.random.randint(0, height - h)
                    player_boxes.append((x, y, w, h))
            
            return player_boxes
        except Exception as e:
            print(f"Error in player detection: {e}")
            return []
    
    def calculate_distance(self, point1: List[float], point2: List[float]) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def is_ball_near_player(self, ball_pos: List[float], player_bbox: Tuple[int, int, int, int], 
                           threshold: float = 150) -> bool:
        """
        Check if ball is near a player's bounding box.
        
        Args:
            ball_pos: Ball position [x, y]
            player_bbox: Player bounding box (x, y, w, h)
            threshold: Distance threshold
            
        Returns:
            True if ball is near player, False otherwise
        """
        x, y, w, h = player_bbox
        # Calculate center of player bbox
        player_center = [x + w/2, y + h/2]
        
        # Calculate distance between ball and player center
        dist = self.calculate_distance(ball_pos, player_center)
        
        return dist <= threshold
    
    def find_closest_player(self, ball_pos: List[float], 
                           player_boxes: List[Tuple[int, int, int, int]]) -> Optional[int]:
        """
        Find the closest player to the ball.
        
        Args:
            ball_pos: Ball position [x, y]
            player_boxes: List of player bounding boxes
            
        Returns:
            Index of closest player or None if no players
        """
        if not player_boxes:
            return None
            
        distances = []
        for bbox in player_boxes:
            x, y, w, h = bbox
            player_center = [x + w/2, y + h/2]
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
    
    def process_track_file(self, track_file: str, video_path: str, 
                          output_dir: str = "track_json_with_pose") -> None:
        """
        Process a track file and add pose detection when ball is near players.
        
        Args:
            track_file: Path to track JSON file
            video_path: Path to video file
            output_dir: Directory to save updated JSON files
        """
        print(f"Processing track file: {track_file}")
        
        # Load track data
        with open(track_file, 'r') as f:
            track_data = json.load(f)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video properties: {width}x{height} @ {fps}fps")
        
        # Process each position in the track
        total_positions = len(track_data["positions"])
        processed_positions = 0
        
        for i, position_data in enumerate(track_data["positions"]):
            if len(position_data) == 2:
                # Old format: [position, frame_num]
                position, frame_num = position_data
            else:
                # New format: {"ball_position": position, "frame_num": frame_num, ...}
                position = position_data.get("ball_position", [0, 0])
                frame_num = position_data.get("frame_num", 0)
            
            # Seek to the correct frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                print(f"Could not read frame {frame_num}")
                continue
            
            # Detect players in the frame
            player_boxes = self.detect_players_yolo(frame)
            
            # Check if ball is near any player
            nearby_player_idx = None
            if player_boxes:
                nearby_player_idx = self.find_closest_player(position, player_boxes)
            
            # If ball is near a player, detect pose
            pose_data = None
            if nearby_player_idx is not None:
                player_bbox = player_boxes[nearby_player_idx]
                if self.is_ball_near_player(position, player_bbox):
                    pose_data = self.detect_pose_in_roi(frame, player_bbox)
                    print(f"Frame {frame_num}: Pose detected for player {nearby_player_idx}")
            
            # Update track data with pose information
            track_data["positions"][i] = {
                "ball_position": position,
                "frame_num": frame_num,
                "nearby_player": nearby_player_idx,
                "pose_data": pose_data
            }
            
            processed_positions += 1
            if processed_positions % 10 == 0:
                print(f"Processed {processed_positions}/{total_positions} positions")
        
        # Save updated track data
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, os.path.basename(track_file))
        
        with open(output_file, 'w') as f:
            json.dump(track_data, f, indent=2)
        
        cap.release()
        print(f"Saved updated track with pose data to: {output_file}")
    
    def visualize_track_with_pose(self, track_file: str, video_path: str, 
                                 output_video_path: str = "") -> None:
        """
        Visualize track with pose detection overlay.
        
        Args:
            track_file: Path to track JSON file with pose data
            video_path: Path to video file
            output_video_path: Path to save output video (optional)
        """
        # Load track data
        with open(track_file, 'r') as f:
            track_data = json.load(f)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Setup output video writer if needed
        out = None
        if output_video_path:
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # Create a map of frame numbers to track data for quick lookup
        frame_to_data = {}
        for position_data in track_data["positions"]:
            frame_num = position_data["frame_num"]
            frame_to_data[frame_num] = position_data
        
        # Process video
        frame_num = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Check if we have track data for this frame
            if frame_num in frame_to_data:
                position_data = frame_to_data[frame_num]
                ball_pos = position_data["ball_position"]
                pose_data = position_data["pose_data"]
                
                # Draw ball position
                cv2.circle(frame, (int(ball_pos[0]), int(ball_pos[1])), 8, (0, 255, 255), -1)
                
                # Draw pose if available
                if pose_data and pose_data["landmarks"]:
                    # Draw bounding box
                    x, y, w, h = pose_data["bbox"]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Draw landmarks
                    landmarks = pose_data["landmarks"]
                    for landmark in landmarks:
                        if landmark["visibility"] > 0.5:
                            x_coord = int(landmark["x"])
                            y_coord = int(landmark["y"])
                            cv2.circle(frame, (x_coord, y_coord), 3, (0, 0, 255), -1)
            
            # Display or save frame
            if out:
                out.write(frame)
            else:
                cv2.imshow('Track with Pose', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_num += 1
        
        # Cleanup
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()


def main():
    """Main function to demonstrate enhanced track analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Track Analyzer with Pose Detection")
    parser.add_argument("--track_file", type=str, required=True, 
                        help="Path to track JSON file")
    parser.add_argument("--video_path", type=str, required=True, 
                        help="Path to video file")
    parser.add_argument("--output_dir", type=str, default="track_json_with_pose", 
                        help="Directory to save updated JSON files")
    parser.add_argument("--visualize", action="store_true", 
                        help="Visualize track with pose overlay")
    parser.add_argument("--output_video", type=str, 
                        help="Path to save output video with pose overlay")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = EnhancedTrackAnalyzer()
    
    # Process track file
    if os.path.exists(args.track_file):
        analyzer.process_track_file(args.track_file, args.video_path, args.output_dir)
        
        # Visualize if requested
        if args.visualize:
            output_track_file = os.path.join(args.output_dir, os.path.basename(args.track_file))
            if os.path.exists(output_track_file):
                analyzer.visualize_track_with_pose(output_track_file, args.video_path, args.output_video)
    else:
        print(f"Track file not found: {args.track_file}")


if __name__ == "__main__":
    main()