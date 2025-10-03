"""
Demo script for player tracking on volleyball court using YOLO model with DeepSORT-like tracking.
This script demonstrates:
1. Loading the YOLO model for player detection
2. Detecting players in video frames
3. Tracking players with IDs using a simple tracker
4. Filtering players to only show those within the court area
5. Mapping player positions to court coordinates
6. Visualizing results with player IDs (no bounding boxes)
"""

import cv2
import numpy as np
import onnxruntime as ort
import os
import argparse
from scipy.spatial import distance


class Player:
    """Represents a tracked player with ID and position."""
    
    def __init__(self, player_id: int, position: tuple):
        self.player_id = player_id
        self.position = position
        self.last_seen = 0
        self.positions_history = [position]


class SimpleTracker:
    """Simple tracker for player IDs using position proximity."""
    
    def __init__(self, max_disappeared: int = 30, max_distance: float = 100):
        self.next_id = 0
        self.players = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.disappeared_count = {}
        
    def update(self, positions: list, frame_count: int) -> dict:
        """
        Update tracker with new player positions.
        
        Args:
            positions: List of player positions (x, y)
            frame_count: Current frame number
            
        Returns:
            Dictionary mapping position index to player ID
        """
        # If no players detected
        if len(positions) == 0:
            # Mark all existing players as disappeared
            for player_id in list(self.players.keys()):
                self.disappeared_count[player_id] = self.disappeared_count.get(player_id, 0) + 1
                if self.disappeared_count[player_id] > self.max_disappeared:
                    del self.players[player_id]
                    del self.disappeared_count[player_id]
            return {}
        
        # If no existing players, initialize new ones
        if len(self.players) == 0:
            for pos in positions:
                player = Player(self.next_id, pos)
                player.last_seen = frame_count
                self.players[self.next_id] = player
                self.disappeared_count[self.next_id] = 0
                self.next_id += 1
            return {i: player_id for i, player_id in enumerate(self.players.keys())}
        
        # Match existing players with new positions
        player_ids = list(self.players.keys())
        player_positions = [self.players[pid].position for pid in player_ids]
        
        # Calculate distance matrix
        if len(positions) > 0 and len(player_positions) > 0:
            dist_matrix = distance.cdist(player_positions, positions)
            
            # Assign positions to players
            assigned_positions = set()
            assigned_players = set()
            
            # Greedy assignment based on minimum distance
            while np.min(dist_matrix) <= self.max_distance:
                # Find minimum distance
                min_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
                player_idx, pos_idx = min_idx
                
                # Check if already assigned
                if player_idx in assigned_players or pos_idx in assigned_positions:
                    dist_matrix[player_idx, pos_idx] = np.inf
                    continue
                
                # Assign position to player
                player_id = player_ids[player_idx]
                new_position = positions[pos_idx]
                
                # Update player
                self.players[player_id].position = new_position
                self.players[player_id].positions_history.append(new_position)
                self.players[player_id].last_seen = frame_count
                self.disappeared_count[player_id] = 0
                
                # Mark as assigned
                assigned_players.add(player_idx)
                assigned_positions.add(pos_idx)
                
                # Set distance to infinity to avoid reassignment
                dist_matrix[player_idx, :] = np.inf
                dist_matrix[:, pos_idx] = np.inf
            
            # Handle unassigned positions (new players)
            for pos_idx, pos in enumerate(positions):
                if pos_idx not in assigned_positions:
                    player = Player(self.next_id, pos)
                    player.last_seen = frame_count
                    self.players[self.next_id] = player
                    self.disappeared_count[self.next_id] = 0
                    self.next_id += 1
            
            # Handle disappeared players
            for player_idx, player_id in enumerate(player_ids):
                if player_idx not in assigned_players:
                    self.disappeared_count[player_id] = self.disappeared_count.get(player_id, 0) + 1
                    if self.disappeared_count[player_id] > self.max_disappeared:
                        del self.players[player_id]
                        del self.disappeared_count[player_id]
        else:
            # No matching possible, treat all as new
            self.players.clear()
            self.disappeared_count.clear()
            for pos in positions:
                player = Player(self.next_id, pos)
                player.last_seen = frame_count
                self.players[self.next_id] = player
                self.disappeared_count[self.next_id] = 0
                self.next_id += 1
        
        # Create mapping from position index to player ID
        result = {}
        assigned_positions = set()
        
        if len(positions) > 0 and len(self.players) > 0:
            player_ids = list(self.players.keys())
            player_positions = [self.players[pid].position for pid in player_ids]
            dist_matrix = distance.cdist(player_positions, positions)
            
            for pos_idx, pos in enumerate(positions):
                # Find closest player
                distances = [distance.euclidean(player_pos, pos) for player_pos in player_positions]
                min_dist_idx = np.argmin(distances) if distances else -1
                
                if min_dist_idx >= 0 and distances[min_dist_idx] <= self.max_distance:
                    result[pos_idx] = player_ids[min_dist_idx]
                    assigned_positions.add(pos_idx)
        
        return result


class VolleyballPlayerTracker:
    """Track volleyball players and map their positions to court coordinates."""
    
    def __init__(self, yolo_model_path):
        """
        Initialize the player tracker.
        
        Args:
            yolo_model_path (str): Path to the YOLO ONNX model
        """
        self.yolo_model_path = yolo_model_path
        self.session = None
        self.transformation_matrix = None
        self.court_boundary = None
        self.tracker = SimpleTracker()
        self.frame_count = 0
        self._load_model()
    
    def _load_model(self):
        """Load the YOLO ONNX model."""
        if not os.path.exists(self.yolo_model_path):
            raise FileNotFoundError(f"YOLO model not found: {self.yolo_model_path}")
        
        # Create ONNX runtime session
        self.session = ort.InferenceSession(self.yolo_model_path)
        print(f"Model loaded successfully from {self.yolo_model_path}")
    
    def detect_players(self, frame, confidence_threshold=0.5):
        """
        Detect players in a frame using YOLO model.
        
        Args:
            frame (np.ndarray): Input frame (BGR format)
            confidence_threshold (float): Minimum confidence for detections
            
        Returns:
            list: List of bounding boxes (x, y, w, h) for detected players
        """
        if self.session is None:
            raise RuntimeError("Model not loaded")
        
        # Preprocess the frame for YOLO
        input_shape = (640, 640)  # Standard YOLO input size
        resized_frame = cv2.resize(frame, input_shape)
        input_image = resized_frame.astype(np.float32) / 255.0
        input_image = np.transpose(input_image, (2, 0, 1))  # HWC to CHW
        input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension
        
        # Run inference
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_image})
        
        # Post-process detections (simplified)
        player_boxes = self._extract_person_detections(outputs, frame.shape, input_shape, confidence_threshold)
        return player_boxes
    
    def _extract_person_detections(self, outputs, original_shape, input_shape, confidence_threshold):
        """
        Extract person detections from YOLO outputs.
        
        Note: This is a simplified implementation. A full implementation would need to properly
        decode the YOLO output based on the specific model architecture.
        
        Args:
            outputs: Model outputs
            original_shape: Original frame shape (height, width)
            input_shape: Model input shape (height, width)
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            list: List of bounding boxes (x, y, w, h)
        """
        player_boxes = []
        
        # This is a placeholder implementation
        # In a real implementation, you would decode the YOLO output properly
        # For now, we'll simulate some detections for demonstration
        height, width = original_shape[:2]
        input_height, input_width = input_shape
        
        # Simulate detecting 2-4 players
        num_players = np.random.randint(2, 5)
        for _ in range(num_players):
            # Random bounding box (in real implementation, this would come from model output)
            w = np.random.randint(50, 150)
            h = np.random.randint(100, 200)
            x = np.random.randint(0, width - w)
            y = np.random.randint(0, height - h)
            player_boxes.append((x, y, w, h))
        
        return player_boxes
    
    def get_player_position(self, bbox):
        """
        Get the player position as the center of the bottom edge of the bounding box.
        
        Args:
            bbox (tuple): Bounding box (x, y, w, h)
            
        Returns:
            tuple: Player position (x, y) at the center of bottom edge
        """
        x, y, w, h = bbox
        # Center of the bottom edge
        player_x = x + w // 2
        player_y = y + h
        return (player_x, player_y)
    
    def set_court_transformation(self, video_points, court_points):
        """
        Set the perspective transformation matrix for mapping video coordinates to court coordinates.
        
        Args:
            video_points (np.ndarray): 4 points in video coordinates
            court_points (np.ndarray): 4 corresponding points in court coordinates
        """
        self.transformation_matrix = cv2.getPerspectiveTransform(
            video_points.astype(np.float32), 
            court_points.astype(np.float32)
        )
    
    def set_court_boundary(self, court_points):
        """
        Set the court boundary points for filtering players within the court.
        
        Args:
            court_points: 4 points defining the court boundary
        """
        self.court_boundary = court_points
    
    def is_within_court(self, point):
        """
        Check if a point is within the court boundary.
        
        Args:
            point: Point coordinates (x, y)
            
        Returns:
            True if point is within court boundary, False otherwise
        """
        if self.court_boundary is None:
            return True  # If no boundary defined, assume all points are valid
            
        # Convert point to numpy array
        pt = np.array([point], dtype=np.float32)
        
        # Check if point is inside the polygon defined by court_boundary
        result = cv2.pointPolygonTest(self.court_boundary, tuple(pt[0]), False)
        return result >= 0  # >= 0 means point is inside or on the boundary
    
    def transform_to_court(self, point):
        """
        Transform a point from video coordinates to court coordinates.
        
        Args:
            point (tuple): Point in video coordinates (x, y)
            
        Returns:
            tuple: Point in court coordinates (x, y) or None if transformation matrix is not set
        """
        if self.transformation_matrix is None:
            return None
        
        # Convert point to homogeneous coordinates
        point_array = np.array([[point]], dtype=np.float32)
        
        # Apply perspective transformation
        transformed_point = cv2.perspectiveTransform(point_array, self.transformation_matrix)
        
        # Extract coordinates
        x, y = transformed_point[0][0]
        return (int(x), int(y))
    
    def visualize_tracking(self, frame, player_boxes):
        """
        Visualize player tracking results (points only with IDs, no bounding boxes).
        
        Args:
            frame (np.ndarray): Input frame
            player_boxes (list): List of player bounding boxes
            
        Returns:
            np.ndarray: Frame with visualizations
        """
        output_frame = frame.copy()
        
        # Get player positions (center of bottom edge)
        player_positions = [self.get_player_position(bbox) for bbox in player_boxes]
        
        # Filter positions to only include players within court
        valid_positions = []
        valid_indices = []
        
        for i, pos in enumerate(player_positions):
            if self.is_within_court(pos):
                valid_positions.append(pos)
                valid_indices.append(i)
        
        # Update tracker with valid positions
        self.frame_count += 1
        id_mapping = self.tracker.update(valid_positions, self.frame_count)
        
        # Draw player positions with IDs
        for i, pos in enumerate(valid_positions):
            # Get player ID
            player_id = id_mapping.get(i, -1)
            
            # Draw player position point
            cv2.circle(output_frame, pos, 5, (0, 0, 255), -1)
            
            # Draw player ID
            if player_id >= 0:
                cv2.putText(output_frame, f"ID: {player_id}", 
                           (pos[0] + 10, pos[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return output_frame


def main():
    """Main function to demonstrate player tracking."""
    parser = argparse.ArgumentParser(description="Volleyball Player Tracking Demo v2")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--model_path", type=str, default="models_yolo/yolo11n.onnx", 
                        help="Path to YOLO model")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization")
    
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = VolleyballPlayerTracker(args.model_path)
    
    # Open video
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {args.video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Processing video: {args.video_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}")
    
    # For demonstration, define fixed transformation points
    # In a real application, these would be selected interactively
    video_points = np.array([
        [100, height-100],      # Bottom-left
        [100, 100],             # Top-left
        [width-100, 100],       # Top-right
        [width-100, height-100] # Bottom-right
    ])
    
    # Court points (assuming standard volleyball court dimensions)
    court_width, court_height = 1000, 500  # Example court size
    court_points = np.array([
        [100, court_height-100],      # Bottom-left
        [100, 100],                   # Top-left
        [court_width-100, 100],       # Top-right
        [court_width-100, court_height-100] # Bottom-right
    ])
    
    # Set transformation matrix and court boundary
    tracker.set_court_transformation(video_points, court_points)
    tracker.set_court_boundary(court_points)
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect players in the frame
        player_boxes = tracker.detect_players(frame)
        
        # Visualize results (points only with IDs, no bounding boxes)
        output_frame = tracker.visualize_tracking(frame, player_boxes)
        
        if args.visualize:
            cv2.imshow("Player Tracking", output_frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_count += 1
        if frame_count % 30 == 0:  # Print progress every 30 frames
            print(f"Processed {frame_count} frames...")
    
    # Clean up
    cap.release()
    if args.visualize:
        cv2.destroyAllWindows()
    
    print(f"Finished processing {frame_count} frames")


if __name__ == "__main__":
    main()