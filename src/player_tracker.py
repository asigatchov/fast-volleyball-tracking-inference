import cv2
import numpy as np
import onnxruntime as ort
import os
from typing import List, Tuple, Optional, Dict
from scipy.spatial import distance


class Player:
    """Represents a tracked player with ID and position."""
    
    def __init__(self, player_id: int, position: Tuple[int, int]):
        self.player_id = player_id
        self.position = position
        self.last_seen = 0
        self.positions_history = [position]


class SimpleTracker:
    """Simple tracker for player IDs using position proximity."""
    
    def __init__(self, max_disappeared: int = 30, max_distance: float = 100):
        self.next_id = 0
        self.players: Dict[int, Player] = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.disappeared_count: Dict[int, int] = {}
        
    def update(self, positions: List[Tuple[int, int]], frame_count: int) -> Dict[int, int]:
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


class PlayerTracker:
    """Class for tracking volleyball players using YOLO model."""
    
    def __init__(self, yolo_model_path: str, court_image_path: str = ""):
        """
        Initialize the PlayerTracker.
        
        Args:
            yolo_model_path: Path to the YOLO ONNX model
            court_image_path: Path to the court schematic image (optional)
        """
        self.yolo_model_path = yolo_model_path
        self.court_image_path = court_image_path
        self.session = None
        self.court_image = None
        self.transformation_matrix = None
        self.court_boundary = None  # Court boundary points for filtering
        
        # Initialize tracker
        self.tracker = SimpleTracker()
        self.frame_count = 0
        
        # Load the YOLO model
        self._load_model()
        
        # Load court image if provided
        if self.court_image_path and os.path.exists(self.court_image_path):
            self.court_image = cv2.imread(self.court_image_path)
    
    def _load_model(self):
        """Load the YOLO ONNX model."""
        if not os.path.exists(self.yolo_model_path):
            raise FileNotFoundError(f"YOLO model not found: {self.yolo_model_path}")
            
        self.session = ort.InferenceSession(self.yolo_model_path)
        
    def detect_players(self, frame: np.ndarray, confidence_threshold: float = 0.5) -> List[Tuple[int, int, int, int]]:
        """
        Detect players in a frame using YOLO model.
        
        Args:
            frame: Input frame (BGR format)
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            List of bounding boxes (x, y, w, h) for detected players
        """
        # Preprocess the frame
        input_shape = (640, 640)  # Standard YOLO input size
        resized_frame = cv2.resize(frame, input_shape)
        input_image = resized_frame.astype(np.float32) / 255.0
        input_image = np.transpose(input_image, (2, 0, 1))  # HWC to CHW
        input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension
        
        # Run inference
        if self.session is None:
            raise RuntimeError("Model not loaded")
            
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_image})
        
        # Post-process detections
        player_boxes = self._postprocess_detections(
            outputs, 
            frame.shape[1], 
            frame.shape[0],
            input_shape[0],
            input_shape[1],
            confidence_threshold
        )
        
        return player_boxes
    
    def _postprocess_detections(self, outputs: List[np.ndarray], 
                               orig_width: int, orig_height: int,
                               input_width: int, input_height: int,
                               confidence_threshold: float) -> List[Tuple[int, int, int, int]]:
        """
        Post-process YOLO detections to extract player bounding boxes.
        
        Args:
            outputs: Model outputs
            orig_width: Original frame width
            orig_height: Original frame height
            input_width: Model input width
            input_height: Model input height
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            List of player bounding boxes (x, y, w, h)
        """
        # This is a simplified implementation
        # In practice, you would need to decode the YOLO output properly
        # based on the specific model architecture
        
        player_boxes = []
        
        # Assuming outputs[0] contains the detection results
        # Format: [batch, num_detections, 6] where 6 is [x1, y1, x2, y2, confidence, class_id]
        if len(outputs) > 0 and outputs[0] is not None:
            detections = outputs[0][0]  # First batch
            
            for detection in detections:
                if len(detection) >= 6:
                    x1, y1, x2, y2, conf, class_id = detection[:6]
                    
                    # Filter by confidence and class (person class ID is typically 0 in COCO)
                    if conf > confidence_threshold and int(class_id) == 0:  # Person class
                        # Convert coordinates to original frame size
                        x1_orig = int(x1 * orig_width / input_width)
                        y1_orig = int(y1 * orig_height / input_height)
                        x2_orig = int(x2 * orig_width / input_width)
                        y2_orig = int(y2 * orig_height / input_height)
                        
                        # Convert to (x, y, w, h) format
                        x = x1_orig
                        y = y1_orig
                        w = x2_orig - x1_orig
                        h = y2_orig - y1_orig
                        
                        player_boxes.append((x, y, w, h))
                
        return player_boxes
    
    def get_player_position(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """
        Get the player position as the center of the bottom edge of the bounding box.
        
        Args:
            bbox: Bounding box (x, y, w, h)
            
        Returns:
            Player position (x, y) at the center of bottom edge
        """
        x, y, w, h = bbox
        # Center of the bottom edge
        player_x = x + w // 2
        player_y = y + h
        return (player_x, player_y)
    
    def set_transformation_matrix(self, video_points: np.ndarray, court_points: np.ndarray):
        """
        Set the perspective transformation matrix for mapping video coordinates to court coordinates.
        
        Args:
            video_points: 4 points in video coordinates
            court_points: 4 corresponding points in court coordinates
        """
        self.transformation_matrix = cv2.getPerspectiveTransform(
            video_points.astype(np.float32), 
            court_points.astype(np.float32)
        )
    
    def set_court_boundary(self, court_points: np.ndarray):
        """
        Set the court boundary points for filtering players within the court.
        
        Args:
            court_points: 4 points defining the court boundary
        """
        self.court_boundary = court_points
    
    def is_within_court(self, point: Tuple[int, int]) -> bool:
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
    
    def transform_to_court(self, point: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """
        Transform a point from video coordinates to court coordinates.
        
        Args:
            point: Point in video coordinates (x, y)
            
        Returns:
            Point in court coordinates (x, y) or None if transformation matrix is not set
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
    
    def draw_player_positions(self, frame: np.ndarray, player_boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        Draw player positions on the frame (without bounding boxes).
        
        Args:
            frame: Input frame
            player_boxes: List of player bounding boxes
            
        Returns:
            Frame with player positions drawn
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
            
            # If transformation matrix is available, also draw on court
            if self.transformation_matrix is not None and self.court_image is not None:
                court_pos = self.transform_to_court(pos)
                if court_pos:
                    # Create overlay of court on frame
                    court_overlay = self.court_image.copy()
                    cv2.circle(court_overlay, court_pos, 5, (0, 0, 255), -1)
                    cv2.putText(court_overlay, f"ID: {player_id}", 
                               (court_pos[0] + 10, court_pos[1] - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Resize court overlay to fit in corner of frame
                    court_h, court_w = court_overlay.shape[:2]
                    scale_factor = 0.2
                    new_court_w = int(court_w * scale_factor)
                    new_court_h = int(court_h * scale_factor)
                    resized_court = cv2.resize(court_overlay, (new_court_w, new_court_h))
                    
                    # Place court in top-right corner
                    if output_frame.shape[0] > new_court_h + 10 and output_frame.shape[1] > new_court_w + 10:
                        output_frame[10:10+new_court_h, -10-new_court_w:-10] = resized_court
                    
        return output_frame


def main():
    """Example usage of the PlayerTracker."""
    # Example usage
    tracker = PlayerTracker(
        yolo_model_path="models_yolo/yolo11n.onnx",
        court_image_path="images/court.jpg"
    )
    
    # Example points for transformation (you would select these interactively)
    video_points = np.array([[100, 100], [100, 400], [500, 400], [500, 100]])
    
    court_points = np.array([[[68.0, 575.0], [69.0, 46.0], [333.0, 46.0], [333.0, 576.0]]])
    
    tracker.set_transformation_matrix(video_points, court_points)
    tracker.set_court_boundary(court_points)
    
    # Process video
    cap = cv2.VideoCapture("video/bl_transhsmash_volar_woman_g2.mp4")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect players
        player_boxes = tracker.detect_players(frame)
        print(player_boxes)
        # Draw player positions
        output_frame = tracker.draw_player_positions(frame, player_boxes)
        
        cv2.imshow("Player Tracking", output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()