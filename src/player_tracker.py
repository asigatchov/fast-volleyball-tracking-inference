import cv2
import numpy as np
import onnxruntime as ort
import os
from typing import List, Tuple, Optional


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
        Draw player positions on the frame.
        
        Args:
            frame: Input frame
            player_boxes: List of player bounding boxes
            
        Returns:
            Frame with player positions drawn
        """
        output_frame = frame.copy()
        
        for bbox in player_boxes:
            # Get player position (center of bottom edge)
            position = self.get_player_position(bbox)
            
            # Draw bounding box
            x, y, w, h = bbox
            cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw player position point
            cv2.circle(output_frame, position, 5, (0, 0, 255), -1)
            
            # If transformation matrix is available, also draw on court
            if self.transformation_matrix is not None and self.court_image is not None:
                court_pos = self.transform_to_court(position)
                if court_pos:
                    # Create overlay of court on frame
                    court_overlay = self.court_image.copy()
                    cv2.circle(court_overlay, court_pos, 5, (0, 0, 255), -1)
                    
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
    court_points = np.array([[50, 50], [50, 350], [450, 350], [450, 50]])
    
    tracker.set_transformation_matrix(video_points, court_points)
    
    # Process video
    cap = cv2.VideoCapture("path/to/video.mp4")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect players
        player_boxes = tracker.detect_players(frame)
        
        # Draw player positions
        output_frame = tracker.draw_player_positions(frame, player_boxes)
        
        cv2.imshow("Player Tracking", output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()