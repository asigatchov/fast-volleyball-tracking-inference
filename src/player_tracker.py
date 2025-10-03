import cv2
import numpy as np
import onnxruntime as ort
import os
from typing import List, Tuple, Optional, Any
from scipy.spatial import distance


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
        """
        input_shape: Tuple[int, int] = (640, 640)
        resized_frame: np.ndarray = cv2.resize(frame, input_shape)
        input_image: np.ndarray = resized_frame.astype(np.float32) / 255.0
        input_image = np.transpose(input_image, (2, 0, 1))
        input_image = np.expand_dims(input_image, axis=0)
        if self.session is None:
            raise RuntimeError("Model not loaded")
        input_name: str = self.session.get_inputs()[0].name
        outputs: List[Any] = self.session.run(None, {input_name: input_image})
        player_boxes: List[Tuple[int, int, int, int]] = self._postprocess_detections(
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
        This implementation handles a common YOLO output format (like YOLOv8)
        and includes Non-Maximum Suppression.
        """
        
        predictions = np.squeeze(outputs[0]).T

        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > confidence_threshold, :]
        scores = scores[scores > confidence_threshold]

        if len(predictions) == 0:
            return []

        class_ids = np.argmax(predictions[:, 4:], axis=1)
        
        person_mask = class_ids == 0
        person_predictions = predictions[person_mask]
        person_scores = scores[person_mask]

        if len(person_predictions) == 0:
            return []

        x_center = person_predictions[:, 0]
        y_center = person_predictions[:, 1]
        w = person_predictions[:, 2]
        h = person_predictions[:, 3]

        x_scale = orig_width / input_width
        y_scale = orig_height / input_height

        x1 = (x_center - w / 2) * x_scale
        y1 = (y_center - h / 2) * y_scale
        w_scaled = w * x_scale
        h_scaled = h * y_scale

        boxes_for_nms = [[int(x), int(y), int(width), int(height)] for x, y, width, height in zip(x1, y1, w_scaled, h_scaled)]
        
        nms_threshold = 0.15
        indices = cv2.dnn.NMSBoxes(boxes_for_nms, person_scores.tolist(), confidence_threshold, nms_threshold)
        
        player_boxes = []
        if hasattr(indices, 'flatten'):
            for i in indices.flatten():
                player_boxes.append(boxes_for_nms[i])
        elif indices is not None:
             for i in indices:
                player_boxes.append(boxes_for_nms[i])
                
        return player_boxes
    
    def get_player_position(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        x, y, w, h = bbox
        player_x: int = x + w // 2
        player_y: int = y + h
        return (player_x, player_y)
    
    def find_closest_player_to_ball(self, ball_pos: Tuple[int, int], 
                                   player_boxes: List[Tuple[int, int, int, int]]) -> Optional[int]:
        if not player_boxes:
            return None
        distances = [distance.euclidean(ball_pos, self.get_player_position(bbox)) for bbox in player_boxes]
        return int(np.argmin(distances))
    
    def set_transformation_matrix(self, video_points: np.ndarray, court_points: np.ndarray):
        self.transformation_matrix = cv2.getPerspectiveTransform(
            video_points.astype(np.float32), 
            court_points.astype(np.float32)
        )
    
    def set_court_boundary(self, court_points: np.ndarray):
        self.court_boundary = court_points
    
    def is_within_court(self, point: Tuple[int, int]) -> bool:
        if self.court_boundary is None:
            return True
        return cv2.pointPolygonTest(self.court_boundary, point, False) >= 0
    
    def transform_to_court(self, point: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        if self.transformation_matrix is None:
            return None
        point_array = np.array([[point]], dtype=np.float32)
        transformed_point = cv2.perspectiveTransform(point_array, self.transformation_matrix)
        x, y = transformed_point[0][0]
        return (int(x), int(y))
    
    def draw_player_positions(self, frame: np.ndarray, player_boxes: List[Tuple[int, int, int, int]], 
                             ball_position: Optional[Tuple[int, int]] = None, 
                             highlight_nearest: bool = False) -> np.ndarray:
        """
        Draw bounding boxes and positions for detected players.
        
        Args:
            frame: Input frame (BGR format)
            player_boxes: List of bounding boxes (x, y, w, h)
            ball_position: Ball position (x, y) (optional)
            highlight_nearest: Whether to highlight the player closest to the ball
            
        Returns:
            Frame with drawn player positions and bounding boxes
        """
        output_frame = frame.copy()
        player_positions = [self.get_player_position(bbox) for bbox in player_boxes]
        
        valid_positions = []
        valid_boxes = []
        valid_indices = []

        print(f"Total detected players: {len(player_positions)}")

        for i, (pos, box) in enumerate(zip(player_positions, player_boxes)):
            if True or self.is_within_court(pos):
                valid_positions.append(pos)
                valid_boxes.append(box)
                valid_indices.append(i)

        print(f"Players within court: {len(valid_positions)}") 
        nearest_player_idx = None
        if highlight_nearest and ball_position and valid_boxes:
            nearest_player_idx = self.find_closest_player_to_ball(ball_position, valid_boxes)

        for i, (pos, box) in enumerate(zip(valid_positions, valid_boxes)):
            # Determine color: yellow for nearest player, green for others
            box_color = (0, 255, 255) if nearest_player_idx is not None and i == nearest_player_idx else (0, 255, 0)
            pos_color = (0, 255, 255) if nearest_player_idx is not None and i == nearest_player_idx else (0, 0, 255)
            
            # Draw bounding box
            x, y, w, h = box
            cv2.rectangle(output_frame, (x, y), (x + w, y + h), box_color, 2)
            
            # Draw player position (center-bottom of bounding box)
            cv2.circle(output_frame, pos, 5, pos_color, -1)

            # Draw on court image if available
            if self.transformation_matrix is not None and self.court_image is not None:
                court_pos = self.transform_to_court(pos)
                if court_pos:
                    court_overlay = self.court_image.copy()
                    cv2.circle(court_overlay, court_pos, 5, pos_color, -1)
                    
                    court_h, court_w = court_overlay.shape[:2]
                    scale_factor = 0.2
                    new_court_w = int(court_w * scale_factor)
                    new_court_h = int(court_h * scale_factor)
                    resized_court = cv2.resize(court_overlay, (new_court_w, new_court_h))
                    
                    if output_frame.shape[0] > new_court_h + 10 and output_frame.shape[1] > new_court_w + 10:
                        output_frame[10:10+new_court_h, -10-new_court_w:-10] = resized_court
                    
        return output_frame

    def detect_players_in_crop(self, frame: np.ndarray, crop_coords: Tuple[int, int, int, int], 
                              ball_position: Optional[Tuple[int, int]] = None, confidence_threshold: float = 0.5
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect players in a crop region of the frame.
        Args:
            frame: Input frame (BGR format)
            crop_coords: Crop coordinates (x1, y1, x2, y2)
            ball_position: Ball position (x, y) in original frame coordinates (optional, not used here)
            confidence_threshold: Minimum confidence for detections
        Returns:
            List of bounding boxes (x, y, w, h) for detected players in original frame coordinates
        """
        x1, y1, x2, y2 = crop_coords
        crop = frame[y1:y2, x1:x2]
        input_shape = (640, 640)
        resized_crop = cv2.resize(crop, input_shape)
        input_image = resized_crop.astype(np.float32) / 255.0
        input_image = np.transpose(input_image, (2, 0, 1))
        input_image = np.expand_dims(input_image, axis=0)
        if self.session is None:
            raise RuntimeError("Model not loaded")
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_image})
        player_boxes = self._postprocess_detections_crop(
            outputs,
            x1, y1,
            crop.shape[1], crop.shape[0],
            input_shape[0], input_shape[1],
            confidence_threshold
        )
        return player_boxes

    def _postprocess_detections_crop(self, outputs: List[np.ndarray], 
                                    offset_x: int, offset_y: int,
                                    orig_width: int, orig_height: int,
                                    input_width: int, input_height: int,
                                    confidence_threshold: float) -> List[Tuple[int, int, int, int]]:
        """
        Post-process YOLO detections from crop and convert to original frame coordinates.
        """
        player_boxes = []
        predictions = np.squeeze(outputs[0]).T
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > confidence_threshold, :]
        scores = scores[scores > confidence_threshold]
        if len(predictions) == 0:
            return []
        class_ids = np.argmax(predictions[:, 4:], axis=1)
        person_mask = class_ids == 0
        person_predictions = predictions[person_mask]
        person_scores = scores[person_mask]
        if len(person_predictions) == 0:
            return []
        x_center = person_predictions[:, 0]
        y_center = person_predictions[:, 1]
        w = person_predictions[:, 2]
        h = person_predictions[:, 3]
        x_scale = orig_width / input_width
        y_scale = orig_height / input_height
        x1 = (x_center - w / 2) * x_scale
        y1 = (y_center - h / 2) * y_scale
        w_scaled = w * x_scale
        h_scaled = h * y_scale
        for x, y, width, height in zip(x1, y1, w_scaled, h_scaled):
            x1_orig = int(x) + offset_x
            y1_orig = int(y) + offset_y
            player_boxes.append((x1_orig, y1_orig, int(width), int(height)))
        return player_boxes
    
def main():
    """Example usage of the PlayerTracker."""
    tracker = PlayerTracker(
        yolo_model_path="models_yolo/yolo11n.onnx",
        court_image_path="images/court.jpg"
    )
    
    video_points = np.array([[67.0, 574.0], [69.0, 44.0], [329.0, 42.0], [332.0, 576.0]], dtype=np.float32)
    court_points = np.array( [[68.0, 575.0], [69.0, 46.0], [333.0, 46.0], [333.0, 576.0]] , dtype=np.float32)
    
    tracker.set_transformation_matrix(video_points, court_points)
    tracker.set_court_boundary(video_points)
    
    video_path = "woman.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    print(f"Processing video: {video_path}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        player_boxes = tracker.detect_players(frame)
        
        print(f"Frame {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}: Found {len(player_boxes)} players")
        
        output_frame = tracker.draw_player_positions(frame, player_boxes)
        
        cv2.imshow("Player Tracking", output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()