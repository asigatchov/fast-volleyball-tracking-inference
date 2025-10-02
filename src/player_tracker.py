import cv2
import numpy as np
import onnxruntime as ort
import os
from typing import List, Tuple, Optional


def pad_frame_to_square(frame):
    h, w, _ = frame.shape
    if h == w:
        return frame, 0, 0
    elif h > w:
        padding = h - w
        pad_left = padding // 2
        pad_right = padding - pad_left
        padded = cv2.copyMakeBorder(frame, 0, 0, pad_left, pad_right,
                                    cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return padded, pad_left, 0
    else:
        padding = w - h
        pad_top = padding // 2
        pad_bottom = padding - pad_top
        padded = cv2.copyMakeBorder(frame, pad_top, pad_bottom, 0, 0,
                                    cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return padded, 0, pad_top

def slice_image_sahi(image, slice_height=640, slice_width=640, overlap_height_ratio=0.2, overlap_width_ratio=0.2):
    """
    Slice image into overlapping patches (SAHI approach).
    Returns list of (patch, x_offset, y_offset).
    """
    img_h, img_w = image.shape[:2]
    step_h = int(slice_height * (1 - overlap_height_ratio))
    step_w = int(slice_width * (1 - overlap_width_ratio))
    patches = []
    for y in range(0, img_h, step_h):
        for x in range(0, img_w, step_w):
            patch = image[y:y+slice_height, x:x+slice_width]
            if patch.shape[0] < slice_height or patch.shape[1] < slice_width:
                # Pad patch to required size
                patch = cv2.copyMakeBorder(
                    patch,
                    0, slice_height - patch.shape[0],
                    0, slice_width - patch.shape[1],
                    cv2.BORDER_CONSTANT, value=(0,0,0)
                )
            patches.append((patch, x, y))
    return patches

def preprocess_yolo_input(image_rgb, input_size=(640, 640)):
    input_data = image_rgb.astype(np.float32) / 255.0
    input_data = cv2.resize(input_data, input_size)
    input_data = np.transpose(input_data, (2, 0, 1))
    input_data = np.expand_dims(input_data, axis=0)
    return input_data

def postprocess_yolo_output(output, original_img_shape, input_size=(640, 640),
                            conf_threshold=0.5, nms_threshold=0.45):
    output = np.squeeze(output)

    if output.shape[0] < output.shape[1]:
        output = output.T

    num_features = output.shape[1]

    if num_features == 5:  # ball model (single class)
        boxes_raw = output[:, :4]
        scores = output[:, 4]
        class_ids = np.zeros(len(scores), dtype=int)
    elif num_features == 84:  # COCO person model
        boxes_raw = output[:, :4]
        class_scores = output[:, 4:]
        scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)
    else:
        return np.array([]).reshape(0,4), np.array([]), np.array([])

    valid_mask = scores > conf_threshold
    boxes_filtered = boxes_raw[valid_mask]
    scores_filtered = scores[valid_mask]
    class_ids_filtered = class_ids[valid_mask]

    if len(boxes_filtered) == 0:
        return np.array([]).reshape(0,4), np.array([]), np.array([])

    img_h, img_w = original_img_shape[:2]
    input_h, input_w = input_size

    scale_x = img_w / input_w
    scale_y = img_h / input_h

    x_center, y_center, width, height = boxes_filtered[:, 0], boxes_filtered[:, 1], boxes_filtered[:, 2], boxes_filtered[:, 3]

    x1 = (x_center - width / 2) * scale_x
    y1 = (y_center - height / 2) * scale_y
    x2 = (x_center + width / 2) * scale_x
    y2 = (y_center + height / 2) * scale_y

    boxes_final = np.clip(np.stack([x1, y1, x2, y2], axis=1), 0, [img_w, img_h, img_w, img_h]).astype(int)

    # NMS
    boxes_nms_input = np.array([[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in boxes_final])
    indices = cv2.dnn.NMSBoxes(boxes_nms_input.tolist(), scores_filtered.tolist(), conf_threshold, nms_threshold)
    if len(indices) == 0:
        return np.array([]).reshape(0,4), np.array([]), np.array([])

    indices = indices.flatten()
    return boxes_final[indices], scores_filtered[indices], class_ids_filtered[indices]

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
        Detect players in a frame using YOLO model (SAHI slicing, no resize).
        Returns bounding boxes in original image coordinates.
        """
        slice_height, slice_width = 640, 640
        overlap_height_ratio, overlap_width_ratio = 0.2, 0.2
        patches = slice_image_sahi(frame, slice_height, slice_width, overlap_height_ratio, overlap_width_ratio)
        all_boxes = []
        all_class_ids = []
        for patch, x_offset, y_offset in patches:
            input_image = preprocess_yolo_input(patch, (slice_width, slice_height))
            if self.session is None:
                raise RuntimeError("Model not loaded")
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: input_image})
            boxes, scores, class_ids = postprocess_yolo_output(
                outputs[0],
                patch.shape,
                input_size=(slice_width, slice_height),
                conf_threshold=confidence_threshold,
                nms_threshold=0.45
            )
            # Map boxes to original image coordinates
            for box, cid in zip(boxes, class_ids):
                if cid == 0:
                    x1, y1, x2, y2 = box
                    x1 += x_offset
                    x2 += x_offset
                    y1 += y_offset
                    y2 += y_offset
                    all_boxes.append((x1, y1, x2, y2))
                    all_class_ids.append(cid)
        # Optionally: merge overlapping boxes (simple NMS on all_boxes)
        if all_boxes:
            boxes_nms_input = np.array([[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in all_boxes])
            scores_dummy = [1.0]*len(all_boxes)
            indices = cv2.dnn.NMSBoxes(boxes_nms_input.tolist(), scores_dummy, 0.1, 0.5)
            if len(indices) > 0:
                indices = indices.flatten()
                all_boxes = [all_boxes[i] for i in indices]
        return all_boxes

    def get_player_position(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """
        Get the player position as the center of the bottom edge of the bounding box.
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Player position (x, y) at the center of bottom edge
        """
        x1, y1, x2, y2 = bbox
        player_x = (x1 + x2) // 2
        player_y = y2
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
            position = self.get_player_position(bbox)
            x1, y1, x2, y2 = bbox
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(output_frame, position, 5, (0, 0, 255), -1)

            # If transformation matrix is available, also draw on court
            if self.transformation_matrix is not None and self.court_image is not None:
                court_pos = self.transform_to_court(position)
                if court_pos:
                    court_overlay = self.court_image.copy()
                    cv2.circle(court_overlay, court_pos, 5, (0, 0, 255), -1)
                    court_h, court_w = court_overlay.shape[:2]
                    scale_factor = 0.2
                    new_court_w = int(court_w * scale_factor)
                    new_court_h = int(court_h * scale_factor)
                    resized_court = cv2.resize(court_overlay, (new_court_w, new_court_h))
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
    cap = cv2.VideoCapture("video/bl_transhsmash_volar_woman_g2.mp4")
    
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