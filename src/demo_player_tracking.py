"""
Demo script for player tracking on volleyball court using YOLO model.
This script demonstrates:
1. Loading the YOLO model for player detection
2. Detecting players in video frames
3. Mapping player positions to court coordinates
4. Visualizing results
"""

import cv2
import numpy as np
import onnxruntime as ort
import os
import argparse


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
    
    def visualize_tracking(self, frame, player_boxes, court_image=None):
        """
        Visualize player tracking results.
        
        Args:
            frame (np.ndarray): Input frame
            player_boxes (list): List of player bounding boxes
            court_image (np.ndarray): Court schematic image (optional)
            
        Returns:
            np.ndarray: Frame with visualizations
        """
        output_frame = frame.copy()
        
        # Draw player detections and positions
        for bbox in player_boxes:
            x, y, w, h = bbox
            
            # Draw bounding box
            cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Get player position (center of bottom edge)
            position = self.get_player_position(bbox)
            
            # Draw player position point
            cv2.circle(output_frame, position, 5, (0, 0, 255), -1)
            
            # If transformation matrix is available, also draw on court
            if self.transformation_matrix is not None:
                court_pos = self.transform_to_court(position)
                if court_pos:
                    # Draw transformed position on frame as text
                    cv2.putText(output_frame, f"({court_pos[0]}, {court_pos[1]})", 
                               (position[0], position[1] - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        return output_frame


def main():
    """Main function to demonstrate player tracking."""
    parser = argparse.ArgumentParser(description="Volleyball Player Tracking Demo")
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
    
    # Set transformation matrix
    tracker.set_court_transformation(video_points, court_points)
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect players in the frame
        player_boxes = tracker.detect_players(frame)
        
        # Visualize results
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