"""
Test script for player tracking functionality.
This script demonstrates how to use the PlayerTracker class.
"""

import cv2
import numpy as np
import os
import sys

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from player_tracker import PlayerTracker


def test_player_tracking(video_path):
    """Test player tracking on a video."""
    
    # Initialize player tracker with YOLO model
    tracker = PlayerTracker(
        yolo_model_path="models_yolo/yolo11n_vb.onnx"
    )
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    print("Starting player tracking. Press 'q' to quit.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect players
        player_boxes = tracker.detect_players(frame)
        print(player_boxes)
        # Draw player positions
        output_frame = tracker.draw_player_positions(frame, player_boxes)
        
        # Display the frame
        cv2.imshow("Player Tracking", output_frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Player tracking finished.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test player tracking with YOLO")
    parser.add_argument("--video_path", type=str, required=True, 
                        help="Path to the input video file")
    args = parser.parse_args()
    
    test_player_tracking(args.video_path)