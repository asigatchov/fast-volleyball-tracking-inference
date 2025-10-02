# Player Tracking on Volleyball Court

This document describes the implementation of player tracking functionality using YOLO model for detecting volleyball players and mapping their positions to a volleyball court schematic.

## Overview

The player tracking system performs the following tasks:
1. Loads and runs the YOLO ONNX model for player detection
2. Detects players in video frames
3. Calculates player positions as the center of the bottom edge of their bounding boxes
4. Maps player positions from video coordinates to court coordinates using perspective transformation
5. Visualizes the results

## Files

- [src/player_tracker.py](src/player_tracker.py) - Main player tracking class
- [src/coort_coordinats.py](src/coort_coordinats.py) - Modified coordinate mapping script with player tracking integration
- [src/demo_player_tracking.py](src/demo_player_tracking.py) - Demonstration script
- [src/test_player_tracker.py](src/test_player_tracker.py) - Test script

## Implementation Details

### Player Detection

The system uses the YOLO model (`models_yolo/yolo11n.onnx`) to detect players in video frames. The detection process involves:

1. Preprocessing the frame to match the model input requirements
2. Running inference using ONNX Runtime
3. Post-processing the outputs to extract player bounding boxes

### Position Calculation

Player positions are calculated as the center of the bottom edge of their bounding boxes. This provides a more accurate representation of where the player is standing on the court.

### Coordinate Mapping

The system uses perspective transformation to map player positions from video coordinates to court coordinates:

1. Interactive selection of 4 points on the video and corresponding points on the court schematic
2. Calculation of the transformation matrix using `cv2.getPerspectiveTransform`
3. Application of the transformation to map player positions to court coordinates

### Visualization

The system provides visualization of:
- Player bounding boxes in the video
- Player positions as points
- Mapped positions on the court schematic (when available)

## Usage

To use the player tracking functionality:

1. Ensure the YOLO model is available at `models_yolo/yolo11n.onnx`
2. Run the coordinate mapping script with player tracking:
   ```bash
   python src/coort_coordinats.py --video_path path/to/video.mp4
   ```

3. Or run the demonstration script:
   ```bash
   python src/demo_player_tracking.py --video_path path/to/video.mp4 --visualize
   ```

## Integration with Existing Code

The player tracking functionality has been integrated into the existing coordinate mapping system in [src/coort_coordinats.py](src/coort_coordinats.py). This allows for seamless use of both ball tracking and player tracking in the same workflow.

## Future Improvements

1. Implement proper YOLO output decoding for accurate player detection
2. Add player ID tracking to follow players across frames
3. Improve the court schematic visualization
4. Add functionality to save player positions to CSV files
5. Optimize performance for real-time tracking

## Dependencies

The player tracking functionality requires the following dependencies (already included in the project):
- `onnxruntime` - For running the YOLO model
- `opencv-python` - For image processing and visualization
- `numpy` - For numerical computations