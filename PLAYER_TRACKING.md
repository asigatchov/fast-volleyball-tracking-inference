# Player Tracking on Volleyball Court

This document describes the implementation of player tracking functionality using YOLO model for detecting volleyball players and mapping their positions to a volleyball court schematic.

## Overview

The player tracking system performs the following tasks:
1. Loads and runs the YOLO ONNX model for player detection
2. Detects players in video frames
3. Calculates player positions as the center of the bottom edge of their bounding boxes
4. Maps player positions from video coordinates to court coordinates using perspective transformation
5. Tracks players with IDs using a DeepSORT-like tracking algorithm
6. Filters players to only show those within the court area
7. Visualizes player positions with IDs (no bounding boxes)

## Files

- [src/player_tracker.py](src/player_tracker.py) - Main player tracking class with DeepSORT-like tracking
- [src/coort_coordinats.py](src/coort_coordinats.py) - Modified coordinate mapping script with player tracking integration
- [src/demo_player_tracking.py](src/demo_player_tracking.py) - Original demonstration script
- [src/demo_player_tracking_v2.py](src/demo_player_tracking_v2.py) - Updated demonstration script with tracking features
- [src/test_player_tracker.py](src/test_player_tracker.py) - Test script
- [src/check_setup.py](src/check_setup.py) - Utility to verify required files

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

### Player Tracking with IDs

A DeepSORT-like tracking algorithm has been implemented to maintain consistent player IDs across frames:

1. Each detected player is assigned a unique ID
2. Player positions are tracked across frames using proximity matching
3. Disappeared players are handled with a timeout mechanism
4. New players are assigned new IDs when they appear

### Court Boundary Filtering

The system filters players to only show those within the court area:

1. Court boundary is defined by 4 points
2. Players outside the court boundary are filtered out
3. Only players within the court are tracked and displayed

### Visualization

The system provides visualization of:
- Player positions as points (no bounding boxes)
- Player IDs displayed next to each player
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
   python src/demo_player_tracking_v2.py --video_path path/to/video.mp4 --visualize
   ```

## Integration with Existing Code

The player tracking functionality has been integrated into the existing coordinate mapping system in [src/coort_coordinats.py](src/coort_coordinats.py). This allows for seamless use of both ball tracking and player tracking in the same workflow.

## Key Features Implemented

1. **YOLO Model Integration**: The system loads and uses the YOLO ONNX model for player detection
2. **Player Position Calculation**: Player positions are calculated as the center of the bottom edge of their bounding boxes for accuracy
3. **DeepSORT-like Tracking**: Players are tracked with consistent IDs across frames
4. **Court Boundary Filtering**: Only players within the court area are displayed
5. **Clean Visualization**: Only points with player IDs are shown (no bounding boxes)
6. **Coordinate Mapping**: Player positions are mapped from video coordinates to court coordinates
7. **Reusable Design**: The functionality is implemented in a modular, reusable way

## Future Improvements

1. Implement proper YOLO output decoding for accurate player detection
2. Improve the tracking algorithm with more sophisticated matching
3. Add player movement analysis and statistics
4. Improve the court schematic visualization
5. Add functionality to save player positions to CSV files
6. Optimize performance for real-time tracking

## Dependencies

The player tracking functionality requires the following dependencies (already included in the project):
- `onnxruntime` - For running the YOLO model
- `opencv-python` - For image processing and visualization
- `numpy` - For numerical computations
- `scipy` - For distance calculations in tracking