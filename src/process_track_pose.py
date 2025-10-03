"""
Script to process track JSON files and add pose detection when ball is near players.
This script analyzes track_0229.json and adds pose information.
"""

import json
import os
import numpy as np
from typing import List, Tuple, Dict, Optional


def calculate_distance(point1: List[float], point2: List[float]) -> float:
    """Calculate Euclidean distance between two points."""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def simulate_player_detection(frame_num: int) -> List[Tuple[int, int, int, int]]:
    """
    Simulate player detection for a given frame.
    In a real implementation, this would use YOLO or another detection model.
    
    Returns:
        List of player bounding boxes (x, y, w, h)
    """
    # For demonstration, we'll create some fixed player positions
    # In practice, these would come from actual player detection
    players = [
        (1000, 600, 80, 160),  # Player 1
        (1200, 500, 80, 160),  # Player 2
        (800, 400, 80, 160),   # Player 3
    ]
    
    # Add some variation based on frame number
    variation = (frame_num % 10) * 2
    varied_players = []
    for x, y, w, h in players:
        varied_players.append((x + variation, y + variation, w, h))
    
    return varied_players


def is_ball_near_player(ball_pos: List[float], player_bbox: Tuple[int, int, int, int], 
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
    dist = calculate_distance(ball_pos, player_center)
    
    return dist <= threshold


def find_closest_player(ball_pos: List[float], 
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
        dist = calculate_distance(ball_pos, player_center)
        distances.append(dist)
        
    closest_idx = np.argmin(distances)
    return int(closest_idx)


def simulate_pose_detection(player_bbox: Tuple[int, int, int, int]) -> Dict:
    """
    Simulate pose detection for a player.
    In a real implementation, this would use MediaPipe Pose.
    
    Args:
        player_bbox: Player bounding box (x, y, w, h)
        
    Returns:
        Dictionary with simulated pose data
    """
    x, y, w, h = player_bbox
    
    # Simulate some pose landmarks (in a real implementation, these would come from MediaPipe)
    # Using a simple stick figure representation
    landmarks = [
        {"x": x + w/2, "y": y + h/4, "z": 0, "visibility": 1.0},      # Head
        {"x": x + w/2, "y": y + h/2, "z": 0, "visibility": 1.0},      # Shoulders
        {"x": x + w/4, "y": y + h/2, "z": 0, "visibility": 1.0},      # Left shoulder
        {"x": x + 3*w/4, "y": y + h/2, "z": 0, "visibility": 1.0},    # Right shoulder
        {"x": x + w/4, "y": y + 3*h/4, "z": 0, "visibility": 1.0},    # Left elbow
        {"x": x + 3*w/4, "y": y + 3*h/4, "z": 0, "visibility": 1.0},  # Right elbow
        {"x": x + w/4, "y": y + h, "z": 0, "visibility": 1.0},        # Left hand
        {"x": x + 3*w/4, "y": y + h, "z": 0, "visibility": 1.0},      # Right hand
        {"x": x + w/2, "y": y + 3*h/4, "z": 0, "visibility": 1.0},    # Hips
        {"x": x + w/4, "y": y + h, "z": 0, "visibility": 1.0},        # Left knee
        {"x": x + 3*w/4, "y": y + h, "z": 0, "visibility": 1.0},      # Right knee
        {"x": x + w/4, "y": y + 5*h/4, "z": 0, "visibility": 1.0},    # Left foot
        {"x": x + 3*w/4, "y": y + 5*h/4, "z": 0, "visibility": 1.0},  # Right foot
    ]
    
    return {
        "landmarks": landmarks,
        "bbox": [x, y, w, h]
    }


def process_track_file(track_file: str, output_dir: str = "track_json_with_pose") -> None:
    """
    Process a track file and add pose detection when ball is near players.
    
    Args:
        track_file: Path to track JSON file
        output_dir: Directory to save updated JSON files
    """
    print(f"Processing track file: {track_file}")
    
    # Load track data
    with open(track_file, 'r') as f:
        track_data = json.load(f)
    
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
        
        # Simulate player detection for this frame
        player_boxes = simulate_player_detection(frame_num)
        
        # Check if ball is near any player
        nearby_player_idx = None
        if player_boxes:
            nearby_player_idx = find_closest_player(position, player_boxes)
        
        # If ball is near a player, detect pose
        pose_data = None
        if nearby_player_idx is not None:
            player_bbox = player_boxes[nearby_player_idx]
            if is_ball_near_player(position, player_bbox):
                pose_data = simulate_pose_detection(player_bbox)
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
    
    print(f"Saved updated track with pose data to: {output_file}")


def main():
    """Main function to process the track_0229.json file."""
    track_file = "track_json/track_0229.json"
    
    if os.path.exists(track_file):
        process_track_file(track_file)
        print("Processing complete!")
        print("Check the 'track_json_with_pose' directory for the updated file.")
    else:
        print(f"Track file not found: {track_file}")
        print("Please make sure the file exists in the track_json directory.")


if __name__ == "__main__":
    main()