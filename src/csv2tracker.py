import pandas as pd
import json
from ball_tracker import BallTracker
import os
import argparse
from tqdm import tqdm

def read_csv_detections(csv_path):
    """Read ball detections from CSV file and organize by frame."""
    df = pd.read_csv(csv_path)
    detections = {}
    for _, row in df.iterrows():
        frame = int(row["Frame"])
        visibility = int(row["Visibility"])
        x = float(row["X"])
        y = float(row["Y"])
        w = float(row.get("W", 20))
        h = float(row.get("H", 20))
        if visibility == 1 and x >= 0 and y >= 0 and w >= 0 and h >= 0:
            det = {
                "x1": x - w / 2,
                "y1": y - h / 2,
                "x2": x + w / 2,
                "y2": y + h / 2,
                "confidence": 1.0,
                "cls_id": 0,
            }
            detections[frame] = det
        else:
            detections[frame] = None
    return detections

def process_detections(detections, file_tracks):
    """Process detections and save ball tracks to text file with progress output."""
    # Initialize BallTracker
    tracker = BallTracker(
        buffer_size=1500, max_disappeared=30, max_distance=200, ball_diameter_cm=21.0
    )

    # Process each frame's detection with progress bar
    total_frames = len(detections)
    for frame_number in tqdm(sorted(detections.keys()), desc="Processing frames", unit="frame"):
        detection = detections.get(frame_number, None)
        det_list = [detection] if detection is not None else []

        # Update tracker with current frame's detection
        main_ball, tracks_dict, deleted_tracks = tracker.update(det_list, frame_number)

        # Save deleted tracks to file
        if file_tracks and len(deleted_tracks) > 0:
            with open(file_tracks, "a") as f:
                for track in deleted_tracks:
                    f.write(json.dumps(track.to_dict()) + "\n")

def main(csv_path):
    """Main function to process CSV and save tracks."""
    # Derive output track file path
    file_tracks = csv_path.replace("predict_ball.csv", ".txt")

    # Remove existing track file if it exists
    # Read detections
    detections = read_csv_detections(csv_path)

    # Process detections and save tracks
    process_detections(detections, file_tracks)
    print(f"Tracks saved to: {file_tracks}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Track ball trajectories from CSV detections and save to text file."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to the CSV file with ball detections",
    )
    args = parser.parse_args()
    main(args.csv_path)