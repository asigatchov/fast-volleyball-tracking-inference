#!/usr/bin/env python3
"""
Test script to demonstrate the --visualize parameter for pose detection.
"""

import argparse
import os
import sys

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.pose_detector import add_pose_to_track_json


def main():
    parser = argparse.ArgumentParser(description="Test pose detection with visualization")
    parser.add_argument("--track_file", type=str, required=True, help="Path to track JSON file")
    parser.add_argument("--video_path", type=str, required=True, help="Path to video file")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.track_file):
        print(f"Error: Track file not found: {args.track_file}")
        return 1
        
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return 1
    
    print(f"Processing track file: {args.track_file}")
    print(f"Using video: {args.video_path}")
    print(f"Visualization enabled: {args.visualize}")
    
    try:
        add_pose_to_track_json(
            track_file=args.track_file,
            video_path=args.video_path,
            visualize=args.visualize
        )
        print("Processing completed successfully!")
    except Exception as e:
        print(f"Error during processing: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())