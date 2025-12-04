"""
Utility script to check if required files for player tracking are available.
"""

import os

def check_files():
    """Check if required files exist."""
    required_files = [
        "models_yolo/yolo11n.onnx",
        "models_yolo/yolo11n_vb.onnx",
        "src/player_tracker.py",
        "src/coort_coordinats.py",
        "src/demo_player_tracking_v2.py"
    ]
    
    print("Checking required files for player tracking:")
    print("-" * 50)
    
    all_files_exist = True
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path} - FOUND")
        else:
            print(f"✗ {file_path} - NOT FOUND")
            all_files_exist = False
    
    print("-" * 50)
    
    if all_files_exist:
        print("All required files are present!")
        print("\nTo use player tracking, run:")
        print("  python src/coort_coordinats.py --video_path path/to/video.mp4")
        print("\nOr to run the demo with tracking:")
        print("  python src/demo_player_tracking_v2.py --video_path path/to/video.mp4 --visualize")
    else:
        print("Some required files are missing.")
        print("Please ensure you have downloaded the YOLO models to the models_yolo directory.")
    
    return all_files_exist

if __name__ == "__main__":
    check_files()