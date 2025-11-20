"""
Simple test script to verify pose detection works.
"""

import os
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

def main():
    # Check if track file exists
    track_file = "track_json/track_0229.json"
    
    if os.path.exists(track_file):
        print(f"Found track file: {track_file}")
        
        # Try to import the processing module
        try:
            from process_track_pose import process_track_file
            print("Successfully imported process_track_pose module")
            
            # Process the track file
            print("Processing track file...")
            process_track_file(track_file)
            print("Done!")
            
        except ImportError as e:
            print(f"Import error: {e}")
        except Exception as e:
            print(f"Error processing track file: {e}")
    else:
        print(f"Track file not found: {track_file}")

if __name__ == "__main__":
    main()