import cv2
import pandas as pd
import argparse
import os
import logging
import numpy as np

def setup_logging():
    """Configure logging with timestamps."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def visualize_rallies(video_file, csv_file, fps=30.0):
    """
    Visualize volleyball rallies from a video using processed CSV data.
    
    Args:
        video_file (str): Path to the input video file
        csv_file (str): Path to the processed CSV file with rally data
        fps (float): Frames per second of the video (default: 30.0)
    """
    logging.info(f"Loading video: {video_file}")
    logging.info(f"Loading CSV: {csv_file}")
    
    # Load CSV
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        logging.error(f"CSV file {csv_file} not found")
        raise
    except pd.errors.EmptyDataError:
        logging.error(f"CSV file {csv_file} is empty")
        raise
    except Exception as e:
        logging.error(f"Error loading CSV {csv_file}: {str(e)}")
        raise
    
    # Validate required columns
    required_columns = ['Rally_ID', 'X_interp', 'Y_interp', 'Time']
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        logging.error(f"Missing required columns in CSV: {missing}")
        raise ValueError(f"CSV must contain columns: {required_columns}")
    
    # Load video
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        logging.error(f"Failed to open video file: {video_file}")
        raise ValueError(f"Cannot open video file: {video_file}")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    logging.info(f"Video FPS: {video_fps}")
    
    # Group frames by Rally_ID
    rally_groups = df.groupby('Rally_ID')
    logging.info(f"Found {len(rally_groups)} rallies in CSV")
    
    # Initialize variables
    frame_idx = 0
    paused = False
    window_name = "Volleyball Rally Visualization"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    for rally_id, rally_df in rally_groups:
        logging.info(f"Visualizing Rally {rally_id}")
        
        # Get time range for the rally
        start_time = rally_df['Time'].min()
        end_time = rally_df['Time'].max()
        start_frame = int(start_time * video_fps)
        end_frame = int(end_time * video_fps)
        
        # Seek to start of rally
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_idx = start_frame
        
        # Create DataFrame index for quick lookup
        rally_df = rally_df.set_index('Time')
        
        while frame_idx <= end_frame:
            ret, frame = cap.read()
            if not ret:
                logging.warning(f"End of video reached at frame {frame_idx}")
                break
            
            # Calculate current time
            current_time = frame_idx / video_fps
            
            # Find closest CSV row by time
            if current_time in rally_df.index:
                row = rally_df.loc[current_time]
            else:
                # Interpolate or find nearest
                closest_time = rally_df.index[np.argmin(np.abs(rally_df.index - current_time))]
                row = rally_df.loc[closest_time]
            
            # Draw ball position
            x, y = int(row['X_interp']), int(row['Y_interp'])
            if not (np.isnan(x) or np.isnan(y)):
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Green circle for ball
            
            # Add text overlays
            cv2.putText(frame, f"Rally ID: {rally_id}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"Frame: {frame_idx}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"Time: {current_time:.2f}s", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Display frame
            cv2.imshow(window_name, frame)
            
            # Handle keyboard input
            key = cv2.waitKey(int(1000 / video_fps)) & 0xFF
            if key == ord('q'):
                logging.info("User terminated visualization")
                break
            elif key == ord(' '):  # Spacebar to pause/resume
                paused = not paused
                while paused:
                    key = cv2.waitKey(100) & 0xFF
                    if key == ord(' '):
                        paused = False
                    elif key == ord('q'):
                        logging.info("User terminated visualization")
                        cap.release()
                        cv2.destroyAllWindows()
                        return
            
            frame_idx += 1
        
        if key == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    logging.info("Visualization completed")

def main():
    parser = argparse.ArgumentParser(description='Visualize volleyball rallies from video and processed CSV')
    parser.add_argument('video_file', type=str, help='Path to input video file')
    parser.add_argument('csv_file', type=str, help='Path to processed CSV file with rally data')
    parser.add_argument('--fps', type=float, default=30.0, help='Frames per second of the video (default: 30.0)')
    
    args = parser.parse_args()
    
    setup_logging()
    
    try:
        visualize_rallies(args.video_file, args.csv_file, args.fps)
    except Exception as e:
        logging.error(f"Visualization failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()