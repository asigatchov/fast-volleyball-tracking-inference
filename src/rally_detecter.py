import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import argparse
import os
import logging

def setup_logging():
    """Configure logging with timestamps."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def process_volleyball_data(csv_file, max_pause=2.0, fps=30.0):
    """
    Process volleyball detection CSV to identify rallies and filter pauses.
    Visibility == 0 indicates no ball detection.
    Pauses > 2 seconds (60 frames at 30 FPS) separate rallies.
    
    Args:
        csv_file (str): Path to input CSV file
        max_pause (float): Maximum pause duration in seconds (default: 2.0)
        fps (float): Frames per second for time calculation if no timestamp (default: 30.0)
    
    Returns:
        pd.DataFrame: Processed data with rally IDs and interpolated coordinates
    """
    logging.info(f"Processing file: {csv_file}")
    logging.info(f"Max pause: {max_pause}s ({max_pause * fps} frames at {fps} FPS)")
    
    # Load data
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        logging.error(f"File {csv_file} not found")
        raise
    except pd.errors.EmptyDataError:
        logging.error(f"File {csv_file} is empty")
        raise
    except Exception as e:
        logging.error(f"Error loading {csv_file}: {str(e)}")
        raise
    
    # Validate columns
    required_columns = ['Visibility', 'X', 'Y']
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        logging.error(f"Missing required columns: {missing}")
        raise ValueError(f"CSV must contain columns: {required_columns}")
    
    # Convert data types
    df['Visibility'] = df['Visibility'].astype(int)
    df['X'] = pd.to_numeric(df['X'], errors='coerce')
    df['Y'] = pd.to_numeric(df['Y'], errors='coerce')
    
    # Calculate time
    if 'Timestamp' in df.columns:
        df['Time'] = pd.to_numeric(df['Timestamp'], errors='coerce')
        logging.info("Using Timestamp column for time calculation")
    else:
        df['Time'] = df.index / fps
        logging.info(f"Using {fps} FPS for time calculation")
    
    # Identify rallies by detecting sequences separated by pauses > max_pause
    df['Is_Pause'] = df['Visibility'] == 0
    df['Pause_Group'] = (df['Is_Pause'] != df['Is_Pause'].shift()).cumsum()
    
    # Calculate duration of each pause group
    pause_info = df.groupby('Pause_Group').agg({
        'Time': ['min', 'max'],
        'Is_Pause': 'first'
    })
    pause_info.columns = ['Time_Min', 'Time_Max', 'Is_Pause']
    pause_info['Duration'] = pause_info['Time_Max'] - pause_info['Time_Min']
    
    # Identify rally boundaries (non-pause groups or pauses <= max_pause)
    rally_groups = []
    current_rally = []
    rally_id = 0
    rallies = []
    
    for idx, row in pause_info.iterrows():
        group = df[df['Pause_Group'] == idx]
        if row['Is_Pause'] and row['Duration'] > max_pause:
            # End current rally if pause is too long
            if current_rally:
                rally_df = pd.concat(current_rally, ignore_index=True)
                rally_df['Rally_ID'] = rally_id
                rallies.append(rally_df)
                rally_id += 1
                current_rally = []
            logging.debug(f"Pause detected (group {idx}): {row['Duration']:.2f}s > {max_pause}s")
        else:
            # Include non-pause or short pause in current rally
            current_rally.append(group)
            logging.debug(f"Added group {idx} to rally (pause: {row['Duration']:.2f}s)")
    
    # Include final rally if exists
    if current_rally:
        rally_df = pd.concat(current_rally, ignore_index=True)
        rally_df['Rally_ID'] = rally_id
        rallies.append(rally_df)
    
    if not rallies:
        logging.warning("No valid rallies found")
        return pd.DataFrame()
    
    # Combine all rallies
    result_df = pd.concat(rallies, ignore_index=True)
    
    # Interpolate coordinates within each rally
    for rally_id in result_df['Rally_ID'].unique():
        rally_df = result_df[result_df['Rally_ID'] == rally_id].copy()
        visible = rally_df[rally_df['Visibility'] == 1]
        
        if len(visible) > 1:
            try:
                f_x = interp1d(visible['Time'], visible['X'], kind='linear', fill_value='extrapolate')
                f_y = interp1d(visible['Time'], visible['Y'], kind='linear', fill_value='extrapolate')
                
                result_df.loc[result_df['Rally_ID'] == rally_id, 'X_interp'] = f_x(rally_df['Time'])
                result_df.loc[result_df['Rally_ID'] == rally_id, 'Y_interp'] = f_y(rally_df['Time'])
                
                # Bounds checking
                result_df['X_interp'] = result_df['X_interp'].clip(min=df['X'].min(), max=df['X'].max())
                result_df['Y_interp'] = result_df['Y_interp'].clip(min=df['Y'].min(), max=df['Y'].max())
            except Exception as e:
                logging.warning(f"Interpolation failed for rally {rally_id}: {str(e)}")
                result_df.loc[result_df['Rally_ID'] == rally_id, 'X_interp'] = rally_df['X']
                result_df.loc[result_df['Rally_ID'] == rally_id, 'Y_interp'] = rally_df['Y']
        else:
            result_df.loc[result_df['Rally_ID'] == rally_id, 'X_interp'] = rally_df['X']
            result_df.loc[result_df['Rally_ID'] == rally_id, 'Y_interp'] = rally_df['Y']
    
    # Filter rallies with sufficient frames
    rally_sizes = result_df.groupby('Rally_ID').size()
    valid_rallies = rally_sizes[rally_sizes > 5].index
    result_df = result_df[result_df['Rally_ID'].isin(valid_rallies)]
    
    logging.info(f"Processed {len(result_df)} frames with {result_df['Rally_ID'].nunique()} rallies")
    return result_df

def main():
    parser = argparse.ArgumentParser(description='Process volleyball detection data')
    parser.add_argument('csv_file', type=str, help='Path to input CSV file')
    parser.add_argument('--max_pause', type=float, default=2.0, help='Maximum pause duration in seconds (default: 2.0)')
    parser.add_argument('--fps', type=float, default=30.0, help='Frames per second for time calculation (default: 30.0)')
    parser.add_argument('--output', type=str, help='Output CSV file path')
    
    args = parser.parse_args()
    
    setup_logging()
    
    try:
        result = process_volleyball_data(args.csv_file, args.max_pause, args.fps)
        
        # Determine output file
        if args.output:
            output_file = args.output
        else:
            output_file = 'processed_' + os.path.basename(args.csv_file)
        
        # Save results
        result.to_csv(output_file, index=False)
        logging.info(f"Results saved to {output_file}")
        print(f"Processed {len(result)} frames")
        print(f"Found {result['Rally_ID'].nunique()} rallies")
        print(f"Results saved to {output_file}")
        
    except Exception as e:
        logging.error(f"Processing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()