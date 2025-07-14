import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import argparse
import os
import logging


def setup_logging():
    """Configure logging with timestamps."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def process_volleyball_data(
    csv_file,
    max_pause=2.0,
    fps=30.0,
    min_flight_duration=1.0,
    time_extension=1.0,
    min_rally_frames=5,
):
    """
    Process volleyball detection CSV to identify rallies and filter pauses.
    Visibility == 0 indicates no ball detection.
    Pauses > max_pause seconds separate rallies.
    Flight trajectories < min_flight_duration seconds are considered passes and excluded.
    Extends each rally by time_extension seconds before and after.

    Args:
        csv_file (str): Path to input CSV file
        max_pause (float): Maximum pause duration in seconds (default: 2.0)
        fps (float): Frames per second for time calculation if no timestamp (default: 30.0)
        min_flight_duration (float): Minimum flight duration in seconds (default: 1.0)
        time_extension (float): Seconds to extend before and after each rally (default: 1.0)
        min_rally_frames (int): Minimum number of frames for a valid rally (default: 5)

    Returns:
        pd.DataFrame: Processed data with rally IDs and interpolated coordinates
    """
    logging.info(f"Processing file: {csv_file}")
    if fps <= 0:
        logging.error("FPS must be greater than 0")
        raise ValueError("FPS must be greater than 0")

    # Load data
    try:
        df = pd.read_csv(csv_file)
        if df.empty:
            logging.error(f"File {csv_file} is empty")
            raise pd.errors.EmptyDataError(f"File {csv_file} is empty")
    except FileNotFoundError:
        logging.error(f"File {csv_file} not found")
        raise
    except Exception as e:
        logging.error(f"Error loading {csv_file}: {str(e)}")
        raise

    # Validate columns
    required_columns = ["Visibility", "X", "Y"]
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        logging.error(f"Missing required columns: {missing}")
        raise ValueError(f"CSV must contain columns: {required_columns}")

    # Convert data types
    df["Visibility"] = df["Visibility"].astype(int)
    df["X"] = pd.to_numeric(df["X"], errors="coerce")
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")

    # Calculate time
    if "Timestamp" in df.columns:
        df["Time"] = pd.to_numeric(df["Timestamp"], errors="coerce")
        logging.info("Using Timestamp column for time calculation")
    else:
        df["Time"] = df.index / fps
        logging.info(f"Using {fps} FPS for time calculation")

    # Identify pauses and rallies
    df["Is_Pause"] = df["Visibility"] == 0
    df["Pause_Group"] = (df["Is_Pause"] != df["Is_Pause"].shift()).cumsum()
    pause_info = df.groupby("Pause_Group").agg(
        {"Time": ["min", "max"], "Is_Pause": "first"}
    )
    pause_info.columns = ["Time_Min", "Time_Max", "Is_Pause"]
    pause_info["Duration"] = pause_info["Time_Max"] - pause_info["Time_Min"]

    # Assign rally IDs based on pauses > max_pause
    rally_id = 0
    df["Rally_ID"] = -1
    current_rally_start = None
    for idx, row in pause_info.iterrows():
        group_indices = df[df["Pause_Group"] == idx].index
        if row["Is_Pause"] and row["Duration"] > max_pause:
            if current_rally_start is not None:
                df.loc[current_rally_start : group_indices[0] - 1, "Rally_ID"] = (
                    rally_id
                )
                rally_id += 1
            current_rally_start = None
        else:
            if current_rally_start is None:
                current_rally_start = group_indices[0]
    # Assign final rally if exists
    if current_rally_start is not None:
        df.loc[current_rally_start:, "Rally_ID"] = rally_id

    # Filter out unassigned frames (Rally_ID == -1)
    df = df[df["Rally_ID"] >= 0].copy()
    if df.empty:
        logging.warning("No valid rallies found")
        return pd.DataFrame()

    # Extend rallies
    extended_rallies = []
    for rid in df["Rally_ID"].unique():
        rally_df = df[df["Rally_ID"] == rid]
        time_min = rally_df["Time"].min() - time_extension
        time_max = rally_df["Time"].max() + time_extension
        extended_frames = df[(df["Time"] >= time_min) & (df["Time"] <= time_max)].copy()
        extended_frames["Rally_ID"] = rid
        extended_rallies.append(extended_frames)
        logging.debug(
            f"Rally {rid}: Extended from {rally_df['Time'].min():.2f}s to {rally_df['Time'].max():.2f}s "
            f"to {time_min:.2f}s to {time_max:.2f}s"
        )

    if not extended_rallies:
        logging.warning("No rallies after extension")
        return pd.DataFrame()

    result_df = pd.concat(extended_rallies, ignore_index=True)

    # Filter short flights within each rally
    filtered_rallies = []
    rally_id = 0
    for rid in result_df["Rally_ID"].unique():
        rally_df = result_df[result_df["Rally_ID"] == rid].copy()
        rally_df["Flight_Group"] = (
            rally_df["Visibility"] != rally_df["Visibility"].shift()
        ).cumsum()
        flight_groups = rally_df.groupby("Flight_Group").filter(
            lambda g: g["Visibility"].iloc[0] == 1
            and (g["Time"].max() - g["Time"].min()) >= min_flight_duration
        )
        if not flight_groups.empty:
            flight_groups["Rally_ID"] = rally_id
            filtered_rallies.append(flight_groups)
            rally_id += 1

    if not filtered_rallies:
        logging.warning("No valid rallies remain after filtering short flights")
        return pd.DataFrame()

    result_df = pd.concat(filtered_rallies, ignore_index=True)

    # Interpolate coordinates per flight segment
    result_df["X_interp"] = result_df["X"]
    result_df["Y_interp"] = result_df["Y"]
    for rid in result_df["Rally_ID"].unique():
        rally_df = result_df[result_df["Rally_ID"] == rid].copy()
        rally_df["Flight_Group"] = (
            rally_df["Visibility"] != rally_df["Visibility"].shift()
        ).cumsum()
        for fgid in rally_df["Flight_Group"].unique():
            flight_df = rally_df[rally_df["Flight_Group"] == fgid]
            if flight_df["Visibility"].iloc[0] == 1 and len(flight_df) > 1:
                visible = flight_df[flight_df["Visibility"] == 1].dropna(
                    subset=["X", "Y"]
                )
                if len(visible) > 1:
                    try:
                        f_x = interp1d(
                            visible["Time"],
                            visible["X"],
                            kind="linear",
                            fill_value="extrapolate",
                        )
                        f_y = interp1d(
                            visible["Time"],
                            visible["Y"],
                            kind="linear",
                            fill_value="extrapolate",
                        )
                        indices = flight_df.index
                        result_df.loc[indices, "X_interp"] = f_x(flight_df["Time"])
                        result_df.loc[indices, "Y_interp"] = f_y(flight_df["Time"])
                    except Exception as e:
                        logging.warning(
                            f"Interpolation failed for rally {rid}, flight {fgid}: {str(e)}"
                        )
                else:
                    logging.debug(
                        f"Rally {rid}, flight {fgid}: Insufficient valid points for interpolation"
                    )

        # Clip interpolated values
        result_df.loc[result_df["Rally_ID"] == rid, "X_interp"] = result_df.loc[
            result_df["Rally_ID"] == rid, "X_interp"
        ].clip(lower=df["X"].min(), upper=df["X"].max())
        result_df.loc[result_df["Rally_ID"] == rid, "Y_interp"] = result_df.loc[
            result_df["Rally_ID"] == rid, "Y_interp"
        ].clip(lower=df["Y"].min(), upper=df["Y"].max())

    # Filter rallies with insufficient frames
    rally_sizes = result_df.groupby("Rally_ID").size()
    valid_rallies = rally_sizes[rally_sizes >= min_rally_frames].index
    result_df = result_df[result_df["Rally_ID"].isin(valid_rallies)]

    if result_df.empty:
        logging.warning(f"No rallies with >= {min_rally_frames} frames")
        return pd.DataFrame()

    logging.info(
        f"Processed {len(result_df)} frames with {result_df['Rally_ID'].nunique()} rallies"
    )
    return result_df


def main():
    parser = argparse.ArgumentParser(description="Process volleyball detection data")
    parser.add_argument("csv_file", type=str, help="Path to input CSV file")
    parser.add_argument(
        "--max_pause",
        type=float,
        default=2.0,
        help="Maximum pause duration in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frames per second for time calculation (default: 30.0)",
    )
    parser.add_argument(
        "--min_flight_duration",
        type=float,
        default=1.0,
        help="Minimum flight duration in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--time_extension",
        type=float,
        default=1.0,
        help="Seconds to extend before and after each rally (default: 1.0)",
    )
    parser.add_argument(
        "--min_rally_frames",
        type=int,
        default=5,
        help="Minimum number of frames for a valid rally (default: 5)",
    )
    parser.add_argument("--output", type=str, help="Output CSV file path")

    args = parser.parse_args()

    setup_logging()

    try:
        result = process_volleyball_data(
            args.csv_file,
            max_pause=args.max_pause,
            fps=args.fps,
            min_flight_duration=args.min_flight_duration,
            time_extension=args.time_extension,
            min_rally_frames=args.min_rally_frames,
        )

        if result.empty:
            logging.warning("No data to save")
            return

        # Determine output file
        output_file = args.output or "processed_" + os.path.basename(args.csv_file)

        # Check if output file exists
        if os.path.exists(output_file):
            logging.warning(f"Output file {output_file} already exists, overwriting")

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
