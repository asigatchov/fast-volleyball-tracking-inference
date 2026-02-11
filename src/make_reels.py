import argparse
import json
import logging
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from constants import DEFAULT_CROP_ASPECT_RATIO, DEFAULT_FPS, DEFAULT_SMOOTH_WINDOW

LOG = logging.getLogger(__name__)


try:
    from scipy.signal import savgol_filter
except Exception:  # pragma: no cover - optional dependency
    savgol_filter = None


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def ensure_reels_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_single_track(track_json_path: str) -> Dict:
    """Load a single track JSON and normalize it for processing."""
    with open(track_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    positions = []
    for item in data["positions"]:
        xy, frame = item
        x, y = xy
        positions.append((float(x), float(y), int(frame)))

    return {
        "start_frame": data["start_frame"],
        "last_frame": data["last_frame"],
        "positions": positions,
    }


def interpolate_positions(
    frame_to_pos: Dict[int, Tuple[float, float]],
    start_frame: int,
    end_frame: int,
    mode: str,
) -> np.ndarray:
    frames = np.arange(start_frame, end_frame + 1)
    if not frame_to_pos:
        return np.zeros_like(frames, dtype=float)

    known_frames = np.array(sorted(frame_to_pos.keys()))
    known_x = np.array([frame_to_pos[f][0] for f in known_frames], dtype=float)

    if mode == "linear":
        return np.interp(frames, known_frames, known_x)

    x_values = np.empty_like(frames, dtype=float)
    last_x = known_x[0]
    idx = 0
    for i, frame_idx in enumerate(frames):
        while idx < len(known_frames) and known_frames[idx] <= frame_idx:
            last_x = known_x[idx]
            idx += 1
        x_values[i] = last_x
    return x_values


def smooth_values(values: np.ndarray, method: str, window: int, polyorder: int) -> np.ndarray:
    if method == "none":
        return values
    if method == "moving_avg":
        if len(values) < 2:
            return values
        window = max(3, min(window, len(values)))
        if window % 2 == 0:
            window += 1
        pad = window // 2
        padded = np.pad(values, (pad, pad), mode="edge")
        return np.convolve(padded, np.ones(window) / window, mode="valid")
    if method == "savitzky_golay":
        if savgol_filter is None:
            raise RuntimeError("savitzky_golay smoothing requires scipy")
        window = max(5, min(window, len(values)))
        if window % 2 == 0:
            window += 1
        polyorder = min(polyorder, window - 1)
        return savgol_filter(values, window_length=window, polyorder=polyorder, mode="interp")
    if method == "kalman":
        return kalman_smooth(values)
    raise ValueError(f"Unknown smoothing method: {method}")


def kalman_smooth(values: np.ndarray, process_var: float = 1e-3, meas_var: float = 1e-1) -> np.ndarray:
    if len(values) == 0:
        return values
    x_est = values[0]
    p = 1.0
    smoothed = []
    for z in values:
        p = p + process_var
        k = p / (p + meas_var)
        x_est = x_est + k * (z - x_est)
        p = (1 - k) * p
        smoothed.append(x_est)
    return np.array(smoothed)


def crop_frame(frame: np.ndarray, center_x: int, crop_width: int, padding: str) -> np.ndarray:
    frame_height, frame_width = frame.shape[:2]
    crop_width = min(crop_width, frame_width)
    left = center_x - crop_width // 2
    right = left + crop_width

    if padding == "none":
        left = max(0, min(left, frame_width - crop_width))
        right = left + crop_width
        return frame[:, left:right]

    pad_left = max(0, -left)
    pad_right = max(0, right - frame_width)
    if pad_left or pad_right:
        if padding == "mirror":
            border = cv2.BORDER_REFLECT_101
            padded = cv2.copyMakeBorder(frame, 0, 0, pad_left, pad_right, border)
        else:
            padded = cv2.copyMakeBorder(
                frame, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
        left += pad_left
        right += pad_left
        return padded[:, left:right]

    return frame[:, left:right]


def crop_and_save_track(
    video_path: str,
    track: Dict,
    output_path: str,
    visualize: bool = False,
    smoothing: str = "moving_avg",
    interpolation: str = "hold",
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
    smooth_polyorder: int = 2,
    margin: float = 0.0,
    padding: str = "none",
) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = DEFAULT_FPS
    delay = int(1000 / fps)

    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError("Failed to read first video frame")

    frame_height, frame_width = frame.shape[:2]
    crop_width = min(int(frame_height * DEFAULT_CROP_ASPECT_RATIO), frame_width)
    crop_height = frame_height

    out = None
    if not visualize:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (crop_width, crop_height))

    frame_to_pos = {int(p[2]): (p[0], p[1]) for p in track["positions"]}
    start_frame = track["start_frame"]
    end_frame = track["last_frame"]

    if not frame_to_pos:
        raise ValueError("Track contains no ball positions")

    x_values = interpolate_positions(frame_to_pos, start_frame, end_frame, interpolation)
    x_smooth = smooth_values(x_values, smoothing, smooth_window, smooth_polyorder)

    total_frames = end_frame - start_frame + 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for i, frame_idx in enumerate(
        tqdm(
            range(start_frame, end_frame + 1),
            desc="Saving frames",
            total=total_frames,
            unit="frame",
        )
    ):
        ret, frame = cap.read()
        if not ret:
            LOG.warning("Frame %s not read - video ended early.", frame_idx)
            break

        center_x = int(x_smooth[i])
        if margin:
            dx = x_smooth[i] - x_smooth[i - 1] if i > 0 else 0.0
            lead = margin * np.sign(dx) if dx != 0 else 0.0
            center_x = int(center_x + lead)

        cropped = crop_frame(frame, center_x, crop_width, padding)

        if out is not None:
            if cropped.shape[1] != crop_width:
                cropped = cv2.resize(cropped, (crop_width, crop_height))
            out.write(cropped)

        if visualize:
            cv2.imshow("Cropped", cropped)
            if cv2.waitKey(delay) & 0xFF == ord("q"):
                break

    cap.release()
    if out is not None:
        out.release()
    if visualize:
        cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize rally clips with ball-centered cropping."
    )
    parser.add_argument("--video_path", required=True, help="Path to video file")
    parser.add_argument("--track_json", help="Path to a single track JSON file")
    parser.add_argument("--track_jsons", nargs="+", help="Paths to multiple track JSON files")
    parser.add_argument("--json_dir", help="Directory with track_*.json files")
    parser.add_argument("--output_dir", default=None, help="Root output directory")
    parser.add_argument("--visualize", action="store_true", help="Show real-time cropped video")
    parser.add_argument(
        "--smoothing",
        choices=["none", "moving_avg", "savitzky_golay", "kalman"],
        default="moving_avg",
        help="Smoothing method for x trajectory",
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=DEFAULT_SMOOTH_WINDOW,
        help="Window size for smoothing",
    )
    parser.add_argument(
        "--smooth_polyorder",
        type=int,
        default=2,
        help="Polynomial order for Savitzky-Golay",
    )
    parser.add_argument(
        "--interpolation",
        choices=["hold", "linear"],
        default="hold",
        help="Interpolation method for missing detections",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="Lead offset in pixels based on movement direction",
    )
    parser.add_argument(
        "--padding",
        choices=["none", "mirror", "black"],
        default="none",
        help="Padding strategy when crop exceeds frame bounds",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.verbose)

    base_name = os.path.splitext(os.path.basename(args.video_path))[0]

    if args.track_jsons:
        json_paths = args.track_jsons
    elif args.track_json:
        json_paths = [args.track_json]
    elif args.json_dir:
        json_paths = sorted(
            [
                os.path.join(args.json_dir, f)
                for f in os.listdir(args.json_dir)
                if f.startswith("track_") and f.endswith(".json")
            ]
        )
    else:
        raise ValueError("Specify --track_json, --track_jsons, or --json_dir")

    reels_dir = os.path.join(args.output_dir, base_name, "reels") if args.output_dir else "reels"
    ensure_reels_dir(reels_dir)

    for track_json_path in json_paths:
        track = load_single_track(track_json_path)

        track_filename = os.path.basename(track_json_path)
        track_basename = os.path.splitext(track_filename)[0]
        try:
            track_number = track_basename.split("_")[-1]
            int(track_number)
        except (ValueError, IndexError):
            track_number = track_basename

        output_path = os.path.join(reels_dir, f"reel_{base_name}_{track_number}.mp4")

        crop_and_save_track(
            args.video_path,
            track,
            output_path,
            visualize=args.visualize,
            smoothing=args.smoothing,
            interpolation=args.interpolation,
            smooth_window=args.smooth_window,
            smooth_polyorder=args.smooth_polyorder,
            margin=args.margin,
            padding=args.padding,
        )

        if not args.visualize:
            LOG.info("Saved: %s", output_path)


if __name__ == "__main__":
    main()
