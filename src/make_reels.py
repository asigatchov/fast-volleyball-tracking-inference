import argparse
import json
import logging
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm
try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover - optional dependency
    Image = None
    ImageDraw = None
    ImageFont = None

try:
    from .constants import DEFAULT_CROP_ASPECT_RATIO, DEFAULT_FPS, DEFAULT_SMOOTH_WINDOW
except ImportError:
    from constants import DEFAULT_CROP_ASPECT_RATIO, DEFAULT_FPS, DEFAULT_SMOOTH_WINDOW

LOG = logging.getLogger(__name__)
WATERMARK_TEXT = "vb-ai.ru"
WATERMARK_FONT_PATH = Path(__file__).resolve().parent / "fonts" / "PlayfairDisplay-MediumItalic.ttf"
BLACK_TRANSITION_SECONDS = 0.3


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
    return load_track_from_payload(data)


def _positive_dimension(value: object) -> Optional[int]:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        return None
    return int(round(float(value)))


def tracking_frame_size(track_payload: Dict) -> Optional[Tuple[int, int]]:
    """Resolution in which track coordinates were produced.

    New track files carry both dimensions in ``tracking_scale``. Older files
    may only have width there, so ``court_info`` is used as a compatible
    fallback. A missing height is handled later with uniform width scaling.
    """
    tracking_scale = track_payload.get("tracking_scale") or {}
    court_info = track_payload.get("court_info") or {}
    width = _positive_dimension(tracking_scale.get("frame_width_px"))
    height = _positive_dimension(tracking_scale.get("frame_height_px"))
    width = width or _positive_dimension(court_info.get("image_width"))
    height = height or _positive_dimension(court_info.get("image_height"))
    if width is None:
        return None
    return width, height or 0


def load_track_from_payload(track_payload: Dict) -> Dict:
    positions = []
    for item in track_payload.get("positions", []):
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        xy, frame = item[0], item[1]
        if not isinstance(xy, (list, tuple)) or len(xy) < 2:
            continue
        x, y = xy[0], xy[1]
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            continue
        if not isinstance(frame, (int, float)):
            continue
        positions.append((float(x), float(y), int(frame)))

    start_frame = track_payload.get("start_frame")
    last_frame = track_payload.get("last_frame")
    if not isinstance(start_frame, int) or not isinstance(last_frame, int):
        raise ValueError("track_json must include integer start_frame and last_frame")

    return {
        "start_frame": start_frame,
        "last_frame": last_frame,
        "positions": positions,
        "tracking_frame_size": tracking_frame_size(track_payload),
    }


def scale_track_to_video(track: Dict, video_width: int, video_height: int) -> Dict:
    """Convert proxy-space positions to the opened video's pixel space."""
    source_size = track.get("tracking_frame_size")
    if not source_size:
        return track

    source_width, source_height = source_size
    if source_width <= 0 or video_width <= 0 or video_height <= 0:
        return track

    scale_x = float(video_width) / float(source_width)
    # Old track JSONs only stored width. Their proxy/full videos have the same
    # aspect ratio, so uniform width scaling is the safest compatible fallback.
    scale_y = (
        float(video_height) / float(source_height)
        if source_height and source_height > 0
        else scale_x
    )
    if math.isclose(scale_x, 1.0) and math.isclose(scale_y, 1.0):
        return track

    LOG.info(
        "Scaling track coordinates: %sx%s -> %sx%s (x=%.4f, y=%.4f)",
        source_width,
        source_height or "?",
        video_width,
        video_height,
        scale_x,
        scale_y,
    )
    scaled = dict(track)
    scaled["positions"] = [
        (float(x) * scale_x, float(y) * scale_y, int(frame))
        for x, y, frame in track["positions"]
    ]
    scaled["tracking_frame_size"] = (video_width, video_height)
    return scaled


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


def add_watermark_top_right(frame: np.ndarray, text: str = WATERMARK_TEXT) -> np.ndarray:
    """Draw a compact watermark in the top-right corner."""
    h, w = frame.shape[:2]
    alpha = 0.2
    scale = max(0.6, min(1.0, w / 1200.0)) * 3.0
    margin = max(10, int(round(scale * 16)))

    if Image is not None and ImageDraw is not None and ImageFont is not None and WATERMARK_FONT_PATH.exists():
        try:
            font_size = max(16, int(round(scale * 24)))
            font = ImageFont.truetype(str(WATERMARK_FONT_PATH), size=font_size)
            rgba = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
            overlay = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
            text_w = right - left
            text_h = bottom - top
            x = max(margin, w - text_w - margin)
            y = margin
            text_alpha = int(round(255 * alpha))

            draw.text((x + 2, y + 2), text, font=font, fill=(0, 0, 0, text_alpha))
            draw.text((x, y), text, font=font, fill=(255, 255, 255, text_alpha))

            composited = Image.alpha_composite(rgba, overlay).convert("RGB")
            return cv2.cvtColor(np.array(composited), cv2.COLOR_RGB2BGR)
        except Exception:
            LOG.exception("Failed to render watermark with TTF font, fallback to OpenCV font")

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(round(scale * 2)))
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    x = max(margin, w - text_w - margin)
    y = margin + text_h

    fallback_overlay = frame.copy()
    cv2.putText(
        fallback_overlay,
        text,
        (x + 2, y + 2),
        font,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(fallback_overlay, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.addWeighted(fallback_overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame


def black_fade_alpha(
    frame_index: int,
    total_frames: int,
    fps: float,
    fade_in: bool,
    fade_out: bool,
    duration_sec: float = BLACK_TRANSITION_SECONDS,
) -> float:
    """Brightness for a fade from/to black at a segment boundary."""
    if total_frames <= 1 or fps <= 0 or duration_sec <= 0:
        return 1.0
    fade_frames = min(
        max(2, int(round(fps * duration_sec))),
        max(2, total_frames // 2),
    )
    alpha = 1.0
    if fade_in and frame_index < fade_frames:
        alpha = min(alpha, frame_index / float(fade_frames - 1))
    fade_out_start = total_frames - fade_frames
    if fade_out and frame_index >= fade_out_start:
        alpha = min(alpha, (total_frames - 1 - frame_index) / float(fade_frames - 1))
    return float(max(0.0, min(1.0, alpha)))


def apply_black_fade(
    frame: np.ndarray,
    frame_index: int,
    total_frames: int,
    fps: float,
    fade_in: bool,
    fade_out: bool,
) -> np.ndarray:
    alpha = black_fade_alpha(
        frame_index, total_frames, fps, fade_in=fade_in, fade_out=fade_out
    )
    if alpha >= 1.0:
        return frame
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=0)


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
    fade_in: bool = False,
    fade_out: bool = False,
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
    track = scale_track_to_video(track, frame_width, frame_height)
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
            cropped = add_watermark_top_right(cropped)
        cropped = apply_black_fade(
            cropped,
            i,
            total_frames,
            fps,
            fade_in=fade_in,
            fade_out=fade_out,
        )

        if out is not None:
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


def crop_and_save_track_payload(
    video_path: str,
    track_payload: Dict,
    output_path: str,
    visualize: bool = False,
    smoothing: str = "moving_avg",
    interpolation: str = "hold",
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
    smooth_polyorder: int = 2,
    margin: float = 0.0,
    padding: str = "none",
    fade_in: bool = False,
    fade_out: bool = False,
) -> None:
    track = load_track_from_payload(track_payload)
    crop_and_save_track(
        video_path=video_path,
        track=track,
        output_path=output_path,
        visualize=visualize,
        smoothing=smoothing,
        interpolation=interpolation,
        smooth_window=smooth_window,
        smooth_polyorder=smooth_polyorder,
        margin=margin,
        padding=padding,
        fade_in=fade_in,
        fade_out=fade_out,
    )


def crop_and_save_track_payloads(
    video_path: str,
    track_payloads: List[Dict],
    output_path: str,
    visualize: bool = False,
    smoothing: str = "moving_avg",
    interpolation: str = "hold",
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
    smooth_polyorder: int = 2,
    margin: float = 0.0,
    padding: str = "none",
    fade_transition: bool = False,
) -> None:
    if not track_payloads:
        raise ValueError("track_payloads must not be empty")

    if len(track_payloads) == 1:
        crop_and_save_track_payload(
            video_path=video_path,
            track_payload=track_payloads[0],
            output_path=output_path,
            visualize=visualize,
            smoothing=smoothing,
            interpolation=interpolation,
            smooth_window=smooth_window,
            smooth_polyorder=smooth_polyorder,
            margin=margin,
            padding=padding,
        )
        return

    output_target = Path(output_path).resolve()
    output_target.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="reel_segments_") as tmp_dir_raw:
        tmp_dir = Path(tmp_dir_raw)
        segment_files: List[Path] = []
        for idx, payload in enumerate(track_payloads, start=1):
            segment_path = tmp_dir / f"segment_{idx:04d}.mp4"
            crop_and_save_track_payload(
                video_path=video_path,
                track_payload=payload,
                output_path=str(segment_path),
                visualize=visualize,
                smoothing=smoothing,
                interpolation=interpolation,
                smooth_window=smooth_window,
                smooth_polyorder=smooth_polyorder,
                margin=margin,
                padding=padding,
                fade_in=fade_transition and idx > 1,
                fade_out=fade_transition and idx < len(track_payloads),
            )
            segment_files.append(segment_path)

        concat_list = tmp_dir / "concat_list.txt"
        concat_list.write_text(
            "\n".join([f"file '{segment_path.name}'" for segment_path in segment_files]),
            encoding="utf-8",
        )

        concat_cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-c",
            "copy",
            str(output_target),
        ]
        try:
            subprocess.run(concat_cmd, cwd=tmp_dir, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"ffmpeg concat failed: {exc.stderr[-2000:] if exc.stderr else str(exc)}"
            ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize rally clips with ball-centered cropping."
    )
    parser.add_argument("--video_path", required=True, help="Path to video file")
    parser.add_argument("--track_json", help="Path to a single track JSON file")
    parser.add_argument("--track_jsons", nargs="+", help="Paths to multiple track JSON files")
    parser.add_argument(
        "--fade-transition",
        "--black-transition",
        dest="fade_transition",
        action="store_true",
        help=(
            "Join --track_jsons with a 0.3s fade to black followed by a "
            "0.3s fade from black"
        ),
    )
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

    if args.fade_transition and (not args.track_jsons or len(args.track_jsons) < 2):
        parser.error("--fade-transition requires at least two files in --track_jsons")

    if args.track_jsons:
        track_payloads = []
        for track_json_path in args.track_jsons:
            with open(track_json_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict):
                raise ValueError(f"Track JSON must contain an object: {track_json_path}")
            track_payloads.append(payload)

        output_path = os.path.join(reels_dir, f"reel_{base_name}_combined.mp4")
        crop_and_save_track_payloads(
            video_path=args.video_path,
            track_payloads=track_payloads,
            output_path=output_path,
            visualize=args.visualize,
            smoothing=args.smoothing,
            interpolation=args.interpolation,
            smooth_window=args.smooth_window,
            smooth_polyorder=args.smooth_polyorder,
            margin=args.margin,
            padding=args.padding,
            fade_transition=args.fade_transition,
        )
        if not args.visualize:
            LOG.info("Saved combined reel: %s", output_path)
        return

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
