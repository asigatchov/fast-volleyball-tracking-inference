#!/usr/bin/env python3
"""
Track Visualization & Export Tool
--------------------------------
Loads ball tracking data from JSON files and visualizes or exports video clips
with overlaid track positions. Supports:
- Interactive preview
- Single combined output video
- Individual track videos
"""

import argparse
import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import cv2
from tqdm import tqdm

from ball_tracker import Track
from constants import DEFAULT_FADE_DURATION

LOG = logging.getLogger(__name__)


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def resolve_video_basename(video_path: str) -> str:
    return os.path.splitext(os.path.basename(video_path))[0]


class BaseExporter:
    def open_track(self, track_id: int) -> bool:
        return True

    def write(self, frame) -> None:
        raise NotImplementedError

    def close_track(self) -> None:
        return None

    def close(self) -> None:
        return None


class CombinedVideoExporter(BaseExporter):
    def __init__(self, output_path: str, fps: float, size: Tuple[int, int]) -> None:
        self.output_path = output_path
        self._writer = None

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(output_path, fourcc, fps, size)
        if not self._writer.isOpened():
            raise ValueError(f"Failed to create combined video writer: {output_path}")

    def write(self, frame) -> None:
        if not self._writer:
            return
        try:
            self._writer.write(frame)
        except cv2.error as exc:
            LOG.error("Failed to write combined frame: %s", exc)

    def close(self) -> None:
        if self._writer:
            self._writer.release()
            self._writer = None


class SplitClipsExporter(BaseExporter):
    def __init__(self, split_dir: str, fps: float, size: Tuple[int, int]) -> None:
        self.split_dir = split_dir
        self._fps = fps
        self._size = size
        self._writer = None
        self._track_path = None
        os.makedirs(split_dir, exist_ok=True)

    def open_track(self, track_id: int) -> bool:
        self.close_track()
        self._track_path = os.path.join(self.split_dir, f"track_{track_id:04d}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(self._track_path, fourcc, self._fps, self._size)
        if not self._writer.isOpened():
            LOG.error("Failed to create video writer for %s", self._track_path)
            self._writer = None
            return False
        return True

    def write(self, frame) -> None:
        if not self._writer:
            return
        try:
            self._writer.write(frame)
        except cv2.error as exc:
            LOG.error("Failed to write split frame: %s", exc)

    def close_track(self) -> None:
        if self._writer:
            self._writer.release()
            self._writer = None

    def close(self) -> None:
        self.close_track()


class TrackProcessor:
    def __init__(
        self,
        json_dir: str,
        video_path: str,
        output_path: Optional[str] = None,
        split_dir: Optional[str] = None,
        fps: float = 30.0,
    ) -> None:
        self.json_dir = json_dir
        self.video_path = video_path
        self.output_path = output_path
        self.split_dir = split_dir
        self.fps = fps
        self.tracks: List[Track] = []
        self.total_processed_frames = 0
        self.total_processing_time = 0.0

    def _validate_json_dir(self) -> None:
        if not self.json_dir:
            raise ValueError("json_dir is required. Provide --json_dir or --output_dir.")
        if not os.path.exists(self.json_dir):
            raise FileNotFoundError(f"JSON directory not found: {self.json_dir}")

    def _load_tracks_from_json(self) -> None:
        self._validate_json_dir()

        json_files = sorted(
            [
                f
                for f in os.listdir(self.json_dir)
                if f.startswith("track_") and f.endswith(".json")
            ]
        )

        for filename in json_files:
            file_path = os.path.join(self.json_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            track = Track.from_dict(data)
            self.tracks.append(track)

        LOG.info("Loaded %s track(s) from %s", len(self.tracks), self.json_dir)

    def _validate_video(self) -> None:
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

    def _create_exporter(self, fps: float, size: Tuple[int, int]) -> Optional[BaseExporter]:
        if self.split_dir:
            return SplitClipsExporter(self.split_dir, fps, size)
        if self.output_path:
            return CombinedVideoExporter(self.output_path, fps, size)
        return None

    def _write_fade_out(self, exporter: BaseExporter, frame, fade_frames: int) -> None:
        fade_pbar = tqdm(
            total=fade_frames,
            desc="Fade-out",
            unit="frame",
            leave=False,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} | {rate_fmt}",
        )
        fade_start = time.time()
        for i in range(fade_frames):
            alpha = 1.0 - (i / fade_frames)
            faded = cv2.convertScaleAbs(frame, alpha=alpha)
            exporter.write(faded)
            fade_pbar.update(1)
        fade_pbar.close()
        fade_time = time.time() - fade_start
        self.total_processing_time += fade_time
        self.total_processed_frames += fade_frames

    def visualize_tracks(self) -> None:
        self._validate_video()
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {self.video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = video_fps if video_fps > 0 else self.fps
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        exporter = self._create_exporter(fps, (width, height))

        fade_frames = int(fps * DEFAULT_FADE_DURATION)

        processed_count = 0
        total_tracks = len(self.tracks)

        overall_start_time = time.time()

        for track in self.tracks:
            track_id = track.track_id
            start_frame = track.start_frame
            end_frame = track.last_frame
            frame_count = end_frame - start_frame + 1

            LOG.info(
                "Processing track %s | Frames: %s-%s (%s)",
                track_id,
                start_frame,
                end_frame,
                frame_count,
            )

            if exporter and not exporter.open_track(track_id):
                continue

            pos_by_frame: Dict[int, Tuple[int, int]] = {}
            for pos in track.positions:
                x, y = pos[0]
                pos_by_frame[int(pos[1])] = (int(x), int(y))

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_num = start_frame
            last_clean_frame = None

            pbar = tqdm(
                total=frame_count,
                desc=f"Track {track_id}",
                unit="frame",
                leave=False,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} | {rate_fmt} [{elapsed}<{remaining}]",
            )
            track_start_time = time.time()

            while frame_num <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    LOG.warning("Failed to read frame %s, stopping track %s", frame_num, track_id)
                    break

                clean_frame = frame.copy()

                pos = pos_by_frame.get(frame_num)
                if pos:
                    px, py = pos
                    cv2.circle(frame, (px, py), 10, (0, 255, 255), -1)
                    elapsed_time = (frame_num - start_frame) / fps
                    text = f"ID:{track_id} ({px},{py}) t:{elapsed_time:.2f}s"
                    cv2.putText(
                        frame,
                        text,
                        (px + 15, py),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                if not self.output_path and not self.split_dir:
                    debug_text = f"Frame: {frame_num}/{total_video_frames}, Track: {track_id}"
                    cv2.putText(
                        frame,
                        debug_text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                    cv2.imshow("Track Visualization", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        pbar.close()
                        cap.release()
                        if exporter:
                            exporter.close()
                        cv2.destroyAllWindows()
                        return

                if exporter:
                    exporter.write(clean_frame)

                last_clean_frame = clean_frame.copy()
                frame_num += 1
                pbar.update(1)

            pbar.close()

            track_time = time.time() - track_start_time
            track_fps = frame_count / track_time if track_time > 0 else 0
            self.total_processed_frames += frame_count
            self.total_processing_time += track_time

            if exporter and last_clean_frame is not None and fade_frames > 0:
                self._write_fade_out(exporter, last_clean_frame, fade_frames)

            if exporter:
                exporter.close_track()
                if isinstance(exporter, SplitClipsExporter):
                    LOG.info("Saved track %s video", track_id)

            processed_count += 1
            LOG.info("Completed track %s (%s/%s)", track_id, processed_count, total_tracks)

        total_time = time.time() - overall_start_time
        avg_fps = self.total_processed_frames / total_time if total_time > 0 else 0

        cap.release()
        if exporter:
            exporter.close()
            if isinstance(exporter, CombinedVideoExporter):
                LOG.info("Combined video saved: %s", self.output_path)

        cv2.destroyAllWindows()

        LOG.info("Processing complete")
        LOG.info("Total tracks processed : %s", total_tracks)
        LOG.info("Total frames processed : %s", self.total_processed_frames)
        LOG.info("Total time             : %.2f sec", total_time)
        LOG.info("Average processing FPS : %.1f FPS", avg_fps)

        if not self.output_path and not self.split_dir:
            LOG.info("Visualization complete. Press any key to exit...")
            cv2.waitKey(0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize or export ball tracking data with FPS monitoring"
    )
    parser.add_argument(
        "--json_dir", type=str, default=None, help="Directory with track_*.json files"
    )
    parser.add_argument(
        "--video_path", type=str, required=True, help="Path to source video"
    )
    parser.add_argument(
        "--output_path", type=str, default=None, help="Combined output video (MP4)"
    )
    parser.add_argument(
        "--split_dir",
        type=str,
        default=None,
        help="Directory for individual track videos (MP4)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Root output directory"
    )
    parser.add_argument(
        "--fps", type=float, default=30.0, help="Output FPS if video has none"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.verbose)

    base_name = resolve_video_basename(args.video_path)

    if args.json_dir is None and args.output_dir:
        args.json_dir = os.path.join(args.output_dir, base_name, "tracks")

    if args.json_dir is None:
        parser.error("Provide --json_dir or --output_dir so tracks can be located")

    if args.output_path is None and args.output_dir and not args.split_dir:
        args.output_path = os.path.join(args.output_dir, base_name, "combined.mp4")

    mode = "Interactive visualization"
    if args.split_dir:
        mode = f"Exporting individual clips -> {args.split_dir}"
    elif args.output_path:
        mode = f"Exporting combined video -> {args.output_path}"

    LOG.info("Mode: %s", mode)

    processor = TrackProcessor(
        json_dir=args.json_dir,
        video_path=args.video_path,
        output_path=args.output_path,
        split_dir=args.split_dir,
        fps=args.fps,
    )
    processor._load_tracks_from_json()
    processor.visualize_tracks()


if __name__ == "__main__":
    main()
