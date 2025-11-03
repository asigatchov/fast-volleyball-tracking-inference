#!/usr/bin/env python3
"""
Track Visualization & Export Tool
--------------------------------
Loads ball tracking data from JSON files and visualizes or exports video clips
with overlaid track positions. Supports:
- Interactive preview
- Single combined output video
- Individual track videos
Features:
- Per-track progress bar with **live FPS**
- Fade-out progress with FPS
- Final **average processing speed**
"""

import os
import cv2
import json
import argparse
from typing import List, Optional
from ball_tracker import Track
from tqdm import tqdm
import time  # <-- NEW: For timing


class TrackProcessor:
    def __init__(
        self,
        json_dir: str,
        video_path: str,
        output_path: Optional[str] = None,
        split_dir: Optional[str] = None,
        fps: float = 30.0,
    ):
        self.json_dir = json_dir
        self.video_path = video_path
        self.output_path = output_path
        self.split_dir = split_dir
        self.fps = fps
        self.tracks: List[Track] = []
        self.total_processed_frames = 0
        self.total_processing_time = 0.0  # For average FPS

    def _load_tracks_from_json(self) -> None:
        """Load all track JSON files from the specified directory."""
        if not os.path.exists(self.json_dir):
            raise FileNotFoundError(f"JSON directory not found: {self.json_dir}")

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

        print(f"Loaded {len(self.tracks)} track(s) from {self.json_dir}")

    def _validate_video(self) -> None:
        """Check if the input video file exists."""
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

    def visualize_tracks(self) -> None:
        """Main function to visualize tracks on video or export clips with FPS tracking."""
        self._validate_video()
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {self.video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = video_fps if video_fps > 0 else self.fps
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Video writers
        combined_writer = None
        if self.output_path and not self.split_dir:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            combined_writer = cv2.VideoWriter(
                self.output_path, fourcc, fps, (width, height)
            )

        if self.split_dir:
            os.makedirs(self.split_dir, exist_ok=True)

        fade_duration = 0.5
        fade_frames = int(fps * fade_duration)

        processed_count = 0
        total_tracks = len(self.tracks)

        overall_start_time = time.time()

        for idx, track in enumerate(self.tracks):
            track_id = track.track_id
            start_frame = track.start_frame
            end_frame = track.last_frame
            frame_count = end_frame - start_frame + 1

            print(
                f"\nProcessing track ID: {track_id} | Frames: {start_frame}–{end_frame} ({frame_count})"
            )

            track_writer = None
            track_video_path = None
            if self.split_dir:
                track_video_path = os.path.join(
                    self.split_dir, f"track_{track_id:04d}.mp4"
                )
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                track_writer = cv2.VideoWriter(
                    track_video_path, fourcc, fps, (width, height)
                )
                if not track_writer.isOpened():
                    print(f"Failed to create video writer for: {track_video_path}")
                    continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_num = start_frame
            last_clean_frame = None

            # --- Progress bar with live FPS ---
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
                    print(
                        f"\nWarning: Failed to read frame {frame_num}, stopping track {track_id}"
                    )
                    break

                clean_frame = frame.copy()

                # Draw annotations
                for pos in track.positions:
                    px, py, pf = int(pos[0][0]), int(pos[0][1]), pos[1]
                    if pf == frame_num:
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

                # Interactive mode
                if not self.output_path and not self.split_dir:
                    debug_text = (
                        f"Frame: {frame_num}/{total_video_frames}, Track: {track_id}"
                    )
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
                        if combined_writer:
                            combined_writer.release()
                        if track_writer:
                            track_writer.release()
                        cv2.destroyAllWindows()
                        return

                # Write clean frame
                if combined_writer:
                    combined_writer.write(clean_frame)
                if track_writer:
                    track_writer.write(clean_frame)

                last_clean_frame = clean_frame.copy()
                frame_num += 1
                pbar.update(1)

            pbar.close()

            track_time = time.time() - track_start_time
            track_fps = frame_count / track_time if track_time > 0 else 0
            self.total_processed_frames += frame_count
            self.total_processing_time += track_time

            # --- Fade-out with FPS ---
            if (
                (combined_writer or track_writer)
                and last_clean_frame is not None
                and fade_frames > 0
            ):
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
                    faded = cv2.convertScaleAbs(last_clean_frame, alpha=alpha)
                    if combined_writer:
                        combined_writer.write(faded)
                    if track_writer:
                        track_writer.write(faded)
                    fade_pbar.update(1)
                fade_pbar.close()
                fade_time = time.time() - fade_start
                self.total_processing_time += fade_time
                self.total_processed_frames += fade_frames

            if track_writer:
                track_writer.release()
                print(
                    f"Saved track {track_id} video: {track_video_path} ({track_fps:.1f} FPS)"
                )

            processed_count += 1
            print(
                f"Completed track {track_id} — {processed_count}/{total_tracks} processed"
            )

        # --- Final Summary ---
        total_time = time.time() - overall_start_time
        avg_fps = self.total_processed_frames / total_time if total_time > 0 else 0

        cap.release()
        if combined_writer:
            combined_writer.release()
            print(f"Combined video saved: {self.output_path}")

        cv2.destroyAllWindows()

        print("\n" + "=" * 60)
        print(f"PROCESSING COMPLETE")
        print(f"   Total tracks processed : {total_tracks}")
        print(f"   Total frames processed : {self.total_processed_frames}")
        print(f"   Total time             : {total_time:.2f} sec")
        print(f"   Average processing FPS : {avg_fps:.1f} FPS")
        print("=" * 60)

        if not self.output_path and not self.split_dir:
            print("Visualization complete. Press any key to exit...")
            cv2.waitKey(0)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize or export ball tracking data with FPS monitoring"
    )
    parser.add_argument(
        "--json_dir", type=str, required=True, help="Directory with track_*.json files"
    )
    parser.add_argument(
        "--video_path", type=str, required=True, help="Path to source video"
    )
    parser.add_argument(
        "--output_path", type=str, default=None, help="Combined output video (AVI)"
    )
    parser.add_argument(
        "--split_dir",
        type=str,
        default=None,
        help="Directory for individual track videos (MP4)",
    )
    parser.add_argument(
        "--fps", type=float, default=30.0, help="Output FPS if video has none"
    )
    args = parser.parse_args()

    mode = "Interactive visualization"
    if args.split_dir:
        mode = f"Exporting individual clips → {args.split_dir}"
    elif args.output_path:
        mode = f"Exporting combined video → {args.output_path}"

    print(f"Mode: {mode}")

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
