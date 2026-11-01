"""Decode only the frames that are wanted: seek to a keyframe instead of decoding the whole video.

``classify_contacts.py`` and ``player_reid.py`` need a few frames around every contact, yet
read the video from the start (``cap.grab()`` for the frames in between). On the uploads the
portal gets (constant-frame-rate MP4) ``cv2.CAP_PROP_POS_FRAMES`` lands on the exact frame:
checked on 2688x1512 and 1920x1080 recordings, 30 random windows, offset 0 every time.

A seek costs the decoding from the previous keyframe (GoPro: up to 250 frames), so it is only
worth it when a keyframe lies between the current position and the wanted frame. The keyframe
positions come from ``ffprobe`` over the packets (no decoding, a few seconds for a match);
without ffprobe a seek is made for gaps above ``fallback_gap`` frames.

Turned on by ``INFERENCE_VIDEO_SEEK=true`` (``seek_enabled()``); off, the readers behave as before.
"""

from __future__ import annotations

import bisect
import logging
import os
import subprocess
from typing import Iterable, Iterator, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

ENV_FLAG = "INFERENCE_VIDEO_SEEK"


def seek_enabled() -> bool:
    return os.getenv(ENV_FLAG, "false").lower() in {"1", "true", "yes", "on"}


def keyframe_indices(video_path: str, timeout: float = 600) -> Optional[list[int]]:
    """Frame indices (presentation order) of the keyframes, None when ffprobe is not usable."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "packet=pts,dts,flags", "-of", "csv=p=0", video_path,
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("ffprobe keyframe listing failed (%s): seeking by a fixed gap", exc)
        return None
    packets = []
    for line in out.splitlines():
        parts = line.split(",")
        if len(parts) < 3:
            continue
        pts = parts[0] if parts[0] not in ("", "N/A") else parts[1]
        try:
            packets.append((int(pts), "K" in parts[2]))
        except ValueError:
            continue
    if not packets:
        return None
    # Presentation order = sorted by pts (B-frames are stored out of order).
    packets.sort(key=lambda item: item[0])
    return [index for index, (_, key) in enumerate(packets) if key]


class SeekingReader:
    """``frames(wanted)`` yields ``(index, bgr)`` of the wanted frames in increasing order.

    Between two wanted frames the reader either decodes forward (``grab``) or, when a keyframe
    lies in between, seeks to the wanted frame (OpenCV decodes from that keyframe).
    """

    def __init__(self, video_path: str, seek: bool = True, fallback_gap: int = 90) -> None:
        self.video_path = video_path
        self.seek = seek
        self.fallback_gap = fallback_gap
        self.keyframes = keyframe_indices(video_path) if seek else None
        self.decoded = 0
        self.seeks = 0

    def _should_seek(self, position: int, target: int) -> bool:
        if not self.seek or target <= position:
            return False
        if self.keyframes is None:
            return target - position > self.fallback_gap
        # the last keyframe at or before the target
        k = bisect.bisect_right(self.keyframes, target) - 1
        return k >= 0 and self.keyframes[k] > position

    def frames(self, wanted: Iterable[int]) -> Iterator[tuple[int, np.ndarray]]:
        targets = sorted(set(int(f) for f in wanted))
        if not targets:
            return
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.video_path}")
        position = 0  # index of the next frame the capture returns
        try:
            for target in targets:
                if self._should_seek(position, target):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                    position = target
                    self.seeks += 1
                while position < target:
                    if not cap.grab():
                        return
                    position += 1
                    self.decoded += 1
                ok, image = cap.read()
                if not ok:
                    return
                position += 1
                self.decoded += 1
                yield target, image
        finally:
            cap.release()
            logger.info(
                "SeekingReader: %d wanted frames, %d decoded, %d seeks (%s)",
                len(targets), self.decoded, self.seeks,
                "keyframes known" if self.keyframes is not None else "fixed gap",
            )
