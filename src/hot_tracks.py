"""Rallies of a video that is still being detected, window by window.

The track calculator works on a whole recording: calibrations over all the ball
detections, tracks merged across pauses. Run as the detector writes its JSONL,
it gets the new frames plus some context before them, and only the tracks that
ended long enough before the last detected frame are taken: later frames can no
longer change them (merges reach a few seconds at most). Those are committed:
written to ``tracks/`` under the next number and handed to the API. The rest wait
for the next window, which starts at the earliest of them.

The state (``hot/state.json``) survives a crash of the worker: a resumed run
goes on from the frontier without committing anything twice.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

# A track that ended this long before the last detected frame is final.
MARGIN_SECONDS = 10.0
# Frames before the frontier the calculator sees again: ball size calibration
# and the serve order need some history.
CONTEXT_SECONDS = 60.0
# A new window once this much video has been detected since the last one.
STEP_SECONDS = 60.0

STATE_NAME = "state.json"

# (frame_start, frame_end, tracks_dir) -> runs the track calculator over the window
WindowRunner = Callable[[int, int, Path], None]


@dataclass
class HotState:
    # Tracks starting before this frame are committed or will never be.
    frontier: int = 0
    # Number of the next committed track file (track_NNNN.json).
    next_number: int = 0
    # Last frame covered by a window.
    processed_to: int = -1


class HotTracks:
    def __init__(self, target_dir: Path, fps: float, run_window: WindowRunner) -> None:
        self.target_dir = Path(target_dir)
        self.tracks_dir = self.target_dir / "tracks"
        self.hot_dir = self.target_dir / "hot"
        self.fps = fps if fps > 0 else 30.0
        self._run_window = run_window
        self.state = self._load_state()

    @property
    def margin_frames(self) -> int:
        return int(round(MARGIN_SECONDS * self.fps))

    @property
    def context_frames(self) -> int:
        return int(round(CONTEXT_SECONDS * self.fps))

    @property
    def step_frames(self) -> int:
        return int(round(STEP_SECONDS * self.fps))

    def reset(self) -> None:
        """A fresh run: the track files and the state of an earlier one go."""
        for path in self.tracks_dir.glob("track_*.json"):
            path.unlink()
        if self.hot_dir.exists():
            shutil.rmtree(self.hot_dir)
        self.state = HotState()
        self._save_state()

    def due(self, last_frame: int) -> bool:
        """Enough new frames for a window."""
        return last_frame - max(self.state.processed_to, self.state.frontier) >= self.step_frames

    def advance(self, last_frame: int, final: bool = False) -> list[str]:
        """Runs a window up to ``last_frame``; returns the names of the committed track files.

        ``final``: the detection is over, every track found is committed.
        """
        if last_frame < self.state.frontier:
            return []
        frame_start = max(0, self.state.frontier - self.context_frames)
        window_dir = self.hot_dir / "window"
        if window_dir.exists():
            shutil.rmtree(window_dir)
        window_dir.mkdir(parents=True)
        self._run_window(frame_start, last_frame, window_dir)

        commit_before = last_frame + 1 if final else last_frame - self.margin_frames
        committed: list[str] = []
        # Tracks are committed in order of start, up to the first one still open: then
        # every committed track starts before the frontier and none comes twice.
        frontier = commit_before
        self.tracks_dir.mkdir(parents=True, exist_ok=True)
        for start, end, content in _window_tracks(window_dir):
            if start < self.state.frontier:
                continue
            if end >= commit_before:
                frontier = min(frontier, start)
                break
            name = f"track_{self.state.next_number:04d}.json"
            content["track_id"] = self.state.next_number
            (self.tracks_dir / name).write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")
            self.state.next_number += 1
            committed.append(name)

        self.state.frontier = max(self.state.frontier, frontier)
        self.state.processed_to = last_frame
        self._save_state()
        return committed

    def _load_state(self) -> HotState:
        try:
            data = json.loads((self.hot_dir / STATE_NAME).read_text(encoding="utf-8"))
            return HotState(**{key: int(data[key]) for key in ("frontier", "next_number", "processed_to")})
        except (OSError, ValueError, KeyError, TypeError):
            return HotState()

    def _save_state(self) -> None:
        self.hot_dir.mkdir(parents=True, exist_ok=True)
        path = self.hot_dir / STATE_NAME
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(asdict(self.state)), encoding="utf-8")
        temporary.replace(path)


def _window_tracks(window_dir: Path) -> list[tuple[int, int, dict]]:
    """``(start frame, last frame, content)`` of the tracks of a window, by start."""
    tracks: list[tuple[int, int, dict]] = []
    for path in window_dir.glob("track_*.json"):
        try:
            content = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        start, end = _frame(content, "start_frame"), _frame(content, "last_frame")
        if start is None or end is None:
            continue
        tracks.append((start, end, content))
    return sorted(tracks, key=lambda item: item[0])


def _frame(content: dict, key: str) -> Optional[int]:
    value = content.get(key)
    return int(value) if isinstance(value, (int, float)) else None
