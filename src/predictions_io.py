"""Detector predictions on disk: the streamed JSONL and the older single JSON.

``predictions.jsonl`` (``ravel-vb-predictions-v3``) is written while the
detector runs, one JSON object per line:

- ``{"type": "header", ...}`` first: model, video, fps, sizes, sampling, and
  ``fields``, the order of the values in the rows of a frame;
- ``{"type":"frame","frame_index":N,"players":[...],"ball":[...]}`` for every
  frame the network saw, in order, empty ones included: the line says the frame
  is done. A player is ``[track_id, score, x1, y1, x2, y2]``, then ``1`` when the
  box was carried over a gap (``interpolated``) and, rarely, a dict of the other
  fields the model gave; the ball is ``[score, x1, y1, x2, y2]``. Boxes are in
  pixels to 0.01 (as the detector always wrote them), scores cut down to 0.001.
  Other classes go to ``"other"`` as dicts;
- ``{"type": "checkpoint", "next_frame": N, ...}`` now and then: the detector
  resumes from the last one after a crash, dropping the lines after it;
- ``{"type": "end", "last_frame": N, "processed_frames": N + 1, "benchmark": {...}}``
  last: the explicit flag that the video is over and nothing more will come.

A reader may meet the file while it grows: a torn last line is not an error,
the frames before it are complete. Once the processing is over the file is kept
gzipped (``predictions.jsonl.gz``). Readers give the predictions back as dicts
in the shape of the older ``predictions.json`` (``ravel-vb-predictions-v1``,
written once at the end), which they read too, as the lines of v2 (a list of
those dicts under ``"predictions"``).
"""

from __future__ import annotations

import gzip
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any, Iterator, Optional

FORMAT_V1 = "ravel-vb-predictions-v1"
FORMAT_V3 = "ravel-vb-predictions-v3"
JSONL_NAME = "predictions.jsonl"
GZIP_NAME = "predictions.jsonl.gz"
JSON_NAME = "predictions.json"
FIELDS = {
    "players": ["track_id", "score", "x1", "y1", "x2", "y2", "interpolated?", "extra?"],
    "ball": ["score", "x1", "y1", "x2", "y2"],
}

# Frame lines start with {"type":"frame","frame_index":N: the index is read from
# the start of the line, so a reader skips frames out of its range unparsed.
_FRAME_PREFIX = re.compile(r'^\{"type": ?"frame", ?"frame_index": ?(\d+)')
# Fields a row stands for; any other field of a prediction rides along in the extra dict.
_PLAYER_FIELDS = {"class_id", "class_name", "score", "bbox_xyxy", "track_id", "interpolated", "frame_index", "time_sec", "query_index"}
_BALL_FIELDS = {"class_id", "class_name", "score", "bbox_xyxy", "frame_index", "time_sec", "query_index"}


def dump_line(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"


def frame_record(frame_index: int, predictions: list[dict]) -> dict[str, Any]:
    """The line of a frame out of the detector's prediction dicts."""
    # Key order matters: _FRAME_PREFIX reads the index without parsing the line.
    record: dict[str, Any] = {"type": "frame", "frame_index": frame_index, "players": []}
    other: list[dict] = []
    for item in predictions:
        name = item.get("class_name")
        if name == "player" and isinstance(item.get("track_id"), int):
            row: list[Any] = [item["track_id"], _score(item), *_box(item)]
            extra = {key: value for key, value in item.items() if key not in _PLAYER_FIELDS}
            if item.get("interpolated") or extra:
                row.append(1 if item.get("interpolated") else 0)
            if extra:
                row.append(extra)
            record["players"].append(row)
        elif name == "ball" and "ball" not in record and set(item) <= _BALL_FIELDS:
            record["ball"] = [_score(item), *_box(item)]
        else:
            other.append({key: value for key, value in item.items() if key not in ("frame_index", "time_sec")})
    if other:
        record["other"] = other
    return record


def expand_frame(record: dict[str, Any]) -> list[dict]:
    """Prediction dicts of a frame line (v3 rows or the v2 list)."""
    frame = int(record["frame_index"])
    if "predictions" in record:
        return record["predictions"] or []
    items: list[dict] = []
    for row in record.get("players") or ():
        item = {
            "class_name": "player",
            "score": row[1],
            "bbox_xyxy": row[2:6],
            "track_id": row[0],
            "interpolated": bool(len(row) > 6 and row[6]),
            "frame_index": frame,
        }
        if len(row) > 7 and isinstance(row[7], dict):
            item.update(row[7])
        items.append(item)
    ball = record.get("ball")
    if ball:
        items.append({"class_name": "ball", "score": ball[0], "bbox_xyxy": ball[1:5], "frame_index": frame})
    for item in record.get("other") or ():
        items.append({**item, "frame_index": frame})
    return items


def is_jsonl(path: str | os.PathLike[str]) -> bool:
    return str(path).endswith((".jsonl", ".jsonl.gz"))


def find_predictions(directory: str | os.PathLike[str]) -> Optional[Path]:
    """The predictions file of a project directory: the JSONL (packed or not), else the older JSON."""
    for name in (GZIP_NAME, JSONL_NAME, JSON_NAME):
        path = Path(directory) / name
        if path.is_file():
            return path
    return None


def pack(path: str | os.PathLike[str]) -> Path:
    """Gzips a finished JSONL next to itself and removes the plain one; returns the packed path."""
    source = Path(path)
    target = source.with_name(source.name + ".gz")
    temporary = target.with_name(target.name + ".tmp")
    with source.open("rb") as plain, gzip.open(temporary, "wb", compresslevel=6) as packed:
        shutil.copyfileobj(plain, packed, 1 << 20)
    temporary.replace(target)
    source.unlink()
    return target


def iter_records(path: str | os.PathLike[str]) -> Iterator[tuple[int, dict[str, Any]]]:
    """``(byte offset after the line, record)`` of every complete line of a JSONL.

    Stops at a torn line: past it the file is being written or was cut by a crash.
    """
    offset = 0
    with _open(path) as handle:
        for raw in handle:
            if not raw.endswith(b"\n"):
                return
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                return
            offset += len(raw)
            if isinstance(record, dict):
                yield offset, record


def iter_frames(
    path: str | os.PathLike[str],
    frame_start: Optional[int] = None,
    frame_end: Optional[int] = None,
) -> Iterator[tuple[int, list[dict]]]:
    """``(frame, predictions)`` of the frames in ``[frame_start, frame_end]``, in order.

    For the JSONL a frame out of the range is skipped without parsing its line.
    """
    if not is_jsonl(path):
        _, predictions = _read_json(path)
        by_frame: dict[int, list[dict]] = {}
        for item in predictions:
            frame = int(item.get("frame_index", -1))
            if _in_range(frame, frame_start, frame_end):
                by_frame.setdefault(frame, []).append(item)
        yield from sorted(by_frame.items())
        return

    with _open(path) as handle:
        for raw in handle:
            if not raw.endswith(b"\n"):
                return
            frame = _frame_of(raw)
            if frame is None:
                continue
            if frame_end is not None and frame > frame_end:
                return
            if not _in_range(frame, frame_start, frame_end):
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                return
            yield frame, expand_frame(record)


def read_meta(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Header fields of the predictions, with the progress of the run.

    For the JSONL: ``last_frame`` (the last frame written), ``complete`` (the end
    line is there) and, once complete, ``benchmark`` / ``processed_frames``. The
    older JSON is complete by construction.
    """
    if not is_jsonl(path):
        meta, predictions = _read_json(path)
        meta = {key: value for key, value in meta.items() if key != "predictions"}
        frames = [int(item.get("frame_index", -1)) for item in predictions]
        meta["last_frame"] = max(frames, default=-1)
        meta["complete"] = True
        meta.setdefault("processed_frames", (meta.get("benchmark") or {}).get("processed_frames"))
        return meta

    meta: dict[str, Any] = {"last_frame": -1, "complete": False}
    with _open(path) as handle:
        first = handle.readline()
        try:
            header = json.loads(first) if first.endswith(b"\n") else {}
        except json.JSONDecodeError:
            header = {}
        if isinstance(header, dict) and header.get("type") == "header":
            meta.update({key: value for key, value in header.items() if key != "type"})
        if str(path).endswith(".gz"):
            meta["last_frame"], end = _scan_state(handle)
        else:
            meta["last_frame"], end = _tail_state(handle)
    if end is not None:
        meta["complete"] = True
        meta["last_frame"] = int(end.get("last_frame", meta["last_frame"]))
        meta["processed_frames"] = end.get("processed_frames")
        meta["benchmark"] = end.get("benchmark", {})
    return meta


def read_predictions(
    path: str | os.PathLike[str],
    frame_start: Optional[int] = None,
    frame_end: Optional[int] = None,
) -> tuple[dict[str, Any], list[dict]]:
    """``(meta, predictions)`` in the shape of the older JSON, from either format."""
    meta = read_meta(path)
    predictions = [item for _, items in iter_frames(path, frame_start, frame_end) for item in items]
    return meta, predictions


def _open(path: str | os.PathLike[str]):
    return gzip.open(path, "rb") if str(path).endswith(".gz") else open(path, "rb")


def _score(item: dict) -> float:
    # Cut down, not rounded: 0.3496 stays below a 0.35 threshold. Rounded scores and
    # boxes to 0.1 px changed a rally and a dozen touches of a match; these change none.
    return math.floor(float(item["score"]) * 1000) / 1000


def _box(item: dict) -> list[float]:
    return [round(float(value), 2) for value in item["bbox_xyxy"]]


def _frame_of(raw: bytes) -> Optional[int]:
    match = _FRAME_PREFIX.match(raw[:64].decode("utf-8", errors="replace"))
    return int(match.group(1)) if match else None


def _read_json(path: str | os.PathLike[str]) -> tuple[dict[str, Any], list[dict]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return {}, payload
    if not isinstance(payload, dict):
        return {}, []
    for key in ("predictions", "detections", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            return payload, value
    return payload, []


def _in_range(frame: int, frame_start: Optional[int], frame_end: Optional[int]) -> bool:
    return (frame_start is None or frame >= frame_start) and (frame_end is None or frame <= frame_end)


def _scan_state(handle: Any) -> tuple[int, Optional[dict[str, Any]]]:
    """Last frame and the end record of a packed JSONL, read through (it cannot be seeked cheaply)."""
    last_frame, end = -1, None
    for raw in handle:
        if not raw.endswith(b"\n"):
            break
        frame = _frame_of(raw)
        if frame is not None:
            last_frame = frame
        elif raw.startswith(b'{"type":"end"'):
            end = json.loads(raw)
    return last_frame, end


def _tail_state(handle: Any, chunk: int = 1 << 20) -> tuple[int, Optional[dict[str, Any]]]:
    """Last written frame and the end record, read from the end of the file."""
    handle.seek(0, os.SEEK_END)
    size = handle.tell()
    start = max(0, size - chunk)
    end = None
    while True:
        handle.seek(start)
        lines = handle.read(size - start).split(b"\n")
        # The first piece may be a line cut by the seek, the last one a torn line.
        complete = lines[1:-1] if start > 0 else lines[:-1]
        for raw in reversed(complete):
            frame = _frame_of(raw)
            if frame is not None:
                return frame, end
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and record.get("type") == "end":
                end = end or record
        if start == 0:
            return -1, end
        start = max(0, start - chunk)
