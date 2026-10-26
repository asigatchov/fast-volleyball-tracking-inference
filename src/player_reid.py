"""Sort the actions of one half of the court into players by appearance (person re-id).

The actor of an action is the detector's player with the action's ``track_id`` at the contact
frame, else the player closest to the ball point. Every player standing on the same half at that
frame is embedded too (``ov/person-reidentification-retail-0287.xml``, 256 numbers per person), and all of them are
clustered into ``k`` identities under one rule: two people seen at the same moment are two
different players (each moment is a Hungarian assignment to the cluster centres). Track ids
mean nothing across rallies, so they are not used for the clustering.

Out: ``<out_dir>/players.json`` and a crop of the actor per action, ``<out_dir>/<frame>.jpg``.
Players are numbered 1..k, the one with the most actions first.

    uv run src/player_reid.py VIDEO PREDICTIONS COURT_JSON ACTIONS_JSON OUT_DIR [--k 12] [--side near]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    from .frame_seek import SeekingReader, seek_enabled
    from .jersey_numbers import decide_number
    from .predictions_io import iter_frames
except ImportError:
    from frame_seek import SeekingReader, seek_enabled
    from jersey_numbers import decide_number
    from predictions_io import iter_frames

REID_HEIGHT, REID_WIDTH = 256, 128
# OSNet (torchreid) wants RGB 0..1 normalised by ImageNet; the Open Model Zoo models take BGR 0..255 as is.
REID_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
REID_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CROP_HEIGHT = 160
# A person's vector is the mean over his track around the contact (+-0.75 s, every 3rd reported
# frame: 7 views), not the contact frame alone (blur, arms up, back to the camera). On two
# hand-sorted matches (206 and 188 actions), the nearest player right / seconds per match:
#   osnet_x0_5_msmt17, contact frame     86% 95%   84 s
#   osnet_x0_5_msmt17, every 3rd frame   94% 98%  135 s
#   osnet_x0_5_msmt17, every frame       94% 99%  258 s
#   retail-0287, contact frame           92% 97%   82 s
#   retail-0287, every 3rd frame         97% 97%  114 s
#   retail-0287, every frame             97% 99%  219 s
# Views 2 and 4 frames apart add little (89% 96%): they are the same pose. Fitting the box into
# 256x128 with margins instead of stretching it lost 2-4 points: not done.
APPEARANCE_WINDOW = 22
APPEARANCE_STEP = 6
# A frame where another player overlaps the box would mix the two people.
APPEARANCE_MAX_OVERLAP = 0.1
ENCODE_BATCH = 512
VECTORS_FILE = "vectors.npz"
DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "ov" / "person-reidentification-retail-0287.xml"
Box = tuple[float, float, float, float]


def centre_line(court: dict[str, Any]) -> tuple[tuple[float, float], tuple[float, float]]:
    """End points of the court's centre line (keypoints ``5_center_left`` / ``6_center_right``)."""
    names = court["categories"][0]["keypoints"]
    points = court["annotations"][0]["keypoints"]
    named = {name: (points[3 * i], points[3 * i + 1]) for i, name in enumerate(names)}
    return named["5_center_left"], named["6_center_right"]


def half_of(box: Box, line: tuple[tuple[float, float], tuple[float, float]]) -> str:
    """``near`` when the feet (box bottom) are below the centre line on the image, else ``far``."""
    (x0, y0), (x1, y1) = line
    x = (box[0] + box[2]) / 2
    y = y0 + (y1 - y0) * (x - x0) / (x1 - x0) if x1 != x0 else (y0 + y1) / 2
    return "near" if box[3] > y else "far"


def pick_actor(players: list[tuple[Optional[int], Box]], track_id: Optional[int], ball: Optional[list[float]]) -> Optional[int]:
    """Index of the acting player in ``players``: by track id, else the box nearest to the ball."""
    if track_id is not None:
        for index, (track, _) in enumerate(players):
            if track == track_id:
                return index
    if ball is None or not players:
        return None
    bx, by = ball

    def distance(box: Box) -> float:
        dx = max(box[0] - bx, 0.0, bx - box[2])
        dy = max(box[1] - by, 0.0, by - box[3])
        return float(np.hypot(dx, dy))

    return min(range(len(players)), key=lambda i: distance(players[i][1]))


def cluster(vectors: np.ndarray, groups: list[list[int]], k: int, seeds: int = 30, iterations: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Labels ``0..k-1`` of unit ``vectors`` and the margin of each (best minus second cosine).

    The members of a group are told apart: each group is matched to distinct centres. The best
    of ``seeds`` random starts by the summed similarity of those matches wins.
    """
    count = len(vectors)
    k = min(k, count)
    labels = np.zeros(count, dtype=int)
    if count == 0 or k == 0:
        return labels, np.zeros(count)
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        centres = vectors[rng.choice(count, k, replace=False)]
        for _ in range(iterations):
            for group in groups:
                rows, cols = linear_sum_assignment(-(vectors[group] @ centres.T))
                labels[np.asarray(group)[rows]] = cols
            updated = np.stack([vectors[labels == c].mean(0) if (labels == c).any() else centres[c] for c in range(k)])
            updated /= np.linalg.norm(updated, axis=1, keepdims=True)
            if np.allclose(updated, centres):
                break
            centres = updated
        score = 0.0
        for group in groups:
            similarity = vectors[group] @ centres.T
            rows, cols = linear_sum_assignment(-similarity)
            score += float(similarity[rows, cols].sum())
        if best is None or score > best[0]:
            best = (score, labels.copy(), centres.copy())
    _, labels, centres = best
    similarity = np.sort(vectors @ centres.T, axis=1)
    margin = similarity[:, -1] - (similarity[:, -2] if k > 1 else 0.0)
    return labels, margin


class ReidEncoder:
    """OpenVINO person re-id: BGR crops -> unit vectors (256 for retail-0287, 512 for OSNet).

    ``name`` (the model file's stem) goes into players.json and vectors.npz: the portal's hint
    margin depends on the model, its cosines spread differently.
    """

    def __init__(self, model_xml: Path = DEFAULT_MODEL, device: str = "CPU") -> None:
        import openvino as ov

        core = ov.Core()
        model = core.read_model(str(model_xml))
        # The Open Model Zoo models are exported for one image at a time.
        model.reshape({model.input(0): [-1, 3, REID_HEIGHT, REID_WIDTH]})
        self._model = core.compile_model(model, device)
        self.name = Path(model_xml).stem
        self._torchreid = self.name.startswith("osnet")

    def __call__(self, crops: list[np.ndarray], batch: int = 32) -> np.ndarray:
        out = []
        for start in range(0, len(crops), batch):
            x = np.stack(
                [cv2.resize(c, (REID_WIDTH, REID_HEIGHT), interpolation=cv2.INTER_LINEAR) for c in crops[start : start + batch]]
            ).astype(np.float32)
            if self._torchreid:
                x = (x[:, :, :, ::-1] / 255.0 - REID_MEAN) / REID_STD
            out.append(self._model(np.ascontiguousarray(x.transpose(0, 3, 1, 2), dtype=np.float32))[0])
        vectors = np.concatenate(out) if out else np.zeros((0, 256), dtype=np.float32)
        return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-6)


def save_vectors(path: Path, by_frame: dict[int, np.ndarray], model: str = "") -> None:
    """``frames`` [N], unit ``vectors`` [N, D] (float16) of the actors and the ``model`` that made them, as ``vectors.npz``."""
    frames = sorted(by_frame)
    vectors = np.stack([by_frame[f] for f in frames]) if frames else np.zeros((0, 256), np.float32)
    np.savez_compressed(path, frames=np.array(frames, np.int64), vectors=vectors.astype(np.float16), model=np.array(model))


def _crop(image: np.ndarray, box: Box, pad: float = 0.0) -> np.ndarray:
    height, width = image.shape[:2]
    x0, y0, x1, y1 = box
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    x0, y0 = int(max(x0 - px, 0)), int(max(y0 - py, 0))
    x1, y1 = int(min(x1 + px, width)), int(min(y1 + py, height))
    # A copy: a view would keep the whole frame alive (2688x1512 is 12 MB, thousands of crops are kept).
    return image[y0 : max(y1, y0 + 1), x0 : max(x1, x0 + 1)].copy()


def _overlap(a: Box, b: Box) -> float:
    """Intersection over union of two boxes."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _track_views(
    boxes: dict[int, list[tuple[Optional[int], Box]]], frame: int, track: Optional[int], window: int, step: int
) -> list[tuple[int, Box]]:
    """Frames and boxes of ``track`` within +-``window`` of ``frame`` where no other player overlaps it."""
    if track is None or window <= 0:
        return []
    views = []
    # Symmetric around the contact: +-step, +-2 step, ... up to +-window.
    for f in sorted(frame + sign * offset for offset in range(step, window + 1, step) for sign in (-1, 1)):
        here = boxes.get(f, [])
        box = next((b for t, b in here if t == track), None)
        if box is not None and all(_overlap(box, b) <= APPEARANCE_MAX_OVERLAP for t, b in here if t != track):
            views.append((f, box))
    return views


def _read_frames(video_path: str, frames: Iterable[int]) -> Iterable[tuple[int, np.ndarray]]:
    """BGR images of ``frames`` in order: one sequential pass, or keyframe seeks with ``INFERENCE_VIDEO_SEEK``."""
    wanted = set(frames)
    if seek_enabled():
        yield from SeekingReader(video_path).frames(wanted)
        return
    final = max(wanted, default=-1)
    cap = cv2.VideoCapture(video_path)
    index = 0
    try:
        while index <= final and cap.grab():
            if index in wanted:
                ok, image = cap.retrieve()
                if ok:
                    yield index, image
            index += 1
    finally:
        cap.release()


def _numbers_by_appearance(
    vectors: np.ndarray, actor_rows: dict[int, int], read: dict[int, str], min_margin: float = 0.03
) -> dict[int, tuple[str, float]]:
    """Actions without a read number get the number whose actions they look most like.

    Each number is the mean re-id vector of the actions it was read on; an action takes the
    closest one when it beats the second by ``min_margin`` (cosine), else stays without a player.
    """
    numbers = sorted(set(read.values()))
    if not numbers:
        return {}
    centres = np.stack([vectors[[actor_rows[f] for f, n in read.items() if n == number]].mean(0) for number in numbers])
    centres /= np.maximum(np.linalg.norm(centres, axis=1, keepdims=True), 1e-6)
    out = {}
    for frame, row in actor_rows.items():
        if frame in read:
            continue
        similarity = centres @ vectors[row]
        order = np.argsort(-similarity)
        margin = float(similarity[order[0]] - (similarity[order[1]] if len(order) > 1 else 0.0))
        if margin >= min_margin:
            out[frame] = (numbers[int(order[0])], margin)
    return out


def _numbers_by_track(actors: dict[int, dict[str, Any]], read: dict[int, str]) -> dict[int, str]:
    """An unread action takes the number read on another action of the same track in the same rally.

    Only when that track got one number in the rally (a track that switched people may carry
    two). First set of 30a64638: +2 right, 0 wrong. The same by re-id likeness inside the rally
    was right 1 time per 2 wrong in one kit colour, so it is not done.
    """
    numbers: dict[tuple[Any, Any], set[str]] = defaultdict(set)
    for frame, number in read.items():
        actor = actors[frame]
        if actor["rally"] is not None and actor["track"] is not None:
            numbers[(actor["rally"], actor["track"])].add(number)
    out = {}
    for frame, actor in actors.items():
        found = numbers.get((actor["rally"], actor["track"]), set())
        if frame not in read and actor["rally"] is not None and actor["track"] is not None and len(found) == 1:
            out[frame] = next(iter(found))
    return out


# Clusters without a number read are numbered from here: no jersey number is that big.
UNNAMED_CLUSTER_BASE = 100


def _numbers_by_cluster(
    cluster_of: dict[int, int], known: dict[int, str], min_votes: int = 2, min_share: float = 0.7
) -> dict[int, str]:
    """The number of each appearance cluster: the one most of its read actions carry.

    Only with ``min_votes`` reads of that number and ``min_share`` of the cluster's reads: a
    cluster that mixes two people gets no number, its actions stay a cluster of their own.
    """
    votes: dict[int, Counter] = defaultdict(Counter)
    for frame, number in known.items():
        if frame in cluster_of:
            votes[cluster_of[frame]][number] += 1
    named = {}
    for cluster_id, counter in votes.items():
        number, count = counter.most_common(1)[0]
        if count >= min_votes and count / sum(counter.values()) >= min_share:
            named[cluster_id] = number
    return named


def assign_players(
    video_path: str,
    predictions_path: str,
    court: dict[str, Any],
    actions: list[dict[str, Any]],
    out_dir: Path,
    encoder,
    k: int = 12,
    side: str = "near",
    numbers=None,
    candidates: Optional[list[str]] = None,
    window: int = 60,
    step: int = 4,
    appearance_window: int = APPEARANCE_WINDOW,
    appearance_step: int = APPEARANCE_STEP,
    fill_by_appearance: bool = False,
    min_numbered: int = 5,
    min_votes: int = 2,
    min_share: float = 0.7,
    name_clusters: bool = True,
    debug_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Players of ``actions`` (``{"frame", "track_id"?, "ball"?: [x, y], "side"?, "rally"?}``) on ``side``; writes players.json.

    Two passes. First every action is clustered by appearance (re-id) into ``k`` players. Then,
    with ``numbers``, the jersey numbers are read and refine it (``mode: "mixed"``): an action
    keeps the number read on it (``source: "number"``) or on its track in the rally (``"track"``);
    the rest take the number of their cluster (``"cluster_number"``) when most of the cluster's
    reads agree on one (``_numbers_by_cluster``); a cluster without one stays a player numbered
    from ``UNNAMED_CLUSTER_BASE`` + 1 (``"cluster"``). Any read number is enough for it; without
    one it is the clusters alone, numbered 1..k (``mode: "cluster"``).

    With ``numbers`` (a ``jersey_numbers.NumberReader``) the jersey number of each actor is read
    on its track (+-``window`` frames, every ``step``-th) and the players are the numbers read
    (``mode: "number"``; ``candidates`` limits them to the team's numbers and adds the ones not
    read); an action whose number was not read takes the one read on its track elsewhere in the
    same rally (``source: "track"``). The rest stays without a player, unless
    ``fill_by_appearance`` gives it the number it looks like (re-id): in one kit colour that was
    right 4 times of 12 on the first set of 30a64638, against 36 of 38 for the read numbers. When the
    numbers can not be read (``mode: "cluster"``: fewer than ``min_numbered`` and 15% of the
    actions, e.g. a kit without numbers on the back, and no ``candidates``), the actions are clustered by appearance into ``k`` players numbered 1..k.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    line = centre_line(court)
    frames = sorted({int(a["frame"]) for a in actions})
    reach = max(window if numbers is not None else 0, appearance_window)
    boxes: dict[int, list[tuple[Optional[int], Box]]] = {}
    if frames:
        for frame, items in iter_frames(predictions_path, max(frames[0] - reach, 0), frames[-1] + reach):
            boxes[frame] = [
                (item.get("track_id"), tuple(float(v) for v in item["bbox_xyxy"]))
                for item in items
                if item.get("class_name") == "player"
            ]

    # actor of each action (index into that frame's players), and the people of the half per frame
    actors: dict[int, dict[str, Any]] = {}
    for action in actions:
        frame = int(action["frame"])
        players = boxes.get(frame, [])
        index = pick_actor(players, action.get("track_id"), action.get("ball"))
        # The pipeline's own side of the contact ("near"/"far", it follows the rally) beats the feet of one
        # frame: a player jumping at the net has them above the centre line.
        actor_side = action.get("side") or (half_of(players[index][1], line) if index is not None else None)
        if index is not None and actor_side == side:
            actors[frame] = {"index": index, "track": players[index][0], "rally": action.get("rally")}
    # frames of the actors' tracks where the number is read
    track_frames: dict[int, list[int]] = defaultdict(list)  # frame -> contact frames whose track is read there
    if numbers is not None:
        for frame, actor in actors.items():
            if actor["track"] is None:
                continue
            for f in range(frame - window, frame + window + 1, step):
                track_frames[f].append(frame)
    # The people of the half at each contact: the actor and those he is told apart from. Each is
    # seen on his track around the contact too, and his vector is the mean of those views.
    people: list[list[tuple[int, Box]]] = []  # views (frame, box) per person, the contact first
    groups: list[list[int]] = []
    for frame, actor in actors.items():
        group = []
        for index, (track, box) in enumerate(boxes.get(frame, [])):
            if index != actor["index"] and half_of(box, line) != side:
                continue
            if box[2] - box[0] < 1 or box[3] - box[1] < 1:
                continue
            group.append(len(people))
            if index == actor["index"]:
                actor["row"] = len(people)
            people.append([(frame, box), *_track_views(boxes, frame, track, appearance_window, appearance_step)])
        groups.append(group)
    views_at: dict[int, list[tuple[int, Box]]] = defaultdict(list)  # frame -> (person, box)
    for person, views in enumerate(people):
        for f, box in views:
            views_at[f].append((person, box))

    torsos: list[np.ndarray] = []
    torso_owner: list[int] = []
    sums = np.zeros((len(people), 0), dtype=np.float32)
    pending: list[tuple[int, np.ndarray]] = []

    def encode_pending() -> None:
        nonlocal sums
        if not pending:
            return
        encoded = encoder([crop for _, crop in pending])
        if sums.shape[1] == 0:
            sums = np.zeros((len(people), encoded.shape[1]), dtype=np.float32)
        np.add.at(sums, [person for person, _ in pending], encoded)
        pending.clear()

    for frame, image in _read_frames(video_path, set(views_at) | set(track_frames)):
        for contact in track_frames.get(frame, []):
            box = next((b for t, b in boxes.get(frame, []) if t == actors[contact]["track"]), None)
            torso = numbers.torso(image, box) if box is not None else None
            if torso is not None:
                torsos.append(torso)
                torso_owner.append(contact)
        for person, box in views_at.get(frame, []):
            crop = _crop(image, box)
            if crop.size:
                pending.append((person, crop))
        actor = actors.get(frame)
        if actor is not None and "row" in actor:
            thumb = _crop(image, people[actor["row"]][0][1], pad=0.08)
            scale = CROP_HEIGHT / max(thumb.shape[0], 1)
            thumb = cv2.resize(thumb, (max(int(thumb.shape[1] * scale), 1), CROP_HEIGHT), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out_dir / f"{frame}.jpg"), thumb, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if len(pending) >= ENCODE_BATCH:
            encode_pending()
    encode_pending()

    if sums.shape[1] == 0:
        sums = np.zeros((len(people), 256), dtype=np.float32)
    norms = np.linalg.norm(sums, axis=1, keepdims=True)
    vectors = sums / np.maximum(norms, 1e-6)
    seen = norms[:, 0] > 0  # a frame past the end of the video gave no crop
    groups = [[person for person in group if seen[person]] for group in groups]
    actor_rows = {frame: a["row"] for frame, a in actors.items() if "row" in a and seen[a["row"]]}

    read: dict[int, str] = {}
    shares: dict[int, float] = {}
    if numbers is not None and torsos:
        reads_by_action: dict[int, list[tuple[str, float]]] = defaultdict(list)
        for owner, text_score in zip(torso_owner, numbers.read(torsos)):
            reads_by_action[owner].append(text_score)
        allowed = {str(c) for c in candidates} if candidates else None
        for frame, reads in reads_by_action.items():
            number, share = decide_number(reads, candidates=allowed)
            if number is not None and frame in actor_rows:
                read[frame] = number
                shares[frame] = share
    enough = numbers is not None and (bool(candidates) and len(read) > 0 or len(read) >= max(min_numbered, 0.15 * len(actor_rows)))

    labels, margins = cluster(vectors, [g for g in groups if g], k)
    cluster_of = {f: int(labels[row]) for f, row in actor_rows.items()}
    # Number the clusters by how many actions they took, 1 = the most.
    taken = Counter(cluster_of.values())
    order = sorted(range(k), key=lambda c: (-taken.get(c, 0), c))
    rank = {cluster_id: position + 1 for position, cluster_id in enumerate(order)}

    # The clusters cover every action, so any reads may name them (the votes guard each name):
    # no ``min_numbered`` here, it is for the numbers standing alone.
    if read and name_clusters:
        mode = "mixed"
        by_track = {f: n for f, n in _numbers_by_track(actors, read).items() if f in actor_rows}
        named = _numbers_by_cluster(cluster_of, {**read, **by_track}, min_votes, min_share)
        assignments = []
        for f, row in sorted(actor_rows.items()):
            c = cluster_of[f]
            if f in read:
                item = {"number": read[f], "source": "number", "margin": round(shares[f], 4)}
            elif f in by_track:
                item = {"number": by_track[f], "source": "track", "margin": None}
            elif c in named:
                item = {"number": named[c], "source": "cluster_number", "margin": round(float(margins[row]), 4)}
            else:
                item = {"number": str(UNNAMED_CLUSTER_BASE + rank[c]), "source": "cluster", "margin": round(float(margins[row]), 4)}
            assignments.append({"frame": f, **item, "crop": f"{f}.jpg"})
        players = sorted({a["number"] for a in assignments} | {str(c) for c in candidates or []}, key=lambda n: int(n))
    elif enough:
        mode = "number"
        by_track = {f: n for f, n in _numbers_by_track(actors, read).items() if f in actor_rows}
        by_look = _numbers_by_appearance(vectors, actor_rows, {**read, **by_track}) if fill_by_appearance else {}
        assignments = [
            {"frame": f, "number": read[f], "source": "number", "margin": round(shares[f], 4), "crop": f"{f}.jpg"}
            if f in read
            else {"frame": f, "number": by_track[f], "source": "track", "margin": None, "crop": f"{f}.jpg"}
            if f in by_track
            else {"frame": f, "number": by_look[f][0], "source": "appearance", "margin": round(by_look[f][1], 4), "crop": f"{f}.jpg"}
            for f in sorted(actor_rows)
            if f in read or f in by_track or f in by_look
        ]
        players = sorted({a["number"] for a in assignments} | {str(c) for c in candidates or []}, key=lambda n: int(n))
    else:
        mode = "cluster"
        assignments = [
            {"frame": f, "number": str(rank[cluster_of[f]]), "source": "cluster", "margin": round(float(margins[row]), 4), "crop": f"{f}.jpg"}
            for f, row in sorted(actor_rows.items())
        ]
        players = [str(n) for n in range(1, k + 1)]
    assigned = {a["frame"] for a in assignments}
    result = {"mode": mode, "k": k, "side": side, "model": getattr(encoder, "name", ""), "players": players,
              "assignments": assignments, "skipped": [f for f in frames if f not in assigned],
              "numbersRead": len(read), "people": len(people)}
    (out_dir / "players.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    # The actors' vectors: the portal ranks the actions by likeness to the ones sorted by hand.
    save_vectors(out_dir / VECTORS_FILE, {f: vectors[row] for f, row in actor_rows.items()}, getattr(encoder, "name", ""))
    if debug_path is not None:
        # What the sorting decided from: to tune it offline without reading the video again.
        debug_path.write_text(json.dumps([
            {"frame": f, "track": actors[f]["track"], "rally": actors[f]["rally"], "read": read.get(f), "share": shares.get(f),
             "cluster": cluster_of[f],
             "vector": [round(float(v), 5) for v in vectors[row]]}
            for f, row in sorted(actor_rows.items())
        ]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video")
    parser.add_argument("predictions")
    parser.add_argument("court_json")
    parser.add_argument("actions_json", help='[{"frame": 3674, "track_id": 132, "ball": [x, y], "side": "near"}, ...]')
    parser.add_argument("out_dir")
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--side", default="near", choices=["near", "far"])
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--numbers", action="store_true", help="read the jersey numbers (pose + PP-OCRv6)")
    parser.add_argument("--candidates", nargs="*", default=None, help="the team's numbers")
    parser.add_argument("--numbers-only", action="store_true", help="no cluster pass: only the actions whose number was read")
    parser.add_argument("--min-votes", type=int, default=2)
    parser.add_argument("--min-share", type=float, default=0.7)
    parser.add_argument("--debug", default=None, help="json with the actors' vectors, tracks and reads")
    args = parser.parse_args()
    reader = None
    if args.numbers:
        from jersey_numbers import NumberReader

        reader = NumberReader(device=args.device)
    result = assign_players(
        args.video,
        args.predictions,
        json.loads(Path(args.court_json).read_text()),
        json.loads(Path(args.actions_json).read_text()),
        Path(args.out_dir),
        ReidEncoder(Path(args.model), args.device),
        args.k,
        args.side,
        numbers=reader,
        candidates=args.candidates,
        debug_path=Path(args.debug) if args.debug else None,
        name_clusters=not args.numbers_only,
        min_votes=args.min_votes,
        min_share=args.min_share,
    )
    per_player = defaultdict(int)
    for item in result["assignments"]:
        per_player[item["number"]] += 1
    sources = Counter(item["source"] for item in result["assignments"])
    print(f"mode {result['mode']}: {len(result['assignments'])} actions assigned {dict(sources)}, {len(result['skipped'])} skipped, "
          f"per player {dict(sorted(per_player.items(), key=lambda t: int(t[0])))}")


if __name__ == "__main__":
    main()
