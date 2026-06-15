Главный вывод: 0001/0007 отличаются от 0004/0012 не одним параметром, а комбинацией “длина эпизода + суммарный путь + покрытие площадки + количество фаз движения”. Это как раз хорошо подходит для простого классификатора по траектории.

import json
import math
import numpy as np
from pathlib import Path


def load_positions(path):
    data = json.load(open(path))
    fps = data.get("fps", 30.0)

    positions = []
    for xy, frame in data["positions"]:
        x, y = xy
        positions.append((int(frame), float(x), float(y)))

    positions.sort()
    return data, positions, fps


def extract_track_features(path):
    data, positions, fps = load_positions(path)

    frames = np.array([p[0] for p in positions], dtype=np.float32)
    xs = np.array([p[1] for p in positions], dtype=np.float32)
    ys = np.array([p[2] for p in positions], dtype=np.float32)

    if len(positions) < 3:
        return None

    frame_diffs = np.diff(frames)
    dt = frame_diffs / fps
    dt = np.maximum(dt, 1.0 / fps)

    dx = np.diff(xs)
    dy = np.diff(ys)

    step_dist = np.sqrt(dx * dx + dy * dy)
    speed = step_dist / dt

    vx = dx / dt
    vy = dy / dt

    # Только почти непрерывные куски, чтобы большие разрывы трека
    # не давали ложные скачки направления.
    continuous = frame_diffs <= 2

    def count_sign_changes(values, mask, eps=60.0):
        values = values[mask]
        signs = np.sign(values)
        signs[np.abs(values) < eps] = 0
        signs = signs[signs != 0]

        if len(signs) < 2:
            return 0

        return int(np.sum(signs[1:] * signs[:-1] < 0))

    vy_sign_changes = count_sign_changes(vy, continuous)
    vx_sign_changes = count_sign_changes(vx, continuous)

    features = {
        "track_id": data.get("track_id"),
        "frame_start": int(frames[0]),
        "frame_end": int(frames[-1]),
        "duration_sec": float((frames[-1] - frames[0] + 1) / fps),
        "points_count": int(len(positions)),
        "coverage": float(len(positions) / (frames[-1] - frames[0] + 1)),

        "x_range_px": float(xs.max() - xs.min()),
        "y_range_px": float(ys.max() - ys.min()),
        "path_len_px": float(step_dist.sum()),

        "median_speed_px_s": float(np.median(speed)),
        "p90_speed_px_s": float(np.percentile(speed, 90)),
        "max_speed_px_s": float(speed.max()),

        "vy_sign_changes": vy_sign_changes,
        "vx_sign_changes": vx_sign_changes,

        "gap_count_gt5": int(np.sum(frame_diffs > 5)),
        "max_gap_frames": int(frame_diffs.max()),

        "has_game_pause": data.get("trajectory_analysis", {}).get("game_pause_frame") is not None,
        "has_rolling": data.get("trajectory_analysis", {}).get("rolling_start_frame") is not None,
    }

    return features


def classify_game_or_prepare(features):
    score = 0

    if features["duration_sec"] >= 4.5:
        score += 2

    if features["path_len_px"] >= 2200:
        score += 2

    if features["x_range_px"] >= 450:
        score += 1

    if features["y_range_px"] >= 500:
        score += 1

    if features["p90_speed_px_s"] >= 900:
        score += 1

    if features["vy_sign_changes"] >= 8:
        score += 1

    # rolling/game_pause — сильный признак подготовки/конца эпизода,
    # но не всегда он будет найден корректно, поэтому штраф мягкий.
    if features["has_game_pause"] or features["has_rolling"]:
        score -= 1

    label = "game" if score >= 5 else "prepare"
    return label, score


for path in sorted(Path(".").glob("track_*.json")):
    features = extract_track_features(path)
    label, score = classify_game_or_prepare(features)

    print(path.name, label, score, features)
