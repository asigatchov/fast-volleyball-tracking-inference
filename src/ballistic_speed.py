"""Single-view ballistic fitting in meters: X across, Y near-to-far, Z up.

Centers drive the image-space fit; diameters are a weak depth prior. Each
manually supplied interval must contain one free flight, without a contact.
"""
from dataclasses import dataclass
import json
import subprocess

import numpy as np
from scipy.optimize import least_squares

GRAVITY = 9.81
MIN_OBSERVATIONS = 6
MIN_DURATION_SECONDS = 0.1


def frame_times(video, fps, frame_count):
    """Use presentation timestamps (also for VFR), with an explicit CFR fallback."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
             "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(video)],
            capture_output=True, text=True, check=True, timeout=60)
        times = np.array([float(f["best_effort_timestamp_time"])
                          for f in json.loads(result.stdout)["frames"]])
        if len(times) > 1 and np.isfinite(times).all() and np.all(np.diff(times) > 0):
            return times - times[0], "video_pts"
    except (OSError, subprocess.SubprocessError, ValueError, KeyError):
        pass
    if frame_count <= 0:
        raise ValueError("No valid frame timestamps or frame count available.")
    print("WARNING: frame PTS unavailable; falling back to frame_index/FPS (assumes CFR).")
    return np.arange(frame_count) / fps, "fps_fallback"


def trajectory(parameters, time):
    t = np.asarray(time, dtype=float)
    points = parameters[:3] + t[..., None] * parameters[3:]
    points[..., 2] -= 0.5 * GRAVITY * t ** 2
    return points


def velocity(parameters, time):
    t = np.asarray(time, dtype=float)
    values = np.broadcast_to(parameters[3:], t.shape + (3,)).copy()
    values[..., 2] -= GRAVITY * t
    return values


@dataclass
class Flight:
    start: int
    end: int
    t0: float
    parameters: np.ndarray
    frames: np.ndarray
    observed: np.ndarray
    diameters: np.ndarray
    errors: np.ndarray
    predicted_diameters: np.ndarray
    inliers: np.ndarray
    confidence: str
    valid: bool
    summary: dict

    def position(self, timestamp):
        return trajectory(self.parameters, timestamp - self.t0)

    def velocity(self, timestamp):
        return velocity(self.parameters, timestamp - self.t0)


def fit_flight(camera, detections, timestamps, start, end, diameter=0.22,
               width=9., length=18., radius_scale=1., net_frames=()):
    frames = np.array([f for f in sorted(detections) if start <= f <= end and f < len(timestamps)], dtype=int)
    if len(frames) < MIN_OBSERVATIONS:
        raise ValueError(f"Segment {start}:{end}: at least {MIN_OBSERVATIONS} center observations required.")
    t0 = float(timestamps[start])
    time = timestamps[frames] - t0
    if time[-1] - time[0] < MIN_DURATION_SECONDS:
        raise ValueError(f"Segment {start}:{end}: observations must span at least {MIN_DURATION_SECONDS} seconds.")
    observed = np.array([detections[f].xy for f in frames])
    raw_size = np.array([2 * detections[f].radius * radius_scale for f in frames])
    size_valid = np.isfinite(raw_size) & (raw_size >= 2) & (raw_size <= camera.intrinsic[0, 2] * 0.3)
    sizes = raw_size.copy()
    # Size smoothing is separate; do not smooth the observed image centers.
    for i in range(len(sizes)):
        neighbors = np.flatnonzero((np.abs(time - time[i]) <= 0.075) & size_valid)
        if size_valid[i] and len(neighbors):
            sizes[i] = np.median(raw_size[neighbors])
    if np.count_nonzero(size_valid) < 3:
        raise ValueError(f"Segment {start}:{end}: at least 3 usable diameter observations required.")
    initial_points = np.array([camera.ball_position(observed[i], sizes[i] / 2, diameter)
                               for i in np.flatnonzero(size_valid)])
    initial_points[:, 2] += 0.5 * GRAVITY * time[size_valid] ** 2
    coefficients = np.linalg.lstsq(np.column_stack((np.ones(sum(size_valid)), time[size_valid])), initial_points, rcond=None)[0]
    initial = np.r_[coefficients[0], coefficients[1]]
    lo = np.array([-2., -3., diameter / 2])
    hi = np.array([width + 2, length + 3, 15.])
    initial[:3] = np.clip(initial[:3], lo + 1e-6, hi - 1e-6)
    initial[3:] = np.clip(initial[3:], -99.9, 99.9)
    duration = float(timestamps[end] - t0)
    grid = np.linspace(0, duration, max(30, end - start + 1))
    net_times = np.array([timestamps[f] - t0 for f in net_frames if start <= f <= end])
    focal = (camera.intrinsic[0, 0] + camera.intrinsic[1, 1]) / 2

    def residual(parameters, mask):
        points = trajectory(parameters, time)
        cam = points @ camera.rotation.T + camera.translation
        depth = np.maximum(cam[:, 2], 0.1)
        # Camera.pixels includes lens distortion when supplied.
        predicted = camera.pixels(points, allow_behind=True)
        errors = [(predicted[mask] - observed[mask]).ravel() / 1.5,
                  (focal * diameter / depth[size_valid] - sizes[size_valid]) / 6.0]
        bound_points = trajectory(parameters, grid)
        bound_depth = (bound_points @ camera.rotation.T + camera.translation)[:, 2]
        errors += [(np.maximum(lo - bound_points, 0) * 100).ravel(),
                   (np.maximum(bound_points - hi, 0) * 100).ravel(),
                   np.maximum(0.1 - bound_depth, 0) * 100]
        if len(net_times):
            errors.append((trajectory(parameters, net_times)[:, 1] - length / 2) / 0.05)
        return np.concatenate(errors)

    mask = np.ones(len(frames), dtype=bool)
    result = least_squares(residual, initial, args=(mask,), loss="soft_l1", f_scale=1.,
                           bounds=(np.r_[lo, [-100.] * 3], np.r_[hi, [100.] * 3]), max_nfev=1000, x_scale="jac")
    error = np.linalg.norm(camera.pixels(trajectory(result.x, time), allow_behind=True) - observed, axis=1)
    median = np.median(error)
    cutoff = max(5., median + 3 * 1.4826 * np.median(np.abs(error - median)))
    mask = error <= cutoff
    enough_inliers = np.count_nonzero(mask) >= 6
    if not enough_inliers:
        mask[:] = True  # Keep diagnostics finite; mark the fit invalid below.
    if np.count_nonzero(mask) >= 6 and not mask.all():
        result = least_squares(residual, result.x, args=(mask,), loss="soft_l1", f_scale=1.,
                               bounds=(np.r_[lo, [-100.] * 3], np.r_[hi, [100.] * 3]), max_nfev=1000, x_scale="jac")
    points = trajectory(result.x, time)
    error = np.linalg.norm(camera.pixels(points, allow_behind=True) - observed, axis=1)
    depths = (points @ camera.rotation.T + camera.translation)[:, 2]
    predicted_sizes = focal * diameter / np.maximum(depths, 0.1)
    rmse = float(np.sqrt(np.mean(error[mask] ** 2)))
    # Check extrema as well as samples: Z can peak between observations.
    extrema = np.r_[grid, np.clip(result.x[5] / GRAVITY, 0, duration)]
    check = trajectory(result.x, extrema)
    valid = bool(result.success and enough_inliers and np.count_nonzero(mask) >= 6 and
                 np.all(check >= lo - 0.02) and np.all(check <= hi + 0.02) and
                 np.all((check @ camera.rotation.T + camera.translation)[:, 2] > 0.1) and
                 (not len(net_times) or np.all(np.abs(trajectory(result.x, net_times)[:, 1] - length / 2) < 0.1)))
    fraction = float(np.mean(mask))
    size_rmse = float(np.sqrt(np.mean((predicted_sizes[size_valid] - raw_size[size_valid]) ** 2)))
    confidence = "low"
    if valid and fraction >= 0.8 and camera.rmse < 5 and size_rmse < 5:
        if np.count_nonzero(mask) > 15 and rmse < 1 and duration >= 0.3:
            confidence = "high"
        elif np.count_nonzero(mask) >= 8 and rmse < 3:
            confidence = "medium"
    speed_samples = np.linalg.norm(velocity(result.x, grid), axis=1)
    summary = dict(start_frame=start, end_frame=end, t0=t0, parameters=result.x.tolist(), valid=valid,
                   confidence=confidence, observations=len(frames), inliers=int(sum(mask)),
                   pixel_rmse=rmse, diameter_rmse_px=size_rmse, duration_s=duration,
                   speed_at_start_mps=float(speed_samples[0]), speed_at_end_mps=float(speed_samples[-1]),
                   max_speed_mps=float(max(speed_samples[0], speed_samples[-1])),
                   mean_speed_mps=float(np.trapezoid(speed_samples, grid) / duration),
                   note="Confidence is heuristic; manual interval must exclude contacts.")
    return Flight(start, end, t0, result.x, frames, observed, raw_size, error, predicted_sizes,
                  mask, confidence, valid, summary)


def _local_velocity(times, centers, index, direction, window, max_gap):
    """Image velocity in px/s from up to `window` contiguous samples on one side of `index`."""
    indices = [index]
    for _ in range(window):
        neighbor = indices[-1] + direction
        if not 0 <= neighbor < len(times) or abs(times[neighbor] - times[indices[-1]]) > max_gap:
            break
        indices.append(neighbor)
    if len(indices) < 3:
        return None
    time = times[indices] - times[index]
    return np.linalg.lstsq(np.column_stack((np.ones(len(time)), time)), centers[indices], rcond=None)[0][1]


def contact_scores(times, centers, window=3, max_gap=0.25, turn_degrees=35., speed_ratio=2., min_speed_px=40.):
    """Rate how sharply the apparent motion breaks at each observation; >= 1 proposes a contact.

    Free flight bends the image track gradually; a contact turns it or changes its
    apparent speed at once. A near-vertical toss also reverses smoothly at its apex,
    so these are proposals only: the ballistic fit decides where a segment ends.
    """
    scores = np.zeros(len(times))
    for i in range(len(times)):
        before = _local_velocity(times, centers, i, -1, window, max_gap)
        after = _local_velocity(times, centers, i, 1, window, max_gap)
        if before is None or after is None:
            continue
        slow, fast = sorted((float(np.linalg.norm(before)), float(np.linalg.norm(after))))
        if fast < min_speed_px:
            continue
        cosine = float(before @ after) / (slow * fast) if slow > 1e-6 else -1.
        turn = float(np.degrees(np.arccos(np.clip(cosine, -1., 1.))))
        scores[i] = max(turn / turn_degrees, fast / max(slow, 1e-6) / speed_ratio)
    return scores


def contact_candidates(times, scores, min_separation=0.15):
    """Strongest proposal per `min_separation` window; one contact must not cut twice."""
    chosen = []
    for index in sorted(np.flatnonzero(scores >= 1.), key=lambda i: -scores[i]):
        if all(abs(times[index] - times[other]) > min_separation for other in chosen):
            chosen.append(int(index))
    return sorted(chosen)


def detect_segments(camera, detections, timestamps, diameter=0.22, width=9., length=18.,
                    radius_scale=1., max_rmse_px=3., max_gap_seconds=0.25, max_splits=4):
    """Find free-flight intervals automatically, as inclusive (start, end) video frames.

    Detection gaps and image-space turns only propose cuts. The ballistic fit itself
    accepts an interval: one that does not fit is split again at its worst
    observation, and neighbours that still fit together are merged, so a smooth
    parabola is not broken at its apex. Intervals that never fit are dropped, so
    every returned segment is a parabola, not a guess at where a contact happened.
    """
    frames = np.array([f for f in sorted(detections) if 0 <= f < len(timestamps)], dtype=int)
    if len(frames) < MIN_OBSERVATIONS:
        return []
    times = np.asarray(timestamps, dtype=float)[frames]
    centers = np.array([detections[f].xy for f in frames], dtype=float)
    scores = contact_scores(times, centers, max_gap=max_gap_seconds)
    fits = {}

    def fit(a, b):
        if (a, b) not in fits:
            try:
                fits[(a, b)] = fit_flight(camera, detections, timestamps, int(frames[a]), int(frames[b]),
                                          diameter, width, length, radius_scale)
            except ValueError:
                fits[(a, b)] = None
        return fits[(a, b)]

    def accepted(flight):
        return flight is not None and flight.valid and flight.summary["pixel_rmse"] <= max_rmse_px

    def pivot(a, b, flight):
        """Interior observation to drop: the worst fitted one, else the sharpest turn."""
        if flight is not None and len(flight.errors) == b - a + 1:
            return a + 1 + int(np.argmax(flight.errors[1:-1]))
        interior = scores[a + 1:b]
        return a + 1 + int(np.argmax(interior)) if np.any(interior > 0) else (a + b) // 2

    def split(a, b, depth=0):
        if b - a + 1 < MIN_OBSERVATIONS or times[b] - times[a] < MIN_DURATION_SECONDS:
            return []
        flight = fit(a, b)
        if accepted(flight):
            return [(a, b)]
        if depth >= max_splits:
            return []
        cut = pivot(a, b, flight)
        return split(a, cut - 1, depth + 1) + split(cut + 1, b, depth + 1)

    runs, first = [], 0
    for i in range(1, len(frames)):
        if times[i] - times[i - 1] > max_gap_seconds:
            runs.append((first, i - 1))
            first = i
    runs.append((first, len(frames) - 1))
    segments = []
    for start, end in runs:
        lower = start
        for candidate in contact_candidates(times[start:end + 1], scores[start:end + 1]):
            index = start + candidate
            if start < index < end:
                segments += split(lower, index - 1)
                lower = index + 1
        segments += split(lower, end)
    index = 0
    while index + 1 < len(segments):
        (a, b), (c, d) = segments[index], segments[index + 1]
        merged = fit(a, d) if times[c] - times[b] <= max_gap_seconds else None
        apart = max(fit(a, b).summary["pixel_rmse"], fit(c, d).summary["pixel_rmse"])
        if accepted(merged) and merged.summary["pixel_rmse"] <= 1.5 * apart + 0.5:
            segments[index:index + 2] = [(a, d)]
        else:
            index += 1
    return [(int(frames[a]), int(frames[b])) for a, b in segments]

def save_plots(path, flights, timestamps):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure(figsize=(12, 10), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(3, 2)
    for index, flight in enumerate(flights):
        times = timestamps[flight.start:flight.end + 1]
        points = flight.position(times)
        speed = np.linalg.norm(flight.velocity(times), axis=1)
        name = f"{flight.start}:{flight.end} ({flight.confidence})"
        for i in range(3):
            axes.flat[i].plot(times, points[:, i], label=name)
        axes.flat[3].plot(times, speed * 3.6, label=name)
        axes.flat[4].scatter(timestamps[flight.frames], flight.diameters, s=8, label=f"{index}: observed")
        axes.flat[4].plot(timestamps[flight.frames], flight.predicted_diameters, label=f"{index}: predicted")
        axes.flat[5].plot(timestamps[flight.frames], flight.errors, label=name)
    for axis, title in zip(axes.flat, ("X (m)", "Y near-to-far (m)", "Z (m)", "Speed (km/h)", "Diameter (px)", "Reprojection error (px)")):
        axis.set_title(title)
        axis.set_xlabel("Video time (s)")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7)
    figure.savefig(path)
