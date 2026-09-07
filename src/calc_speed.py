"""Visualize straight-line ball speed from its apparent radius and court geometry.

Coordinates: x=0..9 m (left to right), y=0..18 m (near to far), z up.
Default physics mode fits a ballistic trajectory on each --segment START:END
(inclusive zero-based video frames, one free flight between contacts); without
--segment the free-flight parabolas are detected automatically.
Centers drive fitting; apparent size is a weak depth prior. Assumptions:
square pixels, image-centered principal point, no lens distortion, fixed camera,
and Radius measures the physical ball's apparent radius (approximately the
minor radius of its projected ellipse). Radius errors directly affect depth.
CSV coordinates/radii must use video pixels. Frame is zero-based by default.
Use --ball-boxes project.json to replace CSV centers/radii with normalized YOLO
ball boxes: radius=min(box_width_px, box_height_px)/2, not half the diagonal.
The smaller side reduces sensitivity to elongation, but is still a size estimate.
--mode floor retains the original ray/floor-intersection approximation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

if __package__:
    from .ballistic_speed import detect_segments, fit_flight, frame_times, save_plots
else:
    from ballistic_speed import detect_segments, fit_flight, frame_times, save_plots


@dataclass
class Detection:
    xy: np.ndarray
    radius: float
    box: np.ndarray | None = None  # pixel xyxy


@dataclass
class Camera:
    intrinsic: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    rmse: float
    dist: np.ndarray = field(default_factory=lambda: np.zeros(5))

    def ball_position(self, xy, radius, diameter):
        if not math.isfinite(radius) or radius <= 0:
            return None
        # Small-sphere pinhole approximation: Z_camera = f * D / (2*r).
        # Z is optical-axis depth, not Euclidean distance to the camera.
        depth = np.mean([self.intrinsic[0, 0], self.intrinsic[1, 1]]) * diameter / (2 * radius)
        normalized = cv2.undistortPoints(np.asarray(xy, float).reshape(1, 1, 2), self.intrinsic, self.dist)[0, 0]
        ray = np.r_[normalized, 1.]
        return self.rotation.T @ (ray * depth - self.translation)

    def pixels(self, xyz, allow_behind=False):
        camera = np.asarray(xyz) @ self.rotation.T + self.translation
        if not allow_behind and np.any(camera[:, 2] <= 0):
            raise ValueError("Projected geometry lies behind the camera.")
        camera[:, 2] = np.maximum(camera[:, 2], 0.1)
        return cv2.projectPoints(camera, np.zeros(3), np.zeros(3), self.intrinsic, self.dist)[0].reshape(-1, 2)


def calibrate_camera(points, width, height, net_height=2.43, court_width=9., court_length=18., intrinsics=None):
    if not all(i in points for i in range(8)):
        raise ValueError("Radius mode requires all eight court/net points (ids 0–7).")
    w, l = court_width, court_length
    world = np.array([(0, 0, 0), (0, l, 0), (w, l, 0), (w, 0, 0),
                      (0, l / 2, 0), (w, l / 2, 0), (0, l / 2, net_height), (w, l / 2, net_height)], np.float32)
    pixels = np.array([points[i] for i in range(8)], np.float32)
    if not np.isfinite(pixels).all():
        raise ValueError("Court/net coordinates must be finite.")
    if intrinsics:
        data = json.loads(intrinsics.read_text())
        intrinsic = np.asarray(data["K"], dtype=float)
        dist = np.asarray(data.get("dist", [0.] * 5), dtype=float)
        if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all() or np.any(np.diag(intrinsic) <= 0):
            raise ValueError("Intrinsics require a finite 3x3 K with positive focal lengths, in video pixels.")
        ok, rvec, tvec = cv2.solvePnP(world, pixels, intrinsic, dist)
        if not ok:
            raise ValueError("solvePnP failed.")
        camera = Camera(intrinsic, cv2.Rodrigues(rvec)[0], tvec.ravel(), 0., dist)
        camera.rmse = float(np.sqrt(np.mean(np.sum((camera.pixels(world) - pixels) ** 2, axis=1))))
        print(f"Known-intrinsics calibration RMSE: {camera.rmse:.2f} px")
        return camera
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT |
             cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST |
             cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
    candidates = []
    for focal in (width * 0.7, width, width * 2, width * 4):
        intrinsic = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.0]])
        try:
            error, intrinsic, _, rotations, translations = cv2.calibrateCamera(
                [world], [pixels], (width, height), intrinsic, None, flags=flags)
            rotation = cv2.Rodrigues(rotations[0])[0]
            translation = translations[0].ravel()
            camera_center = -rotation.T @ translation
            if (np.isfinite(error) and np.isfinite(intrinsic).all() and
                    camera_center[2] > 0 and np.all((world @ rotation.T + translation)[:, 2] > 0)):
                candidates.append(Camera(intrinsic, rotation, translation, error))
        except cv2.error:
            continue
    if not candidates:
        raise ValueError("Cannot estimate a physically valid camera from court/net points.")
    camera = min(candidates, key=lambda item: item.rmse)
    print(f"Camera calibration RMSE: {camera.rmse:.2f} px; focal length: {camera.intrinsic[0, 0]:.1f} px")
    if camera.rmse > 5:
        print("WARNING: calibration residual exceeds 5 px; check annotations/camera assumptions.")
    return camera


def draw_metric_lines(frame, camera, line_width, width=9., length=18.):
    """Project actual metric strips on the floor, not fixed pixel thicknesses."""
    overlay = frame.copy()
    half = line_width / 2
    segments = [((0, 0), (width, 0)), ((width, 0), (width, length)), ((width, length), (0, length)),
                ((0, length), (0, 0)), ((0, length / 2), (width, length / 2))]
    if (width, length) == (9., 18.):
        segments += [((0, 6), (width, 6)), ((0, 12), (width, 12))]
    for start, end in segments:
        a, b = np.array(start, float), np.array(end, float)
        delta = b - a
        offset = np.array([-delta[1], delta[0]]) / np.linalg.norm(delta) * half
        corners = np.array([a + offset, b + offset, b - offset, a - offset])
        pixels = camera.pixels(np.column_stack((corners, np.zeros(4))))
        cv2.fillConvexPoly(overlay, np.rint(pixels).astype(np.int32), (80, 220, 80), cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, dst=frame)


@dataclass
class Speed:
    start: np.ndarray
    end: np.ndarray
    seconds: float
    mps: float
    angle: float  # clockwise from the top of the diagram


def load_court(path: Path, width: int, height: int, court_width=9., court_length=18.):
    data = json.loads(path.read_text(encoding="utf-8"))
    scale = np.array([width / data["frame_width"], height / data["frame_height"]])
    points = {}
    for p in data["keypoints"]:
        if p.get("visible", True) and p.get("x") is not None and p.get("y") is not None:
            points[int(p["id"])] = np.array([p["x"], p["y"]], dtype=float) * scale
    if not all(i in points for i in range(6)):
        raise ValueError("Court JSON requires visible points 1–6 (ids 0–5).")
    pixels = np.array([points[i] for i in range(6)], dtype=np.float64)
    w, l = court_width, court_length
    meters = np.array([(0, 0), (0, l), (w, l), (w, 0), (0, l / 2), (w, l / 2)], dtype=float)
    if not np.isfinite(pixels).all():
        raise ValueError("Court coordinates must be finite.")
    corners = pixels[:4].astype(np.float32)
    if not cv2.isContourConvex(corners) or abs(cv2.contourArea(corners)) < 1:
        raise ValueError("Court corners 1–4 must form a non-degenerate ordered quadrilateral.")
    matrix, _ = cv2.findHomography(pixels, meters, method=0)
    if matrix is None or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) < 3:
        raise ValueError("Cannot fit a floor homography to this annotation.")
    back = cv2.perspectiveTransform(meters[None], np.linalg.inv(matrix))[0]
    error = np.sqrt(np.mean(np.sum((back - pixels) ** 2, axis=1)))
    print(f"Floor calibration reprojection RMSE: {error:.2f} px")
    return matrix, points


def load_ball_boxes(path: Path, width: int, height: int, ball_class: int = 6, frame_base: int = 0):
    data = json.loads(path.read_text(encoding="utf-8"))
    detections = {}
    for frame_key, boxes in data["yolo_boxes"].items():
        frame = int(frame_key) - frame_base
        selected = [b for b in boxes.values() if int(b[0]) == ball_class]
        if not selected:
            continue
        if len(selected) != 1:
            raise ValueError(f"Frame {frame}: expected one ball box, found {len(selected)}.")
        box = np.asarray(selected[0][1:5], dtype=float)
        if frame < 0 or box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= 0):
            raise ValueError(f"Frame {frame}: invalid ball box.")
        center = box[:2] * [width, height]
        size = box[2:] * [width, height]
        detections[frame] = Detection(center, float(min(size) / 2), np.r_[center - size / 2, center + size / 2])
    if not detections:
        raise ValueError(f"No ball boxes with class {ball_class}.")
    return detections


def compare_box_radii(csv_detections, box_detections):
    common = sorted(csv_detections.keys() & box_detections.keys())
    if not common:
        print("No common frames for CSV/box comparison.")
        return
    center_error = [np.max(np.abs(csv_detections[f].xy - box_detections[f].xy)) for f in common]
    diagonal_error = []
    for f in common:
        box = box_detections[f].box
        diagonal_error.append(abs(csv_detections[f].radius - np.linalg.norm(box[2:] - box[:2]) / 2))
    matches = sum(e <= 0.5 for e in diagonal_error)
    print(f"CSV/box comparison: {len(common)} common frames; max center difference {max(center_error):.3f}px; "
          f"CSV radius matches rounded half-diagonal in {matches}/{len(common)} frames.")


def load_ball(path: Path, frame_base: int = 0, require_radius: bool = True):
    detections = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"Frame", "Visibility", "X", "Y"}.issubset(reader.fieldnames or []):
            raise ValueError("Ball CSV requires Frame, Visibility, X, Y columns.")
        if require_radius and "Radius" not in reader.fieldnames:
            raise ValueError("Radius mode requires a Radius column in video pixels.")
        for line, row in enumerate(reader, 2):
            try:
                if float(row["Visibility"]) <= 0:
                    continue
                frame = int(row["Frame"]) - frame_base
                xy = np.array([float(row["X"]), float(row["Y"])])
                if frame < 0 or not np.isfinite(xy).all():
                    raise ValueError("invalid frame or coordinates")
                if frame in detections:
                    raise ValueError(f"duplicate frame {frame}")
                radius = float(row.get("Radius") or 0)
                detections[frame] = Detection(xy, radius)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid CSV row {line}: {exc}") from exc
    return detections


def project(matrix, xy):
    p = matrix @ np.array([*xy, 1.0])
    if not np.isfinite(p).all() or abs(p[2]) < 1e-8:
        return None
    return p[:2] / p[2]


class SpeedWindow:
    def __init__(self, fps: float, seconds: float = 0.5):
        self.fps = fps
        self.seconds = seconds
        self.samples = deque()

    def update(self, frame: int, point: np.ndarray | None) -> Speed | None:
        while self.samples and (frame - self.samples[0][0]) / self.fps > self.seconds + 1e-9:
            self.samples.popleft()
        if point is None:
            return None
        self.samples.append((frame, point))
        if len(self.samples) < 2:
            return None
        first_frame, start = self.samples[0]
        dt = (frame - first_frame) / self.fps
        delta = point - start
        distance = float(np.linalg.norm(delta))
        angle = math.degrees(math.atan2(delta[0], delta[1])) % 360 if np.linalg.norm(delta[:2]) > 1e-9 else math.nan
        return Speed(start, point, dt, distance / dt, angle)


def label(image, text, pos, scale=0.5, color=(235, 235, 235)):
    cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_panel(frame, speed, point, frame_id, fps, window, mode="radius", radius=None, diameter=0.22,
               court_width=9., court_length=18., flight=None, timestamp=None):
    panel = np.full((600, 300, 3), (35, 30, 25), dtype=np.uint8)
    title = {"physics": "BALLISTIC 3D / ESTIMATE", "radius": "RADIUS 3D / ESTIMATE", "floor": "FLOOR HOMOGRAPHY / APPROX."}
    label(panel, title[mode], (10, 24), 0.48)
    label(panel, f"Frame {frame_id} | {timestamp if timestamp is not None else frame_id / fps:.2f}s", (10, 48))
    extra = f" | r={radius:.1f}px" if radius is not None and math.isfinite(radius) else ""
    caption = f"Fit {flight.start}:{flight.end} | {flight.confidence}" if flight else f"Window: {window:.2f}s{extra}"
    label(panel, caption, (10, 71))
    if speed is None:
        label(panel, "Speed: --", (10, 103), 0.7)
        label(panel, "No valid position/radius" if point is None else "Need 2 detections", (10, 130))
    else:
        label(panel, f"{speed.mps * 3.6:.1f} km/h  {speed.mps:.2f} m/s", (10, 102), 0.63)
        heading = "--" if math.isnan(speed.angle) else f"{speed.angle:.0f} deg"
        label(panel, f"Direction: {heading} | dt={speed.seconds:.3f}s", (10, 128), 0.47)
    # Equal scale on both axes: a vertical 18 m x 9 m court.
    origin = np.array([60, 164])
    ppm = 360 / court_length
    def diagram(p):
        # Clipping happens in OpenCV; cap huge coordinates near the horizon.
        return tuple(np.clip(np.rint(origin + np.array([p[0], court_length - p[1]]) * ppm), -100000, 100000).astype(int))
    cv2.rectangle(panel, diagram(np.array([0, 0])), diagram(np.array([court_width, court_length])), (115, 85, 55), -1)
    cv2.rectangle(panel, diagram(np.array([0, 0])), diagram(np.array([court_width, court_length])), (255, 255, 255), 2)
    lines = [court_length / 2]
    if (court_width, court_length) == (9., 18.):
        lines += [6, 12]
    for y in lines:
        cv2.line(panel, diagram(np.array([0, y])), diagram(np.array([court_width, y])), (240, 240, 240), 2 if y == court_length / 2 else 1)
    if flight:
        points = flight.position(np.linspace(flight.t0, flight.t0 + flight.summary["duration_s"], 60))
        polyline = np.array([diagram(p) for p in points], np.int32)
        layer = panel.copy()
        cv2.polylines(layer, [polyline], False, (255, 170, 80), 1, cv2.LINE_AA)
        panel[145:546] = layer[145:546]
    for name, xy in (("2", (45, 162)), ("3", (245, 162)), ("1", (45, 540)), ("4", (245, 540))):
        label(panel, name, xy)
    if speed is not None:
        # Draw on a separate layer and confine drawing to diagram area.
        layer = panel.copy()
        cv2.arrowedLine(layer, diagram(speed.start), diagram(speed.end), (0, 230, 255), 2, cv2.LINE_AA, tipLength=0.2)
        panel[145:546] = layer[145:546]
        if not (0 <= speed.end[0] <= court_width and 0 <= speed.end[1] <= court_length) and not math.isnan(speed.angle):
            # Keep the heading visible even when the floor intersection is far away.
            direction = (speed.end - speed.start)[:2]
            direction = direction * [1, -1]
            end = np.rint(np.array([150, 344]) + 55 * direction / np.linalg.norm(direction)).astype(int)
            cv2.arrowedLine(panel, (150, 344), tuple(end), (0, 230, 255), 2, cv2.LINE_AA, tipLength=0.25)
            label(panel, "Heading", (115, 420), 0.45)
    if point is not None:
        xy = diagram(point)
        if 0 <= xy[0] < 300 and 145 <= xy[1] < 546:
            cv2.circle(panel, xy, max(1, round(diameter * ppm / 2)), (0, 160, 255), -1)
        if not (0 <= point[0] <= court_width and 0 <= point[1] <= court_length):
            label(panel, "Ball XY outside court", (10, 545), 0.46)
        if len(point) == 3:
            text = f"Height: {point[2]:.2f} m"
            if point[2] < diameter / 2:
                text += " ! below floor"
            label(panel, text, (10, 565), 0.46)
    label(panel, f"{court_width:g} x {court_length:g} m | 0 deg = far baseline", (10, 586), 0.46)
    ratio = min(1.0, frame.shape[0] / 620, frame.shape[1] / 640)
    panel = cv2.resize(panel, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
    ph, pw = panel.shape[:2]
    frame[0:ph, frame.shape[1] - pw:] = panel


def draw_flight(frame, camera, flight, timestamp, length, net_height, diameter, detection=None):
    times = np.linspace(flight.t0, flight.t0 + flight.summary["duration_s"], 80)
    points = flight.position(times)
    projected = camera.pixels(points, allow_behind=True)
    cv2.polylines(frame, [np.clip(np.rint(projected), -100000, 100000).astype(np.int32)], False, (255, 100, 255), 1, cv2.LINE_AA)
    current = flight.position(timestamp)
    predicted = camera.pixels([current], allow_behind=True)[0]
    center = tuple(np.clip(np.rint(predicted), -100000, 100000).astype(int))
    cv2.drawMarker(frame, center, (255, 100, 255), cv2.MARKER_CROSS, 12, 2)
    depth = float((camera.rotation @ current + camera.translation)[2])
    if depth > 0:
        radius = np.mean([camera.intrinsic[0, 0], camera.intrinsic[1, 1]]) * diameter / (2 * depth)
        cv2.circle(frame, center, max(1, min(100000, round(radius))), (255, 100, 255), 1, cv2.LINE_AA)
    error = float(np.linalg.norm(predicted - detection.xy)) if detection else None
    caption = f"Fit error: {error:.2f}px" if error is not None else "Fit prediction (missing observation)"
    if not flight.valid:
        caption += " | INVALID FIT"
    label(frame, caption, (12, 48), 0.5, (255, 100, 255))
    side = np.full((195, 330, 3), (35, 30, 25), np.uint8)
    label(side, "Side view: Y / Z (m)", (10, 20))
    def yz(p):
        return tuple(np.clip(np.rint([25 + p[1] / length * 280, 170 - p[2] * 10]), -10000, 10000).astype(int))
    cv2.line(side, (25, 170), (305, 170), (255, 255, 255), 1)
    cv2.line(side, yz([0, length / 2, 0]), yz([0, length / 2, net_height]), (80, 220, 80), 2)
    cv2.polylines(side, [np.array([yz(p) for p in points])], False, (255, 100, 255), 1, cv2.LINE_AA)
    cv2.circle(side, yz(current), 4, (0, 230, 255), -1)
    label(side, f"RMSE {flight.summary['pixel_rmse']:.2f}px | {flight.confidence}", (10, 190), 0.45)
    scale = min(1., frame.shape[0] / 720, frame.shape[1] / 1280)
    side = cv2.resize(side, None, fx=scale, fy=scale)
    h, w = side.shape[:2]
    frame[frame.shape[0] - h - 65:frame.shape[0] - 65, 10:10 + w] = side


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--court", required=True, type=Path, help="Court JSON (ids 0–7)")
    parser.add_argument("--ball", type=Path, help="Ball CSV in video pixels (optional with --ball-boxes)")
    parser.add_argument("--ball-boxes", type=Path, help="Project JSON: use ball box centers and half the smaller side as radius; no CSV fallback")
    parser.add_argument("--ball-class", type=int, default=6, help="Ball class in project YOLO boxes (default 6)")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--window", type=float, default=0.5, help="Trailing window in seconds (default: 0.5)")
    parser.add_argument("--mode", choices=("physics", "radius", "floor"), default="physics")
    parser.add_argument("--segment", action="append", default=[], metavar="START:END", help="One free flight, inclusive zero-based frames; repeat for multiple flights. Omit for automatic detection")
    parser.add_argument("--segment-max-rmse", type=float, default=3., help="Automatic detection: largest accepted reprojection RMSE, pixels (default 3)")
    parser.add_argument("--segment-max-gap", type=float, default=0.25, help="Automatic detection: detection gap that always ends a flight, seconds (default 0.25)")
    parser.add_argument("--net-crossing", action="append", type=int, default=[], help="Optional confirmed net-plane crossing frame (not inferred from 2D overlap)")
    parser.add_argument("--court-width", type=float, default=9.)
    parser.add_argument("--court-length", type=float, default=18.)
    parser.add_argument("--intrinsics", type=Path, help="JSON with K and dist, calibrated at video resolution")
    parser.add_argument("--report", type=Path, help="Save calibration, segment metrics, and fit diagnostics as JSON")
    parser.add_argument("--plots", type=Path, help="Save X/Y/Z, speed, diameter, reprojection plots (PNG/PDF)")
    parser.add_argument("--ball-diameter", type=float, default=0.22, help="Physical diameter, meters")
    parser.add_argument("--net-height", type=float, default=2.43, help="Net height, meters")
    parser.add_argument("--line-width", type=float, default=0.05, help="Court stripe width, meters; for scale overlay only")
    parser.add_argument("--radius-scale", type=float, default=1.0, help="Explicit correction factor for CSV radius; default 1")
    parser.add_argument("--frame-base", type=int, choices=(0, 1), default=0)
    parser.add_argument("--no-show", action="store_true", help="Process without a GUI")
    parser.add_argument("--output", type=Path, help="Optionally save annotated MP4")
    parser.add_argument("--output-csv", type=Path, help="Optionally save per-frame estimates")
    args = parser.parse_args()
    if not args.ball and not args.ball_boxes:
        parser.error("Provide --ball-boxes or --ball")
    if args.mode != "physics" and (args.segment or args.plots or args.net_crossing):
        parser.error("--segment, --plots and --net-crossing require --mode physics")
    if any(not math.isfinite(v) or v <= 0 for v in (args.segment_max_rmse, args.segment_max_gap)):
        parser.error("--segment-max-rmse and --segment-max-gap must be positive and finite")
    if not math.isfinite(args.window) or args.window <= 0:
        parser.error("--window must be positive and finite")
    if any(not math.isfinite(v) or v <= 0 for v in (args.ball_diameter, args.net_height, args.line_width, args.radius_scale, args.court_width, args.court_length)):
        parser.error("Physical dimensions and --radius-scale must be positive and finite")
    if not args.no_show and sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        parser.error("No graphical display. Use --no-show --output preview.mp4, or run in a desktop session.")
    inputs = {p.resolve() for p in (args.video, args.court, args.ball, args.intrinsics) if p}
    if args.ball_boxes:
        inputs.add(args.ball_boxes.resolve())
    outputs = [p.resolve() for p in (args.output, args.output_csv, args.report, args.plots) if p]
    if any(p in inputs or p.exists() for p in outputs) or len(set(outputs)) != len(outputs):
        parser.error("Output paths must be distinct new files and must not overwrite inputs.")
    cap = cv2.VideoCapture(str(args.video))
    writer = stream = None
    count = estimates = processed = 0
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {args.video}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video has no valid FPS.")
        width, height = (int(cap.get(prop)) for prop in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
        matrix, court = load_court(args.court, width, height, args.court_width, args.court_length)
        camera = calibrate_camera(court, width, height, args.net_height, args.court_width, args.court_length, args.intrinsics) if args.mode != "floor" else None
        csv_detections = load_ball(args.ball, args.frame_base, require_radius=camera is not None and not args.ball_boxes) if args.ball else {}
        detections = csv_detections
        if args.ball_boxes:
            detections = load_ball_boxes(args.ball_boxes, width, height, args.ball_class, args.frame_base)
            if csv_detections:
                compare_box_radii(csv_detections, detections)
            print("Using JSON box centers and min(width, height)/2 radii; missing boxes stay missing.")
        timestamps, time_source = frame_times(args.video, fps, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        print(f"Timing: {time_source}; {len(timestamps)} frames")
        flights, frame_flights = [], {}
        if args.mode == "physics":
            intervals = []
            for interval in args.segment:
                try:
                    start, end = map(int, interval.split(":"))
                except ValueError:
                    parser.error("--segment must be START:END, zero-based inclusive video frames")
                if not 0 <= start < end < len(timestamps):
                    parser.error(f"Invalid segment {interval}; video has {len(timestamps)} frames")
                intervals.append((start, end))
            intervals.sort()
            if any(a[1] >= b[0] for a, b in zip(intervals, intervals[1:])):
                parser.error("Segments must not overlap; exclude contact frames between flights")
            if args.segment:
                if any(not any(a <= f <= b for a, b in intervals) for f in args.net_crossing):
                    parser.error("Each --net-crossing frame must be inside a fitted segment")
            else:
                intervals = detect_segments(camera, detections, timestamps, args.ball_diameter, args.court_width,
                                            args.court_length, args.radius_scale, args.segment_max_rmse, args.segment_max_gap)
                print(f"Automatic segmentation: {len(intervals)} free-flight parabolas "
                      f"({', '.join(f'{a}:{b}' for a, b in intervals) or 'none'}); contacts are not labelled.")
                if not intervals:
                    print("No interval fitted a parabola; check detections/calibration or raise --segment-max-rmse.")
                outside = [f for f in args.net_crossing if not any(a <= f <= b for a, b in intervals)]
                if outside:
                    print(f"WARNING: --net-crossing {outside} fall outside every detected segment and are ignored.")
            for start, end in intervals:
                flight = fit_flight(camera, detections, timestamps, start, end, args.ball_diameter,
                                    args.court_width, args.court_length, args.radius_scale, args.net_crossing)
                flights.append(flight)
                for f in range(start, end + 1):
                    frame_flights[f] = flight
                print(f"Fit {start}:{end}: {flight.summary['pixel_rmse']:.2f}px RMSE, {flight.confidence} confidence, "
                      f"valid={flight.valid}, speed {flight.summary['speed_at_start_mps'] * 3.6:.1f} -> {flight.summary['speed_at_end_mps'] * 3.6:.1f} km/h")
        if args.report:
            payload = dict(time_source=time_source, timestamps_s=timestamps.tolist(),
                           court_width=args.court_width, court_length=args.court_length,
                           net_height=args.net_height, ball_diameter=args.ball_diameter,
                           coordinates="X across; Y near-to-far; Z up",
                           calibration=None if camera is None else dict(K=camera.intrinsic.tolist(), R=camera.rotation.tolist(),
                             t=camera.translation.tolist(), dist=camera.dist.tolist(), camera_position=(-camera.rotation.T @ camera.translation).tolist(), rmse_px=camera.rmse),
                           flights=[dict(**f.summary, frames=f.frames.tolist(), observed_centers=f.observed.tolist(),
                                         observed_diameters=[float(d) if np.isfinite(d) else None for d in f.diameters], predicted_diameters=f.predicted_diameters.tolist(),
                                         pixel_errors=f.errors.tolist(), inlier_mask=f.inliers.tolist()) for f in flights])
            with args.report.open("x") as file:
                json.dump(payload, file, indent=2, allow_nan=False)
        if args.plots:
            save_plots(args.plots, flights, timestamps)
        # Evaluate in chronological order once; seeking must not change the
        # trailing window or mix samples from unrelated playback positions.
        window = SpeedWindow(fps, args.window)
        positions, speeds = {}, {}
        for frame_id in sorted(detections):
            if args.mode == "physics":
                break
            item = detections[frame_id]
            item_radius = item.radius * args.radius_scale
            position = camera.ball_position(item.xy, item_radius, args.ball_diameter) if camera else project(matrix, item.xy)
            positions[frame_id] = position
            speeds[frame_id] = window.update(frame_id, position)
        if args.mode == "physics":
            for frame_id, flight in frame_flights.items():
                if not flight.valid:
                    continue
                position = flight.position(timestamps[frame_id])
                v = flight.velocity(timestamps[frame_id])
                positions[frame_id] = position
                angle = math.degrees(math.atan2(v[0], v[1])) % 360 if np.linalg.norm(v[:2]) > 1e-9 else math.nan
                # The arrow is the instantaneous tangent; mps is the analytic derivative.
                speeds[frame_id] = Speed(position, position + v * 0.15, float(timestamps[frame_id] - flight.t0), float(np.linalg.norm(v)), angle)
        frame_count = len(timestamps)
        paused = False
        exporting = bool(args.output or args.output_csv)
        print("Ballistic fit: instantaneous derivative of fitted trajectory; manual intervals must exclude contacts." if args.mode == "physics" else
              "Radius-derived 3D estimate; radius noise affects depth and speed." if camera else
              "Approximate floor-plane speed; airborne ball speed is not recovered.")
        print("Controls: A/D = -1/+1 frame; W/S = +15/-15 frames; Space = pause/resume; Q or Esc = quit.")
        if exporting and not args.no_show:
            print("Exporting full video/CSV before opening interactive playback.")
        if args.output:
            if len(timestamps) > 2 and np.ptp(np.diff(timestamps)) > 0.001:
                print("WARNING: annotated MP4 uses constant FPS; fitting/CSV use original frame PTS.")
            writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not writer.isOpened():
                raise ValueError(f"Cannot create output video: {args.output}")
        if args.output_csv:
            stream = args.output_csv.open("x", newline="", encoding="utf-8")
            csv_writer = csv.writer(stream)
            csv_writer.writerow(["frame", "time_s", "mode", "radius_csv_px", "radius_used_px", "x_m", "y_m", "height_m",
                                 "camera_depth_m", "interval_s", "speed_mps", "speed_kmh", "direction_deg", "status", "size_source",
                                 "vx_mps", "vy_mps", "vz_mps", "confidence", "segment", "reprojection_px", "predicted_diameter_px"])
        if not args.no_show:
            cv2.namedWindow("Ball speed", cv2.WINDOW_NORMAL)
        while True:
            started = time.monotonic()
            ok, frame = cap.read()
            if not ok:
                if exporting and not args.no_show and count:
                    if writer:
                        writer.release()
                        writer = None
                    if stream:
                        stream.close()
                        stream = None
                    exporting = False
                    frame_count = count
                    count = 0
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            processed += 1
            detection = detections.get(count)
            timestamp = float(timestamps[count])
            flight = frame_flights.get(count)
            pixel = detection.xy if detection else None
            radius = detection.radius * args.radius_scale if detection else None
            point = positions.get(count)
            depth = (camera.rotation @ point + camera.translation)[2] if camera and point is not None else ""
            status = "ok" if point is not None else "missing_detection_or_radius"
            if args.mode == "physics":
                status = "outside_segment" if flight is None else "invalid_fit" if not flight.valid else "fitted_observation" if detection else "fitted_missing_detection"
            if camera and point is not None and point[2] < args.ball_diameter / 2:
                status = "below_floor_check_radius"
            speed = speeds.get(count)
            estimates += speed is not None
            if stream:
                csv_detection = csv_detections.get(count)
                v = flight.velocity(timestamp).tolist() if flight and flight.valid else ["", "", ""]
                reprojection = float(np.linalg.norm(camera.pixels([point])[0] - pixel)) if camera and point is not None and detection else ""
                predicted_size = float(np.mean([camera.intrinsic[0, 0], camera.intrinsic[1, 1]]) * args.ball_diameter / depth) if camera and point is not None else ""
                csv_writer.writerow([count, timestamp, args.mode, csv_detection.radius if csv_detection else "", radius if detection else "",
                                     *(point[:2] if point is not None else ("", "")), point[2] if camera and point is not None else "", depth,
                                     *([speed.seconds, speed.mps, speed.mps * 3.6, speed.angle] if speed else ("", "", "", "")), status,
                                     "box_min_side" if args.ball_boxes else "csv_radius", *v,
                                     flight.confidence if flight else "", f"{flight.start}:{flight.end}" if flight else "", reprojection, predicted_size])
            if camera:
                draw_metric_lines(frame, camera, args.line_width, args.court_width, args.court_length)
            if flight:
                draw_flight(frame, camera, flight, timestamp, args.court_length, args.net_height, args.ball_diameter, detection)
            for a, b in ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (6, 7)):
                if a in court and b in court and (camera is None or a == 6):
                    cv2.line(frame, tuple(court[a].astype(int)), tuple(court[b].astype(int)), (80, 220, 80), 1, cv2.LINE_AA)
            if pixel is not None:
                center = tuple(np.clip(pixel, -100000, 100000).astype(int))
                if detection.box is not None:
                    box = np.clip(np.rint(detection.box), -100000, 100000).astype(int)
                    cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), (255, 255, 0), 1, cv2.LINE_AA)
                if math.isfinite(detection.radius) and detection.radius > 0:
                    cv2.circle(frame, center, max(1, min(100000, round(radius))), (0, 230, 255), 1, cv2.LINE_AA)
                    source = "BOX" if detection.box is not None else "CSV"
                    label(frame, f"{source} r={radius:.1f}px", (center[0] + 10, center[1] - 16), 0.45, (0, 230, 255))
                else:
                    cv2.drawMarker(frame, center, (0, 230, 255), cv2.MARKER_CROSS, 10, 1)
            draw_panel(frame, speed, point, count, fps, args.window, args.mode, radius, args.ball_diameter,
                       args.court_width, args.court_length, flight, timestamp)
            if camera:
                label(frame, f"3D size estimate | calibration {camera.rmse:.1f}px | green floor stripes: {args.line_width * 100:g}cm", (12, height - 16), 0.5)
                if point is not None:
                    line_pixels = np.mean([camera.intrinsic[0, 0], camera.intrinsic[1, 1]]) * args.line_width / depth
                    label(frame, f"Optical depth {depth:.2f}m | ball D={args.ball_diameter * 100:g}cm | same-depth {args.line_width * 100:g}cm = {line_pixels:.1f}px", (12, height - 40), 0.5)
            if writer:
                writer.write(frame)
            if not args.no_show and not exporting:
                if frame_count > 0 and count >= frame_count - 1:
                    paused = True
                quit_requested = False
                while True:
                    display = frame.copy()
                    state = "PAUSED" if paused else "PLAYING"
                    label(display, f"{state} | A/D: -1/+1 | W/S: +15/-15 | Space: play/pause | Q: quit", (12, 24), 0.5)
                    cv2.imshow("Ball speed", display)
                    frame_delay = timestamps[count + 1] - timestamp if count + 1 < len(timestamps) else 1 / fps
                    delay = 50 if paused else max(1, round(1000 * (frame_delay - (time.monotonic() - started))))
                    key = cv2.waitKey(delay) & 0xFF
                    if key in (27, ord("q"), ord("Q")) or cv2.getWindowProperty("Ball speed", cv2.WND_PROP_VISIBLE) < 1:
                        quit_requested = True
                        break
                    steps = {ord("a"): -1, ord("d"): 1, ord("w"): 15, ord("s"): -15,
                             ord("A"): -1, ord("D"): 1, ord("W"): 15, ord("S"): -15}
                    if key in steps:
                        count = max(0, count + steps[key])
                        if frame_count > 0:
                            count = min(count, frame_count - 1)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, count)
                        paused = True
                        break
                    if key == 32:
                        paused = not paused
                        if frame_count > 0 and count == frame_count - 1 and not paused:
                            count = 0
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            break
                    if not paused:
                        count += 1
                        break
                if quit_requested:
                    break
            else:
                count += 1
        print(f"Processed {processed} frame visits; {estimates} speed estimates.")
    finally:
        cap.release()
        if writer:
            writer.release()
        if stream:
            stream.close()
        if not args.no_show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
