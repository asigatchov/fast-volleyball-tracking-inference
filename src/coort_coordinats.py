import argparse
import json
from pathlib import Path

import cv2


BASE_ANNOTATION_WIDTH = 1280
BASE_ANNOTATION_HEIGHT = 720
INITIAL_KEYPOINTS = [
    {"id": 0, "name": "1_back_left", "x": 114, "y": 547},
    {"id": 1, "name": "2_back_left", "x": 480, "y": 421},
    {"id": 2, "name": "3_back_right", "x": 731, "y": 433},
    {"id": 3, "name": "4_back_right", "x": 1122, "y": 559},
    {"id": 4, "name": "5_center_left", "x": 391, "y": 452},
    {"id": 5, "name": "6_center_right", "x": 823, "y": 461},
    {"id": 6, "name": "7_net_left", "x": 404, "y": 339},
    {"id": 7, "name": "8_net_right", "x": 815, "y": 346},
]
SKELETON_CONNECTIONS = [
    (0, 4),
    (4, 1),
    (1, 2),
    (2, 5),
    (5, 3),
    (3, 0),
    (4, 5),
    (4, 6),
    (6, 7),
    (7, 5),
]


def scale_initial_keypoints(frame_width, frame_height):
    scaled = []
    for point in INITIAL_KEYPOINTS:
        scaled.append(
            {
                "id": point["id"],
                "name": point["name"],
                "x": int(round(point["x"] / BASE_ANNOTATION_WIDTH * frame_width)),
                "y": int(round(point["y"] / BASE_ANNOTATION_HEIGHT * frame_height)),
                "visible": True,
            }
        )
    return scaled


def make_output_path(video_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{video_path.stem}_coort.json"


def load_existing_annotation(output_path, frame_width, frame_height):
    if not output_path.exists():
        return scale_initial_keypoints(frame_width, frame_height), 0

    with output_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    saved_frame = int(data.get("frame_index", 0))
    points_by_name = {
        point["name"]: point
        for point in data.get("keypoints", [])
        if isinstance(point, dict) and "name" in point
    }

    keypoints = []
    for point in scale_initial_keypoints(frame_width, frame_height):
        saved_point = points_by_name.get(point["name"])
        if saved_point is not None:
            x = saved_point.get("x")
            y = saved_point.get("y")
            visible = bool(saved_point.get("visible", x is not None and y is not None))
            point["x"] = None if x is None else int(x)
            point["y"] = None if y is None else int(y)
            point["visible"] = visible and point["x"] is not None and point["y"] is not None
        keypoints.append(point)

    return keypoints, saved_frame


def save_annotation(output_path, video_path, frame_index, frame_width, frame_height, keypoints):
    payload = {
        "video_path": str(video_path),
        "frame_index": int(frame_index),
        "frame_width": int(frame_width),
        "frame_height": int(frame_height),
        "keypoints": [
            {
                "id": point["id"],
                "name": point["name"],
                "x": point["x"],
                "y": point["y"],
                "visible": bool(point["visible"]),
            }
            for point in keypoints
        ],
    }
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def read_frame(cap, frame_index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    return ok, frame


def clamp_frame_index(frame_index, frame_count):
    return max(0, min(frame_count - 1, frame_index))


def draw_dashed_line(image, start_point, end_point, color, thickness=2, dash_length=12, gap_length=8):
    x1, y1 = start_point
    x2, y2 = end_point
    dx = x2 - x1
    dy = y2 - y1
    distance = int((dx * dx + dy * dy) ** 0.5)
    if distance == 0:
        return

    step = max(1, dash_length + gap_length)
    for offset in range(0, distance, step):
        start_ratio = offset / distance
        end_ratio = min(offset + dash_length, distance) / distance
        sx = int(round(x1 + dx * start_ratio))
        sy = int(round(y1 + dy * start_ratio))
        ex = int(round(x1 + dx * end_ratio))
        ey = int(round(y1 + dy * end_ratio))
        cv2.line(image, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)


def draw_ui(frame, keypoints, current_idx, frame_index, frame_count, output_path):
    canvas = frame.copy()

    for start_idx, end_idx in SKELETON_CONNECTIONS:
        start_point = keypoints[start_idx]
        end_point = keypoints[end_idx]
        if (
            start_point["x"] is None
            or start_point["y"] is None
            or end_point["x"] is None
            or end_point["y"] is None
        ):
            continue
        draw_dashed_line(
            canvas,
            (int(start_point["x"]), int(start_point["y"])),
            (int(end_point["x"]), int(end_point["y"])),
            (0, 0, 255),
            thickness=2,
        )

    for idx, point in enumerate(keypoints):
        x = point["x"]
        y = point["y"]
        if x is None or y is None:
            continue
        is_current = idx == current_idx
        color = (0, 255, 255) if is_current else (0, 200, 0)
        radius = 8 if is_current else 5
        cv2.circle(canvas, (int(x), int(y)), radius, color, -1)
        cv2.putText(
            canvas,
            f"{idx + 1}:{point['name']}",
            (int(x) + 10, int(y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    current_point = keypoints[current_idx]
    current_x = "-" if current_point["x"] is None else str(int(current_point["x"]))
    current_y = "-" if current_point["y"] is None else str(int(current_point["y"]))

    help_lines = [
        "Left click: set current point",
        "Middle click: clear current point",
        "n/p: next/prev point",
        "d/a: frame +1/-1",
        "w/s: frame +15/-15",
        "f: toggle fullscreen",
        "Ctrl+S: save",
        "q: save and exit",
        f"Frame: {frame_index + 1}/{frame_count}",
        f"Point: {current_idx + 1}/{len(keypoints)} {current_point['name']}",
        f"Coord: ({current_x}, {current_y})",
        f"Output: {output_path.name}",
    ]

    y_offset = 28
    for line in help_lines:
        cv2.putText(
            canvas,
            line,
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y_offset += 28

    cv2.imshow("Court Keypoint Annotation", canvas)


def annotate_video(video_path, output_dir):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_count <= 0 or frame_width <= 0 or frame_height <= 0:
        cap.release()
        raise RuntimeError("Не удалось получить параметры видео")

    output_path = make_output_path(video_path, output_dir)
    keypoints, frame_index = load_existing_annotation(output_path, frame_width, frame_height)
    frame_index = clamp_frame_index(frame_index, frame_count)

    window_name = "Court Keypoint Annotation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(frame_width, 1600), min(frame_height, 900))
    state = {"current_idx": 0}
    is_fullscreen = False

    def mouse_callback(event, x, y, flags, param):
        current_point = keypoints[state["current_idx"]]
        if event == cv2.EVENT_LBUTTONDOWN:
            current_point["x"] = int(x)
            current_point["y"] = int(y)
            current_point["visible"] = True
        elif event == cv2.EVENT_MBUTTONDOWN:
            current_point["x"] = None
            current_point["y"] = None
            current_point["visible"] = False

    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        ok, frame = read_frame(cap, frame_index)
        if not ok:
            break

        draw_ui(frame, keypoints, state["current_idx"], frame_index, frame_count, output_path)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("q"):
            save_annotation(output_path, video_path, frame_index, frame_width, frame_height, keypoints)
            break
        if key == 19:
            save_annotation(output_path, video_path, frame_index, frame_width, frame_height, keypoints)
            print(f"Saved annotation to {output_path}")
            continue
        if key == ord("n"):
            state["current_idx"] = min(len(keypoints) - 1, state["current_idx"] + 1)
        elif key == ord("p"):
            state["current_idx"] = max(0, state["current_idx"] - 1)
        elif key == ord("d"):
            frame_index = clamp_frame_index(frame_index + 1, frame_count)
        elif key == ord("a"):
            frame_index = clamp_frame_index(frame_index - 1, frame_count)
        elif key == ord("w"):
            frame_index = clamp_frame_index(frame_index + 15, frame_count)
        elif key == ord("s"):
            frame_index = clamp_frame_index(frame_index - 15, frame_count)
        elif key == ord("f"):
            is_fullscreen = not is_fullscreen
            cv2.setWindowProperty(
                window_name,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL,
            )

    cap.release()
    cv2.destroyAllWindows()
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Manual court keypoint annotation for a video frame.")
    parser.add_argument("video_path", type=Path, help="Path to the video file")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("."),
        help="Directory where <filename>_coort.json will be saved",
    )
    args = parser.parse_args()

    output_path = annotate_video(args.video_path, args.output_dir)
    print(f"Saved annotation to {output_path}")


if __name__ == "__main__":
    main()
