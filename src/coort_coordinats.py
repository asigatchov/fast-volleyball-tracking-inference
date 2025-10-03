import cv2
import numpy as np
import argparse
import json
import os
from pathlib import Path

from player_tracker import PlayerTracker

# Add MediaPipe imports
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not available. Pose estimation will be disabled.")
    mp = None


def select_points(image, title, num_points=4):
    points = []
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < num_points:
            points.append([x, y])
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow(title, image)
    
    cv2.imshow(title, image)
    cv2.setMouseCallback(title, mouse_callback)
    while len(points) < num_points:
        cv2.waitKey(1)
    cv2.destroyWindow(title)
    return np.float32(points)


def get_cache_filename(video_path):
    from pathlib import Path
    video_name = Path(video_path).stem
    return f"coord_cache_{video_name}.json"


def load_cached_points(cache_file):
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            data = json.load(f)
            return np.float32(data['court_points'])
    return None


def save_cached_points(cache_file, points):
    with open(cache_file, 'w') as f:
        json.dump({'court_points': points.tolist()}, f)


def get_video_points_cache_filename(video_path):
    video_name = Path(video_path).stem
    return f"video_points_cache_{video_name}.json"


def load_video_points_cache(cache_file):
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            data = json.load(f)
            return np.float32(data['video_points'])
    return None


def save_video_points_cache(cache_file, points):
    with open(cache_file, 'w') as f:
        json.dump({'video_points': points.tolist()}, f)


def detect_direction_changes(ball_positions, threshold=0.7):
    frames = sorted(ball_positions.keys())
    direction_changes = {}

    if len(frames) < 3:
        return direction_changes

    prev_vec = None
    for i in range(1, len(frames)):
        curr_frame = frames[i]
        prev_frame = frames[i - 1]

        curr_pos = np.array(ball_positions[curr_frame])
        prev_pos = np.array(ball_positions[prev_frame])
        vec = curr_pos - prev_pos

        if prev_vec is not None:
            norm_prev = np.linalg.norm(prev_vec)
            norm_curr = np.linalg.norm(vec)
            if norm_prev == 0 or norm_curr == 0:
                prev_vec = vec
                continue

            cos_angle = np.dot(prev_vec, vec) / (norm_prev * norm_curr)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)

            if angle > threshold:
                direction_changes[curr_frame-3] = curr_pos.tolist()
                print(curr_frame, "Direction change detected at position:", curr_pos.tolist())

        prev_vec = vec

    return direction_changes


def build_frame_to_box_info(direction_changes, window=10):
    """
    Возвращает словарь: frame_num -> {'center': (cx, cy), 'offset': int}
    где offset = frame_num - event_frame (от -20 до +20)
    """
    frame_info = {}
    for event_frame, center in direction_changes.items():
        for offset in range(-window, window + 1):
            show_frame = event_frame + offset
            # Если кадр уже занят другим событием — можно оставить последнее или пропустить
            # Здесь оставляем последнее (перезапись)
            frame_info[show_frame] = {
                'center': center,
                'offset': offset  # отрицательный = до события, >=0 = в/после
            }

    print(f"Detected {len(direction_changes)} direction changes, total frames with boxes: {len(frame_info)}")
    return frame_info


# Global variables for MediaPipe pose estimation
mp_pose = None
mp_drawing = None
mp_drawing_styles = None
pose = None


def initialize_mediapipe():
    """Initialize MediaPipe components for pose estimation."""
    global mp_pose, mp_drawing, mp_drawing_styles, pose
    
    if not MEDIAPIPE_AVAILABLE or mp is None:
        return False
        
    try:
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        pose = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        return True
    except Exception as e:
        print(f"Error initializing MediaPipe: {e}")
        return False


def process_pose_estimation(frame, x1, y1, x2, y2):
    """
    Process pose estimation for a crop region using MediaPipe.
    
    Args:
        frame: Input frame
        x1, y1, x2, y2: Crop region coordinates
        
    Returns:
        Frame with pose landmarks drawn (if MediaPipe is available)
    """
    global pose, mp_pose, mp_drawing, mp_drawing_styles
    
    if not MEDIAPIPE_AVAILABLE or pose is None:
        return frame
    
    # Extract crop region
    crop = frame[y1:y2, x1:x2]
    
    # Convert BGR to RGB for MediaPipe
    rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    results = pose.process(rgb_crop)
    
    # Draw pose landmarks on the crop
    if results.pose_landmarks and mp_drawing and mp_pose:
        mp_drawing.draw_landmarks(
            crop, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
    # Place the processed crop back into the frame
    frame[y1:y2, x1:x2] = crop
    
    return frame


def main(video_path, track_json_path=None):
    # Initialize MediaPipe if available
    if MEDIAPIPE_AVAILABLE:
        if initialize_mediapipe():
            print("MediaPipe pose estimation initialized successfully")
        else:
            print("Failed to initialize MediaPipe pose estimation")
    
    cache_file = get_cache_filename(video_path)
    video_points_cache_file = get_video_points_cache_filename(video_path)

    ball_positions = {}
    start_frame = 0
    last_frame = -1
    if track_json_path:
        with open(track_json_path, 'r') as f:
            track_data = json.load(f)
            start_frame = track_data.get('start_frame', 0)
            last_frame = track_data.get('last_frame', -1)
            for pos_data in track_data.get('positions', []):
                if len(pos_data) == 2:
                    coords, frame_num = pos_data
                    ball_positions[frame_num] = coords

    direction_changes = detect_direction_changes(ball_positions, threshold=0.9)
    frame_to_box = build_frame_to_box_info(direction_changes, window=8)

    cap = cv2.VideoCapture(video_path)
    court_image = cv2.imread('images/court.jpg')
    if not (cap.isOpened() and court_image is not None):
        print("Ошибка загрузки видео или изображения площадки")
        return

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    ret, frame = cap.read()
    if not ret:
        print("Ошибка чтения первого кадра")
        return

    small_court = cv2.resize(court_image, (100, 200))

    # --- Кеширование разметки video_points ---
    video_points = load_video_points_cache(video_points_cache_file)
    if video_points is None:
        frame_copy = frame.copy()
        video_points = select_points(frame_copy, "Select 4 points on video (bottom-left, top-left, top-right, bottom-right)")
        save_video_points_cache(video_points_cache_file, video_points)

    # --- Кеширование разметки court_points ---
    court_points = load_cached_points(cache_file)
    if court_points is None:
        court_copy = court_image.copy()
        court_points = select_points(court_copy, "Select 4 points on court (bottom-left, top-left, top-right, bottom-right)")
        save_cached_points(cache_file, court_points)

    matrix = cv2.getPerspectiveTransform(video_points, court_points)

    player_tracker = PlayerTracker(
        yolo_model_path="models_yolo/yolo11n.onnx",
        court_image_path="images/court.jpg"
    )
    player_tracker.set_transformation_matrix(video_points, court_points)
    player_tracker.set_court_boundary(court_points)

    current_point = None
    def mouse_callback(event, x, y, flags, param):
        nonlocal current_point
        if event == cv2.EVENT_LBUTTONDOWN:
            current_point = (x, y)

    cv2.namedWindow("Video")
    cv2.setMouseCallback("Video", mouse_callback)

    frame_count = start_frame

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if last_frame != -1 and frame_count > last_frame:
            break

        player_boxes = player_tracker.detect_players(frame)
        output_frame = player_tracker.draw_player_positions(frame, player_boxes)

        h, w = output_frame.shape[:2]

        # === Отрисовка сдвинутой рамки с цветом в зависимости от времени ===
        pause_on_this_frame = False
        if frame_count in frame_to_box:
            cx, cy = frame_to_box[frame_count]['center']
            offset = frame_to_box[frame_count]['offset']

            # Смещаем центр рамки вниз на 300 пикселей
            box_cx = cx
            box_cy = cy + 150

            box_size = 640
            x1 = int(box_cx - box_size // 2)
            y1 = int(box_cy - box_size // 2)
            x2 = x1 + box_size
            y2 = y1 + box_size

            # Clip to frame
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            # Цвет: синий до события, зелёный — в событии и после
            color = (255, 0, 0) if offset < 0 else (0, 255, 0)  # BGR
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)

            # Process pose estimation for the crop region when at/after direction change
            if offset >= 0 and MEDIAPIPE_AVAILABLE and pose is not None:
                output_frame = process_pose_estimation(output_frame, x1, y1, x2, y2)

            # Пауза на 0.5 сек при смене направления (offset == 0)
            if offset == 0:
                pause_on_this_frame = True

        # Отображение схемы
        output_frame[0:200, -100:] = small_court

        # Преобразование выбранной точки
        if current_point:
            px = (matrix[0][0]*current_point[0] + matrix[0][1]*current_point[1] + matrix[0][2]) / \
                 (matrix[2][0]*current_point[0] + matrix[2][1]*current_point[1] + matrix[2][2])
            py = (matrix[1][0]*current_point[0] + matrix[1][1]*current_point[1] + matrix[1][2]) / \
                 (matrix[2][0]*current_point[0] + matrix[2][1]*current_point[1] + matrix[2][2])

            cv2.circle(output_frame, current_point, 5, (0, 0, 255), -1)
            small_court_point = (
                int(px * 100 / court_image.shape[1]),
                int(py * 200 / court_image.shape[0])
            )
            cv2.circle(output_frame, (small_court_point[0] + output_frame.shape[1] - 100, small_court_point[1]), 3, (0, 0, 255), -1)

        cv2.imshow("Video", output_frame)

        if pause_on_this_frame:
            cv2.waitKey(int(500))  # 0.5 сек пауза

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Volleyball analysis with directional box (blue before, green at/after).')
    parser.add_argument('--video_path', type=str, required=True, help='Path to the video file')
    parser.add_argument('--track_json', type=str, help='Path to the ball track JSON file')
    args = parser.parse_args()
    main(args.video_path, args.track_json)