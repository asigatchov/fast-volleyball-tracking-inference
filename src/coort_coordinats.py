import cv2
import numpy as np
import argparse
import json
import os

# Import our new player tracker module
from player_tracker import PlayerTracker


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

def load_cached_points(cache_file):
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            data = json.load(f)
            return np.float32(data['court_points'])
    return None

def save_cached_points(cache_file, points):
    with open(cache_file, 'w') as f:
        json.dump({'court_points': points.tolist()}, f)

def main(video_path, cache_file='court_points_cache.json', output_path=None):
    # Загрузка видео и схемы площадки
    cap = cv2.VideoCapture(video_path)
    court_image = cv2.imread('images/court.jpg')
    if cap.isOpened() and court_image is not None:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка чтения первого кадра")
            return

        # Prepare VideoWriter if output_path is specified
        video_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            height, width = frame.shape[:2]
            video_writer = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))

        # Масштабирование схемы для отображения
        small_court = cv2.resize(court_image, (100, 200))

        # Выбор точек на видео
        frame_copy = frame.copy()
        video_points = select_points(frame_copy, "Select 4 points on video (bottom-left, top-left, top-right, bottom-right)")
        
        # Проверка кеша для точек схемы
        court_points = load_cached_points(cache_file)
        if court_points is None:
            # Если кеша нет, запрашиваем разметку
            court_copy = court_image.copy()
            court_points = select_points(court_copy, "Select 4 points on court (bottom-left, top-left, top-right, bottom-right)")
            # Сохраняем точки в кеш
            save_cached_points(cache_file, court_points)
        
        # Создание матрицы преобразования
        matrix = cv2.getPerspectiveTransform(video_points, court_points)
        
        # Initialize player tracker with YOLO model
        player_tracker = PlayerTracker(
            yolo_model_path="models_yolo/yolo11n.onnx",
            court_image_path="images/court.jpg"
        )
        # Set transformation matrix for player tracker
        player_tracker.set_transformation_matrix(video_points, court_points)
        
        # Основной цикл обработки видео
        current_point = None
        def mouse_callback(event, x, y, flags, param):
            nonlocal current_point
            if event == cv2.EVENT_LBUTTONDOWN:
                current_point = (x, y)
        
        cv2.namedWindow("Video")
        cv2.setMouseCallback("Video", mouse_callback)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Detect players using YOLO model
            player_boxes = player_tracker.detect_players(frame)
            
            # Draw player positions on frame
            output_frame = player_tracker.draw_player_positions(frame, player_boxes)
            
            # Отображение схемы в правом верхнем углу
            output_frame[0:200, -100:] = small_court

            # Draw player positions on the small court schematic
            for bbox in player_boxes:
                # Get player position (center of bottom edge)
                player_pos = player_tracker.get_player_position(bbox)
                # Transform to court coordinates
                court_pos = player_tracker.transform_to_court(player_pos)
                if court_pos:
                    # Scale to small_court size
                    small_x = int(court_pos[0] * 100 / court_image.shape[1])
                    small_y = int(court_pos[1] * 200 / court_image.shape[0])
                    # Draw on output_frame (top-right corner)
                    cv2.circle(output_frame, (small_x + output_frame.shape[1] - 100, small_y), 4, (255, 0, 0), -1)

            # Если выбрана точка, преобразуем и отображаем
            if current_point:
                px = (matrix[0][0]*current_point[0] + matrix[0][1]*current_point[1] + matrix[0][2]) / \
                     (matrix[2][0]*current_point[0] + matrix[2][1]*current_point[1] + matrix[2][2])
                py = (matrix[1][0]*current_point[0] + matrix[1][1]*current_point[1] + matrix[1][2]) / \
                     (matrix[2][0]*current_point[0] + matrix[2][1]*current_point[1] + matrix[2][2])
                
                # Отрисовка точек
                cv2.circle(output_frame, current_point, 5, (0, 0, 255), -1)
                court_point = (int(px), int(py))
                small_court_point = (int(px * 100 / court_image.shape[1]), int(py * 200 / court_image.shape[0]))
                cv2.circle(output_frame, (small_court_point[0] + output_frame.shape[1] - 100, small_court_point[1]), 3, (0, 0, 255), -1)
            
            cv2.imshow("Video", output_frame)

            # Save frame to output video if enabled
            if video_writer:
                video_writer.write(output_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
    else:
        print("Ошибка загрузки видео или изображения площадки")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Volleyball court point transformation with caching')
    parser.add_argument('--video_path', type=str, required=True, help='Path to the video file')
    parser.add_argument('--output_path', type=str, default=None, help='Path to save the output video')
    args = parser.parse_args()
    main(args.video_path, output_path=args.output_path)