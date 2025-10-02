import cv2
import numpy as np
import argparse
import json
import os

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

def main(video_path, cache_file='court_points_cache.json'):
    # Загрузка видео и схемы площадки
    cap = cv2.VideoCapture(video_path)
    court_image = cv2.imread('images/court.jpg')
    if cap.isOpened() and court_image is not None:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка чтения первого кадра")
            return
        
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
                
            # Отображение схемы в правом верхнем углу
            frame[0:200, -100:] = small_court
            
            # Если выбрана точка, преобразуем и отображаем
            if current_point:
                px = (matrix[0][0]*current_point[0] + matrix[0][1]*current_point[1] + matrix[0][2]) / \
                     (matrix[2][0]*current_point[0] + matrix[2][1]*current_point[1] + matrix[2][2])
                py = (matrix[1][0]*current_point[0] + matrix[1][1]*current_point[1] + matrix[1][2]) / \
                     (matrix[2][0]*current_point[0] + matrix[2][1]*current_point[1] + matrix[2][2])
                
                # Отрисовка точек
                cv2.circle(frame, current_point, 5, (0, 0, 255), -1)
                court_point = (int(px), int(py))
                small_court_point = (int(px * 100 / court_image.shape[1]), int(py * 200 / court_image.shape[0]))
                cv2.circle(frame, (small_court_point[0] + frame.shape[1] - 100, small_court_point[1]), 3, (0, 0, 255), -1)
            
            cv2.imshow("Video", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Ошибка загрузки видео или изображения площадки")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Volleyball court point transformation with caching')
    parser.add_argument('--video_path', type=str, required=True, help='Path to the video file')
    args = parser.parse_args()
    main(args.video_path)
