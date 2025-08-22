import pandas as pd
import numpy as np
import cv2
import os
import argparse
import math

# === Аргументы командной строки ===
parser = argparse.ArgumentParser(description="Воспроизведение игровых эпизодов с опцией вырезания через ffmpeg, записи в MP4 и сохранения CSV розыгрышей.")
parser.add_argument('--csv_path', type=str, required=True, help='Путь к ball.csv')
parser.add_argument('--video_path', type=str, required=True, help='Путь к видео')
parser.add_argument('--fps', type=float, default=30, help='FPS видео (по умолчанию 30)')
parser.add_argument('--playback_speed', type=float, default=1.0, help='Скорость воспроизведения')
parser.add_argument('--min_duration_sec', type=float, default=3.0, help='Минимальная длительность розыгрыша (сек)')
parser.add_argument('--ffmpeg', action='store_true', help='Вывести команды ffmpeg для вырезания эпизодов')
parser.add_argument('--output_path', type=str, default=None, help='Путь для записи выходного MP4-файла')
args = parser.parse_args()

CSV_PATH = args.csv_path
VIDEO_PATH = args.video_path
FPS = args.fps
PLAYBACK_SPEED = args.playback_speed
MIN_DURATION_SEC = args.min_duration_sec
FFMPEG = args.ffmpeg
OUTPUT_PATH = args.output_path
MAX_JUMP_PX = 200

# Проверка файлов
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV не найден: {CSV_PATH}")
if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(f"Видео не найдено: {VIDEO_PATH}")

# === 1. Загрузка и очистка данных ===
df = pd.read_csv(CSV_PATH)
df['Frame'] = pd.to_numeric(df['Frame'], errors='coerce')
df['Visibility'] = pd.to_numeric(df['Visibility'], errors='coerce')
df['X'] = pd.to_numeric(df['X'], errors='coerce')
df['Y'] = pd.to_numeric(df['Y'], errors='coerce')

# Замена -1 и 0 на NaN
df.loc[(df['X'] == -1) | (df['Visibility'] == 0), ['X', 'Y']] = np.nan

# Интерполяция коротких пропусков
MAX_GAP = 2
df['X_interp'] = df['X'].interpolate(limit=MAX_GAP, limit_direction='both')
df['Y_interp'] = df['Y'].interpolate(limit=MAX_GAP, limit_direction='both')
df['Valid'] = df['X_interp'].notna()

# === 2. Удаление резких прыжков >200px ===
def remove_jumps(x_vals, y_vals, max_jump=200):
    x_clean = x_vals.copy()
    y_clean = y_vals.copy()
    for i in range(1, len(x_vals)):
        if np.isnan(x_vals[i-1]) or np.isnan(x_vals[i]):
            continue
        dx = abs(x_vals[i] - x_vals[i-1])
        dy = abs(y_vals[i] - y_vals[i-1])
        if math.hypot(dx, dy) > max_jump:
            x_clean[i] = np.nan
            y_clean[i] = np.nan
    return x_clean, y_clean

df['segment'] = (~df['Valid']).cumsum()
df['X_clean'] = np.nan
df['Y_clean'] = np.nan

for seg_id, group in df.groupby('segment'):
    if not group['Valid'].any():
        continue
    x_temp = group['X_interp'].values
    y_temp = group['Y_interp'].values
    x_fixed, y_fixed = remove_jumps(x_temp, y_temp, MAX_JUMP_PX)
    df.loc[group.index, 'X_clean'] = x_fixed
    df.loc[group.index, 'Y_clean'] = y_fixed

# Финальные координаты
df['X_final'] = df['X_clean'].interpolate(limit=MAX_GAP, limit_direction='both')
df['Y_final'] = df['Y_clean'].interpolate(limit=MAX_GAP, limit_direction='both')
df['Tracked'] = df['X_final'].notna()

# === 3. Сегментация на розыгрыши ===
df['group'] = (df['Tracked'] != df['Tracked'].shift()).cumsum()
active_segments = df[df['Tracked']].groupby('group')

valid_episodes = []
video_center_x = 960
video_center_y = 540
frame_count = int(df['Frame'].max()) + 1 if not df['Frame'].isna().all() else 0

for name, segment in active_segments:
    frames = segment['Frame'].values
    x_vals = segment['X_final'].values
    y_vals = segment['Y_final'].values
    start_f = int(frames[0])
    end_f = int(frames[-1])
    duration_frames = len(frames)
    duration_sec = duration_frames / FPS

    # Пропуск: слишком короткий
    if duration_sec < MIN_DURATION_SEC:
        continue

    # Пропуск: почти нет движения
    total_movement = sum(
        math.hypot(x_vals[i] - x_vals[i-1], y_vals[i] - y_vals[i-1])
        for i in range(1, len(x_vals)) if not (np.isnan(x_vals[i]) or np.isnan(x_vals[i-1]))
    )
    if total_movement < 30 and duration_sec < 1.5:
        continue

    # Пропуск: внезапное появление в центре
    sudden_center = False
    for i in range(min(3, len(x_vals))):
        if not np.isnan(x_vals[i]) and not np.isnan(y_vals[i]):
            dist_to_center = math.hypot(x_vals[i] - video_center_x, y_vals[i] - video_center_y)
            if dist_to_center < 100:
                sudden_center = True
    if sudden_center and duration_sec < 2.0:
        continue

    # Проверка на резкие скачки
    has_big_jump = False
    for i in range(1, len(x_vals)):
        if np.isnan(x_vals[i]) or np.isnan(x_vals[i-1]):
            continue
        if math.hypot(x_vals[i] - x_vals[i-1], y_vals[i] - y_vals[i-1]) > MAX_JUMP_PX:
            has_big_jump = True
            break
    if has_big_jump:
        continue

    # Добавление 1 сек до и после
    extra_frames = int(FPS)
    new_start = max(0, start_f - extra_frames)
    new_end = min(frame_count - 1, end_f + extra_frames)
    new_frames = np.arange(new_start, new_end + 1)
    new_x_vals = [df.loc[df['Frame'] == f, 'X_final'].iloc[0] if f in df['Frame'].values else np.nan for f in new_frames]
    new_y_vals = [df.loc[df['Frame'] == f, 'Y_final'].iloc[0] if f in df['Frame'].values else np.nan for f in new_frames]
    new_duration_sec = (new_end - new_start + 1) / FPS

    valid_episodes.append({
        'start': new_start,
        'end': new_end,
        'frames': new_frames,
        'x_vals': new_x_vals,
        'y_vals': new_y_vals,
        'duration_sec': new_duration_sec
    })

# Сортировка и объединение пересекающихся розыгрышей
valid_episodes.sort(key=lambda x: x['start'])
merged_episodes = []
if valid_episodes:
    current_ep = valid_episodes[0]
    for next_ep in valid_episodes[1:]:
        if next_ep['start'] <= current_ep['end']:
            current_ep['end'] = max(current_ep['end'], next_ep['end'])
            new_frames = np.arange(current_ep['start'], current_ep['end'] + 1)
            new_x_vals = [df.loc[df['Frame'] == f, 'X_final'].iloc[0] if f in df['Frame'].values else np.nan for f in new_frames]
            new_y_vals = [df.loc[df['Frame'] == f, 'Y_final'].iloc[0] if f in df['Frame'].values else np.nan for f in new_frames]
            current_ep['frames'] = new_frames
            current_ep['x_vals'] = new_x_vals
            current_ep['y_vals'] = new_y_vals
            current_ep['duration_sec'] = (current_ep['end'] - current_ep['start'] + 1) / FPS
        else:
            merged_episodes.append(current_ep)
            current_ep = next_ep
    merged_episodes.append(current_ep)

valid_episodes = merged_episodes

# === 4. Сохранение CSV-файлов для розыгрышей ===
os.makedirs('csv_rally', exist_ok=True)
for i, ep in enumerate(valid_episodes):
    rally_data = pd.DataFrame({
        'Frame': ep['frames'],
        'X': ep['x_vals'],
        'Y': ep['y_vals'],
        'Tracked': [not np.isnan(x) for x in ep['x_vals']]
    })
    rally_data.to_csv(f"csv_rally/rally_{i+1:03d}.csv", index=False)

# === 5. Генерация команд ffmpeg (если указано) ===
if FFMPEG:
    os.makedirs('episods', exist_ok=True)
    print("=== Команды ffmpeg для вырезания эпизодов ===")
    for i, ep in enumerate(valid_episodes):
        start_time = ep['start'] / FPS
        duration = ep['duration_sec']
        hours = int(start_time // 3600)
        minutes = int((start_time % 3600) // 60)
        seconds = start_time % 60
        start_time_str = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
        output_file = f"episods/episod_{i+1:03d}.mp4"
        ffmpeg_cmd = f"ffmpeg -i \"{VIDEO_PATH}\" -ss {start_time_str} -t {duration:.2f} -c copy \"{output_file}\""
        print(ffmpeg_cmd)

# === 6. Открытие видео ===
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError("Не удалось открыть видео.")

video_fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_step = max(1, int(video_fps / FPS))

print("\n=== Воспроизведение игровых эпизодов ===")
print(f"Минимальная длительность розыгрыша: {MIN_DURATION_SEC} сек")
print(f"Найдено валидных розыгрышей: {len(valid_episodes)}")
for i, ep in enumerate(valid_episodes):
    print(f"  Эпизод {i+1}: {ep['start']}–{ep['end']} кадры ({ep['duration_sec']:.2f} с)")
if len(valid_episodes) == 0:
    print("  Нет розыгрышей, удовлетворяющих критериям.")
print("Нажмите 'q' для выхода, 'n' для перехода к следующему розыгрышу.")

# === 7. Настройка записи видео (если указано) ===
writer = None
if OUTPUT_PATH:
    os.makedirs(os.path.dirname(OUTPUT_PATH) or '.', exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (width, height))

# === 8. Воспроизведение ===
current_video_frame = 0
current_episode = None
episode_index = 0
font = cv2.FONT_HERSHEY_SIMPLEX

if valid_episodes:
    cap.set(cv2.CAP_PROP_POS_FRAMES, valid_episodes[0]['start'] - 1)
    current_video_frame = valid_episodes[0]['start'] - 1
    current_episode = valid_episodes[0]
    episode_index = 0
    print(f"▶️ Начало розыгрыша 1 (кадр {valid_episodes[0]['start']})")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if current_video_frame % frame_step != 0:
        current_video_frame += 1
        continue

    csv_frame_idx = current_video_frame // frame_step

    active_ep = None
    if episode_index < len(valid_episodes):
        ep = valid_episodes[episode_index]
        if ep['start'] <= csv_frame_idx <= ep['end']:
            active_ep = ep

    if active_ep:
        if current_episode != active_ep:
            current_episode = active_ep
            print(f"▶️ Начало розыгрыша {episode_index + 1} (кадр {csv_frame_idx})")

        try:
            local_idx = list(active_ep['frames']).index(csv_frame_idx)
            x_meas = active_ep['x_vals'][local_idx]
            y_meas = active_ep['y_vals'][local_idx]
            has_measurement = True
        except:
            has_measurement = False

        if writer:
            writer.write(frame)

        if not OUTPUT_PATH:
            ep_idx = episode_index + 1
            elapsed_sec = (csv_frame_idx - active_ep['start']) / FPS

            cv2.putText(frame, f"Episode: {ep_idx}", (20, 50), font, 1.2, (0, 255, 0), 2)
            cv2.putText(frame, f"Time: {elapsed_sec:.2f}s", (20, 90), font, 1.0, (255, 255, 0), 2)

            if has_measurement and 0 <= x_meas < width and 0 <= y_meas < height:
                cv2.circle(frame, (int(x_meas), int(y_meas)), 4, (0, 255, 0), -1)

            if csv_frame_idx == active_ep['end']:
                if has_measurement and 0 <= x_meas < width and 0 <= y_meas < height:
                    x1, y1 = max(0, int(x_meas) - 20), max(0, int(y_meas) - 20)
                    x2, y2 = min(width, int(x_meas) + 20), min(height, int(y_meas) + 20)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, "END", (int(x_meas) + 25, int(y_meas) - 10),
                                font, 0.8, (0, 0, 255), 2)

            cv2.imshow('Gameplay Only', frame)

        if csv_frame_idx == active_ep['end']:
            episode_index += 1
            if episode_index < len(valid_episodes):
                next_ep = valid_episodes[episode_index]
                cap.set(cv2.CAP_PROP_POS_FRAMES, next_ep['start'] - 1)
                current_video_frame = next_ep['start'] - 1
                current_episode = None
                print(f"Переход к розыгрышу {episode_index + 1} (кадр {next_ep['start']})")
            else:
                print("Все розыгрыши завершены.")
                break
    else:
        if episode_index < len(valid_episodes):
            next_ep = valid_episodes[episode_index]
            cap.set(cv2.CAP_PROP_POS_FRAMES, next_ep['start'] - 1)
            current_video_frame = next_ep['start'] - 1
            current_episode = None
            print(f"Переход к розыгрышу {episode_index + 1} (кадр {next_ep['start']})")
        else:
            break

    if not OUTPUT_PATH:
        delay = int(10 / PLAYBACK_SPEED)
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n') and episode_index < len(valid_episodes) - 1:
            episode_index += 1
            next_ep = valid_episodes[episode_index]
            cap.set(cv2.CAP_PROP_POS_FRAMES, next_ep['start'] - 1)
            current_video_frame = next_ep['start'] - 1
            current_episode = None
            print(f"Переход к розыгрышу {episode_index + 1} (кадр {next_ep['start']})")

    current_video_frame += 1

# === 9. Завершение ===
cap.release()
if writer:
    writer.release()
cv2.destroyAllWindows()
print("Воспроизведение завершено.")