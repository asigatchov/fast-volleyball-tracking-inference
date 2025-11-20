# Fast Volleyball Tracking Inference

High-speed volleyball ball detection and tracking using an optimized ONNX model, achieving **200 FPS** on an Intel Core i5-10400F CPU @ 2.90GHz. This repository provides scripts for real-time inference, outputting ball coordinates to CSV and optional visualized video output.

![Demo](https://raw.githubusercontent.com/asigatchov/fast-volleyball-tracking-inference/refs/heads/master/examples/output.gif)
## Features
- **High Performance**: 200 FPS on modest CPU hardware (Intel Core i5-10400F @ 2.90GHz).
- **Optimized for CPU**: Lightweight ONNX model designed for grayscale video input.
- **4-Step Processing Pipeline**: Detection → Track calculation → Video assembly → Vertical reels.
- **Flexible Output**: Saves ball coordinates to CSV, track JSONs, combined videos, and vertical clips.
- **Customizable**: Adjustable track parameters for analysis and visualization.
- **Pose Detection**: Detect player poses when ball is near using MediaPipe.
- **Direction Change Analysis**: Identify and visualize ball direction changes with pose detection.
- **Easy to Use**: Simple command-line interface with clear options.


[For training used - vball-net](https://github.com/asigatchov/vball-net)

## Installation

### Prerequisites
- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) for dependency management
- Input video file (e.g., `.mp4`)
- Pre-trained ONNX model weights (download link provided below)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/asigatchov/fast-volleyball-tracking-inference.git
   cd fast-volleyball-tracking-inference
   ```

2. Install dependencies using `uv`:
   ```bash
   uv sync
   ```

3. Download the pre-trained ONNX model weights:
   - [Download model.onnx](#) <!-- Replace with actual link to weights, e.g., Google Drive or GitHub Releases -->

## Usage

### 4-Step Processing Pipeline

This repository provides a complete workflow for volleyball ball tracking and video processing:

#### Step 1: Ball Detection (CSV output)

Detect volleyball ball in video and save coordinates to CSV:

```bash
uv run src/inference_onnx_seq9_gray_v2.py \
  --video_path examples/beach_st_lenina_20250622_g1_005.mp4 \
  --model_path models/VballNetFastV1_seq9_grayscale_233_h288_w512.onnx \
  --output_dir output \
  --only_csv
```

**Output:** `output/beach_st_lenina_20250622_g1_005/ball.csv`

**With visualization video:**
```bash
uv run src/inference_onnx_seq9_gray_v2.py \
  --video_path examples/beach_st_lenina_20250622_g1_005.mp4 \
  --model_path models/VballNetFastV1_seq9_grayscale_233_h288_w512.onnx \
  --output_dir output
```

**Output:** `output/beach_st_lenina_20250622_g1_005/ball.csv` + `output/beach_st_lenina_20250622_g1_005/predict.mp4`

**Real-time visualization (no file output):**
```bash
uv run src/inference_onnx_seq9_gray_v2.py \
  --video_path examples/beach_st_lenina_20250622_g1_005.mp4 \
  --model_path models/VballNetFastV1_seq9_grayscale_233_h288_w512.onnx \
  --visualize
```

#### Step 2: Track Calculation (JSON tracks)

Calculate tracks from ball detections and save as JSON files:

```bash
uv run src/track_calculator.py \
  --csv_path output/beach_st_lenina_20250622_g1_005/ball.csv \
  --output_dir output \
  --fps 30
```

**Output:** `output/beach_st_lenina_20250622_g1_005/tracks/track_*.json`

#### Step 3: Video Assembly (skip game pauses)

Create combined video or individual clips from tracks:

**Combined video (all tracks concatenated):**
```bash
uv run src/track_processor.py \
  --video_path examples/beach_st_lenina_20250622_g1_005.mp4 \
  --output_dir output
```

**Output:** `output/beach_st_lenina_20250622_g1_005/combined.mp4`

**Individual clip per track:**
```bash
uv run src/track_processor.py \
  --video_path examples/beach_st_lenina_20250622_g1_005.mp4 \
  --json_dir output/beach_st_lenina_20250622_g1_005/tracks \
  --split_dir output/beach_st_lenina_20250622_g1_005/clips
```

**Output:** `output/beach_st_lenina_20250622_g1_005/clips/track_*.mp4`

#### Step 4: Vertical Reels (9:16 format)

Create vertical video clips (reels) with ball-centered cropping:

![Reel Demo](examples/reel_g.mp4)

**Single track reel:**
```bash
uv run src/make_reels.py \
  --video_path examples/beach_st_lenina_20250622_g1_005.mp4 \
  --track_json output/beach_st_lenina_20250622_g1_005/tracks/track_0001.json \
  --output_dir output
```

**Output:** `output/beach_st_lenina_20250622_g1_005/reels/reel_beach_st_lenina_20250622_g1_005_0001.mp4`

**All tracks as reels:**
```bash
uv run src/make_reels.py \
  --video_path examples/beach_st_lenina_20250622_g1_005.mp4 \
  --json_dir output/beach_st_lenina_20250622_g1_005/tracks \
  --output_dir output
```

**Output:** `output/beach_st_lenina_20250622_g1_005/reels/reel_*.mp4` (one per track)

### Complete Pipeline Example

Run all steps sequentially:

```bash
# Step 1: Detect ball
uv run src/inference_onnx_seq9_gray_v2.py \
  --video_path examples/beach_st_lenina_20250622_g1_005.mp4 \
  --model_path models/VballNetFastV1_seq9_grayscale_233_h288_w512.onnx \
  --output_dir output \
  --only_csv

# Step 2: Calculate tracks
uv run src/track_calculator.py \
  --csv_path output/beach_st_lenina_20250622_g1_005/ball.csv \
  --output_dir output

# Step 3: Create combined video
uv run src/track_processor.py \
  --video_path examples/beach_st_lenina_20250622_g1_005.mp4 \
  --output_dir output

# Step 4: Generate vertical reels
uv run src/make_reels.py \
  --video_path examples/beach_st_lenina_20250622_g1_005.mp4 \
  --json_dir output/beach_st_lenina_20250622_g1_005/tracks \
  --output_dir output
```

**Final output structure:**
```
output/beach_st_lenina_20250622_g1_005/
├── ball.csv                  # Raw detections
├── predict.mp4               # Detection visualization (optional)
├── tracks/
│   ├── track_0001.json
│   ├── track_0002.json
│   └── ...
├── combined.mp4              # All tracks concatenated
├── clips/                    # Individual track clips (optional)
│   ├── track_0001.mp4
│   └── ...
└── reels/                    # Vertical 9:16 crops
    ├── reel_beach_st_lenina_20250622_g1_005_0001.mp4
    └── ...
```

### Pose Detection

Run pose detection on existing track files:

```bash
uv run main.py --mode pose --track_file track_json/track_0037.json --video_path examples/beach_st_lenina_20250622_g1_005.mp4 --visualize
```

### Direction Change Pose Detection

Detect ball direction changes and visualize poses around these points:

```bash
uv run test_direction_change_pose.py --video_path examples/beach_st_lenina_20250622_g1_005.mp4 --track_json track_json/track_0037.json --visualize
```

For more details, see [DIRECTION_CHANGE_POSE_DETECTION.md](DIRECTION_CHANGE_POSE_DETECTION.md)

### Command-Line Options

#### Step 1: Ball Detection
```
usage: inference_onnx_seq9_gray_v2.py [-h] --video_path VIDEO_PATH [--track_length TRACK_LENGTH]
                                      [--output_dir OUTPUT_DIR] --model_path MODEL_PATH
                                      [--visualize] [--only_csv]

options:
  -h, --help            show this help message and exit
  --video_path VIDEO_PATH
                        Path to input video file
  --track_length TRACK_LENGTH
                        Length of the ball track (default: 8)
  --output_dir OUTPUT_DIR
                        Root output directory (default: None)
  --model_path MODEL_PATH
                        Path to ONNX model file
  --visualize           Enable real-time visualization on display
  --only_csv            Save only CSV, skip video output
```

#### Step 2: Track Calculation
```
usage: track_calculator.py [-h] --csv_path CSV_PATH [--output_dir OUTPUT_DIR] [--fps FPS]
                           [--max_distance MAX_DISTANCE] [--min_duration_sec MIN_DURATION_SEC]

options:
  -h, --help            show this help message and exit
  --csv_path CSV_PATH   Path to ball.csv file
  --output_dir OUTPUT_DIR
                        Root output directory (default: output)
  --fps FPS             Frames per second (default: 30.0)
  --max_distance MAX_DISTANCE
                        Max tracking distance (default: 200.0)
  --min_duration_sec MIN_DURATION_SEC
                        Minimum track duration in seconds (default: 1.0)
```

#### Step 3: Video Assembly
```
usage: track_processor.py [-h] [--json_dir JSON_DIR] --video_path VIDEO_PATH
                          [--output_path OUTPUT_PATH] [--split_dir SPLIT_DIR]
                          [--output_dir OUTPUT_DIR] [--fps FPS]

options:
  -h, --help            show this help message and exit
  --json_dir JSON_DIR   Directory with track_*.json files
  --video_path VIDEO_PATH
                        Path to source video
  --output_path OUTPUT_PATH
                        Combined output video path (AVI)
  --split_dir SPLIT_DIR
                        Directory for individual track videos (MP4)
  --output_dir OUTPUT_DIR
                        Root output directory (auto-derives paths)
  --fps FPS             Output FPS if video has none (default: 30.0)
```

#### Step 4: Vertical Reels
```
usage: make_reels.py [-h] --video_path VIDEO_PATH [--track_json TRACK_JSON]
                     [--track_jsons TRACK_JSONS [TRACK_JSONS ...]]
                     [--json_dir JSON_DIR] [--output_dir OUTPUT_DIR] [--visualize]

options:
  -h, --help            show this help message and exit
  --video_path VIDEO_PATH
                        Path to video file
  --track_json TRACK_JSON
                        Path to a single track JSON file
  --track_jsons TRACK_JSONS [TRACK_JSONS ...]
                        Paths to multiple track JSON files
  --json_dir JSON_DIR   Directory with track_*.json files
  --output_dir OUTPUT_DIR
                        Root output directory
  --visualize           Show real-time cropped video
```

### Example
```bash
uv run main.py --mode pose --track_file track_json/track_0037.json --video_path examples/beach_st_lenina_20250622_g1_005.mp4 --visualize
```

This command processes the track file `track_0037.json` with the video, detects player poses when the ball is near, and displays visualization with both ball position and player poses.

## Output

### Step 1: Ball Detection CSV
```csv
Frame,Visibility,X,Y
0,0,-1,-1
1,1,1068,536
2,1,1068,532
3,1,1068,525
...
1008,1,1065,487
```

### Step 2: Track JSON
```json
{
  "track_id": 1,
  "start_frame": 100,
  "last_frame": 250,
  "positions": [
    [[1068, 536], 100],
    [[1068, 532], 101],
    ...
  ]
}
```

### Step 3: Video Outputs
- **combined.mp4**: All tracks concatenated with fade transitions
- **clips/track_*.mp4**: Individual rally clips (optional)

### Step 4: Vertical Reels
- **reels/reel_*.mp4**: 9:16 vertical videos with ball-centered cropping
- Example: [examples/reel_g.mp4](examples/reel_g.mp4)

## Repository Structure
```
fast-volleyball-tracking-inference/
├── examples
│   ├── beach_st_lenina_20250622_g1_005.mp4
│   ├── beach_st_lenina_20250622_g1_005_predict_ball.csv
│   ├── gtu_20250316_002.mp4
│   ├── gtu_20250316_002_predict_ball.csv
│   └── output.gif
├── main.py
├── models
│   ├── VballNetFastV1_155_h288_w512.onnx
│   ├── VballNetFastV1_seq9_grayscale_233_h288_w512.onnx
│   └── VballNetV1_150_h288_w512.onnx
├── pyproject.toml
├── README.md
├── src
│   ├── inference_onnx.py
│   ├── pose_detector.py
│   └── inference_onnx_seq9_gray_v2.py
└── uv.lock
```

## Requirements
- Python >= 3.12
- Dependencies (listed in `pyproject.toml`):
  - `onnxruntime>=1.22.0`
  - `opencv-python>=4.12.0.88`
  - `pandas>=2.3.1`
  - `tqdm>=4.67.1`
  - `mediapipe>=0.10.14`
  - `scipy>=1.14.1`

Install via:
```bash
uv sync
```

## Performance
- **Hardware**: Intel Core i5-10400F @ 2.90GHz
- **FPS**: 200 (detection + CSV output)
- **Input**: Grayscale video frames for optimized processing
- **Output**: CSV with ball coordinates, optional visualized video

## Use Cases
- **Sports Analytics**: Analyze ball movement for volleyball coaching and strategy.
- **Real-Time Tracking**: Integrate into live broadcasts or automated filming systems.
- **Computer Vision Research**: Study lightweight models for real-time detection.
- **Sports Tech**: Build applications for training or performance analysis.

## Model Details
- **Architecture**: Lightweight CNN optimized for CPU inference.
- **Input**: Grayscale video frames.
- **Weights**: Available at [link to weights](#) <!-- Replace with actual link -->.

## Training model

[For training used - vball-net](https://github.com/asigatchov/vball-net)


## License
[MIT License](LICENSE)

## Contact
For questions or feedback, open an issue on GitHub or reach out on
---

Happy tracking! 🏐