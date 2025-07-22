# Fast Volleyball Tracking Inference

High-speed volleyball ball detection and tracking using an optimized ONNX model, achieving **200 FPS** on an Intel Core i5-10400F CPU @ 2.90GHz. This repository provides scripts for real-time inference, outputting ball coordinates to CSV and optional visualized video output.
<table>
   <tr><td>
<img alt="backline" weight="512" src="https://raw.githubusercontent.com/asigatchov/fast-volleyball-tracking-inference/refs/heads/master/examples/output.gif">
   </td><td>
      <img weight="512" src="https://raw.githubusercontent.com/asigatchov/fast-volleyball-tracking-inference/refs/heads/master/examples/sideline.gif" alt="sideline">
</td></tr>   
</table>



## Features
- **High Performance**: 200 FPS on modest CPU hardware (Intel Core i5-10400F @ 2.90GHz).
- **Optimized for CPU**: Lightweight ONNX model designed for grayscale video input.
- **Flexible Output**: Saves ball coordinates to CSV for analysis; optional video visualization.
- **Customizable**: Adjustable track length for visualization.
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

Run the inference script to detect and track a volleyball ball in a video:

```bash
uv run src/inference_onnx_seq9_gray_v2.py --video_path  examples/beach_st_lenina_20250622_g1_005.mp4 --model_path  models/VballNetFastV1_seq9_grayscale_233_h288_w512.onnx --output_dir output/
```

#### Run the inference script to detect and track a volleyball ball  Realtime visualize:

```bash
uv run src/inference_onnx_seq9_gray_v2.py --video_path  examples/beach_st_lenina_20250622_g1_005.mp4 --model_path  models/VballNetFastV1_seq9_grayscale_233_h288_w512.onnx --visualize
```


### Command-Line Options
```
usage: inference_onnx_seq9_gray_v2.py [-h] --video_path VIDEO_PATH [--track_length TRACK_LENGTH] [--output_dir OUTPUT_DIR] --model_path MODEL_PATH
                                      [--visualize] [--only_csv]

Volleyball ball detection and tracking with ONNX

options:
  -h, --help            show this help message and exit
  --video_path VIDEO_PATH
                        Path to input video file
  --track_length TRACK_LENGTH
                        Length of the ball track
  --output_dir OUTPUT_DIR
                        Directory to save output video and CSV
  --model_path MODEL_PATH
                        Path to ONNX model file
  --visualize           Enable visualization on display
  --only_csv            Save only CSV, skip video output
```

### Example
```bash
uv run src/inference_onnx_seq9_gray_v2.py --video_path examples/sample_video.mp4 --model_path weights/model.onnx --output_dir output/ --track_length 10 --visualize
```

This command processes `sample_video.mp4`, saves ball coordinates to `output/coordinates.csv`, and displays a visualized video with a track length of 10 frames.

## Output
- **CSV File**: Contains frame ID and ball coordinates (x, y).
```csv
Frame,Visibility,X,Y
0,0,-1,-1
1,1,1068,536
2,1,1068,532
3,1,1068,525
...
1008,1,1065,487
```
- **Video (Optional)**: Visualized output with tracked ball path, saved to `output/`.

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
