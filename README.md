# Fast Volleyball Ball Tracking → Vertical Reels

A complete high-speed pipeline for volleyball ball detection, tracking, and automatic generation of **9:16 vertical reels** with the ball always centered.
Achieves **~200 FPS** on a regular CPU (Intel i5-10400F) thanks to a lightweight grayscale seq-9 ONNX model.

## Features (fully implemented)

1. Ball detection → `ball.csv`  
2. Track calculation → separate `track_*.json` files  
3. Assembly of all rallies into one horizontal video or individual clips  
4. Creation of **vertical 9:16 reels** with smooth ball-centered cropping (main output)


<table>
   <tr><td>
<img alt="backline" weight="512" src="https://raw.githubusercontent.com/asigatchov/fast-volleyball-tracking-inference/refs/heads/master/examples/output.gif">
   </td><td>
      <img weight="512" src="https://raw.githubusercontent.com/asigatchov/fast-volleyball-tracking-inference/refs/heads/master/examples/sideline.gif" alt="sideline">
</td></tr>
</table>


<video src="https://github.com/asigatchov/fast-volleyball-tracking-inference/raw/refs/heads/master/examples/reel_g.mp4" controls width="100%"></video>

## Model Comparison
| Model                                    | F1    | Precision | Recall | Accuracy | Detection Rate |
|------------------------------------------|-------|-----------|--------|----------|----------------|
| VballNetFastV1_seq9_grayscale_233_h288_w512.onnx | 0.772 | 0.832     | 0.720  | 0.662    | 0.689          |
| VballNetV1b_seq9_grayscale_best.onnx     | 0.855 | 0.818     | 0.896  | 0.767    | 0.840          |
| VballNetV1c_seq9_grayscale_best.onnx     | 0.847 | 0.793     | 0.908  | 0.754    | 0.857          |
| VballNetV1_seq9_grayscale_148_h288_w512.onnx | 0.872 | 0.867     | 0.878  | 0.791    | 0.821          |
| VballNetV1_seq9_grayscale_204_h288_w512.onnx | 0.870 | 0.867     | 0.872  | 0.788    | 0.815          |
| VballNetV1_seq9_grayscale_330_h288_w512.onnx | 0.874 | 0.882     | 0.867  | 0.795    | 0.807          |
| VballNetV2_seq9_grayscale_320_h288_w512.onnx | 0.874 | 0.880     | 0.869  | 0.795    | 0.810          |
| VballNetV2_seq9_grayscale_350_h288_w512.onnx | 0.874 | 0.880     | 0.868  | 0.794    | 0.809          |


## Installation

```bash
# Clone the repository
git clone https://github.com/asigatchov/fast-volleyball-tracking-inference.git
cd fast-volleyball-tracking-inference

# Install dependencies (uv is recommended)
uv sync
# or with pip: pip install -r requirements.txt (if you create one)

## Full pipeline example


VIDEO="examples/beach_st_lenina_20250622_g1_005.mp4"
MODEL="models/VballNetFastV1_seq9_grayscale_233_h288_w512.onnx"
OUT="output"

# 1. Ball detection (CSV only – fastest mode)
uv run src/inference_onnx_seq9_gray_v2.py \
  --video_path $VIDEO \
  --model_path $MODEL \
  --output_dir $OUT \
  --only_csv

# 2. Track calculation from CSV
uv run src/track_calculator.py \
  --csv_path $OUT/beach_st_lenina_20250622_g1_005/ball.csv \
  --output_dir $OUT \
  --fps 30

# 3. (Optional) Assemble all rallies into one horizontal video
uv run src/track_processor.py \
  --video_path $VIDEO \
  --output_dir $OUT



## Output structure
```text

output/beach_st_lenina_20250622_g1_005/
├── ball.csv                  # raw ball coordinates
├── tracks/
│   └── track_0001.json       # one JSON per rally
├── combined.mp4              # all rallies concatenated (optional)
└── reels/
    └── reel_beach_st_lenina_20250622_g1_005_0001.mp4   # vertical 9:16 reels

```


# Individual commands

## Detection only (real-time preview)
```
uv run src/inference_onnx_seq9_gray_v2.py \
  --video_path video.mp4 \
  --model_path model.onnx \
  --visualize
```  

## Single track → vertical reel (with live preview)
```bash
uv run src/make_reels.py \
  --video_path video.mp4 \
  --track_json output/.../tracks/track_0007.json \
  --output_dir output \
  --30 --visualize
```

## All tracks → separate horizontal clips
```bash
uv run src/track_processor.py \
  --video_path video.mp4 \
  --json_dir output/.../tracks \
  --split_dir output/.../clips
```
