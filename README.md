# Fast Volleyball Ball Tracking -> Vertical Reels

## Live demos
- VPS: [Tracking - ball](https://demo.vb-ai.ru/)
- Hugging Face: [Tracking - ball](https://huggingface.co/spaces/asigatchov/volleyball-tracking)

High-speed pipeline for volleyball ball detection, rally extraction, and automatic generation of 9:16 reels.

[![src/ball_tracker.py](examples/ball-tracking.jpg)](https://www.youtube.com/watch?v=TBDrTMFMoFA)

## Pipeline
1. `src/inference_onnx_seq_gray_v2.py` -> detects ball and writes `ball.csv` (and optional `predict.mp4`).
2. `src/track_calculator.py` -> converts `ball.csv` to rally tracks (`track_*.json`).
3. `src/track_processor.py` -> creates combined video (`combined.mp4`) or split rally clips.
4. `src/make_reels.py` -> creates vertical 9:16 reels centered around ball trajectory.

`src/track_calculator_with_court.py` is the court-aware variant of step 2. Given
player detections it also finds ball touches with their technique and tells a
rally apart from a ball handed over for the next serve - see
[PLAYER_CONTACTS.md](PLAYER_CONTACTS.md).

## Installation
```bash
git clone https://github.com/asigatchov/fast-volleyball-tracking-inference.git
cd fast-volleyball-tracking-inference
uv sync
```

The pipeline itself needs nothing beyond that - `--visualize` works on the base install,
since `opencv-python` already ships the GUI backend. The `dev` extra adds `matplotlib`
for the diagnostic plots in `src/test_models.py` and `src/serv_det*.py`, and `pytest`
to run the suite under `tests/`:
```bash
uv sync --extra dev
```

## Quick start (tested)
Example input:
- video: `examples/gtu_20250316_002.mp4`
- model: `models/VballNetFastV1_seq9_grayscale_233_h288_w512.onnx`

```bash
VIDEO="examples/gtu_20250316_002.mp4"
MODEL="models/VballNetFastV1_seq9_grayscale_233_h288_w512.onnx"
OUT="output"

# 1) Detection -> ball.csv
uv run src/inference_onnx_seq_gray_v2.py \
  --video_path "$VIDEO" \
  --model_path "$MODEL" \
  --output_dir "$OUT" \
  --only_csv

# 2) Tracks from CSV -> track_*.json
uv run src/track_calculator.py \
  --csv_path "$OUT/gtu_20250316_002/ball.csv" \
  --output_dir "$OUT"

# 3) Optional: combined horizontal rally video
uv run src/track_processor.py \
  --video_path "$VIDEO" \
  --output_dir "$OUT"

# 4) Vertical reels from tracks
uv run src/make_reels.py \
  --video_path "$VIDEO" \
  --json_dir "$OUT/gtu_20250316_002/tracks" \
  --output_dir "$OUT"
```

## Output structure
```text
output/gtu_20250316_002/
├── ball.csv
├── tracks/
│   └── track_0000.json
├── combined.mp4
└── reels/
    └── reel_gtu_20250316_002_0000.mp4
```

## Key CLI options

### `src/inference_onnx_seq_gray_v2.py`
- `--confidence_threshold` - heatmap threshold for detection postprocess.
- `--visualize` - show live preview.
- `--only_csv` - skip writing output video.

### `src/track_calculator.py`
- `--court_json_path` - optional court annotation JSON. If passed, net/court-aware rally filtering is enabled.
- `--fps`, `--max_distance`, `--min_duration_sec` - main tracking/filtering params.

### `src/track_processor.py`
- By default, only tracks with `rally_classification.is_rally=true` are exported;
  `not_rally` and unclassified tracks are skipped.
- `--include-not-rally` - include all tracks for diagnostic exports.
- `--padding` - seconds to add before and after every rally in the exported video.
- `--output_dir` - auto-resolves `tracks` and `combined.mp4` by video basename.
- `--json_dir` - explicit tracks folder.
- `--split_dir` - export each rally into a separate clip.

### `src/show_rally.py`
Interactive review of tracks over the source video:
```bash
uv run src/show_rally.py output/beach-mixt/tracks /path/to/video.mp4 \
  --players_json_path ../uploads/mix/beach-mixt_predictions.json \
  --court_json_path ../uploads/mix/beach-mixt_court.json
```
- `space` play/pause, `a`/`d` step one frame, `w`/`s` jump 15 frames,
  `n`/`p` next/previous track, `v` switches the main view between video and schematic,
  `b` toggles the left box-only panel (court, player boxes and ball),
  `t` ball path, `h` help, `q` quit.
- Player and court JSON files are auto-detected next to `<clip>/tracks` by the
  `<clip>_predict.json` and `<clip>_court.json` names. Legacy `<clip>_coort.json`
  files are also recognized; explicit paths override auto-detection.
- `--track N` - start from a given track, `--snapshot FILE` - render one frame and exit
  (works without a display).

### `src/make_reels.py`
- `--smoothing {none,moving_avg,savitzky_golay,kalman}`
- `--interpolation {hold,linear}`
- `--margin` - lead offset in movement direction.
- `--padding {none,mirror,black}`


## OpenVino runtime
### `uv run src/inference_openvino_seq_gray_v2.py`
- `--model_xml ./ov/VballNetV4c_seq9_grayscale_20260908_213829.xml`
- `--video_path ./examples/gtu_20250316_002.mp4`
- `--only_csv`
- `--output_dir ./demo-result/`

## Available ONNX models
Benchmark setup:
- runner: `scripts/eval_ov_models.py`, which reuses the decode path of `src/inference_openvino_seq_gray_v2.py`
- dataset: `beach-test-raw` - 3 clips, 886 labelled frames, 1280x720 and 1920x1080
- runtime: OpenVINO on CPU (Intel Core i5-10400F, 12 threads)
- a detection counts as a hit when it lands within 9 px of the label at 1920 frame width; the tolerance is scaled to each clip's resolution, so the 720p clips are judged at 6 px
- `Precision`, `Recall` and `F1` are computed over those hits; frames where label and prediction agree the ball is invisible count as true negatives
- `CPU FPS` times the inference call alone, not the end-to-end pipeline. `scripts/bench_inference_fps.py` feeds the model a synthetic random tensor of its own
  input shape (`[1, seq, H, W]`, float32), runs 10 warm-up inferences and then times 60 more. One call decodes `seq` frames at once, so a run scores `seq / median(latency)`.
  The model is reloaded for each of the 3 repeats, because OpenVINO lays the network out across threads differently from load to load, and the table reports the median of
  those repeats. Video decode, the grayscale+resize preprocess and the heatmap/grid postprocess are all excluded.

| Model | F1 | Precision | Recall | CPU FPS |
| --- | ---: | ---: | ---: | ---: |
| `VballNetV4c_seq9_grayscale_20260908_213829.onnx` | 0.902 | 0.900 | 0.904 | 149.6 |
| `VballNetGridV2b_seq9_grayscale_20260909_001145.onnx` | 0.892 | 0.874 | 0.910 | 122.9 |
| `VballNetGridV3_seq9_grayscale_20260908_225156.onnx` | 0.867 | 0.870 | 0.863 | 229.3 |
| `VballNetFastV1_seq9_grayscale_233_h288_w512.onnx` | 0.799 | 0.778 | 0.822 | 1140.0 |

Reproduce the speed column with:
```bash
uv run scripts/bench_inference_fps.py models/*.onnx
```
Expect the result to move by 3-4% between invocations. A full pipeline run is slower than the number above, since inference then shares the
same cores with video decode - roughly 10% off for `VballNetGridV3` on this machine, and more on the lighter models where decode weighs relatively more.

Earlier checkpoints and their OpenVINO IR counterparts live in `old_models/onnx/` and
`old_models/ov/`; they are kept for reference and are not benchmarked here.

## Notes
- `onnxruntime` can run on CPU if CUDA provider is unavailable.
- All scripts support `--help` and can be launched through `uv run`.
