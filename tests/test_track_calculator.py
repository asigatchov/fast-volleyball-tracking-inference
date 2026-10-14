import io
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

from ball_tracker import BallTracker
from constants import DEFAULT_DETECTION_BOX_RADIUS
from track_calculator import TrackCalculator, TrackCalculatorConfig


ROOT = Path(__file__).resolve().parents[1]


def make_calculator(court_json_path=None) -> TrackCalculator:
    config = TrackCalculatorConfig(
        csv_path=str(ROOT / "beach-20260610/g_man_beach_20260610_p2_predict_ball.csv"),
        output_dir=str(ROOT / "tmp" / "test-output"),
        fps=30.0,
        max_distance=200.0,
        min_duration_sec=1.0,
        max_x_displacement=20.0,
        min_y_displacement=50.0,
        bounce_frames=10,
        court_json_path=court_json_path,
        video_width=1920,
        video_height=1080,
    )
    return TrackCalculator(config)


def build_raw_track(calculator: TrackCalculator, track_id: int):
    df = calculator._load_and_process_csv()
    tracker = BallTracker(
        buffer_size=2500,
        max_disappeared=40,
        max_distance=calculator._scaled_max_distance,
        fps=calculator.config.fps,
    )
    closed = []

    with redirect_stdout(io.StringIO()):
        for frame_num in sorted(df["Frame"].dropna().astype(int).unique()):
            frame_rows = df[df["Frame"] == frame_num]
            detections = []
            for _, row in frame_rows.iterrows():
                if not pd.isna(row["X"]) and not pd.isna(row["Y"]):
                    detections.append(
                        {
                            "x1": row["X"] - DEFAULT_DETECTION_BOX_RADIUS,
                            "y1": row["Y"] - DEFAULT_DETECTION_BOX_RADIUS,
                            "x2": row["X"] + DEFAULT_DETECTION_BOX_RADIUS,
                            "y2": row["Y"] + DEFAULT_DETECTION_BOX_RADIUS,
                            "confidence": row["Visibility"],
                            "cls_id": 0,
                        }
                    )
            _, _, just_closed = tracker.update(detections, frame_num)
            closed.extend(just_closed)

        for active_track in tracker.tracks.values():
            closed.append(active_track)

    return next(track for track in closed if track.track_id == track_id)


def test_no_court_does_not_trim_on_stop_rising_heuristic() -> None:
    calculator = make_calculator(court_json_path=None)
    raw_track = build_raw_track(calculator, track_id=1)

    assert raw_track.last_frame == 332

    analysis = calculator._analyze_track_trajectory(raw_track)
    assert analysis["stop_rising_above_net_frame"] == 264
    assert analysis["game_pause_frame"] is None

    trimmed = calculator._trim_after_game_pause(raw_track)
    assert trimmed.last_frame == 332


def test_rally_features_and_confidence_are_generated() -> None:
    calculator = make_calculator(court_json_path=None)
    raw_track = build_raw_track(calculator, track_id=1)

    analysis = calculator._analyze_track_trajectory(raw_track)
    features = calculator._extract_rally_features(raw_track, analysis)
    classification = calculator._classify_rally(features)

    assert features["frame_end"] == 332
    assert features["duration_sec"] > 4.5
    assert features["path_len_px"] > 0.0
    assert 0.0 < features["coverage"] <= 1.0
    assert "vy_sign_changes" in features

    assert classification["label"] in {"rally", "not_rally"}
    assert 0.0 <= classification["rally_confidence"] <= 1.0
    assert 0.0 <= classification["not_rally_confidence"] <= 1.0
    assert (
        classification["is_rally"]
        == (classification["rally_confidence"] >= 0.5)
    )
