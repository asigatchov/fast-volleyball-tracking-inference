import json
from collections import deque

import numpy as np
import pandas as pd
import pytest

from ball_tracker import Track
from track_calculator_with_court import TrackCalculatorConfig, TrackCalculatorWithCourt

FPS = 30.0


def write_predictions(tmp_path, balls, players=(), name="predictions.json"):
    path = tmp_path / name
    predictions = [
        {
            "class_id": 1,
            "class_name": "ball",
            "score": 0.9,
            "frame_index": frame,
            "bbox_xyxy": [x - 8.0, y - 8.0, x + 8.0, y + 8.0],
        }
        for frame, x, y in balls
    ]
    predictions += [
        {
            "class_id": 0,
            "class_name": "player",
            "score": 0.9,
            "track_id": track_id,
            "frame_index": frame,
            "bbox_xyxy": list(box),
        }
        for frame, track_id, box in players
    ]
    path.write_text(
        json.dumps({"fps": FPS, "source_size": {"width": 1280, "height": 720}, "predictions": predictions}),
        encoding="utf-8",
    )
    return str(path)


def write_court(tmp_path):
    path = tmp_path / "court.json"
    path.write_text(
        json.dumps(
            {
                "frame_width": 1280,
                "frame_height": 720,
                "keypoints": [
                    {"id": 0, "x": 90, "y": 712, "visible": True},
                    {"id": 1, "x": 450, "y": 375, "visible": True},
                    {"id": 2, "x": 770, "y": 358, "visible": True},
                    {"id": 3, "x": 1276, "y": 607, "visible": True},
                    {"id": 4, "x": 374, "y": 446, "visible": True},
                    {"id": 5, "x": 898, "y": 424, "visible": True},
                    {"id": 6, "x": 357, "y": 288, "visible": True},
                    {"id": 7, "x": 888, "y": 263, "visible": True},
                ],
            }
        ),
        encoding="utf-8",
    )
    return str(path)


def make_calculator(tmp_path, players_json_path=None, **overrides) -> TrackCalculatorWithCourt:
    options = dict(
        csv_path=str(tmp_path / "ball.csv"),
        output_dir=str(tmp_path / "out"),
        fps=FPS,
        max_distance=200.0,
        min_duration_sec=0.0,
        max_x_displacement=20.0,
        min_y_displacement=50.0,
        bounce_frames=10,
        court_json_path=write_court(tmp_path),
        video_width=1280,
        video_height=720,
        beach=True,
        players_json_path=players_json_path,
    )
    options.update(overrides)
    calculator = TrackCalculatorWithCourt(TrackCalculatorConfig(**options))
    calculator._build_player_context()
    return calculator


def make_frame(rows) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["Frame", "Visibility", "X", "Y", "Radius"])
    frame["BallSource"] = np.where(frame["X"].notna(), "csv", "none")
    return frame


def make_track(track_id: int, points: list[tuple[float, float, int]]) -> Track:
    track = Track()
    track.track_id = track_id
    track.fps = FPS
    track.start_frame = points[0][2]
    track.last_frame = points[-1][2]
    track.positions = deque([([x, y], frame) for x, y, frame in points], maxlen=3000)
    track.ball_sizes = deque([12.0] * len(points), maxlen=3000)
    track.prediction = [points[-1][0], points[-1][1]]
    return track


def test_missing_ball_frames_are_filled_from_the_players_json(tmp_path) -> None:
    calculator = make_calculator(
        tmp_path,
        write_predictions(tmp_path, [(12, 305.0, 205.0)]),
    )
    frame = make_frame(
        [(index, 1, 300.0 + index, 200.0 + index, 12.0) for index in range(10, 12)]
        + [(12, 0, np.nan, np.nan, 0.0)]
        + [(index, 1, 300.0 + index, 200.0 + index, 12.0) for index in range(13, 16)]
    )

    assert calculator._apply_player_ball_detections(frame) is True
    filled = frame[frame["Frame"] == 12].iloc[0]
    assert filled["X"] == pytest.approx(305.0)
    assert filled["Y"] == pytest.approx(205.0)
    assert filled["Visibility"] == pytest.approx(0.9)
    assert filled["BallSource"] == "players_fill"


def test_players_ball_off_the_flight_is_not_filled(tmp_path) -> None:
    # A second ball 180 px away from the flight must not become a one-frame jump.
    calculator = make_calculator(
        tmp_path,
        write_predictions(tmp_path, [(12, 380.0, 370.0)]),
    )
    frame = make_frame(
        [(index, 1, 300.0 + index, 200.0 + index, 12.0) for index in range(10, 12)]
        + [(12, 0, np.nan, np.nan, 0.0)]
        + [(index, 1, 300.0 + index, 200.0 + index, 12.0) for index in range(13, 16)]
    )

    assert calculator._apply_player_ball_detections(frame) is False
    assert frame[frame["Frame"] == 12].iloc[0]["BallSource"] == "none"


def test_isolated_detection_during_a_pause_is_not_filled(tmp_path) -> None:
    calculator = make_calculator(
        tmp_path,
        write_predictions(tmp_path, [(60, 500.0, 400.0)]),
    )
    frame = make_frame(
        [(index, 1, 300.0 + index, 200.0, 12.0) for index in range(10, 16)]
        + [(index, 0, np.nan, np.nan, 0.0) for index in range(16, 80)]
    )

    assert calculator._apply_player_ball_detections(frame) is False
    assert frame[frame["Frame"] == 60].iloc[0]["BallSource"] == "none"


def test_csv_outlier_is_repaired_from_the_players_json(tmp_path) -> None:
    calculator = make_calculator(
        tmp_path,
        write_predictions(tmp_path, [(13, 313.0, 200.0)]),
    )
    rows = [(index, 1, 300.0 + index, 200.0, 12.0) for index in range(10, 17)]
    rows[3] = (13, 1, 313.0, 460.0, 12.0)  # ball snapped onto a player's hand
    frame = make_frame(rows)

    assert calculator._apply_player_ball_detections(frame) is True
    repaired = frame[frame["Frame"] == 13].iloc[0]
    assert repaired["Y"] == pytest.approx(200.0)
    assert repaired["BallSource"] == "players_repair"


def test_good_csv_detection_is_kept(tmp_path) -> None:
    calculator = make_calculator(
        tmp_path,
        write_predictions(tmp_path, [(13, 316.0, 203.0)]),
    )
    frame = make_frame([(index, 1, 300.0 + index, 200.0, 12.0) for index in range(10, 17)])

    assert calculator._apply_player_ball_detections(frame) is False
    assert frame[frame["Frame"] == 13].iloc[0]["X"] == pytest.approx(313.0)
    assert frame[frame["Frame"] == 13].iloc[0]["BallSource"] == "csv"


def test_ball_is_taken_from_the_players_json_without_a_csv(tmp_path) -> None:
    balls = [(frame, 300.0 + 5 * frame, 400.0 - 3 * frame) for frame in range(10, 20)]
    calculator = make_calculator(
        tmp_path,
        write_predictions(tmp_path, balls),
        csv_path=None,
    )

    frame = calculator._load_csv()

    assert list(frame["Frame"]) == list(range(10, 20))
    assert frame.iloc[0]["X"] == pytest.approx(350.0)
    assert set(frame["BallSource"]) == {"players"}
    assert calculator._observations_by_frame[10].radius_px == pytest.approx(8.0)


def test_frames_without_a_ball_stay_empty_without_a_csv(tmp_path) -> None:
    balls = [(10, 300.0, 200.0), (40, 500.0, 300.0)]
    calculator = make_calculator(tmp_path, write_predictions(tmp_path, balls), csv_path=None)

    frame = calculator._load_csv()

    assert len(frame) == 31
    assert frame[frame["Frame"] == 25].iloc[0]["BallSource"] == "none"
    assert set(calculator._observations_by_frame) == {10, 40}


def test_without_a_csv_a_players_json_with_no_ball_is_an_error(tmp_path) -> None:
    players = [(frame, 5, (560.0, 320.0, 640.0, 500.0)) for frame in range(90, 130)]
    calculator = make_calculator(
        tmp_path,
        write_predictions(tmp_path, [], players),
        csv_path=None,
    )

    with pytest.raises(ValueError, match="no ball detections"):
        calculator._load_csv()


def test_output_directory_is_named_after_the_players_json(tmp_path) -> None:
    calculator = make_calculator(
        tmp_path,
        write_predictions(tmp_path, [(10, 300.0, 200.0)], name="beach-mixt_predictions.json"),
        csv_path=None,
    )

    assert calculator._source_basename() == "beach-mixt"


def test_track_is_split_where_no_detector_sees_the_ball(tmp_path) -> None:
    calculator = make_calculator(tmp_path, write_predictions(tmp_path, [(10, 300.0, 200.0)]))
    calculator._observations_by_frame = {}
    track = make_track(
        1,
        [(300.0, 200.0, frame) for frame in range(10, 20)]
        + [(400.0, 300.0, frame) for frame in range(60, 70)],
    )

    parts = calculator._split_blind_gaps([track])

    assert len(parts) == 2
    assert (parts[0].start_frame, parts[0].last_frame) == (10, 19)
    assert (parts[1].start_frame, parts[1].last_frame) == (60, 69)


def test_track_is_not_split_when_the_other_detector_sees_the_ball(tmp_path) -> None:
    balls = [(frame, 300.0 + frame, 250.0) for frame in range(20, 60)]
    calculator = make_calculator(tmp_path, write_predictions(tmp_path, balls))
    calculator._observations_by_frame = {}
    track = make_track(
        1,
        [(300.0, 200.0, frame) for frame in range(10, 20)]
        + [(400.0, 300.0, frame) for frame in range(60, 70)],
    )

    assert len(calculator._split_blind_gaps([track])) == 1


def test_tracks_split_by_a_touch_are_merged_back(tmp_path) -> None:
    players = [(frame, 5, (560.0, 320.0, 640.0, 500.0)) for frame in range(90, 130)]
    calculator = make_calculator(tmp_path, write_predictions(tmp_path, [], players))
    calculator._observations_by_frame = {}

    falling = make_track(1, [(600.0, 330.0 + 10 * step, 100 + step) for step in range(8)])
    rising = make_track(2, [(610.0, 390.0 - 10 * step, 112 + step) for step in range(8)])

    merged = calculator._merge_contact_split_tracks([falling, rising])

    assert len(merged) == 1
    assert (merged[0].start_frame, merged[0].last_frame) == (100, 119)


def test_distant_tracks_are_not_merged(tmp_path) -> None:
    players = [(frame, 5, (560.0, 320.0, 640.0, 500.0)) for frame in range(90, 130)]
    calculator = make_calculator(tmp_path, write_predictions(tmp_path, [], players))
    calculator._observations_by_frame = {}

    first = make_track(1, [(600.0, 330.0 + 10 * step, 100 + step) for step in range(8)])
    far_away = make_track(2, [(1100.0, 600.0, 112 + step) for step in range(8)])

    assert len(calculator._merge_contact_split_tracks([first, far_away])) == 2
