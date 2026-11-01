from collections import deque

from ball_tracker import Track
from player_contacts import EMPTY_CONTACT_SUMMARY
from track_calculator_with_court import (
    BallSizeFeatures,
    TrackCalculatorConfig,
    TrackCalculatorWithCourt,
    TrackFeatureExtractor,
    court_ball_motion,
    stabilize_court_ball_positions,
)


def make_calculator() -> TrackCalculatorWithCourt:
    config = TrackCalculatorConfig(
        csv_path="unused.csv",
        output_dir="tmp/test-output",
        fps=30.0,
        max_distance=200.0,
        min_duration_sec=0.0,
        max_x_displacement=20.0,
        min_y_displacement=50.0,
        bounce_frames=10,
        court_json_path=None,
        video_width=1920,
        video_height=1080,
        beach=False,
    )
    return TrackCalculatorWithCourt(config)


def make_track(track_id: int, start_frame: int, points: list[tuple[float, float, int]]) -> Track:
    track = Track()
    track.track_id = track_id
    track.fps = 30.0
    track.start_frame = start_frame
    track.last_frame = points[-1][2]
    track.positions = deque([([x, y], frame) for x, y, frame in points], maxlen=3000)
    track.ball_sizes = deque([12.0] * len(points), maxlen=3000)
    track.prediction = [points[-1][0], points[-1][1]]
    return track


def test_foreign_court_trajectory_is_classified_as_technical() -> None:
    extractor = object.__new__(TrackFeatureExtractor)
    features = {
        "duration_sec": 3.6,
        "path_len_m": 8.8,
        "x_range_m": 2.3,
        "vy_sign_changes": 8,
        "post_net_has_second_phase": False,
        "post_net_path_len_m": 0.0,
        "end_zone": "far",
        # This legacy feature is also set by a descending arc on a neighbouring
        # court; it must not exempt the track from the foreign-court check.
        "crosses_net": True,
        "outside_main_court_corridor_ratio": 1.0,
        "player_box_coverage": 1.0,
        "near_player_frame_ratio": 0.0,
    }
    contacts = dict(EMPTY_CONTACT_SUMMARY)
    size = BallSizeFeatures(9.0, 18.0, None, 31.5, 0.45)

    result = extractor._classify_technical_return(
        features, size, "near", contacts, has_player_evidence=True
    )

    assert result["is_technical_return"] is True
    assert result["technical_return_confidence"] > 0.9
    assert "foreign_court_trajectory" in result["reasons"]


def test_court_ball_motion_prefers_smoothed_outgoing_direction() -> None:
    motion = court_ball_motion(
        [(100, -2.0, 1.0), (102, 0.0, 2.0), (104, 4.0, 3.0)]
    )

    assert motion[100] == (1.5, 0.5)
    assert motion[102] == (2.0, 0.5)
    assert motion[104] == (1.5, 0.5)


def test_airborne_ball_is_interpolated_between_player_reach_anchors() -> None:
    positions = [(10, 50.0, 20.0), (15, 80.0, 30.0), (20, 60.0, 25.0)]
    anchors = {10: (-6.0, -2.0, 7), 20: (6.0, 2.0, 9)}

    stabilized = stabilize_court_ball_positions(positions, anchors)

    assert stabilized == [
        (10, -6.0, -2.0, "player_reach", 7),
        (15, 0.0, 0.0, "interpolated_player_reach", None),
        (20, 6.0, 2.0, "player_reach", 9),
    ]


def test_court_view_prefers_receiver_after_serve() -> None:
    interaction = {
        "rally_state": {
            "transitions": [
                {"frame": 100, "contact_type": "serve", "selected_side": "near"},
                {"frame": 140, "contact_type": "dig", "selected_side": "far"},
            ]
        }
    }

    assert TrackCalculatorWithCourt._court_view_expected_side(100, interaction) == "near"
    assert TrackCalculatorWithCourt._court_view_expected_side(120, interaction) == "far"
    assert TrackCalculatorWithCourt._court_view_expected_side(150, interaction) == "far"


def test_merge_airborne_reentry_tracks_when_ball_returns_from_top() -> None:
    calculator = make_calculator()
    first = make_track(
        1,
        100,
        [
            (940.0, 170.0, 100),
            (944.0, 130.0, 102),
            (949.0, 92.0, 104),
            (953.0, 58.0, 106),
        ],
    )
    second = make_track(
        2,
        165,
        [
            (960.0, 52.0, 165),
            (964.0, 88.0, 167),
            (969.0, 126.0, 169),
            (973.0, 164.0, 171),
        ],
    )

    merged = calculator._merge_airborne_reentry_tracks([first, second])

    assert len(merged) == 1
    assert merged[0].start_frame == 100
    assert merged[0].last_frame == 171
    assert [frame for _, frame in merged[0].positions] == [100, 102, 104, 106, 165, 167, 169, 171]


def test_merge_airborne_reentry_tracks_across_a_long_sideways_flight() -> None:
    calculator = make_calculator()
    # The ball leaves the frame moving sideways at ~10 px per frame and stays
    # invisible for 24 frames, so it lands far from where it went out.
    first = make_track(
        1,
        25879,
        [
            (587.0, 75.0, 25954),
            (600.0, 47.0, 25955),
            (612.0, 32.0, 25956),
            (620.0, 17.0, 25957),
            (625.0, 10.0, 25958),
        ],
    )
    second = make_track(
        2,
        25982,
        [
            (870.0, 10.0, 25982),
            (887.0, 40.0, 25984),
            (897.0, 50.0, 25985),
            (907.0, 77.0, 25986),
            (912.0, 92.0, 25987),
        ],
    )

    merged = calculator._merge_airborne_reentry_tracks([first, second])

    assert len(merged) == 1
    assert (merged[0].start_frame, merged[0].last_frame) == (25879, 25987)


def test_do_not_merge_airborne_reentry_tracks_that_land_off_the_predicted_flight() -> None:
    calculator = make_calculator()
    first = make_track(
        1,
        100,
        [
            (940.0, 170.0, 100),
            (944.0, 130.0, 102),
            (949.0, 92.0, 104),
            (953.0, 58.0, 106),
        ],
    )
    # Barely drifting on the way out, yet it comes back 400 px away: a different
    # ball flight, not the same one.
    second = make_track(
        2,
        130,
        [
            (1360.0, 52.0, 130),
            (1364.0, 88.0, 132),
            (1369.0, 126.0, 134),
            (1373.0, 164.0, 136),
        ],
    )

    assert len(calculator._merge_airborne_reentry_tracks([first, second])) == 2


def test_do_not_merge_airborne_reentry_tracks_when_gap_exceeds_three_seconds() -> None:
    calculator = make_calculator()
    first = make_track(
        1,
        100,
        [
            (940.0, 170.0, 100),
            (944.0, 130.0, 102),
            (949.0, 92.0, 104),
            (953.0, 58.0, 106),
        ],
    )
    second = make_track(
        2,
        210,
        [
            (960.0, 52.0, 210),
            (964.0, 88.0, 212),
            (969.0, 126.0, 214),
            (973.0, 164.0, 216),
        ],
    )

    merged = calculator._merge_airborne_reentry_tracks([first, second])

    assert len(merged) == 2
