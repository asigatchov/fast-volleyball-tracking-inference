import json
import math
from dataclasses import replace

import pytest

from player_contacts import (
    CONTACT_DIG,
    CONTACT_OVERHEAD,
    CONTACT_SERVE,
    BallEdge,
    BallSample,
    Contact,
    ContactChoice,
    ContactDetector,
    PlayerStore,
    RallyContactStateMachine,
    summarize_contacts,
)

FPS = 30.0
BOX_HEIGHT = 100.0
BOX_TOP = 300.0
BOX_BOTTOM = 400.0


class FakeCourt:
    """Linear stand-in for the perspective transform: x -> depth, y -> width."""

    camera_position = "backline"
    court_length_m = 16.0
    court_width_m = 8.0
    cm_per_px_scale = 1.0

    def to_court(self, x: float, y: float) -> tuple[float, float]:
        return (x - 640.0) / 40.0, (y - 360.0) / 45.0

    def net_y_at_x(self, x: float) -> float:
        return 200.0


def write_players(tmp_path, boxes, name="players.json"):
    path = tmp_path / name
    path.write_text(
        json.dumps(
            {
                "format": "ravel-vb-predictions-v1",
                "fps": FPS,
                "source_size": {"width": 1280, "height": 720},
                "predictions": boxes,
            }
        ),
        encoding="utf-8",
    )
    return str(path)


def player_boxes(center_x: float, frames: range, track_id: int = 7, half_width: float = 20.0):
    return [
        {
            "class_id": 0,
            "class_name": "player",
            "score": 0.9,
            "track_id": track_id,
            "frame_index": frame,
            "bbox_xyxy": [center_x - half_width, BOX_TOP, center_x + half_width, BOX_BOTTOM],
        }
        for frame in frames
    ]


def make_detector(tmp_path, center_x: float = 600.0):
    store = PlayerStore(
        write_players(tmp_path, player_boxes(center_x, range(0, 40))),
        court=FakeCourt(),
    )
    return ContactDetector(store, FakeCourt(), FPS)


def bounce_samples(contact_y: float, center_x: float = 600.0):
    """Ball falls onto `contact_y` at frame 20 and is played straight back up."""
    samples = []
    for frame in range(10, 21):
        samples.append(BallSample(frame=frame, x=center_x, y=contact_y - (20 - frame) * 10.0))
    for frame in range(21, 31):
        samples.append(BallSample(frame=frame, x=center_x, y=contact_y - (frame - 20) * 10.0))
    return samples


def make_contact(
    frame: int,
    track_id: int,
    side: str,
    contact_type: str = CONTACT_DIG,
    reach_ratio: float = 0.25,
) -> Contact:
    return Contact(
        frame=frame,
        player_track_id=track_id,
        contact_type=contact_type,
        depth_ratio=0.5,
        posture_ratio=0.4,
        reach_ratio=reach_ratio,
        approach_px_frame=0.0,
        direction_change_deg=90.0,
        velocity_residual_px_frame=12.0,
        speed_before_px_frame=8.0,
        speed_after_px_frame=12.0,
        confidence=0.8,
        side=side,
        player_court_x=-4.0 if side == "near" else 4.0,
        player_court_y=0.0,
        in_serve_zone=contact_type == CONTACT_SERVE,
        ball_x=480.0 if side == "near" else 800.0,
        ball_y=330.0,
        ball_above_net=False,
        gap_frames=1,
    )


def make_serve_start(side: str = "far") -> BallEdge:
    return BallEdge(
        frame=10,
        origin="serve_zone",
        player_track_id=9,
        side=side,
        player_court_x=8.0,
        player_court_y=0.0,
        in_serve_zone=True,
        reach_ratio=0.1,
        depth_ratio=0.2,
        posture_ratio=0.4,
        ball_x=960.0,
        ball_y=320.0,
    )


def test_ball_played_at_waist_height_is_a_dig(tmp_path) -> None:
    detector = make_detector(tmp_path)

    contacts = detector.detect(bounce_samples(BOX_TOP + 0.5 * BOX_HEIGHT))

    assert len(contacts) == 1
    assert contacts[0].contact_type == CONTACT_DIG
    assert contacts[0].player_track_id == 7
    assert math.isclose(contacts[0].depth_ratio, 0.5, abs_tol=0.05)


def test_ball_played_at_head_height_is_an_overhead_pass(tmp_path) -> None:
    detector = make_detector(tmp_path)

    contacts = detector.detect(bounce_samples(BOX_TOP + 0.1 * BOX_HEIGHT))

    assert len(contacts) == 1
    assert contacts[0].contact_type == CONTACT_OVERHEAD


def test_contact_uses_adjacent_player_box_when_overlap_lowers_score(tmp_path) -> None:
    target = player_boxes(600.0, range(0, 40), track_id=51)
    target[20]["score"] = 0.65
    distractor = player_boxes(300.0, range(0, 40), track_id=50)
    detector = ContactDetector(
        PlayerStore(
            write_players(tmp_path, target + distractor, "overlap-score-drop.json"),
            court=FakeCourt(),
            score_threshold=0.70,
        ),
        FakeCourt(),
        FPS,
    )

    contacts = detector.detect(bounce_samples(BOX_TOP + 0.1 * BOX_HEIGHT))

    assert len(contacts) == 1
    assert contacts[0].frame == 20
    assert contacts[0].player_track_id == 51


def test_head_high_ball_played_from_a_lunge_is_a_dig(tmp_path) -> None:
    lunging = ContactDetector(
        PlayerStore(
            write_players(
                tmp_path,
                player_boxes(600.0, range(0, 40), half_width=40.0),
                "lunge.json",
            ),
            court=FakeCourt(),
        ),
        FakeCourt(),
        FPS,
    )

    contacts = lunging.detect(bounce_samples(BOX_TOP + 0.1 * BOX_HEIGHT))

    assert len(contacts) == 1
    assert contacts[0].contact_type == CONTACT_DIG
    assert contacts[0].posture_ratio > 0.65


def test_free_flight_over_a_player_is_not_a_contact(tmp_path) -> None:
    detector = make_detector(tmp_path)
    gravity = detector.gravity_px_per_frame2
    samples = [
        BallSample(
            frame=frame,
            x=600.0,
            y=250.0 - 12.0 * (frame - 20) + 0.5 * gravity * (frame - 20) ** 2,
        )
        for frame in range(10, 31)
    ]

    assert detector.detect(samples) == []


def test_player_proximity_reports_a_trajectory_far_from_all_players(tmp_path) -> None:
    detector = make_detector(tmp_path, center_x=600.0)
    samples = [
        BallSample(frame=frame, x=80.0, y=180.0, radius_px=8.0)
        for frame in range(10, 31)
    ]

    proximity = detector.player_proximity(samples)

    assert proximity["player_box_coverage"] == 1.0
    assert proximity["near_player_frame_ratio"] == 0.0
    assert proximity["within_reach_frame_ratio"] == 0.0
    assert proximity["min_player_reach_ratio"] > 1.5


def test_start_from_the_serve_zone_is_separated_from_a_court_handover(tmp_path) -> None:
    serving = ContactDetector(
        PlayerStore(
            write_players(tmp_path, player_boxes(940.0, range(0, 20)), "serve.json"),
            court=FakeCourt(),
        ),
        FakeCourt(),
        FPS,
    )
    handing_over = ContactDetector(
        PlayerStore(
            write_players(tmp_path, player_boxes(700.0, range(0, 20)), "handover.json"),
            court=FakeCourt(),
        ),
        FakeCourt(),
        FPS,
    )

    serve_start = serving.detect_start(
        [BallSample(frame=frame, x=940.0, y=330.0 - frame) for frame in range(0, 10)]
    )
    handover_start = handing_over.detect_start(
        [BallSample(frame=frame, x=700.0, y=330.0 - frame) for frame in range(0, 10)]
    )

    assert serve_start.origin == "serve_zone"
    assert serve_start.at_serve_zone
    assert handover_start.origin == "in_court"
    assert handover_start.at_player_in_court


def test_serve_struck_off_a_toss_at_head_height_is_a_serve(tmp_path) -> None:
    serving = ContactDetector(
        PlayerStore(
            write_players(tmp_path, player_boxes(940.0, range(0, 40)), "serve_toss.json"),
            court=FakeCourt(),
        ),
        FakeCourt(),
        FPS,
    )
    # The toss drops slowly onto the player, then the strike sends it away fast.
    tossed = [BallSample(frame=frame, x=940.0, y=290.0 + 2.0 * frame) for frame in range(0, 13)]
    struck = [
        BallSample(frame=frame, x=940.0 - 25.0 * (frame - 12), y=316.0 - 4.0 * (frame - 12))
        for frame in range(13, 25)
    ]

    analysis = serving.analyze(tossed + struck)

    assert analysis.start.at_serve_zone
    serves = [contact for contact in analysis.contacts if contact.contact_type == "serve"]
    assert len(serves) == 1
    assert serves[0].depth_ratio > 0.0  # struck at head height, not above it
    assert analysis.rally_state["phase"] == "serve_flight"
    assert analysis.rally_state["serving_side"] == serves[0].side


def toss_and_strike(gravity: float, x: float = 940.0, toss_frames: int = 30):
    """Ball lowered in the hand, tossed up, falling back and struck away."""
    lowered = [BallSample(frame=frame, x=x, y=320.0 + 2.0 * frame) for frame in range(0, 10)]
    speed = 0.5 * gravity * toss_frames
    tossed = [
        BallSample(frame=10 + t, x=x, y=338.0 - speed * t + 0.5 * gravity * t * t)
        for t in range(1, toss_frames + 1)
    ]
    struck_at = 10 + toss_frames
    struck = [
        BallSample(frame=struck_at + t, x=x - 25.0 * t, y=338.0 - 4.0 * t)
        for t in range(1, 13)
    ]
    return lowered + tossed + struck, struck_at


def serving_detector(tmp_path, boxes, name: str) -> ContactDetector:
    return ContactDetector(PlayerStore(write_players(tmp_path, boxes, name), court=FakeCourt()), FakeCourt(), FPS)


def test_toss_and_strike_of_one_player_are_one_serve(tmp_path) -> None:
    detector = serving_detector(tmp_path, player_boxes(940.0, range(0, 60)), "toss.json")
    samples, struck_at = toss_and_strike(detector.gravity_px_per_frame2)
    raw = detector._detect_choices(samples)
    assert [choice.best.player_track_id for choice in raw[:2]] == [7, 7]  # the toss and the strike

    analysis = detector.analyze(samples)

    assert analysis.contacts[0].contact_type == CONTACT_SERVE
    assert abs(analysis.contacts[0].frame - struck_at) <= 2
    assert len(analysis.contacts) == len(raw) - 1
    assert sum(contact.contact_type == CONTACT_SERVE for contact in analysis.contacts) == 1


def test_touches_of_two_players_are_not_a_toss(tmp_path) -> None:
    detector = serving_detector(tmp_path, player_boxes(940.0, range(0, 60)), "two.json")
    samples, _ = toss_and_strike(detector.gravity_px_per_frame2)
    choices = detector._detect_choices(samples)
    other = replace(choices[1], best=replace(choices[1].best, player_track_id=8))

    merged = detector._merge_serve_toss([choices[0], other, *choices[2:]], samples)

    assert len(merged) == len(choices)


def test_two_touches_without_the_ball_going_up_are_not_a_toss(tmp_path) -> None:
    detector = serving_detector(tmp_path, player_boxes(940.0, range(0, 60)), "flat.json")
    samples, _ = toss_and_strike(detector.gravity_px_per_frame2)
    choices = detector._detect_choices(samples)
    flat = [sample for sample in samples if not choices[0].best.frame < sample.frame < choices[1].best.frame]
    flat += [
        BallSample(frame=frame, x=940.0, y=choices[0].best.ball_y + 1.0)
        for frame in range(choices[0].best.frame + 1, choices[1].best.frame)
    ]

    merged = detector._merge_serve_toss(choices, sorted(flat, key=lambda sample: sample.frame))

    assert len(merged) == len(choices)


def test_server_touches_right_after_the_serve_are_one_serve(tmp_path) -> None:
    detector = make_detector(tmp_path)
    contacts = [
        make_contact(10, 9, "far", CONTACT_SERVE),  # the toss, marked as the serve
        replace(make_contact(24, 9, "far"), velocity_residual_px_frame=60.0),  # the strike
        make_contact(40, 9, "far"),  # the ball leaving the hand
        make_contact(60, 3, "near"),
        make_contact(90, 9, "far"),  # after somebody else played it: a real touch
    ]

    settled, kept = detector._settle_serve(contacts)

    assert kept == [1, 3, 4]
    assert [(contact.frame, contact.contact_type) for contact in settled] == [
        (24, CONTACT_SERVE),
        (60, CONTACT_DIG),
        (90, CONTACT_DIG),
    ]


def test_server_touch_long_after_the_serve_is_kept(tmp_path) -> None:
    detector = make_detector(tmp_path)
    late = 10 + int(FPS * 2.5)  # the return was not seen, the ball had time to come back
    contacts = [make_contact(10, 9, "far", CONTACT_SERVE), make_contact(late, 9, "far")]

    settled, kept = detector._settle_serve(contacts)

    assert kept == [0, 1]
    assert [contact.contact_type for contact in settled] == [CONTACT_SERVE, CONTACT_DIG]


def test_fast_ball_deflected_in_the_serving_zone_is_not_a_serve(tmp_path) -> None:
    detector = ContactDetector(
        PlayerStore(
            write_players(tmp_path, player_boxes(940.0, range(0, 40)), "deep_dig.json"),
            court=FakeCourt(),
        ),
        FakeCourt(),
        FPS,
    )
    # Already flying fast when it reaches the deep player: a rally touch.
    arriving = [BallSample(frame=frame, x=700.0 + 20.0 * frame, y=250.0 + 6.0 * frame) for frame in range(0, 13)]
    leaving = [
        BallSample(frame=frame, x=940.0 - 18.0 * (frame - 12), y=328.0 - 5.0 * (frame - 12))
        for frame in range(13, 25)
    ]

    analysis = detector.analyze(arriving + leaving)

    assert all(contact.contact_type != "serve" for contact in analysis.contacts)


def test_rally_starts_where_the_handled_ball_leaves_the_player(tmp_path) -> None:
    serving = ContactDetector(
        PlayerStore(
            write_players(tmp_path, player_boxes(940.0, range(0, 80)), "handling.json"),
            court=FakeCourt(),
        ),
        FakeCourt(),
        FPS,
    )
    # Two seconds of the ball being carried and tossed next to the player...
    handled = [
        BallSample(frame=frame, x=940.0 + (frame % 4), y=340.0 - (frame % 6) * 2.0)
        for frame in range(0, 40)
    ]
    # ... and then the strike that sends it across the court.
    flying = [
        BallSample(frame=frame, x=940.0 - 25.0 * (frame - 39), y=340.0 - 6.0 * (frame - 39))
        for frame in range(40, 60)
    ]

    analysis = serving.analyze(handled + flying)

    assert analysis.rally_start_frame is not None
    assert 36 <= analysis.rally_start_frame <= 44
    assert any(contact.contact_type == "serve" for contact in analysis.contacts)


def test_rally_start_is_not_moved_for_a_trajectory_that_starts_in_flight(tmp_path) -> None:
    detector = make_detector(tmp_path)
    # A ball arriving fast and dug straight back only brushes the player, so
    # there is no handling phase to cut away.
    contact_y = BOX_TOP + 0.5 * BOX_HEIGHT
    dug = [BallSample(frame=frame, x=600.0, y=contact_y - (20 - frame) * 30.0) for frame in range(10, 21)]
    dug += [BallSample(frame=frame, x=600.0, y=contact_y - (frame - 20) * 30.0) for frame in range(21, 31)]

    analysis = detector.analyze(dug)

    assert analysis.contacts
    assert analysis.rally_start_frame is None


def facing_players(frames: range):
    """Two overlapping boxes whose feet land on opposite sides of the net."""
    near = [
        {
            "class_id": 0,
            "class_name": "player",
            "score": 0.9,
            "track_id": 1,
            "frame_index": frame,
            "bbox_xyxy": [540.0, BOX_TOP, 660.0, BOX_BOTTOM],
        }
        for frame in frames
    ]
    far = [
        {
            "class_id": 0,
            "class_name": "player",
            "score": 0.9,
            "track_id": 2,
            "frame_index": frame,
            "bbox_xyxy": [620.0, BOX_TOP, 740.0, BOX_BOTTOM],
        }
        for frame in frames
    ]
    return near + far


def perspective_overlapping_players(frames: range):
    """A tall foreground box obscures a shorter player on the far half."""
    near = [
        {
            "class_id": 0,
            "class_name": "player",
            "score": 0.9,
            "track_id": 1,
            "frame_index": frame,
            "bbox_xyxy": [560.0, 200.0, 680.0, 400.0],
        }
        for frame in frames
    ]
    far = [
        {
            "class_id": 0,
            "class_name": "player",
            "score": 0.9,
            "track_id": 2,
            "frame_index": frame,
            "bbox_xyxy": [620.0, 300.0, 740.0, 400.0],
        }
        for frame in frames
    ]
    return near + far


def test_ball_player_scale_selects_obscured_far_player(tmp_path) -> None:
    detector = ContactDetector(
        PlayerStore(
            write_players(
                tmp_path,
                perspective_overlapping_players(range(0, 40)),
                "perspective.json",
            ),
            court=FakeCourt(),
        ),
        FakeCourt(),
        FPS,
    )
    no_size = bounce_samples(350.0, center_x=630.0)
    small_ball = [
        BallSample(frame=item.frame, x=item.x, y=item.y, radius_px=6.0)
        for item in no_size
    ]

    geometry_contact = detector.detect(no_size)[0]
    scaled_contact = detector.detect(small_ball)[0]

    assert geometry_contact.player_track_id == 1
    assert geometry_contact.side == "near"
    assert scaled_contact.player_track_id == 2
    assert scaled_contact.side == "far"
    assert scaled_contact.ball_player_radius_ratio == pytest.approx(0.06)
    assert scaled_contact.inferred_player_height_cm == pytest.approx(175.0, abs=0.2)
    assert scaled_contact.size_mismatch == 0.0


def test_fourth_touch_in_a_row_is_given_to_the_other_side(tmp_path) -> None:
    detector = ContactDetector(
        PlayerStore(
            write_players(tmp_path, facing_players(range(0, 120)), "facing.json"),
            court=FakeCourt(),
        ),
        FakeCourt(),
        FPS,
    )
    # Four touches in a row at the same spot, ambiguous between the two players.
    samples = []
    for bounce in range(4):
        base = bounce * 20
        samples += [BallSample(frame=base + step, x=650.0, y=200.0 + 15.0 * step) for step in range(0, 11)]
        samples += [
            BallSample(frame=base + 10 + step, x=650.0, y=350.0 - 15.0 * step) for step in range(1, 10)
        ]

    contacts = detector.detect(sorted({sample.frame: sample for sample in samples}.values(), key=lambda s: s.frame))
    sides = [contact.side for contact in contacts]

    assert len(sides) >= 4
    assert sides[:3] == [sides[0]] * 3  # a team may keep the ball for three touches
    assert sides[3] != sides[0]  # the fourth has to belong to the other side


def test_first_two_touches_after_serve_prefer_the_receiving_side() -> None:
    samples = [
        BallSample(frame=frame, x=430.0, y=300.0, radius_px=8.0, confidence=0.9)
        for frame in range(20, 61)
    ]
    choices = [
        ContactChoice(best=make_contact(10, 9, "far", CONTACT_SERVE), alternatives=()),
        ContactChoice(
            best=make_contact(30, 10, "far", reach_ratio=0.20),
            alternatives=(make_contact(30, 1, "near", reach_ratio=0.30),),
        ),
        ContactChoice(
            best=make_contact(50, 11, "far", reach_ratio=0.22),
            alternatives=(make_contact(50, 2, "near", reach_ratio=0.32),),
        ),
    ]
    machine = RallyContactStateMachine(FakeCourt(), FPS, samples, make_serve_start())

    contacts = machine.resolve(choices)

    assert [contact.side for contact in contacts] == ["far", "near", "near"]
    assert [event["selection_reason"] for event in machine.transitions[1:]] == [
        "post_serve_receiver",
        "post_serve_receiver",
    ]
    assert machine.snapshot()["phase"] == "possession"


def test_ball_side_vote_ignores_low_confidence_radius_jump() -> None:
    samples = [
        BallSample(frame=frame, x=430.0, y=300.0, radius_px=8.0, confidence=0.9)
        for frame in range(17, 24)
    ]
    samples.append(
        BallSample(frame=20, x=900.0, y=300.0, radius_px=80.0, confidence=0.05)
    )
    choice = ContactChoice(
        best=make_contact(20, 10, "far", reach_ratio=0.20),
        alternatives=(make_contact(20, 1, "near", reach_ratio=0.30),),
    )
    machine = RallyContactStateMachine(FakeCourt(), FPS, samples, start=None)

    contacts = machine.resolve([choice])

    assert contacts[0].side == "near"
    assert machine.ball_side == "near"
    assert machine.ball_side_probability > 0.9
    assert machine.transitions[0]["selection_reason"] == "stable_ball_side"


def test_summary_counts_side_exchanges(tmp_path) -> None:
    detector = make_detector(tmp_path)
    samples = bounce_samples(BOX_TOP + 0.5 * BOX_HEIGHT)
    analysis = detector.analyze(samples)

    summary = summarize_contacts(analysis.contacts, FPS, analysis.start, analysis.end)

    assert summary["contact_count"] == 1
    assert summary["dig_count"] == 1
    assert summary["side_switch_count"] == 0
    assert summary["start_origin"] == "in_court"


def test_ball_ending_at_a_player_inside_the_court_is_reported(tmp_path) -> None:
    detector = make_detector(tmp_path)

    # The ball flies down into the player and stays there: a caught ball.
    caught = [BallSample(frame=frame, x=600.0, y=200.0 + 8.0 * frame) for frame in range(10, 25)]
    end = detector.detect_end(caught)

    assert end.at_player_in_court
    assert end.player_track_id == 7


def test_disabled_store_yields_no_contacts() -> None:
    detector = ContactDetector(PlayerStore(None), FakeCourt(), FPS)

    assert not detector.enabled
    assert detector.detect(bounce_samples(350.0)) == []
