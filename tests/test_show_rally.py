import json

import cv2
import numpy as np
import pytest

from player_contacts import PlayerStore
from show_rally import JUMP_FRAMES, RallyViewer, discover_auxiliary_json, load_tracks

FPS = 30.0
WIDTH, HEIGHT = 320, 180
FRAME_COUNT = 60


def make_track_json(track_id: int, first_frame: int, last_frame: int) -> dict:
    payload = {
        "track_id": track_id,
        "frame_observations": [
            {"frame": frame, "x": 100.0 + frame, "y": 80.0, "radius_px": 8.0, "radius_med6_px": 8.0}
            for frame in range(first_frame, last_frame + 1)
        ],
        "rally_classification": {"label": "rally", "is_rally": True, "rally_confidence": 0.8},
        "rally_features": {"duration_sec": (last_frame - first_frame + 1) / FPS},
        "trajectory_analysis": {"serve_side": "near", "serve_side_source": "player_serve_zone_start"},
        "court_info": {"image_width": WIDTH, "image_height": HEIGHT},
        "player_interaction": {
            "enabled": True,
            "ball_start": {
                "frame": first_frame,
                "origin": "serve_zone",
                "player_track_id": 3,
                "ball_x": 100.0 + first_frame,
                "ball_y": 80.0,
            },
            "contacts": [
                {
                    "frame": first_frame + 2,
                    "player_track_id": 3,
                    "contact_type": "dig",
                    "depth_ratio": 0.5,
                    "confidence": 0.9,
                    "side": "near",
                    "ball_x": 102.0 + first_frame,
                    "ball_y": 80.0,
                }
            ],
            "summary": {"contact_count": 1, "dig_count": 1, "start_origin": "serve_zone"},
        },
    }
    payload["court_view"] = {
        "coordinate_system": "court_metres",
        "court_length_m": 16.0,
        "court_width_m": 8.0,
        "frames": [
            {
                "frame": frame,
                "ball": {"court_x": -2.0 + (frame - first_frame) * 0.1, "court_y": 0.5},
                "ball_direction": {"dx": 0.1, "dy": 0.0},
                "players": [
                    {"track_id": 3, "court_x": -3.0, "court_y": 1.0, "score": 0.9},
                    {"track_id": 4, "court_x": 3.0, "court_y": -1.0, "score": 0.9},
                ],
            }
            for frame in range(first_frame, last_frame + 1)
        ],
    }
    return payload


@pytest.fixture
def tracks_dir(tmp_path):
    directory = tmp_path / "tracks"
    directory.mkdir()
    for index, (first, last) in enumerate(((5, 15), (30, 44))):
        (directory / f"track_{index:04d}.json").write_text(
            json.dumps(make_track_json(index, first, last)), encoding="utf-8"
        )
    return str(directory)


@pytest.fixture
def video_file(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (WIDTH, HEIGHT))
    for index in range(FRAME_COUNT):
        frame = np.full((HEIGHT, WIDTH, 3), index * 3 % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return str(path)


def make_viewer(tracks_dir, video_file):
    return RallyViewer(load_tracks(tracks_dir), video_file, PlayerStore(None), FPS)


def test_tracks_are_loaded_in_playback_order(tracks_dir) -> None:
    tracks = load_tracks(tracks_dir)

    assert [track.first_frame for track in tracks] == [5, 30]
    assert tracks[0].contacts[0]["contact_type"] == "dig"
    assert tracks[0].ball_start["origin"] == "serve_zone"


def test_render_includes_video_and_box_view(tracks_dir, video_file) -> None:
    viewer = make_viewer(tracks_dir, video_file)
    try:
        canvas = viewer.render()
    finally:
        viewer.close()

    assert canvas.shape[1] == WIDTH * 2
    assert canvas.shape[0] > HEIGHT  # header and timeline are stacked around the frame


def test_top_down_maps_far_side_above_near_side(tracks_dir, video_file) -> None:
    viewer = make_viewer(tracks_dir, video_file)
    try:
        near = viewer._court_to_top_down(-4.0, 0.0, (HEIGHT, WIDTH, 3))
        far = viewer._court_to_top_down(4.0, 0.0, (HEIGHT, WIDTH, 3))
    finally:
        viewer.close()

    assert far[1] < near[1]


def test_b_toggles_the_box_view_panel(tracks_dir, video_file) -> None:
    viewer = make_viewer(tracks_dir, video_file)
    try:
        assert viewer.render().shape[1] == WIDTH * 2
        viewer._handle_key(ord("b"))
        assert viewer.render().shape[1] == WIDTH
        viewer._handle_key(ord("b"))
        assert viewer.render().shape[1] == WIDTH * 2
    finally:
        viewer.close()


def test_h_switches_to_a_single_video_panel(tracks_dir, video_file) -> None:
    viewer = make_viewer(tracks_dir, video_file)
    try:
        viewer._handle_key(ord("h"))
        assert viewer.render().shape[1] == WIDTH
        viewer._handle_key(ord("H"))
        assert viewer.render().shape[1] == WIDTH * 2
    finally:
        viewer.close()


def test_auxiliary_json_is_discovered_next_to_clip_tracks(tmp_path) -> None:
    clip_dir = tmp_path / "clip_001"
    tracks = clip_dir / "tracks"
    tracks.mkdir(parents=True)
    players = tmp_path / "clip_001_predict.json"
    court = tmp_path / "clip_001_coort.json"
    players.write_text("{}", encoding="utf-8")
    court.write_text("{}", encoding="utf-8")

    assert discover_auxiliary_json(str(tracks)) == (str(players), str(court))


def test_stepping_keys_move_by_one_and_fifteen_frames(tracks_dir, video_file) -> None:
    viewer = make_viewer(tracks_dir, video_file)
    try:
        viewer.select(0)
        start = viewer._frame

        viewer._handle_key(ord("d"))
        assert viewer._frame == start + 1
        viewer._handle_key(ord("a"))
        assert viewer._frame == start
        viewer._handle_key(ord("w"))
        assert viewer._frame == start + JUMP_FRAMES
        viewer._handle_key(ord("s"))
        assert viewer._frame == start
    finally:
        viewer.close()


def test_seeking_is_clamped_to_the_track_context(tracks_dir, video_file) -> None:
    viewer = make_viewer(tracks_dir, video_file)
    try:
        viewer.select(0)
        for _ in range(50):
            viewer._handle_key(ord("a"))

        assert viewer._frame == viewer._first_visible_frame()
        assert viewer._frame >= 0
    finally:
        viewer.close()


def test_track_navigation_clamps_at_the_ends(tracks_dir, video_file) -> None:
    viewer = make_viewer(tracks_dir, video_file)
    try:
        viewer._handle_key(ord("n"))
        assert viewer.track.first_frame == 30
        assert viewer._frame == 30

        viewer._handle_key(ord("n"))  # already the last one
        assert viewer.track.first_frame == 30

        viewer._handle_key(ord("p"))
        assert viewer.track.first_frame == 5
    finally:
        viewer.close()


def test_space_toggles_playback_and_q_stops_the_loop(tracks_dir, video_file) -> None:
    viewer = make_viewer(tracks_dir, video_file)
    try:
        assert viewer._handle_key(ord(" ")) is True
        assert viewer._playing is True
        assert viewer._handle_key(ord("d")) is True
        assert viewer._playing is False  # stepping pauses playback
        assert viewer._handle_key(ord("q")) is False
    finally:
        viewer.close()


def test_track_without_player_data_still_renders(tmp_path, video_file) -> None:
    directory = tmp_path / "plain"
    directory.mkdir()
    data = make_track_json(0, 5, 15)
    del data["player_interaction"]
    (directory / "track_0000.json").write_text(json.dumps(data), encoding="utf-8")

    viewer = RallyViewer(load_tracks(str(directory)), video_file, PlayerStore(None), FPS)
    try:
        assert viewer.track.contacts == []
        assert viewer.track.ball_start is None
        assert viewer.render() is not None
    finally:
        viewer.close()
