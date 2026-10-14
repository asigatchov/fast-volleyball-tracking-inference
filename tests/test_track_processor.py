import json

from track_processor import TrackProcessor, build_parser


def write_track(directory, track_id, classification=None) -> None:
    payload = {
        "positions": [[[100.0, 80.0], track_id * 10]],
        "prediction": [100.0, 80.0],
        "start_frame": track_id * 10,
        "last_frame": track_id * 10 + 5,
        "ball_sizes": [8.0],
        "track_id": track_id,
    }
    if classification is not None:
        payload["rally_classification"] = classification
    (directory / f"track_{track_id:04d}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_only_rally_tracks_are_loaded_by_default(tmp_path) -> None:
    write_track(tmp_path, 1, {"label": "rally", "is_rally": True})
    write_track(tmp_path, 2, {"label": "not_rally", "is_rally": False})
    write_track(tmp_path, 3)
    processor = TrackProcessor(str(tmp_path), "unused.mp4")

    processor._load_tracks_from_json()

    assert [item.track_id for item in processor.tracks] == [1]


def test_rally_label_is_used_when_boolean_is_missing(tmp_path) -> None:
    write_track(tmp_path, 1, {"label": "rally"})
    processor = TrackProcessor(str(tmp_path), "unused.mp4")

    processor._load_tracks_from_json()

    assert [item.track_id for item in processor.tracks] == [1]


def test_include_not_rally_restores_all_tracks(tmp_path) -> None:
    write_track(tmp_path, 1, {"label": "rally", "is_rally": True})
    write_track(tmp_path, 2, {"label": "not_rally", "is_rally": False})
    write_track(tmp_path, 3)
    processor = TrackProcessor(
        str(tmp_path), "unused.mp4", include_not_rally=True
    )

    processor._load_tracks_from_json()

    assert [item.track_id for item in processor.tracks] == [1, 2, 3]


def test_cli_include_not_rally_flag() -> None:
    args = build_parser().parse_args(
        ["--video_path", "video.mp4", "--json_dir", "tracks", "--include-not-rally"]
    )

    assert args.include_not_rally is True


def test_cli_no_mark_flag() -> None:
    args = build_parser().parse_args(
        ["--video_path", "video.mp4", "--json_dir", "tracks", "--no-mark"]
    )

    assert args.no_mark is True


def test_cli_padding_is_float() -> None:
    args = build_parser().parse_args(
        ["--video_path", "video.mp4", "--json_dir", "tracks", "--padding", "1.5"]
    )

    assert args.padding == 1.5


def test_clip_has_ten_context_frames_on_both_sides() -> None:
    assert TrackProcessor._clip_frame_range(100, 200, 1000) == (90, 210)


def test_clip_context_is_clamped_to_video_bounds() -> None:
    assert TrackProcessor._clip_frame_range(5, 20, 25) == (0, 24)


def test_clip_padding_is_converted_from_seconds_using_video_fps() -> None:
    assert TrackProcessor._clip_frame_range(
        100, 200, 1000, fps=25.0, padding=2.0
    ) == (50, 250)


def test_negative_padding_is_rejected(tmp_path) -> None:
    try:
        TrackProcessor(str(tmp_path), "unused.mp4", padding=-0.1)
    except ValueError as exc:
        assert str(exc) == "padding must be greater than or equal to 0"
    else:
        raise AssertionError("negative padding must be rejected")
