import json

import numpy as np
import pytest

import make_reels
from make_reels import (
    apply_black_fade,
    black_fade_alpha,
    build_parser,
    clip_frame_range,
    crop_frame,
    crop_and_save_track_payloads,
    load_single_track,
    load_track_from_payload,
    scale_track_to_video,
    tracking_frame_size,
)


def proxy_track_payload() -> dict:
    return {
        "start_frame": 10,
        "last_frame": 12,
        "positions": [
            [[320.0, 180.0], 10],
            [[640.0, 360.0], 12],
        ],
        "tracking_scale": {
            "reference_width_px": 1920,
            "frame_width_px": 1280,
            "frame_height_px": 720,
            "frame_width_coeff": 2 / 3,
        },
    }


def test_proxy_coordinates_are_scaled_to_full_hd() -> None:
    track = load_track_from_payload(proxy_track_payload())

    scaled = scale_track_to_video(track, 1920, 1080)

    assert scaled["positions"] == [
        (480.0, 270.0, 10),
        (960.0, 540.0, 12),
    ]


def test_proxy_coordinates_are_scaled_independently_per_axis() -> None:
    track = load_track_from_payload(proxy_track_payload())

    scaled = scale_track_to_video(track, 2560, 1080)

    assert scaled["positions"][1] == pytest.approx((1280.0, 540.0, 12))


def test_old_tracking_scale_uses_court_height() -> None:
    payload = proxy_track_payload()
    del payload["tracking_scale"]["frame_height_px"]
    payload["court_info"] = {"image_width": 1280, "image_height": 720}

    assert tracking_frame_size(payload) == (1280, 720)


def test_track_without_source_size_keeps_original_coordinates() -> None:
    payload = proxy_track_payload()
    del payload["tracking_scale"]
    track = load_track_from_payload(payload)

    assert scale_track_to_video(track, 1920, 1080) is track


def test_single_track_loader_preserves_tracking_scale(tmp_path) -> None:
    path = tmp_path / "track_0001.json"
    path.write_text(json.dumps(proxy_track_payload()), encoding="utf-8")

    track = load_single_track(str(path))

    assert track["tracking_frame_size"] == (1280, 720)


def test_black_transition_has_03_second_fades_at_30_fps() -> None:
    assert black_fade_alpha(0, 60, 30.0, fade_in=True, fade_out=False) == 0.0
    assert black_fade_alpha(8, 60, 30.0, fade_in=True, fade_out=False) == 1.0
    assert black_fade_alpha(51, 60, 30.0, fade_in=False, fade_out=True) == 1.0
    assert black_fade_alpha(59, 60, 30.0, fade_in=False, fade_out=True) == 0.0


def test_black_fade_produces_a_black_boundary_frame() -> None:
    frame = np.full((4, 4, 3), 200, dtype=np.uint8)

    faded = apply_black_fade(frame, 0, 60, 30.0, fade_in=True, fade_out=False)

    assert np.count_nonzero(faded) == 0


def test_multiple_payloads_get_fades_only_at_internal_boundaries(
    tmp_path, monkeypatch
) -> None:
    calls = []

    def fake_crop(**kwargs):
        calls.append((kwargs["fade_in"], kwargs["fade_out"]))

    monkeypatch.setattr(make_reels, "crop_and_save_track_payload", fake_crop)
    monkeypatch.setattr(make_reels.subprocess, "run", lambda *args, **kwargs: None)

    crop_and_save_track_payloads(
        "video.mp4",
        [proxy_track_payload(), proxy_track_payload(), proxy_track_payload()],
        str(tmp_path / "combined.mp4"),
        fade_transition=True,
    )

    assert calls == [(False, True), (True, True), (True, False)]


def test_cli_fade_transition_flag() -> None:
    args = build_parser().parse_args(
        [
            "--video_path",
            "video.mp4",
            "--track_jsons",
            "track_1.json",
            "track_2.json",
            "--fade-transition",
        ]
    )

    assert args.fade_transition is True


def test_crop_near_the_edge_stays_inside_the_frame() -> None:
    frame = np.arange(100, dtype=np.uint8).reshape(1, 100, 1).repeat(3, axis=2)

    assert crop_frame(frame, 5, 20)[0, :, 0].tolist() == list(range(0, 20))
    assert crop_frame(frame, 98, 20)[0, :, 0].tolist() == list(range(80, 100))


def test_padding_adds_seconds_around_the_rally_within_video_bounds() -> None:
    assert clip_frame_range(100, 200, 1000, fps=30.0, padding=0.33) == (90, 210)
    assert clip_frame_range(5, 995, 1000, fps=30.0, padding=1.0) == (0, 999)


def test_cli_padding_is_seconds_with_033_default() -> None:
    assert build_parser().parse_args(["--video_path", "v.mp4"]).padding == pytest.approx(0.33)
    args = build_parser().parse_args(["--video_path", "v.mp4", "--padding", "1.5"])
    assert args.padding == pytest.approx(1.5)
