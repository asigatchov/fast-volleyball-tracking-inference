import json

from court_coordinates import (
    build_parser,
    load_existing_annotation,
    make_legacy_output_path,
    make_output_path,
    scale_initial_keypoints,
)


def test_loads_keypoints_and_frame_from_json(tmp_path) -> None:
    annotation = tmp_path / "court.json"
    annotation.write_text(
        json.dumps(
            {
                "frame_index": 42,
                "frame_width": 1280,
                "frame_height": 720,
                "keypoints": [
                    {
                        "id": 0,
                        "name": "1_back_left",
                        "x": 100,
                        "y": 200,
                        "visible": True,
                    },
                    {
                        "id": 1,
                        "name": "2_back_left",
                        "x": None,
                        "y": None,
                        "visible": False,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    points, frame_index = load_existing_annotation(annotation, 1280, 720)

    assert frame_index == 42
    assert points[0]["x"] == 100
    assert points[0]["y"] == 200
    assert points[0]["visible"] is True
    assert points[1]["x"] is None
    assert points[1]["y"] is None
    assert points[1]["visible"] is False


def test_loaded_keypoints_are_scaled_to_video_size(tmp_path) -> None:
    annotation = tmp_path / "court.json"
    annotation.write_text(
        json.dumps(
            {
                "frame_width": 640,
                "frame_height": 360,
                "keypoints": [
                    {
                        "name": "1_back_left",
                        "x": 100,
                        "y": 50,
                        "visible": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    points, _ = load_existing_annotation(annotation, 1280, 720)

    assert (points[0]["x"], points[0]["y"]) == (200, 100)


def test_missing_annotation_uses_default_keypoints(tmp_path) -> None:
    points, frame_index = load_existing_annotation(tmp_path / "missing.json", 1280, 720)

    assert points == scale_initial_keypoints(1280, 720)
    assert frame_index == 0


def test_output_name_uses_correct_court_spelling(tmp_path) -> None:
    video = tmp_path / "clip.LRF"

    assert make_output_path(video, tmp_path).name == "clip_court.json"
    assert make_legacy_output_path(video, tmp_path).name == "clip_coort.json"


def test_cli_accepts_json_court_and_both_output_dir_spellings(tmp_path) -> None:
    parser = build_parser()
    court = tmp_path / "court.json"

    dashed = parser.parse_args(
        ["video.LRF", "--json-court", str(court), "--output-dir", str(tmp_path)]
    )
    underscored = parser.parse_args(["video.LRF", "--output_dir", str(tmp_path)])

    assert dashed.json_court == court
    assert dashed.output_dir == tmp_path
    assert underscored.output_dir == tmp_path
