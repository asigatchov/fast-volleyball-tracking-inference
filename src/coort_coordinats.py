"""Compatibility wrapper for the corrected :mod:`court_coordinates` command."""

from court_coordinates import (
    annotate_video,
    build_parser,
    load_existing_annotation,
    main,
    make_legacy_output_path,
    make_output_path,
    save_annotation,
    scale_initial_keypoints,
)

__all__ = [
    "annotate_video",
    "build_parser",
    "load_existing_annotation",
    "main",
    "make_legacy_output_path",
    "make_output_path",
    "save_annotation",
    "scale_initial_keypoints",
]


if __name__ == "__main__":
    main()
