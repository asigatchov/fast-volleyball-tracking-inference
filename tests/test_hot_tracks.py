import json

from hot_tracks import HotTracks

FPS = 10.0  # margin 100 frames, context 600, step 600


def track(start, end):
    return {"start_frame": start, "last_frame": end, "track_id": 99, "positions": []}


class Calculator:
    """Stands in for track_calculator_with_court: the tracks of the whole video, cut to the window."""

    def __init__(self, tracks):
        self.tracks = tracks
        self.windows = []

    def __call__(self, frame_start, frame_end, window_dir):
        self.windows.append((frame_start, frame_end))
        for index, (start, end) in enumerate(self.tracks):
            if frame_start <= start <= frame_end:
                (window_dir / f"track_{index:04d}.json").write_text(json.dumps(track(start, min(end, frame_end))))


def committed(tmp_path):
    rows = []
    for path in sorted((tmp_path / "tracks").glob("track_*.json")):
        content = json.loads(path.read_text())
        rows.append((path.name, content["start_frame"], content["track_id"]))
    return rows


def test_tracks_are_committed_once_they_cannot_change_and_never_twice(tmp_path):
    calculator = Calculator([(100, 300), (700, 880), (950, 1400), (1500, 1700)])
    hot = HotTracks(tmp_path, FPS, calculator)

    assert hot.due(500) is False
    assert hot.due(700) is True
    # Up to 1000: the first two ended more than 100 frames ago; the third is still in the air.
    assert hot.advance(1000) == ["track_0000.json", "track_0001.json"]
    assert hot.state.frontier == 900

    # The third one is still open at 1450, and the fourth starts after it: both wait.
    assert hot.advance(1450) == []
    assert hot.state.frontier == 950

    # A worker that died resumes from the saved state.
    hot = HotTracks(tmp_path, FPS, calculator)
    assert hot.advance(1600) == ["track_0002.json"]
    assert hot.advance(1750, final=True) == ["track_0003.json"]
    assert committed(tmp_path) == [
        ("track_0000.json", 100, 0),
        ("track_0001.json", 700, 1),
        ("track_0002.json", 950, 2),
        ("track_0003.json", 1500, 3),
    ]
    # Each window starts a minute of context before the frontier.
    assert calculator.windows[0] == (0, 1000)
    assert calculator.windows[-1] == (1500 - 600, 1750)


def test_a_new_run_starts_from_nothing(tmp_path):
    hot = HotTracks(tmp_path, FPS, Calculator([(100, 200)]))
    assert hot.advance(500, final=True) == ["track_0000.json"]
    hot.reset()
    assert committed(tmp_path) == []
    assert hot.state.frontier == 0 and hot.state.next_number == 0
