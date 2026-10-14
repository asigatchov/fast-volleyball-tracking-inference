import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.classify_contacts import classify_tracks, crop_plan, load_boxes, read_clips

WIDTH, HEIGHT, FRAMES = 320, 240, 30


class FakeClassifier:
    """Stands in for the OpenVINO model: the brighter the clip, the surer the Attack."""

    class_names = ["Serve", "Receive", "Set", "Attack", "NoAction"]
    half_window = 3
    stored_size = 32
    crop_scale = 1.5
    crop = None

    def __call__(self, clips: np.ndarray) -> np.ndarray:
        attack = clips.reshape(len(clips), -1).mean(axis=1) / 255.0
        rows = np.zeros((len(clips), 5), dtype=np.float32)
        rows[:, 3] = attack
        rows[:, 4] = 1.0 - attack
        return rows


def prediction(frame: int, name: str, box: tuple[float, float, float, float], track_id: int | None = None) -> dict:
    return {"frame_index": frame, "class_name": name, "bbox_xyxy": list(box), "track_id": track_id}


class ClassifyContactsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.video = str(root / "video.mp4")
        writer = cv2.VideoWriter(self.video, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (WIDTH, HEIGHT))
        for index in range(FRAMES):
            # The left half is white from frame 10 on, the right half stays black.
            image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            if index >= 10:
                image[:, : WIDTH // 2] = 255
            writer.write(image)
        writer.release()

        predictions = []
        for frame in range(0, FRAMES, 2):
            predictions.append(prediction(frame, "player", (40, 80, 80, 180), track_id=1))  # left player
            predictions.append(prediction(frame, "player", (240, 80, 280, 180), track_id=2))  # right player
        predictions.append(prediction(20, "ball", (70, 70, 80, 80)))  # at the left player
        self.predictions = str(root / "predictions.json")
        Path(self.predictions).write_text(json.dumps({"predictions": predictions}))

        self.tracks = root / "tracks"
        self.tracks.mkdir()
        rally = {
            "rally_classification": {"is_rally": True},
            "player_interaction": {
                "contacts": [
                    {"frame": 20},  # ball box next to the left player
                    {"frame": 24, "ball_x": 250.0, "ball_y": 90.0},  # no ball box nearby: the contact position is used
                    {"frame": 21},  # odd frame: the detector saw no players there
                ]
            },
        }
        (self.tracks / "track_0001.json").write_text(json.dumps(rally))
        not_rally = {"rally_classification": {"is_rally": False}, "player_interaction": {"contacts": [{"frame": 20}]}}
        (self.tracks / "track_0002.json").write_text(json.dumps(not_rally))

    def tearDown(self):
        self.tmp.cleanup()

    def test_crop_is_built_around_the_player_closest_to_the_ball(self):
        boxes = load_boxes(Path(self.predictions), WIDTH, HEIGHT)
        x0, y0, side = crop_plan(boxes, {"frame": 20}, 3, WIDTH, HEIGHT, 1.5)
        # Player 40..80 x 80..180 with the ball at 70..80 x 70..80: 110 px tall, scaled by 1.5.
        self.assertEqual(side, 165)
        self.assertAlmostEqual(x0 + side / 2, 60, delta=1)
        self.assertAlmostEqual(y0 + side / 2, 125, delta=1)
        self.assertIsNone(crop_plan(boxes, {"frame": 21}, 3, WIDTH, HEIGHT, 1.5))
        self.assertIsNone(crop_plan(boxes, {"frame": 2}, 3, WIDTH, HEIGHT, 1.5))  # no ball anywhere near

    def test_player_crop_depends_on_player_height_only(self):
        boxes = load_boxes(Path(self.predictions), WIDTH, HEIGHT)
        crop = {"mode": "player", "scale": 2.4, "lift": 0.3, "view": 0.85}
        x0, y0, side = crop_plan(boxes, {"frame": 20}, 3, WIDTH, HEIGHT, 1.5, crop=crop)
        # Player 40..80 x 80..180: 100 px tall, centre (60, 130); the ball box does not change the window.
        self.assertEqual(side, round(2.4 * 100 / 0.85))
        self.assertAlmostEqual(x0 + side / 2, 60, delta=0.5)
        self.assertAlmostEqual(y0 + side / 2, 130 - 30, delta=0.5)

    def test_player_crop_takes_the_player_named_by_the_contact(self):
        boxes = load_boxes(Path(self.predictions), WIDTH, HEIGHT)
        crop = {"mode": "player", "scale": 2.4, "lift": 0.3, "view": 0.85}
        # The ball box is at the left player, but the contact detector named the right one (240..280).
        x0, _, side = crop_plan(boxes, {"frame": 20, "player_track_id": 2}, 3, WIDTH, HEIGHT, 1.5, crop=crop)
        self.assertAlmostEqual(x0 + side / 2, 260, delta=0.5)
        # An unknown track falls back to the ball; no ball and no track gives nothing to crop.
        x0, _, side = crop_plan(boxes, {"frame": 20, "player_track_id": 9}, 3, WIDTH, HEIGHT, 1.5, crop=crop)
        self.assertAlmostEqual(x0 + side / 2, 60, delta=0.5)
        self.assertIsNone(crop_plan(boxes, {"frame": 2}, 3, WIDTH, HEIGHT, 1.5, crop=crop))
        # A named player is enough even where the ball was not detected.
        self.assertIsNotNone(crop_plan(boxes, {"frame": 2, "player_track_id": 1}, 3, WIDTH, HEIGHT, 1.5, crop=crop))

    def test_ball_goes_to_the_big_player_not_to_a_small_one_behind(self):
        from src.classify_contacts import contact_player

        # A far player (20 px tall) stands in the picture right at the ball above the server's head.
        boxes = {5: {"player": [(0.5, 0.6, 0.1, 0.5), (0.52, 0.3, 0.02, 0.08)], "ball": [], "player_tracks": [1, 2]}}
        ball = (0.52, 0.32, 0.02, 0.03)
        self.assertEqual(contact_player(boxes, {"frame": 5}, ball, WIDTH, HEIGHT), (0.52, 0.3, 0.02, 0.08))
        ball_above_server = (0.5, 0.33, 0.02, 0.03)
        self.assertEqual(contact_player(boxes, {"frame": 5}, ball_above_server, WIDTH, HEIGHT), (0.5, 0.6, 0.1, 0.5))

    def test_clip_window_repeats_the_edge_frames(self):
        clips = read_clips(self.video, {1: (0, 0, 100), 28: (0, 0, 100)}, 3, 32)
        self.assertEqual(clips[1].shape, (7, 32, 32, 3))
        self.assertEqual(int(clips[1].max()), 0)  # frames -2..4 are all black
        self.assertGreater(int(clips[28].min()), 200)  # frames 25..31, the tail repeats frame 29

    def test_actions_are_written_into_rally_contacts(self):
        counts = classify_tracks(self.video, self.predictions, str(self.tracks), FakeClassifier())
        self.assertEqual(counts, {"Attack": 1, "NoAction": 1, "unclassified": 1})

        contacts = json.loads((self.tracks / "track_0001.json").read_text())["player_interaction"]["contacts"]
        self.assertEqual(contacts[0]["action"]["type"], "Attack")  # left player, white half
        self.assertGreater(contacts[0]["action"]["score"], 0.6)
        self.assertEqual(set(contacts[0]["action"]["probabilities"]), set(FakeClassifier.class_names))
        self.assertEqual(contacts[1]["action"]["type"], "NoAction")  # right player, black half
        self.assertNotIn("action", contacts[2])

        untouched = json.loads((self.tracks / "track_0002.json").read_text())
        self.assertNotIn("action", untouched["player_interaction"]["contacts"][0])


if __name__ == "__main__":
    unittest.main()
