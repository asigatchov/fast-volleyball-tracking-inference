import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.jersey_numbers import decide_number
from src.player_reid import assign_players, cluster, half_of, pick_actor

WIDTH, HEIGHT, FRAMES = 320, 240, 12
COURT = {
    "categories": [{"keypoints": ["1_back_left", "2_back_left", "3_back_right", "4_back_right", "5_center_left", "6_center_right"]}],
    "annotations": [{"keypoints": [0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 100, 2, 320, 100, 2]}],
}
# Two near players told apart by colour, one far player; red stands left, blue right.
RED, BLUE, FAR = (20, 120, 60, 200), (220, 120, 260, 200), (140, 30, 160, 80)


class ColourEncoder:
    """Stands in for OSNet: the mean colour of the crop."""

    def __call__(self, crops):
        vectors = np.stack([c.reshape(-1, 3).mean(0) + 1.0 for c in crops]).astype(np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class PlayerReidTests(unittest.TestCase):
    def test_half_of_uses_the_feet_against_the_centre_line(self):
        line = ((0.0, 100.0), (320.0, 100.0))
        self.assertEqual(half_of(RED, line), "near")
        self.assertEqual(half_of(FAR, line), "far")

    def test_pick_actor_prefers_the_track_then_the_ball(self):
        players = [(1, RED), (2, BLUE)]
        self.assertEqual(pick_actor(players, 2, [30.0, 130.0]), 1)
        self.assertEqual(pick_actor(players, None, [30.0, 130.0]), 0)
        self.assertIsNone(pick_actor(players, None, None))

    def test_cluster_keeps_people_of_one_moment_apart(self):
        a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
        vectors = np.stack([a, b, a, b, a * 0.9 + b * 0.1]).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        labels, margin = cluster(vectors, [[0, 1], [2, 3], [4]], k=2)
        self.assertNotEqual(labels[0], labels[1])
        self.assertEqual(labels[0], labels[2])
        self.assertEqual(labels[0], labels[4])
        self.assertTrue((margin >= 0).all())

    def test_assign_players_groups_by_appearance_and_writes_crops(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = str(root / "video.mp4")
            writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (WIDTH, HEIGHT))
            lines = []
            for frame in range(FRAMES):
                image = np.full((HEIGHT, WIDTH, 3), 128, np.uint8)
                cv2.rectangle(image, RED[:2], RED[2:], (0, 0, 255), -1)
                cv2.rectangle(image, BLUE[:2], BLUE[2:], (255, 0, 0), -1)
                writer.write(image)
                players = [[1, 0.9, *RED], [2, 0.9, *BLUE], [3, 0.9, *FAR]]
                lines.append(json.dumps({"type": "frame", "frame_index": frame, "players": players}))
            writer.release()
            predictions = root / "predictions.jsonl"
            predictions.write_text(json.dumps({"type": "header"}) + "\n" + "\n".join(lines) + "\n")
            actions = [
                {"frame": 2, "track_id": 1},
                {"frame": 5, "track_id": None, "ball": [240.0, 110.0]},
                {"frame": 8, "track_id": 1},
                {"frame": 9, "track_id": 3},  # far side: not ours
            ]
            result = assign_players(video, str(predictions), COURT, actions, root / "players", ColourEncoder(), k=2)
            # No numbers: nobody gets a player, the actions are only grouped by appearance.
            self.assertEqual(result["mode"], "cluster")
            self.assertEqual(result["assignments"], [])
            self.assertEqual(result["players"], [])
            self.assertEqual(result["skipped"], [2, 5, 8, 9])
            groups = result["clusters"]
            self.assertEqual(set(groups), {"2", "5", "8"})
            self.assertEqual(groups["2"], groups["8"])
            self.assertNotEqual(groups["2"], groups["5"])
            self.assertEqual(groups["2"], 1)  # two actions: the first group
            self.assertTrue((root / "players" / "2.jpg").is_file())
            self.assertTrue((root / "players" / "players.json").is_file())
            saved = np.load(root / "players" / "vectors.npz")
            self.assertEqual(list(saved["frames"]), [2, 5, 8])
            self.assertEqual(saved["vectors"].shape, (3, 3))

            # The numbers on the shirts are read: the players are those numbers, an action whose
            # number was not read stays without a player.
            result = assign_players(video, str(predictions), COURT, actions, root / "numbers", ColourEncoder(), k=2,
                                    numbers=FakeNumbers({RED: "17", BLUE: None}), window=2, step=1, min_numbered=1)
            self.assertEqual(result["mode"], "number")
            self.assertEqual([(a["frame"], a["number"], a["source"]) for a in result["assignments"]], [(2, "17", "number"), (8, "17", "number")])
            self.assertEqual(result["players"], ["17"])
            self.assertEqual(result["skipped"], [5, 9])

            # The team's numbers: a read outside them is dropped, the ones not read are still players.
            result = assign_players(video, str(predictions), COURT, actions, root / "candidates", ColourEncoder(), k=2,
                                    numbers=FakeNumbers({RED: "17", BLUE: "4"}), candidates=["4", "17", "23"], window=2, step=1)
            self.assertEqual({a["frame"]: a["number"] for a in result["assignments"]}, {2: "17", 5: "4", 8: "17"})
            self.assertEqual(result["players"], ["4", "17", "23"])

    def test_the_vector_is_the_mean_over_the_track_around_the_contact(self):
        # The red player turns green on the contact frame only (blur, a turn): his track around it is still red.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = str(root / "video.mp4")
            writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (WIDTH, HEIGHT))
            lines = []
            for frame in range(FRAMES):
                image = np.full((HEIGHT, WIDTH, 3), 128, np.uint8)
                cv2.rectangle(image, RED[:2], RED[2:], (0, 255, 0) if frame == 6 else (0, 0, 255), -1)
                writer.write(image)
                # Track 2 stands on track 1 at frame 8: that view would mix them, it is left out.
                players = [[1, 0.9, *RED]] + ([[2, 0.9, *RED]] if frame == 8 else [])
                lines.append(json.dumps({"type": "frame", "frame_index": frame, "players": players}))
            writer.release()
            predictions = root / "predictions.jsonl"
            predictions.write_text(json.dumps({"type": "header"}) + "\n" + "\n".join(lines) + "\n")
            actions = [{"frame": 6, "track_id": 1}]

            def vector(window):
                assign_players(video, str(predictions), COURT, actions, root / str(window), ColourEncoder(), k=1,
                               appearance_window=window, appearance_step=2)
                return np.load(root / str(window) / "vectors.npz")["vectors"][0].astype(np.float32)

            blue, green, red = vector(0)  # BGR
            self.assertGreater(green, red)
            blue, green, red = vector(4)  # frames 2, 4, 6 and 10: three red views to one green
            self.assertGreater(red, 2 * green)

    def test_a_number_read_on_a_track_goes_to_its_other_actions_in_the_rally(self):
        from src.player_reid import _numbers_by_track

        actors = {
            10: {"track": 5, "rally": "a"},
            20: {"track": 5, "rally": "a"},  # same track, same rally: takes 17
            30: {"track": 5, "rally": "b"},  # another rally: track ids mean nothing there
            40: {"track": 6, "rally": "a"},
            50: {"track": 6, "rally": "a"},
            60: {"track": 6, "rally": "a"},  # its track read as two numbers: unsure
        }
        self.assertEqual(_numbers_by_track(actors, {10: "17", 40: "12", 50: "15"}), {20: "17"})

    def test_decide_number_votes_single_digits_into_the_two_digit_number(self):
        self.assertEqual(decide_number([("17", 0.9), ("1", 0.95), ("7", 0.8)]), ("17", 1.0))
        # A lone "1" is as often half of "17": no number without a two-digit read.
        self.assertEqual(decide_number([("1", 0.99), ("1", 0.97)])[0], None)
        # Unless "1" is one of the team's numbers.
        self.assertEqual(decide_number([("1", 0.99)], candidates={"1", "12"})[0], "1")
        self.assertEqual(decide_number([("COMMUN", 0.9), ("15", 0.4)])[0], None)  # words and unsure reads do not vote
        self.assertEqual(decide_number([("15", 0.9), ("12", 0.9), ("20", 0.9)])[0], None)  # no number has half of the vote


class FakeNumbers:
    """Stands in for jersey_numbers.NumberReader: the 'torso' is the box, its number by box."""

    def __init__(self, by_box):
        self.by_box = by_box

    def torso(self, image, box):
        return np.full((4, 4, 3), list(self.by_box).index(tuple(box)), np.uint8)

    def read(self, torsos):
        return [(self.by_box[list(self.by_box)[int(t[0, 0, 0])]] or "", 0.9) for t in torsos]


if __name__ == "__main__":
    unittest.main()
