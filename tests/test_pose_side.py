import unittest

import numpy as np

from pose_side import (
    ANKLE_THR,
    L_ANKLE,
    L_ELBOW,
    L_HIP,
    L_SHOULDER,
    L_WRIST,
    R_ANKLE,
    R_ELBOW,
    R_HIP,
    R_SHOULDER,
    R_WRIST,
    Person,
    resolve_contact,
    window_frames,
)

NET_Y = 600.0


class FakeCourt:
    """Camera behind the end line: the net is the row y = 600, 50 px a metre, near below it."""

    def to_court(self, x: float, y: float) -> tuple[float, float]:
        return (NET_Y - y) / 50.0, x / 50.0


def person(frame: int, x: float, feet_y: float, hand: tuple[float, float] | None = None, height: float = 300.0) -> Person:
    """An upright player standing at (x, feet_y), one hand at ``hand`` (else both hands at the hips)."""
    keypoints = np.zeros((17, 2))
    top = feet_y - height
    keypoints[L_SHOULDER] = (x - 30, top + 0.2 * height)
    keypoints[R_SHOULDER] = (x + 30, top + 0.2 * height)
    keypoints[L_HIP] = (x - 20, top + 0.5 * height)
    keypoints[R_HIP] = (x + 20, top + 0.5 * height)
    keypoints[L_ANKLE] = (x - 15, feet_y)
    keypoints[R_ANKLE] = (x + 15, feet_y)
    keypoints[L_ELBOW] = (x - 40, top + 0.4 * height)
    keypoints[L_WRIST] = (x - 40, top + 0.55 * height)
    keypoints[R_ELBOW] = (x + 40, top + 0.4 * height)
    if hand is None:
        keypoints[R_WRIST] = (x + 40, top + 0.55 * height)
    else:
        keypoints[R_ELBOW] = (hand[0] - 10, hand[1] + 40)
        keypoints[R_WRIST] = (hand[0], hand[1] + 15)
    box = np.array([x - 60, top, x + 60, feet_y + 5])
    return Person(frame=frame, box=box, keypoints=keypoints, scores=np.ones(17))


class PoseSideTest(unittest.TestCase):
    def test_window_is_every_step_before_the_contact_and_the_touch(self):
        self.assertEqual(window_frames(100, 15, 3), [85, 88, 91, 94, 97, 98, 100, 102])

    def test_a_jumping_attacker_at_the_net_keeps_his_half(self):
        # Near attacker: on the floor at y 650 (1 m before the net), then jumps: at the hit
        # his feet are at y 560, which projects onto the far half.
        ball = (500.0, 300.0)
        people = {
            85: [person(85, 500, 650)],
            88: [person(88, 500, 648)],
            91: [person(91, 500, 640)],
            94: [person(94, 500, 610)],
            97: [person(97, 500, 580)],
            98: [person(98, 500, 570, hand=ball)],
            100: [person(100, 500, 560, hand=ball)],
            102: [person(102, 500, 565)],
        }
        contact = {"frame": 100, "ball_x": ball[0], "ball_y": ball[1], "player_track_id": 7, "side": "far"}
        players = {100: [(7, people[100][0].box)]}
        evidence = resolve_contact(contact, people, players, {}, FakeCourt(), ("near", "far"), window_frames(100, 15, 3))
        self.assertIsNotNone(evidence)
        self.assertEqual(evidence["side"], "near")
        self.assertEqual(evidence["feet_frame"], 85)
        self.assertEqual(evidence["toucher"], "detector")

    def test_the_hand_on_the_ball_beats_the_detectors_blocker(self):
        # The detector names a far blocker (feet at y 520) whose box the ball crosses; the
        # near player (feet at y 700) has his hand on it.
        ball = (500.0, 420.0)
        frames = window_frames(100, 15, 3)
        people = {
            frame: [person(frame, 500, 700, hand=ball if frame >= 98 else None), person(frame, 700, 520, height=200)]
            for frame in frames
        }
        players = {100: [(3, people[100][1].box), (9, people[100][0].box)]}
        contact = {"frame": 100, "ball_x": ball[0], "ball_y": ball[1], "player_track_id": 3, "side": "far"}
        evidence = resolve_contact(contact, people, players, {}, FakeCourt(), ("near", "far"), frames)
        self.assertEqual(evidence["toucher"], "pose")
        self.assertEqual(evidence["track_id"], 9)
        self.assertEqual(evidence["side"], "near")

    def test_feet_on_the_net_line_decide_nothing(self):
        ball = (500.0, 300.0)
        frames = window_frames(100, 15, 3)
        people = {frame: [person(frame, 500, NET_Y + 2, hand=ball)] for frame in frames}
        for group in people.values():
            group[0].scores[[L_ANKLE, R_ANKLE]] = ANKLE_THR
        contact = {"frame": 100, "ball_x": ball[0], "ball_y": ball[1]}
        evidence = resolve_contact(contact, people, {}, {}, FakeCourt(), ("near", "far"), frames)
        self.assertNotIn("side", evidence)


if __name__ == "__main__":
    unittest.main()
