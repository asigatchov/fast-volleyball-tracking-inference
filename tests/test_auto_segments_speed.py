"""Automatic parabola detection: synthetic flights, no video or annotation files."""
import unittest

import numpy as np

from src.ballistic_speed import contact_scores, detect_segments, trajectory
from src.calc_speed import Camera, Detection

FPS = 30.


def make_camera():
    center = np.array([4.5, -20., 8.])
    forward = np.array([4.5, 9., 0.]) - center
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    rotation = np.array([right, np.cross(forward, right), forward])
    return Camera(np.array([[2200., 0, 640], [0, 2200., 360], [0, 0, 1]]), rotation, -rotation @ center, 0.2)


class AutoSegmentTests(unittest.TestCase):
    def setUp(self):
        self.camera = make_camera()
        self.timestamps = np.arange(200) / FPS

    def detections(self, flights, noise=0.3, skip=()):
        """Project each (parameters, first_frame, last_frame) flight the way the detector sees it."""
        rng = np.random.default_rng(11)
        detections = {}
        for parameters, first, last in flights:
            frames = np.arange(first, last + 1)
            points = trajectory(np.asarray(parameters, float), self.timestamps[frames] - self.timestamps[first])
            self.assertGreater(points[:, 2].min(), 0.11, "synthetic flight must stay above the floor")
            pixels = self.camera.pixels(points) + rng.normal(0, noise, (len(frames), 2))
            depths = (points @ self.camera.rotation.T + self.camera.translation)[:, 2]
            radii = 2200 * 0.22 / (2 * depths) * (1 + rng.normal(0, 0.05, len(frames)))
            for frame, pixel, radius in zip(frames, pixels, radii):
                if frame not in skip:
                    detections[int(frame)] = Detection(pixel, float(radius))
        return detections

    def rally(self, contacts=4, span=26):
        """One serve plus `contacts` returns, each flight starting where the previous one ended."""
        parameters, spans, first = [np.array([1., 1., 1.8, 1.2, 8., 4.5])], [(0, span)], 0
        for index in range(contacts):
            first = spans[-1][1] + 1
            previous = trajectory(parameters[-1], self.timestamps[first] - self.timestamps[spans[-1][0]])
            outgoing = [-1.2, -8., 4.5] if index % 2 == 0 else [1.2, 8., 4.5]
            parameters.append(np.r_[previous, outgoing])
            spans.append((first, first + span))
        return [(p, a, b) for p, (a, b) in zip(parameters, spans)], spans

    def test_smooth_flight_stays_one_parabola(self):
        serve = [3., 1., 2., 0.5, 9., 4.5]

        segments = detect_segments(self.camera, self.detections([(serve, 0, 27)]), self.timestamps)

        self.assertEqual(segments, [(0, 27)])

    def test_apex_of_a_vertical_toss_is_not_taken_for_a_contact(self):
        # The apparent direction reverses completely at the top, yet nobody touched
        # the ball: the fit must keep the two halves together.
        toss = [4., 3., 1.2, 0.05, 0.2, 3.5]

        segments = detect_segments(self.camera, self.detections([(toss, 0, 24)]), self.timestamps)

        self.assertEqual(segments, [(0, 24)])

    def test_contact_splits_the_track_and_the_contact_frame_is_excluded(self):
        serve = np.array([3., 1., 2., 0.5, 9., 4.5])
        returned = np.r_[trajectory(serve, 1.0), [-0.5, -9., 5.]]

        segments = detect_segments(
            self.camera, self.detections([(serve, 0, 29), (returned, 31, 57)]), self.timestamps)

        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0][0], 0)
        self.assertEqual(segments[1][1], 57)
        self.assertLessEqual(segments[0][1], 30)
        self.assertGreaterEqual(segments[1][0], 30)

    def test_rally_without_gaps_recovers_every_flight(self):
        flights, spans = self.rally()

        segments = detect_segments(self.camera, self.detections(flights), self.timestamps)

        self.assertEqual(len(segments), len(spans))
        for (start, end), (first, last) in zip(segments, spans):
            self.assertLessEqual(abs(start - first), 2)
            self.assertLessEqual(abs(end - last), 2)

    def test_detection_gap_always_ends_a_flight(self):
        # Nothing rules out a contact behind a long occlusion, so the two visible
        # parts are fitted separately even though they are one parabola.
        serve = [3., 1., 2., 0.5, 9., 4.5]

        segments = detect_segments(
            self.camera, self.detections([(serve, 0, 27)], skip=range(12, 22)), self.timestamps)

        self.assertEqual(segments, [(0, 11), (22, 27)])

    def test_track_shorter_than_a_fit_returns_nothing(self):
        serve = [3., 1., 2., 0.5, 9., 4.5]
        detections = {f: d for f, d in self.detections([(serve, 0, 27)]).items() if f < 5}

        self.assertEqual(detect_segments(self.camera, detections, self.timestamps), [])

    def test_scores_rate_the_contact_above_the_free_flight(self):
        serve = np.array([3., 1., 2., 0.5, 9., 4.5])
        returned = np.r_[trajectory(serve, 1.0), [-0.5, -9., 5.]]
        detections = self.detections([(serve, 0, 29), (returned, 31, 57)])
        frames = np.array(sorted(detections))

        scores = contact_scores(self.timestamps[frames], np.array([detections[f].xy for f in frames]))

        # Free flight also scores above the 1.0 proposal threshold now and then;
        # what the detector relies on is that the contact stands well above it.
        self.assertLessEqual(abs(int(frames[int(np.argmax(scores))]) - 30), 2)
        self.assertLess(max(scores[(frames < 25) | (frames > 36)]), 0.5 * scores.max())


if __name__ == '__main__':
    unittest.main()
