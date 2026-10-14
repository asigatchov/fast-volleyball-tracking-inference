import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from src.calc_speed import Camera, Detection
from src.ballistic_speed import fit_flight, trajectory, velocity, frame_times


class FlightTests(unittest.TestCase):
    def setUp(self):
        center = np.array([4.5, -20., 8.])
        forward = np.array([4.5, 9., 0.]) - center
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0, 0, 1])
        right /= np.linalg.norm(right)
        rotation = np.array([right, np.cross(forward, right), forward])
        self.camera = Camera(np.array([[2200., 0, 640], [0, 2200., 360], [0, 0, 1]]), rotation, -rotation @ center, 0.2)
        self.parameters = np.array([3., 2., 2.5, 1.5, 9., 5.])
        # Nonuniform timestamps, so fitting must not assume frame_index / FPS.
        self.times = np.r_[0., np.cumsum(np.tile([0.025, 0.04], 15))]

    def detections(self, noisy=False):
        rng = np.random.default_rng(7)
        points = trajectory(self.parameters, self.times)
        pixels = self.camera.pixels(points)
        depths = (points @ self.camera.rotation.T + self.camera.translation)[:, 2]
        radii = 2200 * 0.22 / (2 * depths)
        if noisy:
            pixels += rng.normal(0, 0.3, pixels.shape)
            pixels[15] += [70, -90]
            radii *= 1 + rng.normal(0, 0.07, radii.shape)
        return {i: Detection(p, r) for i, (p, r) in enumerate(zip(pixels, radii)) if i not in (4, 5, 22)}

    def test_recovers_clean_flight_with_vfr_and_missing_frames(self):
        fit = fit_flight(self.camera, self.detections(), self.times, 0, 30)
        self.assertTrue(fit.valid)
        np.testing.assert_allclose(fit.parameters, self.parameters, atol=0.08)
        np.testing.assert_allclose(fit.velocity(self.times[22]), velocity(self.parameters, self.times[22]), atol=0.08)

    def test_outlier_and_size_noise_do_not_dominate_speed(self):
        fit = fit_flight(self.camera, self.detections(True), self.times, 0, 30)
        self.assertTrue(fit.valid)
        self.assertFalse(fit.inliers[list(fit.frames).index(15)])
        truth = np.linalg.norm(velocity(self.parameters, self.times[-1]))
        self.assertLess(abs(fit.summary['speed_at_end_mps'] - truth), 1.)

    def test_short_track_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'at least 6'):
            fit_flight(self.camera, self.detections(), self.times, 0, 3)

    def test_velocity_is_analytic_derivative(self):
        t = .41
        derivative = (trajectory(self.parameters, t + 1e-5) - trajectory(self.parameters, t - 1e-5)) / 2e-5
        np.testing.assert_allclose(velocity(self.parameters, t), derivative, atol=1e-8)

    def test_pts_are_normalized_without_assuming_constant_fps(self):
        data = SimpleNamespace(stdout='{"frames": [{"best_effort_timestamp_time": "2.0"}, {"best_effort_timestamp_time": "2.025"}, {"best_effort_timestamp_time": "2.065"}]}')
        with patch('src.ballistic_speed.subprocess.run', return_value=data):
            times, source = frame_times('video.mp4', 30, 3)
        np.testing.assert_allclose(times, [0, .025, .065])
        self.assertEqual(source, 'video_pts')

    def test_segment_t0_uses_selected_frame_time(self):
        fit = fit_flight(self.camera, self.detections(), self.times, 6, 30)
        self.assertTrue(fit.valid)
        np.testing.assert_allclose(fit.position(self.times[6]), trajectory(self.parameters, self.times[6]), atol=.1)
        np.testing.assert_allclose(fit.velocity(self.times[6]), velocity(self.parameters, self.times[6]), atol=.1)


if __name__ == '__main__':
    unittest.main()
