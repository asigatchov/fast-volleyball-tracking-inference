import tempfile
import unittest
import json
from pathlib import Path

import cv2
import numpy as np

from src.calc_speed import Camera, SpeedWindow, calibrate_camera, load_ball, load_ball_boxes


class BallDepthTests(unittest.TestCase):
    def setUp(self):
        intrinsic = np.array([[1800., 0, 640], [0, 1800., 360], [0, 0, 1]])
        center = np.array([4.5, -17., 8.])
        forward = np.array([4.5, 9., 0.]) - center
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0, 0, 1])
        right /= np.linalg.norm(right)
        rotation = np.array([right, np.cross(forward, right), forward])
        self.camera = Camera(intrinsic, rotation, -rotation @ center, 0)

    def test_size_recovers_known_3d_positions(self):
        for world in (np.array([3., 7., 2.5]), np.array([6., 15., 4.])):
            xy = self.camera.pixels([world])[0]
            depth = (self.camera.rotation @ world + self.camera.translation)[2]
            radius = 1800 * 0.22 / (2 * depth)
            np.testing.assert_allclose(self.camera.ball_position(xy, radius, 0.22), world, atol=1e-10)
            self.assertIsNone(self.camera.ball_position(xy, 0, 0.22))
            self.assertIsNone(self.camera.ball_position(xy, float('nan'), 0.22))

    def test_calibration_from_floor_and_net(self):
        world = np.array([(0, 0, 0), (0, 18, 0), (9, 18, 0), (9, 0, 0),
                          (0, 9, 0), (9, 9, 0), (0, 9, 2.43), (9, 9, 2.43)])
        pixels = self.camera.pixels(world)
        fitted = calibrate_camera(dict(enumerate(pixels)), 1280, 720)
        self.assertLess(fitted.rmse, 0.001)
        np.testing.assert_allclose(fitted.intrinsic, self.camera.intrinsic, atol=0.02)
        np.testing.assert_allclose(fitted.pixels(world), pixels, atol=0.001)

    def test_sparse_3d_speed_and_expiry(self):
        window = SpeedWindow(30, 0.5)
        self.assertIsNone(window.update(0, np.array([0., 0., 1.])))
        self.assertIsNone(window.update(8, None))
        speed = window.update(15, np.array([0., 0., 4.]))
        self.assertEqual(speed.mps, 6.)
        self.assertTrue(np.isnan(speed.angle))
        self.assertIsNone(window.update(31, np.array([1., 0., 4.])))

    def test_csv_radius_and_missing_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ball.csv'
            path.write_text('Frame,Visibility,X,Y,Radius\n1,1,100,200,14\n2,0,0,0,0\n16,1,150,250,0\n')
            detections = load_ball(path, frame_base=1)
        self.assertEqual(set(detections), {0, 15})
        self.assertEqual(detections[0].radius, 14)
        self.assertEqual(detections[15].radius, 0)

    def test_box_radius_uses_pixel_short_side_not_diagonal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'boxes.json'
            path.write_text(json.dumps({'yolo_boxes': {
                '0': {'1': [6, 0.5, 0.25, 0.02, 0.04], '2': [5, 0.1, 0.1, 0.1, 0.1]},
                '3': {'2': [5, 0.1, 0.1, 0.1, 0.1]}}}))
            detections = load_ball_boxes(path, 1280, 720)
        self.assertEqual(set(detections), {0})
        np.testing.assert_allclose(detections[0].xy, [640, 180])
        self.assertAlmostEqual(detections[0].radius, 12.8)
        np.testing.assert_allclose(detections[0].box, [627.2, 165.6, 652.8, 194.4])


if __name__ == '__main__':
    unittest.main()
