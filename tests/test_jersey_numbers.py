import unittest

import numpy as np

from src.jersey_numbers import POSE_XML, PoseModel, torso_crop


@unittest.skipUnless(POSE_XML.exists(), "pose model not exported")
class PoseModelTest(unittest.TestCase):
    def test_keypoints_land_in_the_crop_whatever_its_shape(self):
        # Top-down: any crop is warped to the model input, a 400x30 sliver included (it broke the YOLO export).
        model = PoseModel()
        for shape in ((400, 30), (30, 400), (400, 160), (60, 60)):
            keypoints = model(np.full((*shape, 3), 120, np.uint8))
            self.assertEqual(keypoints.shape, (17, 3))
            self.assertTrue(np.isfinite(keypoints).all())
            # Nothing lands outside the box the model saw: the crop padded x1.25 and widened or
            # heightened to the 3:4 input, around the crop's centre.
            h, w = shape
            box_w, box_h = max(1.25 * w, 1.25 * h * 0.75), max(1.25 * h, 1.25 * w / 0.75)
            self.assertTrue((abs(keypoints[:, 0] - w / 2) <= box_w / 2 + 1).all())
            self.assertTrue((abs(keypoints[:, 1] - h / 2) <= box_h / 2 + 1).all())


class TorsoCropTest(unittest.TestCase):
    def test_the_patch_runs_from_the_shoulders_to_above_the_hips(self):
        person = np.zeros((200, 100, 3), np.uint8)
        keypoints = np.zeros((17, 3), np.float32)
        keypoints[[5, 6]] = [[70, 50, 1.0], [30, 50, 1.0]]
        keypoints[[11, 12]] = [[65, 150, 1.0], [35, 150, 1.0]]
        patch = torso_crop(person, keypoints)
        self.assertEqual(patch.shape[:2], (80, 52))  # 50..130 (80% of the 100 px torso), 24..76 (15% wider)
        keypoints[11, 2] = 0.1
        self.assertIsNone(torso_crop(person, keypoints))


if __name__ == "__main__":
    unittest.main()
