import unittest

import numpy as np

from tools.publish_reachable_path_odom import sample_polyline


class ReachableOdomTests(unittest.TestCase):
    def test_polyline_sampling_is_deterministic_and_bounded(self):
        points = np.asarray([[0.0, 0.0, 1.0], [2.0, 0.0, 1.0], [2.0, 3.0, 1.0]])
        position, direction, length = sample_polyline(points, 3.0)
        self.assertEqual(length, 5.0)
        self.assertTrue(np.allclose(position, [2.0, 1.0, 1.0]))
        self.assertTrue(np.allclose(direction, [0.0, 1.0, 0.0]))
        end, _, _ = sample_polyline(points, 100.0)
        self.assertTrue(np.allclose(end, points[-1]))


if __name__ == "__main__":
    unittest.main()
