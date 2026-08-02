import unittest

import numpy as np
import torch

from policy.dynamic.camera_model import depth_to_pointcloud
from policy.dynamic.clustering import StandardDBSCANClustering
from tests.dynamic_helpers import camera_model, test_config


class DepthToPointCloudTests(unittest.TestCase):
    def test_max_depth_saturation_is_not_an_obstacle(self):
        camera = camera_model()
        depth = np.full((camera.height, camera.width), camera.max_depth, dtype=np.float32)
        self.assertEqual(depth_to_pointcloud(depth, camera).shape, (0, 3))
        torch_depth = torch.full((camera.height, camera.width), camera.max_depth)
        self.assertEqual(tuple(depth_to_pointcloud(torch_depth, camera).shape), (0, 3))

    def test_float_depth_uses_intrinsics_and_filters_invalid_values(self):
        camera = camera_model()
        depth = np.zeros((camera.height, camera.width), dtype=np.float32)
        depth[48, 80] = 2.0
        depth[47, 79] = np.nan
        depth[1, 1] = np.inf
        points = depth_to_pointcloud(depth, camera)
        self.assertEqual(points.dtype, np.float32)
        self.assertEqual(points.shape, (1, 3))
        self.assertTrue(np.allclose(points[0], [0.01, 0.01, 2.0], atol=1e-6))

    def test_uint16_millimetres_and_stride(self):
        base = camera_model()
        camera = type(base)(**{**base.__dict__, "depth_scale": 0.001})
        depth = np.zeros((camera.height, camera.width), dtype=np.uint16)
        depth[0, 0] = 1000
        depth[2, 2] = 2000
        points = depth_to_pointcloud(depth, camera, stride=2)
        self.assertEqual(points.shape, (2, 3))
        self.assertTrue(np.allclose(points[:, 2], [1.0, 2.0]))

    def test_torch_stays_torch_and_invalid_shape_fails(self):
        camera = camera_model()
        depth = torch.ones(camera.height, camera.width)
        points = depth_to_pointcloud(depth, camera, stride=4)
        self.assertTrue(torch.is_tensor(points))
        self.assertEqual(points.dtype, torch.float32)
        self.assertEqual(points.device, depth.device)
        with self.assertRaisesRegex(ValueError, "depth shape"):
            depth_to_pointcloud(torch.ones(3, 4), camera)


class StandardDBSCANTests(unittest.TestCase):
    def setUp(self):
        self.clusterer = StandardDBSCANClustering(test_config(
            cluster_eps=0.18, cluster_min_samples=5
        ))

    def cluster(self, points):
        points = np.asarray(points, dtype=np.float64)
        return self.clusterer.cluster(points, points, 1.0)

    def test_empty_and_all_noise(self):
        observations, labels = self.cluster(np.empty((0, 3)))
        self.assertEqual(observations, ())
        self.assertEqual(labels.shape, (0,))
        observations, labels = self.cluster(np.array([[0, 0, 1], [2, 2, 2], [4, 4, 4]]))
        self.assertEqual(observations, ())
        self.assertTrue(np.all(labels == -1))

    def test_one_and_two_obstacles_include_statistics(self):
        rng = np.random.default_rng(3)
        one = rng.normal([0, 0, 5], 0.03, (20, 3))
        observations, _ = self.cluster(one)
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].point_count, 20)
        self.assertEqual(observations[0].bounding_box_world.shape, (2, 3))
        self.assertTrue(np.isfinite(observations[0].position_covariance).all())
        two = np.vstack((one, rng.normal([0.8, 0, 5], 0.03, (20, 3))))
        observations, _ = self.cluster(two)
        self.assertEqual(len(observations), 2)

    def test_nearby_distinct_obstacles_do_not_merge_and_sparse_is_noise(self):
        rng = np.random.default_rng(9)
        cloud = np.vstack((
            rng.normal([0, 0, 5], 0.02, (12, 3)),
            rng.normal([0.42, 0, 5], 0.02, (12, 3)),
            np.array([[1.5, 0, 5], [1.55, 0, 5]]),
        ))
        observations, labels = self.cluster(cloud)
        self.assertEqual(len(observations), 2)
        self.assertEqual(int(np.sum(labels == -1)), 2)

    def test_fixed_input_is_deterministic(self):
        rng = np.random.default_rng(11)
        cloud = np.vstack((rng.normal([0, 0, 5], 0.03, (20, 3)),
                           rng.normal([1, 0, 5], 0.03, (20, 3))))
        first_observations, first_labels = self.cluster(cloud)
        second_observations, second_labels = self.cluster(cloud)
        self.assertTrue(np.array_equal(first_labels, second_labels))
        self.assertTrue(np.array_equal(
            np.stack([item.centroid_world for item in first_observations]),
            np.stack([item.centroid_world for item in second_observations]),
        ))


if __name__ == "__main__":
    unittest.main()
