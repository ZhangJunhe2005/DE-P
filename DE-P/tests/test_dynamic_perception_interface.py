from dataclasses import replace
import unittest

import numpy as np
import torch

from policy.dynamic.dynamic_perception import DynamicPerception
from policy.dynamic.types import DynamicPerceptionConfig
from tests.dynamic_helpers import (
    camera_cloud_from_world,
    camera_model,
    pose,
    test_config,
    world_cluster,
)


class DynamicConfigurationTests(unittest.TestCase):
    def test_yaml_is_complete_and_invalid_values_fail(self):
        config = DynamicPerceptionConfig.from_global_config()
        self.assertGreater(config.cluster_eps, 0)
        with self.assertRaisesRegex(ValueError, "dynamic_enter_speed"):
            replace(config, dynamic_enter_speed=0.1, dynamic_exit_speed=0.2).validate()
        with self.assertRaisesRegex(ValueError, "cluster_eps"):
            replace(config, cluster_eps=0).validate()
        with self.assertRaisesRegex(ValueError, "unknown"):
            DynamicPerceptionConfig.from_mapping({"cluster_size": 10})
        with self.assertRaisesRegex(ValueError, "source"):
            replace(config, source="scan").validate()


class DynamicPerceptionInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.config = test_config(
            foreground_mode="none",
            dynamic_min_cluster_extent=0.02,
            cluster_eps=0.15,
            cluster_min_samples=5,
            association_distance_threshold=1.0,
            association_mahalanobis_threshold=100.0,
            min_confirmed_hits=2,
            dynamic_min_confirmed_hits=3,
            dynamic_max_velocity_std=2.0,
        )
        self.camera = camera_model()

    def test_empty_input_reset_and_diagnostics(self):
        perception = DynamicPerception(self.config, feature_shape=(4, 6))
        result = perception.update(np.empty((0, 3)), pose(timestamp=0), 0, self.camera)
        self.assertEqual(result.all_tracks, ())
        self.assertEqual(tuple(result.attention_map.shape), (1, 1, 4, 6))
        self.assertEqual(float(result.attention_map.max()), 0.0)
        self.assertEqual(result.diagnostics["clustering_algorithm"], "sklearn.cluster.DBSCAN")
        perception.reset()
        result = perception.update(np.empty((0, 3)), pose(timestamp=0), 0, self.camera)
        self.assertEqual(result.all_tracks, ())

    def test_pose_timestamp_and_frame_timestamp_validation(self):
        perception = DynamicPerception(self.config)
        with self.assertRaisesRegex(ValueError, "too stale"):
            perception.update(np.empty((0, 3)), pose(timestamp=0), 1.0, self.camera)
        perception.update(np.empty((0, 3)), pose(timestamp=1.0), 1.0, self.camera)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            perception.update(np.empty((0, 3)), pose(timestamp=1.0), 1.0, self.camera)
        with self.assertRaisesRegex(ValueError, "out-of-order"):
            perception.update(np.empty((0, 3)), pose(timestamp=0.5), 0.5, self.camera)

    def test_cpu_clustering_rejects_implicit_tensor_conversion(self):
        perception = DynamicPerception(self.config)
        with self.assertRaisesRegex(TypeError, "NumPy"):
            perception.update(torch.empty(0, 3), pose(timestamp=0), 0, self.camera)

    def test_end_to_end_moving_track_projects_and_builds_attention(self):
        perception = DynamicPerception(self.config, feature_shape=(3, 5))
        offsets = world_cluster([0, 0, 0], seed=33, count=35, scale=0.02)
        result = None
        for frame in range(12):
            timestamp = frame * 0.1
            center = np.array([0.04 * frame, 0, 5])
            current_pose = pose(timestamp=timestamp)
            cloud = camera_cloud_from_world(offsets + center, current_pose)
            result = perception.update(cloud, current_pose, timestamp, self.camera)
        self.assertEqual(len(result.all_tracks), 1)
        self.assertEqual(len(result.confirmed_tracks), 1)
        self.assertEqual(len(result.dynamic_tracks), 1)
        self.assertEqual(len(result.projected_dynamic_tracks), 1)
        self.assertGreater(float(result.attention_map.max()), 0.0)
        self.assertLessEqual(float(result.attention_map.max()), self.config.attention_max)
        self.assertTrue(torch.isfinite(result.attention_map).all())


if __name__ == "__main__":
    unittest.main()
