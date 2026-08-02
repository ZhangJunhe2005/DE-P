import unittest

import numpy as np

from policy.dynamic.clustering import StandardDBSCANClustering
from policy.dynamic.track_manager import TrackManager
from tests.dynamic_helpers import observation, test_config


class ClusterSplitMergeTests(unittest.TestCase):
    def test_planar_cluster_keeps_extent_dependent_centroid_uncertainty(self):
        config = test_config(foreground_mode="none", cluster_eps=0.5, cluster_min_samples=3)
        x, y = np.meshgrid(np.linspace(-1, 1, 12), np.linspace(-1, 1, 12))
        points = np.stack((x.ravel(), y.ravel(), np.full(x.size, 5.0)), axis=1)
        observations, _ = StandardDBSCANClustering(config).cluster(points, points, 0.0)
        self.assertEqual(len(observations), 1)
        self.assertGreater(observations[0].position_covariance[0, 0], 1e-3)
        self.assertGreater(observations[0].extent[0], 1.5)

    def test_split_merge_velocity_jump_is_rejected(self):
        config = test_config(
            foreground_mode="none",
            association_distance_threshold=5.0,
            association_mahalanobis_threshold=1e6,
            association_point_count_ratio_max=2.0,
            association_bbox_margin=0.01,
            physically_plausible_acceleration_max=5.0,
        )
        manager = TrackManager(config)
        first = observation([0, 0, 5], 0.0)
        second = observation([0.05, 0, 5], 0.1)
        manager.update((first,), 0.0)
        manager.update((second,), 0.1)
        prior_velocity = manager.snapshot()[0].velocity_world.copy()
        jump = observation([1.0, 0, 5], 0.2)
        manager.update((jump,), 0.2)
        details = manager.last_diagnostics["match_details"]
        if details:
            self.assertTrue(details[0]["split_merge_suspected"] or
                            np.linalg.norm(details[0]["raw_centroid_velocity"]) > 5.0)
            self.assertLessEqual(
                np.linalg.norm(manager.snapshot()[0].velocity_world - prior_velocity), 1e-6
            )
        else:
            self.assertEqual(len(manager.last_diagnostics["unmatched_observation_indices"]), 1)


if __name__ == "__main__":
    unittest.main()
