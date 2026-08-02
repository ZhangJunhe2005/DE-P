import unittest

import numpy as np

from policy.dynamic.track_manager import TrackManager
from tests.dynamic_helpers import observation, test_config


class VelocityOutlierTests(unittest.TestCase):
    def test_single_centroid_outlier_cannot_create_dynamic_track(self):
        config = test_config(
            foreground_mode="none",
            association_distance_threshold=10.0,
            association_mahalanobis_threshold=1e9,
            physically_plausible_speed_max=3.0,
            physically_plausible_acceleration_max=10.0,
            min_confirmed_hits=2,
            dynamic_min_confirmed_hits=3,
            dynamic_max_velocity_std=10.0,
        )
        manager = TrackManager(config)
        manager.update((observation([0, 0, 5], 0.0),), 0.0)
        manager.update((observation([0.02, 0, 5], 0.1),), 0.1)
        tracks = manager.update((observation([5.0, 0, 5], 0.2),), 0.2)
        original = next(track for track in tracks if track.track_id == 0)
        self.assertFalse(original.is_dynamic)
        self.assertLessEqual(np.linalg.norm(original.velocity_world), 3.0)


if __name__ == "__main__":
    unittest.main()
