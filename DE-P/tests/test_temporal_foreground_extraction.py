import unittest

import numpy as np

from policy.dynamic.temporal_foreground import TemporalVoxelForeground
from tests.dynamic_helpers import test_config


class TemporalForegroundTests(unittest.TestCase):
    def config(self, **overrides):
        values = dict(
            foreground_mode="temporal_voxel",
            background_voxel_size=0.2,
            background_match_distance=0.25,
            background_min_persistence=3,
            foreground_min_persistence=2,
            background_update_speed_limit=0.2,
            foreground_reset_gap=0.5,
            background_max_age=5,
            background_max_voxels=100,
        )
        values.update(overrides)
        return test_config(**values)

    def test_static_world_points_are_suppressed_after_warmup(self):
        extractor = TemporalVoxelForeground(self.config())
        points = np.asarray([[0.01, 0.01, 5.0], [0.04, 0.02, 5.02]])
        masks = [extractor.extract(points, frame * 0.1) for frame in range(8)]
        self.assertTrue(all(not mask.any() for mask in masks))
        self.assertGreater(extractor.last_diagnostics["stable_background_voxels"], 0)
        self.assertEqual(extractor.last_diagnostics["future_frames_used"], 0)

    def test_persistent_moving_voxels_become_foreground(self):
        extractor = TemporalVoxelForeground(self.config())
        static = np.asarray([[0.01, 0.01, 5.0], [0.04, 0.02, 5.02]])
        for frame in range(4):
            extractor.extract(static, frame * 0.1)
        foreground_counts = []
        for frame in range(4, 12):
            center = np.asarray([0.12 * (frame - 4), 1.0, 5.0])
            actor = center + np.asarray([[0.0, 0.0, 0.0], [0.02, 0.01, 0.01]])
            mask = extractor.extract(np.concatenate((static, actor)), frame * 0.1)
            foreground_counts.append(int(mask.sum()))
        self.assertGreater(sum(foreground_counts), 0)

    def test_gap_resets_stream_and_capacity_is_bounded(self):
        extractor = TemporalVoxelForeground(self.config(background_max_voxels=5))
        points = np.arange(30, dtype=float).reshape(10, 3)
        extractor.extract(points, 0.0)
        self.assertLessEqual(extractor.voxel_count, 5)
        extractor.extract(points, 1.0)
        self.assertTrue(extractor.last_diagnostics["reset_due_gap"])
        self.assertLessEqual(extractor.voxel_count, 5)

    def test_duplicate_and_out_of_order_timestamps_fail(self):
        extractor = TemporalVoxelForeground(self.config())
        extractor.extract(np.zeros((1, 3)), 0.1)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            extractor.extract(np.zeros((1, 3)), 0.1)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            extractor.extract(np.zeros((1, 3)), 0.0)


if __name__ == "__main__":
    unittest.main()
