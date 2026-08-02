import unittest

import torch

from loss.dynamic_types import DynamicObstacleBatch
from tools.analyze_candidate_capacity_oracle import (
    apply_profile,
    concatenate_obstacles,
    sobol_bank,
)


class Phase8JRCapacityOracleTests(unittest.TestCase):
    def test_sobol_bank_is_bounded_deterministic_and_preserves_current_15(self):
        current = torch.linspace(-0.9, 0.9, 15 * 9).reshape(1, 15, 9)
        first, ids = sobol_bank(current, 64)
        second, second_ids = sobol_bank(current, 64)
        self.assertTrue(torch.equal(first[:, :15], current))
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(ids, second_ids))
        self.assertLessEqual(float(first.abs().max()), 1.0)
        self.assertEqual(tuple(first.shape), (1, 64, 9))

    def test_temporal_profile_remains_inside_terminal_parameter_bounds(self):
        raw = torch.ones(2, 15, 9)
        result = apply_profile(raw, {
            "radius_scale": 0.45,
            "velocity_scale": 0.0,
            "acceleration_scale": 0.0,
        })
        self.assertLessEqual(float(result.abs().max()), 1.0)
        self.assertTrue(torch.equal(result[:, :, 3:], torch.zeros_like(
            result[:, :, 3:]
        )))

    def test_variable_obstacle_counts_are_padded_without_creating_valid_tracks(self):
        empty = DynamicObstacleBatch.empty(1)
        one = DynamicObstacleBatch(
            positions_world=torch.zeros(1, 1, 3),
            velocities_world=torch.zeros(1, 1, 3),
            position_covariances=torch.eye(3).reshape(1, 1, 3, 3),
            radii=torch.ones(1, 1),
            track_timestamps=torch.zeros(1, 1),
            sample_timestamps=torch.zeros(1),
            confidence=torch.ones(1, 1),
            valid_mask=torch.ones(1, 1, dtype=torch.bool),
            dynamic_mask=torch.ones(1, 1, dtype=torch.bool),
            observable_mask=torch.ones(1, 1, dtype=torch.bool),
        )
        result = concatenate_obstacles([empty, one])
        self.assertEqual(tuple(result.positions_world.shape), (2, 1, 3))
        self.assertEqual(result.valid_mask.tolist(), [[False], [True]])


if __name__ == "__main__":
    unittest.main()
