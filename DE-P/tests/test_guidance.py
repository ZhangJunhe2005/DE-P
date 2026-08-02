import unittest

import torch

from config.config import cfg
from loss.guidance_loss import GuidanceLoss
from tests.baseline_helpers import finite_tensor


class GuidanceBoundaryTests(unittest.TestCase):
    def test_goal_distance_boundaries_and_gradients_are_finite(self):
        module = GuidanceLoss()
        distances = torch.tensor([0.0, 1e-8, module.goal_length, 2.0 * module.goal_length])
        fixed = torch.zeros(4, 3, 3)
        predicted = torch.zeros(4, 3, 3, requires_grad=True)
        predicted.data[:, 0, 0] = 1.0
        predicted.data[:, 1, 0] = 0.5
        goals = torch.zeros(4, 3)
        goals[:, 0] = distances
        losses = module(fixed, predicted, goals)
        losses.sum().backward()
        self.assertTrue(finite_tensor(losses))
        self.assertTrue(finite_tensor(predicted.grad))

    def test_perpendicular_weight_uses_yaml_and_stays_in_range(self):
        module = GuidanceLoss()
        distances = torch.tensor([0.0, 1e-8, module.goal_length / 2, module.goal_length, 2 * module.goal_length])
        weights = module.perpendicular_weight(distances)
        self.assertTrue(finite_tensor(weights))
        self.assertGreaterEqual(float(weights.min()), 0.0)
        self.assertLessEqual(float(weights.max()), float(cfg["guidance_perp_max"]))
        self.assertAlmostEqual(float(weights[0]), float(cfg["guidance_perp_max"]), places=7)
        self.assertEqual(float(weights[-1]), 0.0)


if __name__ == "__main__":
    unittest.main()

