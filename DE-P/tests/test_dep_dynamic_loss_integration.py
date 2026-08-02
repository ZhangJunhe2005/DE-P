from dataclasses import replace
import unittest

import torch

from loss.dynamic_types import DynamicLossConfig
from loss.loss_function import DEPLoss
from tests.test_dynamic_safety_loss import obstacle_batch


class DepDynamicLossIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.enabled_config = replace(
            DynamicLossConfig.from_global_config(), enabled=True,
            target_source="constant_velocity"
        )
        cls.loss = DEPLoss(dynamic_loss_config=cls.enabled_config)

    def inputs(self):
        start = torch.zeros(15, 3, 3)
        prediction = torch.zeros(15, 3, 3, requires_grad=True)
        prediction.data[:, 0, 0] = 2.0
        goal = torch.tensor([[2.0, 0.0, 0.0]]).repeat(15, 1)
        map_id = torch.zeros(1, dtype=torch.long)
        return start, prediction, goal, map_id

    def test_dynamic_cost_enters_total_and_detached_score_label(self):
        start, prediction, goal, map_id = self.inputs()
        result = self.loss(
            start, prediction, goal, map_id,
            dynamic_obstacles=obstacle_batch((1, 0, 0)),
            return_details=True,
        )
        self.assertGreater(float(result.dynamic_safety_cost.mean()), 0)
        expected = (result.smooth_cost + result.static_safety_cost
                    + result.guidance_cost + result.dynamic_safety_cost)
        self.assertTrue(torch.equal(result.trajectory_cost, expected))
        label = result.detached_score_label()
        self.assertFalse(label.requires_grad)
        self.assertTrue(torch.equal(label, expected.detach()))
        result.trajectory_cost.mean().backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_none_obstacles_is_exact_zero(self):
        start, prediction, goal, map_id = self.inputs()
        result = self.loss(
            start, prediction, goal, map_id,
            dynamic_obstacles=None, return_details=True,
        )
        self.assertTrue(torch.equal(
            result.dynamic_safety_cost, torch.zeros_like(result.dynamic_safety_cost)
        ))

    def test_default_disabled_preserves_three_value_api(self):
        start, prediction, goal, map_id = self.inputs()
        original_config = self.loss.dynamic_loss_config
        self.loss.dynamic_loss_config = replace(original_config, enabled=False)
        try:
            legacy = self.loss(start, prediction, goal, map_id)
            details = self.loss(
                start, prediction, goal, map_id,
                dynamic_obstacles=obstacle_batch((1, 0, 0)), return_details=True,
            )
        finally:
            self.loss.dynamic_loss_config = original_config
        self.assertEqual(len(legacy), 3)
        self.assertTrue(torch.equal(legacy[0], details.smooth_cost))
        self.assertTrue(torch.equal(legacy[1], details.static_safety_cost))
        self.assertTrue(torch.equal(legacy[2], details.guidance_cost))
        self.assertTrue(torch.equal(
            details.dynamic_safety_cost, torch.zeros_like(details.dynamic_safety_cost)
        ))


if __name__ == "__main__":
    unittest.main()
