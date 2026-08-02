import unittest
from unittest import mock

import numpy as np
import torch

from tests.baseline_helpers import DATASET_ROOT, finite_tensor, get_single_map_loss


class LossNumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loss_module = get_single_map_loss()
        cls.device = cls.loss_module.device

    def _simple_trajectories(self):
        pose = np.loadtxt(DATASET_ROOT / "pose-0.csv", delimiter=",", skiprows=1, max_rows=1).astype(np.float32)
        origin = torch.tensor(pose[:3], device=self.device)
        state = torch.zeros(15, 3, 3, device=self.device)
        prediction = torch.zeros(15, 3, 3, device=self.device)
        state[:, 0, :] = origin
        prediction[:, 0, :] = origin + torch.tensor([2.0, 0.0, 0.0], device=self.device)
        prediction.requires_grad_(True)
        goal = (origin + torch.tensor([5.0, 0.0, 0.0], device=self.device)).repeat(15, 1)
        map_id = torch.tensor([0], dtype=torch.long, device=self.device)
        return state, prediction, goal, map_id

    def test_loss_components_and_backward_are_finite(self):
        state, prediction, goal, map_id = self._simple_trajectories()
        smooth, safety, guidance = self.loss_module(state, prediction, goal, map_id)
        total = (smooth + safety + guidance).mean()
        total.backward()
        print(
            "BASELINE_LOSS",
            {
                "smooth": float(smooth.mean().detach().cpu()),
                "safety": float(safety.mean().detach().cpu()),
                "guidance": float(guidance.mean().detach().cpu()),
                "total": float(total.detach().cpu()),
            },
        )
        for value in (smooth, safety, guidance, total, prediction.grad):
            self.assertTrue(finite_tensor(value))

    def _scan_cost(self):
        distances = torch.linspace(0.0, 10.0, 101, device=self.device)
        costs = self.loss_module.safety_loss.cost_function(distances)
        return distances, costs

    def test_safety_cost_scan_is_finite(self):
        distances, costs = self._scan_cost()
        sampled = [(round(float(d), 1), round(float(c), 8)) for d, c in zip(distances[::5], costs[::5])]
        print("BASELINE_SAFETY_SCAN", sampled)
        self.assertTrue(finite_tensor(costs))

    def test_safety_cost_should_monotonically_decrease(self):
        _, costs = self._scan_cost()
        self.assertTrue(bool((torch.diff(costs) <= 1e-7).all().item()))
        self.assertLess(float(costs[-1].cpu()), 1e-5)

    def test_safety_cost_should_be_continuous_at_d0(self):
        d0 = self.loss_module.safety_loss.d0
        eps = 1e-5
        points = torch.tensor([d0 - eps, d0, d0 + eps], device=self.device)
        costs = self.loss_module.safety_loss.cost_function(points)
        self.assertLess(float(torch.max(torch.abs(torch.diff(costs))).cpu()), 1e-3)

    def test_safety_cost_derivative_is_consistent_and_decreasing(self):
        d0 = self.loss_module.safety_loss.d0
        h = 1e-3
        points = torch.tensor([d0 - h, d0, d0 + h], device=self.device)
        costs = self.loss_module.safety_loss.cost_function(points)
        derivative_left = (costs[1] - costs[0]) / h
        derivative_right = (costs[2] - costs[1]) / h
        self.assertLess(float(derivative_left.cpu()), 0.0)
        self.assertLess(float(derivative_right.cpu()), 0.0)
        self.assertLess(float(torch.abs(derivative_left - derivative_right).cpu()), 1e-2)

    def test_safety_cost_extremes_and_distance_gradient_are_finite(self):
        distances = torch.tensor([-1.0, 0.0, self.loss_module.safety_loss.d0, 10.0, 100.0],
                                 device=self.device, requires_grad=True)
        costs = self.loss_module.safety_loss.cost_function(distances)
        costs.sum().backward()
        self.assertTrue(finite_tensor(costs))
        self.assertTrue(finite_tensor(distances.grad))
        self.assertTrue(bool((distances.grad[:4] < 0).all().item()))

    def test_out_of_bounds_queries_use_configured_high_risk_cost(self):
        safety = self.loss_module.safety_loss
        fake_sdf = torch.full((1, 1, 3, 3, 3), 10.0, device=self.device)
        origin = torch.zeros(1, 3, device=self.device)
        shape = torch.full((1, 3), 3.0, device=self.device)
        # voxel_size=0.2: 0.2 m maps to the center voxel, 0.0 to the
        # configured unknown boundary, and 1.0 lies outside this 3^3 crop.
        positions = torch.tensor([[[0.2, 0.2, 0.2], [0.0, 0.2, 0.2], [1.0, 0.2, 0.2]]], device=self.device)
        with mock.patch.object(safety, "get_batch_sdf", return_value=(fake_sdf, origin, shape)):
            costs, _ = safety.get_distance_cost(positions, torch.tensor([0], device=self.device))
        self.assertLess(float(costs[0, 0].cpu()), safety.out_of_bounds_cost)
        self.assertEqual(float(costs[0, 1].cpu()), safety.out_of_bounds_cost)
        self.assertEqual(float(costs[0, 2].cpu()), safety.out_of_bounds_cost)

    def test_distance_field_is_signed(self):
        sdf = self.loss_module.safety_loss.sdf_maps[0]
        sdf_min = float(sdf.min().cpu())
        negative_count = int((sdf < 0).sum().cpu())
        zero_count = int((sdf == 0).sum().cpu())
        print("BASELINE_SDF_STATS", {"min": sdf_min, "negative_count": negative_count, "zero_count": zero_count})
        self.assertLess(sdf_min, 0.0)
        self.assertGreater(negative_count, 0)


if __name__ == "__main__":
    unittest.main()
