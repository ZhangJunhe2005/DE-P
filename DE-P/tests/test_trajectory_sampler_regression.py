import math
import unittest

import torch

from config.config import cfg
from loss.trajectory_sampler import QuinticTrajectorySampler


def coefficient_map(duration):
    matrix = torch.zeros(6, 6)
    for derivative in range(3):
        matrix[2 * derivative, derivative] = math.factorial(derivative)
        for power in range(derivative, 6):
            matrix[2 * derivative + 1, power] = (
                math.factorial(power) / math.factorial(power - derivative)
                * duration ** (power - derivative)
            )
    permutation = torch.zeros(6, 6)
    permutation[[0, 2, 4, 1, 3, 5], [0, 1, 2, 3, 4, 5]] = 1
    return torch.inverse(matrix) @ permutation


class TrajectorySamplerRegressionTests(unittest.TestCase):
    def test_matches_historical_loop_elementwise(self):
        torch.manual_seed(9)
        duration = float(cfg["sgm_time"])
        mapping = coefficient_map(duration)
        fixed = torch.randn(15, 3, 3)
        predicted = torch.randn(15, 3, 3)
        sampler = QuinticTrajectorySampler(mapping, duration, 30)
        actual, actual_times = sampler(fixed, predicted)

        batch_mapping = mapping.unsqueeze(0).expand(15, -1, -1)
        coefficients = torch.zeros(15, 18)
        for axis in range(3):
            derivative = torch.cat([fixed[:, axis], predicted[:, axis]], dim=1).unsqueeze(-1)
            coefficients[:, axis * 6:(axis + 1) * 6] = (
                batch_mapping @ derivative
            ).squeeze()
        dt = duration / 30
        times = torch.linspace(dt, duration, 30).view(1, -1, 1).expand(15, -1, -1)
        powers = torch.stack(
            [torch.ones_like(times), times, times ** 2, times ** 3,
             times ** 4, times ** 5], dim=-1
        ).squeeze(-2)
        expected = torch.stack([
            torch.sum(powers * coefficients[:, axis * 6:(axis + 1) * 6].unsqueeze(1), dim=-1)
            for axis in range(3)
        ], dim=-1)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(actual_times, torch.linspace(dt, duration, 30)))

    def test_grouped_shape_and_endpoint(self):
        duration = float(cfg["sgm_time"])
        sampler = QuinticTrajectorySampler(coefficient_map(duration), duration, 30)
        fixed = torch.zeros(30, 3, 3)
        predicted = torch.zeros_like(fixed)
        predicted[:, 0, 0] = 2.0
        positions, times = sampler.grouped(fixed, predicted, batch_size=2)
        self.assertEqual(tuple(positions.shape), (2, 15, 30, 3))
        self.assertTrue(torch.allclose(positions[:, :, -1, 0], torch.full((2, 15), 2.0), atol=1e-5))
        self.assertAlmostEqual(float(times[-1]), duration, places=6)


if __name__ == "__main__":
    unittest.main()
