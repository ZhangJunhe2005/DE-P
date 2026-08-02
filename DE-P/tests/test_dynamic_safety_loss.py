from dataclasses import replace
import unittest

import torch

from config.config import cfg
from loss.dynamic_safety_loss import DynamicCollisionLoss
from loss.dynamic_types import DynamicLossConfig, DynamicObstacleBatch
from loss.trajectory_sampler import QuinticTrajectorySampler
from tests.test_trajectory_sampler_regression import coefficient_map


def obstacle_batch(position, velocity=(0, 0, 0), covariance=0.0, radius=0.2,
                   timestamp=1.0, sample_timestamp=1.0, confidence=1.0,
                   valid=True, dynamic=True):
    position = torch.tensor(position, dtype=torch.float32).reshape(1, -1, 3)
    obstacles = position.shape[1]
    velocity = torch.tensor(velocity, dtype=torch.float32)
    if velocity.ndim == 1:
        velocity = velocity.reshape(1, 1, 3).expand(1, obstacles, 3).clone()
    else:
        velocity = velocity.reshape(1, obstacles, 3)
    return DynamicObstacleBatch(
        positions_world=position,
        velocities_world=velocity,
        position_covariances=torch.eye(3).reshape(1, 1, 3, 3).expand(1, obstacles, 3, 3).clone() * covariance,
        radii=torch.full((1, obstacles), radius),
        track_timestamps=torch.full((1, obstacles), timestamp),
        sample_timestamps=torch.tensor([sample_timestamp]),
        confidence=torch.full((1, obstacles), confidence),
        valid_mask=torch.full((1, obstacles), valid, dtype=torch.bool),
        dynamic_mask=torch.full((1, obstacles), dynamic, dtype=torch.bool),
    )


class DynamicCollisionLossTests(unittest.TestCase):
    def setUp(self):
        self.duration = float(cfg["sgm_time"])
        self.sampler = QuinticTrajectorySampler(
            coefficient_map(self.duration), self.duration, 30
        )
        self.config = replace(
            DynamicLossConfig.from_global_config(), enabled=True,
            target_source="constant_velocity",
        )
        self.loss = DynamicCollisionLoss(self.sampler, self.config)
        self.fixed = torch.zeros(15, 3, 3)
        self.predicted = torch.zeros_like(self.fixed)
        self.predicted[:, 0, 0] = 2.0

    def cost(self, obstacles, predicted=None):
        return self.loss(self.fixed, predicted if predicted is not None else self.predicted, obstacles)[0]

    def test_empty_padding_static_and_stale_are_exact_zero(self):
        cases = [
            DynamicObstacleBatch.empty(1, timestamp=1.0),
            obstacle_batch((1, 0, 0), valid=False),
            obstacle_batch((1, 0, 0), dynamic=False),
            obstacle_batch((1, 0, 0), timestamp=0.0, sample_timestamp=1.0),
        ]
        for case in cases:
            with self.subTest(obstacles=case.max_obstacles):
                cost = self.cost(case)
                self.assertTrue(torch.equal(cost, torch.zeros_like(cost)))

    def test_collision_is_higher_than_far_away_and_temporally_separated(self):
        collision = self.cost(obstacle_batch((1, 0, 0))).mean()
        far = self.cost(obstacle_batch((100, 100, 100))).mean()
        leaves_early = self.cost(obstacle_batch((1, 0, 0), velocity=(0, 8, 0))).mean()
        arrives_late = self.cost(obstacle_batch((1, -10, 0), velocity=(0, 1, 0))).mean()
        departing = self.cost(obstacle_batch((1, 0, 0), velocity=(8, 0, 0))).mean()
        self.assertGreater(float(collision), 10 * float(far))
        self.assertGreater(float(collision), 3 * float(leaves_early))
        self.assertGreater(float(collision), 10 * float(arrives_late))
        self.assertGreater(float(collision), float(departing))

    def test_head_on_crossing_covariance_and_radius(self):
        head_on = self.cost(obstacle_batch((2.5, 0, 0), velocity=(-1, 0, 0))).mean()
        lateral = self.cost(obstacle_batch((1, -1, 0), velocity=(0, 1.2, 0))).mean()
        base = self.cost(obstacle_batch((1, 0.8, 0), covariance=0, radius=0.1)).mean()
        uncertain = self.cost(obstacle_batch((1, 0.8, 0), covariance=0.25, radius=0.1)).mean()
        large = self.cost(obstacle_batch((1, 0.8, 0), covariance=0, radius=0.8)).mean()
        self.assertGreater(float(head_on), 0)
        self.assertGreater(float(lateral), 0)
        self.assertGreater(float(uncertain), float(base))
        self.assertGreater(float(large), float(base))

    def test_max_aggregation_is_bounded_and_padding_does_not_change_cost(self):
        single = obstacle_batch((1, 0, 0))
        duplicate = obstacle_batch(((1, 0, 0), (1, 0, 0)))
        self.assertTrue(torch.allclose(self.cost(single), self.cost(duplicate)))
        padded = DynamicObstacleBatch(
            positions_world=duplicate.positions_world,
            velocities_world=duplicate.velocities_world,
            position_covariances=duplicate.position_covariances,
            radii=duplicate.radii,
            track_timestamps=duplicate.track_timestamps,
            sample_timestamps=duplicate.sample_timestamps,
            confidence=duplicate.confidence,
            valid_mask=torch.tensor([[True, False]]),
            dynamic_mask=duplicate.dynamic_mask,
        )
        self.assertTrue(torch.allclose(self.cost(single), self.cost(padded)))

    def test_timestamp_error_and_extremes_are_finite(self):
        with self.assertRaisesRegex(ValueError, "future track_timestamp"):
            self.cost(obstacle_batch((1, 0, 0), timestamp=2.0, sample_timestamp=1.0))
        for obstacles in (
            obstacle_batch((1e5, -1e5, 1e5), velocity=(1e4, -1e4, 1e4)),
            obstacle_batch((0, 0, 0), velocity=(-1e4, 1e4, 0), covariance=1e3, radius=1e3),
        ):
            self.assertTrue(torch.isfinite(self.cost(obstacles)).all())

    def test_trajectory_gradient_is_finite_and_pushes_away(self):
        predicted = self.predicted.clone().requires_grad_(True)
        cost = self.cost(obstacle_batch((1, 0.2, 0)), predicted).mean()
        cost.backward()
        self.assertTrue(torch.isfinite(predicted.grad).all())
        self.assertGreater(float(predicted.grad[:, 1, 0].mean()), 0.0)


if __name__ == "__main__":
    unittest.main()
