"""Phase 8I invariants for candidate, score-label, and risk alignment."""

from __future__ import annotations

import unittest

import torch

from loss.dynamic_safety_loss import DynamicCollisionLoss
from loss.dynamic_types import DynamicLossConfig, DynamicObstacleBatch, RiskMetricsConfig
from loss.loss_function import DEPLossOutput
from tools.generate_phase8i_estimated_cache import phase8h_perception_config


class _FixedSampler:
    def grouped(self, fixed, predicted, batch):
        trajectories = torch.zeros(batch, 15, 30, 3, dtype=predicted.dtype)
        trajectories[..., 0] = torch.arange(15, dtype=predicted.dtype)[None, :, None]
        return trajectories, torch.linspace(1.7 / 30, 1.7, 30, dtype=predicted.dtype)


def _empty_obstacles(batch=1):
    return DynamicObstacleBatch(
        positions_world=torch.empty(batch, 0, 3),
        velocities_world=torch.empty(batch, 0, 3),
        position_covariances=torch.empty(batch, 0, 3, 3),
        radii=torch.empty(batch, 0),
        track_timestamps=torch.empty(batch, 0, dtype=torch.float64),
        sample_timestamps=torch.zeros(batch, dtype=torch.float64),
        confidence=torch.empty(batch, 0),
        valid_mask=torch.empty(batch, 0, dtype=torch.bool),
        dynamic_mask=torch.empty(batch, 0, dtype=torch.bool),
        observable_mask=torch.empty(batch, 0, dtype=torch.bool),
    )


class Phase8IScoreSemanticsTests(unittest.TestCase):
    def test_estimated_cache_uses_frozen_phase8h_range_image_mode(self):
        self.assertEqual(
            phase8h_perception_config().foreground_mode, "range_image_hybrid"
        )

    def test_score_and_endstate_use_identical_row_major_candidate_index(self):
        endstate = torch.zeros(1, 9, 3, 5)
        for vertical in range(3):
            for horizon in range(5):
                endstate[:, :, vertical, horizon] = vertical * 5 + horizon
        score = torch.arange(15, dtype=torch.float32).reshape(1, 3, 5)
        endstate_flat = endstate.permute(0, 2, 3, 1).reshape(15, 9)
        self.assertTrue(torch.equal(endstate_flat[:, 0], score.reshape(15)))

    def test_sampler_candidate_order_is_stable(self):
        sampler = _FixedSampler()
        first, times_first = sampler.grouped(torch.zeros(15, 3, 3), torch.zeros(15, 3, 3), 1)
        second, times_second = sampler.grouped(torch.zeros(15, 3, 3), torch.zeros(15, 3, 3), 1)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(times_first, times_second))
        self.assertTrue(torch.equal(first[0, :, 0, 0], torch.arange(15).float()))

    def test_selected_complete_trajectory_uses_selected_score_index(self):
        trajectories, _ = _FixedSampler().grouped(
            torch.zeros(15, 3, 3), torch.zeros(15, 3, 3), 1
        )
        scores = torch.tensor([4, 3, 2, 1, 0] + [5] * 10, dtype=torch.float32)
        selected = int(scores.argmin())
        self.assertEqual(selected, 4)
        self.assertTrue(torch.all(trajectories[0, selected, :, 0] == selected))

    def test_score_label_is_detached_and_not_an_input(self):
        base = torch.arange(15, dtype=torch.float32, requires_grad=True)
        zero = base * 0
        output = DEPLossOutput(
            smooth_cost=base, static_safety_cost=zero, guidance_cost=zero,
            dynamic_safety_cost=zero, raw_dynamic_safety_cost=zero,
            dynamic_training_objective=zero.mean(), dynamic_diagnostics=None,
        )
        label = output.detached_score_label()
        self.assertFalse(label.requires_grad)
        self.assertEqual(label.shape, (15,))

    def test_no_target_dynamic_cost_is_exactly_zero(self):
        config = DynamicLossConfig(
            enabled=True, weight=0.1, uav_radius=0.3, default_obstacle_radius=0.3,
            covariance_sigma=2.0, covariance_growth_rate=0.05, temperature=0.2,
            time_discount=0.2, max_track_age=0.5, obstacle_aggregation="max",
            eval_points=30, target_source="recorded_future_gt",
        )
        risk = RiskMetricsConfig(
            clearance_threshold=0.0, cost_epsilon=0.4804530139182014,
            cvar_fraction=4 / 15, hard_window_threshold=0.15,
        )
        loss = DynamicCollisionLoss(_FixedSampler(), config, risk)
        value, diagnostics = loss(
            torch.zeros(15, 3, 3), torch.zeros(15, 3, 3), _empty_obstacles()
        )
        self.assertTrue(torch.equal(value, torch.zeros_like(value)))
        self.assertTrue(diagnostics.candidate_safe_mask.all())

    def test_time_grid_is_seconds_and_metric_trajectory_is_meters(self):
        trajectory, times = _FixedSampler().grouped(
            torch.zeros(15, 3, 3), torch.zeros(15, 3, 3), 1
        )
        self.assertAlmostEqual(float(times[-1]), 1.7, places=6)
        self.assertEqual(tuple(trajectory.shape), (1, 15, 30, 3))


if __name__ == "__main__":
    unittest.main()
