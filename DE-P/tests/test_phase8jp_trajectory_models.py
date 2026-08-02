import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from loss.dynamic_types import DynamicObstacleBatch
from tools.analyze_piecewise_trajectory_oracle import (
    ACCELERATION_LIMIT,
    HORIZON,
    JERK_LIMIT,
    VELOCITY_LIMIT,
    classify_outcome,
    evaluate_trajectory_samples,
    oracle_config,
    report_scheme,
    waypoint_world,
)
from tools.phase8jp_trajectory_models import (
    integrate_piecewise_constant_jerk,
    quintic_coefficients,
    sample_piecewise_quintic,
)


def state(position=(0, 0, 0), velocity=(0, 0, 0),
          acceleration=(0, 0, 0)):
    return torch.tensor(
        [position, velocity, acceleration], dtype=torch.float64
    ).T


class PositiveDistance:
    def get_distance_cost(self, trajectories, map_id):
        shape = trajectories.shape[:-1]
        return torch.zeros(shape), torch.ones(shape)


class Phase8JPTrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.start = state()[None, None]
        self.midpoint = state((1, 0.5, 0), (0.5, 0, 0))[None, None]
        self.end = state((2, 0, 0))[None, None]

    def test_01_quintic_start_state_is_exact(self):
        coefficient = quintic_coefficients(
            self.start, self.midpoint, HORIZON * 0.5
        )
        self.assertTrue(torch.allclose(coefficient[..., 0], self.start[..., 0]))
        self.assertTrue(torch.allclose(coefficient[..., 1], self.start[..., 1]))
        self.assertTrue(torch.allclose(
            2 * coefficient[..., 2], self.start[..., 2]
        ))

    def test_02_piecewise_position_is_continuous(self):
        first = sample_piecewise_quintic(
            self.start, self.midpoint, self.end,
            HORIZON * 0.5, HORIZON * 0.5, 30,
        )
        junction = first.position[..., 14, :]
        self.assertTrue(torch.allclose(junction, self.midpoint[..., :, 0]))

    def test_03_piecewise_velocity_is_continuous(self):
        samples = sample_piecewise_quintic(
            self.start, self.midpoint, self.end,
            HORIZON * 0.5, HORIZON * 0.5, 30,
        )
        self.assertTrue(torch.allclose(
            samples.velocity[..., 14, :], self.midpoint[..., :, 1]
        ))

    def test_04_piecewise_acceleration_is_continuous(self):
        samples = sample_piecewise_quintic(
            self.start, self.midpoint, self.end,
            HORIZON * 0.5, HORIZON * 0.5, 30,
        )
        self.assertTrue(torch.allclose(
            samples.acceleration[..., 14, :], self.midpoint[..., :, 2],
            atol=1e-10,
        ))

    def test_05_time_split_sums_to_formal_horizon(self):
        samples = sample_piecewise_quintic(
            self.start, self.midpoint, self.end,
            HORIZON * 0.3, HORIZON * 0.7, 30,
        )
        self.assertAlmostEqual(float(samples.times[-1]), HORIZON, places=6)
        self.assertEqual(samples.times.numel(), 30)

    def test_06_waypoint_body_to_world_transform(self):
        start = state((10, 20, 30))[None]
        rotation = torch.eye(3, dtype=torch.float64)[None]
        raw = torch.zeros(1, 1, 3, dtype=torch.float64)
        transformed = waypoint_world(start, rotation, raw)
        expected = torch.tensor([[[12.5, 20, 30]]], dtype=torch.float64)
        self.assertTrue(torch.equal(transformed, expected))

    def test_07_brake_trajectory_respects_audit_bounds(self):
        start = state(velocity=(2, 0, 0))[None, None]
        jerk = torch.zeros(1, 1, 8, 3, dtype=torch.float64)
        jerk[..., :4, 0] = -1
        jerk[..., 4:, 0] = 1
        samples = integrate_piecewise_constant_jerk(start, jerk, HORIZON, 4)
        self.assertLessEqual(
            float(torch.linalg.vector_norm(samples.velocity, dim=-1).max()),
            VELOCITY_LIMIT,
        )
        self.assertLessEqual(
            float(torch.linalg.vector_norm(samples.acceleration, dim=-1).max()),
            ACCELERATION_LIMIT,
        )
        self.assertLessEqual(
            float(torch.linalg.vector_norm(samples.jerk, dim=-1).max()),
            JERK_LIMIT,
        )

    def test_08_hover_has_no_instantaneous_motion(self):
        samples = integrate_piecewise_constant_jerk(
            self.start, torch.zeros(1, 1, 8, 3, dtype=torch.float64),
            HORIZON, 4,
        )
        self.assertTrue(torch.equal(samples.position, torch.zeros_like(
            samples.position
        )))
        self.assertTrue(torch.equal(samples.velocity, torch.zeros_like(
            samples.velocity
        )))

    def test_09_static_only_dynamic_cost_is_exactly_zero(self):
        samples = integrate_piecewise_constant_jerk(
            self.start, torch.zeros(1, 1, 8, 3, dtype=torch.float64),
            HORIZON, 4,
        )
        trainer = SimpleNamespace(
            device=torch.device("cpu"),
            dynamic_dep_loss=SimpleNamespace(
                safety_loss=PositiveDistance(),
            ),
        )
        metrics = evaluate_trajectory_samples(
            trainer, samples, torch.zeros(1, dtype=torch.long),
            DynamicObstacleBatch.empty(1),
        )
        self.assertTrue(torch.equal(
            metrics["dynamic_cost"], torch.zeros_like(metrics["dynamic_cost"])
        ))

    def test_10_candidate_permutation_preserves_samples(self):
        start = self.start.expand(1, 3, -1, -1).clone()
        jerk = torch.arange(1 * 3 * 8 * 3, dtype=torch.float64).reshape(
            1, 3, 8, 3
        ) / 100
        permutation = torch.tensor([2, 0, 1])
        first = integrate_piecewise_constant_jerk(start, jerk, HORIZON, 2)
        second = integrate_piecewise_constant_jerk(
            start[:, permutation], jerk[:, permutation], HORIZON, 2
        )
        self.assertTrue(torch.allclose(
            first.position[:, permutation], second.position
        ))

    def test_11_direct_collocation_state_is_continuous(self):
        jerk = torch.randn(1, 1, 8, 3, dtype=torch.float64)
        full = integrate_piecewise_constant_jerk(
            self.start, jerk, HORIZON, 4
        )
        first = integrate_piecewise_constant_jerk(
            self.start, jerk[..., :4, :], HORIZON / 2, 4
        )
        boundary = torch.stack((
            first.position[..., -1, :],
            first.velocity[..., -1, :],
            first.acceleration[..., -1, :],
        ), dim=-1)
        second = integrate_piecewise_constant_jerk(
            boundary, jerk[..., 4:, :], HORIZON / 2, 4
        )
        self.assertTrue(torch.allclose(
            full.position[..., -1, :], second.position[..., -1, :],
            atol=1e-10,
        ))

    def test_12_physical_limits_are_declared_and_positive(self):
        self.assertGreater(VELOCITY_LIMIT, 0)
        self.assertGreater(ACCELERATION_LIMIT, 0)
        self.assertGreater(JERK_LIMIT, 0)
        self.assertEqual(
            oracle_config()["jerk_limit_status"].split(";")[0], "audit-only"
        )

    def test_13_instance_masks_are_forbidden(self):
        self.assertIs(oracle_config()["instance_mask_used"], False)

    def test_14_production_test_is_forbidden(self):
        self.assertIs(oracle_config()["production_test_used"], False)

    def test_15_future_gt_role_is_offline_evaluation_only(self):
        self.assertEqual(
            oracle_config()["future_gt_role"], "offline safety evaluator only"
        )

    def test_16_optimizer_failure_is_not_intrinsic_failure(self):
        failed = {"success": False}
        self.assertEqual(
            classify_outcome(
                failed, failed, failed, failed, failed,
                optimizer_failure=True,
            ),
            "optimizer_failure",
        )

    def test_17_resume_configuration_hash_is_deterministic(self):
        self.assertEqual(
            oracle_config()["config_sha256"],
            oracle_config()["config_sha256"],
        )

    def test_18_scheme_report_contains_required_contract(self):
        result = {
            "success": True,
            "physical": True,
            "optimizer_termination": "projected_gradient_complete",
        }
        record = {
            "sequence_id": "sequence",
            "frame_index": 0,
            "map_id": 12,
            "scenario": "crossing",
            "category": "hard_risk",
            "actor_count": 1,
            "optimizer_failure": False,
            "offline_latency_seconds": {"o5a": 0.1},
            "o5a": result,
        }
        report = report_scheme([record], "o5a", 142)
        for key in (
            "joint_coverage_failure_fraction",
            "physical_feasibility_rate_of_recovered",
            "optimizer_failure_count",
            "offline_latency_seconds_mean_per_window",
            "records",
        ):
            self.assertIn(key, report)


if __name__ == "__main__":
    unittest.main()
