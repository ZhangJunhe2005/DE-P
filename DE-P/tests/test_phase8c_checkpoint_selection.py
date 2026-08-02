import unittest

from tools.phase8c_checkpoint_selection import CheckpointSelector


CONFIG = {
    "balanced_hard_constraints": {
        "top1_collision_max": 0.05,
        "no_target_dynamic_tolerance": 0.0,
        "static_warning_ratio": 1.10,
        "static_hard_ratio": 1.15,
        "estimated_gt_dynamic_cvar_gap_max": 0.25,
        "estimated_gt_top1_collision_gap_max": 0.10,
    },
    "balanced_weights": {
        "dynamic_cvar": 1.0,
        "selected_clearance": 1.0,
        "oracle_regret": 0.5,
        "static_cost": 0.25,
        "score_loss": 0.1,
    },
    "balanced_scales": {
        "dynamic_cvar": 0.5,
        "selected_clearance": 0.3,
        "oracle_regret": 0.1,
    },
}


def metrics(**updates):
    result = {
        "top1_collision_fraction": 0.02,
        "dynamic_cvar": 0.2,
        "selected_min_clearance": 0.1,
        "raw_dynamic_mean": 0.05,
        "oracle_regret": 0.02,
        "static_cost": 2.0,
        "score_loss": 1.0,
        "no_target_dynamic_mean": 0.0,
        "static_smoke_pass": True,
        "static_metrics_finite": True,
        "estimated_gt_dynamic_cvar_gap": 0.02,
        "estimated_gt_top1_collision_gap": 0.01,
        "checkpoint_suite": "valid_estimated",
        "suites": {
            "valid_estimated": {}, "valid_gt": {}, "valid_static": {},
        },
    }
    result.update(updates)
    return result


class CheckpointSelectionTests(unittest.TestCase):
    def setUp(self):
        self.selector = CheckpointSelector(CONFIG, {"static_cost": 2.0, "score_loss": 1.0})

    def test_dynamic_priority_is_lexicographic_not_total_loss(self):
        first = self.selector.consider(metrics(top1_collision_fraction=0.02, dynamic_cvar=0.1))
        second = self.selector.consider(metrics(top1_collision_fraction=0.01, dynamic_cvar=10.0))
        self.assertTrue(first["dynamic_improved"])
        self.assertTrue(second["dynamic_improved"])

    def test_balanced_hard_constraints_precede_score(self):
        result = self.selector.consider(metrics(static_cost=2.31))
        self.assertFalse(result["balanced_feasible"])
        self.assertFalse(result["balanced_improved"])
        self.assertFalse(result["balanced_constraints"]["static_regression"])

    def test_no_target_must_be_exact_zero(self):
        result = self.selector.consider(metrics(no_target_dynamic_mean=1e-17))
        self.assertFalse(result["balanced_feasible"])

    def test_state_round_trip_preserves_resume_ordering(self):
        self.selector.consider(metrics())
        state = self.selector.state_dict()
        resumed = CheckpointSelector(CONFIG, {"static_cost": 2.0, "score_loss": 1.0})
        resumed.load_state_dict(state)
        self.assertEqual(resumed.state_dict()["best_dynamic_key"], state["best_dynamic_key"])
        self.assertEqual(
            resumed.state_dict()["best_balanced_score"], state["best_balanced_score"]
        )


if __name__ == "__main__":
    unittest.main()
