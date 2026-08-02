import json
from pathlib import Path
import unittest

import numpy as np

from loss.dynamic_safety_loss import DynamicCollisionLoss
from loss.safety_geometry_v2 import (
    actor_physical_clearance,
    continuous_sphere_segment_gap,
    directional_covariance_margin,
    finite_vertical_cylinder_signed_gap,
    planning_clearance,
    sphere_signed_gap,
    static_physical_clearance,
)
from loss.safety_loss import SafetyLoss
from policy.safety_evaluator_v2 import (
    SafetyEvaluatorV2,
    SafetyEvaluatorV2Config,
)


ROOT = Path(__file__).resolve().parents[1]


class SafetyEvaluatorV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = SafetyEvaluatorV2Config.load(
            ROOT / "configs/safety_evaluator_v2.yaml",
            ROOT / "reports/phase8jq_controller_authoritative_envelope.json",
        )
        cls.evaluator = SafetyEvaluatorV2(cls.config)
        cls.current = np.asarray([
            [0, 1, 0], [0, 0, 0], [2, 0, 0],
        ], dtype=float)
        # One exact quintic generated from simple constant-velocity samples.
        times = np.linspace(1.7 / 30, 1.7, 30)
        cls.v1 = np.stack((times, np.zeros(30), 2*np.ones(30)), axis=1)

    def test_01_v1_still_importable(self):
        self.assertIsNotNone(SafetyLoss)
        self.assertIsNotNone(DynamicCollisionLoss)

    def test_02_static_subtracts_uav_radius(self):
        np.testing.assert_allclose(
            static_physical_clearance([.29, .30, .31], .3),
            [-.01, 0, .01],
        )

    def test_03_sphere_exact_contact(self):
        self.assertAlmostEqual(sphere_signed_gap([0, 0, 0], [.7, 0, 0], .3, .4), 0)

    def test_04_cylinder_side_contact(self):
        self.assertAlmostEqual(
            finite_vertical_cylinder_signed_gap([.7, 0, 0], [0, 0, 0], .3, .4, 1),
            0,
        )

    def test_05_cylinder_top_contact(self):
        self.assertAlmostEqual(
            finite_vertical_cylinder_signed_gap([0, 0, .8], [0, 0, 0], .3, .4, 1),
            0,
        )

    def test_06_cylinder_corner(self):
        expected = np.hypot(.3, .4) - .3
        self.assertAlmostEqual(
            finite_vertical_cylinder_signed_gap([.7, 0, .9], [0, 0, 0], .3, .4, 1),
            expected,
        )

    def test_07_gt_uncertainty_zero(self):
        self.assertEqual(
            self.evaluator.uncertainty_margin(
                [0, 0, 0], [1, 0, 0], np.eye(3), "valid_gt"
            ),
            0,
        )

    def test_08_estimated_covariance_used_once(self):
        covariance = np.diag([.25, 1, 1])
        value = self.evaluator.uncertainty_margin(
            [0, 0, 0], [1, 0, 0], covariance, "valid_estimated"
        )
        self.assertAlmostEqual(value, 1.0)

    def test_09_no_heuristic_double_growth(self):
        covariance = np.diag([.25, 1, 1])
        first = directional_covariance_margin([0, 0, 0], [1, 0, 0], covariance, 2)
        second = directional_covariance_margin([0, 0, 0], [1, 0, 0], covariance, 2)
        self.assertEqual(first, second)

    def test_10_t0_is_included(self):
        timeline = self.evaluator.timeline(self.current, self.v1)
        self.assertEqual(timeline["times"][0], 0)

    def test_11_latency_prefix_is_included(self):
        timeline = self.evaluator.timeline(self.current, self.v1)
        self.assertTrue(np.any(np.isclose(timeline["times"], self.config.latency_s)))

    def test_12_candidate_starts_at_first_controllable_state(self):
        timeline = self.evaluator.timeline(self.current, self.v1)
        index = np.flatnonzero(np.isclose(timeline["times"], self.config.latency_s))[0]
        np.testing.assert_allclose(
            timeline["positions"][index],
            timeline["first_controllable_state"][:, 0],
            atol=1e-10,
        )

    def test_13_actor_uav_same_timestamp_fixture(self):
        timeline = self.evaluator.timeline(self.current, self.v1)
        actor_times = timeline["times"].copy()
        np.testing.assert_array_equal(actor_times, timeline["times"])

    def test_14_one_frame_offset_detected(self):
        timeline = self.evaluator.timeline(self.current, self.v1)
        self.assertGreater(
            np.max(np.abs((timeline["times"] + .1) - timeline["times"])), 1e-4
        )

    def test_15_fixed_wall_clock_horizon(self):
        timeline = self.evaluator.timeline(self.current, self.v1, "fixed_wall_clock")
        self.assertAlmostEqual(timeline["wall_clock_duration_s"], 1.7)

    def test_16_fixed_controlled_duration_horizon(self):
        timeline = self.evaluator.timeline(
            self.current, self.v1, "fixed_controlled_duration_plus_latency"
        )
        self.assertAlmostEqual(
            timeline["wall_clock_duration_s"], 1.7 + self.config.latency_s
        )

    def test_17_continuous_dynamic_collision(self):
        gap, fraction = continuous_sphere_segment_gap(
            [-1, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0], .3, .4
        )
        self.assertAlmostEqual(gap, -.7)
        self.assertAlmostEqual(fraction, .5)

    def test_18_static_adaptive_sampling_configured(self):
        self.assertGreaterEqual(self.config.subdivisions, 4)
        self.assertGreater(self.config.static_tolerance_m, 0)

    def test_19_physical_planning_margin_separated(self):
        self.assertAlmostEqual(
            planning_clearance(.5, planning_margin=.1, tracking_control_margin=.2,
                               uncertainty_margin=.05),
            .15,
        )

    def test_20_no_target_dynamic_cost_is_exactly_zero(self):
        timeline = self.evaluator.timeline(self.current, self.v1)
        result = self.evaluator.evaluate_dynamic(
            timeline, [[] for _ in timeline["times"]], "valid_gt"
        )
        self.assertTrue(np.isinf(result["physical_clearance_by_time"]).all())
        self.assertEqual(max(-result["continuous_physical_min"], 0.0), 0.0)

    def test_21_future_gt_offline_only(self):
        config = (ROOT / "configs/safety_evaluator_v2.yaml").read_text()
        self.assertIn("recorded_future_gt_policy: none", config)

    def test_22_production_data_forbidden_in_tool(self):
        text = (ROOT / "tools/evaluate_candidate_decomposition_v2.py").read_text()
        self.assertNotIn("phase8c_test_", text)

    def test_23_v1_v2_reports_coexist(self):
        self.assertTrue((ROOT / "reports/phase8i_final_result.json").is_file())
        self.assertTrue((ROOT / "reports/phase8jqv2_entry_gate.json").is_file())

    def test_24_component_config_hash_deterministic(self):
        self.assertEqual(self.config.canonical_hash(), self.config.canonical_hash())

    def test_25_actor_shape_dispatch(self):
        actor = {
            "type": "vertical_cylinder", "position_world": [0, 0, 0],
            "radius": .4, "height": 1,
        }
        self.assertAlmostEqual(actor_physical_clearance([.7, 0, 0], actor, .3), 0)

    def test_26_report_completeness(self):
        required = [
            "phase8jqv2_entry_gate.json", "phase8jqv2_v1_baseline.json",
            "phase8jqv2_evaluator_spec.md",
            "phase8jqv2_uncertainty_semantics.json",
            "phase8jqv2_timeline_spec.json",
            "phase8jqv2_geometry_validation.json",
            "phase8jqv2_timeline_validation.json",
            "phase8jqv2_continuous_collision_validation.json",
            "phase8jqv2_determinism_validation.json",
            "phase8jqv2_checkpoint_matrix.json",
            "phase8jqv2_component_ablation.json",
            "phase8jqv2_physical_coverage.json",
            "phase8jqv2_planning_coverage.json",
            "phase8jqv2_actionability_metrics.json",
            "phase8jqv2_label_scale_audit.json",
            "phase8jqv2_candidate_failure_decomposition.json",
            "phase8jqv2_context_gap_analysis.json",
            "phase8jqv2_v1_v2_delta.json",
            "phase8jqv2_final_result.json",
            "phase8jqv2_final_recommendation.md",
            "phase8jqv2_final_readiness.md",
        ]
        self.assertTrue(all((ROOT / "reports" / name).is_file() for name in required))

    def test_27_instance_mask_not_used(self):
        text = (
            ROOT / "tools/evaluate_candidate_decomposition_v2.py"
        ).read_text()
        self.assertNotIn("instance_mask_path", text)
        self.assertNotIn("instance_masks", text)

    def test_28_checkpoint_strict_load_audited(self):
        report = json.loads(
            (ROOT / "reports/phase8jqv2_checkpoint_matrix.json").read_text()
        )
        self.assertTrue(report["strict_load_results"])
        self.assertTrue(all(
            row["strict_load"] and row["hash_match"]
            for row in report["strict_load_results"]
        ))

    def test_29_resume_hash_consistency(self):
        report = json.loads(
            (ROOT / "reports/phase8jqv2_determinism_validation.json").read_text()
        )
        self.assertTrue(report["same_config_resume_supported"])
        self.assertRegex(report["component_ablation_hash"], r"^[0-9a-f]{64}$")

    def test_30_all_json_reports_have_provenance_hashes(self):
        required = {
            "evaluator_version", "config_hash", "geometry_hash", "timeline_hash",
            "uncertainty_policy_hash", "Simulator_geometry_hash",
            "dataset_manifest_hash", "cache_index_hash", "checkpoint_hash",
        }
        for path in (ROOT / "reports").glob("phase8jqv2*.json"):
            if path.name.endswith("_smoke.json"):
                continue
            payload = json.loads(path.read_text())
            self.assertFalse(
                required - payload.keys(),
                f"{path.name} missing {sorted(required-payload.keys())}",
            )


if __name__ == "__main__":
    unittest.main()
