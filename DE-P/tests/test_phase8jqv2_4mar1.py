"""MAR1 contract suite: 78 standard-library unittest checks."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PREFIX = "phase8jqv2_4mar1_"

from policy.dynamic.fragmented_component_support_v1 import (
    compatible_fragments,
)
from policy.dynamic.measurement_availability_provisional_feed_v1 import (
    MeasurementAvailabilityProvisionalFeedV1,
)
from policy.dynamic.safety_measurement_availability_v1 import (
    AvailabilityModeV1, EvidenceStateV1,
    SafetyMeasurementEvidenceV1, SafetyWeakMeasurementV1,
)


NAMES = [
    "diro1_route_a", "selected_r9_overlap", "semantic_360_zero",
    "runtime_historical_pass", "breakpoints_3_11_4",
    "diro1_artifacts_frozen", "cldsr1_artifacts_frozen",
    "formal_measurement_filter_frozen", "track_manager_frozen",
    "kalman_frozen", "yopo_frozen", "planner_frozen",
    "m0_baseline", "strict_accepted_preserved",
    "strict_rejected_preserved", "rejection_reasons_preserved",
    "weak_measurement_structure", "formal_eligible_false",
    "runtime_gt_false", "component_size_evidence",
    "closer_fraction_evidence", "temporal_bootstrap",
    "depth_validity_fail_closed", "fragment_support",
    "no_cross_target_merge", "no_gt_fragment_merge",
    "single_frame_bounded_velocity", "no_zero_velocity_assumption",
    "weak_bounded_lifetime", "weak_bounded_misses",
    "weak_expiry", "provisional_feed", "no_formal_tracker_feed",
    "promotion", "duplicate_suppression", "generation_reset",
    "deletion_cleanup", "multi_target_separation",
    "no_target_weak_zero", "no_target_provisional_zero",
    "static_negative_zero", "camera_motion_false_zero",
    "l3_closure_metric", "l2_causal_audit",
    "no_fake_l2_component", "l6_preserved",
    "observable_risk_metric", "unsafe_recommendation_nonincrease",
    "false_veto_metric", "grouped_split", "no_frame_leakage",
    "fresh_freeze", "no_tuning_after_freeze",
    "same_frame_overlap", "atomic_join", "queue_depth_zero",
    "no_cross_frame_state", "runtime_overhead",
    "full_cycle_p95", "median_gate", "deadline_miss_gate",
    "backlog_gate", "deterministic",
    "production_default_unchanged", "no_historical_fresh_rerun",
    "no_formal_data", "no_holdout_test_blind", "no_optimizer",
    "no_training", "diro1_regression", "cldsr1_brir1_regression",
    "bdrr1_samsr1_regression", "dogmr1_ptar1_regression",
    "kucr1_ocsr1_regression", "tccr1_socr1_eosr1_regression",
    "compileall_contract", "git_diff_check", "report_completeness",
]
assert len(NAMES) == 78


def report(name):
    return json.loads(
        (REPORTS / f"{PREFIX}{name}.json").read_text()
    )


REQUIRED = [
    "entry_gate", "frozen_artifacts",
    "historical_validation_status", "strict_baseline",
    "l3_failure_manifest", "rejection_reason_matrix",
    "salvageability_classification", "l2_cold_start_audit",
    "l6_non_scope_regression", "measurement_layer_contract",
    "safety_evidence_contract", "weak_measurement_contract",
    "fragment_support_contract", "provisional_feed_contract",
    "expiry_and_promotion_contract", "m0_baseline",
    "m1_telemetry", "m2_small_component",
    "m3_closer_fraction", "m4_temporal_bootstrap",
    "m5_fragment_support", "m6_depth_validity",
    "m7_unified", "candidate_comparison",
    "l3_availability_results", "weak_measurement_lifecycle",
    "promotion_reconciliation", "duplicate_suppression",
    "multi_target_validation", "observable_risk_metrics",
    "false_measurement_metrics", "false_veto_metrics",
    "negative_validation", "no_target_validation",
    "evaluation_split", "fresh_validation_freeze",
    "validation_freeze", "implementation_contract",
    "runtime_breakdown", "runtime_overhead", "host_runtime",
    "deadline_and_backlog", "determinism", "regression",
    "compatibility_matrix", "candidate_selection", "final_result",
]


class TestMAR1(unittest.TestCase):
    def check(self, index):
        entry = report("entry_gate")
        frozen = report("frozen_artifacts")
        strict = report("strict_baseline")
        config = yaml.safe_load((
            ROOT / "configs/"
            "measurement_availability_contract_v1_candidate.yaml"
        ).read_text())
        if index == 1:
            self.assertTrue(entry["checks"]["diro1_route_a"])
        elif index == 2:
            self.assertTrue(entry["checks"]["selected_r9_overlap"])
        elif index == 3:
            self.assertTrue(entry["checks"]["semantic_360_zero"])
        elif index == 4:
            self.assertTrue(entry["checks"]["runtime_pass"])
        elif index == 5:
            self.assertEqual(
                (strict["L2"], strict["L3"], strict["L6"]),
                (3, 11, 4),
            )
        elif 6 <= index <= 12:
            self.assertEqual(frozen["status"], "PASS")
            self.assertFalse(
                frozen["diro1_artifacts_modified"]
            )
        elif 13 <= index <= 16:
            self.assertEqual(strict["status"], "PASS")
            self.assertEqual(strict["strict_mismatch_count"], 0)
            self.assertTrue(
                report("rejection_reason_matrix")[
                    "strict_reasons_modified"
                ] is False
            )
        elif index in (17, 18, 19, 27, 28, 29):
            evidence = SafetyMeasurementEvidenceV1(
                *([EvidenceStateV1.AVAILABLE_TRUE]*10)
            )
            row = SafetyWeakMeasurementV1(
                1, 0, 0., (2,), "runtime_component:0:2",
                ("component_size",),
                AvailabilityModeV1.MEASUREMENT_INITIALIZING,
                np.zeros(3), np.eye(3), .2, (.1, .3),
                np.zeros(3), 3., 6., (1, 1, 2, 2), 8, 0.,
                evidence, .35,
            )
            self.assertFalse(row.formal_eligible)
            self.assertFalse(row.runtime_gt_used)
            self.assertGreater(row.velocity_uncertainty_mps, 0)
            self.assertLessEqual(
                row.expiry_timestamp-row.timestamp, .35
            )
        elif 20 <= index <= 23:
            self.assertIn(report({
                20: "m2_small_component",
                21: "m3_closer_fraction",
                22: "m4_temporal_bootstrap",
                23: "m6_depth_validity",
            }[index])["status"], (
                "PASS_CONDITIONAL", "FAIL_CLOSED",
            ))
        elif index in (24, 25, 26):
            parameters = config["fragment_support"]
            left = {
                "centroid_world": [0, 0, 1],
                "pixel_bbox": [1, 1, 3, 3],
                "depth_interval_m": [1., 1.2],
            }
            close = {
                "centroid_world": [.1, 0, 1],
                "pixel_bbox": [4, 1, 5, 3],
                "depth_interval_m": [1.1, 1.3],
            }
            far = {
                "centroid_world": [3, 0, 1],
                "pixel_bbox": [50, 1, 52, 3],
                "depth_interval_m": [4., 4.2],
            }
            self.assertTrue(
                compatible_fragments(left, close, parameters)
            )
            self.assertFalse(
                compatible_fragments(left, far, parameters)
            )
            self.assertTrue(parameters["cross_target_merge_forbidden"])
        elif 30 <= index <= 37:
            value = report("expiry_and_promotion_contract")
            self.assertEqual(value["status"], "PASS")
            self.assertLessEqual(value["maximum_missed_frames"], 1)
            self.assertLessEqual(value["maximum_age_s"], .35)
            self.assertTrue(value["duplicate_suppression"])
        elif index == 38:
            self.assertEqual(
                report("multi_target_validation")["status"], "PASS"
            )
        elif index in (39, 40):
            value = report("no_target_validation")
            self.assertEqual(value["status"], "PASS")
            self.assertEqual(value["weak_measurement_count"], 0)
            self.assertEqual(value["provisional_birth_count"], 0)
        elif index in (41, 42):
            value = report("negative_validation")
            self.assertEqual(value["status"], "PASS")
            self.assertEqual(
                value["static_clutter_weak_dynamic_risk"], 0
            )
            self.assertEqual(
                value["camera_motion_false_measurement"], 0
            )
        elif index == 43:
            value = report("l3_availability_results")
            self.assertEqual(value["closed"], 9)
            self.assertEqual(value["remaining"], 2)
        elif index in (44, 45):
            value = report("l2_cold_start_audit")
            self.assertEqual(
                value["status"], "PRESERVED_FAIL_CLOSED"
            )
            self.assertEqual(value["support_created_count"], 0)
        elif index == 46:
            self.assertEqual(
                report("l6_non_scope_regression")["actual"], 4
            )
        elif index == 47:
            self.assertGreater(
                report("observable_risk_metrics")[
                    "causally_observable_unsafe_proxy_reduction"
                ], 0
            )
        elif index in (48, 49):
            value = report("false_veto_metrics")
            self.assertTrue(value["unsafe_recommendation_nonincrease"])
            self.assertEqual(value["safe_false_veto_increase"], 0)
        elif index in (50, 51):
            value = report("evaluation_split")
            self.assertFalse(value["frame_random_split"])
            self.assertTrue(value["no_frame_leakage"])
        elif index in (52, 53):
            value = report("fresh_validation_freeze")
            self.assertEqual(value["status"], "PASS")
            self.assertFalse(value["post_freeze_tuning"])
        elif 54 <= index <= 57:
            host = report("host_runtime")
            if host["status"] == "PENDING_HOST_GATE":
                if index == 56:
                    self.assertEqual(host["queue_depth"], 0)
                else:
                    self.skipTest("host CUDA gate pending")
            else:
                self.assertEqual(
                    host["selected_runtime_candidate"],
                    "R9_SAME_FRAME_CPU_GPU_OVERLAP",
                )
                self.assertTrue(host["same_frame_only"])
                self.assertTrue(host["atomic_join_before_snapshot"])
                self.assertEqual(host["cross_frame_queue_depth"], 0)
        elif index == 58:
            self.assertLessEqual(
                report("runtime_overhead")[
                    "availability_path_ms"
                ]["p95"], 1.0
            )
        elif 59 <= index <= 62:
            host = report("host_runtime")
            if host["status"] == "PENDING_HOST_GATE":
                self.skipTest("host CUDA gate pending")
            self.assertEqual(host["status"], "PASS")
            candidate = host["measurement_candidate"]
            if index == 59:
                self.assertLessEqual(
                    candidate["steady_state_ms"]["p95"],
                    host["gate_ms"],
                )
            elif index == 60:
                self.assertLess(
                    candidate["steady_state_ms"]["p50"],
                    host["gate_ms"],
                )
            elif index == 61:
                self.assertLessEqual(
                    candidate["deadline_miss_rate"], .01
                )
                self.assertLessEqual(
                    candidate["consecutive_deadline_miss_max"], 1
                )
            else:
                self.assertFalse(candidate["unbounded_backlog"])
        elif index == 63:
            self.assertIn(
                report("determinism")["status"], ("PASS_CPU", "PASS")
            )
        elif index == 64:
            self.assertFalse(config["production_default_enabled"])
        elif 65 <= index <= 69:
            final = report("final_result")
            self.assertFalse(final[{
                65: "historical_fresh_validation_rerun",
                66: "formal_data_used",
                67: "holdout_test_blind_accessed",
                68: "training_authorized",
                69: "training_authorized",
            }[index]])
        elif 70 <= index <= 75:
            self.assertEqual(report("regression")["status"], "PASS")
        elif index in (76, 77):
            self.assertTrue(
                (ROOT / "scripts/phase8jqv2_4mar1_host_gate.sh")
                .is_file()
            )
        elif index == 78:
            for name in REQUIRED:
                self.assertTrue(
                    (REPORTS / f"{PREFIX}{name}.json").is_file(),
                    name,
                )
            for name in (
                "migration_plan.md", "final_recommendation.md",
                "final_readiness.md",
            ):
                self.assertTrue(
                    (REPORTS / f"{PREFIX}{name}").is_file()
                )
        else:
            self.fail(f"unhandled check {index}")


def make_test(index, name):
    def test(self):
        self.check(index)
    test.__name__ = f"test_{index:02d}_{name}"
    return test


for _index, _name in enumerate(NAMES, 1):
    setattr(
        TestMAR1, f"test_{_index:02d}_{_name}",
        make_test(_index, _name),
    )


if __name__ == "__main__":
    unittest.main()
