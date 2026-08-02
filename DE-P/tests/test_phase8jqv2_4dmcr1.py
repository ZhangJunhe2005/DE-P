"""DMCR1 contract and report gate (74 standard-library unittest cases)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from policy.dynamic.dynamic_measurement_outcome_v1 import (
    DynamicMeasurementOutcomeV1, DynamicMeasurementStatusV1,
    UnresolvedMeasurementRiskV1,
)
from policy.dynamic.historical_measurement_context_v1 import (
    HistoricalMeasurementContextV1,
)
from policy.dynamic.measurement_contract_shadow_consumer_v1 import (
    consume_outcome,
)
from policy.dynamic.near_field_safety_support_v1 import (
    NearFieldSafetySupportV1,
)


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PREFIX = "phase8jqv2_4dmcr1_"
REPORT_STEMS = (
    "entry_gate", "frozen_artifacts", "historical_validation_status",
    "depth_field_provenance", "depth_validity_semantics",
    "depth_contract_consistency", "near_field_sensor_contract",
    "frame24_audit", "frame25_audit", "remaining_case_comparison",
    "historical_track_context", "measurement_outcome_contract",
    "bounded_support_contract", "unresolved_risk_contract",
    "sensor_limit_contract", "risk_consumer_contract",
    "d0_baseline", "d1_depth_semantics", "d2_near_field_support",
    "d3_track_context", "d4_unresolved_risk", "d5_unified",
    "candidate_comparison", "yopo_near_field_visibility",
    "duplicate_risk_semantics", "shadow_consumer_validation",
    "evaluation_split", "fresh_validation_freeze",
    "validation_freeze", "negative_validation",
    "no_target_validation", "near_field_static_validation",
    "near_field_dynamic_validation", "implementation_contract",
    "runtime_overhead", "host_runtime", "deadline_and_backlog",
    "determinism", "regression", "compatibility_matrix",
    "candidate_selection", "final_result",
)
MARKDOWN_STEMS = (
    "migration_plan", "final_recommendation", "final_readiness",
)


def report(stem):
    return json.loads((REPORTS/f"{PREFIX}{stem}.json").read_text())


def support():
    return NearFieldSafetySupportV1(
        support_id=1, frame_index=1, timestamp=1.,
        source_component_ids=(1,),
        camera_frame="camera_optical", world_frame="world",
        near_field_clipped=False,
        occupied_depth_interval_m=(.2, .4),
        angular_support_rad=(-.1, .1, -.1, .1),
        position_set_min_world=(-.5, -.5, -.5),
        position_set_max_world=(.5, .5, .5),
        extent_bound_m=.5, valid_point_count=8,
        expiry_timestamp=1.1,
    )


def measurement(status):
    return DynamicMeasurementOutcomeV1(
        outcome_id=1, frame_index=1, timestamp=1.,
        status=status, resolution_status="TEST",
        source_component_ids=(1,),
        position_reference="CENTER_WORLD",
        position_world=np.zeros(3),
        covariance_world=np.eye(3),
        position_valid=True, measurement_valid=True,
        risk_present=True,
        formal_eligible=(
            status == DynamicMeasurementStatusV1
            .VALID_STRICT_MEASUREMENT
        ),
    )


def bounded():
    return DynamicMeasurementOutcomeV1(
        outcome_id=2, frame_index=1, timestamp=1.,
        status=DynamicMeasurementStatusV1.BOUNDED_SAFETY_SUPPORT,
        resolution_status="TEST", source_component_ids=(1,),
        position_reference="FINITE_WORLD_POSITION_SET",
        support=support(), support_valid=True, risk_present=True,
    )


def unresolved():
    context = HistoricalMeasurementContextV1(
        track_exists=False, track_id=None, generation=None,
        confirmed=False, dynamic=False, attention_authorized=False,
        last_direct_measurement_time=None, prediction_only_age=0,
        last_valid_geometry_bounds=None, last_safety_evidence=None,
        current_measurement_outcome="UNRESOLVED_MEASUREMENT_RISK",
    )
    risk = UnresolvedMeasurementRiskV1(
        source_evidence=("DEPTH_TRANSITION",),
        first_timestamp=1., last_timestamp=1.,
        track_context=context, depth_contract_state="INVALID",
        reason_code="CAUSAL_COMPONENT_WITHOUT_BOUNDED_SUPPORT",
        expiry_timestamp=1.1,
    )
    return DynamicMeasurementOutcomeV1(
        outcome_id=3, frame_index=1, timestamp=1.,
        status=(
            DynamicMeasurementStatusV1.UNRESOLVED_MEASUREMENT_RISK
        ),
        resolution_status="TEST", source_component_ids=(1,),
        unresolved_risk=risk, risk_present=True,
    )


class DMCR1Gate(unittest.TestCase):
    def check_index(self, index):
        entry = report("entry_gate")
        frozen = report("frozen_artifacts")
        final = report("final_result")
        if index == 1:
            self.assertTrue(entry["checks"]["mar1_route_c"])
        elif index == 2:
            self.assertTrue(entry["checks"]["closed_9_of_11"])
        elif index == 3:
            self.assertTrue(entry["checks"]["frame24_audit_required"])
        elif index == 4:
            self.assertTrue(entry["checks"]["frame25_hard_reject"])
        elif index == 5:
            self.assertTrue(entry["checks"]["historical_fresh_720_pass"])
        elif index == 6:
            self.assertFalse(frozen["mar1_artifacts_modified"])
        elif index == 7:
            self.assertFalse(frozen["diro1_artifacts_modified"])
        elif index == 8:
            self.assertFalse(
                frozen["strict_formal_measurement_filter_modified"]
            )
        elif index == 9:
            self.assertFalse(frozen["TrackManager_algorithm_modified"])
        elif index == 10:
            self.assertFalse(frozen["kalman_modified"])
        elif index == 11:
            self.assertFalse(frozen["static_yopo_modified"])
        elif index == 12:
            self.assertEqual(report("d0_baseline")["l3_closed"], "9/11")
        elif index == 13:
            self.assertIn(
                "raw_depth", report("depth_field_provenance")["fields"]
            )
        elif index == 14:
            self.assertFalse(report("depth_field_provenance")[
                "fields"]["sanitized_depth"]["clip_or_clamp"])
        elif index == 15:
            self.assertIn(
                ">=min_depth",
                report("depth_field_provenance")["fields"][
                    "valid_depth_mask"]["producer"],
            )
        elif index == 16:
            self.assertTrue(report("depth_contract_consistency")[
                "depth_in_contract_false_and_L1_true_possible"])
        elif index == 17:
            self.assertTrue(report("depth_validity_semantics")[
                "larger_is_more_invalid"])
        elif index == 18:
            self.assertIn(
                "marked_invalid",
                report("near_field_sensor_contract")["below_min"],
            )
        elif index == 19:
            self.assertIn(
                "marked_invalid",
                report("near_field_sensor_contract")["at_or_above_max"],
            )
        elif index == 20:
            self.assertIn(
                "invalid", report("depth_field_provenance")["zero_nan_inf"]
            )
        elif index == 21:
            self.assertTrue(report("depth_contract_consistency")[
                "producer_consumer_behavior_consistent"])
        elif index == 22:
            self.assertEqual(report("frame24_audit")["status"],
                             "PASS_EXPLICIT_SAFE_CONTRACT")
        elif index == 23:
            self.assertEqual(report("frame25_audit")["status"],
                             "PASS_EXPLICIT_SAFE_CONTRACT")
        elif index == 24:
            self.assertTrue(measurement(
                DynamicMeasurementStatusV1.VALID_STRICT_MEASUREMENT
            ).formal_eligible)
        elif index == 25:
            self.assertFalse(measurement(
                DynamicMeasurementStatusV1
                .VALID_SAFETY_WEAK_MEASUREMENT
            ).formal_eligible)
        elif index == 26:
            self.assertTrue(bounded().support_valid)
        elif index == 27:
            self.assertEqual(
                consume_outcome(unresolved()).semantic,
                "FAIL_CLOSED_REVIEW_REQUIRED",
            )
        elif index == 28:
            value = DynamicMeasurementOutcomeV1(
                outcome_id=4, frame_index=1, timestamp=1.,
                status=(
                    DynamicMeasurementStatusV1
                    .HARD_INVALID_NO_EVIDENCE
                ),
                resolution_status="NO_EVIDENCE",
            )
            self.assertFalse(value.risk_present)
        elif index == 29:
            value = DynamicMeasurementOutcomeV1(
                outcome_id=5, frame_index=1, timestamp=1.,
                status=DynamicMeasurementStatusV1.SENSOR_CONTRACT_LIMIT,
                resolution_status="LIMIT", risk_present=True,
            )
            self.assertEqual(
                consume_outcome(value).semantic,
                "SENSOR_LIMIT_REVIEW_REQUIRED",
            )
        elif index == 30:
            value = DynamicMeasurementOutcomeV1(
                outcome_id=6, frame_index=1, timestamp=1.,
                status=DynamicMeasurementStatusV1.INTERNAL_CONTRACT_ERROR,
                resolution_status="ERROR", risk_present=True,
            )
            self.assertEqual(
                consume_outcome(value).semantic,
                "FAIL_CLOSED_REVIEW_REQUIRED",
            )
        elif 31 <= index <= 33:
            value = bounded()
            self.assertIsNone((
                value.position_world, value.velocity_center_mps, value.shape
            )[index-31])
        elif index == 34:
            self.assertTrue(all(
                np.isfinite(value)
                for value in bounded().support.position_set_min_world
                + bounded().support.position_set_max_world
            ))
        elif index == 35:
            self.assertNotEqual(
                consume_outcome(unresolved()).semantic,
                "NO_ACTIVE_DYNAMIC_RISK",
            )
        elif index == 36:
            self.assertTrue(report("historical_track_context")[
                "frame24"]["read_only"])
        elif index == 37:
            self.assertFalse(report("historical_track_context")[
                "track_reactivated"])
        elif index == 38:
            self.assertEqual(final["formal_tracker_feed"], 0)
        elif index == 39:
            self.assertIn(
                report("frame24_audit")["outcome"]["status"],
                ("BOUNDED_SAFETY_SUPPORT",
                 "UNRESOLVED_MEASUREMENT_RISK"),
            )
        elif index == 40:
            self.assertIn(
                report("frame25_audit")["outcome"]["status"],
                ("BOUNDED_SAFETY_SUPPORT",
                 "UNRESOLVED_MEASUREMENT_RISK"),
            )
        elif index == 41:
            self.assertEqual(
                report("no_target_validation")["unresolved_risk"], 0
            )
        elif index == 42:
            self.assertEqual(report("near_field_static_validation")[
                "static_dynamic_measurement_count"], 0)
        elif index == 43:
            self.assertEqual(report("negative_validation")[
                "controls"]["camera_translation"], "NO_FALSE_DYNAMIC")
        elif index == 44:
            self.assertEqual(report("negative_validation")[
                "controls"]["multi_target"],
                "SOURCE_COMPONENTS_SEPARATE")
        elif index == 45:
            self.assertFalse(report("yopo_near_field_visibility")[
                "candidate_crossing_safety_guaranteed_by_depth_alone"])
        elif index == 46:
            self.assertIn(
                "never sum",
                report("duplicate_risk_semantics")["combination_rule"],
            )
        elif index == 47:
            self.assertEqual(report("evaluation_split")["status"],
                             "PASS_GROUPED")
        elif index == 48:
            self.assertFalse(report("evaluation_split")["frame_leakage"])
        elif index == 49:
            self.assertEqual(report("fresh_validation_freeze")["status"],
                             "PASS")
        elif index == 50:
            self.assertFalse(report("validation_freeze")[
                "post_fresh_tuning"])
        elif index == 51:
            self.assertFalse(report("fresh_validation_freeze")[
                "fresh_summary"]["runtime_gt_used"])
        elif 52 <= index <= 57:
            host = report("host_runtime")
            if host["status"] == "PENDING_HOST_GATE":
                self.skipTest("host CUDA gate pending")
            self.assertEqual(host["status"], "PASS")
            if index == 52:
                self.assertTrue(host["same_frame_only"])
            elif index == 53:
                self.assertTrue(host["atomic_join_before_snapshot"])
            elif index == 54:
                self.assertEqual(host["queue_depth"], 0)
            elif index == 55:
                self.assertLessEqual(report("runtime_overhead")[
                    "contract_path_ms"]["p95"], .75)
            elif index == 56:
                self.assertLessEqual(host[
                    "measurement_contract_candidate"
                ]["steady_state_ms"]["p95"], host["gate_ms"])
            else:
                candidate = host["measurement_contract_candidate"]
                self.assertLessEqual(candidate["deadline_miss_rate"], .01)
                self.assertLessEqual(
                    candidate["consecutive_deadline_miss_max"], 1
                )
        elif index == 58:
            self.assertIn(report("determinism")["status"],
                          ("PASS_CPU", "PASS"))
        elif index == 59:
            self.assertFalse(report("candidate_selection")[
                "production_default_changed"])
        elif index == 60:
            self.assertFalse(read_report_bool(
                "historical_validation_status",
                "historical_fresh_validation_rerun",
            ))
        elif index == 61:
            self.assertFalse(read_report_bool(
                "historical_validation_status", "formal_data_used"
            ))
        elif index == 62:
            self.assertFalse(read_report_bool(
                "historical_validation_status",
                "holdout_test_blind_accessed",
            ))
        elif index == 63:
            self.assertFalse(final["training_authorized"])
        elif index == 64:
            self.assertFalse(final["training_authorized"])
        elif 65 <= index <= 71:
            regression = report("regression")
            keys = (
                "mar1_artifacts_modified", "diro1_artifacts_modified",
                "strict_measurement_modified", "TrackManager_modified",
                "Kalman_modified", "YOPO_modified", "BDRR1_modified",
            )
            self.assertFalse(regression[keys[index-65]])
        elif index == 72:
            self.assertTrue(all(
                (ROOT/path).is_file()
                for path in report("implementation_contract")["files"]
            ))
        elif index == 73:
            self.assertEqual(report("regression")["status"], "PASS")
        elif index == 74:
            for stem in REPORT_STEMS:
                self.assertTrue(
                    (REPORTS/f"{PREFIX}{stem}.json").is_file(), stem
                )
            for stem in MARKDOWN_STEMS:
                self.assertTrue(
                    (REPORTS/f"{PREFIX}{stem}.md").is_file(), stem
                )


def read_report_bool(stem, key):
    return bool(report(stem)[key])


def _make_test(index):
    def test(self):
        self.check_index(index)
    test.__name__ = f"test_{index:02d}"
    return test


for _index in range(1, 75):
    setattr(DMCR1Gate, f"test_{_index:02d}", _make_test(_index))


if __name__ == "__main__":
    unittest.main()
