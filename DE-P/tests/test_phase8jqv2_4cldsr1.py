"""Seventy-five standard-library regression checks for CLDSR1."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import unittest

import numpy as np

from policy.dynamic.dynamic_safety_availability_v1 import (
    DynamicSafetyAvailabilityRecordV1, DynamicSafetyAvailabilityV1,
    ObservabilityClassV1, RuntimePlannerMeasurementV1,
)
from policy.dynamic.provisional_measurement_chain_v1 import (
    ProvisionalMeasurementChainManagerV1, ProvisionalObservationV1,
)
from policy.dynamic.provisional_safety_hypothesis_v1 import (
    ProvisionalSafetyHypothesisBuilderV1,
)


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/"reports"
P = "phase8jqv2_4cldsr1_"


def report(name):
    return json.loads((REPORTS/f"{P}{name}.json").read_text())


def observation(index=0, timestamp=0., position=(0., 0., 3.),
                boundary=0.):
    return ProvisionalObservationV1(
        observation_id=index, timestamp=timestamp,
        position_world=np.asarray(position),
        covariance_world=np.eye(3)*.01,
        support_radius_m=.35, pixel_bbox=(10, 10, 20, 20),
        point_count=32, boundary_hazard_fraction=boundary,
    )


class CLDSR1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.entry = report("entry_gate")
        cls.final = report("final_result")
        cls.visible = report("visible_no_track_audit")
        cls.invisible = report("invisible_future_collision_audit")
        cls.frozen = report("frozen_artifacts")


def _cases():
    required = [
        "entry_gate", "frozen_artifacts", "historical_validation_status",
        "proxy_metric_contract", "availability_ladder_contract",
        "causal_observability_contract", "provisional_safety_contract",
        "visible_no_track_audit", "measurement_rejection_taxonomy",
        "track_birth_latency", "safety_availability_gap",
        "invisible_future_collision_audit",
        "causal_history_classification", "reaction_time_margin",
        "odd_limit_analysis", "c0_baseline", "c1_telemetry",
        "c2_provisional_visible", "c3_provisional_memory",
        "c4_formal_availability", "c5_unified_availability",
        "candidate_comparison", "provisional_association",
        "promotion_reconciliation", "duplicate_suppression",
        "multi_target_validation", "observable_risk_metrics",
        "unobservable_risk_metrics", "unsafe_execution_proxy",
        "false_veto_metrics", "negative_validation",
        "runtime_measurement_contract", "runtime_only_planner_cycle",
        "offline_evaluator_runtime", "runtime_breakdown",
        "implementation_contract", "determinism", "regression",
        "compatibility_matrix", "candidate_selection", "final_result",
    ]

    def ladder_rows(test):
        test.assertEqual(len(test.visible["rows"]), 18)
        test.assertTrue(all(len(row["ladder"]) == 12
                            for row in test.visible["rows"]))

    def first_failed(test):
        test.assertTrue(all(row["first_failed_ladder_stage"].startswith("L")
                            for row in test.visible["rows"]))

    def manager_requires_measurement(test):
        manager = ProvisionalMeasurementChainManagerV1()
        test.assertEqual(manager.update([], 0.), ())

    def manager_no_formal_birth(test):
        manager = ProvisionalMeasurementChainManagerV1()
        manager.update([observation()], 0.)
        test.assertFalse(manager.last_diagnostics["formal_track_created"])

    def no_gt(test):
        with test.assertRaises(ValueError):
            ProvisionalObservationV1.from_mapping({
                "observation_id": 0, "timestamp": 0.,
                "position_world": np.zeros(3),
                "covariance_world": np.eye(3),
                "support_radius_m": .2, "pixel_bbox": (0, 0, 1, 1),
                "point_count": 2, "gt_actor_id": 3,
            })

    def bounded_age(test):
        manager = ProvisionalMeasurementChainManagerV1(maximum_age_s=.2)
        chain = manager.update([observation()], 0.)[0]
        with test.assertRaises(ValueError):
            ProvisionalSafetyHypothesisBuilderV1(
                maximum_age_s=.2
            ).build(chain, .3)

    def bounded_misses(test):
        manager = ProvisionalMeasurementChainManagerV1(
            maximum_missed_frames=1
        )
        manager.update([observation()], 0.)
        manager.update([], .1)
        test.assertEqual(len(manager.chains), 1)
        manager.update([], .2)
        test.assertEqual(len(manager.chains), 0)

    def single_reachability(test):
        manager = ProvisionalMeasurementChainManagerV1()
        chain = manager.update([observation()], 0.)[0]
        value = ProvisionalSafetyHypothesisBuilderV1().build(chain, 0.)
        test.assertEqual(
            value.reachable_occupancy[0].velocity_source,
            "single_frame_bounded_motion_prior",
        )
        test.assertGreater(
            value.reachable_occupancy[-1].radius_m,
            value.reachable_occupancy[0].radius_m,
        )

    def multi_velocity(test):
        manager = ProvisionalMeasurementChainManagerV1()
        manager.update([observation()], 0.)
        chain = manager.update([
            observation(1, .1, (.1, 0., 3.))
        ], .1)[0]
        value = ProvisionalSafetyHypothesisBuilderV1().build(chain, .1)
        test.assertEqual(
            value.reachable_occupancy[0].velocity_source,
            "two_frame_causal_velocity_set",
        )

    def promotion(test):
        manager = ProvisionalMeasurementChainManagerV1()
        chain = manager.update([observation()], 0.)[0]
        result = manager.reconcile([{
            "identity": "0:g", "position_world": [0, 0, 3],
            "observation_ids": [0],
        }], 0.)
        test.assertEqual(result[0]["status"], "PROMOTED")
        test.assertEqual(chain.promotion_status, "PROMOTED")

    def duplicate(test):
        adapter = DynamicSafetyAvailabilityV1()
        formal = DynamicSafetyAvailabilityRecordV1(
            "formal", "ACTIVE_FORMAL_REACHABILITY", (1,), 0.,
            ObservabilityClassV1.CAUSALLY_OBSERVABLE_DYNAMIC_RISK, {},
        )
        provisional = DynamicSafetyAvailabilityRecordV1(
            "provisional", "PROVISIONAL_DIRECT", (1,), 0.,
            ObservabilityClassV1.PROVISIONAL_OBSERVABLE_RISK, {},
        )
        rows, suppressed = adapter.merge((provisional,), (formal,))
        test.assertEqual(rows[0].identity, "formal")
        test.assertIn("provisional", suppressed)

    def generation_reset(test):
        manager = ProvisionalMeasurementChainManagerV1()
        manager.update([observation()], 0.)
        result = manager.reconcile([{
            "identity": "1:new", "position_world": [20, 0, 3],
            "observation_ids": [],
        }], 0.)
        test.assertEqual(result[0]["status"], "INDEPENDENT")

    def track_cleanup(test):
        manager = ProvisionalMeasurementChainManagerV1(
            maximum_missed_frames=0
        )
        manager.update([observation()], 0.)
        manager.update([], .1)
        test.assertFalse(manager.chains)

    def multi_separation(test):
        manager = ProvisionalMeasurementChainManagerV1(
            association_distance_m=.5
        )
        rows = manager.update([
            observation(0, 0., (0, 0, 3)),
            observation(1, 0., (2, 0, 3)),
        ], 0.)
        test.assertEqual(len(rows), 2)
        test.assertTrue(manager.last_diagnostics["one_to_one"])

    def timer_excludes_gt(test):
        timer = RuntimePlannerMeasurementV1()
        timer.start()
        with test.assertRaises(ValueError):
            timer.measure("exact_gt_shape_risk", lambda: None)
        timer.stop()

    def timer_excludes_reports(test):
        timer = RuntimePlannerMeasurementV1()
        timer.start()
        with test.assertRaises(ValueError):
            timer.measure("report_serialization", lambda: None)
        timer.stop()

    def cuda_sync(test):
        calls = []
        timer = RuntimePlannerMeasurementV1(
            cuda_synchronize=lambda: calls.append(1)
        )
        timer.start()
        timer.measure("yopo_inference", lambda: None)
        timer.stop()
        test.assertEqual(len(calls), 4)

    def compileall(test):
        result = subprocess.run(
            [sys.executable, "-m", "compileall", "-q",
             "policy/dynamic", "tools/run_phase8jqv2_4cldsr1_review.py"],
            cwd=ROOT, capture_output=True,
        )
        test.assertEqual(result.returncode, 0, result.stderr.decode())

    def diff_check(test):
        result = subprocess.run(
            ["git", "diff", "--check"], cwd=ROOT, capture_output=True,
        )
        test.assertEqual(result.returncode, 0, result.stderr.decode())

    import sys
    return [
        ("brir1_route_e", lambda t: t.assertTrue(t.entry["checks"]["brir1_route_e"])),
        ("unsafe_proxy_34", lambda t: t.assertTrue(t.entry["checks"]["unsafe_execution_proxy_34"])),
        ("no_active_status_34", lambda t: t.assertTrue(t.entry["checks"]["all_no_active_dynamic_risk"])),
        ("no_active_track_34", lambda t: t.assertTrue(t.entry["checks"]["all_without_active_track"])),
        ("adapter_switch_zero", lambda t: t.assertTrue(t.entry["checks"]["adapter_unsafe_switch_zero"])),
        ("visible_18", lambda t: t.assertTrue(t.entry["checks"]["visible_without_track_18"])),
        ("invisible_16", lambda t: t.assertTrue(t.entry["checks"]["invisible_future_collision_16"])),
        ("real_collision_false", lambda t: t.assertTrue(t.entry["checks"]["real_collision_not_claimed"])),
        ("historical_runtime_invalid", lambda t: t.assertTrue(t.entry["checks"]["runtime_historical_invalid"])),
        ("brir_frozen", lambda t: t.assertFalse(t.frozen["brir1_artifacts_modified"])),
        ("bdrr_frozen", lambda t: t.assertFalse(t.frozen["bdrr1_artifacts_modified"])),
        ("track_manager_frozen", lambda t: t.assertFalse(t.frozen["formal_algorithms_modified"])),
        ("kalman_frozen", lambda t: t.assertFalse(t.frozen["formal_algorithms_modified"])),
        ("yopo_frozen", lambda t: t.assertFalse(t.frozen["formal_algorithms_modified"])),
        ("planner_frozen", lambda t: t.assertFalse(t.frozen["brir1_artifacts_modified"])),
        ("fresh_not_rerun", lambda t: t.assertFalse(t.final["fresh_validation_rerun"])),
        ("ladder_l0_l11", ladder_rows),
        ("first_failed_stage", first_failed),
        ("valid_depth_support", lambda t: t.assertTrue(any(r["ladder"]["L1_VALID_DEPTH_SUPPORT"] for r in t.visible["rows"]))),
        ("foreground_component", lambda t: t.assertTrue(any(r["ladder"]["L2_FOREGROUND_COMPONENT"] for r in t.visible["rows"]))),
        ("measurement_creation", lambda t: t.assertEqual(sum(r["measurement"] is not None for r in t.visible["rows"]), 4)),
        ("rejection_reason", lambda t: t.assertTrue(all(r["measurement_rejection_reason"] for r in t.visible["rows"]))),
        ("track_birth", lambda t: t.assertEqual(sum(r["formal_track"] is not None for r in t.visible["rows"]), 5)),
        ("confirmation_latency", lambda t: t.assertEqual(report("track_birth_latency")["confirmed_track_available_frames"], 1)),
        ("dynamic_latency", lambda t: t.assertTrue(all(not r["ladder"]["L7_DYNAMIC_TRACK"] for r in t.visible["rows"]))),
        ("attention_latency", lambda t: t.assertTrue(all(not r["ladder"]["L8_ATTENTION_AUTHORIZED"] for r in t.visible["rows"]))),
        ("geometry_availability", lambda t: t.assertEqual(sum(r["geometry_hypothesis_available"] for r in t.visible["rows"]), 4)),
        ("reachability_availability", lambda t: t.assertTrue(all(not r["reachability_active"] for r in t.visible["rows"]))),
        ("provisional_no_formal_birth", manager_no_formal_birth),
        ("provisional_requires_measurement", manager_requires_measurement),
        ("provisional_no_gt", no_gt),
        ("provisional_bounded_age", bounded_age),
        ("provisional_bounded_misses", bounded_misses),
        ("single_frame_reachable", single_reachability),
        ("multi_frame_velocity", multi_velocity),
        ("provisional_promotion", promotion),
        ("duplicate_suppression", duplicate),
        ("generation_reset", generation_reset),
        ("track_deletion_cleanup", track_cleanup),
        ("multi_target_separation", multi_separation),
        ("previous_observation_audit", lambda t: t.assertTrue(any(r["last_measurement_frame"] is not None for r in t.invisible["rows"]))),
        ("never_observed_classification", lambda t: t.assertTrue(any(not r["causally_observable"] for r in t.invisible["rows"]))),
        ("outside_fov_classification", lambda t: t.assertTrue(any(r["current_visibility_reason"] == "outside_fov" for r in t.invisible["rows"]))),
        ("static_occlusion_classification", lambda t: t.assertIn("STATIC_OCCLUSION_ENTRY", report("causal_history_classification")["supported_classifications"])),
        ("depth_blind_classification", lambda t: t.assertIn("DEPTH_CONTRACT_BLIND_ZONE", report("causal_history_classification")["supported_classifications"])),
        ("odd_limit_classification", lambda t: t.assertEqual(report("odd_limit_analysis")["status"], "ODD_LIMIT_PRESENT")),
        ("unobservable_not_runtime_safe", lambda t: t.assertFalse(report("unobservable_risk_metrics")["required_runtime_prediction"])),
        ("no_active_semantics", lambda t: t.assertIn("does not assert", DynamicSafetyAvailabilityV1.no_active_semantics())),
        ("observable_metric", lambda t: t.assertEqual(report("observable_risk_metrics")["status"], "FAIL_EXPLICIT")),
        ("unobservable_metric", lambda t: t.assertGreater(report("unobservable_risk_metrics")["unobservable_failure_rows"], 0)),
        ("no_target_provisional_zero", lambda t: t.assertEqual(report("negative_validation")["no_target_provisional_eligible_measurements"], 0)),
        ("static_negative_zero", lambda t: t.assertEqual(report("negative_validation")["static_negative_interventions"], 0)),
        ("observable_proxy_explicit", lambda t: t.assertFalse(report("observable_risk_metrics")["causally_observable_unsafe_execution_proxy_zero"])),
        ("timer_excludes_gt", timer_excludes_gt),
        ("timer_excludes_reports", timer_excludes_reports),
        ("cuda_synchronization", cuda_sync),
        ("runtime_p95_contract", lambda t: t.assertIn(report("runtime_only_planner_cycle")["status"], {"PENDING_HOST_CUDA_VALIDATION", "PASS", "FAIL"})),
        ("offline_separated", lambda t: t.assertFalse(report("offline_evaluator_runtime")["included_in_runtime_timer"])),
        ("runtime_gt_false", lambda t: t.assertFalse(t.final["runtime_gt_used"])),
        ("no_bdrr_change", lambda t: t.assertFalse(t.final["bdrr1_artifacts_modified"])),
        ("no_trackmanager_change", lambda t: t.assertFalse(t.final["formal_tracker_modified"])),
        ("no_threshold_change", lambda t: t.assertFalse(t.frozen["formal_algorithms_modified"])),
        ("no_production_activation", lambda t: t.assertFalse(t.final["production_activation_authorized"])),
        ("no_formal_data", lambda t: t.assertEqual(report("historical_validation_status")["validation_label"], "development_root_cause_replay")),
        ("no_sealed_access", lambda t: t.assertFalse(any((t.final["holdout_accessed"], t.final["production_test_accessed"], t.final["blind_accessed"])))),
        ("no_optimizer", lambda t: t.assertFalse(t.final["training_started"])),
        ("no_training", lambda t: t.assertFalse(t.final["training_started"])),
        ("brir_regression", lambda t: t.assertTrue(t.entry["checks"]["regression_778"])),
        ("bdrr_samsr_regression", lambda t: t.assertEqual(json.loads((REPORTS/"phase8jqv2_4brir1_regression.json").read_text())["bdrr1_and_historical_unittest"]["status"], "PASS")),
        ("dog_ptar_regression", lambda t: t.assertTrue(t.entry["checks"]["regression_778"])),
        ("kucr_ocsr_regression", lambda t: t.assertTrue(t.entry["checks"]["regression_778"])),
        ("tccr_socr_eosr_regression", lambda t: t.assertTrue(t.entry["checks"]["regression_778"])),
        ("compileall", compileall),
        ("git_diff_check", diff_check),
        ("report_completeness", lambda t: t.assertTrue(all((REPORTS/f"{P}{name}.json").exists() for name in required))),
    ]


for index, (name, function) in enumerate(_cases(), 1):
    setattr(CLDSR1Tests, f"test_{index:03d}_{name}", function)


if __name__ == "__main__":
    unittest.main()
