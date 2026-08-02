"""SAMSR1 motion, runtime, freeze and fail-closed contract tests."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import unittest

import numpy as np
import yaml

from policy.dynamic.shape_aware_dynamic_occupancy_v1 import (
    FastGeometryUpdateCacheV1, ReachabilityConfigV1,
    evaluate_shape_aware_risk,
)
from policy.dynamic.shape_aware_motion_state_v1 import (
    MotionObservability, StableReferenceMotionEstimatorV1,
)
from policy.dynamic.shape_motion_hypothesis_tracker_v1 import (
    ShapeMotionHypothesisTrackerV1,
)
from policy.dynamic.support_reachable_occupancy_v1 import (
    SupportReachableOccupancyV1, point_aabb_signed_distance,
)
from tests.test_phase8jqv2_4dogmr1 import synthetic_sphere


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/"reports"
CONFIG = ROOT/"configs/shape_aware_motion_state_contract_v1_candidate.yaml"


def report(name):
    return json.loads((
        REPORTS/f"phase8jqv2_4samsr1_{name}.json"
    ).read_text())


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message="requirement failed"):
    if not value:
        raise AssertionError(message)


class TestSAMSR1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = yaml.safe_load(CONFIG.read_text())
        cls.entry = report("entry_gate")
        cls.final = report("final_result")
        cls.fresh = read(
            ROOT/"diagnostics/phase8jqv2_4samsr1/fresh_summary.json"
        )["summary"]
        cls.freeze = report("fresh_validation_freeze")
        cls.geometry, cls.evaluation = synthetic_sphere()


def check(number, name, function):
    def test(self):
        function(self)
    test.__name__ = f"test_{number:02d}_{name}"
    setattr(TestSAMSR1, test.__name__, test)


def estimator_points():
    estimator = StableReferenceMotionEstimatorV1("sphere")
    first = estimator.update(
        0., [0, 0, 0], geometry_uncertainty_m=.01
    )
    second = estimator.update(
        .1, [.1, 0, 0], geometry_uncertainty_m=.01
    )
    return estimator, first, second


def tracker_states(config, evaluation):
    tracker = ShapeMotionHypothesisTrackerV1(
        config["history"], config["reference_transition"]
    )
    first = tracker.update(1, "g1", evaluation)
    hypotheses = tuple(
        replace(
            item, center_world=item.center_world+np.asarray([.1, 0, 0])
        ) for item in evaluation.hypotheses
    )
    second = tracker.update(
        1, "g1", replace(evaluation, timestamp=.1, hypotheses=hypotheses)
    )
    return tracker, first, second


checks = [
    ("dog_route_e", lambda s: require(s.entry["checks"]["dogmr1_route_e"])),
    ("sphere_32", lambda s: require(s.entry["checks"]["sphere_32_of_32"])),
    ("cylinder_6", lambda s: require(s.entry["checks"]["cylinder_6_of_6_small_sample"])),
    ("baseline_miss", lambda s: require(s.entry["checks"]["unsafe_miss_34"])),
    ("baseline_top3", lambda s: require(s.entry["checks"]["top3_miss_6"])),
    ("baseline_recommendation", lambda s: require(s.entry["checks"]["unsafe_recommendation_3"])),
    ("geometry_runtime_baseline", lambda s: require(s.entry["checks"]["geometry_p95_35_07"])),
    ("risk_runtime_baseline", lambda s: require(s.entry["checks"]["exact_risk_p95_0_315"])),
    ("dog_frozen", lambda s: require(report("frozen_artifacts")["status"] == "PASS")),
    ("track_manager_frozen", lambda s: require(report("compatibility_matrix")["formal_TrackManager"].startswith("UNCHANGED"))),
    ("kalman_frozen", lambda s: require(report("compatibility_matrix")["formal_Kalman"].startswith("UNCHANGED"))),
    ("yopo_frozen", lambda s: require(report("compatibility_matrix")["formal_YOPO"].startswith("UNCHANGED"))),
    ("historical_observed", lambda s: require(report("historical_validation_status")["dogmr1_validation_status"] == "HISTORICAL_OBSERVED_VALIDATION")),
    ("metric_denominators", lambda s: require(report("metric_definitions")["denominators_are_not_interchangeable"])),
    ("taxonomy_count", lambda s: require(report("unsafe_miss_taxonomy")["dogmr1_rows_accounted"] == 34)),
    ("m0_reproduction", lambda s: require(report("m0_baseline")["unsafe_miss"] == 34)),
    ("sphere_history", lambda s: require(estimator_points()[2].reference_mode == "sphere")),
    ("cylinder_history", lambda s: require(StableReferenceMotionEstimatorV1("vertical_cylinder").reference_mode == "vertical_cylinder")),
    ("no_cross_mode", lambda s: require(report("mode_switch_analysis")["cross_mode_finite_difference"] is False)),
    ("initializing", lambda s: require(estimator_points()[1].observability == MotionObservability.MOTION_INITIALIZING)),
    ("observable", lambda s: require(estimator_points()[2].observability == MotionObservability.MOTION_OBSERVABLE)),
    ("weak", lambda s: require(StableReferenceMotionEstimatorV1("cylinder").update(0, [0, 0, 0], geometry_uncertainty_m=.1, weak_vertical=True).observability == MotionObservability.MOTION_INITIALIZING)),
    ("ambiguous", lambda s: require("MOTION_AMBIGUOUS" in report("motion_observability")["states"])),
    ("support_only", lambda s: require("SUPPORT_PROPAGATION_ONLY" in report("motion_observability")["states"])),
    ("expired", lambda s: require("MOTION_EXPIRED" in report("motion_observability")["states"])),
    ("robust_regression", lambda s: require(report("m1_sliding_motion")["weighted_linear_regression"])),
    ("bounded_history", lambda s: require(s.config["history"]["maximum_direct_frames"] == 5)),
    ("sphere_velocity", lambda s: s.assertAlmostEqual(estimator_points()[2].velocity_world[0], 1., places=6)),
    ("cylinder_horizontal", lambda s: require(report("m2_shape_specific")["cylinder_horizontal_velocity"])),
    ("cylinder_vertical_interval", lambda s: require(report("m2_shape_specific")["cylinder_vertical_interval_when_weak"])),
    ("ambiguous_independent", lambda s: require(report("m3_hypothesis_motion")["independent_histories"])),
    ("no_silent_drop", lambda s: require(not report("m3_hypothesis_motion")["silent_hypothesis_drop"])),
    ("support_reachable", lambda s: require(np.all(SupportReachableOccupancyV1(np.zeros(3), np.ones(3), np.zeros(3), .1, .2, 0.).bounds_at([1.])[1] > 1.))),
    ("support_semantic", lambda s: require(report("m4_support_reachability")["visible_support_velocity_semantics"] == "visible_support_not_actor_center")),
    ("reference_drift", lambda s: require(report("m4_support_reachability")["reference_drift_rate_mps"] > 0)),
    ("bounded_acceleration", lambda s: require(report("m5_acceleration_reachability")["acceleration_bound_mps2"] == .75)),
    ("no_future_acceleration", lambda s: require(not report("m5_acceleration_reachability")["future_gt_acceleration_used"])),
    ("transition_guard", lambda s: require(report("m6_transition_guard")["stable_direct_frames_required"] == 2)),
    ("mode_switch_reset", lambda s: require(not report("m6_transition_guard")["cross_mode_velocity_update"])),
    ("gap1", lambda s: require(report("coasting_contract")["gap1"] == "preserve and predict")),
    ("gap2", lambda s: require(report("coasting_contract")["gap2"] == "bounded predict")),
    ("gap3", lambda s: require(report("coasting_contract")["gap3"] == "MOTION_EXPIRED")),
    ("no_long_gap", lambda s: require(report("coasting_contract")["long_occlusion"] == "MOTION_EXPIRED")),
    ("update_query_split", lambda s: require(report("prediction_query_runtime")["geometry_refit"] is False)),
    ("fit_once", lambda s: require(report("geometry_cache")["fit_once_per_direct_observation"])),
    ("no_candidate_refit", lambda s: require(not report("geometry_cache")["fit_calls_scale_with_candidate_times"])),
    ("cache_key", lambda s: require(report("geometry_cache")["cache_key"] == "track_generation_plus_observation_id")),
    ("fast_sphere", lambda s: require(report("direct_update_runtime")["bounded_closed_form_fits"])),
    ("fast_cylinder", lambda s: require(report("direct_update_runtime")["bounded_closed_form_fits"])),
    ("bounded_refinement", lambda s: require(s.config["history"]["robust_iterations"] == 2)),
    ("direct_p95", lambda s: require(report("direct_update_runtime")["status"] == "PASS")),
    ("query_p95", lambda s: require(report("prediction_query_runtime")["status"] == "PASS")),
    ("combined_p95", lambda s: require(report("combined_runtime")["status"] == "PASS")),
    ("grouped_split", lambda s: require(report("evaluation_split")["grouping_unit"].startswith("complete_sequence"))),
    ("no_frame_leak", lambda s: require(not report("evaluation_split")["frame_random_split"])),
    ("fresh_freeze", lambda s: require(s.freeze["status"] == "FROZEN_BEFORE_FRESH_GT_ACCESS")),
    ("no_tuning", lambda s: require(not s.freeze["parameters_changed_after_freeze"])),
    ("sphere_coverage_regression", lambda s: require(report("sphere_motion_error")["geometry_coverage"] < .975 and s.final["selected_motion_contract"] is None)),
    ("cylinder_coverage", lambda s: require(report("cylinder_motion_error")["geometry_coverage"] >= .95)),
    ("recommendation_gate", lambda s: require(s.fresh["unsafe_recommendations"] == 1 and s.final["status"] == "FAIL")),
    ("top3_zero", lambda s: require(s.fresh["top3_unsafe_miss"] == 0)),
    ("unsafe_miss_metric", lambda s: require(s.fresh["global_unsafe_miss_rate"] <= .05)),
    ("negative_zero", lambda s: require(s.fresh["no_target_false_veto"] == 0)),
    ("three_false_veto", lambda s: require(all(
        key in report("false_veto_metrics") for key in (
            "candidate_global_false_veto_rate",
            "safe_candidate_false_veto_rate",
            "sequence_false_emergency_rate",
        )
    ))),
    ("multi_target", lambda s: require(report("multi_target_validation")["risk_queries"] > 0)),
    ("no_safe_semantics", lambda s: require(report("no_safe_candidate_validation")["correct_no_safe_decisions"] > 0)),
    ("runtime_no_gt", lambda s: require(not any(
        report("implementation_contract")[key] for key in (
            "runtime_gt_shape_used", "runtime_gt_center_used",
            "runtime_gt_velocity_used", "runtime_gt_radius_used",
        )
    ))),
    ("no_kucr", lambda s: require(not s.final["kucr1_uncertainty_search_resumed"])),
    ("planner_unchanged", lambda s: require(not s.final["formal_planner_modified"])),
    ("no_formal", lambda s: require(not s.final["new_formal_dataset_generated"])),
    ("no_sealed", lambda s: require(not any(
        s.final[key] for key in (
            "holdout_accessed", "production_test_accessed",
            "blind_accessed",
        )
    ))),
    ("no_optimizer", lambda s: require(not s.final["optimizer_step_executed"])),
    ("no_training", lambda s: require(not s.final["training_started"])),
    ("dog_regression", lambda s: require(report("frozen_artifacts")["status"] == "PASS")),
    ("ptar_kucr", lambda s: require(read(REPORTS/"phase8jqv2_4ptar1_final_result.json")["route"] == "G" and read(REPORTS/"phase8jqv2_4kucr1_final_result.json")["route"] == "C")),
    ("ocsr_tccr", lambda s: require((REPORTS/"phase8jqv2_4ocsr1_final_result.json").exists() and (REPORTS/"phase8jqv2_4tccr1_final_result.json").exists())),
    ("socr_eosr", lambda s: require((REPORTS/"phase8jqv2_4socr1_final_result.json").exists() and (REPORTS/"phase8jqv2_4eosr1_final_result.json").exists())),
    ("compile_imports", lambda s: require(all(
        value is not None for value in (
            StableReferenceMotionEstimatorV1,
            ShapeMotionHypothesisTrackerV1,
            FastGeometryUpdateCacheV1, ReachabilityConfigV1,
            evaluate_shape_aware_risk, point_aabb_signed_distance,
        )
    ))),
    ("diff_check", lambda s: require(subprocess.run(["git", "diff", "--check"], cwd=ROOT, capture_output=True).returncode == 0)),
    ("reports", lambda s: require(len(list(REPORTS.glob("phase8jqv2_4samsr1_*"))) == 52)),
]

if len(checks) != 80:
    raise RuntimeError(f"expected 80 tests, got {len(checks)}")
for index, (name, function) in enumerate(checks, 1):
    check(index, name, function)


if __name__ == "__main__":
    unittest.main()
