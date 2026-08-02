"""BDRR1 time, reachability, risk, freeze and fail-closed tests."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import unittest

import numpy as np
import yaml

from policy.dynamic.asynchronous_multi_target_risk_v1 import (
    evaluate_asynchronous_reachability_risk,
)
from policy.dynamic.bounded_dynamic_reachability_v1 import (
    BoundedDynamicReachabilityBuilderV1,
    DynamicReachabilityStateV1,
    IntervalSetV1,
    ReachabilityStatus,
)
from policy.dynamic.shape_aware_motion_state_v1 import MotionObservability
from policy.dynamic.shape_reachable_occupancy_v1 import (
    predict_reachable_occupancy,
    reachable_signed_distance,
)
from policy.dynamic.stale_geometry_time_contract_v1 import (
    TimeOriginMode,
    resolve_effective_horizon,
)


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
CONFIG = ROOT / "configs/bounded_dynamic_reachability_contract_v1_candidate.yaml"
PREFIX = "phase8jqv2_4bdrr1_"


def report(name):
    return json.loads((REPORTS / f"{PREFIX}{name}.json").read_text())


def prior(prefix, name):
    return json.loads((REPORTS / f"{prefix}_{name}.json").read_text())


def require(value, message="requirement failed"):
    if not value:
        raise AssertionError(message)


def state(shape="sphere", *, track=1, generation="g1", age=.1):
    return DynamicReachabilityStateV1(
        track_id=track,
        generation=generation,
        hypothesis_id=f"{shape}:0",
        shape_type=shape,
        source_geometry_timestamp=1.0,
        state_timestamp=1.0 + age,
        query_timestamp=1.0 + age,
        geometry_age_s=age,
        motion_observability=MotionObservability.MOTION_OBSERVABLE,
        position_set_world=IntervalSetV1(
            np.array([-.05, -.05, -.05]),
            np.array([.05, .05, .05]),
        ),
        velocity_set_world_mps=IntervalSetV1(
            np.array([.8, -.1, -.1]), np.array([1.2, .1, .1])
        ),
        acceleration_set_world_mps2=IntervalSetV1(
            np.full(3, -.2), np.full(3, .2)
        ),
        radius_interval_m=(.28, .32),
        half_height_interval_m=(.45, .55),
        center_z_interval_m=(-.05, .05),
        reference_drift_rate_mps=.02,
        status=ReachabilityStatus.STALE_BOUNDED_REACHABILITY,
        expiry_timestamp=1.25,
        position_reference_timestamp=1.0,
        runtime_gt_used=False,
    )


class TestBDRR1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = yaml.safe_load(CONFIG.read_text())
        cls.entry = report("entry_gate")
        cls.final = report("final_result")
        cls.fresh = json.loads(
            (ROOT / "diagnostics/phase8jqv2_4bdrr1/fresh_summary.json")
            .read_text()
        )["summary"]
        cls.builder = BoundedDynamicReachabilityBuilderV1(
            cls.config,
            {
                "use_velocity_set": True,
                "use_acceleration_set": True,
                "use_reference_drift": True,
            },
        )


def check(number, name, function):
    def test(self):
        function(self)
    test.__name__ = f"test_{number:02d}_{name}"
    setattr(TestBDRR1, test.__name__, test)


def stale_horizon():
    return resolve_effective_horizon(
        geometry_timestamp=1.0,
        motion_state_timestamp=1.1,
        current_query_timestamp=1.1,
        candidate_relative_times_s=np.array([0., .2]),
        position_reference_timestamp=1.0,
    )


def current_horizon():
    return resolve_effective_horizon(
        geometry_timestamp=1.0,
        motion_state_timestamp=1.1,
        current_query_timestamp=1.1,
        candidate_relative_times_s=np.array([0., .2]),
        position_reference_timestamp=1.1,
    )


def risk_result(states=None, unresolved=False):
    candidates = np.array([
        [[0., 0., 0.], [.2, 0., 0.]],
        [[0., 2., 0.], [.2, 2., 0.]],
    ])
    return evaluate_asynchronous_reachability_risk(
        candidates, np.array([0., .2]), states or (), TestBDRR1.builder,
        uav_radius_m=.1, required_margin_m=.05,
        candidate_scores=np.array([0., 1.]),
        unresolved_dynamic_risk=unresolved,
    )


checks = [
    ("samsr1_route_e", lambda s: require(s.entry["checks"]["samsr1_route_e"])),
    ("unsafe_miss_two", lambda s: require(s.entry["checks"]["unsafe_miss_2"])),
    ("historical_top3_zero", lambda s: require(s.entry["checks"]["top3_unsafe_miss_0"])),
    ("historical_unsafe_recommendation_one", lambda s: require(s.entry["checks"]["unsafe_recommendation_1"])),
    ("historical_sphere_coverage", lambda s: require(s.entry["checks"]["sphere_coverage_94_44"])),
    ("historical_cylinder_velocity", lambda s: require(s.entry["checks"]["cylinder_velocity_p95_0_691"])),
    ("historical_runtime", lambda s: require(s.entry["checks"]["direct_geometry_runtime_pass"] and s.entry["checks"]["cached_risk_runtime_pass"])),
    ("samsr1_frozen", lambda s: require(not report("frozen_artifacts")["samsr1_artifacts_modified"])),
    ("dogmr1_frozen", lambda s: require(not report("frozen_artifacts")["dogmr1_artifacts_modified"])),
    ("track_manager_frozen", lambda s: require(not report("frozen_artifacts")["formal_track_manager_modified"])),
    ("kalman_frozen", lambda s: require(not report("frozen_artifacts")["formal_kalman_modified"])),
    ("yopo_frozen", lambda s: require(not report("frozen_artifacts")["formal_yopo_modified"])),
    ("historical_observed", lambda s: require(report("historical_validation_status")["samsr1_validation_status"] == "HISTORICAL_OBSERVED_VALIDATION")),
    ("failure_manifest", lambda s: require(report("failure_case_manifest")["status"] == "PASS")),
    ("two_misses_mapped", lambda s: require(report("failure_case_manifest")["rows_accounted"] == 2)),
    ("unsafe_recommendation_mapped", lambda s: require(report("failure_case_manifest")["unsafe_recommendation_is_one_of_two_misses"])),
    ("geometry_timestamp", lambda s: s.assertEqual(stale_horizon().geometry_timestamp, 1.0)),
    ("motion_timestamp", lambda s: s.assertEqual(stale_horizon().motion_state_timestamp, 1.1)),
    ("query_timestamp", lambda s: s.assertEqual(stale_horizon().current_query_timestamp, 1.1)),
    ("candidate_relative_time", lambda s: np.testing.assert_allclose(stale_horizon().candidate_relative_times_s, [0., .2])),
    ("effective_horizon", lambda s: np.testing.assert_allclose(stale_horizon().effective_prediction_horizons_s, [.1, .3])),
    ("no_missing_geometry_age", lambda s: require(s.fresh["time_integrity"]["missing_geometry_age"] == 0)),
    ("no_double_geometry_age", lambda s: np.testing.assert_allclose(current_horizon().effective_prediction_horizons_s, [0., .2])),
    ("per_track_timestamps", lambda s: require(report("per_track_timestamp_validation")["per_track_unique_ages"] > 1)),
    ("no_global_timestamp", lambda s: require(not report("per_track_timestamp_validation")["global_geometry_timestamp_used"])),
    ("cache_track_key", lambda s: require("track_id" in report("cache_generation_validation")["key_fields"])),
    ("cache_generation_key", lambda s: require("generation" in report("cache_generation_validation")["key_fields"])),
    ("track_deletion_cleanup", lambda s: (setattr(s.builder, "_keys", {(1, "g1", "sphere:0"), (2, "g2", "sphere:0")}), require(s.builder.delete_missing({1}) == ((2, "g2", "sphere:0"),)))),
    ("generation_reset", lambda s: (s.builder.clear(), require(len(s.builder._keys) == 0))),
    ("b0_reproduced", lambda s: require(report("b0_baseline")["missed_unsafe"] == 2)),
    ("b1_stale_age", lambda s: require(report("b1_stale_age")["status"] == "DEVELOPMENT_EVALUATED_TIME_ORIGIN_NOT_SUFFICIENT")),
    ("b2_sphere_velocity", lambda s: require(report("b2_velocity_set")["sphere_half_width_mps"] == [.12, .12, .12])),
    ("b2_cylinder_horizontal", lambda s: require(report("b2_velocity_set")["cylinder_half_width_mps"][:2] == [.25, .25])),
    ("b2_cylinder_vertical", lambda s: require(report("b2_velocity_set")["cylinder_half_width_mps"][2] == .5)),
    ("b3_acceleration", lambda s: require(report("b3_acceleration")["bound_mps2"] == .55)),
    ("no_future_acceleration", lambda s: require(not report("b3_acceleration")["future_acceleration_used"])),
    ("b4_reference_drift", lambda s: require(report("b4_reference_drift")["bounds"]["weak_or_ambiguous_rate_mps"] == .12)),
    ("no_double_drift", lambda s: require(report("reference_drift_contract")["double_count_guard"].startswith("applied_once"))),
    ("b5_sphere_reachability", lambda s: require(np.all(np.isfinite(reachable_signed_distance(np.array([[0., 0., 0.], [.2, 0., 0.]]), predict_reachable_occupancy(state("sphere"), stale_horizon())))))),
    ("b5_finite_cylinder", lambda s: require(reachable_signed_distance(np.array([[0., 0., 2.], [.2, 0., 2.]]), predict_reachable_occupancy(state("vertical_cylinder"), stale_horizon()))[0] > 0)),
    ("b5_ambiguous_hypotheses", lambda s: require(report("b5_shape_reachability")["ambiguous_hypotheses_preserved"])),
    ("b5_support", lambda s: require(report("b5_shape_reachability")["support_reachability_preserved"])),
    ("b6_asynchronous", lambda s: require(len(risk_result([state(track=1), state(track=2, age=.2)])["per_track_time_rows"]) == 2)),
    ("all_tracks_participate", lambda s: require(report("multi_target_contract")["all_live_nonexpired_tracks_participate"])),
    ("minimum_robust_clearance", lambda s: require(risk_result([state()])["candidate_rows"][0]["robust_minimum_clearance_m"] <= risk_result([state()])["candidate_rows"][1]["robust_minimum_clearance_m"])),
    ("b7_unresolved", lambda s: require(risk_result(unresolved=True)["decision_status"] == "UNRESOLVED_DYNAMIC_RISK")),
    ("unresolved_not_no_active", lambda s: require(report("unresolved_risk_validation")["unresolved_semantics"].startswith("fail_closed"))),
    ("b8_unified", lambda s: require(report("b8_unified")["status"] == "FRESH_DEVELOPMENT_PASS")),
    ("direct_sphere_coverage", lambda s: require(report("sphere_reachability_coverage")["direct_geometry_coverage"] == 1.0)),
    ("predicted_sphere_coverage", lambda s: require(report("sphere_reachability_coverage")["predicted_occupancy_coverage"] >= .975)),
    ("cylinder_occupancy_coverage", lambda s: require(report("cylinder_reachability_coverage")["predicted_occupancy_coverage"] >= .95)),
    ("cylinder_small_sample", lambda s: require(report("cylinder_reachability_coverage")["status"] == "DEVELOPMENT_PASS_SMALL_SAMPLE")),
    ("candidate_global_false_veto", lambda s: require(s.fresh["candidate_global_false_veto_rate"] <= .30)),
    ("safe_candidate_false_veto", lambda s: require(s.fresh["safe_candidate_false_veto_rate"] <= .30)),
    ("sequence_false_emergency", lambda s: require(s.fresh["sequence_false_emergency_rate"] <= .10)),
    ("selected_safe_loss", lambda s: require(s.fresh["selected_safe_loss_rate"] == 0)),
    ("gap1", lambda s: require(s.config["expiry"]["gap1_frames"] == 1)),
    ("gap2", lambda s: require(s.config["expiry"]["gap2_frames"] == 2)),
    ("gap3_expiry", lambda s: require(s.config["expiry"]["gap3"] == "EXPIRED")),
    ("long_gap_expiry", lambda s: require(s.config["expiry"]["long_gap"] == "EXPIRED")),
    ("fresh_top3_zero", lambda s: require(s.fresh["top3_unsafe_miss"] == 0)),
    ("fresh_unsafe_recommendation_zero", lambda s: require(s.fresh["unsafe_recommendations"] == 0)),
    ("fresh_multi_target_stale_zero", lambda s: require(s.fresh["multi_target_stale_miss"] == 0)),
    ("fresh_no_target_false_veto_zero", lambda s: require(s.fresh["no_target_false_veto"] == 0)),
    ("no_safe_candidate", lambda s: require(report("no_safe_candidate_validation")["correct_no_safe_candidate"] > 0)),
    ("no_unsafe_fallback", lambda s: require(not report("no_safe_candidate_validation")["unsafe_fallback_allowed"])),
    ("grouped_split", lambda s: require(report("evaluation_split")["grouping_unit"].startswith("complete_sequence"))),
    ("no_frame_leakage", lambda s: require(not report("evaluation_split")["frame_level_leakage"])),
    ("fresh_freeze", lambda s: require(report("fresh_validation_freeze")["status"] == "FROZEN_BEFORE_FRESH_GT_ACCESS")),
    ("no_tuning_after_freeze", lambda s: require(not report("validation_freeze")["parameters_changed_after_freeze"])),
    ("runtime_no_gt", lambda s: require(not s.fresh["runtime_gt_used"])),
    ("direct_geometry_runtime", lambda s: require(report("runtime")["checks"]["direct_geometry_p95"])),
    ("cached_reachability_runtime", lambda s: require(report("runtime")["checks"]["cached_risk_p95"])),
    ("combined_runtime", lambda s: require(all(report("runtime")["checks"].values()))),
    ("deterministic", lambda s: require(report("determinism")["source_hashes_match"])),
    ("no_kucr_u1_u7", lambda s: require(not s.final["kucr1_uncertainty_search_resumed"])),
    ("planner_unchanged", lambda s: require(not s.final["formal_planner_modified"])),
    ("no_formal_data", lambda s: require(not s.final["new_formal_dataset_generated"])),
    ("no_sealed_data", lambda s: require(not any(s.final[key] for key in ("holdout_accessed", "production_test_accessed", "blind_accessed")))),
    ("no_optimizer", lambda s: require(not s.final["optimizer_step_executed"])),
    ("no_training", lambda s: require(not s.final["training_started"])),
    ("samsr1_regression", lambda s: require(prior("phase8jqv2_4samsr1", "final_result")["route"] == "E")),
    ("dog_ptar_regression", lambda s: require(prior("phase8jqv2_4dogmr1", "final_result")["route"] == "E" and prior("phase8jqv2_4ptar1", "final_result")["route"] == "G")),
    ("kucr_ocsr_regression", lambda s: require(prior("phase8jqv2_4kucr1", "final_result")["route"] == "C" and (REPORTS / "phase8jqv2_4ocsr1_final_result.json").exists())),
    ("tccr_socr_eosr_regression", lambda s: require(all((REPORTS / f"{name}_final_result.json").exists() for name in ("phase8jqv2_4tccr1", "phase8jqv2_4socr1", "phase8jqv2_4eosr1")))),
    ("compileall", lambda s: require(subprocess.run(["python", "-m", "compileall", "-q", "policy/dynamic", "tools", "tests/test_phase8jqv2_4bdrr1.py"], cwd=ROOT).returncode == 0)),
    ("git_diff_check", lambda s: require(subprocess.run(["git", "diff", "--check"], cwd=ROOT, capture_output=True).returncode == 0)),
    ("report_completeness", lambda s: require(len(list(REPORTS.glob(f"{PREFIX}*"))) == 53)),
]

if len(checks) != 88:
    raise RuntimeError(f"expected 88 tests, got {len(checks)}")
for index, (name, function) in enumerate(checks, 1):
    check(index, name, function)


if __name__ == "__main__":
    unittest.main()
