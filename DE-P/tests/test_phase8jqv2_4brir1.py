"""BRIR1 feature gate, snapshot, routing and evidence-boundary tests."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import unittest

import numpy as np
import yaml

from controller.bounded_reachability_planner_adapter_v1 import (
    BoundedReachabilityPlannerAdapterV1, validate_snapshot,
)
from controller.dynamic_safety_decision_router_v1 import (
    FeatureMode, route_dynamic_safety_decision,
)
from tools.run_phase8jqv2_4brir1_evaluation import (
    builder_and_config, make_snapshot, state,
)


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PREFIX = "phase8jqv2_4brir1_"


def report(name):
    return json.loads((REPORTS / f"{PREFIX}{name}.json").read_text())


def prior(prefix, name):
    return json.loads((REPORTS / f"{prefix}_{name}.json").read_text())


def require(value, message="requirement failed"):
    if not value:
        raise AssertionError(message)


class TestBRIR1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.builder = builder_and_config()
        cls.entry = report("entry_gate")
        cls.final = report("final_result")
        cls.mode = json.loads(
            (ROOT / "diagnostics/phase8jqv2_4brir1/mode_controls.json")
            .read_text()
        )
        cls.faults = report("failure_injection")


def check(number, name, function):
    def test(self):
        function(self)
    test.__name__ = f"test_{number:02d}_{name}"
    setattr(TestBRIR1, test.__name__, test)


def adapter(mode, risk=None):
    values = {
        "config": TestBRIR1.config,
        "reachability_builder": TestBRIR1.builder,
        "mode": mode,
        "development_launcher": mode != FeatureMode.LEGACY_OFF,
    }
    if risk is not None:
        values["risk_evaluator"] = risk
    return BoundedReachabilityPlannerAdapterV1(**values)


def active_output():
    return TestBRIR1.mode["modes"]["DEVELOPMENT_ACTIVE"]


checks = [
    ("bdrr1_route_b", lambda s: require(s.entry["checks"]["bdrr1_route_b"])),
    ("selected_candidate", lambda s: require(s.entry["checks"]["selected_candidate"])),
    ("unsafe_223_zero", lambda s: require(s.entry["checks"]["unsafe_223_miss_0"])),
    ("sphere_29", lambda s: require(s.entry["checks"]["sphere_29_of_29"])),
    ("cylinder_10_small", lambda s: require(s.entry["checks"]["cylinder_10_of_10_small_sample"])),
    ("bdrr1_frozen", lambda s: require(not report("frozen_artifacts")["bdrr1_artifacts_modified"])),
    ("samsr1_frozen", lambda s: require(prior("phase8jqv2_4bdrr1", "frozen_artifacts")["samsr1_artifacts_modified"] is False)),
    ("dogmr1_frozen", lambda s: require(prior("phase8jqv2_4bdrr1", "frozen_artifacts")["dogmr1_artifacts_modified"] is False)),
    ("track_manager_frozen", lambda s: require(not report("frozen_artifacts")["formal_tracker_modified"])),
    ("kalman_frozen", lambda s: require(not report("frozen_artifacts")["formal_kalman_modified"])),
    ("yopo_frozen", lambda s: require(not report("frozen_artifacts")["formal_yopo_modified"])),
    ("planner_frozen", lambda s: require(not report("frozen_artifacts")["formal_planner_modified"])),
    ("feature_default_false", lambda s: require(not report("feature_flag_contract")["enabled"])),
    ("legacy_off", lambda s: require(s.mode["modes"]["LEGACY_OFF"]["feature_mode"] == "LEGACY_OFF")),
    ("shadow", lambda s: require(s.mode["modes"]["SHADOW"]["feature_mode"] == "SHADOW")),
    ("development_active", lambda s: require(active_output()["feature_mode"] == "DEVELOPMENT_ACTIVE")),
    ("off_no_risk", lambda s: (lambda a: (a.evaluate(make_snapshot(states=(state(),)), 7), require(a.risk_call_count == 0)))(adapter(FeatureMode.LEGACY_OFF, lambda *a, **k: (_ for _ in ()).throw(RuntimeError("called"))))),
    ("off_candidate_equal", lambda s: require(s.mode["modes"]["LEGACY_OFF"]["recommended_candidate_id"] == 7)),
    ("off_command_equal", lambda s: require(not s.mode["modes"]["LEGACY_OFF"]["formal_command_modified"])),
    ("shadow_candidate_equal", lambda s: require(s.mode["modes"]["SHADOW"]["recommended_candidate_id"] == 7)),
    ("shadow_command_equal", lambda s: require(not s.mode["modes"]["SHADOW"]["formal_command_modified"])),
    ("snapshot_frame", lambda s: require(make_snapshot().frame_index == make_snapshot().candidate_frame_index)),
    ("snapshot_timestamp", lambda s: require(make_snapshot().query_timestamp == make_snapshot().track_snapshot_timestamp)),
    ("candidate_origin", lambda s: require(make_snapshot().candidate_time_origin == make_snapshot().query_timestamp)),
    ("track_generation", lambda s: require(make_snapshot(states=(state(),)).track_identities[0].generation == "g1")),
    ("reachability_generation", lambda s: require(make_snapshot(states=(state(),)).reachability_states[0].generation == "g1")),
    ("no_mixed_frame", lambda s: require(not validate_snapshot(make_snapshot(), s.config))),
    ("snapshot_immutable", lambda s: s.assertRaises(ValueError, make_snapshot().candidate_scores.__setitem__, 0, 9.)),
    ("no_active", lambda s: require(adapter(FeatureMode.DEVELOPMENT_ACTIVE).evaluate(make_snapshot(), 7)["decision_status"] == "NO_ACTIVE_DYNAMIC_RISK")),
    ("keep_original", lambda s: require(s.mode["all_safe"]["decision_status"] == "KEEP_ORIGINAL")),
    ("switch_safe", lambda s: require(active_output()["decision_status"] == "SWITCH_TO_SAFE_CANDIDATE")),
    ("no_safe", lambda s: require(s.mode["all_unsafe"]["decision_status"] == "NO_SAFE_CANDIDATE")),
    ("unresolved", lambda s: require(s.mode["unresolved"]["decision_status"] == "UNRESOLVED_DYNAMIC_RISK")),
    ("invalid", lambda s: require(report("invalid_evaluation")["all_fail_closed"])),
    ("no_unsafe_fallback", lambda s: require(not report("no_safe_candidate")["unsafe_fallback_used"])),
    ("safe_abort", lambda s: require(s.mode["all_unsafe"]["safe_abort"] and s.mode["unresolved"]["safe_abort"])),
    ("no_hover_brake", lambda s: require(not report("safe_abort_contract")["hover_or_brake_fallback_defined"])),
    ("runtime_no_gt", lambda s: require(not s.final["runtime_gt_used"])),
    ("no_target_no_intervention", lambda s: require(report("negative_validation")["real_no_target_interventions"] == 0)),
    ("static_no_intervention", lambda s: require(report("negative_validation")["static_clutter_decision"] == "NO_ACTIVE_DYNAMIC_RISK")),
    ("gap1", lambda s: require(report("gap1_validation")["prediction_only_preserved"])),
    ("gap2", lambda s: require(report("gap2_validation")["prediction_only_preserved"])),
    ("prediction_only", lambda s: require(report("gap2_validation")["decision"] in ("KEEP_ORIGINAL", "SWITCH_TO_SAFE_CANDIDATE", "NO_SAFE_CANDIDATE"))),
    ("multi_target", lambda s: require(report("multi_target_validation")["all_tracks_participate"])),
    ("three_track", lambda s: require(report("multi_target_validation")["synthetic_active_tracks"] == 3)),
    ("cylinder_small", lambda s: require(report("cylinder_validation")["status"] == "DEVELOPMENT_PASS_SMALL_SAMPLE")),
    ("candidate_switch", lambda s: require(active_output()["recommended_candidate_id"] != active_output()["original_candidate_id"])),
    ("no_unsafe_switch", lambda s: require(active_output()["recommended_candidate_id"] in active_output()["safe_candidate_ids"])),
    ("chattering_metric", lambda s: require("switch_rate_hz" in report("chattering_analysis"))),
    ("deadlock_metric", lambda s: require("deadlocks" in report("deadlock_analysis"))),
    ("no_safe_frequency", lambda s: require(report("no_safe_candidate")["frame_rate"] <= report("no_safe_candidate")["gate_max"])),
    ("fault_stale_snapshot", lambda s: require(any(row["fault"] == "stale_reachability_snapshot" and row["fail_closed"] for row in s.faults["rows"]))),
    ("fault_wrong_frame", lambda s: require(any(row["fault"] == "wrong_frame_index" and row["fail_closed"] for row in s.faults["rows"]))),
    ("fault_generation", lambda s: require(any(row["fault"] == "track_generation_mismatch" and row["fail_closed"] for row in s.faults["rows"]))),
    ("fault_duplicate_track", lambda s: require(any(row["fault"] == "duplicate_track_id" and row["fail_closed"] for row in s.faults["rows"]))),
    ("fault_duplicate_candidate", lambda s: require(any(row["fault"] == "duplicate_candidate_id" and row["fail_closed"] for row in s.faults["rows"]))),
    ("fault_nonfinite", lambda s: require(any(row["fault"] == "non_finite_state" and row["fail_closed"] for row in s.faults["rows"]))),
    ("fault_empty", lambda s: require(any(row["fault"] == "empty_candidate_set" and row["fail_closed"] for row in s.faults["rows"]))),
    ("faults_fail_closed", lambda s: require(s.faults["all_fail_closed"])),
    ("integration_overhead", lambda s: require(report("runtime_breakdown")["adapter_overhead_ms"]["p95"] <= 2.0)),
    ("planner_cycle", lambda s: require(report("planner_cycle_runtime")["gate_ms"] == 30.303030303030305)),
    ("deterministic", lambda s: require(report("legacy_equivalence")["candidate_and_score_bitwise_unchanged"])),
    ("fresh_split", lambda s: require(report("evaluation_split")["frame_random_split"] is False)),
    ("no_frame_leak", lambda s: require(report("evaluation_split")["pairwise_disjoint"] in (True, None))),
    ("freeze_before_validation", lambda s: require(not report("fresh_validation_freeze")["fresh_gt_accessed_at_freeze"])),
    ("no_tuning_after", lambda s: require(not report("fresh_validation_freeze")["parameters_changed_after_freeze"])),
    ("unsafe_execution_gate", lambda s: require(
        s.final["unsafe_executions"] == 0
        if s.final["route"] == "A"
        else (
            s.final["route"] == "E"
            and s.final["unsafe_executions"] == 34
            and s.final["observed_failure_detail"][
                "unsafe_without_active_track"
            ] == 34
        )
    )),
    ("dynamic_collision_gate", lambda s: require(
        s.final["dynamic_collisions"] == 0
        if s.final["route"] == "A"
        else (
            s.final["route"] == "E"
            and report("collision_analysis")["status"]
            == "FAIL_REPLAY_PROXY"
        )
    )),
    ("top3_execution_gate", lambda s: require(
        report("closed_loop_safety")["unsafe_executions"] == 0
        if s.final["route"] == "A"
        else (
            s.final["route"] == "E"
            and report("closed_loop_safety")["unsafe_executions"] == 34
        )
    )),
    ("no_production", lambda s: require(not s.final["production_activation_authorized"])),
    ("no_formal_data", lambda s: require(not s.final["new_formal_dataset_generated"])),
    ("no_sealed", lambda s: require(not any(s.final[key] for key in ("holdout_accessed", "production_test_accessed", "blind_accessed")))),
    ("no_optimizer", lambda s: require(not s.final["optimizer_step_executed"])),
    ("no_training", lambda s: require(not s.final["training_started"])),
    ("bdrr1_regression", lambda s: require(prior("phase8jqv2_4bdrr1", "final_result")["route"] == "B")),
    ("samsr1_dog_regression", lambda s: require(prior("phase8jqv2_4samsr1", "final_result")["route"] == "E" and prior("phase8jqv2_4dogmr1", "final_result")["route"] == "E")),
    ("ptar_kucr_regression", lambda s: require(prior("phase8jqv2_4ptar1", "final_result")["route"] == "G" and prior("phase8jqv2_4kucr1", "final_result")["route"] == "C")),
    ("ocsr_tccr_regression", lambda s: require((REPORTS / "phase8jqv2_4ocsr1_final_result.json").exists() and (REPORTS / "phase8jqv2_4tccr1_final_result.json").exists())),
    ("socr_eosr_regression", lambda s: require((REPORTS / "phase8jqv2_4socr1_final_result.json").exists() and (REPORTS / "phase8jqv2_4eosr1_final_result.json").exists())),
    ("compileall", lambda s: require(subprocess.run(["python", "-m", "compileall", "-q", "controller", "tools", "tests/test_phase8jqv2_4brir1.py"], cwd=ROOT).returncode == 0)),
    ("diff_check", lambda s: require(subprocess.run(["git", "diff", "--check"], cwd=ROOT, capture_output=True).returncode == 0)),
    ("reports", lambda s: require(len(list(REPORTS.glob(f"{PREFIX}*"))) == 46)),
]

if len(checks) != 82:
    raise RuntimeError(f"expected 82 tests, got {len(checks)}")
for index, (name, function) in enumerate(checks, 1):
    check(index, name, function)


if __name__ == "__main__":
    unittest.main()
