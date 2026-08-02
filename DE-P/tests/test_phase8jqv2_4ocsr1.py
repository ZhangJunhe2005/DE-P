"""Standard-library regression suite for Phase 8J-Q2.4-OCSR1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import yaml

from controller.dynamic_safety_shadow_adapter_v2 import (
    BoundedCoastingSafetyAdapter, CoastingSafetyConfig, predict_state,
)
from tools.evaluate_yopo_dynamic_candidate_risk_v1 import (
    compare_shadow_with_gt, evaluate_gt_candidate_risk,
)


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def report(suffix):
    return json.loads((
        REPORTS / f"phase8jqv2_4ocsr1_{suffix}.json"
    ).read_text())


def require(value, message="requirement failed"):
    if not value:
        raise AssertionError(message)


def track(
    frame, *, track_id=0, generation="0:0", direct=True,
    dynamic=True, attention=True, confirmed=True, missed=0,
    position=(2., 0., 0.), velocity=(1., 0., 0.),
    covariance=None, consecutive=3,
):
    return {
        "track_id": track_id,
        "state_generation": generation,
        "position_world": np.asarray(position, dtype=float),
        "velocity_world": np.asarray(velocity, dtype=float),
        "state_covariance": (
            np.eye(6) * .01 if covariance is None else covariance
        ),
        "age": frame + 3,
        "missed_count": missed,
        "is_confirmed": confirmed,
        "is_dynamic": dynamic,
        "attention_authorized": attention,
        "measurement_present": direct,
        "consecutive_direct_hits": consecutive,
        "split_merge_suspected": False,
        "birth_frame": 0,
        "birth_observation_id": 0,
    }


def transition(*rows):
    adapter = BoundedCoastingSafetyAdapter()
    outputs = []
    for frame, row in enumerate(rows):
        outputs.append(adapter.update_safety_states(
            row, frame * .1, frame
        ))
    return adapter, outputs


CHECKS = []


def check(name):
    def decorator(function):
        CHECKS.append((name, function))
        return function
    return decorator


@check("01_tccr_route_b_entry")
def _():
    require(report("entry_gate")["checks"]["route_b"])


@check("02_confirmed_dynamic_33")
def _():
    require(report("entry_gate")["checks"]["confirmed_dynamic_tracks"])


@check("03_gap1_26_entry")
def _():
    require(report("entry_gate")["checks"]["gap1_events"])


@check("04_gap2_16_entry")
def _():
    require(report("entry_gate")["checks"]["gap2_events"])


@check("05_negative_15_entry")
def _():
    require(report("entry_gate")["checks"]["negative_sequences"])


@check("06_adapter_v1_frozen")
def _():
    require(report("frozen_artifacts")["tccr1_adapter_v1_frozen"])


@check("07_track_manager_frozen")
def _():
    require(report("regression")["TrackManager_algorithm_modified"] is False)


@check("08_detector_frozen")
def _():
    require(report("regression")["detector_algorithm_modified"] is False)


@check("09_checkpoint_frozen")
def _():
    require(report("regression")["static_yopo_checkpoint_modified"] is False)


@check("10_recent_dynamic_bounded_memory")
def _():
    require(report("recent_dynamic_memory")["bounded"])


@check("11_id_generation_protection")
def _():
    adapter, _ = transition([track(0)])
    rows, _, expiry, _ = adapter.update_safety_states(
        [track(1, generation="1:1")], .1, 1
    )
    require(any(row["expiry_reason"] == "id_generation_changed" for row in expiry))


@check("12_deleted_track_expiry")
def _():
    adapter, _ = transition([track(0)])
    _, _, expiry, _ = adapter.update_safety_states([], .1, 1)
    require(expiry[0]["expiry_reason"] == "track_deleted")


@check("13_missed_count_expiry")
def _():
    adapter, _ = transition([track(0)])
    rows, _, _, _ = adapter.update_safety_states(
        [track(1, direct=False, dynamic=False, attention=False, missed=3)],
        .2, 1,
    )
    require(rows[0]["safety_state"] == "EXPIRED")


@check("14_time_expiry")
def _():
    adapter, _ = transition([track(0)])
    rows, _, _, _ = adapter.update_safety_states(
        [track(1, direct=False, dynamic=False, attention=False, missed=2)],
        .3, 1,
    )
    require(rows[0]["reason"] == "coasting_time_exceeded")


@check("15_position_uncertainty_expiry")
def _():
    adapter, _ = transition([track(0)])
    covariance = np.eye(6) * .01
    covariance[:3, :3] = np.eye(3)
    rows, _, _, _ = adapter.update_safety_states(
        [track(1, direct=False, dynamic=False, attention=False, missed=1,
               covariance=covariance)], .1, 1,
    )
    require(rows[0]["reason"] == "position_uncertainty_exceeded")


@check("16_velocity_uncertainty_expiry")
def _():
    adapter, _ = transition([track(0)])
    covariance = np.eye(6) * .01
    covariance[3:, 3:] = np.eye(3) * 4.
    rows, _, _, _ = adapter.update_safety_states(
        [track(1, direct=False, dynamic=False, attention=False, missed=1,
               covariance=covariance)], .1, 1,
    )
    require(rows[0]["reason"] == "velocity_uncertainty_exceeded")


@check("17_invalid_covariance")
def _():
    invalid = np.eye(6)
    invalid[0, 0] = np.nan
    adapter = BoundedCoastingSafetyAdapter()
    _, rejected, _, _ = adapter.update_safety_states(
        [track(0, covariance=invalid)], 0., 0
    )
    require(rejected[0]["safety_state"] == "INVALID")


@check("18_observed_dynamic")
def _():
    _, outputs = transition([track(0)])
    require(outputs[0][0][0]["safety_state"] == "OBSERVED_DYNAMIC")


@check("19_coasting_dynamic")
def _():
    _, outputs = transition(
        [track(0)],
        [track(1, direct=False, dynamic=False, attention=False, missed=1)],
    )
    require(outputs[1][0][0]["safety_state"] == "COASTING_DYNAMIC")


@check("20_reacquired_uncertain")
def _():
    _, outputs = transition(
        [track(0)],
        [track(1, direct=True, dynamic=False, attention=False,
               velocity=(1., 0., 0.), consecutive=1)],
    )
    require(outputs[1][0][0]["safety_state"] == "REACQUIRED_UNCERTAIN")


@check("21_observed_non_dynamic")
def _():
    _, outputs = transition([
        track(0, dynamic=False, attention=False, velocity=(0., 0., 0.))
    ])
    require(outputs[0][0][0]["safety_state"] == "OBSERVED_NON_DYNAMIC")


@check("22_expired_state")
def _():
    _, outputs = transition([
        track(0, direct=False, dynamic=False, attention=False, missed=1)
    ])
    require(outputs[0][0][0]["safety_state"] == "EXPIRED")


@check("23_invalid_evaluation")
def _():
    adapter = BoundedCoastingSafetyAdapter()
    result = adapter.evaluate(
        np.zeros((1, 2, 2)), [.1, .2], [],
        timestamp=0., frame_index=0, original_candidate_id=0,
    )
    require(result["decision_status"] == "INVALID_EVALUATION")


@check("24_coasting_cannot_birth")
def _():
    _, outputs = transition([
        track(0, direct=False, dynamic=False, attention=False, missed=1)
    ])
    require(outputs[0][0][0]["safety_state"] != "COASTING_DYNAMIC")


@check("25_coasting_requires_prior_dynamic")
def _():
    _, outputs = transition([
        track(0, direct=True, dynamic=False, attention=False)
    ], [
        track(1, direct=False, dynamic=False, attention=False, missed=1)
    ])
    require(outputs[1][0][0]["safety_state"] != "COASTING_DYNAMIC")


@check("26_gap1_coverage")
def _():
    result = report("gap1_validation")
    require(result["covered"] == result["events"] == 26)


@check("27_gap2_evaluated")
def _():
    require(report("gap2_validation")["events"] == 16)


@check("28_gap3_not_allowed")
def _():
    require(report("long_occlusion_validation")["gap3"]["covered"] == 0)


for number, horizon in enumerate((1, 2, 3), 29):
    @check(f"{number:02d}_prediction_horizon_{horizon}")
    def _horizon(horizon=horizon):
        require(
            report("prediction_error_by_horizon")["by_horizon"][
                "validation"
            ][str(horizon)]["sample_count"] > 0
        )


@check("32_gt_not_fed_to_prediction")
def _():
    require(not report("prediction_error_by_horizon")["gt_feedback_to_prediction"])


for number, key in enumerate((
    "coverage_1sigma", "coverage_2sigma", "coverage_3sigma",
), 33):
    @check(f"{number:02d}_{key}")
    def _coverage(key=key):
        require(key in report("covariance_calibration")["by_horizon"]["validation"]["2"])


@check("36_mahalanobis_calibration")
def _():
    require(report("covariance_calibration")["status"] == "FAIL")


@check("37_bounded_parameter_grid")
def _():
    require(len(report("covariance_calibration")["parameter_grid"]["covariance_sigma"]) == 4)


@check("38_case_grouped_split")
def _():
    split = report("evaluation_split")
    require(not set(split["calibration_case_ids"]) & set(split["validation_case_ids"]))


@check("39_no_tuning_after_freeze")
def _():
    require(not report("validation_freeze")["post_validation_tuning_allowed"])


@check("40_gt_collision")
def _():
    gt = evaluate_gt_candidate_risk(
        np.zeros((1, 2, 3)), [.1, .2],
        np.zeros((1, 2, 3)), [.3],
    )
    require(gt["candidate_rows"][0]["classification"] == "GT_COLLISION")


@check("41_gt_safe")
def _():
    gt = evaluate_gt_candidate_risk(
        np.ones((1, 2, 3)) * 10., [.1, .2],
        np.zeros((1, 2, 3)), [.3],
    )
    require(gt["candidate_rows"][0]["classification"] == "GT_SAFE")


@check("42_multi_target_minimum_clearance")
def _():
    require(len(report("multi_target_validation")["distinct_limiting_track_ids"]) >= 2)


def evaluate_decision(candidates, original=0, tracks=None):
    adapter = BoundedCoastingSafetyAdapter()
    tracks = [track(0, position=(0., 0., 0.), velocity=(0., 0., 0.))] if tracks is None else tracks
    return adapter.evaluate(
        np.asarray(candidates, dtype=float), [.1, .2], tracks,
        timestamp=0., frame_index=0, original_candidate_id=original,
        candidate_scores=np.arange(len(candidates), dtype=float),
    )


@check("43_unsafe_veto")
def _():
    result = evaluate_decision([[[0., 0., 0.], [0., 0., 0.]]])
    require(result["candidate_rows"][0]["would_veto"])


@check("44_safe_retain")
def _():
    result = evaluate_decision([[[10., 0., 0.], [10., 0., 0.]]])
    require(not result["candidate_rows"][0]["would_veto"])


@check("45_missed_unsafe_reported")
def _():
    require(report("candidate_level_metrics")["missed_unsafe"] > 0)


@check("46_false_veto_metric")
def _():
    require(report("candidate_level_metrics")["safe_candidate_false_veto_rate"] > 0)


@check("47_keep_original")
def _():
    result = evaluate_decision([[[10., 0., 0.], [10., 0., 0.]]])
    require(result["decision_status"] == "KEEP_ORIGINAL")


@check("48_switch_to_safe")
def _():
    result = evaluate_decision([
        [[0., 0., 0.], [0., 0., 0.]],
        [[10., 0., 0.], [10., 0., 0.]],
    ])
    require(
        result["decision_status"] == "SWITCH_TO_SAFE_CANDIDATE"
        and result["recommended_candidate_id"] == 1
    )


@check("49_no_safe_candidate")
def _():
    result = evaluate_decision([
        [[0., 0., 0.], [0., 0., 0.]],
        [[.1, 0., 0.], [.1, 0., 0.]],
    ])
    require(result["decision_status"] == "NO_SAFE_CANDIDATE")


@check("50_no_unsafe_fallback")
def _():
    result = evaluate_decision([
        [[0., 0., 0.], [0., 0., 0.]],
        [[.1, 0., 0.], [.1, 0., 0.]],
    ])
    require(result["recommended_candidate_id"] is None)


@check("51_no_active_dynamic_risk")
def _():
    result = evaluate_decision(
        [[[0., 0., 0.], [0., 0., 0.]]], tracks=[]
    )
    require(result["decision_status"] == "NO_ACTIVE_DYNAMIC_RISK")


@check("52_invalid_evaluation_output")
def _():
    adapter = BoundedCoastingSafetyAdapter()
    result = adapter.evaluate(
        np.full((1, 2, 3), np.nan), [.1, .2], [],
        timestamp=0., frame_index=0, original_candidate_id=0,
    )
    require(result["recommended_candidate_id"] is None)


for number, key in enumerate((
    "false_observed_dynamic", "false_coasting",
    "false_candidate_veto", "false_no_safe_candidate",
), 53):
    @check(f"{number:02d}_negative_{key}")
    def _negative(key=key):
        require(report("negative_validation")[key] == 0)


@check("57_long_occlusion_expiry")
def _():
    require(report("long_occlusion_validation")["status"] == "PASS")


@check("58_actor_stop_expiry")
def _():
    require(report("long_occlusion_validation")["stable_stop_expiry_supported"])


@check("59_track_deletion_expiry")
def _():
    require(report("long_occlusion_validation")["track_deletion_expiry_supported"])


@check("60_adapter_deterministic")
def _():
    candidates = [[[10., 0., 0.], [10., 0., 0.]]]
    left = evaluate_decision(candidates)
    right = evaluate_decision(candidates)
    left.pop("runtime_ms"); right.pop("runtime_ms")
    require(left == right)


FINAL_FALSE_KEYS = (
    "formal_command_modified", "runtime_gt_used",
    "new_maps_generated", "new_dataset_generated", "holdout_accessed",
    "formal_generation_started", "production_test_accessed",
    "blind_accessed", "optimizer_step_executed", "training_started",
)
for number, key in enumerate(FINAL_FALSE_KEYS, 61):
    @check(f"{number:02d}_{key}_false")
    def _final_false(key=key):
        require(report("final_result")[key] is False)


@check("71_eosr_failure_preserved")
def _():
    require(report("final_result")["natural_eosr1_tracker_gate"] == "FAIL_SEPARATE")


@check("72_tccr_regression")
def _():
    require(report("regression")["frozen_artifact_hashes_current"])


@check("73_socr_eosr_regression")
def _():
    require(report("frozen_artifacts")["eosr1_witnesses_frozen"])


@check("74_route_c")
def _():
    final = report("final_result")
    require(final["status"] == "FAIL" and final["route"] == "C")


@check("75_report_completeness")
def _():
    required = [
        "entry_gate", "frozen_artifacts", "evaluation_split",
        "validation_freeze", "safety_state_machine",
        "recent_dynamic_memory", "expiry_semantics",
        "prediction_error_by_horizon", "covariance_calibration",
        "position_error_distribution", "velocity_error_distribution",
        "gt_candidate_risk", "contract_comparison",
        "candidate_level_metrics", "decision_level_metrics",
        "no_safe_candidate_validation", "gap1_validation",
        "gap2_validation", "long_occlusion_validation",
        "negative_validation", "multi_target_validation",
        "adapter_v2_contract", "planner_interface_spec",
        "compatibility_matrix", "runtime", "determinism",
        "regression", "final_result",
    ]
    require(all((
        REPORTS / f"phase8jqv2_4ocsr1_{name}.json"
    ).is_file() for name in required))


class TestPhase8JOCSR1(unittest.TestCase):
    pass


def wrapper(function):
    def method(self):
        function()
    return method


for test_name, function in CHECKS:
    setattr(TestPhase8JOCSR1, f"test_{test_name}", wrapper(function))


if __name__ == "__main__":
    unittest.main(verbosity=2)
