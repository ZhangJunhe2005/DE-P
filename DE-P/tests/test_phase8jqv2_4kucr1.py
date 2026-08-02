"""KUCR1 fail-closed integrity-audit tests using unittest only."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np
import yaml

from controller.dynamic_safety_shadow_adapter_v2 import (
    BoundedCoastingSafetyAdapter,
)


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/"reports"


def load(suffix):
    return json.loads((
        REPORTS/f"phase8jqv2_4kucr1_{suffix}.json"
    ).read_text())


def require(value):
    if not value:
        raise AssertionError("KUCR1 requirement failed")


CHECKS = []


def check(name):
    def register(function):
        CHECKS.append((name, function))
        return function
    return register


@check("01_ocsr_route_c")
def _(): require(load("entry_gate")["checks"]["ocsr1_route_c"])


@check("02_adapter_v2_frozen")
def _(): require(load("frozen_artifacts")["ocsr1_adapter_v2_modified"] is False)


@check("03_gt_evaluator_v1_frozen")
def _():
    require("tools/evaluate_yopo_dynamic_candidate_risk_v1.py" in load("frozen_artifacts")["artifacts"])


@check("04_track_manager_frozen")
def _(): require(not load("regression")["TrackManager_algorithm_modified"])


@check("05_kalman_qr_frozen")
def _():
    result=load("regression")
    require(not result["kalman_process_model_modified"] and not result["kalman_measurement_model_modified"])


@check("06_yopo_frozen")
def _(): require(load("compatibility_matrix")["YOPO"]=="unchanged")


@check("07_historical_validation_observed")
def _():
    require(load("historical_validation_status")["ocsr1_validation_status"]=="HISTORICAL_OBSERVED_VALIDATION")


@check("08_timeline_origin")
def _(): require(load("timeline_integrity")["sample_time_origin"].startswith("relative"))


@check("09_no_double_prediction")
def _(): require(not load("timeline_integrity")["double_prediction_detected"])


@check("10_sample_time_alignment")
def _():
    require(all(abs(row["prediction_gt_time_difference_s"])<1e-5 for row in load("timeline_integrity")["manual_replays"]))


@check("11_gt_time_alignment")
def _(): require(load("timeline_integrity")["gt_timestamp_alignment"]=="PASS")


@check("12_world_frame_alignment")
def _(): require(load("coordinate_frame_integrity")["world_frame_alignment"]=="PASS")


@check("13_state_order")
def _():
    require(load("coordinate_frame_integrity")["kalman_state_order"]==["x","y","z","vx","vy","vz"])


@check("14_covariance_block_order")
def _():
    require(load("coordinate_frame_integrity")["covariance_blocks"]==["Sigma_pp","Sigma_pv","Sigma_vp","Sigma_vv"])


@check("15_seconds_unit")
def _(): require(load("timeline_integrity")["seconds_unit"])


@check("16_raw_baseline_reproduced")
def _(): require(load("u0_raw")["status"]=="HISTORICAL_BASELINE_REPRODUCED")


@check("17_whitened_residual_invalid_origin")
def _():
    require(load("whitened_residual_analysis")["status"]=="INVALID_FOR_ACTOR_CENTER_CALIBRATION")


@check("18_per_axis_bias")
def _(): require(len(load("bias_analysis")["by_horizon"]["1"]["mean_error_vector_m"])==3)


@check("19_negative_correlation_explained")
def _():
    require("surface" in load("error_root_cause")["negative_std_error_correlation_explanation"])


@check("20_grouped_split_fail_closed")
def _(): require(load("evaluation_split")["frame_split_performed"] is False)


@check("21_no_frame_leakage")
def _(): require(not load("evaluation_split")["frame_split_performed"])


@check("22_fresh_validation_not_accessed")
def _(): require(not load("fresh_validation_freeze")["fresh_validation_gt_read"])


@check("23_raw_replay_blocked")
def _(): require(load("uncertainty_replay_manifest")["records_created"]==0)


for number,suffix in enumerate((
    "u1_scalar","u2_affine_floor","u3_horizon","u4_conformal",
    "u5_directional","u6_acceleration","u7_shadow_kalman",
),24):
    @check(f"{number:02d}_{suffix}_blocked")
    def _candidate(suffix=suffix):
        require(load(suffix)["status"]=="NOT_RUN_BLOCKED_BY_INTEGRITY_GATE")


@check("31_u7_order_guard")
def _(): require(load("u7_shadow_kalman")["parameters_evaluated"]==0)


@check("32_bounded_candidate_search_not_expanded")
def _():
    document=yaml.safe_load((ROOT/"configs/kalman_uncertainty_contract_v1_candidate.yaml").read_text())
    require(set(document["candidates"])=={"U0_RAW","U1_SCALAR","U2_AFFINE_FLOOR","U3_HORIZON","U4_CONFORMAL","U5_DIRECTIONAL","U6_ACCELERATION","U7_SHADOW_KALMAN"})


@check("33_no_validation_retuning")
def _(): require(load("candidate_selection")["selected_candidate"] is None)


for number,suffix in enumerate((
    "empirical_coverage","empirical_coverage","normalized_error",
    "normalized_error","decision_risk_metrics",
    "top_ranked_candidate_safety","candidate_risk_metrics",
    "false_veto_analysis","decision_risk_metrics",
),34):
    @check(f"{number:02d}_{suffix}_blocked")
    def _metric(suffix=suffix):
        require(load(suffix)["status"]=="NOT_RUN_BLOCKED_BY_INTEGRITY_GATE")


def decision(candidates, tracks, original=0):
    adapter=BoundedCoastingSafetyAdapter()
    return adapter.evaluate(
        np.asarray(candidates,float),[.1,.2],tracks,
        timestamp=0.,frame_index=0,original_candidate_id=original,
        candidate_scores=np.arange(len(candidates),dtype=float),
    )


def active_track():
    return {
        "track_id":0,"state_generation":"0:0",
        "position_world":np.zeros(3),"velocity_world":np.zeros(3),
        "state_covariance":np.eye(6)*.01,"age":3,"missed_count":0,
        "is_confirmed":True,"is_dynamic":True,
        "attention_authorized":True,"measurement_present":True,
        "consecutive_direct_hits":3,
    }


@check("43_keep_original")
def _():
    result=decision([[[10,0,0],[10,0,0]]],[active_track()])
    require(result["decision_status"]=="KEEP_ORIGINAL")


@check("44_switch_safe")
def _():
    result=decision([[[0,0,0],[0,0,0]],[[10,0,0],[10,0,0]]],[active_track()])
    require(result["decision_status"]=="SWITCH_TO_SAFE_CANDIDATE")


@check("45_no_safe_candidate")
def _():
    result=decision([[[0,0,0],[0,0,0]],[[.1,0,0],[.1,0,0]]],[active_track()])
    require(result["decision_status"]=="NO_SAFE_CANDIDATE")


@check("46_no_unsafe_fallback")
def _():
    result=decision([[[0,0,0],[0,0,0]]],[active_track()])
    require(result["recommended_candidate_id"] is None)


@check("47_negative_false_veto_zero")
def _(): require(load("negative_validation")["false_veto"]==0)


@check("48_negative_false_coasting_zero")
def _(): require(load("negative_validation")["false_coasting"]==0)


@check("49_multi_target_historical")
def _(): require(load("multi_target_validation")["tracks"]==3)


@check("50_gap1_historical")
def _(): require(load("gap1_validation")["coverage"]=="26/26")


@check("51_gap2_not_approved")
def _(): require(load("gap2_validation")["status"]=="HISTORICAL_ONLY_NOT_APPROVED")


@check("52_gap3_expiry")
def _(): require(not load("gap3_non_scope")["automatically_enabled"])


@check("53_state_machine_unchanged")
def _(): require(load("adapter_v2_1_contract")["state_machine_change"] is False)


@check("54_runtime_gt_false")
def _(): require(not load("final_result")["runtime_gt_used"])


@check("55_adapter_v2_1_not_created")
def _(): require(not load("adapter_v2_1_contract")["adapter_v2_1_created"])


@check("56_deterministic")
def _(): require(load("determinism")["status"]=="PASS")


FINAL_FALSE=(
    "new_maps_generated","new_formal_dataset_generated","holdout_accessed",
    "formal_v3_entry_created","optimizer_step_executed","training_started",
)
for number,key in enumerate(FINAL_FALSE,57):
    @check(f"{number:02d}_{key}_false")
    def _false(key=key): require(load("final_result")[key] is False)


@check("63_eosr_failure_preserved")
def _(): require(load("final_result")["natural_eosr1_tracker_gate"]=="FAIL_SEPARATE")


@check("64_ocsr_regression")
def _(): require(load("regression")["ocsr1_validation_results_modified"] is False)


@check("65_prior_phase_regression")
def _(): require(load("regression")["frozen_hashes_current"])


@check("66_route_c")
def _():
    final=load("final_result")
    require(final["route"]=="C" and final["status"]=="FAIL")


@check("67_reference_origin_specific_cause")
def _():
    require("surface_cluster_centroid" in load("final_result")["specific_cause"])


@check("68_report_completeness")
def _():
    required=[
        "entry_gate","frozen_artifacts","historical_validation_status",
        "timeline_integrity","coordinate_frame_integrity","error_root_cause",
        "whitened_residual_analysis","bias_analysis",
        "scenario_conditioned_error","uncertainty_replay_manifest",
        "evaluation_split","fresh_validation_freeze","validation_freeze",
        "u0_raw","u1_scalar","u2_affine_floor","u3_horizon",
        "u4_conformal","u5_directional","u6_acceleration",
        "u7_shadow_kalman","candidate_comparison","empirical_coverage",
        "normalized_error","envelope_sharpness","candidate_risk_metrics",
        "decision_risk_metrics","top_ranked_candidate_safety",
        "no_safe_candidate_validation","false_veto_analysis",
        "adapter_v2_1_contract","negative_validation",
        "multi_target_validation","gap1_validation","gap2_validation",
        "gap3_non_scope","runtime","determinism","regression",
        "candidate_selection","compatibility_matrix","final_result",
    ]
    require(all((REPORTS/f"phase8jqv2_4kucr1_{name}.json").is_file() for name in required))


class TestPhase8JKUCR1(unittest.TestCase):
    pass


def wrap(function):
    def method(self): function()
    return method


for name,function in CHECKS:
    setattr(TestPhase8JKUCR1,f"test_{name}",wrap(function))


if __name__=="__main__":
    unittest.main(verbosity=2)
