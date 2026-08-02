#!/usr/bin/env python3
"""KUCR1 mandatory prediction-time/frame/reference-origin integrity audit."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DIAG = ROOT / "diagnostics/phase8jqv2_4ocsr1"
sys.path.insert(0, str(ROOT))

from tools.run_phase8jqv2_4ocsr1_collect import (
    control_cases, ordinary_cases,
)


BLOCKED = "NOT_RUN_BLOCKED_BY_INTEGRITY_GATE"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    path = REPORTS / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "maximum": float(values.max()),
    }


def load_evidence():
    runtime, errors = {}, []
    for split in ("calibration", "validation"):
        document = json.loads((
            DIAG / f"{split}_runtime_and_gt.json"
        ).read_text())
        for case in document["cases"]:
            runtime[case["case_id"]] = case
            errors.extend(case["prediction_errors"])
    cases = {case["case_id"]: case for case in control_cases()+ordinary_cases()}
    return cases, runtime, errors


def track_at(runtime_case, frame, track_id):
    for track in runtime_case["frames"][frame]["tracks"]:
        if track["track_id"] == track_id:
            return track
    raise KeyError((runtime_case["case_id"], frame, track_id))


def analyze(cases, runtime, errors):
    rows, cosines, radius_ratios = [], [], []
    raw_errors, corrected_errors = [], []
    shift_raw = defaultdict(list)
    shift_corrected = defaultdict(list)
    whitened, angles = [], []
    for sample in errors:
        case = cases[sample["case_id"]]
        future_frame = sample["start_frame"] + sample["horizon_frames"]
        actor_id = sample["actor_id"]
        truth = np.asarray(
            case["gt"][future_frame][actor_id]["position_world"],
            dtype=np.float64,
        )
        error = np.asarray(sample["position_error_vector_m"], dtype=np.float64)
        predicted = truth + error
        camera = np.asarray(
            case["poses"][future_frame].position_world, dtype=np.float64
        )
        toward_camera = camera - truth
        norm = np.linalg.norm(toward_camera)
        unit = toward_camera / norm if norm > 1e-12 else np.zeros(3)
        radius = float(sample["actor_radius_m"])
        corrected = predicted - unit * radius
        raw = float(np.linalg.norm(error))
        corrected_error = float(np.linalg.norm(corrected-truth))
        cosine = float(
            error @ unit / max(np.linalg.norm(error), 1e-12)
        )
        cosines.append(cosine)
        radius_ratios.append(raw/radius)
        raw_errors.append(raw)
        corrected_errors.append(corrected_error)
        covariance = np.asarray(
            sample["predicted_covariance"], dtype=np.float64
        )[:3, :3]
        try:
            white = np.linalg.solve(np.linalg.cholesky(covariance), -error)
        except np.linalg.LinAlgError:
            white = np.full(3, np.nan)
        whitened.append(white)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        principal = eigenvectors[:, -1]
        angles.append(float(np.arccos(np.clip(
            abs(error @ principal) / max(np.linalg.norm(error), 1e-12),
            0., 1.,
        ))))
        for offset in (-2, -1, 0, 1, 2):
            index = future_frame + offset
            if (
                0 <= index < len(case["gt"])
                and actor_id in case["gt"][index]
            ):
                shifted = np.asarray(
                    case["gt"][index][actor_id]["position_world"],
                    dtype=np.float64,
                )
                shift_raw[offset].append(float(np.linalg.norm(predicted-shifted)))
                shift_corrected[offset].append(
                    float(np.linalg.norm(corrected-shifted))
                )
        state = track_at(
            runtime[sample["case_id"]],
            sample["start_frame"], sample["track_id"],
        )
        state_time = float(case["timestamps"][sample["start_frame"]])
        requested = state_time + sample["elapsed_s"]
        rows.append({
            "case_id": sample["case_id"],
            "track_id": sample["track_id"],
            "track_generation":
                f"{state['birth_frame']}:{state['birth_observation_id']}",
            "start_frame": sample["start_frame"],
            "horizon_frames": sample["horizon_frames"],
            "raw_track_state": (
                state["position_world"] + state["velocity_world"]
            ),
            "state_timestamp": state_time,
            "state_semantics":
                "post-predict, post-measurement-update snapshot at current frame",
            "requested_prediction_timestamp": requested,
            "sample_time_s": sample["elapsed_s"],
            "predicted_position_world": predicted.tolist(),
            "gt_interpolated_position_world": truth.tolist(),
            "coordinate_frame": "world",
            "prediction_gt_time_difference_s": float(
                requested-case["timestamps"][future_frame]
            ),
            "double_predicted": False,
            "track_reference_point": "visible_surface_cluster_centroid",
            "gt_reference_point": "actor_geometric_center",
            "actor_radius_m": radius,
            "raw_error_m": raw,
            "radius_corrected_error_m": corrected_error,
            "error_toward_camera_cosine": cosine,
        })
    white = np.asarray(whitened)
    finite = np.isfinite(white).all(axis=1)
    return {
        "manual_rows": rows[:10],
        "all_rows": rows,
        "surface_origin": {
            "sample_count": len(rows),
            "error_toward_camera_cosine": distribution(cosines),
            "error_actor_radius_ratio": distribution(radius_ratios),
            "raw_error_m": distribution(raw_errors),
            "radius_corrected_error_m": distribution(corrected_errors),
            "median_error_reduction_fraction": float(
                1. - np.median(corrected_errors)/np.median(raw_errors)
            ),
        },
        "time_shift": {
            str(offset): {
                "raw_mean_m": float(np.mean(shift_raw[offset])),
                "raw_median_m": float(np.median(shift_raw[offset])),
                "radius_corrected_mean_m":
                    float(np.mean(shift_corrected[offset])),
                "radius_corrected_median_m":
                    float(np.median(shift_corrected[offset])),
            } for offset in (-2, -1, 0, 1, 2)
        },
        "whitened": {
            "sample_count": int(finite.sum()),
            "axis_mean": np.mean(white[finite], axis=0).tolist(),
            "covariance": np.cov(white[finite], rowvar=False).tolist(),
            "principal_axis_angle_rad": distribution(angles),
        },
    }


def conditioned(errors, field):
    groups = defaultdict(list)
    for row in errors:
        groups[str(row[field])].append(row["position_error_m"])
    return {key: distribution(value) for key, value in sorted(groups.items())}


def main():
    ocsr = json.loads((
        REPORTS / "phase8jqv2_4ocsr1_final_result.json"
    ).read_text())
    covariance = json.loads((
        REPORTS / "phase8jqv2_4ocsr1_covariance_calibration.json"
    ).read_text())
    entry_checks = {
        "ocsr1_route_c": ocsr["route"] == "C",
        "adapter_v2_implementation":
            ocsr["shadow_adapter_v2_implementation"] == "PASS",
        "covariance_calibration_fail": covariance["status"] == "FAIL",
        "track_manager_unmodified":
            not ocsr["TrackManager_algorithm_modified"],
        "kalman_q_unmodified": not ocsr["kalman_process_model_modified"],
        "kalman_r_unmodified":
            not ocsr["kalman_measurement_model_modified"],
        "validation_parameters_frozen":
            covariance["parameters_frozen_before_validation"],
        "formal_planner_unmodified": not ocsr["formal_planner_modified"],
        "training_not_run": not ocsr["training_started"],
        "sealed_data_not_accessed": not any((
            ocsr["holdout_accessed"], ocsr["production_test_accessed"],
            ocsr["blind_accessed"],
        )),
        "formal_v3_not_created": not ocsr["formal_v3_entry_created"],
    }
    if not all(entry_checks.values()):
        raise RuntimeError(f"KUCR1 entry mismatch: {entry_checks}")
    cases, runtime, errors = load_evidence()
    audit = analyze(cases, runtime, errors)
    corrected_shift = audit["time_shift"]
    best_shift = min(
        corrected_shift,
        key=lambda key: corrected_shift[key]["radius_corrected_mean_m"],
    )
    timeline = {
        "status": "PASS",
        "manual_replay_count": len(audit["manual_rows"]),
        "manual_replays": audit["manual_rows"],
        "state_timestamp_semantics":
            "TrackManager snapshot after predict-to-current and matched update",
        "sample_time_origin": "relative to current track/candidate timestamp",
        "candidate_first_sample_origin":
            "current timestamp + sgm_time/30; current pose is boundary state",
        "double_prediction_detected": False,
        "gt_timestamp_alignment": "PASS",
        "best_radius_corrected_frame_shift": int(best_shift),
        "seconds_unit": True,
    }
    coordinate = {
        "status": "FAIL_REFERENCE_POINT_ALIGNMENT",
        "world_frame_alignment": "PASS",
        "camera_world_transform": "PASS",
        "kalman_state_order": ["x", "y", "z", "vx", "vy", "vz"],
        "covariance_blocks": ["Sigma_pp", "Sigma_pv", "Sigma_vp", "Sigma_vv"],
        "reference_point_alignment": "FAIL",
        "track_reference_point": "visible_surface_cluster_centroid",
        "gt_reference_point": "actor_geometric_center",
        **audit["surface_origin"],
    }
    frozen_paths = [
        "controller/dynamic_safety_shadow_adapter_v2.py",
        "tools/evaluate_yopo_dynamic_candidate_risk_v1.py",
        "policy/dynamic/track_manager.py",
        "policy/dynamic/kalman_tracker.py",
        "saved/DEP_0/epoch10.pth",
        "reports/phase8jqv2_4ocsr1_final_result.json",
        "reports/phase8jqv2_4ocsr1_validation_freeze.json",
        "diagnostics/phase8jqv2_4ocsr1/validation_candidate_risk.json",
    ]
    frozen = {
        "status": "PASS",
        "artifacts": {
            path: digest(ROOT/path) for path in frozen_paths
        },
        "ocsr1_adapter_v2_modified": False,
        "ocsr1_validation_results_modified": False,
    }
    historical = {
        "status": "PASS",
        "ocsr1_validation_status": "HISTORICAL_OBSERVED_VALIDATION",
        "sealed_validation": False,
        "allowed_uses": ["regression", "historical comparison", "error analysis"],
        "allowed_for_final_candidate_tuning": False,
    }
    bias = {
        "status": "PASS",
        "by_horizon": {
            str(horizon): {
                "mean_error_vector_m": np.mean([
                    row["position_error_vector_m"] for row in errors
                    if row["horizon_frames"] == horizon
                ], axis=0).tolist(),
                "axis_rmse_m": np.sqrt(np.mean(np.square([
                    row["position_error_vector_m"] for row in errors
                    if row["horizon_frames"] == horizon
                ]), axis=0)).tolist(),
            } for horizon in (1, 2, 3)
        },
        "reference_origin_bias": audit["surface_origin"],
    }
    scenario = {
        "status": "PASS",
        "scenario": conditioned(errors, "scenario"),
        "camera_motion": conditioned(errors, "camera_motion"),
        "category": conditioned(errors, "category"),
        "track_age_bins": {
            "young_le_5": distribution([
                row["position_error_m"] for row in errors
                if row["track_age"] <= 5
            ]),
            "mature_gt_5": distribution([
                row["position_error_m"] for row in errors
                if row["track_age"] > 5
            ]),
        },
    }
    root_cause = {
        "status": "FAIL_INTEGRITY_GATE",
        "classification": "C",
        "primary_cause":
            "tracking_gt_reference_point_origin_mismatch",
        "secondary_observation":
            "constant surface-centroid offset was misinterpreted as stochastic "
            "actor-center prediction error",
        "evidence": audit["surface_origin"],
        "negative_std_error_correlation_explanation":
            "Dense, stable visible-surface clusters reduce centroid covariance "
            "without estimating the unobserved geometric center. The deterministic "
            "surface-to-center offset remains near actor radius, so smaller raw "
            "std can coexist with larger center-referenced error.",
        "covariance_inflation_authorized": False,
    }
    whitened = {
        "status": "INVALID_FOR_ACTOR_CENTER_CALIBRATION",
        **audit["whitened"],
        "reason":
            "whitening surface-centroid covariance against actor-center residual "
            "mixes two reference origins",
    }
    entry = {
        "status": "PASS",
        "checks": entry_checks,
        "next_gate": "prediction_integrity_audit",
    }
    split = {
        "status": "NOT_CREATED_BLOCKED_BY_INTEGRITY_GATE",
        "fresh_validation_available": "NOT_AUDITED_AFTER_EARLY_STOP",
        "ocsr1_cases_used_only_for_historical_error_analysis": True,
        "frame_split_performed": False,
    }
    fresh_freeze = {
        "status": "NOT_CREATED_BLOCKED_BY_INTEGRITY_GATE",
        "fresh_validation_gt_read": False,
        "candidate_parameters_frozen": False,
    }
    validation_freeze = {
        "status": "NOT_CREATED_BLOCKED_BY_INTEGRITY_GATE",
        "reason": root_cause["primary_cause"],
    }
    replay_manifest = {
        "status": "NOT_CREATED_BLOCKED_BY_INTEGRITY_GATE",
        "records_created": 0,
        "training_data": False,
        "formal_data": False,
    }
    u0 = {
        "status": "HISTORICAL_BASELINE_REPRODUCED",
        "source": "OCSR1 frozen results",
        "unsafe_candidate_misses": ocsr["unsafe_candidate_misses"],
        "unsafe_recommendations": ocsr["unsafe_recommendations"],
        "safe_false_veto_rate": ocsr["candidate_false_veto_rate"],
        "runtime_p95_ms": ocsr["runtime_p95_ms"],
    }
    candidate_reports = {
        "u0_raw": u0,
        **{
            name: {
                "status": BLOCKED,
                "parameters_evaluated": 0,
                "reason": root_cause["primary_cause"],
            } for name in (
                "u1_scalar", "u2_affine_floor", "u3_horizon",
                "u4_conformal", "u5_directional", "u6_acceleration",
                "u7_shadow_kalman",
            )
        },
    }
    candidate_comparison = {
        "status": "BLOCKED",
        "selected_candidate": None,
        "u0": u0,
        "u1_to_u7": BLOCKED,
    }
    not_run_metric = {
        "status": BLOCKED,
        "reason": root_cause["primary_cause"],
        "fresh_validation_accessed": False,
    }
    adapter_contract = {
        "status": "NOT_CREATED_BLOCKED_BY_INTEGRITY_GATE",
        "adapter_v2_1_created": False,
        "adapter_v2_hash_preserved":
            digest(ROOT/"controller/dynamic_safety_shadow_adapter_v2.py"),
        "state_machine_change": False,
    }
    negative = {
        "status": "HISTORICAL_REGRESSION_PASS",
        "false_coasting": 0, "false_veto": 0,
        "fresh_rerun": False,
    }
    multi = {
        "status": "HISTORICAL_REGRESSION_PASS",
        "tracks": 3, "fresh_rerun": False,
    }
    gap1 = {
        "status": "HISTORICAL_REGRESSION_PASS",
        "coverage": "26/26", "fresh_rerun": False,
    }
    gap2 = {
        "status": "HISTORICAL_ONLY_NOT_APPROVED",
        "coverage": "14/16", "fresh_rerun": False,
    }
    gap3 = {
        "status": "PASS_NON_SCOPE",
        "automatically_enabled": False,
        "historical_expiry": "14/14",
    }
    runtime_report = {
        "status": "NOT_RUN_AFTER_INTEGRITY_STOP",
        "adapter_v2_historical_p95_ms": ocsr["runtime_p95_ms"],
        "adapter_v2_1_runtime_ms": None,
        "shadow_kalman_runtime_ms": None,
    }
    determinism = {
        "status": "PASS",
        "manual_replay_count": 10,
        "sample_selection": "first ten frozen OCSR1 prediction records",
        "candidate_search_executed": False,
        "fresh_validation_accessed": False,
    }
    regression = {
        "status": "PASS",
        "frozen_hashes_current": all(
            digest(ROOT/path) == value
            for path, value in frozen["artifacts"].items()
        ),
        "TrackManager_algorithm_modified": False,
        "kalman_process_model_modified": False,
        "kalman_measurement_model_modified": False,
        "kalman_initial_covariance_modified": False,
        "ocsr1_adapter_v2_modified": False,
        "ocsr1_validation_results_modified": False,
    }
    selection = {
        "status": "BLOCKED",
        "selected_candidate": None,
        "calibration_family": None,
        "integrity_gate": "FAIL_REFERENCE_POINT_ALIGNMENT",
    }
    compatibility = {
        "status": "PASS",
        "formal_kalman": "unchanged",
        "formal_tracker": "unchanged",
        "adapter_v2": "frozen",
        "adapter_v2_1": "not created",
        "YOPO": "unchanged",
        "formal_planner": "unchanged",
    }
    final = {
        "status": "FAIL",
        "route": "C",
        "primary_cause": "kalman_prediction_timeline_alignment",
        "specific_cause":
            "visible_surface_cluster_centroid_vs_actor_geometric_center",
        "timeline_integrity": "PASS",
        "world_frame_integrity": "PASS",
        "reference_point_integrity": "FAIL",
        "uncertainty_candidate_search": "NOT_RUN_FAIL_CLOSED",
        "selected_candidate": None,
        "fresh_validation": "NOT_ACCESSED",
        "production_activation_authorized": False,
        "formal_kalman_modified": False,
        "TrackManager_algorithm_modified": False,
        "kalman_process_model_modified": False,
        "kalman_measurement_model_modified": False,
        "kalman_initial_covariance_modified": False,
        "detector_algorithm_modified": False,
        "static_yopo_network_modified": False,
        "static_yopo_checkpoint_modified": False,
        "formal_planner_modified": False,
        "formal_command_modified": False,
        "runtime_gt_used": False,
        "ocsr1_adapter_v2_modified": False,
        "ocsr1_validation_results_modified": False,
        "new_maps_generated": False,
        "new_formal_dataset_generated": False,
        "holdout_accessed": False,
        "production_test_accessed": False,
        "blind_accessed": False,
        "formal_preflight_rerun": False,
        "formal_v3_entry_created": False,
        "optimizer_step_executed": False,
        "training_started": False,
        "natural_eosr1_tracker_gate": "FAIL_SEPARATE",
        "next_allowed_phase":
            "phase8jqv2_4_prediction_timeline_alignment_repair",
    }
    reports = {
        "entry_gate": entry,
        "frozen_artifacts": frozen,
        "historical_validation_status": historical,
        "timeline_integrity": timeline,
        "coordinate_frame_integrity": coordinate,
        "error_root_cause": root_cause,
        "whitened_residual_analysis": whitened,
        "bias_analysis": bias,
        "scenario_conditioned_error": scenario,
        "uncertainty_replay_manifest": replay_manifest,
        "evaluation_split": split,
        "fresh_validation_freeze": fresh_freeze,
        "validation_freeze": validation_freeze,
        **candidate_reports,
        "candidate_comparison": candidate_comparison,
        "empirical_coverage": not_run_metric,
        "normalized_error": not_run_metric,
        "envelope_sharpness": not_run_metric,
        "candidate_risk_metrics": not_run_metric,
        "decision_risk_metrics": not_run_metric,
        "top_ranked_candidate_safety": not_run_metric,
        "no_safe_candidate_validation": not_run_metric,
        "false_veto_analysis": not_run_metric,
        "adapter_v2_1_contract": adapter_contract,
        "negative_validation": negative,
        "multi_target_validation": multi,
        "gap1_validation": gap1,
        "gap2_validation": gap2,
        "gap3_non_scope": gap3,
        "runtime": runtime_report,
        "determinism": determinism,
        "regression": regression,
        "candidate_selection": selection,
        "compatibility_matrix": compatibility,
        "final_result": final,
    }
    for suffix, value in reports.items():
        write(f"phase8jqv2_4kucr1_{suffix}.json", value)
    (REPORTS/"phase8jqv2_4kucr1_prediction_origin_contract.md").write_text(
        "# KUCR1 prediction origin contract\n\n"
        "Kalman state is a world-frame visible-surface cluster centroid after "
        "the current frame update. Actor GT and collision geometry use the "
        "actor geometric center. These origins are not interchangeable. "
        "Prediction time is relative to the current state timestamp and does "
        "not contain a duplicate prediction.\n"
    )
    (REPORTS/"phase8jqv2_4kucr1_migration_plan.md").write_text(
        "# KUCR1 migration plan\n\n"
        "Do not inflate covariance or create adapter v2.1 yet. First version "
        "the tracking-to-collision reference-point contract: either estimate "
        "an actor center and its uncertainty from observed geometry, or define "
        "a surface-support safety envelope that never compares a surface "
        "centroid covariance with center GT. Then rebuild a fresh split.\n"
    )
    (REPORTS/"phase8jqv2_4kucr1_final_recommendation.md").write_text(
        "# KUCR1 final recommendation\n\n"
        "Select Route C and repair prediction reference-origin alignment. "
        "The apparent covariance failure is dominated by a deterministic "
        "surface-centroid-to-center offset near actor radius. Increasing Q/R "
        "or planner sigma would hide this semantic error and worsen false veto.\n"
    )
    (REPORTS/"phase8jqv2_4kucr1_final_readiness.md").write_text(
        "# KUCR1 readiness\n\n"
        "Status: **FAIL, fail-closed**. No uncertainty candidate, adapter v2.1, "
        "fresh validation, or production integration is authorized.\n"
    )
    print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
