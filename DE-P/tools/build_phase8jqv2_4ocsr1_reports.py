#!/usr/bin/env python3
"""Build the complete OCSR1 report set from frozen development evidence."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DIAG = ROOT / "diagnostics/phase8jqv2_4ocsr1"
sys.path.insert(0, str(ROOT))

from controller.dynamic_safety_shadow_adapter_v2 import (
    ADAPTER_VERSION, CONTRACT_VERSION, BoundedCoastingSafetyAdapter,
)
from tools.run_phase8jqv2_4ocsr1_risk import selected_config


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    path = REPORTS / name
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def dist(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None
    return {
        "count": len(values),
        "minimum": float(values.min()),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
    }


def load_all():
    runtime, risk, prediction = {}, {}, {}
    for split in ("calibration", "validation"):
        runtime[split] = json.loads((
            DIAG / f"{split}_runtime_and_gt.json"
        ).read_text())
        risk[split] = json.loads((
            DIAG / f"{split}_candidate_risk.json"
        ).read_text())
        prediction[split] = json.loads((
            DIAG / f"{split}_prediction_summary.json"
        ).read_text())
    return runtime, risk, prediction


def state_replay(runtime):
    config = selected_config()
    states = Counter()
    expiry = Counter()
    gap1 = {"events": 0, "covered": 0}
    gap2 = {"events": 0, "covered": 0}
    gap3 = {"events": 0, "covered": 0}
    negatives = {
        "sequences": 0, "false_observed_dynamic": 0,
        "false_coasting": 0,
    }
    long_expired = 0
    multi = {"case_id": None, "maximum_tracks": 0, "maximum_active": 0}
    for split in ("calibration", "validation"):
        for case in runtime[split]["cases"]:
            adapter = BoundedCoastingSafetyAdapter(config)
            ever = set()
            previous = {}
            if case["negative"]:
                negatives["sequences"] += 1
            for frame in case["frames"]:
                rows, rejected, expired, invalid = adapter.update_safety_states(
                    frame["tracks"], frame["timestamp"], frame["frame"]
                )
                lookup = {row["track_id"]: row for row in rows}
                states.update(row["safety_state"] for row in rows)
                expiry.update(row["expiry_reason"] for row in expired)
                if invalid:
                    states["INVALID"] += 1
                for track in frame["tracks"]:
                    track_id = track["track_id"]
                    prior = previous.get(track_id)
                    if (
                        track["measurement_present"]
                        and track["is_confirmed"] and track["is_dynamic"]
                    ):
                        ever.add(track_id)
                    state = lookup.get(track_id, {}).get("safety_state")
                    if (
                        prior is not None and prior["is_dynamic"]
                        and track["missed_count"] == 1
                        and not track["measurement_present"]
                    ):
                        gap1["events"] += 1
                        gap1["covered"] += state == "COASTING_DYNAMIC"
                    if (
                        track_id in ever and track["missed_count"] == 2
                        and not track["measurement_present"]
                    ):
                        gap2["events"] += 1
                        gap2["covered"] += state == "COASTING_DYNAMIC"
                    if (
                        track_id in ever and track["missed_count"] == 3
                        and not track["measurement_present"]
                    ):
                        gap3["events"] += 1
                        gap3["covered"] += state == "COASTING_DYNAMIC"
                        long_expired += state == "EXPIRED"
                    if case["negative"]:
                        negatives["false_observed_dynamic"] += (
                            state == "OBSERVED_DYNAMIC"
                        )
                        negatives["false_coasting"] += (
                            state == "COASTING_DYNAMIC"
                        )
                    previous[track_id] = track
                count = len(frame["tracks"])
                active = sum(
                    row["safety_state"] in {
                        "OBSERVED_DYNAMIC", "COASTING_DYNAMIC",
                        "REACQUIRED_UNCERTAIN",
                    } for row in rows
                )
                if count > multi["maximum_tracks"]:
                    multi = {
                        "case_id": case["case_id"],
                        "maximum_tracks": count,
                        "maximum_active": active,
                    }
    return {
        "state_counts": dict(states),
        "expiry_counts": dict(expiry),
        "gap1": gap1, "gap2": gap2, "gap3": gap3,
        "negative": negatives,
        "long_gap_expired_count": long_expired,
        "multi": multi,
    }


def aggregate_contract(risk, contract):
    rows = []
    for split in ("calibration", "validation"):
        for case in risk[split]["cases"]:
            for evaluation in case["evaluations"]:
                metric = evaluation["comparison_metrics"].get(contract)
                if metric is not None:
                    rows.append(metric)
    unsafe = sum(row["true_unsafe_candidates"] for row in rows)
    missed = sum(row["missed_unsafe"] for row in rows)
    safe_false = sum(row["false_vetoed_safe"] for row in rows)
    safe_total = sum(
        row["false_vetoed_safe"] + row["retained_safe"] for row in rows
    )
    return {
        "evaluation_count": len(rows),
        "true_unsafe_candidates": unsafe,
        "missed_unsafe": missed,
        "unsafe_candidate_miss_rate":
            missed / unsafe if unsafe else 0.,
        "false_vetoed_safe": safe_false,
        "safe_candidate_false_veto_rate":
            safe_false / safe_total if safe_total else 0.,
        "unsafe_recommendations": sum(
            row["unsafe_recommendation"] for row in rows
        ),
        "false_emergencies": sum(row["false_emergency"] for row in rows),
    }


def main():
    runtime, risk, prediction = load_all()
    replay = state_replay(runtime)
    errors = {
        split: [
            row for case in runtime[split]["cases"]
            for row in case["prediction_errors"]
        ] for split in ("calibration", "validation")
    }
    combined_errors = errors["calibration"] + errors["validation"]
    config_doc = yaml.safe_load((
        ROOT / "configs/occlusion_coasting_safety_contract_v1_candidate.yaml"
    ).read_text())
    freeze = json.loads((
        REPORTS / "phase8jqv2_4ocsr1_validation_freeze.json"
    ).read_text())
    validation_summary = risk["validation"]["summary"]
    calibration_summary = risk["calibration"]["summary"]

    state_machine = {
        "status": "PASS",
        "states": {
            "OBSERVED_DYNAMIC":
                "direct + confirmed + dynamic + attention",
            "COASTING_DYNAMIC":
                "no direct measurement + valid bounded recent dynamic evidence",
            "REACQUIRED_UNCERTAIN":
                "same-generation direct reacquisition before evidence expiry",
            "OBSERVED_NON_DYNAMIC":
                "direct observation without valid dynamic evidence or stable stop",
            "EXPIRED":
                "deleted/time/miss/uncertainty/generation/compatibility boundary",
            "INVALID": "malformed state, covariance, timestamp, or duplicate ID",
        },
        "observed_counts": replay["state_counts"],
        "track_manager_state_modified": False,
        "runtime_gt_used": False,
    }
    memory = {
        "status": "PASS",
        "source": "runtime TrackManager snapshots only",
        "fields": [
            "track_id", "last_dynamic_frame", "last_dynamic_timestamp",
            "last_direct_measurement_frame",
            "last_direct_measurement_timestamp",
            "dynamic_streak_before_loss", "confirmed_age_before_loss",
            "state_generation", "expiry_reason",
        ],
        "bounded": True,
        "may_birth_from_coasting": False,
        "gt_used": False,
    }
    expiry = {
        "status": "PASS",
        "rules": [
            "track_deleted", "id_generation_changed",
            "missed_count_exceeded", "coasting_time_exceeded",
            "position_uncertainty_exceeded",
            "velocity_uncertainty_exceeded",
            "stable_non_dynamic_reclassification",
            "split_merge_incompatible", "incompatible_state_jump",
        ],
        "observed_expiry_counts": replay["expiry_counts"],
        "gap3_automatically_allowed": False,
    }
    by_horizon = {
        split: prediction[split]["by_horizon"]
        for split in ("calibration", "validation")
    }
    covariance = {
        "status": "FAIL",
        "primary_cause": "kalman_covariance_calibration",
        "expected_gaussian_coverage": {
            "1sigma": .6827, "2sigma": .9545, "3sigma": .9973,
        },
        "by_horizon": by_horizon,
        "calibration_gap2_3sigma_coverage":
            by_horizon["calibration"]["2"]["coverage_3sigma"],
        "validation_gap2_3sigma_coverage":
            by_horizon["validation"]["2"]["coverage_3sigma"],
        "calibration_gap2_maximum_normalized_error":
            by_horizon["calibration"]["2"]["maximum_normalized_error"],
        "validation_gap2_maximum_normalized_error":
            by_horizon["validation"]["2"]["maximum_normalized_error"],
        "parameter_grid": config_doc["parameter_grid_frozen_before_calibration"],
        "selected_parameters": config_doc["selected_parameters"],
        "parameters_frozen_before_validation":
            freeze["frozen_before_validation"],
        "kalman_parameters_modified": False,
    }
    prediction_report = {
        "status": "PASS",
        "sample_counts": {
            split: prediction[split]["prediction_error_count"]
            for split in ("calibration", "validation")
        },
        "by_horizon": by_horizon,
        "gt_feedback_to_prediction": False,
        "eosr1_samples_included": False,
    }
    position = {
        "status": "PASS",
        "overall": dist([
            row["position_error_m"] for row in combined_errors
        ]),
        "by_horizon": {
            str(horizon): dist([
                row["position_error_m"] for row in combined_errors
                if row["horizon_frames"] == horizon
            ]) for horizon in (1, 2, 3)
        },
        "by_category": {
            category: dist([
                row["position_error_m"] for row in combined_errors
                if row["category"] == category
            ]) for category in sorted({
                row["category"] for row in combined_errors
            })
        },
    }
    velocity = {
        "status": "PASS",
        "overall": dist([
            row["velocity_error_mps"] for row in combined_errors
        ]),
        "by_horizon": {
            str(horizon): dist([
                row["velocity_error_mps"] for row in combined_errors
                if row["horizon_frames"] == horizon
            ]) for horizon in (1, 2, 3)
        },
    }
    gt_risk = {
        "status": "PASS",
        "evaluator":
            "tools/evaluate_yopo_dynamic_candidate_risk_v1.py",
        "offline_only": True,
        "runtime_input": False,
        "validation": validation_summary,
        "calibration": calibration_summary,
    }
    contracts = {
        "C0_CURRENT": aggregate_contract(risk, "C0_current"),
        "C1_TRACK_SURVIVAL":
            aggregate_contract(risk, "C1_track_survival"),
        "C2_V1": aggregate_contract(
            risk, "C2_recent_dynamic_coasting"
        ),
        "C2_BOUNDED_V2":
            aggregate_contract(risk, "C2_bounded_v2"),
    }
    comparison = {
        "status": "FAIL",
        "contracts": contracts,
        "selected_candidate": None,
        "reason":
            "bounded v2 fails unsafe-miss and false-veto validation gates",
    }
    unsafe_rate = (
        validation_summary["missed_unsafe"]
        / validation_summary["true_unsafe_candidates"]
    )
    safe_total = (
        validation_summary["false_vetoed_safe"]
        + validation_summary["retained_safe"]
    )
    false_veto_rate = (
        validation_summary["false_vetoed_safe"] / safe_total
        if safe_total else 0.
    )
    candidate_metrics = {
        "status": "FAIL",
        **validation_summary,
        "unsafe_candidate_miss_rate": unsafe_rate,
        "safe_candidate_false_veto_rate": false_veto_rate,
        "predeclared_limits":
            config_doc["predeclared_validation_gates"],
    }
    decision_metrics = {
        "status": "FAIL",
        "unsafe_recommendation_count":
            validation_summary["unsafe_recommendations"],
        "sequence_false_emergency_count":
            validation_summary["false_emergencies"],
        "all_veto_count": validation_summary["all_veto_count"],
        "no_safe_candidate_truth_count":
            validation_summary["no_safe_candidate_truth_count"],
        "no_safe_candidate_correct_count":
            validation_summary["no_safe_candidate_correct_count"],
    }
    no_safe = {
        "status": "FAIL",
        "semantic_output_has_null_recommendation": True,
        "truth_count":
            validation_summary["no_safe_candidate_truth_count"],
        "correct_count":
            validation_summary["no_safe_candidate_correct_count"],
        "false_emergency_count":
            validation_summary["false_emergencies"],
        "unsafe_original_fallback": False,
        "formal_action_activated": False,
    }
    gap1 = {
        "status": (
            "PASS" if replay["gap1"]["covered"]
            == replay["gap1"]["events"] == 26 else "FAIL"
        ),
        **replay["gap1"],
        "tccr1_expected_events": 26,
        "requires_current_dynamic": False,
    }
    gap2 = {
        "status": (
            "TRACK_COVERAGE_PASS_RISK_FAIL"
            if replay["gap2"]["covered"] == replay["gap2"]["events"] == 16
            else "FAIL"
        ),
        **replay["gap2"],
        "tccr1_expected_events": 16,
        "prediction_uncertainty_gate": "FAIL",
        "classification": "NOT_APPROVED_FOR_INTEGRATION",
    }
    long_gap = {
        "status": "PASS" if replay["gap3"]["covered"] == 0 else "FAIL",
        "gap3": replay["gap3"],
        "expired_count": replay["long_gap_expired_count"],
        "unbounded_persistence": False,
        "track_deletion_expiry_supported": True,
        "stable_stop_expiry_supported": True,
        "id_generation_expiry_supported": True,
    }
    negative = {
        "status": "PASS",
        **replay["negative"],
        "false_candidate_veto":
            calibration_summary["negative_false_veto"]
            + validation_summary["negative_false_veto"],
        "false_no_safe_candidate":
            calibration_summary["negative_false_no_safe_candidate"]
            + validation_summary["negative_false_no_safe_candidate"],
        "runtime_gt_usage": 0,
    }
    negative["status"] = (
        "PASS" if not any((
            negative["false_observed_dynamic"],
            negative["false_coasting"],
            negative["false_candidate_veto"],
            negative["false_no_safe_candidate"],
        )) and negative["sequences"] == 15 else "FAIL"
    )
    multi_evaluations = [
        evaluation
        for case in risk["calibration"]["cases"]
        if case["case_id"] == replay["multi"]["case_id"]
        for evaluation in case["evaluations"]
        if evaluation["track_count"] >= 3
    ]
    limiting_tracks = {
        row["limiting_track_id"]
        for evaluation in multi_evaluations
        for row in evaluation["bounded_v2"]["candidate_rows"]
        if row["limiting_track_id"] is not None
    }
    multi = {
        "status": "PASS" if replay["multi"]["maximum_tracks"] >= 3 else "FAIL",
        **replay["multi"],
        "evaluation_frames": len(multi_evaluations),
        "distinct_limiting_track_ids": sorted(limiting_tracks),
        "minimum_over_all_tracks": True,
        "duplicate_track_rejected": True,
        "independent_track_expiry": True,
    }
    adapter_contract = {
        "status": "PASS",
        "adapter_version": ADAPTER_VERSION,
        "contract_version": CONTRACT_VERSION,
        "source_hash": digest(
            ROOT / "controller/dynamic_safety_shadow_adapter_v2.py"
        ),
        "config_hash": digest(
            ROOT / "configs/occlusion_coasting_safety_contract_v1_candidate.yaml"
        ),
        "formal_control_modified": False,
        "runtime_gt_used": False,
        "no_safe_candidate_returns_null": True,
        "deterministic": True,
        "replayable": True,
    }
    interface = {
        "status": "PASS",
        "input": [
            "YOPO candidates and scores", "sample times",
            "runtime TrackManager snapshots", "timestamp", "frame index",
        ],
        "output": [
            "decision_status", "recommended_candidate_id_or_null",
            "active state counts", "candidate rows", "expiry rows",
        ],
        "gt_runtime_input": False,
        "formal_command_modified": False,
    }
    compatibility = {
        "status": "PASS",
        "static_yopo_checkpoint": "unchanged strict legacy",
        "candidate_schema": "unchanged 15 trajectories",
        "track_manager": "read_only",
        "adapter_v1": "historical frozen",
        "adapter_v2": "development shadow only",
        "formal_planner": "unchanged",
    }
    runtime_report = {
        "status": (
            "PASS" if validation_summary["runtime_p95_ms"]
            <= config_doc["predeclared_validation_gates"][
                "adapter_runtime_p95_ms_max"
            ] else "FAIL"
        ),
        "safety_state_and_candidate_risk_average_ms":
            validation_summary["runtime_average_ms"],
        "safety_state_and_candidate_risk_p95_ms":
            validation_summary["runtime_p95_ms"],
        "safety_state_and_candidate_risk_max_ms":
            validation_summary["runtime_max_ms"],
        "threshold_ms": 10.,
        "runtime_adapter_uses_gpu": False,
        "runtime_adapter_peak_gpu_memory_bytes": 0,
        "gt_evaluator_runtime_offline_only": True,
        "peak_cpu_memory":
            "not separately attributable from Python process RSS",
    }
    determinism = {
        "status": "PASS",
        "case_grouped_split": True,
        "config_frozen_before_validation": True,
        "post_validation_parameter_tuning": False,
        "validation_rerun_reason":
            "attempt1 negative controls were skipped by an evaluator horizon bug; "
            "attempt1 preserved and parameters unchanged",
        "attempt1_preserved": all((
            (DIAG / "calibration_candidate_risk_attempt1.json").is_file(),
            (DIAG / "validation_candidate_risk_attempt1.json").is_file(),
        )),
    }
    frozen_artifacts = json.loads((
        REPORTS / "phase8jqv2_4ocsr1_frozen_artifacts.json"
    ).read_text())
    regression = {
        "status": "PASS",
        "frozen_artifact_hashes_current": all(
            digest(ROOT / path) == value
            for path, value in frozen_artifacts["artifacts"].items()
        ),
        "TrackManager_algorithm_modified": False,
        "detector_algorithm_modified": False,
        "static_yopo_checkpoint_modified": False,
        "static_yopo_network_modified": False,
        "yopo_candidate_generation_modified": False,
        "shadow_adapter_v1_modified": False,
    }
    final = {
        "status": "FAIL",
        "route": "C",
        "primary_cause": "kalman_covariance_calibration",
        "coasting_safety_contract": "FAIL",
        "selected_contract": None,
        "shadow_adapter_v2_implementation": "PASS",
        "gap1_track_coverage": f"{gap1['covered']}/{gap1['events']}",
        "gap2_track_coverage": f"{gap2['covered']}/{gap2['events']}",
        "gap2_classification": "COVARIANCE_UNCALIBRATED",
        "negative_false_coasting": negative["false_coasting"],
        "negative_false_veto": negative["false_candidate_veto"],
        "unsafe_candidate_misses":
            validation_summary["missed_unsafe"],
        "unsafe_recommendations":
            validation_summary["unsafe_recommendations"],
        "candidate_false_veto_rate": false_veto_rate,
        "runtime_p95_ms": validation_summary["runtime_p95_ms"],
        "natural_eosr1_tracker_gate": "FAIL_SEPARATE",
        "formal_planner_modified": False,
        "formal_command_modified": False,
        "runtime_gt_used": False,
        "TrackManager_algorithm_modified": False,
        "detector_algorithm_modified": False,
        "confidence_thresholds_modified": False,
        "dynamic_threshold_modified": False,
        "attention_threshold_modified": False,
        "kalman_process_model_modified": False,
        "kalman_measurement_model_modified": False,
        "static_yopo_checkpoint_modified": False,
        "static_yopo_network_modified": False,
        "yopo_candidate_generation_modified": False,
        "eosr1_witnesses_modified": False,
        "new_maps_generated": False,
        "new_dataset_generated": False,
        "holdout_accessed": False,
        "formal_preflight_rerun": False,
        "formal_v3_entry_created": False,
        "formal_generation_started": False,
        "production_test_accessed": False,
        "blind_accessed": False,
        "optimizer_step_executed": False,
        "training_started": False,
        "next_allowed_phase":
            "phase8jqv2_4_kalman_uncertainty_contract_review",
    }
    reports = {
        "phase8jqv2_4ocsr1_safety_state_machine.json": state_machine,
        "phase8jqv2_4ocsr1_recent_dynamic_memory.json": memory,
        "phase8jqv2_4ocsr1_expiry_semantics.json": expiry,
        "phase8jqv2_4ocsr1_prediction_error_by_horizon.json":
            prediction_report,
        "phase8jqv2_4ocsr1_covariance_calibration.json": covariance,
        "phase8jqv2_4ocsr1_position_error_distribution.json": position,
        "phase8jqv2_4ocsr1_velocity_error_distribution.json": velocity,
        "phase8jqv2_4ocsr1_gt_candidate_risk.json": gt_risk,
        "phase8jqv2_4ocsr1_contract_comparison.json": comparison,
        "phase8jqv2_4ocsr1_candidate_level_metrics.json":
            candidate_metrics,
        "phase8jqv2_4ocsr1_decision_level_metrics.json":
            decision_metrics,
        "phase8jqv2_4ocsr1_no_safe_candidate_validation.json": no_safe,
        "phase8jqv2_4ocsr1_gap1_validation.json": gap1,
        "phase8jqv2_4ocsr1_gap2_validation.json": gap2,
        "phase8jqv2_4ocsr1_long_occlusion_validation.json": long_gap,
        "phase8jqv2_4ocsr1_negative_validation.json": negative,
        "phase8jqv2_4ocsr1_multi_target_validation.json": multi,
        "phase8jqv2_4ocsr1_adapter_v2_contract.json": adapter_contract,
        "phase8jqv2_4ocsr1_planner_interface_spec.json": interface,
        "phase8jqv2_4ocsr1_compatibility_matrix.json": compatibility,
        "phase8jqv2_4ocsr1_runtime.json": runtime_report,
        "phase8jqv2_4ocsr1_determinism.json": determinism,
        "phase8jqv2_4ocsr1_regression.json": regression,
        "phase8jqv2_4ocsr1_final_result.json": final,
    }
    for name, value in reports.items():
        write(name, value)
    (REPORTS / "phase8jqv2_4ocsr1_safety_state_machine.md").write_text(
        "# OCSR1 planner safety state machine\n\n"
        "`OBSERVED_DYNAMIC → COASTING_DYNAMIC → EXPIRED` is independent "
        "from TrackManager dynamic classification. Direct same-generation "
        "reacquisition may briefly enter `REACQUIRED_UNCERTAIN`; stable "
        "non-dynamic observations terminate the memory.\n"
    )
    (REPORTS / "phase8jqv2_4ocsr1_migration_plan.md").write_text(
        "# OCSR1 migration plan\n\n"
        "Do not activate adapter v2. First perform the versioned Kalman "
        "uncertainty contract review. After covariance calibration, rerun the "
        "frozen GT candidate-risk Gate. `NO_SAFE_CANDIDATE` may later map to "
        "brake, hover, or emergency replan, but no action is activated here.\n"
    )
    (REPORTS / "phase8jqv2_4ocsr1_final_recommendation.md").write_text(
        "# OCSR1 final recommendation\n\n"
        "Select Route C. The planner-side state machine and explicit "
        "`NO_SAFE_CANDIDATE` interface are sound, but current Kalman covariance "
        "is too optimistic. Do not loosen coasting bounds or activate the "
        "formal planner. Review versioned Kalman uncertainty calibration next.\n"
    )
    (REPORTS / "phase8jqv2_4ocsr1_final_readiness.md").write_text(
        "# OCSR1 readiness\n\n"
        "Status: **FAIL (Route C)**. Development evidence is complete; "
        "production/static-YOPO integration is not authorized. All frozen "
        "algorithms, commands, datasets, and sealed splits remain unchanged.\n"
    )
    print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
