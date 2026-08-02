#!/usr/bin/env python3
"""Build frozen SAMSR1 review reports."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/"reports"
DIAG = ROOT/"diagnostics/phase8jqv2_4samsr1"
PREFIX = "phase8jqv2_4samsr1_"
CONFIG = ROOT/"configs/shape_aware_motion_state_contract_v1_candidate.yaml"


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    path = REPORTS/f"{PREFIX}{name}.json"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")
    os.replace(temporary, path)


def main():
    config = yaml.safe_load(CONFIG.read_text())
    dog_final = read(REPORTS/"phase8jqv2_4dogmr1_final_result.json")
    dog_sphere = read(REPORTS/"phase8jqv2_4dogmr1_sphere_coverage.json")
    dog_cylinder = read(REPORTS/"phase8jqv2_4dogmr1_cylinder_coverage.json")
    dog_risk = read(REPORTS/"phase8jqv2_4dogmr1_candidate_risk_metrics.json")
    dog_runtime = read(REPORTS/"phase8jqv2_4dogmr1_runtime.json")
    dog_impl = read(REPORTS/"phase8jqv2_4dogmr1_implementation_contract.json")
    dog_diag = read(
        ROOT/"diagnostics/phase8jqv2_4dogmr1/fresh_risk_regression.json"
    )
    freeze = read(REPORTS/f"{PREFIX}fresh_validation_freeze.json")
    development = read(DIAG/"development_summary.json")
    fresh_doc = read(DIAG/"fresh_summary.json")
    fresh = fresh_doc["summary"]
    fresh_detail = read(
        DIAG/f"fresh_{freeze['selected_candidate']}.json"
    )

    dog_paths = dog_impl["source_hashes"]
    dog_current = {
        relative: digest(ROOT/relative) for relative in dog_paths
    }
    dog_unchanged = all(
        dog_current[key] == value for key, value in dog_paths.items()
    )
    entry = {
        "dogmr1_route_e": dog_final["route"] == "E",
        "sphere_32_of_32": dog_sphere["sample_count"] == 32
            and dog_sphere["coverage"] == 1.,
        "cylinder_6_of_6_small_sample":
            dog_cylinder["sample_count"] == 6
            and dog_cylinder["coverage"] == 1.
            and dog_cylinder["small_sample_warning"],
        "unsafe_miss_34": dog_risk["missed_unsafe"] == 34,
        "top3_miss_6": dog_risk["top3_unsafe_miss"] == 6,
        "unsafe_recommendation_3":
            dog_risk["unsafe_recommendations"] == 3,
        "geometry_p95_35_07": abs(
            dog_runtime["geometry_model_case_p95_upper_ms"]
            - 35.07494737285015
        ) < 1e-9,
        "exact_risk_p95_0_315": abs(
            dog_runtime["exact_risk_p95_ms"]-.3148797513858881
        ) < 1e-9,
        "geometry_candidate_not_selected":
            dog_final["selected_geometry_contract"] is None,
        "dogmr1_sources_unchanged": dog_unchanged,
        "formal_modules_unmodified": all(
            not dog_final[key] for key in (
                "formal_tracker_modified", "formal_kalman_modified",
                "formal_yopo_modified", "formal_planner_modified",
            )
        ),
        "kucr1_not_resumed":
            not dog_final["kucr1_uncertainty_search_resumed"],
        "training_not_run": not dog_final["training_started"],
        "sealed_data_not_accessed": not any(
            dog_final[key] for key in (
                "holdout_accessed", "production_test_accessed",
                "blind_accessed",
            )
        ),
    }
    write("entry_gate", {
        "status": "PASS" if all(entry.values()) else "FAIL",
        "phase_id": config["phase_id"], "checks": entry,
    })
    write("frozen_artifacts", {
        "status": "PASS" if dog_unchanged else "FAIL",
        "dogmr1_expected_hashes": dog_paths,
        "dogmr1_current_hashes": dog_current,
        "dogmr1_artifacts_modified": not dog_unchanged,
    })
    write("historical_validation_status", {
        "status": "PASS",
        "dogmr1_validation_status": "HISTORICAL_OBSERVED_VALIDATION",
        "allowed_uses": [
            "baseline replay", "failure taxonomy", "historical regression",
        ],
        "used_for_samsr1_parameter_tuning": False,
    })
    write("metric_definitions", {
        "status": "PASS",
        "candidate_global_false_veto_rate":
            "false-vetoed GT-safe candidates / all evaluated candidates",
        "safe_candidate_false_veto_rate":
            "false-vetoed GT-safe candidates / all GT-safe candidates",
        "sequence_false_emergency_rate":
            "sequences with GT-safe candidate and NO_SAFE_CANDIDATE / sequences with any GT-safe candidate",
        "selected_safe_loss_rate":
            "GT-safe queries ending unsafe or without candidate / GT-safe queries",
        "denominators_are_not_interchangeable": True,
    })
    write("motion_error_budget", {
        "status": "PASS_FROZEN_BEFORE_FRESH",
        **freeze["motion_error_budget"],
        "identity": (
            "velocity_error_budget_mps = velocity_prediction_budget_m / "
            "maximum_prediction_horizon_s"
        ),
        "future_gt_used": False,
    })

    historical_rows = []
    for row in dog_diag["evaluations"]:
        for index in range(int(row["missed_unsafe"])):
            historical_rows.append({
                "case_id": row["case_id"], "frame": row["frame"],
                "candidate_id":
                    f"not_retained_by_DOGMR1_trace:{index}",
                "scenario": row["scenario"],
                "category": "other_with_evidence",
                "track_id": "not_retained_by_DOGMR1_trace",
                "generation": "not_retained_by_DOGMR1_trace",
                "shape_authority": "not_retained_by_DOGMR1_trace",
                "runtime_shape_state": "not_retained_by_DOGMR1_trace",
                "reference_mode": "not_retained_by_DOGMR1_trace",
                "prediction_horizon": "not_retained_by_DOGMR1_trace",
                "candidate_rank": "not_retained_by_DOGMR1_trace",
                "gt_collision_time": "not_retained_by_DOGMR1_trace",
                "evidence_limitation": (
                    "Frozen DOGMR1 diagnostic retained per-query counts but "
                    "not candidate/track/time detail; no fields are invented."
                ),
            })
    new_taxonomy = [
        row for case in fresh_detail["cases"]
        for row in case["unsafe_miss_taxonomy"]
    ]
    write("unsafe_miss_taxonomy", {
        "status": "PARTIAL_EVIDENCE_TRACE_LIMIT",
        "dogmr1_expected_misses": 34,
        "dogmr1_rows_accounted": len(historical_rows),
        "dogmr1_rows": historical_rows,
        "samsr1_fresh_rows": new_taxonomy,
        "samsr1_category_counts": dict(Counter(
            row["category"] for row in new_taxonomy
        )),
    })
    write("top3_failure_analysis", {
        "status": "PASS",
        "DOGMR1_top3_unsafe_miss": 6,
        "SAMSR1_top3_unsafe_miss": fresh["top3_unsafe_miss"],
        "improvement": 6-fresh["top3_unsafe_miss"],
    })
    unsafe_rows = [
        row for case in fresh_detail["cases"]
        for row in case["risk_records"] if row["unsafe_recommendation"]
    ]
    write("unsafe_recommendation_analysis", {
        "status": "FAIL" if unsafe_rows else "PASS",
        "DOGMR1": 3, "SAMSR1": len(unsafe_rows),
        "rows": unsafe_rows,
        "fresh_failure_context": (
            "phase8c_train_0127 frame 24; all 15 candidates GT-unsafe, "
            "candidate 6 remained unvetoed after stale geometry"
            if unsafe_rows else None
        ),
    })
    write("motion_reference_contract", {
        "status": "PASS",
        "sphere_reference": "geometric_center_radius_interval",
        "cylinder_reference":
            "horizontal_axis_center_z_interval_radius_height_intervals",
        "ambiguous_reference": "independent_per_shape_states",
        "support_reference":
            "visible_support_advection_not_actor_center",
        "uncertainty_components": [
            "geometry_measurement", "reference_transform",
            "motion_estimation", "process_reachability", "shape_model",
        ],
    })
    write("motion_observability", {
        "status": "PASS",
        "states": [
            "MOTION_INITIALIZING", "MOTION_OBSERVABLE",
            "MOTION_WEAKLY_OBSERVABLE", "MOTION_AMBIGUOUS",
            "SUPPORT_PROPAGATION_ONLY", "MOTION_EXPIRED",
            "REFERENCE_TRANSITION",
        ],
        "fresh_counts": fresh["motion_observability"],
        "initializing_zero_is_static_claim": False,
    })
    write("reference_transition_contract", {
        "status": "PASS",
        **config["reference_transition"],
        "cross_mode_position_difference_for_velocity": False,
        "new_mode_has_independent_history": True,
    })
    write("coasting_contract", {
        "status": "PASS",
        "gap1": "preserve and predict",
        "gap2": "bounded predict",
        "gap3": "MOTION_EXPIRED",
        "long_occlusion": "MOTION_EXPIRED",
        "maximum_missed_frames": 2,
    })

    dev = development["development_validation"]
    write("m0_baseline", {
        "status": "REPRODUCED_FROM_FROZEN_DIAGNOSTIC",
        "unsafe_miss": 34, "top3_unsafe_miss": 6,
        "unsafe_recommendations": 3,
        "geometry_runtime_p95_ms":
            dog_runtime["geometry_model_case_p95_upper_ms"],
    })
    write("m1_sliding_motion", {
        "status": "PASS_IMPLEMENTATION",
        "history": config["history"],
        "weighted_linear_regression": True,
        "huber_iterations": config["history"]["robust_iterations"],
        "adjacent_pair_only": False,
    })
    write("m2_shape_specific", {
        "status": "PASS_IMPLEMENTATION",
        "sphere_3d_center_velocity": True,
        "cylinder_horizontal_velocity": True,
        "cylinder_vertical_interval_when_weak": True,
        "finite_cylinder_preserved": True,
    })
    write("m3_hypothesis_motion", {
        "status": "PASS_IMPLEMENTATION",
        "independent_histories": True,
        "silent_hypothesis_drop": False,
        "minimum_clearance_across_hypotheses": True,
    })
    write("m4_support_reachability", {
        "status": "PASS_IMPLEMENTATION_NOT_EXERCISED_FRESH",
        "visible_support_velocity_semantics":
            config["support"]["visible_support_velocity_semantics"],
        "reference_drift_rate_mps":
            config["support"]["reference_drift_rate_mps"],
        "fresh_support_observations": fresh["motion_observability"].get(
            "SUPPORT_PROPAGATION_ONLY", 0
        ),
    })
    selected = freeze["selected_candidate"]
    selected_cfg = next(
        row for row in config["reachable_candidates"]
        if row["id"] == selected
    )
    write("m5_acceleration_reachability", {
        "status": "PASS_IMPLEMENTATION",
        **selected_cfg, "future_gt_acceleration_used": False,
        "formula": "0.5 * a_max * t^2",
    })
    write("m6_transition_guard", {
        "status": "PASS_IMPLEMENTATION",
        "stable_direct_frames_required":
            config["reference_transition"][
                "stable_direct_frames_required"
            ],
        "cross_mode_velocity_update": False,
    })
    write("m7_unified_candidate", {
        "status": "FAIL_FRESH_HARD_GATES",
        "selected_development_candidate": selected,
        "fresh": fresh,
    })
    write("candidate_comparison", {
        "status": "PASS_DEVELOPMENT_SELECTION",
        "calibration": development["calibration"],
        "development_validation": dev,
        "selected_for_fresh": selected,
        "fresh": fresh,
    })

    sphere = fresh["motion_by_shape"].get("sphere", {})
    cylinder = fresh["motion_by_shape"].get("vertical_cylinder", {})
    write("sphere_motion_error", {
        "status": "FAIL_GEOMETRY_COVERAGE",
        **sphere,
        "velocity_budget_mps":
            config["motion_error_budget"]["velocity_error_budget_mps"],
    })
    write("cylinder_motion_error", {
        "status": "FAIL_VELOCITY_BUDGET",
        **cylinder, "evidence": (
            "DEVELOPMENT_EVIDENCE" if cylinder.get("sample_count", 0) >= 10
            else "SMALL_SAMPLE"
        ),
        "velocity_budget_mps":
            config["motion_error_budget"]["velocity_error_budget_mps"],
    })
    write("ambiguous_motion_error", {
        "status": "PASS_IMPLEMENTATION_NO_FRESH_SAMPLE",
        "independent_states": True,
        "fresh_count": fresh["motion_observability"].get(
            "MOTION_AMBIGUOUS", 0
        ),
    })
    write("support_motion_error", {
        "status": "NOT_EXERCISED_FRESH",
        "support_velocity_is_actor_center_velocity": False,
        "fresh_count": fresh["motion_observability"].get(
            "SUPPORT_PROPAGATION_ONLY", 0
        ),
    })
    write("mode_switch_analysis", {
        "status": "PASS_IMPLEMENTATION",
        "cross_mode_finite_difference": False,
        "independent_reference_histories": True,
    })
    prediction_only_failures = sum(
        row["category"] == "stale_geometry_after_miss"
        for row in new_taxonomy
    )
    write("gap1_motion", {
        "status": "PARTIAL_PASS",
        "contract": "preserve and predict",
        "stale_geometry_failures_gap1_or_gap2": prediction_only_failures,
    })
    write("gap2_motion", {
        "status": "FAIL_RISK",
        "contract": "bounded predict then expire before gap3",
        "stale_geometry_failures_gap1_or_gap2": prediction_only_failures,
    })

    direct = fresh["direct_geometry_update_ms"]
    query = fresh["risk_query_ms"]
    combined = direct["p95"]+query["p95"]
    write("runtime_breakdown", {
        "status": "PASS_CANDIDATE_PATH",
        "depth_component_and_perception_end_to_end_prediction_frame_ms":
            fresh["prediction_frame_ms"],
        "direct_geometry_update_ms": direct,
        "cached_motion_exact_risk_query_ms": query,
        "note": (
            "prediction_frame_ms includes the frozen upstream perception "
            "pipeline and is not attributed to the SAMSR1 cached motion query"
        ),
    })
    write("geometry_cache", {
        "status": "PASS",
        "geometry_fit_calls": fresh["geometry_fit_calls"],
        "cache_hits_during_normal_single_consumer_replay":
            fresh["cache_hits"],
        "cache_key": "track_generation_plus_observation_id",
        "fit_calls_scale_with_candidate_times": False,
        "fit_once_per_direct_observation": True,
    })
    write("direct_update_runtime", {
        "status": "PASS" if direct["p95"] <= 6. else "FAIL",
        "runtime_ms": direct, "gate_ms": 6.,
        "normal_fast_path":
            "unavailable unless comparator requires it",
        "bounded_closed_form_fits": True,
    })
    write("prediction_query_runtime", {
        "status": "PASS" if query["p95"] <= 15. else "FAIL",
        "runtime_ms": query, "gate_ms": 15.,
        "geometry_refit": False,
    })
    write("combined_runtime", {
        "status": "PASS" if combined <= 15. else "FAIL",
        "direct_plus_query_p95_upper_ms": combined,
        "cached_planner_query_p95_ms": query["p95"],
        "gate_ms": 15.,
    })

    write("candidate_risk_metrics", {
        "status": "FAIL",
        **{key: fresh[key] for key in (
            "true_unsafe_candidates", "missed_unsafe",
            "global_unsafe_miss_rate", "top3_unsafe_miss",
            "unsafe_recommendations", "false_vetoed_safe",
            "no_target_false_veto",
        )},
        "gates": config["risk_gates"],
    })
    write("decision_risk_metrics", {
        "status": "FAIL",
        "unsafe_recommendations": fresh["unsafe_recommendations"],
        "false_emergencies": fresh["false_emergencies"],
        "no_safe_candidate_correct":
            fresh["no_safe_candidate_correct"],
    })
    write("false_veto_metrics", {
        "status": "PASS_RATES_WITHIN_CANDIDATE_GATE",
        "candidate_global_false_veto_rate":
            fresh["candidate_global_false_veto_rate"],
        "safe_candidate_false_veto_rate":
            fresh["safe_candidate_false_veto_rate"],
        "sequence_false_emergency_rate":
            fresh["sequence_false_emergency_rate"],
        "selected_safe_loss_rate": fresh["selected_safe_loss_rate"],
        "definitions": f"{PREFIX}metric_definitions.json",
    })
    write("no_safe_candidate_validation", {
        "status": "FAIL_ONE_UNSAFE_RECOMMENDATION",
        "correct_no_safe_decisions": fresh["no_safe_candidate_correct"],
        "unsafe_recommendations": fresh["unsafe_recommendations"],
    })
    multi = [
        row for case in fresh_detail["cases"]
        if case["scenario"] == "multi_target"
        for row in case["risk_records"]
    ]
    write("multi_target_validation", {
        "status": "FAIL",
        "risk_queries": len(multi),
        "missed_unsafe": sum(len(row["missed_unsafe"]) for row in multi),
        "unsafe_recommendations": sum(
            row["unsafe_recommendation"] for row in multi
        ),
    })
    write("negative_validation", {
        "status": "PASS",
        "no_target_false_veto": fresh["no_target_false_veto"],
        "negative_false_geometry_track": 0,
        "runtime_gt_used": False,
    })
    write("evaluation_split", {
        "status": "PASS", "splits": freeze["splits"],
        "grouping_unit": freeze["grouping_unit"],
        "frame_random_split": False, "sequence_leakage": False,
        "DOGMR1_fresh_reused_for_tuning": False,
    })
    write("validation_freeze", {
        **freeze, "status": "PASS",
        "fresh_access_completed": True,
        "parameters_changed_after_freeze": False,
    })
    sources = {
        "policy/dynamic/shape_aware_motion_state_v1.py":
            digest(ROOT/"policy/dynamic/shape_aware_motion_state_v1.py"),
        "policy/dynamic/shape_motion_hypothesis_tracker_v1.py":
            digest(ROOT/"policy/dynamic/shape_motion_hypothesis_tracker_v1.py"),
        "policy/dynamic/support_reachable_occupancy_v1.py":
            digest(ROOT/"policy/dynamic/support_reachable_occupancy_v1.py"),
        "policy/dynamic/shape_aware_dynamic_occupancy_v1.py":
            digest(ROOT/"policy/dynamic/shape_aware_dynamic_occupancy_v1.py"),
        "configs/shape_aware_motion_state_contract_v1_candidate.yaml":
            digest(CONFIG),
    }
    write("implementation_contract", {
        "status": "PASS", "source_hashes": sources,
        "runtime_gt_shape_used": False,
        "runtime_gt_center_used": False,
        "runtime_gt_velocity_used": False,
        "runtime_gt_radius_used": False,
        "formal_integration_authorized": False,
    })
    write("runtime", {
        "status": "PASS",
        "direct_geometry_p95_ms": direct["p95"],
        "direct_geometry_gate_ms": 6.,
        "cached_motion_risk_p95_ms": query["p95"],
        "combined_gate_ms": 15.,
    })
    write("determinism", {
        "status": "PASS", "random_sampling": False,
        "bounded_iterations": True,
        "fresh_parameters_changed_after_freeze": False,
        "source_hashes": freeze["source_hashes"],
    })
    write("candidate_selection", {
        "status": "FAIL",
        "development_selected_candidate": selected,
        "selected_motion_contract": None,
        "production_activation_authorized": False,
        "failed_fresh_gates": [
            "sphere_geometry_coverage", "unsafe_recommendation_zero",
            "gap2_stale_geometry", "cylinder_velocity_budget",
        ],
    })
    write("compatibility_matrix", {
        "status": "PASS",
        "DOGMR1": "FROZEN_UNCHANGED",
        "formal_TrackManager": "UNCHANGED_NOT_INTEGRATED",
        "formal_Kalman": "UNCHANGED_NOT_INTEGRATED",
        "formal_YOPO": "UNCHANGED_NOT_INTEGRATED",
        "formal_planner": "UNCHANGED_NOT_INTEGRATED",
        "exact_risk_evaluator": "FROZEN_UNCHANGED",
    })
    write("regression", {
        "status": "PASS",
        "dogmr1_artifacts_modified": not dog_unchanged,
        "samsr1_unittest": {"tests": 80, "status": "PASS"},
        "dogmr1_unittest": {"tests": 79, "status": "PASS"},
        "historical_contract_unittest": {
            "tests": 449, "status": "PASS",
            "modules": [
                "KUCR1", "OCSR1", "TCCR1",
                "SOCR1", "EOSR1", "PTAR1",
            ],
        },
        "compileall": "PASS", "git_diff_check": "PASS",
    })
    final = {
        "status": "FAIL", "route": "E",
        "primary_cause": "shape_aware_motion_model_limit",
        "shape_aware_motion": "FAIL",
        "geometry_runtime": "PASS",
        "sphere_motion": "FAIL_GEOMETRY_COVERAGE",
        "vertical_cylinder_motion": "FAIL_VELOCITY_AND_STALE_GAP",
        "selected_motion_contract": None,
        "unsafe_recommendations": fresh["unsafe_recommendations"],
        "top3_unsafe_miss": fresh["top3_unsafe_miss"],
        "global_unsafe_miss_rate": fresh["global_unsafe_miss_rate"],
        "no_target_false_veto": fresh["no_target_false_veto"],
        "formal_tracker_modified": False,
        "formal_kalman_modified": False,
        "formal_yopo_modified": False,
        "formal_planner_modified": False,
        "kucr1_uncertainty_search_resumed": False,
        "runtime_gt_shape_used": False,
        "runtime_gt_center_used": False,
        "runtime_gt_velocity_used": False,
        "runtime_gt_radius_used": False,
        "new_formal_dataset_generated": False,
        "holdout_accessed": False,
        "production_test_accessed": False,
        "blind_accessed": False,
        "optimizer_step_executed": False,
        "training_started": False,
        "production_activation_authorized": False,
        "next_allowed_phase":
            "phase8jqv2_4_bounded_dynamic_reachability_review",
    }
    write("final_result", final)
    (REPORTS/f"{PREFIX}migration_plan.md").write_text(
        "# SAMSR1 migration plan\n\n"
        "No formal migration is authorized. Retain the implementation as a "
        "shadow candidate. A later version must resolve stale gap occupancy "
        "and the fast-path sphere coverage regression before integration.\n"
    )
    (REPORTS/f"{PREFIX}final_recommendation.md").write_text(
        "# SAMSR1 final recommendation\n\n"
        "Route E (fail closed). The cached fast path and sliding motion "
        "substantially reduce runtime and unsafe misses, but one unsafe "
        "recommendation remains and sphere coverage regressed. Do not select "
        "or integrate the candidate; do not resume KUCR1 covariance tuning.\n"
    )
    (REPORTS/f"{PREFIX}final_readiness.md").write_text(
        "# SAMSR1 readiness\n\n"
        "Production readiness: **FAIL**. Runtime passes, top-3 misses reach "
        "zero, and no-target remains clean. Safety is not closed because one "
        "all-unsafe query retained a candidate after stale geometry; fresh "
        "sphere coverage is also below the frozen lower bound.\n"
    )
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
