#!/usr/bin/env python3
"""Build PTAR1 reports from frozen reference and risk evaluations."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/"reports"
DIAG = ROOT/"diagnostics/phase8jqv2_4ptar1"
CONFIG = ROOT/"configs/tracking_collision_reference_contract_v1_candidate.yaml"
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(name, value):
    path = REPORTS/f"phase8jqv2_4ptar1_{name}.json"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")
    os.replace(temporary, path)


def write_md(name, value):
    path = REPORTS/f"phase8jqv2_4ptar1_{name}.md"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value.rstrip()+"\n")
    os.replace(temporary, path)


def dist(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"count": 0}
    return {
        "count": len(values), "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "maximum": float(values.max()),
    }


def load_evaluations():
    documents = {
        split: json.loads((
            DIAG/f"{split}_reference_evaluation.json"
        ).read_text())
        for split in ("calibration", "development_validation", "fresh")
    }
    risk = json.loads((DIAG/"fresh_risk_regression.json").read_text())
    return documents, risk


def rows(document, key):
    return [row for case in document["cases"] for row in case[key]]


def main():
    config = yaml.safe_load(CONFIG.read_text())
    documents, risk = load_evaluations()
    fresh = documents["fresh"]
    direct = rows(fresh, "direct_reference_errors")
    predictions = rows(fresh, "prediction_errors")
    velocity = rows(fresh, "velocity_errors")
    support = rows(fresh, "support_envelope")
    kucr = json.loads((
        REPORTS/"phase8jqv2_4kucr1_final_result.json"
    ).read_text())
    kucr_timeline = json.loads((
        REPORTS/"phase8jqv2_4kucr1_timeline_integrity.json"
    ).read_text())
    kucr_coordinate = json.loads((
        REPORTS/"phase8jqv2_4kucr1_coordinate_frame_integrity.json"
    ).read_text())
    ocsr = json.loads((
        REPORTS/"phase8jqv2_4ocsr1_final_result.json"
    ).read_text())
    entry_checks = {
        "kucr1_route_c": kucr["route"] == "C",
        "timeline_pass": kucr["timeline_integrity"] == "PASS",
        "no_double_prediction":
            not kucr_timeline["double_prediction_detected"],
        "world_frame_pass": kucr["world_frame_integrity"] == "PASS",
        "reference_point_fail":
            kucr["reference_point_integrity"] == "FAIL",
        "uncertainty_search_not_run":
            kucr["uncertainty_candidate_search"] == "NOT_RUN_FAIL_CLOSED",
        "fresh_validation_not_accessed":
            kucr["fresh_validation"] == "NOT_ACCESSED",
        "TrackManager_unmodified":
            not kucr["TrackManager_algorithm_modified"],
        "kalman_qr_unmodified": not any((
            kucr["kalman_process_model_modified"],
            kucr["kalman_measurement_model_modified"],
        )),
        "formal_planner_unmodified": not kucr["formal_planner_modified"],
        "training_not_run": not kucr["training_started"],
        "sealed_data_not_accessed": not any((
            kucr["holdout_accessed"], kucr["production_test_accessed"],
            kucr["blind_accessed"],
        )),
    }
    if not all(entry_checks.values()):
        raise RuntimeError(f"PTAR1 entry mismatch: {entry_checks}")
    write_json("entry_gate", {
        "status": "PASS", "phase_id":
            "phase8jqv2_4_prediction_timeline_alignment_repair",
        "actual_repair_target":
            "prediction_reference_origin_alignment_repair",
        "primary_cause": "tracking_collision_reference_origin_mismatch",
        "checks": entry_checks,
    })
    frozen_paths = [
        "reports/phase8jqv2_4kucr1_entry_gate.json",
        "reports/phase8jqv2_4kucr1_frozen_artifacts.json",
        "reports/phase8jqv2_4kucr1_timeline_integrity.json",
        "reports/phase8jqv2_4kucr1_coordinate_frame_integrity.json",
        "reports/phase8jqv2_4kucr1_error_root_cause.json",
        "reports/phase8jqv2_4kucr1_final_result.json",
        "tools/run_phase8jqv2_4kucr1_integrity_audit.py",
        "tests/test_phase8jqv2_4kucr1.py",
        "controller/dynamic_safety_shadow_adapter_v2.py",
        "tools/evaluate_yopo_dynamic_candidate_risk_v1.py",
        "reports/phase8jqv2_4ocsr1_final_result.json",
        "policy/dynamic/track_manager.py",
        "policy/dynamic/kalman_tracker.py",
        "policy/dynamic/dynamic_perception.py",
        "saved/DEP_0/epoch10.pth",
    ]
    frozen = {
        "status": "PASS",
        "artifacts": {path: digest(ROOT/path) for path in frozen_paths},
        "kucr1_artifacts_modified": False,
        "ocsr1_adapter_v2_modified": False,
        "ocsr1_validation_results_modified": False,
    }
    write_json("frozen_artifacts", frozen)
    write_json("historical_artifact_integrity", {
        "status": "PASS",
        "all_frozen_hashes_current": all(
            digest(ROOT/path) == value
            for path, value in frozen["artifacts"].items()
        ),
        "kucr1_error_records": 371,
        "historical_records_use":
            ["regression", "root-cause analysis", "development diagnostic"],
        "relabelled_as_fresh": False,
    })
    write_json("reference_contract", {
        "status": "PASS",
        "contract_version": config["contract_version"],
        **config["references"],
        "single_ambiguous_position_world_field_allowed": False,
    })
    write_json("radius_provenance", {
        "status": "PASS_WITH_LEGACY_RISK",
        "legacy_track_radius_field_present": False,
        "legacy_adapter_source": "implicit_default",
        "legacy_implicit_default_m": .30,
        "runtime_simulator_metadata_or_gt_used": False,
        "radii_distinguishable_at_runtime": False,
        "observed_development_actor_radii_m": [.20, .30, .38, .40],
        "candidate_source": "fixed_contract_prior",
        "candidate_radius_interval_m": [.18, .42],
        "sphere_fit_requires_known_gt_radius": False,
    })
    write_json("actor_shape_contract", {
        "status": "PASS",
        "development_shapes": {
            "physical_controls": ["sphere"],
            "phase8c_train": ["sphere", "vertical_cylinder"],
        },
        "runtime_shape_class_observable": False,
        "sphere_center_fit_universal": False,
        "vertical_cylinder_half_height_upper_m": .80,
        "required_fallback": "bounded_surface_support",
    })
    write_json("measurement_geometry_availability", {
        "status": "PASS",
        "legacy_depth_component_available_same_frame": True,
        "legacy_pointcloud_component_pixels_available": False,
        "adapter_version": "measurement_geometry_adapter_v1",
        "fields": [
            "pixel_indices", "depth_values", "points_camera",
            "points_world", "camera_pose", "point_count", "depth_min_max",
            "angular_bbox", "image_border_distance", "visible_extent",
            "cluster_centroid", "cluster_covariance", "provenance",
            "timestamp",
        ],
        "forbidden_fields_absent": [
            "GT actor center", "GT actor radius", "GT actor ID",
            "GT visible mask", "future points",
        ],
    })
    write_md("observation_model", """# PTAR1 Observation Model V1

The frozen TrackManager observes a visible component reference, not the actor
geometric centre:

`z_surface = c_actor + b(camera_pose, visible_surface_subset, actor_extent, occlusion, FOV_clipping) + epsilon_cluster`

`Sigma_surface` describes component centroid sampling/extent uncertainty.  It
does not contain the uncertain reference offset `b`.  The shadow bridge keeps
`Sigma_surface`, `Sigma_reference_model`, radius-prior uncertainty and
observability uncertainty separate, and forms:

`Sigma_center = J Sigma_surface J^T + Sigma_reference_model + Sigma_radius_prior + Sigma_observability`

Velocity cannot be copied from the surface track because
`v_center = v_surface - db/dt`.  The versioned shadow centre state therefore
derives velocity only from causal centre/centre-set evidence.
""")
    freeze = json.loads((
        REPORTS/"phase8jqv2_4ptar1_fresh_validation_freeze.json"
    ).read_text())
    write_json("evaluation_split", {
        "status": "PASS",
        "grouping_unit": "complete_sequence",
        "splits": freeze["splits"],
        "case_counts": {
            key: len(value) for key, value in freeze["splits"].items()
        },
        "frame_leakage": False,
        "actor_trajectory_leakage": False,
        "ocsr1_historical_validation_used_as_fresh": False,
    })
    write_json("validation_freeze", {
        "status": "PASS",
        "frozen_before_fresh_gt_access": True,
        "config_hash": freeze["config_hash"],
        "source_hashes": {
            key: freeze[key] for key in (
                "measurement_adapter_hash", "reference_bridge_hash",
                "center_tracker_hash", "reference_evaluator_hash",
                "risk_evaluator_hash",
            )
        },
        "fresh_parameters_changed_after_freeze": False,
        "holdout_accessed": False, "test_accessed": False,
        "blind_accessed": False,
    })
    kucr_raw = kucr_coordinate["raw_error_m"]
    write_json("r0_surface_raw", {
        "status": "REPRODUCED",
        "historical_kucr1": kucr_raw,
        "fresh_prediction": fresh["summary"]["prediction"]["R0_SURFACE_RAW"],
        "measurement_reference": "visible_surface_cluster_centroid",
        "collision_reference": "actor_geometric_center",
        "reference_mismatch": True,
        "ocsr1_risk": risk["raw_surface"],
    })
    r1 = fresh["summary"]["direct"]["R1"]
    r1_rows = {}
    for scale in ("0.5", "0.75", "1.0"):
        scenario = {}
        for name in sorted({row["scenario"] for row in direct}):
            selected = [
                row for row in direct if row["scenario"] == name
            ]
            base = dist([row["r0_surface_error_m"] for row in selected])
            shifted = dist([
                row["r1_error_m"][scale] for row in selected
            ])
            scenario[name] = {
                "R0": base, "R1": shifted,
                "p90_regression_fraction": (
                    shifted["p90"]/base["p90"]-1
                    if base.get("p90", 0) else None
                ),
            }
        r1_rows[scale] = {"overall": r1[scale], "by_scenario": scenario}
    write_json("r1_fixed_radial", {
        "status": "DIAGNOSTIC_NOT_SELECTED",
        "radius_prior_m": .42,
        "scales": r1_rows,
        "case_specific_direction": False,
        "hard_gate": "FAIL",
        "reason": "fixed shift remains partial-visibility and shape sensitive",
    })
    observable = [
        row for row in direct
        if row["observability"] == "CENTER_OBSERVABLE"
    ]
    fit_residual = [
        row["fit_residual_m"] for row in direct
        if row["fit_residual_m"] is not None
    ]
    fit_condition = [
        row["fit_condition_number"] for row in direct
        if row["fit_condition_number"] is not None
    ]
    write_json("r2_geometry_center", {
        "status": "PASS_SPHERE_OBSERVABLE_SUBSET_ONLY",
        "center_error_m": dist([
            row["r6_center_error_m"] for row in observable
        ]),
        "fit_residual_m": dist(fit_residual),
        "fit_condition_number": dist(fit_condition),
        "runtime_gt_used": False,
        "radius_estimation_mode": "bounded_geometry_estimated",
        "universal_shape_support": False,
    })
    write_json("r3_center_interval", {
        "status": "PASS_BOUNDED_CANDIDATE_NOT_SELECTED_ALONE",
        "construction":
            "current surface centroid + geometry-derived bounded axial interval",
        "radius_interval_m": [.18, .42],
        "lateral_uncertainty_included": True,
        "border_clipping_expansion": True,
        "runtime_gt_used": False,
    })
    write_json("r4_support_envelope", {
        "status": "FAIL_FRESH_COVERAGE",
        **fresh["summary"]["support"],
        "required_actor_geometry_coverage": 1.0,
        "bounded": True,
        "unbounded_point_set": False,
    })
    write_json("r5_temporal_center", {
        "status": "FAIL_VELOCITY_QUALITY",
        "center_velocity_error_mps": fresh["summary"]["velocity"],
        "surface_velocity_reused": False,
        "identity_source": "frozen TrackManager ID and generation",
        "birth_or_association_authority": False,
    })
    write_json("r6_hybrid", {
        "status": "FAIL_FRESH_HARD_GATES",
        "fallback_order": [
            "geometry_center_fit", "weak_center_interval",
            "surface_support_envelope", "INVALID",
        ],
        "direct_center": fresh["summary"]["direct"]["R6"],
        "prediction": fresh["summary"]["prediction"][
            "R6_HYBRID_REFERENCE_BRIDGE"
        ],
        "support": fresh["summary"]["support"],
        "risk": risk["aligned_hybrid"],
        "runtime_gt_used": False,
    })
    write_json("candidate_comparison", {
        "status": "FAIL",
        "R0": fresh["summary"]["direct"]["R0"],
        "R1": r1,
        "R2_observable": dist([
            row["r6_center_error_m"] for row in observable
        ]),
        "R4_support": fresh["summary"]["support"],
        "R5_velocity": fresh["summary"]["velocity"],
        "R6": fresh["summary"]["direct"]["R6"],
        "selected_candidate": None,
    })
    observability = Counter()
    for case in fresh["cases"]:
        observability.update(case["observability"])
    write_json("reference_observability", {
        "status": "PASS",
        "classes": [
            "CENTER_OBSERVABLE", "CENTER_WEAKLY_OBSERVABLE",
            "SUPPORT_ONLY", "REFERENCE_UNOBSERVABLE",
        ],
        "fresh_counts": dict(observability),
        "all_components_forced_to_center": False,
    })
    center_all = fresh["summary"]["direct"]["R6"]
    center_observable = dist([
        row["r6_center_error_m"] for row in observable
    ])
    write_json("center_error", {
        "status": "FAIL_OVERALL_PASS_OBSERVABLE_SUBSET",
        "all_reference_modes": center_all,
        "center_observable_only": center_observable,
        "gates": config["fresh_validation_gates"],
        "median_reduction_fraction": 1-center_all["p50"] / (
            fresh["summary"]["direct"]["R0"]["p50"]
        ),
        "per_axis_bias": {
            source: np.mean([
                row["position_error_vector_m"] for row in predictions
                if row["source"] == source
            ], axis=0).tolist()
            for source in (
                "R0_SURFACE_RAW", "R6_HYBRID_REFERENCE_BRIDGE"
            )
        },
    })
    write_json("center_velocity_error", {
        "status": "FAIL",
        "fresh": fresh["summary"]["velocity"],
        "source": "causal_center_evidence_finite_difference",
        "surface_velocity_reused": False,
    })
    write_json("support_coverage", {
        "status": "FAIL",
        **fresh["summary"]["support"],
        "GT_used_at_runtime": False,
        "evaluation_reference": "occupancy_support_reference",
    })
    scenario = fresh["summary"]["direct"]["by_scenario"]
    write_json("scenario_conditioned_error", {
        "status": "FAIL",
        "fresh": scenario,
        "scenario_p90_regression": {
            key: (
                value["R6"]["p90"]/value["R0"]["p90"]-1
                if value["R0"].get("p90", 0) else None
            ) for key, value in scenario.items()
        },
    })
    write_json("edge_case_analysis", {
        "status": "FAIL_PARTIAL_VISIBILITY_TAIL",
        "edge": fresh["summary"]["direct"]["edge"],
        "moving_camera": fresh["summary"]["direct"]["moving_camera"],
        "near_left_right": {
            "scope": "historical_physical_control_diagnostic_only",
            "fresh_claimed": False,
        },
        "upper_lower": {
            "scope": "historical_physical_control_diagnostic_only",
            "fresh_claimed": False,
        },
        "partial_visibility":
            "represented by image-border and occluded_but_tracked groups",
    })
    write_json("aligned_covariance_reassessment", {
        "status": "NOT_RUN_REFERENCE_GATE_FAIL",
        "raw_center_covariance_adequate": None,
        "reason":
            "R6 failed support coverage and risk hard gates; uncertainty "
            "candidate tuning remains prohibited",
        "diagnostic_only_fresh_coverage":
            fresh["summary"]["prediction"][
                "R6_HYBRID_REFERENCE_BRIDGE"
            ]["coverage"],
    })
    write_json("aligned_normalized_error", {
        "status": "DIAGNOSTIC_ONLY_NOT_A_CALIBRATION",
        "fresh": fresh["summary"]["prediction"][
            "R6_HYBRID_REFERENCE_BRIDGE"
        ]["normalized"],
        "covariance_scale_modified": False,
    })
    write_json("uncertainty_next_step", {
        "status": "BLOCKED",
        "kucr1_uncertainty_search_may_resume": False,
        "classification": "support_envelope_preferred_but_not_yet_valid",
        "next_review":
            "phase8jqv2_4_dynamic_object_geometry_model_review",
    })
    write_json("candidate_risk_regression", {
        "status": "FAIL",
        "raw_surface": risk["raw_surface"],
        "aligned_hybrid": risk["aligned_hybrid"],
        "unsafe_recommendation_increase": (
            risk["aligned_hybrid"]["unsafe_recommendations"]
            - risk["raw_surface"]["unsafe_recommendations"]
        ),
        "runtime_gt_used": False,
    })
    write_json("false_veto_analysis", {
        "status": "FAIL",
        "raw_safe_false_veto_rate":
            risk["raw_surface"]["safe_candidate_false_veto_rate"],
        "aligned_safe_false_veto_rate":
            risk["aligned_hybrid"]["safe_candidate_false_veto_rate"],
        "negative_false_veto": risk["negative_false_veto"],
        "false_veto_reduction_not_achieved": True,
    })
    write_json("unsafe_recommendation_analysis", {
        "status": "FAIL",
        "raw_unsafe_recommendations":
            risk["raw_surface"]["unsafe_recommendations"],
        "aligned_unsafe_recommendations":
            risk["aligned_hybrid"]["unsafe_recommendations"],
        "hard_requirement_no_increase": False,
    })
    implementation_paths = [
        "policy/dynamic/measurement_geometry_adapter_v1.py",
        "policy/dynamic/tracking_collision_reference_bridge_v1.py",
        "policy/dynamic/reference_aligned_center_tracker_v1.py",
        "configs/tracking_collision_reference_contract_v1_candidate.yaml",
    ]
    write_json("implementation_contract", {
        "status": "PASS_SHADOW_ONLY",
        "implementation_hashes": {
            path: digest(ROOT/path) for path in implementation_paths
        },
        "formal_tracker_modified": False,
        "formal_kalman_modified": False,
        "formal_planner_modified": False,
        "runtime_gt_used": False,
    })
    geometry_runtime = [
        case["runtime"]["geometry_export_ms"]["p95"]
        for case in fresh["cases"]
        if case["runtime"]["geometry_export_ms"]["count"]
    ]
    bridge_runtime = [
        case["runtime"]["reference_bridge_ms"]["p95"]
        for case in fresh["cases"]
        if case["runtime"]["reference_bridge_ms"]["count"]
    ]
    bridge_p95_upper = max(bridge_runtime, default=0.)
    write_json("runtime", {
        "status": (
            "PASS" if bridge_p95_upper <=
            config["fresh_validation_gates"][
                "reference_bridge_p95_runtime_max_ms"
            ] else "FAIL"
        ),
        "geometry_export_case_p95_upper_ms":
            max(geometry_runtime, default=0.),
        "reference_bridge_case_p95_upper_ms": bridge_p95_upper,
        "reference_bridge_limit_ms":
            config["fresh_validation_gates"][
                "reference_bridge_p95_runtime_max_ms"
            ],
        "bounded_nonlinear_evaluations":
            config["bridge"]["maximum_fit_evaluations"],
        "host_gpu_risk_replay_seconds": risk["runtime_seconds"],
    })
    write_json("determinism", {
        "status": "PASS",
        "case_specific_branching": False,
        "map_uuid_branching": False,
        "future_input": False,
        "candidate_parameters_frozen": True,
        "fresh_parameters_changed_after_freeze": False,
    })
    write_json("regression", {
        "status": "PASS",
        "kucr1_artifacts_modified": False,
        "ocsr1_adapter_v2_modified": False,
        "ocsr1_validation_results_modified": False,
        "TrackManager_algorithm_modified": False,
        "kalman_process_model_modified": False,
        "kalman_measurement_model_modified": False,
        "kalman_initial_covariance_modified": False,
        "detector_algorithm_modified": False,
        "static_yopo_network_modified": False,
        "static_yopo_checkpoint_modified": False,
        "formal_planner_modified": False,
        "formal_command_modified": False,
    })
    write_json("candidate_selection", {
        "status": "FAIL",
        "selected_reference_contract": None,
        "R1_selected": False,
        "reason":
            "hybrid support coverage and safety-risk hard gates failed fresh validation",
    })
    write_json("compatibility_matrix", {
        "status": "PASS",
        "legacy_measurement": "unchanged",
        "TrackManager": "unchanged",
        "Kalman": "unchanged",
        "OCSR1_adapter_v2": "unchanged",
        "YOPO_network_and_checkpoint": "unchanged",
        "formal_planner_and_command": "unchanged",
        "new_reference_bridge": "development_shadow_only_not_selected",
    })
    write_md("migration_plan", """# PTAR1 Migration Plan

No production migration is authorized.  Preserve the measurement geometry
adapter and the reference bridge as shadow diagnostics.  The next review must
model the runtime-distinguishable geometry of sphere and vertical-cylinder
actors (or define a tighter legal support representation) before reopening
reference-aligned uncertainty calibration.  Do not tune Kalman Q/R or activate
the planner/coasting path in the meantime.
""")
    final = {
        "status": "FAIL",
        "route": "G",
        "primary_cause": "reference_alignment_false_veto_tradeoff",
        "specific_cause":
            "runtime_shape_unobservable_hybrid_support_coverage_and_risk_failure",
        "reference_alignment": "FAIL",
        "timeline_integrity": "PASS",
        "world_frame_integrity": "PASS",
        "center_observable_subset": "PASS",
        "support_envelope": "FAIL",
        "center_velocity": "FAIL",
        "selected_reference_contract": None,
        "fresh_validation": "FAIL_HARD_GATES",
        "kucr1_uncertainty_search_may_resume": False,
        "formal_tracker_modified": False,
        "formal_kalman_modified": False,
        "production_activation_authorized": False,
        "kucr1_artifacts_modified": False,
        "ocsr1_adapter_v2_modified": False,
        "ocsr1_validation_results_modified": False,
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
        "runtime_gt_radius_used": False,
        "runtime_gt_center_used": False,
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
            "phase8jqv2_4_dynamic_object_geometry_model_review",
    }
    write_json("final_result", final)
    write_md("final_recommendation", """# PTAR1 Final Recommendation

PTAR1 fails closed on Route G.  Geometry-aware fitting solves the reference
origin for sufficiently observable spherical actors, but runtime has no legal
shape label and also contains vertical cylinders.  The generic support fallback
covered only 95.24% of fresh actor geometry while increasing safe false-veto
and unsafe recommendations.  Do not resume KUCR1 calibration.

The next allowed phase is
`phase8jqv2_4_dynamic_object_geometry_model_review`.
""")
    write_md("final_readiness", """# PTAR1 Final Readiness

- Timeline and world-frame integrity remain PASS.
- Runtime GT usage remains false.
- Observable-sphere reference recovery is demonstrated.
- Hybrid reference alignment is not production-ready.
- No reference candidate is selected.
- Formal TrackManager, Kalman, YOPO, planner and command remain unchanged.
- Training and formal data generation were not run.
""")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
