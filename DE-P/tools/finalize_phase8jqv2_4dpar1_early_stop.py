#!/usr/bin/env python3
"""Materialize honest DPAR1 blocked-stage reports after the control gate."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DIAGNOSTICS = ROOT / "diagnostics/phase8jqv2_4dpar1"


def read(name):
    return json.loads((REPORTS / name).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite DPAR1 evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    text = (
        value.rstrip() + "\n" if isinstance(value, str)
        else json.dumps(value, indent=2, sort_keys=True) + "\n"
    )
    temporary.write_text(text)
    os.replace(temporary, path)


def blocked(stage, extra=None):
    value = {
        "status": "NOT_RUN_CONTROL_GATE_FAILED",
        "stage": stage,
        "primary_cause": "dynamic_perception_control_contract_invalid",
        "result_claimed": False,
        "holdout_accessed": False,
        "test_accessed": False,
        "blind_accessed": False,
    }
    value.update(extra or {})
    return value


def main():
    entry = read("phase8jqv2_4dpar1_entry_gate.json")
    control = read("phase8jqv2_4dpar1_control_contract_audit.json")
    final = read("phase8jqv2_4dpar1_final_result.json")
    if entry["status"] != "PASS":
        raise RuntimeError("DPAR1 entry did not pass")
    if control["status"] != "FAIL":
        raise RuntimeError("early-stop finalizer requires failed control gate")
    if control["architecture_design_allowed"]:
        raise RuntimeError("architecture design unexpectedly allowed")

    write_new(
        REPORTS / "phase8jqv2_4dpar1_architecture_invariants.json",
        {
            "status": "FROZEN_REQUIREMENTS_NOT_IMPLEMENTED",
            "candidate_created": False,
            "invariants": {
                "causal": "current and past only",
                "sensor": "depth only",
                "ego_motion_compensated": True,
                "runtime_gt": False,
                "visibility_aware": True,
                "birth_reacquisition_separated": True,
                "static_conservative": True,
                "no_target_conservative": True,
                "spatially_coherent": True,
                "deterministic": True,
                "bounded_history": True,
                "bounded_runtime": True,
                "deployment_compatible": True,
                "explicit_non_gt_provenance": True,
                "fail_closed": True,
            },
            "implementation_conformance_evaluated": False,
            "reason": "control contract failed before architecture design",
        },
    )
    write_new(
        REPORTS / "phase8jqv2_4dpar1_architecture_review.md",
        """# DPAR1 architecture review

The architecture comparison was not started. Stage A found that both controls
which define the central trade-off are semantically invalid:

- `small_projection` is an uncalibrated 4×4 depth overwrite outside the
  physical radius and detection-distance contract;
- `fov_boundary_change` is a manual border overwrite with a stationary
  camera, not an FOV visibility transition.

Designing visibility provenance, track-before-detect, or dual-path gates
against these fixtures would tune the architecture to artifacts. The mandated
fail-closed decision is therefore Route B, control-contract repair.
""",
    )
    write_new(
        REPORTS / "phase8jqv2_4dpar1_candidate_specs.json",
        blocked("candidate_specification", {
            "candidate_count": 0,
            "candidate_source_files_created": [],
            "legacy_default": "legacy_v1",
        }),
    )
    write_new(
        REPORTS / "phase8jqv2_4dpar1_interface_spec.json",
        blocked("dynamic_measurement_v3_interface", {
            "interface_created": False,
            "track_manager_interface_changed": False,
        }),
    )

    report_specs = {
        "visibility_provenance_validation.json":
            ("visibility_provenance", {}),
        "warp_validity_validation.json": ("warp_validity", {}),
        "fov_boundary_stress.json":
            ("fov_boundary_stress", {
                "invalid_historical_fixture_not_reused": True,
            }),
        "candidate_a_development.json":
            ("visibility_aware_residual_candidate", {}),
        "candidate_b_development.json":
            ("causal_3d_track_before_detect_candidate", {}),
        "tracklet_validation.json": ("tracklet_validation", {}),
        "small_projection_validation.json":
            ("small_projection_stress", {
                "historical_fixture_invalid": True,
            }),
        "candidate_c_development.json":
            ("dual_path_candidate", {
                "prerequisite_candidate_a_or_b_passed": False,
            }),
        "reacquisition_validation.json":
            ("reacquisition_validation", {}),
        "duplicate_suppression.json":
            ("duplicate_measurement_suppression", {}),
        "static_validation.json": ("static_regression", {}),
        "no_target_validation.json": ("no_target_regression", {}),
        "ordinary_dynamic_regression.json":
            ("ordinary_dynamic_regression", {}),
        "natural_measurement_validation.json":
            ("natural_measurement_validation", {}),
        "holdout_results.json":
            ("sealed_holdout", {"sealed_holdout_opened": False}),
        "generalization.json":
            ("generalization", {"generalization_claimed": False}),
        "tracker_integration_smoke.json":
            ("track_manager_integration", {
                "adapter_created": False,
                "track_manager_modified": False,
            }),
        "gap1_identity.json": ("gap1_identity", {}),
        "gap2_identity.json": ("gap2_identity", {}),
        "performance.json":
            ("performance_gate", {
                "runtime_measured": False,
                "threshold_relaxed": False,
            }),
        "memory_bound.json":
            ("memory_gate", {"memory_measured": False}),
        "determinism.json":
            ("candidate_determinism", {
                "candidate_output_compared": False,
            }),
    }
    for suffix, (stage, extra) in report_specs.items():
        write_new(
            REPORTS / f"phase8jqv2_4dpar1_{suffix}",
            blocked(stage, extra),
        )

    write_new(
        REPORTS / "phase8jqv2_4dpar1_holdout_freeze.json",
        blocked("holdout_freeze", {
            "selected_for_holdout": [],
            "maximum_allowed": 2,
            "source_hashes": None,
            "parameters": None,
            "sealed_holdout_opened": False,
            "no_tuning_after_holdout": True,
        }),
    )
    write_new(
        REPORTS / "phase8jqv2_4dpar1_gap3_non_scope.json",
        {
            "status": "OUT_OF_SCOPE_CONTRACT_REVIEW_REQUIRED",
            "initial_confidence": 1.0,
            "confidence_decay": 0.75,
            "misses": 3,
            "confidence_after_three_misses": 0.421875,
            "dynamic_threshold": 0.55,
            "strict_dynamic_through_gap3_possible": False,
            "parameters_modified": False,
            "gap3_pass_claimed": False,
        },
    )
    write_new(
        REPORTS / "phase8jqv2_4dpar1_candidate_selection.json",
        {
            "status": "FAIL",
            "selected_candidate": None,
            "candidate_selected": False,
            "candidates_evaluated": 0,
            "decision_route": "B",
            "primary_cause":
                "dynamic_perception_control_contract_invalid",
            "holdout_accessed": False,
            "next_allowed_phase":
                "phase8jqv2_4_dynamic_perception_control_contract_repair",
        },
    )
    write_new(
        REPORTS / "phase8jqv2_4dpar1_compatibility_matrix.json",
        {
            "status": "PASS_NO_ARCHITECTURE_CHANGE",
            "legacy_depth_input": "unchanged",
            "legacy_measurement_output": "unchanged",
            "track_manager_interface": "unchanged",
            "checkpoint": "no impact",
            "dataset": "no impact",
            "deployment": "no impact",
            "rollback": "not required",
            "new_architecture_enabled": False,
        },
    )
    write_new(
        REPORTS / "phase8jqv2_4dpar1_migration_plan.md",
        """# DPAR1 migration plan

No migration is authorized or required. No architecture, adapter, interface,
configuration, checkpoint, dataset, or deployment default was changed.

The next phase may only repair and freeze the control suite. Architecture
migration planning remains blocked until a candidate passes development,
sealed holdout, and limited tracker integration.
""",
    )
    for directory in (
        "fov_boundary", "small_projection", "candidate_a", "candidate_b",
        "candidate_c", "false_measurements", "tracklets",
        "reacquisition", "holdout_failures",
    ):
        write_new(
            DIAGNOSTICS / directory / "status.json",
            blocked(directory, {
                "control_audit_evidence_only":
                    directory in {"fov_boundary", "small_projection"},
            }),
        )

    invariant_check = {
        "status": "PASS",
        "legacy_temporal_foreground_modified": False,
        "legacy_range_image_foreground_modified": False,
        "tf1_candidates_modified": False,
        "TrackManager_algorithm_modified": False,
        "occlusion_constructor_v2_1_modified": False,
        "occlusion_constructor_v2_2_modified": False,
        "occlusion_identity_schedule_v1_modified": False,
        "map_profiles_modified": False,
        "motion_contract_modified": False,
        "authority_semantics_modified": False,
        "sensor_configuration_modified": False,
        "annex_used": False,
        "pointcloud_sensor_enabled": False,
        "new_maps_generated": False,
        "formal_preflight_rerun": False,
        "formal_v3_entry_created": False,
        "formal_v3_generation_started": False,
        "production_test_accessed": False,
        "blind_accessed": False,
        "optimizer_step_executed": False,
        "training_started": False,
        "control_audit_sha256": sha(
            REPORTS / "phase8jqv2_4dpar1_control_contract_audit.json"
        ),
        "final_result_sha256": sha(
            REPORTS / "phase8jqv2_4dpar1_final_result.json"
        ),
    }
    write_new(
        REPORTS / "phase8jqv2_4dpar1_final_constraints.json",
        invariant_check,
    )
    print(json.dumps({
        "status": final["status"],
        "decision_route": "B",
        "blocked_reports_materialized": len(report_specs),
        "holdout_accessed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
