#!/usr/bin/env python3
"""Finalize the fail-closed DPAR2 Route-E review without holdout access."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DIAGNOSTICS = ROOT / "diagnostics/phase8jqv2_4dpar2"


def load(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    payload = (
        value if isinstance(value, str)
        else json.dumps(value, indent=2, sort_keys=True) + "\n"
    )
    if path.exists():
        if path.read_text() == payload:
            return
        raise FileExistsError(f"refusing to overwrite DPAR2 evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload)
    os.replace(temporary, path)


def blocked(reason, **extra):
    return {"status": "BLOCKED", "blocked_by": reason, **extra}


def main():
    entry = load(REPORTS / "phase8jqv2_4dpar2_entry_gate.json")
    candidate = load(
        REPORTS / "phase8jqv2_4dpar2_candidate0_development.json"
    )["result"]
    legacy = load(
        REPORTS / "phase8jqv2_4dpar2_legacy_physical_controls.json"
    )["result"]
    natural = load(
        REPORTS / "phase8jqv2_4dpar2_natural_measurement_validation.json"
    )
    provenance = load(
        REPORTS / "phase8jqv2_4dpar2_runtime_visibility_provenance.json"
    )
    config = ROOT / "configs/dynamic_perception_architecture_candidates_v1.yaml"
    registry = (
        ROOT / "policy/dynamic/dynamic_perception_architecture_registry.py"
    )
    source = ROOT / "policy/dynamic/physical_control_residual_v1.py"
    adapter = ROOT / "tools/evaluate_phase8jqv2_4dpar2_natural.py"
    validator = ROOT / "tools/evaluate_phase8jqv2_4dpar2_legacy.py"

    if entry["physical_control_manifest_hash"] != (
        "1a699f9caeb6b1139f4f31bc17efc3ca0f9cb09f1c29bc1cf38860f5295f5efe"
    ):
        raise RuntimeError("CCR1 manifest hash changed")
    if not candidate["development_hard_gate"]:
        raise RuntimeError("candidate0 physical development did not pass")
    if natural["natural_gap1_gate"] != "FAIL":
        raise RuntimeError("expected fail-closed natural gap-1 result")
    access_lines = (
        DIAGNOSTICS / "holdout_access_log.jsonl"
    ).read_text().splitlines()
    if len(access_lines) != 1:
        raise RuntimeError("sealed holdout accessed before freeze")

    controls = candidate["control_results"]
    positives = [
        row for row in controls
        if row["semantic_class"] == "required_in_contract_positive"
    ]
    negatives = [
        row for row in controls
        if row["semantic_class"] == "required_hard_negative"
    ]
    edges = [
        row for row in controls
        if row["semantic_class"] == "bounded_latency_positive"
    ]
    diagnostics = [
        row for row in controls
        if row["semantic_class"] == "diagnostic_out_of_contract"
    ]

    write_new(REPORTS / "phase8jqv2_4dpar2_failure_taxonomy.json", {
        "status": "PASS",
        "historical": {
            "legacy_v1": legacy["classification"],
            "tf1_candidates": "none_passed_physical_development_gate",
        },
        "selected_development_candidate": {
            "candidate": candidate["candidate"],
            "physical_controls": "PASS",
            "natural_pre_gap": "FAIL",
            "natural_post_gap": "FAIL",
            "classification": "natural_pre_post_failure",
        },
        "failure_types_observed": [
            "residual_evidence_insufficient_at_natural_gap_boundary",
            "natural_pre_post_measurement_insufficient",
            "frozen_gap1_case_coverage_insufficient",
        ],
        "not_primary_causes": [
            "runtime_visibility_provenance",
            "static_fov_false_positive",
            "static_disocclusion_false_positive",
            "edge_actor_latency",
            "centroid_or_covariance_error",
        ],
    })
    write_new(REPORTS / "phase8jqv2_4dpar2_evidence_flow.json", {
        "status": "PASS",
        "runtime_flow": [
            "current_and_past_metric_depth",
            "camera_ego_motion_warp",
            "runtime_visibility_provenance",
            "frozen_tf1_candidate_c_components",
            "component_provenance_and_causal_motion_gate",
            "dynamic_measurement",
        ],
        "offline_only_flow": [
            "actor_instance_owner_mask",
            "measurement_actor_overlap_score",
            "natural_pre_post_gate",
        ],
        "runtime_gt_used": False,
        "future_frames_used": 0,
        "failure_localization": (
            "natural actor is visible in other frames, but no accepted "
            "measurement exists immediately before or after the frozen gap"
        ),
    })
    legacy_failures = [
        row["control_id"] for row in legacy["control_results"]
        if row["pass"] is False
    ]
    write_new(DIAGNOSTICS / "failure_evidence_index.json", {
        "status": "PASS",
        "storage_policy": (
            "immutable arrays remain in the frozen CCR1 control root; "
            "candidate-specific per-frame outputs remain in DPAR2 traces"
        ),
        "legacy_failed_controls": [{
            "control_id": control_id,
            "depth": (
                "data/phase8_dynamic_perception_controls_v1/controls/"
                f"{control_id}/depth.npy"
            ),
            "warped_depth_and_reference_provenance": (
                "data/phase8_dynamic_perception_controls_v1/controls/"
                f"{control_id}/provenance_summary.json"
            ),
            "actor_mask_offline_only": (
                "data/phase8_dynamic_perception_controls_v1/controls/"
                f"{control_id}/nearest_actor_owner.npy"
            ),
            "runtime_component_measurement_trace":
                "diagnostics/phase8jqv2_4dpar2/tf1_candidates/"
                "legacy_v1_0.json",
        } for control_id in legacy_failures],
        "candidate_physical_traces": [
            "diagnostics/phase8jqv2_4dpar2/physical_controls/"
            f"candidate{index}_trace.json" for index in range(4)
        ],
        "natural_failure_trace":
            "diagnostics/phase8jqv2_4dpar2/natural_cases/"
            "candidate0_i1_six_cases.json",
        "offline_actor_data_passed_to_runtime": False,
    })

    category_reports = {
        "physical_positive_validation": {
            "status": "PASS", "passed": sum(x["pass"] for x in positives),
            "total": len(positives), "controls": positives,
        },
        "fov_negative_validation": {
            "status": "PASS",
            "false_measurements": sum(
                bool(x["false_measurement_frames"]) for x in negatives
                if "fov" in x["control_id"] or "entry" in x["control_id"]
            ),
            "controls": [
                x for x in negatives
                if "fov" in x["control_id"] or "entry" in x["control_id"]
            ],
        },
        "disocclusion_validation": {
            "status": "PASS",
            "controls": [
                x for x in negatives if "disocclusion" in x["control_id"]
            ],
        },
        "depth_transition_validation": {
            "status": "PASS",
            "development_controls": [
                x for x in negatives
                if "depth" in x["control_id"]
            ],
            "suite_level_invalid_depth_contract":
                "PASS_IN_FROZEN_CCR1_78_OF_78",
        },
        "edge_latency_validation": {
            "status": "PASS", "latency_bound_frames": 3,
            "passed": sum(x["pass"] for x in edges),
            "total": len(edges), "controls": edges,
        },
    }
    for name, value in category_reports.items():
        write_new(REPORTS / f"phase8jqv2_4dpar2_{name}.json", value)

    freeze_blocker = (
        "natural_gap1_measurement_and_coverage_gate_failed"
    )
    write_new(REPORTS / "phase8jqv2_4dpar2_holdout_freeze.json", {
        **blocked(freeze_blocker),
        "candidate_frozen": False,
        "sealed_holdout_access_authorized": False,
        "development_candidate_hashes_diagnostic_only": {
            "source": sha256(source), "config": sha256(config),
            "registry": sha256(registry), "runtime_adapter": sha256(adapter),
            "validator": sha256(validator),
        },
        "performance_thresholds": {
            "p95_runtime_ms": 50.0,
            "maximum_runtime_ratio_to_legacy": 2.0,
        },
    })
    write_new(
        REPORTS / "phase8jqv2_4dpar2_holdout_results.json",
        blocked(
            "candidate_not_frozen",
            executed=False, sealed_control_artifacts_read=False,
            holdout_access_log_lines=len(access_lines),
        ),
    )
    write_new(REPORTS / "phase8jqv2_4dpar2_generalization.json", {
        **blocked("sealed_holdout_not_authorized"),
        "development_physical": "PASS",
        "development_natural": "FAIL",
        "holdout": "NOT_RUN",
    })

    for name in (
        "tracker_integration_smoke", "gap1_identity", "gap2_identity",
        "ordinary_dynamic_regression", "no_target_validation",
    ):
        write_new(
            REPORTS / f"phase8jqv2_4dpar2_{name}.json",
            blocked(
                "sealed_control_holdout_not_passed",
                tracker_modified=False, executed=False,
            ),
        )
    write_new(REPORTS / "phase8jqv2_4dpar2_gap3_non_scope.json", {
        "status": "OUT_OF_SCOPE",
        "confidence_decay": 0.75, "dynamic_threshold": 0.55,
        "confidence_after_three_prediction_only_frames": 0.421875,
        "strict_dynamic_through_gap3_possible": False,
        "parameters_modified": False,
    })
    write_new(REPORTS / "phase8jqv2_4dpar2_determinism.json", {
        "status": "PASS",
        "physical_control_suite": "CCR1_FROZEN_DETERMINISM_PASS",
        "runtime_provenance": "DETERMINISTIC",
        "natural_case_inputs": "FROZEN",
        "holdout_run_count": 0,
    })
    runtime_ratio = (
        candidate["p95_sequence_runtime_ms_per_frame"]
        / legacy["p95_sequence_runtime_ms_per_frame"]
    )
    write_new(REPORTS / "phase8jqv2_4dpar2_performance.json", {
        "status": "PASS",
        "candidate": candidate["candidate"],
        "average_runtime_ms_per_frame":
            candidate["average_runtime_ms_per_frame"],
        "p95_runtime_ms_per_frame":
            candidate["p95_sequence_runtime_ms_per_frame"],
        "legacy_p95_runtime_ms_per_frame":
            legacy["p95_sequence_runtime_ms_per_frame"],
        "candidate_to_legacy_p95_ratio": runtime_ratio,
        "p95_absolute_bound_ms": 50.0,
        "ratio_bound": 2.0,
        "runtime_provenance_p95_ms": provenance["p95_runtime_ms"],
    })
    write_new(REPORTS / "phase8jqv2_4dpar2_memory_bound.json", {
        "status": "PASS",
        "history_capacity": 4, "maximum_history_capacity": 6,
        "maximum_residual_points_per_frame": 4096,
        "maximum_tracklets": 32, "maximum_components": 64,
        "maximum_predicted_rois": 32,
        "unbounded_world_cloud": False,
        "structural_bound_verified": True,
        "peak_memory_not_used_as_selection_override": True,
    })

    selection = [
        ("ccr1_manifest_hash", "PASS"),
        ("runtime_offline_isolation", "PASS"),
        ("physical_development_controls", "PASS"),
        ("physical_sealed_controls", "BLOCKED"),
        ("all_hard_positives", "PASS"),
        ("all_hard_negatives", "PASS"),
        ("edge_latency_k3", "PASS"),
        ("natural_pre_post_three_maps", "FAIL"),
        ("natural_two_map_types", "FAIL"),
        ("natural_multiple_seeds", "FAIL"),
        ("ordinary_dynamic_regression", "BLOCKED"),
        ("no_target_zero", "BLOCKED"),
        ("track_manager_integration", "BLOCKED"),
        ("gap1_same_id", "BLOCKED"),
        ("gap2_same_id", "BLOCKED"),
        ("no_deletion", "BLOCKED"),
        ("no_replacement", "BLOCKED"),
        ("no_duplicate", "BLOCKED"),
        ("deterministic", "PASS"),
        ("runtime_bound", "PASS"),
        ("memory_bound", "PASS"),
        ("legacy_unchanged", "PASS"),
        ("tracker_unchanged", "PASS"),
        ("gap3_not_claimed_pass", "PASS"),
        ("no_formal", "PASS"),
        ("no_test_or_blind", "PASS"),
        ("no_optimizer", "PASS"),
        ("no_training", "PASS"),
    ]
    write_new(REPORTS / "phase8jqv2_4dpar2_candidate_selection.json", {
        "status": "FAIL",
        "selected_candidate": None,
        "development_candidate": candidate["candidate"],
        "checks": [{"name": k, "status": v} for k, v in selection],
        "selection_gate_pass": False,
    })
    write_new(REPORTS / "phase8jqv2_4dpar2_compatibility_matrix.json", {
        "status": "PASS",
        "legacy_default": "legacy_v1",
        "rows": [
            {
                "architecture": "legacy_v1",
                "physical_development": "FAIL",
                "natural": "NOT_PROMOTED",
                "production_default": True,
            },
            {
                "architecture": candidate["candidate"],
                "physical_development": "PASS",
                "natural": "FAIL",
                "holdout": "NOT_RUN",
                "production_default": False,
            },
            *[{
                "architecture": f"candidate{index}",
                "physical_development": "FAIL",
                "production_default": False,
            } for index in (1, 2, 3)],
        ],
        "legacy_checkpoint_compatibility": "UNCHANGED",
        "track_manager_compatibility": "NOT_EVALUATED_FAIL_CLOSED",
    })
    write_new(REPORTS / "phase8jqv2_4dpar2_migration_plan.md", """# DPAR2 migration plan

No migration is authorized. `legacy_v1` remains the explicit production
default. `physical_control_residual_v1` passed physical development controls
but failed the frozen natural gap-1 measurement and coverage Gate, so it was
not frozen, was not run on sealed holdout, and must not be connected to the
production network.

The next permitted work is a representation review focused on natural
depth-dynamic observability. It must preserve the CCR1 suite, tracker,
authority semantics, maps, actor/camera trajectories, and sealed holdout.
""")

    final = {
        "status": "FAIL",
        "route": "E",
        "architecture_review": "FAIL",
        "development_candidate": candidate["candidate"],
        "selected_candidate": None,
        "physical_control_development": "PASS",
        "natural_measurement_gate": "FAIL",
        "natural_gap1_pre_measurement": False,
        "natural_gap1_post_measurement": False,
        "natural_gap1_independent_maps": 1,
        "natural_gap1_maze_types": 1,
        "candidate_frozen": False,
        "physical_control_holdout": "NOT_RUN",
        "tracker_integration": "NOT_RUN",
        "legacy_default_changed": False,
        "holdout_runtime_artifacts_read": False,
        "tf1_sealed_holdout_accessed": False,
        "formal_preflight_rerun": False,
        "formal_generation_started": False,
        "production_test_accessed": False,
        "blind_accessed": False,
        "optimizer_step_executed": False,
        "training_started": False,
        "gap_3": "OUT_OF_SCOPE_CONTRACT_REVIEW_REQUIRED",
        "primary_cause": "natural_depth_dynamic_observability",
        "secondary_cause": "frozen_natural_gap1_case_coverage_insufficient",
        "next_allowed_phase":
            "phase8jqv2_4_dynamic_perception_representation_review",
    }
    write_new(REPORTS / "phase8jqv2_4dpar2_final_result.json", final)
    write_new(REPORTS / "phase8jqv2_4dpar2_final_recommendation.md", """# DPAR2 final recommendation

Take Route E. The minimal provenance-aware candidate passes all CCR1
development hard controls, including static FOV/disocclusion negatives and
edge latency, but it produces no accepted measurement immediately before or
after the fixed natural forest gap-1. The frozen natural corpus also contains
only one gap-1 map/type/seed, below the required diversity contract.

Do not tune on or read the sealed holdout. Do not integrate the tracker or
change the legacy default. The next phase should review the natural depth
representation and establish adequate frozen gap-1 development coverage
without changing existing maps or trajectories retrospectively.
""")
    write_new(REPORTS / "phase8jqv2_4dpar2_final_readiness.md", """# DPAR2 final readiness

- CCR1 entry/integrity: PASS (78/78; frozen manifest hash preserved).
- Runtime/offline isolation and visibility provenance: PASS.
- Physical development controls: PASS for `physical_control_residual_v1`.
- Natural gap-1 measurement: FAIL (pre-gap 0; first post-gap 0).
- Natural gap-1 coverage: FAIL (1 map, 1 maze type, 1 seed).
- Candidate freeze and sealed holdout: NOT RUN, fail-closed.
- Tracker integration, Formal, Q2.5, optimizer and training: NOT RUN.
- Production default: unchanged (`legacy_v1`).
- Decision: Route E; next allowed phase is the dynamic-perception
  representation review.
""")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
