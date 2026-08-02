#!/usr/bin/env python3
"""Freeze and verify the Phase 8J-V2 entry contract."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main():
    required = [
        "reports/phase8h_blind_once_quality.json",
        "reports/phase8h_final_perception_gate.json",
        "reports/phase8h_final_readiness.md",
        "reports/.phase8h_blind_once.lock",
        "reports/phase8i_final_result.json",
        "reports/phase8i_final_recommendation.md",
        "reports/phase8i_candidate_failure_decomposition.json",
        "reports/phase8i_checkpoint_matrix.json",
        "reports/phase8i_estimated_cache_validation.json",
        "reports/phase8j_final_result.json",
        "reports/phase8j_coverage_gate.json",
        "reports/phase8jr_final_result.json",
        "reports/phase8jr_capacity_gate.json",
        "reports/phase8jp_final_result.json",
        "reports/phase8jp_capacity_gate.json",
        "reports/phase8jq_final_result.json",
        "reports/phase8jq_final_taxonomy.json",
        "reports/phase8jq_controller_authoritative_envelope.json",
        "reports/phase8jq_generator_contract.md",
        "reports/phase8jqv2_final_result.json",
        "reports/phase8jqv2_final_recommendation.md",
        "reports/phase8jqv2_final_readiness.md",
        "reports/phase8jqv2_evaluator_spec.md",
        "reports/phase8jqv2_checkpoint_matrix.json",
        "reports/phase8jqv2_component_ablation.json",
        "reports/phase8jqv2_static_validation.json",
        "reports/phase8jqv2_candidate_failure_decomposition.json",
        "reports/phase8jqv2_actionability_metrics.json",
        "reports/phase8jqv2_label_scale_audit.json",
        "reports/phase8jqv2_context_gap_analysis.json",
        "configs/safety_evaluator_v2.yaml",
        "loss/safety_geometry_v2.py",
        "policy/safety_evaluator_v2.py",
        "loss/safety_loss.py",
        "loss/dynamic_safety_loss.py",
        "loss/trajectory_sampler.py",
        "loss/coverage_loss.py",
        "policy/phase8j_coverage_trainer.py",
        "policy/dep_trainer.py",
        "policy/dep_network.py",
        "policy/models/head.py",
        "policy/checkpoint_utils.py",
    ]
    missing = [name for name in required if not (ROOT / name).is_file()]
    if missing:
        raise RuntimeError(f"missing Phase 8J-V2 entry inputs: {missing}")

    phase8h = json.loads(
        (REPORTS / "phase8h_final_perception_gate.json").read_text()
    )
    phase8h_lock = json.loads(
        (REPORTS / ".phase8h_blind_once.lock").read_text()
    )
    phase8i_cache = json.loads(
        (REPORTS / "phase8i_estimated_cache_validation.json").read_text()
    )
    phase8i_matrix = json.loads(
        (REPORTS / "phase8i_checkpoint_matrix.json").read_text()
    )
    v2_entry = json.loads(
        (REPORTS / "phase8jqv2_entry_gate.json").read_text()
    )
    v2 = json.loads((REPORTS / "phase8jqv2_final_result.json").read_text())
    determinism = json.loads(
        (REPORTS / "phase8jqv2_determinism_validation.json").read_text()
    )
    static = json.loads(
        (REPORTS / "phase8jqv2_static_validation.json").read_text()
    )
    cache_index = Path(phase8i_cache["index"])

    checkpoints = [
        row for row in phase8i_matrix["checkpoints"]
        if row["analyzed_full_phase8i"]
    ]
    checkpoint_checks = []
    for row in checkpoints:
        path = Path(row["path"])
        checkpoint_checks.append({
            "name": row["name"],
            "path": str(path),
            "expected_sha256": row["sha256"],
            "observed_sha256": sha256(path),
            "hash_match": sha256(path) == row["sha256"],
            "available": path.is_file(),
        })

    risk_files = sorted(
        (ROOT / "diagnostics").glob("production_*_*.json")
    )
    frozen = {
        "v2_config_hash": sha256(ROOT / "configs/safety_evaluator_v2.yaml"),
        "v2_geometry_hash": sha256(ROOT / "loss/safety_geometry_v2.py"),
        "v2_timeline_hash": sha256(ROOT / "policy/safety_evaluator_v2.py"),
        "v2_uncertainty_policy_hash": v2["uncertainty_policy_hash"],
        "Simulator_geometry_hash": v2["Simulator_geometry_hash"],
        "v2_semantic_determinism_hash": determinism["semantic_results_hash"],
        "phase8h_implementation_hash": phase8h_lock["implementation_sha256"],
        "phase8h_config_hash": phase8h_lock["config_sha256"],
        "phase8h_evaluator_hash": phase8h_lock["evaluator_sha256"],
        "estimated_cache_index_hash": sha256(cache_index),
        "estimated_cache_content_hash": phase8i_cache["index_content_hash"],
        "dataset_manifest_hash": sha256(
            ROOT / "data/phase8_dynamic_production/dataset_manifest.yaml"
        ),
        "static_map_catalog_hash": sha256(
            ROOT / "configs/static_map_catalog.yaml"
        ),
        "dynamic_map_catalog_hash": sha256(
            ROOT / "data/phase8_dynamic_production/map_catalog.yaml"
        ),
        "risk_set_manifest_hashes": {
            str(path.relative_to(ROOT)): sha256(path) for path in risk_files
        },
        "v1_static_evaluator_hash": sha256(ROOT / "loss/safety_loss.py"),
        "v1_dynamic_evaluator_hash": sha256(
            ROOT / "loss/dynamic_safety_loss.py"
        ),
        "v1_phase8i_report_hash": sha256(
            REPORTS / "phase8i_candidate_failure_decomposition.json"
        ),
        "checkpoint_hashes": {
            row["name"]: row["sha256"] for row in checkpoints
        },
    }
    checks = {
        "phase8h_perception_pass": phase8h.get("status") == "PASS",
        "phase8h_blind_run_count_one": (
            phase8h.get("phase8h_blind_run_count") == 1
        ),
        "rebaseline_status_pass": v2.get("status") == "PASS",
        "rebaseline_ready": v2.get("rebaseline_ready") is True,
        "route_c": v2.get("route") == "C",
        "evaluator_v2": v2.get("evaluator_version") == "v2",
        "authorized_next_phase": (
            v2.get("next_allowed_phase")
            == "phase8j_coverage_then_score_v2"
        ),
        "network_weights_unmodified": (
            v2.get("network_weights_modified") is False
        ),
        "training_unexecuted": v2.get("training_executed") is False,
        "production_test_unused": v2.get("production_test_used") is False,
        "phase8h_frozen": v2.get("phase8h_perception_frozen") is True,
        "v2_config_hash_match": (
            frozen["v2_config_hash"] == v2["config_hash"]
        ),
        "v2_geometry_hash_match": (
            frozen["v2_geometry_hash"] == v2["geometry_hash"]
        ),
        "v2_timeline_hash_match": (
            frozen["v2_timeline_hash"] == v2["timeline_hash"]
        ),
        "v2_determinism_pass": determinism.get("status") == "PASS",
        "v2_semantic_hash_match": (
            determinism.get("semantic_results_hash")
            == "41ff8b9203ca7fe92325addae7be2fef1b51569281763fa34a1a78915871bd45"
        ),
        "estimated_cache_pass": phase8i_cache.get("status") == "PASS",
        "estimated_cache_hash_match": (
            frozen["estimated_cache_index_hash"]
            == phase8i_cache["index_sha256"]
        ),
        "estimated_cache_strictly_causal": (
            phase8i_cache.get("strictly_causal") is True
        ),
        "ten_checkpoint_hashes_match": (
            len(checkpoint_checks) == 10
            and all(row["available"] and row["hash_match"]
                    for row in checkpoint_checks)
        ),
        "phase8b_unavailable_declared": (
            phase8i_matrix["phase8b"]["available"] is False
        ),
        "static_full_sequential": (
            static.get("status") == "PASS"
            and static.get("window_count") == 10000
            and static.get("full_sequential") is True
        ),
        "v1_static_hash_preserved": (
            frozen["v1_static_evaluator_hash"]
            == v2_entry["frozen_hashes"]["v1_static_evaluator_sha256"]
        ),
        "v1_dynamic_hash_preserved": (
            frozen["v1_dynamic_evaluator_hash"]
            == v2_entry["frozen_hashes"]["v1_dynamic_evaluator_sha256"]
        ),
        "v1_report_hash_preserved": (
            frozen["v1_phase8i_report_hash"]
            == v2_entry["frozen_hashes"]["v1_phase8i_report_sha256"]
        ),
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "phase": "8J-V2",
        "checks": checks,
        "frozen_hashes": frozen,
        "checkpoint_audit": checkpoint_checks,
        "phase8b": phase8i_matrix["phase8b"],
        "required_input_hashes": {
            name: sha256(ROOT / name) for name in required
        },
        "diagnostic_catalog": {
            "phase8i": {
                str(path.relative_to(ROOT)): sha256(path)
                for path in sorted((ROOT / "diagnostics/phase8i").glob("*"))
                if path.is_file()
            },
            "phase8jqv2": {
                str(path.relative_to(ROOT)): sha256(path)
                for path in sorted((ROOT / "diagnostics/phase8jqv2").glob("*"))
                if path.is_file()
            },
        },
        "network_weights_modified": False,
        "training_executed": False,
        "production_test_used": False,
        "blind_used": False,
    }
    atomic_json(REPORTS / "phase8jv2_entry_gate.json", report)
    if report["status"] != "PASS":
        raise RuntimeError(f"Phase 8J-V2 entry Gate failed: {checks}")

    actionability = json.loads(
        (REPORTS / "phase8jqv2_actionability_metrics.json").read_text()
    )
    labels = json.loads(
        (REPORTS / "phase8jqv2_label_scale_audit.json").read_text()
    )
    planning = json.loads(
        (REPORTS / "phase8jqv2_planning_coverage.json").read_text()
    )
    physical = json.loads(
        (REPORTS / "phase8jqv2_physical_coverage.json").read_text()
    )
    decision_static = static["results"]["fixed_050_seed8403"]
    corrected_static = static["results"]["corrected_static_initialization"]
    expected = {
        "valid_estimated_planning_joint": (516, 2052),
        "valid_gt_physical_joint": (501, 2052),
        "preventable_estimated_planning": (367, 1903),
        "combined_selection": (746, 2052),
        "physical_label_inversion": (0, 2052),
        "planning_label_inversion": (0, 2052),
        "valid_static_decision": (3680, 10000),
        "already_unsafe": (142, 2052),
        "valid_static_corrected": (2697, 10000),
    }
    observed = {
        "valid_estimated_planning_joint": planning["valid_estimated"][
            "joint_coverage_failure"
        ],
        "valid_gt_physical_joint": physical["valid_gt"][
            "joint_coverage_failure"
        ],
        "preventable_estimated_planning": planning["valid_estimated"][
            "preventable_coverage_failure"
        ],
        "combined_selection": labels["combined_selection_failure"],
        "physical_label_inversion": labels["physical_label_inversion"],
        "planning_label_inversion": labels["planning_label_inversion"],
        "valid_static_decision": decision_static[
            "physical_static_coverage_failure"
        ],
        "already_unsafe": actionability["valid_estimated"][
            "already_unsafe_fraction"
        ],
        "valid_static_corrected": corrected_static[
            "physical_static_coverage_failure"
        ],
    }
    reproduction_checks = {
        name: (
            value["numerator"] == expected[name][0]
            and value["denominator"] == expected[name][1]
        )
        for name, value in observed.items()
    }
    reproduction_checks["semantic_determinism_hash"] = checks[
        "v2_semantic_hash_match"
    ]
    baseline = {
        "status": (
            "PASS" if all(reproduction_checks.values()) else "FAIL"
        ),
        "evaluator_version": "v2",
        "checks": reproduction_checks,
        "metrics": observed,
        "semantic_determinism_hash": determinism[
            "semantic_results_hash"
        ],
        "frozen_hashes": frozen,
        "v1_metrics_excluded_from_gate": True,
        "network_weights_modified": False,
        "training_executed": False,
        "production_test_used": False,
    }
    atomic_json(
        REPORTS / "phase8jv2_baseline_reproduction.json", baseline
    )
    if baseline["status"] != "PASS":
        raise RuntimeError("Phase 8J-V2 baseline reproduction failed")
    print(json.dumps({
        "status": "PASS",
        "entry_gate": str(REPORTS / "phase8jv2_entry_gate.json"),
        "baseline": str(
            REPORTS / "phase8jv2_baseline_reproduction.json"
        ),
        "checkpoint_count": len(checkpoints),
        "phase8b_available": False,
    }, indent=2))


if __name__ == "__main__":
    main()
