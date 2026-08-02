#!/usr/bin/env python3
"""Freeze Phase 8J-Q2 inputs before any V2 validation result is inspected."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from ruamel.yaml import YAML


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
    path.parent.mkdir(parents=True, exist_ok=True)
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
        "reports/phase8i_score_semantics_audit.md",
        "reports/phase8i_score_label_scale_audit.json",
        "reports/phase8i_context_gap_analysis.json",
        "reports/phase8i_checkpoint_matrix.json",
        "reports/phase8i_estimated_cache_validation.json",
        "reports/phase8j_final_result.json",
        "reports/phase8j_coverage_gate.json",
        "reports/phase8jr_final_result.json",
        "reports/phase8jr_capacity_gate.json",
        "reports/phase8jr_capacity_oracle.json",
        "reports/phase8jp_final_result.json",
        "reports/phase8jp_capacity_gate.json",
        "reports/phase8jp_parameterization_comparison.json",
        "reports/phase8jq_final_result.json",
        "reports/phase8jq_final_taxonomy.json",
        "reports/phase8jq_final_recommendation.md",
        "reports/phase8jq_controller_authoritative_envelope.json",
        "reports/phase8jq_safety_geometry_audit.json",
        "reports/phase8jq_timeline_semantics_audit.json",
        "reports/phase8jq_failure_time_localization.json",
        "reports/phase8jq_independent_feasibility_solver.json",
        "reports/phase8jq_scenario_feasibility_audit.json",
        "reports/phase8jq_generator_contract.md",
        "reports/phase8jq_continuous_collision_audit.json",
        "reports/phase8jq_sensitivity_matrix.json",
        "loss/safety_loss.py",
        "loss/dynamic_safety_loss.py",
        "loss/trajectory_sampler.py",
        "tools/run_phase8i_failure_decomposition.py",
        "policy/dynamic_sequence_dataset.py",
        "policy/dynamic_collate.py",
        "configs/safety_evaluator_v2.yaml",
    ]
    missing = [name for name in required if not (ROOT / name).is_file()]
    phase8h = json.loads((REPORTS / "phase8h_final_perception_gate.json").read_text())
    lock = json.loads((REPORTS / ".phase8h_blind_once.lock").read_text())
    phase8jq = json.loads((REPORTS / "phase8jq_final_result.json").read_text())
    cache = json.loads((REPORTS / "phase8i_estimated_cache_validation.json").read_text())
    matrix = json.loads((REPORTS / "phase8i_checkpoint_matrix.json").read_text())
    index_path = Path(cache["index"])
    current = {
        "phase8h_implementation_sha256": lock["implementation_sha256"],
        "phase8h_config_sha256": lock["config_sha256"],
        "phase8h_evaluator_sha256": lock["evaluator_sha256"],
        "estimated_cache_index_sha256": sha256(index_path),
        "estimated_cache_index_content_sha256": cache["index_content_hash"],
        "production_manifest_sha256": sha256(
            ROOT / "data/phase8_dynamic_production/dataset_manifest.yaml"
        ),
        "static_map_catalog_sha256": sha256(ROOT / "configs/static_map_catalog.yaml"),
        "dynamic_map_catalog_sha256": sha256(
            ROOT / "data/phase8_dynamic_production/map_catalog.yaml"
        ),
        "risk_set_manifest_sha256": sha256(
            ROOT / "diagnostics/phase8c_production_risk_set_manifest.json"
        ),
        "phase8i_checkpoint_matrix_sha256": sha256(
            REPORTS / "phase8i_checkpoint_matrix.json"
        ),
        "v1_dynamic_evaluator_sha256": sha256(ROOT / "loss/dynamic_safety_loss.py"),
        "v1_static_evaluator_sha256": sha256(ROOT / "loss/safety_loss.py"),
        "v1_phase8i_report_sha256": sha256(
            REPORTS / "phase8i_candidate_failure_decomposition.json"
        ),
        "phase8jq_final_sha256": sha256(REPORTS / "phase8jq_final_result.json"),
        "v2_config_sha256": sha256(ROOT / "configs/safety_evaluator_v2.yaml"),
    }
    config = YAML(typ="safe").load(ROOT / "configs/safety_evaluator_v2.yaml")
    decision = next(
        row for row in matrix["checkpoints"]
        if row["name"] == config["gate"]["decision_checkpoint"]
    )
    common = {
        "evaluator_version": str(config["evaluator_version"]),
        "config_hash": current["v2_config_sha256"],
        "geometry_hash": sha256(ROOT / "loss/safety_geometry_v2.py"),
        "timeline_hash": sha256(ROOT / "policy/safety_evaluator_v2.py"),
        "uncertainty_policy_hash": canonical_hash({
            "policy": config["uncertainty"]["estimated_policy"],
            "confidence_sigma": config["uncertainty"]["confidence_sigma"],
            "propagate_in_evaluator": config["uncertainty"][
                "propagate_in_evaluator"
            ],
        }),
        "Simulator_geometry_hash": canonical_hash({
            "source": sha256(
                "/home/zjh/YOPO/Simulator/src/src/dynamic_actor.cpp"
            ),
            "config": sha256(
                "/home/zjh/YOPO/Simulator/src/config/config.yaml"
            ),
        }),
        "dataset_manifest_hash": current["production_manifest_sha256"],
        "cache_index_hash": current["estimated_cache_index_sha256"],
        "checkpoint_hash": decision["sha256"],
        "checkpoint_hashes": {
            row["name"]: row["sha256"] for row in matrix["checkpoints"]
            if row["analyzed_full_phase8i"]
        },
    }
    checks = {
        "all_required_inputs_present": not missing,
        "phase8h_perception_gate_pass": phase8h.get("status") == "PASS",
        "phase8h_blind_run_count_one": phase8h.get("phase8h_blind_run_count") == 1,
        "phase8jq_status_pass": phase8jq.get("status") == "PASS",
        "phase8jq_primary_cause": (
            phase8jq.get("primary_cause") == "evaluator_or_safety_semantics"
        ),
        "authorized_next_phase": (
            phase8jq.get("next_allowed_phase")
            == "phase8jq_v2_safety_semantics_rebaseline"
        ),
        "old_evaluator_unmodified_in_phase8jq": phase8jq.get("evaluator_modified") is False,
        "network_weights_unmodified": phase8jq.get("network_weights_modified") is False,
        "dataset_not_rebuilt": phase8jq.get("dataset_rebuilt") is False,
        "production_test_unused": phase8jq.get("production_test_used") is False,
        "cache_pass": cache.get("status") == "PASS",
        "cache_range_image_hybrid": cache.get("effective_foreground_mode") == "range_image_hybrid",
        "cache_strictly_causal": cache.get("strictly_causal") is True,
        "cache_index_hash_matches": current["estimated_cache_index_sha256"] == cache["index_sha256"],
        "cache_has_no_test": cache.get("production_test_generated") is False,
        "checkpoint_matrix_preserved": matrix.get("test_used_for_selection") is False,
    }
    report = {
        **common,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "phase": "8J-Q2",
        "checks": checks,
        "missing": missing,
        "frozen_hashes": current,
        "required_inputs": {
            name: {
                "sha256": sha256(ROOT / name) if (ROOT / name).is_file() else None,
                "bytes": (ROOT / name).stat().st_size if (ROOT / name).is_file() else None,
            }
            for name in required
        },
        "checkpoint_matrix_status": matrix["status"],
        "phase8b_checkpoint_missing_declared": matrix["phase8b"]["available"] is False,
        "production_test_used": False,
        "network_weights_modified": False,
    }
    atomic_json(REPORTS / "phase8jqv2_entry_gate.json", report)
    if report["status"] != "PASS":
        raise RuntimeError(f"Phase 8J-Q2 entry failed: {checks}, missing={missing}")

    phase8i = json.loads((REPORTS / "phase8i_final_result.json").read_text())
    phase8jq_taxonomy = json.loads((REPORTS / "phase8jq_final_taxonomy.json").read_text())
    phase8jr = json.loads((REPORTS / "phase8jr_capacity_gate.json").read_text())
    phase8jp = json.loads((REPORTS / "phase8jp_baseline_reproduction.json").read_text())
    baseline = {
        **common,
        "status": "PASS",
        "denominator_warning": (
            "Phase 8I/8J fractions use 2052 full validation windows; "
            "Phase 8J-Q taxonomy uses 142 parameterization-limited windows."
        ),
        "phase8i_full_validation_2052": {
            "dynamic_coverage_failure": 0.0892,
            "joint_coverage_failure": phase8i["joint_coverage_failure_fraction"],
            "label_inversion": 0.2232,
            "model_ranking_failure": 0.0877,
            "combined_selection_failure": phase8i["combined_selection_failure_fraction"],
            "predicted_label_spearman": 0.831,
        },
        "phase8j_c_k3_full_validation_2052": {
            "dynamic_coverage_failure_approx": 0.089,
            "joint_coverage_failure_approx": 0.108,
            "q10_joint_safe_candidate_count": 0,
        },
        "phase8jr_full_validation_2052": {
            "o1": phase8jr["valid_estimated"]["o1_direct_optimization"],
            "o2": phase8jr["valid_estimated"]["o2_dense_512"],
            "o3": phase8jr["valid_estimated"]["o3_temporal_bank"],
        },
        "phase8jp_full_validation_2052": {
            "unresolved_fraction": 142 / 2052,
            "unresolved_count": 142,
            "source_taxonomy": phase8jp["taxonomy"],
        },
        "phase8jq_subset_142": {
            "valid_estimated_counts": phase8jq_taxonomy["suites"]["valid_estimated"]["counts"],
            "strongly_supported_unavoidable": 0,
            "unresolved": 0,
        },
        "source_hashes": current,
    }
    atomic_json(REPORTS / "phase8jqv2_v1_baseline.json", baseline)
    print(json.dumps({
        "status": "PASS",
        "entry": str(REPORTS / "phase8jqv2_entry_gate.json"),
        "baseline": str(REPORTS / "phase8jqv2_v1_baseline.json"),
        "v2_config_sha256": current["v2_config_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
