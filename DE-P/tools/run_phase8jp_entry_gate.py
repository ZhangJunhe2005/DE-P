#!/usr/bin/env python3
"""Fail-closed Phase 8J-P entry and baseline reproduction."""

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


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def main():
    required_inputs = [
        "reports/phase8h_final_perception_gate.json",
        "reports/phase8h_final_readiness.md",
        "reports/phase8i_final_result.json",
        "reports/phase8i_final_recommendation.md",
        "reports/phase8i_candidate_failure_decomposition.json",
        "reports/phase8i_estimated_cache_validation.json",
        "reports/phase8j_final_result.json",
        "reports/phase8j_coverage_gate.json",
        "reports/phase8j_coverage_summary.json",
        "reports/phase8j_candidate_collapse_audit.md",
        "reports/phase8jr_final_result.json",
        "reports/phase8jr_final_readiness.md",
        "reports/phase8jr_capacity_gate.json",
        "reports/phase8jr_capacity_oracle.json",
        "reports/phase8jr_capacity_failure_taxonomy.md",
        "reports/phase8jr_temporal_candidate_analysis.json",
        "reports/phase8jr_proposal_design.md",
    ]
    phase8h = json.loads(
        (REPORTS / "phase8h_final_perception_gate.json").read_text()
    )
    phase8i = json.loads((REPORTS / "phase8i_final_result.json").read_text())
    phase8jr = json.loads((REPORTS / "phase8jr_final_result.json").read_text())
    capacity_gate_path = REPORTS / "phase8jr_capacity_gate.json"
    capacity_path = REPORTS / "phase8jr_capacity_oracle.json"
    capacity_gate = json.loads(capacity_gate_path.read_text())
    capacity = json.loads(capacity_path.read_text())
    prior_entry = json.loads(
        (REPORTS / "phase8jr_entry_gate.json").read_text()
    )
    cache = json.loads(
        (REPORTS / "phase8i_estimated_cache_validation.json").read_text()
    )
    current_hashes = {
        "cache_index_sha256": sha256(Path(cache["index"])),
        "dataset_manifest_sha256": sha256(
            ROOT / "data/phase8_dynamic_production/dataset_manifest.yaml"
        ),
        "static_map_catalog_sha256": sha256(
            ROOT / "configs/static_map_catalog.yaml"
        ),
        "dynamic_map_catalog_sha256": sha256(
            ROOT / "data/phase8_dynamic_production/map_catalog.yaml"
        ),
        "risk_set_manifest_sha256": sha256(
            ROOT / "diagnostics/phase8c_production_risk_set_manifest.json"
        ),
        "phase8jr_capacity_oracle_sha256": sha256(capacity_path),
        "phase8jr_capacity_gate_sha256": sha256(capacity_gate_path),
    }
    expected_data = prior_entry["data_hashes"]["current"]
    checks = {
        "all_required_inputs_present": all(
            (ROOT / path).is_file() for path in required_inputs
        ),
        "phase8h_perception_gate_pass": phase8h.get("status") == "PASS",
        "perception_frozen": (
            phase8i.get("perception_frozen") is True
            and phase8jr.get("perception_frozen") is True
        ),
        "phase8jr_capacity_gate_fail": capacity_gate.get("status") == "FAIL",
        "capacity_insufficient": (
            phase8jr.get("capacity_sufficient") is False
            and capacity_gate.get("capacity_sufficient") is False
        ),
        "training_not_executed": phase8jr.get("training_executed") is False,
        "score_not_executed": phase8jr.get("score_stage_executed") is False,
        "production_test_unused": phase8jr.get("production_test_used") is False,
        "cache_index_unchanged": (
            current_hashes["cache_index_sha256"]
            == cache["index_sha256"]
            == prior_entry["cache"]["index_sha256"]
        ),
        "cache_range_image_hybrid": (
            cache["effective_foreground_mode"] == "range_image_hybrid"
        ),
        "dataset_manifest_unchanged": (
            current_hashes["dataset_manifest_sha256"]
            == expected_data["production_dataset_manifest_sha256"]
        ),
        "static_catalog_unchanged": (
            current_hashes["static_map_catalog_sha256"]
            == expected_data["static_map_catalog_sha256"]
        ),
        "dynamic_catalog_unchanged": (
            current_hashes["dynamic_map_catalog_sha256"]
            == expected_data["dynamic_map_catalog_sha256"]
        ),
        "risk_manifest_unchanged": (
            current_hashes["risk_set_manifest_sha256"]
            == expected_data["risk_set_manifest_sha256"]
        ),
        "o0_o4_audit_complete": capacity.get("capacity_audit_complete") is True,
        "unresolved_count_142": (
            capacity["suites"]["valid_estimated"]["taxonomy"]["counts"].get(
                "parameterization_limited"
            ) == 142
        ),
    }
    failures = [key for key, passed in checks.items() if not passed]
    entry = {
        "status": "PASS" if not failures else "FAIL",
        "phase": "8J-P",
        "checks": checks,
        "hashes": current_hashes,
        "phase8h_frozen_hashes": prior_entry["frozen_hashes"],
        "entry_failures": failures,
        "required_inputs": {
            path: {
                "present": (ROOT / path).is_file(),
                "sha256": (
                    sha256(ROOT / path) if (ROOT / path).is_file() else None
                ),
            }
            for path in required_inputs
        },
        "network_training_allowed": False,
        "score_training_allowed": False,
        "production_test_allowed": False,
    }
    atomic_json(REPORTS / "phase8jp_entry_gate.json", entry)
    if failures:
        raise RuntimeError(f"Phase 8J-P entry Gate failed: {failures}")

    estimated = capacity["suites"]["valid_estimated"]
    gt = capacity["suites"]["valid_gt"]
    baseline = {
        "status": "PASS",
        "valid_estimated": {
            "o0": estimated["o0_current_15"][
                "joint_coverage_failure_fraction"
            ],
            "o1": estimated["o1_direct_optimization"][
                "joint_coverage_failure_fraction"
            ],
            "o2_dense_512": estimated["o2_dense_same_bound"][
                "joint_coverage_failure_fraction_by_count"
            ]["512"],
            "o3": estimated["o3_temporal_bank"][
                "joint_coverage_failure_fraction"
            ],
            "o4_diagnostic": estimated[
                "o4_extended_horizon_diagnostic"
            ]["joint_coverage_failure_fraction"],
        },
        "valid_gt": {
            "o0": gt["o0_current_15"]["joint_coverage_failure_fraction"],
            "o1": gt["o1_direct_optimization"][
                "joint_coverage_failure_fraction"
            ],
            "o2_dense_512": gt["o2_dense_same_bound"][
                "joint_coverage_failure_fraction_by_count"
            ]["512"],
            "o3": gt["o3_temporal_bank"][
                "joint_coverage_failure_fraction"
            ],
            "o4_diagnostic": gt[
                "o4_extended_horizon_diagnostic"
            ]["joint_coverage_failure_fraction"],
        },
        "taxonomy": estimated["taxonomy"]["counts"],
        "unresolved_not_intrinsically_classified": True,
        "source_capacity_oracle": str(capacity_path.resolve()),
        "source_capacity_oracle_sha256": sha256(capacity_path),
        "network_weights_modified": False,
        "production_test_used": False,
    }
    atomic_json(REPORTS / "phase8jp_baseline_reproduction.json", baseline)
    print(json.dumps({
        "status": "PASS",
        "entry": str(REPORTS / "phase8jp_entry_gate.json"),
        "baseline": str(REPORTS / "phase8jp_baseline_reproduction.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
