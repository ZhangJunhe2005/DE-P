#!/usr/bin/env python3
"""Finalize only the dataset semantic repair; never start Q2.5 evaluator."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/"data/phase8_authoritative_v1"
NEW = ROOT/"data/phase8_authoritative_v2"
REPORTS = ROOT/"reports"
OLD_HASH = "a81319d734871c48d49058e22a36a733baead3d85db2ec39206221d3b154f723"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")


def map_hashes(root):
    manifest = json.loads((root/"manifests/dataset_manifest.json").read_text())
    result = {}
    for record in manifest["maps"]:
        value = json.loads((root/record["path"]).read_text())
        result[(value["split"], value["map_uuid"])] = (
            value["authority_hash"], value["occupancy_hash"]
        )
    return result


def main():
    integrity = json.loads((
        REPORTS/"phase8jqv2_4_formal_generation_summary.json"
    ).read_text())
    semantics = json.loads((
        REPORTS/"phase8jqv2_4_dataset_semantic_validation.json"
    ).read_text())
    old_unchanged = sha256(
        OLD/"manifests/dataset_manifest.json"
    ) == OLD_HASH
    new_hash = sha256(NEW/"manifests/dataset_manifest.json")
    authority_equal = map_hashes(OLD) == map_hashes(NEW)
    checks = {
        "file_integrity": integrity["status"] == "PASS",
        "semantic_validation": semantics["status"] == "PASS",
        "train_frames": integrity["counts"]["train_frames"] == 1_000_000,
        "valid_frames": integrity["counts"]["valid_frames"] == 100_000,
        "unknown_zero": integrity["counts"]["unknown"] == 0,
        "loader_deterministic": integrity["loader_deterministic"],
        "authority_maps_identical": authority_equal,
        "old_dataset_unchanged": old_unchanged,
        "new_manifest_distinct": new_hash != OLD_HASH,
        "test_not_accessed": integrity["test_access_count"] == 0,
        "blind_not_accessed": integrity["blind_access_count"] == 0,
    }
    passed = all(checks.values())
    entry = {
        "status": "PASS" if passed else "FAIL",
        "phase": "phase8jqv2_5_entry_gate_rerun",
        "checks": checks,
        "old_dataset_version": "phase8_authoritative_v1",
        "old_root_manifest_hash": OLD_HASH,
        "new_dataset_version": "phase8_authoritative_v2",
        "new_root_manifest_hash": new_hash,
        "dataset_semantic_audit_complete": True,
        "q2_5_full_gate_complete": False,
        "coverage_training_started": False,
        "score_training_started": False,
        "optimizer_step_executed": False,
        "production_test_used": False,
        "blind_used": False,
        "network_weights_modified": False,
        "next_allowed_phase": (
            "phase8jqv2_5_safety_evaluator_v2_1_rebaseline"
            if passed else "phase8jqv2_4_state_semantics_repair"
        ),
    }
    write(REPORTS/"phase8jqv2_5_entry_gate_rerun.json", entry)
    final = {
        **entry,
        "status": entry["status"],
        "state_semantics_repair_complete": passed,
        "training_ready": False,
        "reason_training_not_ready":
            "Q2.5 evaluator/rebaseline/capacity gates are outside this phase",
    }
    write(REPORTS/"phase8jqv2_5_final_result_rerun.json", final)
    recommendation = (
        "# Phase 8J-Q2.5 entry rerun\n\n"
        f"**{entry['status']}** for the dataset state-semantics entry Gate.\n\n"
        "Safety Evaluator V2.1, R0/R1, capacity and training were not run. "
        f"Next allowed phase: `{entry['next_allowed_phase']}`.\n"
    )
    (REPORTS/"phase8jqv2_5_final_recommendation_rerun.md").write_text(
        recommendation
    )
    print(json.dumps(final, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
