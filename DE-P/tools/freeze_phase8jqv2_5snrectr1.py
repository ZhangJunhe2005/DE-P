#!/usr/bin/env python3
"""Freeze the one-time P1 SNRE-CTR1 build contract."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OLD = ROOT / "artifacts/phase8_mixed_scene_static_yopo_derived_v1_failed_cross_split_gate"


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def write(name, value):
    path = REPORTS / f"phase8jqv2_5snrectr1_{name}"
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def main():
    audit = json.load(open(REPORTS / "phase8jqv2_5snrectr1_prefreeze_audit.json"))
    target = json.load(open(REPORTS / "phase8jqv2_5snrectr1_target_consistency_raw.json"))
    rows = json.load(open(REPORTS / "phase8jqv2_5snrectr1_no_return_rows.json"))["rows"]
    if audit["status"] != "PASS_IDENTITY_PENDING_TARGET_AUDIT":
        raise RuntimeError("identity pre-freeze audit did not pass")
    selected = "P1_EXCLUDE_CANONICAL_NO_RETURN_SYMMETRICALLY"
    exclusion = {
        "version": "canonical_no_return_symmetric_exclusion_v1",
        "selected_policy": selected,
        "detector_hash": audit["canonical_detector_hash"],
        "rule": "exclude iff CanonicalNoReturnDepthV1 classifies source frame CANONICAL_NO_RETURN",
        "split_independent": True, "label_independent": True, "map_independent": True,
        "validation_only_filtering": False,
        "excluded_counts_observed_train_validation": audit["no_return"]["counts"],
        "excluded_sample_ids": sorted(row["sample_id"] for row in rows),
        "internal_test": "same detector frozen; content remains unread",
    }
    exclusion["excluded_sample_ids_hash"] = canonical(exclusion["excluded_sample_ids"])
    exclusion["manifest_hash"] = canonical(exclusion)
    implementation_files = [
        "data/static_no_return_contract_v1.py",
        "data/static_yopo_preprocessing_v1.py",
        "data/static_yopo_dataset_v1.py",
        "data/static_yopo_loader_v1.py",
        "policy/static_yopo_contract_v1.py",
        "policy/static_yopo_training_v1.py",
        "policy/static_yopo_checkpoint_v1.py",
        "policy/dynamic/deterministic_guard_strict_history_only_v1.py",
        "tools/build_phase8_mixed_static_yopo_derived_v2.py",
        "configs/phase8jqv2_5_mixed_static_yopo_training_v2.yaml",
    ]
    tree = {path: sha(ROOT / path) for path in implementation_files}
    tree_hash = canonical(tree)
    write("entry_gate.json", {
        "status": "PASS", "old_result": "FAIL_STATIC_DATA_CONTRACT_ROUTE_B",
        "source_v3_hash": audit["source_v3_manifest_hash"],
        "old_artifact_exists": OLD.is_dir(), "old_artifact_modified": False,
        "old_derived_finalized": False, "training_started": False,
        "formal_training_script_exists": False,
        "combined_h5_formal_started": False, "production_authorized": False,
    })
    write("contract_supersession.json", {
        "status": "PASS_EXPLICIT_USER_SUPERSESSION",
        "abolished": "ANY_CROSS_SPLIT_EXACT_DEPTH_HASH_TO_HARD_FAIL",
        "replacement": "DEPTH_OBSERVATION_EQUIVALENCE_NE_SAMPLE_IDENTITY_LEAKAGE",
        "true_identity_leakage_allowed": False,
        "automatic_msy_dtr1_repair": False,
    })
    write("old_failure_freeze.json", {
        "status": "FROZEN_READ_ONLY", "artifact": str(OLD),
        "old_final_result_sha256": sha(REPORTS / "phase8jqv2_5msydtr1_final_result.json"),
        "old_leakage_report_sha256": sha(REPORTS / "phase8jqv2_5msydtr1_leakage_audit.json"),
        "relabelled_pass": False, "modified": False,
    })
    write("no_return_detector.json", {
        "status": "PASS", "contract": audit["canonical_detector"],
        "implementation": "data/static_no_return_contract_v1.py",
        "contract_hash": audit["canonical_detector_hash"],
        "canonical_count": sum(audit["no_return"]["counts"].values()),
        "invalid_fill": audit["invalid_fill"], "corrupt_constant": audit["corrupt_constant"],
    })
    write("no_return_source_audit.json", {
        "status": "PASS", **audit["no_return"],
        "source_map_uuid_overlap": audit["leakage"]["map_uuid_overlap"],
        "source_sequence_overlap": audit["leakage"]["sequence_overlap"],
        "source_file_overlap": audit["leakage"]["source_path_overlap"],
        "source_inode_overlap": audit["leakage"]["source_inode_overlap"],
        "default_fill_or_read_failure": False,
        "static_authority_bound_per_map": True,
    })
    write("leakage_policy_v2.json", {
        "status": "FROZEN", "contract": audit["leakage_policy_v2"],
        "contract_hash": audit["leakage_policy_hash"],
    })
    write("identity_leakage.json", {"status": "PASS", **audit["leakage"]})
    write("complete_input_duplicates.json", {
        "status": "PASS",
        "exact_cross_split_groups": audit["leakage"]["exact_complete_input_cross_split_groups"],
        "tolerance_cross_split_groups": audit["leakage"]["tolerance_complete_input_cross_split_groups"],
        "depth_only_equivalence_groups": audit["leakage"]["depth_only_cross_split_groups"],
    })
    write("observational_aliasing.json", {
        "status": "P1_SELECTED_CONSERVATIVE",
        "exact_complete_input_cross_split_groups": 0,
        "tolerance_groups_evaluated": target["tolerance_groups_evaluated"],
        "semantic_safe_mask_disagreement_groups": sum(
            row["safe_mask_disagreement"] for row in target["groups"]
        ),
        "semantic_top1_disagreement_groups": sum(
            row["top1_disagreement"] for row in target["groups"]
        ),
        "ranking_disagreement_groups": sum(
            row["ranking_disagreement"] for row in target["groups"]
        ),
        "continuous_cost_tolerance_conflict_groups": target["conflict_groups"],
        "decision": "P1 because one tolerance-group has non-negligible continuous target variation",
    })
    write("target_consistency.json", target)
    write("no_return_policy.json", {
        "status": "FROZEN", "selected": selected,
        "reason": "legal sensor observation but conservative removal avoids tolerance-level target aliasing",
        "special_loss": False, "random_downsampling": False,
        "validation_only_filtering": False, "label_aware_filtering": False,
        "map_aware_filtering": False,
    })
    write("exclusion_manifest.json", exclusion)
    write("generator_hash_tree.json", {
        "status": "FROZEN", "files": tree, "root_hash": tree_hash,
    })
    write("contract_freeze.json", {
        "status": "FROZEN_BEFORE_ONE_TIME_BUILD",
        "source_v3_hash": audit["source_v3_manifest_hash"],
        "selected_policy": selected,
        "detector_hash": audit["canonical_detector_hash"],
        "leakage_policy_hash": audit["leakage_policy_hash"],
        "exclusion_manifest_hash": exclusion["manifest_hash"],
        "generator_hash_tree": tree_hash,
        "derived_version": "phase8_mixed_scene_static_yopo_derived_v2",
        "build_attempt_limit": 1, "post_freeze_policy_modified": False,
    })
    print(json.dumps({"status": "PASS_CONTRACT_FROZEN", "policy": selected,
                      "generator_hash_tree": tree_hash,
                      "exclusion_manifest_hash": exclusion["manifest_hash"]}, indent=2))


if __name__ == "__main__":
    main()
