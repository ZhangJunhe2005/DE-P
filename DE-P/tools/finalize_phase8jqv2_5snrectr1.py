#!/usr/bin/env python3
"""Terminal SNRE-CTR1 reporting after the one-time post-build Gate."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DERIVED = ROOT / "data/phase8_mixed_scene_static_yopo_derived_v2"


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(name, value):
    path = REPORTS / f"phase8jqv2_5snrectr1_{name}"
    temporary = path.with_suffix(path.suffix + ".tmp")
    if isinstance(value, str):
        temporary.write_text(value)
    else:
        with open(temporary, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
    os.replace(temporary, path)


def main():
    manifest_path = DERIVED / "manifests/dataset_manifest.json"
    manifest = json.load(open(manifest_path))
    split = json.load(open(DERIVED / "manifests/split_manifest.json"))
    duplicate = json.load(open(DERIVED / "manifests/duplicate_audit.json"))
    map_counts = manifest["map_type_sample_counts"]
    required = {"cave", "forest", "pillar", "room", "wall"}
    train_types = set(map_counts["train"])
    validation_types = set(map_counts["validation"])
    missing_train = sorted(required - train_types)
    missing_validation = sorted(required - validation_types)
    if missing_validation != ["pillar"]:
        raise RuntimeError(f"unexpected map coverage result: {missing_validation}")
    common = {
        "terminal_gate": "P1_POST_EXCLUSION_TRAIN_VALIDATION_FIVE_MAP_TYPES",
        "status": "NOT_RUN_DUE_TO_TERMINAL_DATA_GATE",
        "long_training_started": False,
        "internal_test_accessed": False,
    }
    write("dataset_schema.json", {
        "status": "PASS", "schema_version": manifest["schema_version"],
        "representation": manifest["representation"],
        "data_route": manifest["data_route"], "static_depth_only": True,
        "actor_input_used": False, "composed_depth_used": False,
    })
    write("split_manifest.json", {"status": "PASS_IDENTITY_GATES", **split})
    write("dataset_hash_tree.json", {
        "status": "PASS", "dataset_manifest_sha256": sha(manifest_path),
        "split_manifest_sha256": sha(DERIVED / "manifests/split_manifest.json"),
        "duplicate_audit_sha256": sha(DERIVED / "manifests/duplicate_audit.json"),
        "map_authority_sha256": sha(DERIVED / "manifests/map_authority.json"),
        "source_v3_sha256": manifest["source_v3_manifest_hash"],
    })
    write("duplicate_audit.json", {"status": "PASS_LEAKAGE_POLICY_V2", **duplicate})
    write("map_distribution.json", {
        "status": "FAIL_VALIDATION_MAP_TYPE_COVERAGE",
        "sample_counts": map_counts,
        "required_map_types": sorted(required),
        "train_missing": missing_train,
        "validation_missing": missing_validation,
        "validation_pillar_samples": 0,
        "filter_changed_split": False,
        "post_freeze_split_repair_permitted": False,
    })
    before = {"train": 360000, "calibration": 25680, "validation": 25140}
    write("no_return_distribution.json", {
        "status": "PASS_DIAGNOSTIC",
        "policy": manifest["no_return_policy"],
        "counts_before": before, "excluded": manifest["excluded_counts"],
        "ratios_before": {
            key: manifest["excluded_counts"][key] / before[key] for key in before
        },
        "counts_after": manifest["split_counts"],
        "dominates_any_split": False,
    })
    for name in (
        "loader_validation.json", "backward_smoke.json", "optimizer_shakedown.json",
        "checkpoint_resume.json", "training_script_verify.json",
        "combined_h5_dryrun.json",
    ):
        write(name, dict(common))
    write("dataset_finalization.json", {
        "status": "BUILT_BUT_NOT_TRAINING_ELIGIBLE",
        "derived_dataset": str(DERIVED),
        "manifest_sha256": sha(manifest_path),
        "build_attempt_count": 1, "second_build_started": False,
        "policy_modified_after_freeze": False,
        "formal_training_authorized": False,
        "reason": "validation has zero pillar samples after frozen P1 build",
    })
    final = {
        "status": "FAIL_OBSERVATIONAL_ALIASING", "route": "D",
        "selected_no_return_policy": "P1_EXCLUDE_CANONICAL_NO_RETURN_SYMMETRICALLY",
        "p0_rejected_reason": "one tolerance group exceeded frozen continuous target-cost tolerance",
        "p1_rejected_reason": "post-build validation lacks pillar map type",
        "derived_dataset": "phase8_mixed_scene_static_yopo_derived_v2",
        "derived_manifest_sha256": sha(manifest_path),
        "identity_leakage": 0, "source_group_leakage": 0,
        "complete_input_cross_split_groups": 0,
        "formal_training_authorized": False,
        "training_script_created": False, "long_training_started": False,
        "combined_h5_formal_started": False, "production_qualified": False,
        "production_activation_authorized": False,
        "build_attempt_count": 1, "third_derived_build_authorized": False,
        "next_allowed_phase": None,
    }
    write("final_result.json", final)
    write("final_readiness.md", """# SNRE-CTR1 final readiness

Final status: `FAIL_OBSERVATIONAL_ALIASING` (terminal Route D).

The old depth-only Hard FAIL was correctly superseded. All 10,568 repeated
20 m frames were canonical no-return observations, with zero invalid/corrupt
fill, zero source/map/sequence/sample/path/inode leakage, and zero exact or
tolerance complete-input overlap across train and validation.

The frozen P1 policy symmetrically excluded canonical no-return and the one-time
derived v2 build completed: train 350,813; calibration 25,362; validation
23,759. Ordinary complete-input and provenance leakage remained zero.

The post-build P1 Gate failed because validation contains no pillar samples.
The contract requires all five map types in both train and validation and
forbids a post-freeze split change or another build. Therefore loader/backward,
checkpoint, training launcher, and combined-H5 dry-run were not executed.
No training or production activation occurred.
""")
    write("final_recommendation.md", """# SNRE-CTR1 terminal recommendation

Do not train from derived v2 under this stage. No authorized launcher was
created.

SNRE-CTR1 permits no CTR2, split repair, third derived build, or policy change.
Any future continuation requires a new explicit user contract that supersedes
the map-type coverage/split boundary; it cannot be described as an automatic
SNRE-CTR1 repair.
""")
    print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
