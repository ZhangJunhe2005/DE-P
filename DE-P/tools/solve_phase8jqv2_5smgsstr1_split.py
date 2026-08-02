#!/usr/bin/env python3
"""Deterministic structural-only map-group split solver for SMGSS-TR1."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
V2 = ROOT / "data/phase8_mixed_scene_static_yopo_derived_v2"
SOURCE = ROOT / "data/phase8_dynamic_evidence_formal_v3"
PREFIX = "phase8jqv2_5smgsstr1"
REQUIRED_TYPES = ("cave", "forest", "pillar", "room", "wall")
SOURCE_HASH = "a1acfa2fee7e8336917316cb3f07ac005c1ce1354a8bda8de10d02f2438354f6"
V2_HASH = "749a075ce2d45a93ffd12c12bc4b30fc88d82c7678ca7d1ec09d778b5701e662"
P1_POLICY = "P1_EXCLUDE_CANONICAL_NO_RETURN_SYMMETRICALLY"


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
    path = REPORTS / f"{PREFIX}_{name}"
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)
    return path


def decode(values):
    return [bytes(value).decode() for value in values]


def main():
    if sha(SOURCE / "manifests/dataset_manifest.json") != SOURCE_HASH:
        raise RuntimeError("source V3 changed")
    if sha(V2 / "manifests/dataset_manifest.json") != V2_HASH:
        raise RuntimeError("derived v2 changed")
    previous = json.load(open(REPORTS / "phase8jqv2_5snrectr1_final_result.json"))
    if previous["status"] != "FAIL_OBSERVATIONAL_ALIASING":
        raise RuntimeError("SNRE terminal Route D not frozen")
    v2_manifest = json.load(open(V2 / "manifests/dataset_manifest.json"))
    authority_rows = json.load(open(V2 / "manifests/map_authority.json"))["maps"]
    authority = {row["map_uuid"]: row for row in authority_rows}
    work_units = [
        json.load(open(path))
        for path in sorted((SOURCE / "manifests/work_units").glob("*.json"))
    ]
    group_units = defaultdict(list)
    for row in work_units:
        if row["split"] in {"calibration", "validation"}:
            group_units[row["map_uuid"]].append(row)
    post_p1_counts = Counter()
    post_p1_sequences = defaultdict(set)
    current_split = {}
    for split in ("calibration", "validation"):
        base = V2 / "indices" / split
        maps = decode(np.load(base / "map_uuid.npy", mmap_mode="r"))
        sequences = decode(np.load(base / "sequence_id.npy", mmap_mode="r"))
        for map_uuid, sequence in zip(maps, sequences):
            post_p1_counts[map_uuid] += 1
            post_p1_sequences[map_uuid].add(sequence)
            current_split[map_uuid] = split
    groups = []
    for map_uuid in sorted(group_units):
        row = authority[map_uuid]
        units = group_units[map_uuid]
        source_splits = {unit["source_split"] for unit in units}
        if source_splits != {"valid"}:
            raise RuntimeError("development pool contains non-valid source")
        if map_uuid not in current_split:
            raise RuntimeError(f"P1 removed an entire development map group: {map_uuid}")
        groups.append({
            "map_uuid": map_uuid,
            "map_seed": row["map_seed"],
            "map_type": row["map_type"],
            "source_split": "valid",
            "current_split": current_split[map_uuid],
            "post_p1_sample_count": post_p1_counts[map_uuid],
            "source_sequence_count": len(units),
            "post_p1_sequence_count": len(post_p1_sequences[map_uuid]),
            "source_sequences": sorted(unit["sequence_id"] for unit in units),
            "source_work_unit_hashes": sorted(canonical(unit) for unit in units),
            "camera_trajectory_family": sorted(unit["sequence_id"] for unit in units),
            "renderer_seed_family": sorted(unit["sequence_id"] for unit in units),
            "static_authority_hash": row["authority_manifest_hash"],
        })
    if len(groups) > 20:
        raise RuntimeError("bounded exact solver expected at most 20 map groups")
    feasible = []
    for bits in itertools.product((0, 1), repeat=len(groups)):
        assignment = {
            group["map_uuid"]: ("validation" if bit else "calibration")
            for group, bit in zip(groups, bits)
        }
        calibration = [group for group in groups if assignment[group["map_uuid"]] == "calibration"]
        validation = [group for group in groups if assignment[group["map_uuid"]] == "validation"]
        if not calibration or not validation:
            continue
        validation_types = {group["map_type"] for group in validation}
        if validation_types != set(REQUIRED_TYPES):
            continue
        if any(group["post_p1_sample_count"] <= 0 or group["post_p1_sequence_count"] <= 0
               for group in validation):
            continue
        movement = sum(
            assignment[group["map_uuid"]] != group["current_split"] for group in groups
        )
        validation_count = sum(group["post_p1_sample_count"] for group in validation)
        per_type_samples = {
            map_type: sum(group["post_p1_sample_count"] for group in validation
                          if group["map_type"] == map_type)
            for map_type in REQUIRED_TYPES
        }
        balance_range = max(per_type_samples.values()) - min(per_type_samples.values())
        per_type_maps = {
            map_type: sum(group["map_type"] == map_type for group in validation)
            for map_type in REQUIRED_TYPES
        }
        double_covered = sum(count >= 2 for count in per_type_maps.values())
        lexical = tuple(
            assignment[group["map_uuid"]] for group in sorted(groups, key=lambda x: x["map_uuid"])
        )
        objective = (
            movement, abs(validation_count - 23759), balance_range,
            -double_covered, lexical,
        )
        feasible.append({
            "assignment": assignment, "objective": objective,
            "validation_samples": validation_count,
            "calibration_samples": sum(group["post_p1_sample_count"] for group in calibration),
            "validation_type_samples": per_type_samples,
            "validation_type_maps": per_type_maps,
            "moved_map_uuids": sorted(
                group["map_uuid"] for group in groups
                if assignment[group["map_uuid"]] != group["current_split"]
            ),
        })
    if not feasible:
        write("split_feasibility.json", {
            "status": "FAIL_NO_FEASIBLE_MAP_GROUP_SPLIT",
            "candidate_assignment_count": 2 ** len(groups),
            "feasible_assignment_count": 0,
        })
        raise SystemExit(2)
    feasible.sort(key=lambda row: tuple(row["objective"]))
    chosen = feasible[0]
    tied = sum(tuple(row["objective"]) == tuple(chosen["objective"]) for row in feasible)
    manifest = {
        "version": "static_yopo_development_split_v3",
        "pool": "DEVELOPMENT_EVALUATION_POOL_V1",
        "source_v3_manifest_hash": SOURCE_HASH,
        "derived_v2_manifest_hash": V2_HASH,
        "p1_policy": P1_POLICY,
        "canonical_detector_hash": v2_manifest["canonical_detector_hash"],
        "leakage_policy_hash": v2_manifest["leakage_policy_hash"],
        "solver": "deterministic_exhaustive_map_group_solver_v1",
        "objective_priority": [
            "movement_count", "validation_sample_delta_from_23759",
            "validation_type_sample_range", "negative_double_covered_type_count",
            "map_uuid_lexical_assignment",
        ],
        "objective": chosen["objective"],
        "assignment": chosen["assignment"],
        "moved_map_uuids": chosen["moved_map_uuids"],
        "train": "UNCHANGED_RAW_TRAIN_P1",
        "internal_test": "SEALED_UNREAD_P1_RULE_ONLY",
        "calibration_samples": chosen["calibration_samples"],
        "validation_samples": chosen["validation_samples"],
        "validation_type_samples": chosen["validation_type_samples"],
        "validation_type_maps": chosen["validation_type_maps"],
        "groups": groups,
    }
    split_path = write("split_manifest.json", manifest)
    split_hash = sha(split_path)
    before_distribution = v2_manifest["map_type_sample_counts"]
    after_calibration = Counter()
    after_validation = Counter()
    for group in groups:
        target = chosen["assignment"][group["map_uuid"]]
        (after_validation if target == "validation" else after_calibration)[
            group["map_type"]
        ] += group["post_p1_sample_count"]
    write("entry_gate.json", {
        "status": "PASS", "previous_status": previous["status"],
        "previous_route": previous["route"], "v2_hash": V2_HASH,
        "source_hash": SOURCE_HASH, "v2_modified": False,
        "train_split_modified": False, "internal_test_accessed": False,
        "training_started": False,
    })
    write("explicit_supersession.json", {
        "status": "PASS_EXPLICIT_USER_SUPERSESSION",
        "abolished": "fixed raw calibration/validation to derived calibration/validation mapping",
        "replacement": "DEVELOPMENT_EVALUATION_POOL_V1 atomic map-group repartition",
        "automatic_snre_repair": False, "supersession_count": 1,
    })
    write("v2_failure_freeze.json", {
        "status": "FROZEN_READ_ONLY", "manifest_hash": V2_HASH,
        "promoted": False, "modified": False,
    })
    write("source_freeze.json", {
        "status": "PASS", "source_hash": SOURCE_HASH, "source_modified": False,
        "internal_test_accessed": False,
    })
    write("development_pool.json", {
        "status": "PASS", "pool_version": "DEVELOPMENT_EVALUATION_POOL_V1",
        "map_group_count": len(groups),
        "source_splits": ["raw calibration", "raw validation"],
        "raw_train_included": False, "internal_test_included": False,
        "samples_after_p1": sum(group["post_p1_sample_count"] for group in groups),
    })
    write("map_group_inventory.json", {
        "status": "PASS", "group_key": "map_uuid", "groups": groups,
    })
    write("split_feasibility.json", {
        "status": "PASS", "candidate_assignment_count": 2 ** len(groups),
        "feasible_assignment_count": len(feasible),
        "selected_objective": chosen["objective"],
        "selected_tie_count": tied, "unique_assignment": tied == 1,
        "hard_constraints": "PASS",
    })
    write("candidate_assignments.json", {
        "status": "PASS_BOUNDED_EXACT_ENUMERATION",
        "candidate_assignment_count": 2 ** len(groups),
        "feasible_assignment_count": len(feasible),
        "top_candidates": feasible[:min(20, len(feasible))],
    })
    write("selected_map_group_assignment.json", {
        "status": "FROZEN", **chosen,
        "before_map_distribution": before_distribution,
        "after_map_distribution": {
            "train": before_distribution["train"],
            "calibration": dict(after_calibration),
            "validation": dict(after_validation),
        },
        "split_manifest_sha256": split_hash,
    })
    code_files = [
        "tools/solve_phase8jqv2_5smgsstr1_split.py",
        "data/static_no_return_contract_v1.py",
        "data/static_yopo_preprocessing_v1.py",
        "policy/static_yopo_contract_v1.py",
        "policy/static_yopo_training_v1.py",
        "policy/static_yopo_checkpoint_v1.py",
    ]
    code_hashes = {path: sha(ROOT / path) for path in code_files}
    write("contract_freeze.json", {
        "status": "FROZEN_BEFORE_DERIVED_V3_BUILD",
        "split_manifest_sha256": split_hash,
        "assignment": chosen["assignment"],
        "source_hash": SOURCE_HASH, "v2_hash": V2_HASH,
        "p1_policy": P1_POLICY,
        "canonical_detector_hash": v2_manifest["canonical_detector_hash"],
        "leakage_policy_hash": v2_manifest["leakage_policy_hash"],
        "preprocessing_hash": v2_manifest["preprocessing_hash"],
        "normalization_hash": v2_manifest["normalization_hash"],
        "model_contract_hash": v2_manifest["model_contract_hash"],
        "loss_contract_hash": v2_manifest["loss_contract_hash"],
        "code_hashes": code_hashes,
        "derived_v3_build_attempt_limit": 1,
    })
    print(json.dumps({
        "status": "PASS_SPLIT_FROZEN", "groups": len(groups),
        "feasible_assignments": len(feasible), "selected": chosen,
        "split_manifest_sha256": split_hash,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
