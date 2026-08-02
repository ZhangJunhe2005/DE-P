#!/usr/bin/env python3
"""One-time P0 reference-view build for the frozen SMGSS split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.static_no_return_contract_v1 import (
    CanonicalNoReturnDepthV1, NoReturnClass, exact_complete_input_hash,
    normalize_observation_v1,
)
from data.static_yopo_manifest_v1 import (
    SOURCE_V3_MANIFEST_SHA256, assert_source_identity, canonical_hash,
    sample_identity as v2_sample_identity, sha256_file,
)
from data.static_yopo_preprocessing_v1 import DEFAULT_STATIC_YOPO_PREPROCESSOR_V1
from policy.static_yopo_contract_v1 import (
    LOSS_CONTRACT_HASH, MODEL_CONTRACT_HASH, NORMALIZATION_CONTRACT_HASH,
)
from tools.build_phase8_mixed_static_yopo_derived_v1 import INDEX_FIELDS, SOURCE, write_json

VERSION = "phase8_mixed_scene_static_yopo_derived_v3"
SCHEMA = "static_yopo_sample_v3_p1_map_group_split"
DEFAULT_OUTPUT = ROOT / "data" / VERSION
V2 = ROOT / "data/phase8_mixed_scene_static_yopo_derived_v2"
SPLIT_REPORT = ROOT / "reports/phase8jqv2_5smgsstr1_split_manifest.json"
NO_RETURN_HASH = "4d35e620a40043e03db2ae014c204faf098583893638e1d4d2198879f98221ba"
P1_POLICY_HASH = "edd1002b81d07ff8c767ac3231365ea72ea090cef24d4e6d99e7bba533cc3756"


def load_units():
    return [json.load(open(path)) for path in sorted(
        (SOURCE / "manifests/work_units").glob("*.json")
    )]


def sequence_root(row):
    return SOURCE / "raw_authoritative" / row["suite"] / row["source_split"] / row["sequence_id"]


def frames(row):
    return [json.loads(line) for line in open(sequence_root(row) / "frames.jsonl")]


def v3_identity(sequence, frame_index, map_uuid, depth_hash, observation,
                split_manifest_hash):
    observation_hash = hashlib.sha256(
        np.asarray(observation, dtype="<f4").tobytes()
    ).hexdigest()
    return canonical_hash({
        "source_v3_manifest_hash": SOURCE_V3_MANIFEST_SHA256,
        "source_sequence_id": sequence, "source_frame": int(frame_index),
        "map_uuid": map_uuid, "static_depth_hash": depth_hash,
        "observation_hash": observation_hash,
        "p1_policy_hash": P1_POLICY_HASH,
        "preprocessing_hash": DEFAULT_STATIC_YOPO_PREPROCESSOR_V1.contract_hash,
        "split_manifest_hash": split_manifest_hash, "schema_version": SCHEMA,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--authorize-one-time-build", action="store_true")
    args = parser.parse_args()
    if not args.authorize_one_time_build:
        raise SystemExit("--authorize-one-time-build is required")
    destination = args.output.expanduser().resolve()
    staging = destination.parent / f".{destination.name}.smgsstr1-staging"
    if destination.exists() or staging.exists():
        raise FileExistsError("one-time v3 destination/staging already exists")
    assert_source_identity(SOURCE)
    if sha256_file(V2 / "manifests/dataset_manifest.json") != (
        "749a075ce2d45a93ffd12c12bc4b30fc88d82c7678ca7d1ec09d778b5701e662"
    ):
        raise RuntimeError("derived v2 freeze mismatch")
    split_hash = sha256_file(SPLIT_REPORT)
    split_contract = json.load(open(SPLIT_REPORT))
    assignment = split_contract["assignment"]
    detector = CanonicalNoReturnDepthV1()
    units = load_units()

    def target_split(row):
        if row["split"] == "train":
            return "train"
        if row["split"] == "internal_test":
            return "internal_test"
        if row["split"] in {"calibration", "validation"}:
            return assignment[row["map_uuid"]]
        raise RuntimeError(f"unexpected source split {row['split']}")

    content_units = [row for row in units if target_split(row) != "internal_test"]
    counts, excluded = Counter(), Counter()
    for number, row in enumerate(content_units, 1):
        target = target_split(row)
        depth = None
        rows = frames(row)
        if len(rows) != 60:
            raise RuntimeError(f"partial source sequence {row['sequence_id']}")
        for frame in rows:
            counts[target] += 1
            if frame["static_depth_frame_hash"] == NO_RETURN_HASH:
                if depth is None:
                    depth = np.load(sequence_root(row) / "static_depth.npy", mmap_mode="r")
                classification = detector.classify(
                    depth[int(frame["frame_index"])], NO_RETURN_HASH
                )
                if classification != NoReturnClass.CANONICAL:
                    raise RuntimeError("P1 replay mismatch")
                excluded[target] += 1
        if number % 1000 == 0:
            print(json.dumps({"status": "PRECOUNT", "units": number,
                              "total": len(content_units)}), flush=True)
    retained = {split: counts[split] - excluded[split]
                for split in ("train", "calibration", "validation")}
    expected = {
        "train": 350813,
        "calibration": split_contract["calibration_samples"],
        "validation": split_contract["validation_samples"],
    }
    if retained != expected:
        raise RuntimeError(f"frozen assignment count mismatch {retained} != {expected}")
    staging.mkdir(parents=True)
    v2_authority = json.load(open(V2 / "manifests/map_authority.json"))
    write_json(staging / "manifests/map_authority.json", {
        "authority_source": "frozen V3 raw authority",
        "ply_wrapper_reused_read_only_from": str(V2),
        **v2_authority,
    })
    v2_catalog = YAML(typ="safe").load(V2 / "map_catalog.yaml")
    v2_catalog["catalog_version"] = 3
    v2_catalog["dataset_version"] = VERSION
    # The v2 catalog was produced in an atomic staging directory.  A published
    # reference view must never retain those ephemeral absolute paths.
    stable_v2_maps = (V2 / "maps").resolve()
    for entry in v2_catalog["maps"]:
        source = Path(entry["static_ply"])
        entry["static_ply"] = str(stable_v2_maps / source.name)
        if not Path(entry["static_ply"]).is_file():
            raise FileNotFoundError(
                f"stable read-only PLY wrapper is missing: {entry['static_ply']}"
            )
    with open(staging / "map_catalog.yaml", "w") as stream:
        YAML().dump(v2_catalog, stream)
    for entry in v2_authority["maps"]:
        source = Path(entry["static_ply"])
        entry["static_ply"] = str(stable_v2_maps / source.name)
    write_json(staging / "manifests/map_authority.json", {
        "authority_source": "frozen V3 raw authority",
        "ply_wrapper_reused_read_only_from": str(V2),
        **v2_authority,
    })
    map_rows = v2_authority["maps"]
    map_ids = {row["map_uuid"]: int(row["map_id"]) for row in map_rows}
    map_types = {row["map_uuid"]: row["map_type"] for row in map_rows}
    fields = dict(INDEX_FIELDS)
    fields["previous_v2_sample_id"] = ("S64", ())
    arrays = {}
    for split, count in retained.items():
        base = staging / "indices" / split
        base.mkdir(parents=True)
        arrays[split] = {
            name: open_memmap(base / f"{name}.npy", mode="w+", dtype=dtype,
                              shape=(count, *shape))
            for name, (dtype, shape) in fields.items()
        }
    cursors = Counter()
    identities, previous_identities = set(), set()
    map_splits, sequence_splits = defaultdict(set), defaultdict(set)
    source_paths, source_inodes = defaultdict(set), defaultdict(set)
    complete_splits, network_depth_splits = defaultdict(set), defaultdict(set)
    distributions = defaultdict(Counter)
    per_map_counts, sequence_counts = defaultdict(Counter), defaultdict(set)
    depth_stats = defaultdict(lambda: {"min": float("inf"), "max": float("-inf"),
                                       "finite": 0, "values": 0})
    for number, row in enumerate(content_units, 1):
        split, sequence, map_uuid = target_split(row), row["sequence_id"], row["map_uuid"]
        root = sequence_root(row)
        depth_path = root / "static_depth.npy"
        depth = np.load(depth_path, mmap_mode="r")
        stat = os.stat(depth_path)
        map_splits[map_uuid].add(split)
        sequence_splits[sequence].add(split)
        source_paths[split].add(str(depth_path.resolve()))
        source_inodes[split].add((stat.st_dev, stat.st_ino))
        for frame_index, frame in enumerate(frames(row)):
            if frame["static_depth_frame_hash"] == NO_RETURN_HASH:
                continue
            observation = np.asarray(
                frame["velocity_body"] + frame["acceleration_body"] + frame["goal_body"],
                dtype=np.float32,
            )
            raw = np.asarray(depth[frame_index])
            network = DEFAULT_STATIC_YOPO_PREPROCESSOR_V1(raw)
            network_hash = hashlib.sha256(network.tobytes()).hexdigest()
            complete_hash = exact_complete_input_hash(
                network_hash, normalize_observation_v1(observation)
            )
            network_depth_splits[network_hash].add(split)
            complete_splits[complete_hash].add(split)
            identity = v3_identity(
                sequence, frame_index, map_uuid, frame["static_depth_frame_hash"],
                observation, split_hash,
            )
            previous = v2_sample_identity(
                sequence, frame_index, map_uuid, frame["static_depth_frame_hash"], observation
            )
            if identity in identities or previous in previous_identities:
                raise RuntimeError("sample identity duplication")
            identities.add(identity)
            previous_identities.add(previous)
            index = cursors[split]
            values = {
                "sample_id": identity, "sequence_id": sequence,
                "frame_index": frame_index, "source_split": row["source_split"],
                "map_uuid": map_uuid, "suite": row["suite"],
                "depth_path": str(depth_path.resolve()),
                "depth_hash": frame["static_depth_frame_hash"],
                "observation": observation,
                "position_world": np.asarray(frame["position_world"], dtype=np.float32),
                "quaternion_xyzw": np.asarray(frame["quaternion_world_from_body"], dtype=np.float32),
                "map_id": map_ids[map_uuid], "previous_v2_sample_id": previous,
            }
            for field, value in values.items():
                arrays[split][field][index] = value
            cursors[split] += 1
            distributions[split][map_types[map_uuid]] += 1
            per_map_counts[split][map_uuid] += 1
            sequence_counts[split].add(sequence)
            finite = raw[np.isfinite(raw)]
            depth_stats[split]["finite"] += finite.size
            depth_stats[split]["values"] += raw.size
            depth_stats[split]["min"] = min(depth_stats[split]["min"], float(finite.min()))
            depth_stats[split]["max"] = max(depth_stats[split]["max"], float(finite.max()))
        if number % 500 == 0:
            print(json.dumps({"status": "INDEXING", "units": number,
                              "total": len(content_units)}), flush=True)
    for group in arrays.values():
        for array in group.values():
            array.flush()
    if dict(cursors) != expected:
        raise RuntimeError("final index count mismatch")
    map_leaks = {key: sorted(value) for key, value in map_splits.items() if len(value) > 1}
    sequence_leaks = {key: sorted(value) for key, value in sequence_splits.items() if len(value) > 1}
    path_overlap = set.intersection(*source_paths.values())
    inode_overlap = set.intersection(*source_inodes.values())
    complete_leaks = [key for key, value in complete_splits.items() if len(value) > 1]
    if map_leaks or sequence_leaks or path_overlap or inode_overlap or complete_leaks:
        raise RuntimeError("post-build split leakage Hard FAIL")
    internal = [row for row in units if target_split(row) == "internal_test"]
    write_json(staging / "sealed/internal_test_identity.json", {
        "content_accessed": False, "sequence_count": len(internal),
        "sample_count_before_p1_rule": sum(int(row["sample_count"]) for row in internal),
        "p1_policy_hash": P1_POLICY_HASH,
        "sequence_identities": [{
            "sequence_id": row["sequence_id"], "map_uuid": row["map_uuid"],
            "work_unit_hash": canonical_hash(row), "sample_count": row["sample_count"],
        } for row in internal],
    })
    split_manifest = {
        "version": "static_yopo_development_split_v3",
        "solver_split_manifest_sha256": split_hash,
        "assignment": assignment, "counts_before_p1": dict(counts),
        "p1_excluded": dict(excluded), "counts_after_p1": retained,
        "map_type_sample_counts": {key: dict(value) for key, value in distributions.items()},
        "per_map_sample_counts": {key: dict(value) for key, value in per_map_counts.items()},
        "sequence_counts": {key: len(value) for key, value in sequence_counts.items()},
        "map_leakage": 0, "sequence_leakage": 0, "source_path_overlap": 0,
        "source_inode_overlap": 0, "complete_input_cross_split_groups": 0,
        "train_membership": "UNCHANGED_FROM_DERIVED_V2",
        "internal_test_accessed": False,
    }
    write_json(staging / "manifests/split_manifest.json", split_manifest)
    duplicate = {
        "canonical_no_return_excluded": dict(excluded),
        "ordinary_network_depth_cross_split_groups": sum(
            len(value) > 1 for value in network_depth_splits.values()
        ),
        "ordinary_complete_input_cross_split_groups": 0,
        "sample_identity_overlap": 0, "provenance_overlap": 0,
    }
    write_json(staging / "manifests/duplicate_audit.json", duplicate)
    manifest = {
        "status": "COMPLETE_FROZEN", "dataset_version": VERSION,
        "schema_version": SCHEMA, "source_v3_manifest_hash": SOURCE_V3_MANIFEST_SHA256,
        "derived_v2_reference_hash": sha256_file(V2 / "manifests/dataset_manifest.json"),
        "data_route": "D1_V3_STATIC_ONLY", "representation": "P0_REFERENCE_VIEW",
        "no_return_policy": "P1_EXCLUDE_CANONICAL_NO_RETURN_SYMMETRICALLY",
        "p1_policy_hash": P1_POLICY_HASH,
        "canonical_detector_hash": "9954b6599a7b6bf22c2aa94456fa024beecd478d86615174d04522387702b793",
        "leakage_policy_hash": "f26a823d513aa73ce53959326cdc4dba018b4cf532285510241d37867178667f",
        "split_counts": retained, "excluded_counts": dict(excluded),
        "indexed_sample_count": sum(retained.values()),
        "solver_split_manifest_sha256": split_hash,
        "split_hash": sha256_file(staging / "manifests/split_manifest.json"),
        "duplicate_audit_hash": sha256_file(staging / "manifests/duplicate_audit.json"),
        "preprocessing_hash": DEFAULT_STATIC_YOPO_PREPROCESSOR_V1.contract_hash,
        "normalization_hash": NORMALIZATION_CONTRACT_HASH,
        "model_contract_hash": MODEL_CONTRACT_HASH, "loss_contract_hash": LOSS_CONTRACT_HASH,
        "map_authority_hash": sha256_file(staging / "manifests/map_authority.json"),
        "map_type_sample_counts": {key: dict(value) for key, value in distributions.items()},
        "depth_stats": dict(depth_stats),
        "source_depth": {"dtype": "float32", "unit": "m", "shape": [96, 160]},
        "static_depth_copied": False, "maps_regenerated": False,
        "source_modified": False, "actor_input_used": False, "composed_depth_used": False,
        "train_split_modified": False, "internal_test_accessed": False,
        "training_started": False,
    }
    write_json(staging / "manifests/dataset_manifest.json", manifest)
    manifest_hash = sha256_file(staging / "manifests/dataset_manifest.json")
    write_json(staging / "generation_state/BUILD_COMPLETE.json", {
        "status": "PASS", "manifest_sha256": manifest_hash,
        "build_attempt_count": 1,
    })
    os.replace(staging, destination)
    print(json.dumps({
        "status": "PASS", "output": str(destination),
        "manifest_sha256": manifest_hash, "split_counts": retained,
        "map_types": {key: sorted(value) for key, value in distributions.items()},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
