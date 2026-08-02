#!/usr/bin/env python3
"""One-time SNRE-CTR1 P1 build from frozen V3 static_depth only."""

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
    CanonicalNoReturnDepthV1, NoReturnClass, StaticDatasetLeakagePolicyV2,
    contract_hash, exact_complete_input_hash, normalize_observation_v1,
)
from data.static_yopo_manifest_v1 import (
    SOURCE_V3_MANIFEST_SHA256, assert_source_identity, canonical_hash,
    sample_identity, sha256_file,
)
from data.static_yopo_preprocessing_v1 import DEFAULT_STATIC_YOPO_PREPROCESSOR_V1
from policy.static_yopo_contract_v1 import (
    LOSS_CONTRACT_HASH, MODEL_CONTRACT_HASH, NORMALIZATION_CONTRACT_HASH,
)
from tools.build_phase8_mixed_static_yopo_derived_v1 import (
    INDEX_FIELDS, MAP_TYPE, SOURCE, _write_authority_ply, load_work_units, write_json,
)

VERSION = "phase8_mixed_scene_static_yopo_derived_v2"
DEFAULT_OUTPUT = ROOT / "data" / VERSION
NO_RETURN_HASH = "4d35e620a40043e03db2ae014c204faf098583893638e1d4d2198879f98221ba"


def authority_maps(output):
    rows, numeric = [], {}
    for source_split in ("train", "valid"):
        base = SOURCE / "raw_authoritative/geometry_authority" / source_split
        for map_dir in sorted(path for path in base.iterdir() if path.is_dir()):
            metadata = json.load(open(map_dir / "occupancy_metadata.json"))
            provenance = json.load(open(map_dir / "mixed_scene_provenance.json"))
            uuid = metadata["map_uuid"]
            map_id = len(numeric)
            numeric[uuid] = map_id
            ply = output / "maps" / f"pointcloud-{map_id}.ply"
            _write_authority_ply(map_dir / "raw_cloud.bin", ply, metadata["source_point_count"])
            rows.append({
                "map_id": map_id, "map_uuid": uuid, "source_split": source_split,
                "map_type": MAP_TYPE[int(provenance["maze_type"])],
                "map_seed": provenance["seed"], "profile_name": provenance["profile_name"],
                "authority_manifest": str((map_dir / "authority_manifest.json").resolve()),
                "authority_manifest_hash": sha256_file(map_dir / "authority_manifest.json"),
                "occupancy": str((map_dir / "occupancy.bin").resolve()),
                "occupancy_hash": metadata["occupancy_hash"],
                "raw_cloud": str((map_dir / "raw_cloud.bin").resolve()),
                "raw_cloud_hash": metadata["source_cloud_hash"],
                "static_ply": str(ply.resolve()), "static_ply_hash": sha256_file(ply),
                "resolution_m": metadata["occupancy_resolution_m"],
                "frame_id": metadata["frame_id"], "uav_radius_m": metadata["uav_radius_m"],
            })
    write_json(output / "manifests/map_authority.json", {"maps": rows})
    yaml = YAML()
    with open(output / "map_catalog.yaml", "w") as stream:
        yaml.dump({
            "catalog_version": 2, "dataset_version": VERSION,
            "authority": "V3 SGARAW1 payload exact binary PLY wrapping",
            "maps": [{"map_id": row["map_id"], "map_uuid": row["map_uuid"],
                      "static_ply": row["static_ply"],
                      "static_map_sha256": row["static_ply_hash"]} for row in rows],
        }, stream)
    return rows, numeric


def create_arrays(output, split, count):
    base = output / "indices" / split
    base.mkdir(parents=True, exist_ok=True)
    return {
        name: open_memmap(base / f"{name}.npy", mode="w+", dtype=dtype,
                          shape=(count, *shape))
        for name, (dtype, shape) in INDEX_FIELDS.items()
    }


def sequence_root(row):
    return SOURCE / "raw_authoritative" / row["suite"] / row["source_split"] / row["sequence_id"]


def frame_rows(row):
    return [json.loads(line) for line in open(sequence_root(row) / "frames.jsonl")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--authorize-one-time-build", action="store_true")
    args = parser.parse_args()
    if not args.authorize_one_time_build:
        raise SystemExit("--authorize-one-time-build is required")
    destination = args.output.expanduser().resolve()
    staging = destination.parent / f".{destination.name}.snrectr1-staging"
    if destination.exists() or staging.exists():
        raise FileExistsError("one-time v2 destination/staging already exists")
    assert_source_identity(SOURCE)
    detector = CanonicalNoReturnDepthV1()
    policy = StaticDatasetLeakagePolicyV2()
    units = load_work_units()
    content_splits = {"train", "calibration", "validation"}
    selected_units = [row for row in units if row["split"] in content_splits]
    counts, excluded, classification = Counter(), Counter(), Counter()
    # Pre-count is part of this single build and fixes allocation before any output.
    for number, row in enumerate(selected_units, 1):
        frames = frame_rows(row)
        if len(frames) != 60:
            raise RuntimeError(f"partial source sequence: {row['sequence_id']}")
        depth = None
        for frame in frames:
            counts[row["split"]] += 1
            if frame["static_depth_frame_hash"] == NO_RETURN_HASH:
                if depth is None:
                    depth = np.load(sequence_root(row) / "static_depth.npy", mmap_mode="r")
                cls = detector.classify(
                    depth[int(frame["frame_index"])], frame["static_depth_frame_hash"]
                )
                classification[cls] += 1
                if cls != NoReturnClass.CANONICAL:
                    raise RuntimeError(f"P1 detector rejected {row['sequence_id']}")
                excluded[row["split"]] += 1
        if number % 1000 == 0:
            print(json.dumps({"status": "PRECOUNT", "units": number,
                              "total": len(selected_units)}), flush=True)
    retained = {split: counts[split] - excluded[split] for split in content_splits}
    staging.mkdir(parents=True)
    maps, map_ids = authority_maps(staging)
    arrays = {split: create_arrays(staging, split, retained[split])
              for split in content_splits}
    cursors = Counter()
    sample_ids = set()
    map_splits, sequence_splits = defaultdict(set), defaultdict(set)
    processed_depth_splits, complete_splits = defaultdict(set), defaultdict(set)
    map_type_counts = defaultdict(Counter)
    no_return_map_counts = defaultdict(Counter)
    stats = defaultdict(lambda: {"min": float("inf"), "max": float("-inf"),
                                 "finite": 0, "values": 0})
    map_lookup = {row["map_uuid"]: row for row in maps}
    for number, row in enumerate(selected_units, 1):
        split, sequence = row["split"], row["sequence_id"]
        root = sequence_root(row)
        depth = np.load(root / "static_depth.npy", mmap_mode="r")
        frames = frame_rows(row)
        map_uuid = row["map_uuid"]
        map_splits[map_uuid].add(split)
        sequence_splits[sequence].add(split)
        map_type = map_lookup[map_uuid]["map_type"]
        for frame_index, frame in enumerate(frames):
            if frame["static_depth_frame_hash"] == NO_RETURN_HASH:
                no_return_map_counts[split][map_uuid] += 1
                continue
            observation = np.asarray(
                frame["velocity_body"] + frame["acceleration_body"] + frame["goal_body"],
                dtype=np.float32,
            )
            raw = np.asarray(depth[frame_index])
            network = DEFAULT_STATIC_YOPO_PREPROCESSOR_V1(raw)
            network_hash = hashlib.sha256(network.tobytes()).hexdigest()
            normalized = normalize_observation_v1(observation)
            complete_hash = exact_complete_input_hash(network_hash, normalized)
            processed_depth_splits[network_hash].add(split)
            complete_splits[complete_hash].add(split)
            identity = sample_identity(
                sequence, frame_index, map_uuid, frame["static_depth_frame_hash"],
                observation,
            )
            if identity in sample_ids:
                raise RuntimeError("duplicate sample identity")
            sample_ids.add(identity)
            index = cursors[split]
            values = {
                "sample_id": identity, "sequence_id": sequence,
                "frame_index": frame_index, "source_split": row["source_split"],
                "map_uuid": map_uuid, "suite": row["suite"],
                "depth_path": str((root / "static_depth.npy").resolve()),
                "depth_hash": frame["static_depth_frame_hash"],
                "observation": observation,
                "position_world": np.asarray(frame["position_world"], dtype=np.float32),
                "quaternion_xyzw": np.asarray(frame["quaternion_world_from_body"], dtype=np.float32),
                "map_id": map_ids[map_uuid],
            }
            for field, value in values.items():
                arrays[split][field][index] = value
            cursors[split] += 1
            map_type_counts[split][map_type] += 1
            finite = raw[np.isfinite(raw)]
            stats[split]["finite"] += finite.size
            stats[split]["values"] += raw.size
            stats[split]["min"] = min(stats[split]["min"], float(finite.min()))
            stats[split]["max"] = max(stats[split]["max"], float(finite.max()))
        if number % 500 == 0:
            print(json.dumps({"status": "INDEXING", "units": number,
                              "total": len(selected_units)}), flush=True)
    for group in arrays.values():
        for array in group.values():
            array.flush()
    if dict(cursors) != retained:
        raise RuntimeError(f"retained count mismatch: {dict(cursors)} != {retained}")
    map_leaks = {key: sorted(value) for key, value in map_splits.items() if len(value) > 1}
    sequence_leaks = {key: sorted(value) for key, value in sequence_splits.items() if len(value) > 1}
    complete_leaks = [key for key, value in complete_splits.items() if len(value) > 1]
    if map_leaks or sequence_leaks or complete_leaks:
        raise RuntimeError("LeakagePolicyV2 Hard FAIL")
    internal = [row for row in units if row["split"] == "internal_test"]
    sealed = {
        "content_accessed": False, "sample_count_before_frozen_rule": sum(
            int(row["sample_count"]) for row in internal
        ), "sequence_count": len(internal),
        "frozen_filter": detector.contract(),
        "policy": "P1_EXCLUDE_CANONICAL_NO_RETURN_SYMMETRICALLY",
        "sequence_identities": [{
            "sequence_id": row["sequence_id"], "map_uuid": row["map_uuid"],
            "work_unit_hash": canonical_hash(row), "sample_count": row["sample_count"],
        } for row in internal],
    }
    write_json(staging / "sealed/internal_test_identity.json", sealed)
    split_manifest = {
        "policy": "P1_EXCLUDE_CANONICAL_NO_RETURN_SYMMETRICALLY",
        "counts_before": dict(counts), "excluded": dict(excluded),
        "counts_after": retained, "internal_test_content_accessed": False,
        "map_leakage": 0, "sequence_leakage": 0,
        "complete_input_cross_split_groups": 0,
    }
    write_json(staging / "manifests/split_manifest.json", split_manifest)
    duplicate = {
        "canonical_no_return_depth_only_group": NO_RETURN_HASH,
        "canonical_excluded": dict(excluded),
        "ordinary_network_depth_cross_split_groups": sum(
            len(value) > 1 for value in processed_depth_splits.values()
        ),
        "ordinary_complete_input_cross_split_groups": 0,
        "identity_leakage": 0, "provenance_leakage": 0,
    }
    write_json(staging / "manifests/duplicate_audit.json", duplicate)
    manifest = {
        "status": "COMPLETE_FROZEN", "dataset_version": VERSION,
        "schema_version": "static_yopo_sample_v2_p1",
        "source_v3_manifest_hash": SOURCE_V3_MANIFEST_SHA256,
        "data_route": "D1_V3_STATIC_ONLY", "representation": "P0_REFERENCE_VIEW",
        "no_return_policy": "P1_EXCLUDE_CANONICAL_NO_RETURN_SYMMETRICALLY",
        "canonical_detector_hash": contract_hash(detector.contract()),
        "leakage_policy_hash": contract_hash(policy.contract()),
        "split_counts": retained, "excluded_counts": dict(excluded),
        "indexed_sample_count": sum(retained.values()),
        "internal_test_rule_frozen_content_unread": True,
        "split_hash": sha256_file(staging / "manifests/split_manifest.json"),
        "duplicate_audit_hash": sha256_file(staging / "manifests/duplicate_audit.json"),
        "preprocessing_hash": DEFAULT_STATIC_YOPO_PREPROCESSOR_V1.contract_hash,
        "normalization_hash": NORMALIZATION_CONTRACT_HASH,
        "model_contract_hash": MODEL_CONTRACT_HASH,
        "loss_contract_hash": LOSS_CONTRACT_HASH,
        "map_authority_hash": sha256_file(staging / "manifests/map_authority.json"),
        "map_type_sample_counts": {key: dict(value) for key, value in map_type_counts.items()},
        "no_return_map_counts_before_exclusion": {
            key: dict(value) for key, value in no_return_map_counts.items()
        },
        "depth_stats_after_exclusion": dict(stats),
        "source_depth": {"dtype": "float32", "unit": "m", "shape": [96, 160]},
        "source_mutated": False, "depth_copied": False,
        "composed_depth_used": False, "actor_input_used": False,
        "validation_only_filtering": False, "label_aware_filtering": False,
        "map_aware_filtering": False, "internal_test_accessed": False,
        "training_started": False,
    }
    write_json(staging / "manifests/dataset_manifest.json", manifest)
    manifest_hash = sha256_file(staging / "manifests/dataset_manifest.json")
    write_json(staging / "generation_state/BUILD_COMPLETE.json", {
        "status": "PASS", "manifest_sha256": manifest_hash,
        "build_attempt_count": 1,
    })
    os.replace(staging, destination)
    print(json.dumps({"status": "PASS", "output": str(destination),
                      "manifest_sha256": manifest_hash,
                      "split_counts": retained, "excluded": dict(excluded)},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
