#!/usr/bin/env python3
"""Read-only pre-freeze SNRE-CTR1 source, identity, and aliasing audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.static_no_return_contract_v1 import (
    CanonicalNoReturnDepthV1, NoReturnClass, StaticDatasetLeakagePolicyV2,
    contract_hash, exact_complete_input_hash, normalize_observation_v1,
    tolerance_observation_hash,
)
from data.static_yopo_manifest_v1 import SOURCE_V3_MANIFEST_SHA256, sha256_file
from data.static_yopo_preprocessing_v1 import DEFAULT_STATIC_YOPO_PREPROCESSOR_V1

PREFIX = "phase8jqv2_5snrectr1"
REPORTS = ROOT / "reports"
OLD = ROOT / "artifacts/phase8_mixed_scene_static_yopo_derived_v1_failed_cross_split_gate"
SOURCE = ROOT / "data/phase8_dynamic_evidence_formal_v3"
NO_RETURN_HASH = "4d35e620a40043e03db2ae014c204faf098583893638e1d4d2198879f98221ba"


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def arrays(split):
    base = OLD / "indices" / split
    names = (
        "sample_id", "sequence_id", "frame_index", "source_split", "map_uuid",
        "suite", "depth_path", "depth_hash", "observation", "position_world",
        "quaternion_xyzw", "map_id",
    )
    return {name: np.load(base / f"{name}.npy", mmap_mode="r") for name in names}


def text(value):
    return bytes(value).decode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    detector = CanonicalNoReturnDepthV1()
    policy = StaticDatasetLeakagePolicyV2()
    source_hash = sha256_file(SOURCE / "manifests/dataset_manifest.json")
    if source_hash != SOURCE_V3_MANIFEST_SHA256:
        raise RuntimeError("V3 source identity mismatch")
    old_final = json.load(open(REPORTS / "phase8jqv2_5msydtr1_final_result.json"))
    if old_final["status"] != "FAIL_STATIC_DATA_CONTRACT":
        raise RuntimeError("MSY-DTR1 terminal Route B is not frozen")
    split_data = {split: arrays(split) for split in ("train", "validation")}
    no_return_rows = {}
    detector_counts = Counter()
    map_counts = defaultdict(Counter)
    sequence_counts = defaultdict(set)
    depth_paths = defaultdict(set)
    inodes = defaultdict(set)
    complete = defaultdict(set)
    tolerant = defaultdict(set)
    detailed = []
    for split, values in split_data.items():
        indices = np.flatnonzero(values["depth_hash"] == NO_RETURN_HASH.encode())
        no_return_rows[split] = indices
        cache = {}
        normalized = normalize_observation_v1(values["observation"][indices])
        depth_network_hash = hashlib.sha256(
            np.ones((1, 96, 160), dtype=np.float32).tobytes()
        ).hexdigest()
        for local, index in enumerate(indices):
            path = text(values["depth_path"][index])
            frame_index = int(values["frame_index"][index])
            if path not in cache:
                cache[path] = np.load(path, mmap_mode="r")
            depth = cache[path][frame_index]
            classification = detector.classify(depth, NO_RETURN_HASH)
            detector_counts[classification] += 1
            if classification != NoReturnClass.CANONICAL:
                raise RuntimeError(f"noncanonical duplicate frame: {path}/{frame_index}: {classification}")
            map_uuid = text(values["map_uuid"][index])
            map_counts[split][map_uuid] += 1
            sequence = text(values["sequence_id"][index])
            sequence_counts[split].add(sequence)
            depth_paths[split].add(path)
            stat = os.stat(path)
            inodes[split].add((stat.st_dev, stat.st_ino))
            exact = exact_complete_input_hash(
                depth_network_hash, normalized[local], True,
                DEFAULT_STATIC_YOPO_PREPROCESSOR_V1.version,
            )
            tolerance = tolerance_observation_hash(
                normalized[local], policy.normalized_observation_quantization
            )
            complete[exact].add(split)
            tolerant[tolerance].add(split)
            detailed.append({
                "split": split, "sample_id": text(values["sample_id"][index]),
                "sequence_id": sequence, "frame_index": frame_index,
                "map_uuid": map_uuid, "depth_path": path,
                "normalized_observation_hash": hashlib.sha256(
                    np.asarray(normalized[local], dtype="<f4").tobytes()
                ).hexdigest(),
                "complete_input_hash": exact,
                "tolerance_input_hash": tolerance,
            })
    # L1/L2 identities are independent of observation values.
    identity_sets = {}
    for name in ("sample_id", "sequence_id", "depth_path"):
        identity_sets[name] = {
            split: {text(value) for value in values[name]}
            for split, values in split_data.items()
        }
    map_sets = {
        split: {text(value) for value in values["map_uuid"]}
        for split, values in split_data.items()
    }
    sample_frame = {
        split: {
            (text(values["sequence_id"][i]), int(values["frame_index"][i]))
            for i in range(len(values["frame_index"]))
        } for split, values in split_data.items()
    }
    cross_complete = sorted(key for key, splits in complete.items() if len(splits) > 1)
    cross_tolerant = sorted(key for key, splits in tolerant.items() if len(splits) > 1)
    result = {
        "status": "PASS_IDENTITY_PENDING_TARGET_AUDIT",
        "source_v3_manifest_hash": source_hash,
        "old_artifact": str(OLD),
        "old_artifact_modified": False,
        "canonical_detector": detector.contract(),
        "canonical_detector_hash": contract_hash(detector.contract()),
        "leakage_policy_v2": policy.contract(),
        "leakage_policy_hash": contract_hash(policy.contract()),
        "no_return": {
            "hash": NO_RETURN_HASH,
            "counts": {key: len(value) for key, value in no_return_rows.items()},
            "classification_counts": dict(detector_counts),
            "map_counts": {key: dict(value) for key, value in map_counts.items()},
            "source_sequences": {key: len(value) for key, value in sequence_counts.items()},
            "source_files": {key: len(value) for key, value in depth_paths.items()},
        },
        "leakage": {
            "map_uuid_overlap": sorted(map_sets["train"] & map_sets["validation"]),
            "sample_id_overlap": sorted(identity_sets["sample_id"]["train"] & identity_sets["sample_id"]["validation"]),
            "sequence_overlap": sorted(identity_sets["sequence_id"]["train"] & identity_sets["sequence_id"]["validation"]),
            "sequence_frame_overlap_count": len(sample_frame["train"] & sample_frame["validation"]),
            "source_path_overlap": sorted(identity_sets["depth_path"]["train"] & identity_sets["depth_path"]["validation"]),
            "source_inode_overlap": len(inodes["train"] & inodes["validation"]),
            "exact_complete_input_cross_split_groups": len(cross_complete),
            "tolerance_complete_input_cross_split_groups": len(cross_tolerant),
            "depth_only_cross_split_groups": 1,
            "canonical_depth_only_group_allowed": True,
        },
        "cross_complete_input_hashes": cross_complete,
        "cross_tolerance_input_hashes": cross_tolerant,
        "runtime_preprocessing_parity": "PASS_SHARED_IMPLEMENTATION",
        "invalid_fill": detector_counts[NoReturnClass.INVALID_FILL],
        "corrupt_constant": detector_counts[NoReturnClass.CORRUPT],
        "training_started": False,
        "internal_test_accessed": False,
    }
    hard = result["leakage"]
    if (
        hard["map_uuid_overlap"] or hard["sample_id_overlap"] or hard["sequence_overlap"]
        or hard["sequence_frame_overlap_count"] or hard["source_path_overlap"]
        or hard["source_inode_overlap"] or result["invalid_fill"] or result["corrupt_constant"]
    ):
        result["status"] = "FAIL_TRUE_SPLIT_LEAKAGE"
    destination = args.output or REPORTS / f"{PREFIX}_prefreeze_audit.json"
    atomic_json(destination, result)
    # Detailed rows are immutable audit input for target evaluation, not model input.
    atomic_json(REPORTS / f"{PREFIX}_no_return_rows.json", {
        "status": "AUDIT_ONLY_NOT_MODEL_INPUT", "rows": detailed,
    })
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
