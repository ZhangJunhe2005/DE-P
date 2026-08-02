#!/usr/bin/env python3
"""Build an actor-free V4 static YOPO reference view from authoritative raw data."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from numpy.lib.format import open_memmap
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.static_yopo_preprocessing_v1 import DEFAULT_STATIC_YOPO_PREPROCESSOR_V1
from policy.static_yopo_contract_v2 import (
    LOSS_CONTRACT_V2_HASH,
    MODEL_CONTRACT_V2_HASH,
    NORMALIZATION_CONTRACT_V2_HASH,
)
from tools.build_phase8_mixed_static_yopo_derived_v1 import (
    INDEX_FIELDS,
    _write_authority_ply,
    write_json,
)


SOURCE = ROOT / "data/route_a_v4_raw_static"
OUTPUT = ROOT / "data/route_a_v4_static_yopo"
RAW_DATASET_VERSION = "route_a_v4_raw_static_v1"
DERIVED_DATASET_VERSION = "route_a_v4_static_yopo"
DERIVED_SCHEMA_VERSION = "static_yopo_sample_v4_route_a"
CATALOG_VERSION = 4
MAP_TYPE = {1: "cave", 2: "pillar", 5: "forest", 6: "room", 7: "wall"}
SPLIT_CONTRACT_VERSION = "route_a_v4_split_contract_v1"


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(source_hash, sequence, frame, map_uuid, depth_hash):
    return hashlib.sha256(json.dumps({
        "source_manifest": source_hash,
        "sequence": sequence,
        "frame": int(frame),
        "map_uuid": map_uuid,
        "static_depth_hash": depth_hash,
        "observation_contract": "route_a_wide_state_v1",
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sequence_rows():
    result = []
    for path in sorted((SOURCE / "manifests/sequences").glob("*.json")):
        row = json.loads(path.read_text())
        if row["suite"] != "static":
            raise RuntimeError("V4 static source contains a dynamic suite")
        if row["split"] not in {"train", "valid"}:
            raise RuntimeError("unexpected V4 source split")
        result.append(row)
    return result


def sequence_root(row):
    return SOURCE / row["suite"] / row["split"] / row["sequence_id"]


def static_depth_path(row):
    """Resolve a depth payload that is proven to be actor-free.

    The authoritative generator omits the redundant ``static_depth.npy`` file
    when evidence-authority outputs are disabled.  For a static-only sequence,
    ``depth.npy`` is then byte-semantically the same rendered depth.  Do not
    apply this fallback to dynamic suites: prove the equivalence from the
    persisted manifest, render diagnostics, and per-frame provenance first.
    """
    root = sequence_root(row)
    manifest_path = (
        SOURCE / "manifests/sequences" / f"{row['sequence_id']}.json"
    )
    manifest = json.loads(manifest_path.read_text())
    relative_base = f"{row['suite']}/{row['split']}/{row['sequence_id']}"
    static_relative = f"{relative_base}/static_depth.npy"
    depth_relative = f"{relative_base}/depth.npy"
    files = manifest.get("files", {})
    if static_relative in files:
        result = root / "static_depth.npy"
    elif depth_relative in files:
        if row["suite"] != "static":
            raise RuntimeError(
                f"refusing composed-depth fallback for non-static suite: "
                f"{row['sequence_id']}"
            )
        diagnostics = json.loads(
            (root / "render_diagnostics.json").read_text()
        )
        if (
            diagnostics.get("actor_count") != 0
            or diagnostics.get("actor_depth_pixels_total") != 0
            or diagnostics.get("frames_with_actor_depth") != 0
            or diagnostics.get("static_depth_semantic_hash")
            != diagnostics.get("composed_depth_semantic_hash")
        ):
            raise RuntimeError(
                f"depth payload is not proven actor-free: {row['sequence_id']}"
            )
        frames = [
            json.loads(line)
            for line in (root / "frames.jsonl").read_text().splitlines()
        ]
        if len(frames) != int(row["frame_count"]) or any(
            frame.get("dynamic_depth_composited") is not False
            or frame.get("actor_metadata") != []
            for frame in frames
        ):
            raise RuntimeError(
                f"frame provenance is not actor-free: {row['sequence_id']}"
            )
        result = root / "depth.npy"
    else:
        raise FileNotFoundError(
            f"no authoritative depth payload: {row['sequence_id']}"
        )
    if not result.is_file():
        raise FileNotFoundError(
            f"manifested depth payload is missing: {result}"
        )
    return result


def retained_indices(row):
    depth = np.load(static_depth_path(row), mmap_mode="r")
    if depth.dtype != np.float32 or depth.shape[1:] != (96, 160):
        raise RuntimeError(f"invalid V4 depth array: {row['sequence_id']}")
    return [
        index for index in range(len(depth))
        if not np.all(depth[index] == np.float32(20.0))
    ]


def build_maps(staging):
    map_rows, catalog, numeric = [], [], {}
    manifests = [
        json.loads(path.read_text())
        for path in sorted((SOURCE / "manifests/maps").glob("*.json"))
    ]
    for number, row in enumerate(sorted(manifests, key=lambda value: value["map_uuid"])):
        map_uuid = row["map_uuid"]
        numeric[map_uuid] = number
        authority = (
            SOURCE / f"geometry_authority/{row['split']}/{map_uuid}"
        )
        metadata = json.loads((authority / "occupancy_metadata.json").read_text())
        provenance = json.loads(
            (authority / "mixed_scene_provenance.json").read_text()
        )
        ply = staging / f"maps/pointcloud-{number}.ply"
        published_ply = OUTPUT / f"maps/pointcloud-{number}.ply"
        _write_authority_ply(
            authority / "raw_cloud.bin", ply, metadata["source_point_count"]
        )
        ply_hash = sha(ply)
        map_type = MAP_TYPE[int(provenance["maze_type"])]
        map_rows.append({
            "map_id": number,
            "map_uuid": map_uuid,
            "source_split": row["split"],
            "map_type": map_type,
            "size_class": provenance["resolved_parameters"]["size_class"],
            "profile_name": provenance["profile_name"],
            "map_seed": provenance["seed"],
            "frame_id": "odom",
            "resolution_m": metadata["occupancy_resolution_m"],
            "uav_radius_m": metadata["uav_radius_m"],
            "authority_manifest": str(authority / "authority_manifest.json"),
            "authority_manifest_hash": metadata["artifact_manifest_hash"],
            "occupancy": str(authority / "occupancy.bin"),
            "occupancy_hash": metadata["occupancy_hash"],
            "raw_cloud": str(authority / "raw_cloud.bin"),
            "raw_cloud_hash": metadata["source_cloud_hash"],
            "static_ply": str(published_ply.resolve()),
            "static_ply_hash": ply_hash,
        })
        catalog.append({
            "map_id": number,
            "map_uuid": map_uuid,
            "static_ply": str(published_ply.resolve()),
            "static_map_sha256": ply_hash,
        })
    write_json(staging / "manifests/map_authority.json", {
        "authority_source": f"{DERIVED_DATASET_VERSION}_authoritative_mixed_maps",
        "maps": map_rows,
    })
    (staging / "map_catalog.yaml").write_text(yaml.safe_dump({
        "catalog_version": CATALOG_VERSION,
        "dataset_version": DERIVED_DATASET_VERSION,
        "authority": "Route-A canonical authority exact binary PLY wrapping",
        "maps": catalog,
    }, sort_keys=False))
    return map_rows, numeric


def write_split_contract(root, counts, excluded):
    index_files = {}
    for split in ("train", "validation"):
        index_files[split] = {}
        for field in INDEX_FIELDS:
            relative = Path("indices") / split / f"{field}.npy"
            path = root / relative
            if not path.is_file():
                raise FileNotFoundError(f"missing V4 split index: {path}")
            index_files[split][str(relative)] = sha(path)
    authority = json.loads(
        (root / "manifests/map_authority.json").read_text()
    )
    map_membership = {
        "train": sorted(
            row["map_uuid"] for row in authority["maps"]
            if row["source_split"] == "train"
        ),
        "validation": sorted(
            row["map_uuid"] for row in authority["maps"]
            if row["source_split"] == "valid"
        ),
    }
    overlap = sorted(set(map_membership["train"]) & set(
        map_membership["validation"]
    ))
    if overlap:
        raise RuntimeError(f"V4 map split leakage: {overlap}")
    contract = {
        "version": SPLIT_CONTRACT_VERSION,
        "assignment": {
            "train": "raw train, complete map UUID groups",
            "validation": "raw valid, complete map UUID groups",
        },
        "sample_counts": {
            key: int(value) for key, value in counts.items()
        },
        "excluded_canonical_no_return": {
            key: int(value) for key, value in excluded.items()
        },
        "map_membership": map_membership,
        "map_leakage": 0,
        "sequence_leakage": 0,
        "index_file_hashes": index_files,
        "internal_test_accessed": False,
        "training_started": False,
    }
    path = root / "manifests/split_manifest.json"
    write_json(path, contract)
    return sha(path)


def validate_split_contract(root, manifest):
    path = root / "manifests/split_manifest.json"
    if not path.is_file():
        raise RuntimeError("V4 split manifest is missing")
    actual_hash = sha(path)
    if actual_hash != manifest.get("split_hash"):
        raise RuntimeError("V4 split manifest hash mismatch")
    contract = json.loads(path.read_text())
    if contract.get("version") != SPLIT_CONTRACT_VERSION:
        raise RuntimeError("V4 split contract version mismatch")
    for entries in contract.get("index_file_hashes", {}).values():
        for relative, expected in entries.items():
            index_path = root / relative
            if not index_path.is_file() or sha(index_path) != expected:
                raise RuntimeError(f"V4 split index hash mismatch: {relative}")
    return actual_hash


def main():
    global SOURCE, OUTPUT, RAW_DATASET_VERSION
    global DERIVED_DATASET_VERSION, DERIVED_SCHEMA_VERSION
    global CATALOG_VERSION, SPLIT_CONTRACT_VERSION
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorize-build", action="store_true")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--raw-dataset-version", default=RAW_DATASET_VERSION
    )
    parser.add_argument(
        "--derived-dataset-version", default=DERIVED_DATASET_VERSION
    )
    parser.add_argument(
        "--derived-schema-version", default=DERIVED_SCHEMA_VERSION
    )
    parser.add_argument(
        "--split-contract-version", default=SPLIT_CONTRACT_VERSION
    )
    parser.add_argument("--catalog-version", type=int, default=CATALOG_VERSION)
    args = parser.parse_args()
    if not args.authorize_build:
        raise SystemExit("--authorize-build is required")
    SOURCE = args.source.expanduser().resolve()
    OUTPUT = args.output.expanduser().resolve()
    RAW_DATASET_VERSION = args.raw_dataset_version
    DERIVED_DATASET_VERSION = args.derived_dataset_version
    DERIVED_SCHEMA_VERSION = args.derived_schema_version
    SPLIT_CONTRACT_VERSION = args.split_contract_version
    CATALOG_VERSION = args.catalog_version
    source_manifest = SOURCE / "manifests/dataset_manifest.json"
    if not source_manifest.is_file():
        raise FileNotFoundError("V4 raw dataset is incomplete")
    source_document = json.loads(source_manifest.read_text())
    if source_document.get("dataset_version") != RAW_DATASET_VERSION:
        raise RuntimeError("V4 raw dataset identity mismatch")
    if not (
        SOURCE / "generation_state/completion/FULL_GENERATION_COMPLETE"
    ).is_file():
        raise RuntimeError("V4 raw completion marker missing")
    if OUTPUT.exists():
        raise FileExistsError("V4 derived output already exists")
    staging = OUTPUT.parent / f".{OUTPUT.name}.staging"
    if staging.exists():
        raise FileExistsError("V4 derived staging already exists")
    units = sequence_rows()
    counts, excluded, retained = Counter(), Counter(), {}
    for number, row in enumerate(units, 1):
        target = "train" if row["split"] == "train" else "validation"
        keep = retained_indices(row)
        retained[row["sequence_id"]] = keep
        counts[target] += len(keep)
        excluded[target] += int(row["frame_count"]) - len(keep)
        if number % 500 == 0:
            print(json.dumps({
                "status": "PRECOUNT", "sequences": number, "total": len(units)
            }), flush=True)
    staging.mkdir(parents=True)
    map_rows, map_ids = build_maps(staging)
    arrays = {}
    for split in ("train", "validation"):
        base = staging / f"indices/{split}"
        base.mkdir(parents=True)
        arrays[split] = {
            name: open_memmap(
                base / f"{name}.npy", mode="w+", dtype=dtype,
                shape=(counts[split], *shape),
            )
            for name, (dtype, shape) in INDEX_FIELDS.items()
        }
    cursors = Counter()
    per_type = defaultdict(Counter)
    map_lookup = {row["map_uuid"]: row for row in map_rows}
    source_hash = sha(source_manifest)
    identities = set()
    for number, row in enumerate(units, 1):
        target = "train" if row["split"] == "train" else "validation"
        root = sequence_root(row)
        frames = [
            json.loads(line) for line in (root / "frames.jsonl").read_text().splitlines()
        ]
        depth_path = static_depth_path(row)
        for frame_index in retained[row["sequence_id"]]:
            frame = frames[frame_index]
            sample_id = identity(
                source_hash, row["sequence_id"], frame_index, row["map_uuid"],
                frame["static_depth_frame_hash"],
            )
            if sample_id in identities:
                raise RuntimeError("V4 sample identity duplicate")
            identities.add(sample_id)
            raw_observation = np.asarray(
                frame["velocity_body"] + frame["acceleration_body"]
                + frame["goal_body"], dtype=np.float32,
            )
            values = {
                "sample_id": sample_id,
                "sequence_id": row["sequence_id"],
                "frame_index": frame_index,
                "source_split": row["split"],
                "map_uuid": row["map_uuid"],
                "suite": "static",
                "depth_path": str(depth_path.resolve()),
                "depth_hash": frame["static_depth_frame_hash"],
                "observation": raw_observation,
                "position_world": np.asarray(frame["position_world"], dtype=np.float32),
                "quaternion_xyzw": np.asarray(
                    frame["quaternion_world_from_body"], dtype=np.float32
                ),
                "map_id": map_ids[row["map_uuid"]],
            }
            index = cursors[target]
            for field, value in values.items():
                arrays[target][field][index] = value
            cursors[target] += 1
            per_type[target][map_lookup[row["map_uuid"]]["map_type"]] += 1
        if number % 500 == 0:
            print(json.dumps({
                "status": "INDEXING", "sequences": number, "total": len(units)
            }), flush=True)
    for group in arrays.values():
        for array in group.values():
            array.flush()
    if dict(cursors) != dict(counts):
        raise RuntimeError("V4 final index count mismatch")
    split_hash = write_split_contract(staging, counts, excluded)
    manifest = {
        "status": "COMPLETE_FROZEN",
        "dataset_version": DERIVED_DATASET_VERSION,
        "schema_version": DERIVED_SCHEMA_VERSION,
        "source_dataset_manifest_hash": source_hash,
        "split_contract_version": SPLIT_CONTRACT_VERSION,
        "split_hash": split_hash,
        "split_counts": dict(counts),
        "excluded_canonical_no_return": dict(excluded),
        "map_type_sample_counts": {
            split: dict(values) for split, values in per_type.items()
        },
        "map_authority_hash": sha(staging / "manifests/map_authority.json"),
        "map_catalog_hash": sha(staging / "map_catalog.yaml"),
        "preprocessing_hash":
            DEFAULT_STATIC_YOPO_PREPROCESSOR_V1.contract_hash,
        "normalization_hash": NORMALIZATION_CONTRACT_V2_HASH,
        "model_contract_hash": MODEL_CONTRACT_V2_HASH,
        "loss_contract_hash": LOSS_CONTRACT_V2_HASH,
        "observation_contract": "route_a_wide_state_v1",
        "raw_observation_used_for_training": False,
        "actor_input_used": False,
        "composed_depth_used": False,
        "internal_test_accessed": False,
        "training_started": False,
    }
    write_json(staging / "manifests/dataset_manifest.json", manifest)
    write_json(staging / "generation_state/BUILD_COMPLETE.json", {
        "status": "PASS",
        "manifest_sha256": sha(staging / "manifests/dataset_manifest.json"),
    })
    os.replace(staging, OUTPUT)
    print(json.dumps({
        "status": "PASS",
        "output": str(OUTPUT),
        "split_counts": dict(counts),
        "training_started": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
