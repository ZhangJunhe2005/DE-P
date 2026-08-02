#!/usr/bin/env python3
"""Build the frozen P0 reference view without copying V3 depth tensors."""

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

from data.static_yopo_manifest_v1 import (
    DATA_ROUTE, DATASET_VERSION, REPRESENTATION, SCHEMA_VERSION,
    SOURCE_V3_MANIFEST_SHA256, assert_source_identity, canonical_hash,
    sample_identity, sha256_file,
)
from data.static_yopo_preprocessing_v1 import DEFAULT_STATIC_YOPO_PREPROCESSOR_V1
from policy.static_yopo_contract_v1 import (
    LOSS_CONTRACT_HASH, MODEL_CONTRACT_HASH, NORMALIZATION_CONTRACT_HASH,
)

SOURCE = ROOT / "data" / "phase8_dynamic_evidence_formal_v3"
DEFAULT_OUTPUT = ROOT / "data" / DATASET_VERSION
INDEX_FIELDS = {
    "sample_id": ("S64", ()),
    "sequence_id": ("S32", ()),
    "frame_index": ("<i2", ()),
    "source_split": ("S5", ()),
    "map_uuid": ("S36", ()),
    "suite": ("S7", ()),
    "depth_path": ("S240", ()),
    "depth_hash": ("S64", ()),
    "observation": ("<f4", (9,)),
    "position_world": ("<f4", (3,)),
    "quaternion_xyzw": ("<f4", (4,)),
    "map_id": ("<i2", ()),
}
MAP_TYPE = {1: "cave", 2: "pillar", 5: "forest", 6: "room", 7: "wall"}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def load_work_units():
    work_root = SOURCE / "manifests" / "work_units"
    return [json.load(open(path, encoding="utf-8")) for path in sorted(work_root.glob("*.json"))]


def _write_authority_ply(source, destination, point_count):
    expected_bytes = int(point_count) * 3 * 4
    container_header = 16
    if source.stat().st_size != expected_bytes + container_header:
        raise RuntimeError(f"raw authority cloud size mismatch: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {point_count}\n"
        "property float x\nproperty float y\nproperty float z\nend_header\n"
    ).encode("ascii")
    with open(destination, "wb") as target, open(source, "rb") as raw:
        magic = raw.read(8)
        stored_count = int.from_bytes(raw.read(8), "little")
        if magic != b"SGARAW1\x00" or stored_count != int(point_count):
            raise RuntimeError(f"raw authority cloud container header mismatch: {source}")
        target.write(header)
        for chunk in iter(lambda: raw.read(8 * 1024 * 1024), b""):
            target.write(chunk)


def authority_maps(output, materialize_ply):
    rows = []
    numeric = {}
    for source_split in ("train", "valid"):
        base = SOURCE / "raw_authoritative" / "geometry_authority" / source_split
        for map_dir in sorted(path for path in base.iterdir() if path.is_dir()):
            metadata = json.load(open(map_dir / "occupancy_metadata.json", encoding="utf-8"))
            provenance = json.load(open(map_dir / "mixed_scene_provenance.json", encoding="utf-8"))
            uuid = metadata["map_uuid"]
            numeric[uuid] = len(numeric)
            row = {
                "map_id": numeric[uuid], "map_uuid": uuid,
                "source_split": source_split,
                "map_type": MAP_TYPE[int(provenance["maze_type"])],
                "authority_manifest": str((map_dir / "authority_manifest.json").resolve()),
                "authority_manifest_hash": sha256_file(map_dir / "authority_manifest.json"),
                "occupancy": str((map_dir / "occupancy.bin").resolve()),
                "occupancy_hash": metadata["occupancy_hash"],
                "raw_cloud": str((map_dir / "raw_cloud.bin").resolve()),
                "raw_cloud_hash": metadata["source_cloud_hash"],
                "resolution_m": metadata["occupancy_resolution_m"],
                "frame_id": metadata["frame_id"],
                "uav_radius_m": metadata["uav_radius_m"],
                "source_point_count": metadata["source_point_count"],
            }
            if materialize_ply:
                ply = output / "maps" / f"pointcloud-{numeric[uuid]}.ply"
                _write_authority_ply(
                    map_dir / "raw_cloud.bin", ply, metadata["source_point_count"]
                )
                row["static_ply"] = str(ply.resolve())
                row["static_ply_hash"] = sha256_file(ply)
            rows.append(row)
    write_json(output / "manifests" / "map_authority.json", {"maps": rows})
    if materialize_ply:
        yaml = YAML()
        yaml.default_flow_style = False
        with open(output / "map_catalog.yaml", "w", encoding="utf-8") as stream:
            yaml.dump({
                "catalog_version": 1,
                "dataset_version": DATASET_VERSION,
                "authority": "V3 raw_cloud.bin exact binary PLY wrapping",
                "maps": [{
                    "map_id": row["map_id"], "map_uuid": row["map_uuid"],
                    "static_ply": row["static_ply"],
                    "static_map_sha256": row["static_ply_hash"],
                } for row in rows],
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


def build(output, pilot=0):
    assert_source_identity(SOURCE)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite derived root: {output}")
    output.mkdir(parents=True)
    maps, map_ids = authority_maps(output, materialize_ply=not pilot)
    units = load_work_units()
    indexed_units = [row for row in units if row["split"] in {"train", "validation"}]
    if pilot:
        chosen = []
        seen = set()
        for row in indexed_units:
            key = (row["split"], row["map_uuid"])
            if key not in seen:
                chosen.append(row)
                seen.add(key)
            if len(chosen) >= pilot:
                break
        indexed_units = chosen
    counts = Counter(row["split"] for row in indexed_units)
    counts = {key: value * 60 for key, value in counts.items()}
    arrays = {split: create_arrays(output, split, count) for split, count in counts.items()}
    cursor = Counter()
    depth_hash_splits = defaultdict(set)
    sample_ids = set()
    map_split = defaultdict(set)
    map_type_counts = defaultdict(Counter)
    depth_stats = defaultdict(lambda: {"min": float("inf"), "max": float("-inf"),
                                       "finite": 0, "values": 0})
    for unit_no, row in enumerate(indexed_units, 1):
        split = row["split"]
        sequence = row["sequence_id"]
        suite = row["suite"]
        source_split = row["source_split"]
        sequence_root = (
            SOURCE / "raw_authoritative" / suite / source_split / sequence
        )
        depth_path = sequence_root / "static_depth.npy"
        depth = np.load(depth_path, mmap_mode="r")
        if depth.dtype != np.float32 or depth.shape != (60, 96, 160):
            raise RuntimeError(f"invalid source depth: {sequence}")
        frames = [
            json.loads(line) for line in
            open(sequence_root / "frames.jsonl", encoding="utf-8")
        ]
        if len(frames) != 60:
            raise RuntimeError(f"partial source sequence: {sequence}")
        map_uuid = row["map_uuid"]
        map_split[map_uuid].add(split)
        map_type = next(item["map_type"] for item in maps if item["map_uuid"] == map_uuid)
        map_type_counts[split][map_type] += 60
        for frame_index, frame in enumerate(frames):
            if frame["frame_index"] != frame_index or frame["map_uuid"] != map_uuid:
                raise RuntimeError(f"source identity mismatch: {sequence}/{frame_index}")
            observation = np.asarray(
                frame["velocity_body"] + frame["acceleration_body"] + frame["goal_body"],
                dtype=np.float32,
            )
            if observation.shape != (9,) or not np.isfinite(observation).all():
                raise RuntimeError(f"invalid observation: {sequence}/{frame_index}")
            depth_hash = frame["static_depth_frame_hash"]
            if pilot or frame_index in {0, 59}:
                actual = hashlib.sha256(np.asarray(depth[frame_index]).tobytes()).hexdigest()
                if actual != depth_hash:
                    raise RuntimeError(f"static depth hash mismatch: {sequence}/{frame_index}")
            identity = sample_identity(
                sequence, frame_index, map_uuid, depth_hash, observation
            )
            if identity in sample_ids:
                raise RuntimeError(f"duplicate sample identity: {identity}")
            sample_ids.add(identity)
            depth_hash_splits[depth_hash].add(split)
            index = cursor[split]
            values = {
                "sample_id": identity, "sequence_id": sequence,
                "frame_index": frame_index, "source_split": source_split,
                "map_uuid": map_uuid, "suite": suite,
                "depth_path": str(depth_path.resolve()), "depth_hash": depth_hash,
                "observation": observation,
                "position_world": np.asarray(frame["position_world"], dtype=np.float32),
                "quaternion_xyzw": np.asarray(
                    frame["quaternion_world_from_body"], dtype=np.float32
                ),
                "map_id": map_ids[map_uuid],
            }
            for field, value in values.items():
                arrays[split][field][index] = value
            cursor[split] += 1
            values_depth = np.asarray(depth[frame_index])
            finite = values_depth[np.isfinite(values_depth)]
            stats = depth_stats[split]
            stats["finite"] += finite.size
            stats["values"] += values_depth.size
            if finite.size:
                stats["min"] = min(stats["min"], float(finite.min()))
                stats["max"] = max(stats["max"], float(finite.max()))
        if unit_no % 500 == 0:
            print(json.dumps({"status": "INDEXING", "work_units": unit_no,
                              "total": len(indexed_units)}), flush=True)
    for split_arrays in arrays.values():
        for value in split_arrays.values():
            value.flush()
    leaks = {uuid: sorted(value) for uuid, value in map_split.items() if len(value) > 1}
    if leaks:
        raise RuntimeError(f"map split leakage: {leaks}")
    cross = [key for key, value in depth_hash_splits.items() if len(value) > 1]
    if cross:
        raise RuntimeError(f"cross-split exact static depth duplicates: {len(cross)}")

    sealed = {}
    for split in ("calibration", "internal_test"):
        selected = [row for row in units if row["split"] == split]
        sealed[split] = {
            "content_accessed": False,
            "sample_count": sum(int(row["sample_count"]) for row in selected),
            "sequence_count": len(selected),
            "sequence_identities": [{
                "sequence_id": row["sequence_id"],
                "map_uuid": row["map_uuid"],
                "work_unit_hash": canonical_hash(row),
                "sample_count": row["sample_count"],
            } for row in selected],
        }
        write_json(output / "sealed" / f"{split}_identity.json", sealed[split])

    split_contract = {
        "train_source": "raw train",
        "validation_source": "raw valid deterministic map-group partition",
        "calibration": "identity only, unopened",
        "internal_test": "sealed identity only, unopened",
        "sample_counts": {
            **{key: int(value) for key, value in counts.items()},
            **{key: value["sample_count"] for key, value in sealed.items()},
        },
        "map_leakage": 0, "sequence_leakage": 0,
        "cross_split_exact_depth_duplicates_indexed_splits": 0,
    }
    write_json(output / "manifests" / "split_manifest.json", split_contract)
    split_hash = sha256_file(output / "manifests" / "split_manifest.json")
    manifest = {
        "status": "PILOT_NON_FORMAL" if pilot else "COMPLETE_FROZEN",
        "dataset_version": DATASET_VERSION, "schema_version": SCHEMA_VERSION,
        "data_route": DATA_ROUTE, "representation": REPRESENTATION,
        "source_root": str(SOURCE.resolve()),
        "source_v3_manifest_hash": SOURCE_V3_MANIFEST_SHA256,
        "sample_count": sum(counts.values()) + sum(
            value["sample_count"] for value in sealed.values()
        ),
        "indexed_sample_count": sum(counts.values()),
        "split_counts": split_contract["sample_counts"],
        "split_hash": split_hash,
        "preprocessing_version": DEFAULT_STATIC_YOPO_PREPROCESSOR_V1.version,
        "preprocessing_hash": DEFAULT_STATIC_YOPO_PREPROCESSOR_V1.contract_hash,
        "normalization_hash": NORMALIZATION_CONTRACT_HASH,
        "model_contract_hash": MODEL_CONTRACT_HASH,
        "loss_contract_hash": LOSS_CONTRACT_HASH,
        "map_authority_hash": sha256_file(output / "manifests" / "map_authority.json"),
        "map_type_sample_counts": {
            split: dict(values) for split, values in map_type_counts.items()
        },
        "source_depth": {"dtype": "float32", "unit": "m", "shape": [96, 160]},
        "depth_stats": dict(depth_stats),
        "forbidden_model_inputs": [
            "composed_depth", "actor_owner", "actor_metadata", "actor_future",
            "dynamic_actionability", "scenario", "map_type",
        ],
        "source_mutated": False, "depth_copied": False,
        "calibration_content_accessed": False,
        "internal_test_content_accessed": False,
        "test_blind_production_accessed": False,
        "pilot_work_units": int(pilot),
    }
    write_json(output / "manifests" / "dataset_manifest.json", manifest)
    manifest["derived_manifest_hash"] = sha256_file(
        output / "manifests" / "dataset_manifest.json"
    )
    write_json(output / "generation_state" / "BUILD_COMPLETE.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pilot-work-units", type=int, default=0)
    parser.add_argument("--formal-final-build", action="store_true")
    args = parser.parse_args()
    if bool(args.pilot_work_units) == bool(args.formal_final_build):
        raise SystemExit("select exactly one of --pilot-work-units or --formal-final-build")
    build(args.output.expanduser().resolve(), args.pilot_work_units)


if __name__ == "__main__":
    main()
