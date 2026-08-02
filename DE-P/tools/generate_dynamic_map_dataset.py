#!/usr/bin/env python3
"""Build a deterministic, bounded multi-map dynamic preflight dataset.

Production-scale generation is deliberately plan-only here; real depth recording
must be launched separately with full access after capacity approval.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import open3d as o3d
from ruamel.yaml import YAML

from generate_synthetic_dynamic_sequences import dump_yaml, generate_sequence


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def map_entry(map_id, path):
    cloud = o3d.io.read_point_cloud(str(path))
    points = len(cloud.points)
    if points == 0:
        raise ValueError(f"rejecting completely blank static map: {path}")
    bounds = [cloud.get_min_bound().tolist(), cloud.get_max_bound().tolist()]
    extent = max(1e-6, (bounds[1][0] - bounds[0][0]) * (bounds[1][1] - bounds[0][1])
                 * max(1.0, bounds[1][2] - bounds[0][2]))
    density = points / extent
    if density > 5000:
        raise ValueError(f"rejecting implausibly dense map: {path}")
    return {
        "map_id": map_id, "map_seed": 8000 + map_id,
        "static_ply": str(path.resolve()), "static_map_sha256": sha256(path),
        "bounds": bounds, "obstacle_density": density,
        "reachability": "inherited_from_existing_YOPO_pose_set",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--static-root", type=Path,
                        default=Path("/home/zjh/YOPO/dataset"))
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--production-plan-only", action="store_true")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if args.production_plan_only:
        output.mkdir(parents=True)
        dump_yaml(output / "generation_plan.yaml", {
            "status": "AWAITING_USER_CAPACITY_APPROVAL",
            "generator": "/home/zjh/YOPO/Simulator/src/src/dataset_generator.cpp",
            "static_format": "pointcloud-<map_id>.ply + pose-<map_id>.csv",
            "official_dynamic_sensor": "depth",
            "formal_pointcloud_training_allowed": False,
        })
        return
    definitions = {
        "train": [(0, "empty"), (0, "crossing"), (1, "head_on_cylinder"),
                  (1, "multi_target"), (1, "waypoint_ping_pong"),
                  (0, "delayed_linear"), (1, "crossing_occlusion")],
        "valid": [(2, "empty"), (2, "delayed_linear"), (2, "multi_target")],
        "test": [(3, "crossing_occlusion"), (3, "waypoint_ping_pong"),
                 (3, "head_on_cylinder")],
    }
    entries = {}
    for map_id in sorted({item[0] for values in definitions.values() for item in values}):
        entries[map_id] = map_entry(map_id, args.static_root / f"pointcloud-{map_id}.ply")
    (output / "splits").mkdir(parents=True)
    sequence_number = 1
    split_ids = {}
    for split, values in definitions.items():
        split_ids[split] = []
        for map_id, scenario in values:
            sequence_id = f"phase8_{split}_{sequence_number:04d}"
            seed = 800000 + sequence_number
            generate_sequence(output, sequence_id, seed, scenario, args.frames,
                              map_id=map_id,
                              static_map_sha256=entries[map_id]["static_map_sha256"])
            split_ids[split].append(sequence_id)
            sequence_number += 1
        (output / "splits" / f"{split}.txt").write_text(
            "".join(f"{value}\n" for value in split_ids[split]), encoding="utf-8"
        )
    dump_yaml(output / "map_catalog.yaml", {"catalog_version": 1,
                                              "maps": list(entries.values())})
    dump_yaml(output / "dataset_manifest.yaml", {
        "dataset_version": "dep_dynamic_sequence_v1", "sensor_source": "depth",
        "time_unit": "second", "distance_unit": "meter",
        "map_catalog": "map_catalog.yaml",
        "map_splits": {split: sorted({map_id for map_id, _ in values})
                       for split, values in definitions.items()},
        "splits": {split: f"splits/{split}.txt" for split in definitions},
        "generation_scope": "bounded_synthetic_preflight_only",
        "formal_pointcloud_training_allowed": False,
    })
    print(f"PHASE8_MAP_DATASET_RESULT status=PASS maps={len(entries)} sequences={sequence_number - 1}")


if __name__ == "__main__":
    main()
