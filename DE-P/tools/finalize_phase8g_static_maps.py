#!/usr/bin/env python3
"""Build a split-isolated catalog for newly generated Phase 8G maps."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ruamel.yaml import YAML


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--split", action="append", nargs=5, metavar=(
        "NAME", "COUNT", "MAP_SEED", "POSE_SEED", "ACTOR_SEED"
    ), required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    yaml = YAML(typ="safe")
    entries, hashes = [], set()
    next_id = 0
    split_counts = {}
    for name, count_text, map_seed_text, pose_seed_text, actor_seed_text in args.split:
        count, map_seed, pose_seed, actor_seed = map(
            int, (count_text, map_seed_text, pose_seed_text, actor_seed_text)
        )
        split_counts[name] = count
        directory = root / name
        generation = yaml.load(directory / "generation_metadata.yaml")
        actual = tuple(int(generation[key]) for key in (
            "map_seed", "pose_seed", "actor_seed"
        ))
        if actual != (map_seed, pose_seed, actor_seed):
            raise ValueError(f"{name}: seed mismatch {actual}")
        reachability = json.loads(
            (directory / "reachability_metadata.json").read_text()
        )
        indexed = {int(item["map_id"]): item for item in reachability["maps"]}
        if set(indexed) != set(range(count)):
            raise ValueError(f"{name}: expected map IDs 0..{count - 1}")
        for local_id in range(count):
            ply = directory / f"pointcloud-{local_id}.ply"
            pose = directory / f"pose-{local_id}.csv"
            item = indexed[local_id]
            ply_hash, pose_hash = digest(ply), digest(pose)
            if ply_hash != item["ply_sha256"] or pose_hash != item["pose_csv_sha256"]:
                raise ValueError(f"{name}/{local_id}: reachability hash mismatch")
            if ply_hash in hashes:
                raise ValueError("map leaked across Phase 8G splits")
            hashes.add(ply_hash)
            entries.append({
                "map_id": next_id, "split": name, "local_map_id": local_id,
                "map_seed": map_seed + local_id, "pose_seed": pose_seed,
                "actor_seed_base": actor_seed + local_id * 1000,
                "static_ply": str(ply), "pose_csv": str(pose),
                "static_map_sha256": ply_hash, "pose_csv_sha256": pose_hash,
                "reachability": item,
            })
            next_id += 1
    catalog = {
        "catalog_version": 3,
        "generation_scope": "phase8g_instance_perception_protocol",
        "split_counts": split_counts,
        "maps": entries,
    }
    output = YAML()
    with (root / "map_catalog.yaml").open("w", encoding="utf-8") as stream:
        output.dump(catalog, stream)
    print(json.dumps({"status": "PASS", "maps": len(entries),
                      "split_counts": split_counts}, indent=2))


if __name__ == "__main__":
    main()
