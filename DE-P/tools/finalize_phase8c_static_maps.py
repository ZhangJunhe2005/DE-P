#!/usr/bin/env python3
"""Build an auditable catalog for split-isolated Phase 8C static maps."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ruamel.yaml import YAML


SPLITS = {"train": 12, "valid": 3, "test": 3}
SEEDS = {
    "train": (810000, 820000, 830000),
    "valid": (820000, 920000, 930000),
    "test": (830000, 1020000, 1030000),
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reachability_by_numeric_id(reach, count, split="unknown"):
    items = reach.get("maps", [])
    if len(items) != count:
        raise ValueError(f"{split}: reachability map count mismatch")
    result = {int(item["map_id"]): item for item in items}
    if len(result) != count or set(result) != set(range(count)):
        raise ValueError(
            f"{split}: reachability map IDs must be exactly 0..{count - 1}, "
            f"got {sorted(result)}"
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    yaml = YAML(typ="safe")
    entries, hashes, next_id = [], set(), 0
    for split, count in SPLITS.items():
        directory = root / split
        reach = json.loads((directory / "reachability_metadata.json").read_text())
        generation = yaml.load(directory / "generation_metadata.yaml")
        expected = SEEDS[split]
        actual = tuple(int(generation[name]) for name in ("map_seed", "pose_seed", "actor_seed"))
        if actual != expected or int(generation["env_num"]) != count:
            raise ValueError(f"{split}: generator seed/count mismatch {actual}")
        reach_by_id = reachability_by_numeric_id(reach, count, split)
        for local_id in range(count):
            item = reach_by_id[local_id]
            ply = directory / f"pointcloud-{local_id}.ply"
            pose = directory / f"pose-{local_id}.csv"
            start_goal = directory / f"start_goal-{local_id}.csv"
            for path in (ply, pose, start_goal):
                if not path.is_file():
                    raise FileNotFoundError(path)
            ply_hash = file_sha256(ply)
            pose_hash = file_sha256(pose)
            if ply_hash != item["ply_sha256"] or pose_hash != item["pose_csv_sha256"]:
                raise ValueError(f"{split}/{local_id}: reachability file hash mismatch")
            if item["ply_sha256"] in hashes:
                raise ValueError("static map hash leaked across splits")
            hashes.add(item["ply_sha256"])
            entries.append({
                "map_id": next_id,
                "split": split,
                "local_map_id": local_id,
                "map_seed": expected[0] + local_id,
                "pose_seed": expected[1],
                "actor_seed_base": expected[2] + local_id * 1000,
                "static_ply": str(ply.resolve()),
                "pose_csv": str(pose.resolve()),
                "start_goal_csv": str(start_goal.resolve()),
                "static_map_sha256": item["ply_sha256"],
                "pose_csv_sha256": item["pose_csv_sha256"],
                "reachability": item,
            })
            next_id += 1
    catalog = {
        "catalog_version": 2,
        "generation_scope": "phase8c_formal_static_maps",
        "split_counts": SPLITS,
        "maps": entries,
    }
    yaml_out = YAML()
    with (root / "map_catalog.yaml").open("w", encoding="utf-8") as stream:
        yaml_out.dump(catalog, stream)
    (root / "static_map_manifest.json").write_text(
        json.dumps({
            "status": "PASS", "map_count": len(entries),
            "split_counts": SPLITS, "unique_hash_count": len(hashes),
            "pointcloud_used_for_training": False,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "maps": len(entries)}, indent=2))


if __name__ == "__main__":
    main()
