#!/usr/bin/env python3
"""Plan and generate the four-scene Route-A V4.6 authority maps."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import uuid

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.phase8jqv2_4m1_mixed_maps as backend


PROFILE_PATH = ROOT / "configs/mixed_scene_map_profiles_v4_6_route_a.yaml"
DEFAULT_OUTPUT = ROOT / "data/route_a_v4_6_mixed_maps"
NAMESPACE = uuid.UUID("cf2f84ed-4bb3-5fcf-a6a5-e64c0e4a28c6")
COUNTS = {"train": 48, "valid": 12}
SEEDS = {"train": 846000000, "valid": 847000000}
EXPECTED_TYPES = {"cave", "pillar", "forest", "wall"}


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_profiles(path=PROFILE_PATH):
    document = yaml.safe_load(Path(path).read_text())
    if document.get("profiles_version") != "mixed_scene_map_profiles_v4_6_route_a":
        raise RuntimeError("V4.6 profile version mismatch")
    if set(document.get("supported_map_types", ())) != EXPECTED_TYPES:
        raise RuntimeError("V4.6 must contain exactly four supported map types")
    rows = []
    for name, values in document["profiles"].items():
        row = {**document["common"], **values, "profile_name": name}
        if row.get("map_type") not in EXPECTED_TYPES or int(row["maze_type"]) == 6:
            raise RuntimeError(f"room/unknown profile entered V4.6: {name}")
        rows.append(row)
    if len(rows) != 8:
        raise RuntimeError("V4.6 requires large/narrow profiles for four types")
    return rows


def selected_profiles(split, profiles):
    by_size = {
        size: sorted(
            (row for row in profiles if row["size_class"] == size),
            key=lambda row: int(row["maze_type"]),
        )
        for size in ("large", "narrow")
    }
    if any(len(rows) != 4 for rows in by_size.values()):
        raise RuntimeError("V4.6 profile matrix is incomplete")
    # 48 train = 40 large + 8 narrow; 12 valid = 8 large + 4 narrow.
    # Both splits therefore contain exactly equal map counts for all 4 types.
    narrow_count = 8 if split == "train" else 4
    large_count = COUNTS[split] - narrow_count
    return [
        by_size["large"][index % 4] for index in range(large_count)
    ] + [
        by_size["narrow"][index % 4] for index in range(narrow_count)
    ]


def plan_path(output):
    return Path(output) / "manifests/map_plan.json"


def make_plan(output):
    output = Path(output)
    result = {
        "status": "FROZEN_NOT_STARTED",
        "version": "route_a_v4_6_four_scene_map_plan_v1",
        "profile_hash": backend.sha256(PROFILE_PATH),
        "namespace": str(NAMESPACE),
        "supported_map_types": sorted(EXPECTED_TYPES),
        "room_included": False,
        "maps": {},
    }
    profiles = load_profiles()
    for split in ("train", "valid"):
        rows = []
        for index, profile in enumerate(selected_profiles(split, profiles)):
            seed = SEEDS[split] + index
            identity = f"{split}:{profile['profile_name']}:{seed}:{index}"
            rows.append({
                "map_id": f"{split}_map_{index:04d}",
                "map_uuid": str(uuid.uuid5(NAMESPACE, identity)),
                "seed": seed,
                "maze_type": int(profile["maze_type"]),
                "map_type": profile["map_type"],
                "semantic_name": profile["semantic_name"],
                "profile_name": profile["profile_name"],
                "size_class": profile["size_class"],
                "profile_hash": backend.profile_hash(profile),
            })
        counts = Counter(row["map_type"] for row in rows)
        expected = COUNTS[split] // len(EXPECTED_TYPES)
        if counts != Counter({name: expected for name in EXPECTED_TYPES}):
            raise RuntimeError(f"unbalanced V4.6 {split} plan: {counts}")
        result["maps"][split] = rows
    result["plan_hash"] = canonical_hash(result["maps"])
    backend.atomic_json(plan_path(output), result)
    print(json.dumps(result, indent=2, sort_keys=True))


def _generate_one(arguments):
    output, split, profile, planned, index = arguments
    backend.NAMESPACE = NAMESPACE
    row = backend.build_one(
        output, split, profile, int(planned["seed"]), index, False
    )
    if row["map_uuid"] != planned["map_uuid"]:
        raise RuntimeError("V4.6 map UUID mismatch")
    if not row["ordinary_navigation_capable"]:
        raise RuntimeError(f"V4.6 map is not navigation-capable: {row['map_uuid']}")
    row["route_a_profile_version"] = "mixed_scene_map_profiles_v4_6_route_a"
    row["size_class"] = planned["size_class"]
    row["map_type"] = planned["map_type"]
    return index, row


def generate(output, split, workers):
    output = Path(output)
    plan_file = plan_path(output)
    if not plan_file.is_file():
        raise FileNotFoundError("run the V4.6 plan command first")
    plan = json.loads(plan_file.read_text())
    profiles = {row["profile_name"]: row for row in load_profiles()}
    work = [
        (output, split, profiles[row["profile_name"]], row, index)
        for index, row in enumerate(plan["maps"][split])
    ]
    rows = [None] * len(work)
    workers = max(1, min(int(workers), len(work)))
    if workers == 1:
        completed = map(_generate_one, work)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        futures = [executor.submit(_generate_one, item) for item in work]
        completed = (future.result() for future in as_completed(futures))
    try:
        for count, (index, row) in enumerate(completed, 1):
            rows[index] = row
            print(json.dumps({
                "status": "MAP_PROGRESS", "split": split,
                "complete": count, "total": len(work), "workers": workers,
            }), flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    if any(row is None for row in rows):
        raise RuntimeError("V4.6 map generation returned an incomplete manifest")
    backend.atomic_json(output / f"manifests/{split}_maps.json", {
        "status": "PASS",
        "split": split,
        "profile_version": "mixed_scene_map_profiles_v4_6_route_a",
        "room_included": False,
        "maps": rows,
        "map_type_counts": dict(Counter(row["map_type"] for row in rows)),
        "size_class_counts": dict(Counter(row["size_class"] for row in rows)),
    })
    print(json.dumps({"status": "PASS", "split": split, "maps": len(rows)}))


def status(output):
    output = Path(output)
    plan = json.loads(plan_path(output).read_text()) \
        if plan_path(output).is_file() else {"maps": {}}
    result = {"status": "READY", "splits": {}}
    for split in ("train", "valid"):
        expected = len(plan.get("maps", {}).get(split, ()))
        manifest = output / f"manifests/{split}_maps.json"
        complete = len(json.loads(manifest.read_text())["maps"]) \
            if manifest.is_file() else 0
        result["splits"][split] = f"{complete}/{expected}"
        if expected != COUNTS[split] or complete != expected:
            result["status"] = "INCOMPLETE"
    print(json.dumps(result, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("plan", "generate", "status"))
    parser.add_argument("--split", choices=("train", "valid"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "plan":
        make_plan(args.output)
    elif args.command == "generate":
        if not args.split:
            parser.error("generate requires --split")
        generate(args.output, args.split, args.workers)
    else:
        status(args.output)


if __name__ == "__main__":
    main()
