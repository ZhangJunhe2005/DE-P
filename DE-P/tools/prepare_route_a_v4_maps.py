#!/usr/bin/env python3
"""Prepare/generate independent Route-A V4 mixed maps using the YOPO backend."""

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


PROFILE_PATH = ROOT / "configs/mixed_scene_map_profiles_v4_route_a.yaml"
OUTPUT = ROOT / "data/route_a_v4_mixed_maps"
PLAN = OUTPUT / "manifests/map_plan.json"
NAMESPACE = uuid.UUID("039df2a4-7d42-5b76-b7ba-943175d1eb36")
COUNTS = {"train": 48, "valid": 12}
SEEDS = {"train": 842000000, "valid": 843000000}


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_profiles():
    document = yaml.safe_load(PROFILE_PATH.read_text())
    if document["profiles_version"] != "mixed_scene_map_profiles_v4_route_a":
        raise RuntimeError("V4 profile version mismatch")
    common = document["common"]
    result = []
    for name, values in document["profiles"].items():
        row = {**common, **values, "profile_name": name}
        result.append(row)
    return result


def selected_profiles(split, profiles):
    by_size = {
        size: sorted(
            (row for row in profiles if row["size_class"] == size),
            key=lambda row: int(row["maze_type"]),
        )
        for size in ("large", "narrow")
    }
    count = COUNTS[split]
    narrow_count = 8 if split == "train" else 2
    large_count = count - narrow_count
    # Large maps always cover all five types. Narrow maps are a bounded stress
    # subset and rotate by split so their presence cannot define a map type.
    rows = [
        by_size["large"][index % 5] for index in range(large_count)
    ]
    offset = 0 if split == "train" else 3
    rows.extend(
        by_size["narrow"][(offset + index) % 5]
        for index in range(narrow_count)
    )
    return rows


def make_plan():
    profiles = load_profiles()
    result = {
        "status": "FROZEN_NOT_STARTED",
        "version": "route_a_v4_map_plan_v1",
        "profile_hash": backend.sha256(PROFILE_PATH),
        "namespace": str(NAMESPACE),
        "large_map_share": {},
        "maps": {},
    }
    for split in ("train", "valid"):
        rows = []
        selected = selected_profiles(split, profiles)
        for index, profile in enumerate(selected):
            seed = SEEDS[split] + index
            identity = f"{split}:{profile['profile_name']}:{seed}:{index}"
            rows.append({
                "map_id": f"{split}_map_{index:04d}",
                "map_uuid": str(uuid.uuid5(NAMESPACE, identity)),
                "seed": seed,
                "maze_type": int(profile["maze_type"]),
                "semantic_name": profile["semantic_name"],
                "profile_name": profile["profile_name"],
                "size_class": profile["size_class"],
                "profile_hash": backend.profile_hash(profile),
            })
        result["maps"][split] = rows
        result["large_map_share"][split] = (
            sum(row["size_class"] == "large" for row in rows) / len(rows)
        )
    result["plan_hash"] = canonical_hash(result["maps"])
    backend.atomic_json(PLAN, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def _generate_one(arguments):
    split, profile, planned, index = arguments
    backend.NAMESPACE = NAMESPACE
    row = backend.build_one(
        OUTPUT, split, profile, int(planned["seed"]), index, False,
    )
    if row["map_uuid"] != planned["map_uuid"]:
        raise RuntimeError("V4 map UUID mismatch")
    if not row["ordinary_navigation_capable"]:
        raise RuntimeError(
            f"V4 static map is not navigation-capable: {row['map_uuid']}"
        )
    row["route_a_profile_version"] = "mixed_scene_map_profiles_v4_route_a"
    row["size_class"] = planned["size_class"]
    return index, row


def generate(split, workers):
    if not PLAN.is_file():
        raise FileNotFoundError("run the plan command first")
    plan = json.loads(PLAN.read_text())
    profiles = {row["profile_name"]: row for row in load_profiles()}
    work = [
        (split, profiles[planned["profile_name"]], planned, index)
        for index, planned in enumerate(plan["maps"][split])
    ]
    rows = [None] * len(work)
    workers = max(1, min(int(workers), len(work)))
    if workers == 1:
        completed = map(_generate_one, work)
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
        if workers > 1:
            executor.shutdown(wait=True, cancel_futures=True)
    backend.atomic_json(OUTPUT / f"manifests/{split}_maps.json", {
        "status": "PASS",
        "split": split,
        "maps": rows,
        "map_type_counts": dict(Counter(row["semantic_name"] for row in rows)),
        "size_class_counts": dict(Counter(row["size_class"] for row in rows)),
    })
    print(json.dumps({"status": "PASS", "split": split, "maps": len(rows)}))


def status():
    plan = json.loads(PLAN.read_text()) if PLAN.is_file() else {"maps": {}}
    result = {"status": "READY", "splits": {}}
    for split in ("train", "valid"):
        expected = len(plan.get("maps", {}).get(split, ()))
        manifest = OUTPUT / f"manifests/{split}_maps.json"
        complete = 0
        if manifest.is_file():
            complete = len(json.loads(manifest.read_text())["maps"])
        result["splits"][split] = f"{complete}/{expected}"
        if not expected or complete != expected:
            result["status"] = "INCOMPLETE"
    print(json.dumps(result, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("plan", "generate", "status"))
    parser.add_argument("--split", choices=("train", "valid"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.command == "plan":
        make_plan()
    elif args.command == "generate":
        if not args.split:
            parser.error("generate requires --split")
        generate(args.split, args.workers)
    else:
        status()


if __name__ == "__main__":
    main()
