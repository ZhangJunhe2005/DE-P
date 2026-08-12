#!/usr/bin/env python3
"""Plan and generate immutable four-scene Route-A V4.7 authority maps."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import uuid

import numpy as np
from scipy import ndimage
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.phase8jqv2_4m1_mixed_maps as backend


PROFILE_VERSION = "mixed_scene_map_profiles_v4_7_route_a"
PROFILE_PATH = ROOT / "configs/mixed_scene_map_profiles_v4_7_route_a.yaml"
DEFAULT_OUTPUT = ROOT / "data/route_a_v4_7_mixed_maps"
NAMESPACE = uuid.UUID("86f8e108-484e-4e5b-a84b-af0e18d72131")
COUNTS = {"train": 48, "valid": 12}
SEEDS = {"train": 856000000, "valid": 857000000}
EXPECTED_TYPES = {"cave", "pillar", "forest", "wall"}
PILLAR_SEED_STRIDE = 10_000
PILLAR_DISTRIBUTION_CONTRACT = {
    "version": "route_a_v4_7_pillar_distribution_v1",
    "sampling_source": "original_yopo_uniform_random_with_seed_rejection",
    "height_slice_m": [0.4, 1.2],
    "raster_resolution_m": 0.25,
    "coarse_cell_size_m": 10.0,
    "local_window_size_m": 6.0,
    "minimum_covered_cell_fraction": 0.85,
    "maximum_coarse_cell_coefficient_of_variation": 0.75,
    "maximum_coarse_cell_to_mean_ratio": 3.0,
    "maximum_local_window_occupied_fraction": 0.18,
    "maximum_seed_attempts": 16,
}


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_once_json(path, value):
    """Create a manifest once; accept an exact deterministic replay only."""
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise RuntimeError(f"refusing to overwrite existing V4.7 data: {path}")
        return
    backend.atomic_json(path, value)


def load_profiles(path=PROFILE_PATH):
    document = yaml.safe_load(Path(path).read_text())
    if document.get("profiles_version") != PROFILE_VERSION:
        raise RuntimeError("V4.7 profile version mismatch")
    if set(document.get("supported_map_types", ())) != EXPECTED_TYPES:
        raise RuntimeError("V4.7 must contain exactly four supported map types")
    rows = []
    for name, values in document["profiles"].items():
        row = {**document["common"], **values, "profile_name": name}
        if row.get("map_type") not in EXPECTED_TYPES or int(row["maze_type"]) == 6:
            raise RuntimeError(f"room/unknown profile entered V4.7: {name}")
        rows.append(row)
    if len(rows) != 8:
        raise RuntimeError("V4.7 requires large/narrow profiles for four types")
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
        raise RuntimeError("V4.7 profile matrix is incomplete")
    # 48 train = 40 large + 8 narrow; 12 valid = 8 large + 4 narrow.
    # Both splits contain equal counts for the four supported map types.
    narrow_count = 8 if split == "train" else 4
    large_count = COUNTS[split] - narrow_count
    return [
        by_size["large"][index % 4] for index in range(large_count)
    ] + [
        by_size["narrow"][index % 4] for index in range(narrow_count)
    ]


def pillar_distribution_metrics(points, profile):
    """Measure low-altitude spatial dispersion without inferring pillar IDs.

    The original generator emits pillar surfaces rather than center metadata.
    A low-altitude XY occupancy raster is therefore the authoritative proxy:
    coarse cells detect map-wide imbalance, while a rolling 6 m window detects
    a local clump that could seal a forward camera view.
    """
    points = np.asarray(points, dtype=np.float64)
    contract = PILLAR_DISTRIBUTION_CONTRACT
    length_x = float(profile["x_length"])
    length_y = float(profile["y_length"])
    resolution = float(contract["raster_resolution_m"])
    z_min, z_max = map(float, contract["height_slice_m"])
    selected = points[
        (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    ]
    bins_x = int(round(length_x / resolution))
    bins_y = int(round(length_y / resolution))
    x_index = np.floor((selected[:, 0] + 0.5 * length_x) / resolution).astype(int)
    y_index = np.floor((selected[:, 1] + 0.5 * length_y) / resolution).astype(int)
    valid = (
        (x_index >= 0) & (x_index < bins_x)
        & (y_index >= 0) & (y_index < bins_y)
    )
    occupied = np.zeros((bins_x, bins_y), dtype=bool)
    occupied[x_index[valid], y_index[valid]] = True

    coarse_size = float(contract["coarse_cell_size_m"])
    coarse_x = int(round(length_x / coarse_size))
    coarse_y = int(round(length_y / coarse_size))
    block_x = bins_x // coarse_x
    block_y = bins_y // coarse_y
    trimmed = occupied[:coarse_x * block_x, :coarse_y * block_y]
    counts = trimmed.reshape(
        coarse_x, block_x, coarse_y, block_y
    ).sum(axis=(1, 3)).astype(np.float64)
    mean = float(counts.mean())
    coefficient = float(counts.std() / mean) if mean > 0.0 else float("inf")
    maximum_ratio = float(counts.max() / mean) if mean > 0.0 else float("inf")
    covered_fraction = float(np.mean(counts > 0.0))
    window = max(1, int(round(
        float(contract["local_window_size_m"]) / resolution
    )))
    local_fraction = float(ndimage.uniform_filter(
        occupied.astype(np.float64), size=window, mode="constant"
    ).max())
    passed = bool(
        covered_fraction + 1e-12
        >= float(contract["minimum_covered_cell_fraction"])
        and coefficient
        <= float(contract["maximum_coarse_cell_coefficient_of_variation"])
        + 1e-12
        and maximum_ratio
        <= float(contract["maximum_coarse_cell_to_mean_ratio"]) + 1e-12
        and local_fraction
        <= float(contract["maximum_local_window_occupied_fraction"]) + 1e-12
    )
    return {
        "contract_version": contract["version"],
        "passed": passed,
        "covered_cell_fraction": covered_fraction,
        "coarse_cell_coefficient_of_variation": coefficient,
        "coarse_cell_to_mean_ratio": maximum_ratio,
        "maximum_local_window_occupied_fraction": local_fraction,
    }


def _seed_candidates(split, profile, index):
    base = int(SEEDS[split] + index)
    attempts = (
        int(PILLAR_DISTRIBUTION_CONTRACT["maximum_seed_attempts"])
        if profile["map_type"] == "pillar" else 1
    )
    return [base + attempt * PILLAR_SEED_STRIDE for attempt in range(attempts)]


def _candidate_uuid(split, profile, seed, index):
    identity = f"{split}:{profile['profile_name']}:{seed}:{index}"
    return str(uuid.uuid5(NAMESPACE, identity))


def plan_path(output):
    return Path(output) / "manifests/map_plan.json"


def make_plan(output):
    output = Path(output)
    result = {
        "status": "FROZEN_NOT_STARTED",
        "version": "route_a_v4_7_four_scene_map_plan_v2",
        "profile_hash": backend.sha256(PROFILE_PATH),
        "namespace": str(NAMESPACE),
        "supported_map_types": sorted(EXPECTED_TYPES),
        "room_included": False,
        "pillar_distribution_contract": PILLAR_DISTRIBUTION_CONTRACT,
        "maps": {},
    }
    profiles = load_profiles()
    for split in ("train", "valid"):
        rows = []
        for index, profile in enumerate(selected_profiles(split, profiles)):
            candidates = _seed_candidates(split, profile, index)
            rows.append({
                "map_id": f"{split}_map_{index:04d}",
                "map_uuid": _candidate_uuid(
                    split, profile, candidates[0], index
                ),
                "seed": candidates[0],
                "seed_candidates": candidates,
                "map_uuid_candidates": [
                    _candidate_uuid(split, profile, seed, index)
                    for seed in candidates
                ],
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
            raise RuntimeError(f"unbalanced V4.7 {split} plan: {counts}")
        result["maps"][split] = rows
    result["plan_hash"] = canonical_hash(result["maps"])
    write_once_json(plan_path(output), result)
    print(json.dumps(result, indent=2, sort_keys=True))


def _generate_one(arguments):
    output, split, profile, planned, index = arguments
    backend.NAMESPACE = NAMESPACE
    selected_seed = int(planned["seed"])
    selected_uuid = str(planned["map_uuid"])
    distribution = None
    seed_attempt = 0
    if profile["map_type"] == "pillar":
        selected = None
        for attempt, (seed, candidate_uuid) in enumerate(zip(
            planned["seed_candidates"], planned["map_uuid_candidates"]
        )):
            with tempfile.TemporaryDirectory(
                prefix="route-a-v47-pillar-", dir="/tmp"
            ) as temporary:
                raw = Path(temporary) / "pillar.bin"
                points = backend.invoke_generator(profile, int(seed), raw)
            metrics = pillar_distribution_metrics(points, profile)
            if metrics["passed"]:
                selected = (int(seed), str(candidate_uuid), attempt, metrics)
                break
        if selected is None:
            raise RuntimeError(
                "V4.7 pillar seed candidates exhausted without satisfying "
                "the spatial-dispersion contract"
            )
        selected_seed, selected_uuid, seed_attempt, distribution = selected
    row = backend.build_one(
        output, split, profile, selected_seed, index, False
    )
    if row["map_uuid"] != selected_uuid:
        raise RuntimeError("V4.7 map UUID mismatch")
    if not row["ordinary_navigation_capable"]:
        raise RuntimeError(f"V4.7 map is not navigation-capable: {row['map_uuid']}")
    row["route_a_profile_version"] = PROFILE_VERSION
    row["size_class"] = planned["size_class"]
    row["map_type"] = planned["map_type"]
    if distribution is not None:
        row["pillar_distribution_contract"] = dict(
            PILLAR_DISTRIBUTION_CONTRACT
        )
        row["pillar_distribution_metrics"] = distribution
        row["pillar_seed_attempt"] = int(seed_attempt)
    return index, row


def generate(output, split, workers):
    output = Path(output)
    plan_file = plan_path(output)
    if not plan_file.is_file():
        raise FileNotFoundError("run the V4.7 plan command first")
    plan = json.loads(plan_file.read_text())
    if plan.get("namespace") != str(NAMESPACE):
        raise RuntimeError("V4.7 plan namespace mismatch")
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
        raise RuntimeError("V4.7 map generation returned an incomplete manifest")
    manifest = {
        "status": "PASS",
        "split": split,
        "profile_version": PROFILE_VERSION,
        "room_included": False,
        "maps": rows,
        "map_type_counts": dict(Counter(row["map_type"] for row in rows)),
        "size_class_counts": dict(Counter(row["size_class"] for row in rows)),
    }
    write_once_json(output / f"manifests/{split}_maps.json", manifest)
    print(json.dumps({"status": "PASS", "split": split, "maps": len(rows)}))


def status(output):
    output = Path(output)
    plan = json.loads(plan_path(output).read_text()) \
        if plan_path(output).is_file() else {"maps": {}}
    result = {"status": "READY", "splits": {}, "pillar_distribution": {}}
    for split in ("train", "valid"):
        expected = len(plan.get("maps", {}).get(split, ()))
        manifest = output / f"manifests/{split}_maps.json"
        complete = len(json.loads(manifest.read_text())["maps"]) \
            if manifest.is_file() else 0
        result["splits"][split] = f"{complete}/{expected}"
        if manifest.is_file():
            rows = json.loads(manifest.read_text())["maps"]
            pillars = [row for row in rows if row["map_type"] == "pillar"]
            passed = sum(bool(
                row.get("pillar_distribution_metrics", {}).get("passed")
            ) for row in pillars)
            result["pillar_distribution"][split] = (
                f"{passed}/{len(pillars)}"
            )
            if passed != len(pillars):
                result["status"] = "INCOMPLETE"
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
