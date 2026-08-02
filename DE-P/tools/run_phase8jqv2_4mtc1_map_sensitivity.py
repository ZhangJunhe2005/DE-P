#!/usr/bin/env python3
"""Predeclared, bounded development-only map-type sensitivity experiment."""

from __future__ import annotations

from collections import Counter
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy import ndimage
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from authoritative_dataset.exact_backend_v1 import ExactAuthorityBVH
from authoritative_dataset.occlusion_constructor_v2_2 import (
    canonical_surface_patches, feasibility_interval,
)
from phase8jqv2_4m1_mixed_maps import build_one

OUTPUT = ROOT / "data/phase8_gap1_map_type_capability_review_v1"
REPORTS = ROOT / "reports"
DIAGNOSTICS = ROOT / "diagnostics/phase8jqv2_4mtc1/parameter_sensitivity"
PROFILE_PATH = ROOT / "configs/mixed_scene_map_profiles_v2.yaml"
TYPE_NAME = {1: "cave", 2: "pillar", 6: "room", 7: "wall"}
SEED_BASE = {1: 1_732_100_000, 2: 1_732_200_000,
             6: 1_732_600_000, 7: 1_732_700_000}
STANDOFFS = (.355, .45, .60, .80, 1.00, 1.20)
SPEEDS = (1.40, 1.55, 1.70)


def atomic_json(path: Path, value) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() == payload:
            return
        raise FileExistsError(f"refusing to overwrite MTC1 evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload)
    os.replace(temporary, path)


def planned_profiles():
    document = yaml.safe_load(PROFILE_PATH.read_text())
    common = document["common"]
    bases = {}
    for name, row in document["profiles"].items():
        bases.setdefault(int(row["maze_type"]), {
            **common, **row, "profile_name": name,
        })
    specs = {
        1: [
            {"fill": value} for value in (.06, .10, .14, .18, .22)
        ],
        2: [
            {"obstacle_number": 36, "width_min": .8, "width_max": 1.3},
            {"obstacle_number": 54, "width_min": 1.2, "width_max": 2.0},
            {"obstacle_number": 78, "width_min": .7, "width_max": 1.2},
        ],
        6: [
            {"room_number": 2, "max_windows": 1,
             "window_size_min": 1.0, "window_size_max": 1.4},
            {"room_number": 3, "max_windows": 1,
             "window_size_min": 1.2, "window_size_max": 1.8},
            {"room_number": 4, "max_windows": 2,
             "window_size_min": 1.2, "window_size_max": 1.8},
        ],
        7: [
            {"wall_number": 22, "wall_width_min": 3.0,
             "wall_width_max": 7.0, "wall_thick": .2},
            {"wall_number": 32, "wall_width_min": 2.5,
             "wall_width_max": 6.0, "wall_thick": .3},
            {"wall_number": 44, "wall_width_min": 1.5,
             "wall_width_max": 4.5, "wall_thick": .4},
        ],
    }
    rows = []
    for maze_type, variants in specs.items():
        for profile_index, changes in enumerate(variants):
            profile = dict(bases[maze_type])
            profile.update(changes)
            profile["profile_name"] = (
                f"mtc1_{TYPE_NAME[maze_type]}_p{profile_index}"
            )
            profile["occlusion_annex"] = False
            for seed_offset in range(2):
                rows.append({
                    "maze_type": maze_type,
                    "natural_type": TYPE_NAME[maze_type],
                    "profile_index": profile_index,
                    "profile": profile,
                    "seed": SEED_BASE[maze_type] + profile_index * 2
                    + seed_offset,
                    "development_only": True,
                })
    return rows


def largest_free_fraction(occupied):
    free = ~occupied[::2, ::2, ::2]
    labels, count = ndimage.label(
        free, structure=ndimage.generate_binary_structure(3, 1)
    )
    if not count or not free.sum():
        return 0.0
    sizes = np.bincount(labels.ravel())
    return float(sizes[1:].max() / free.sum())


def metrics(row):
    backend = ExactAuthorityBVH(row["authority_root"])
    authority_metadata = json.loads(
        (Path(row["authority_root"]) / "occupancy_metadata.json").read_text()
    )
    occupied = np.asarray(backend.map.flat, dtype=bool).reshape(
        tuple(map(int, backend.map.dimensions))
    )
    patches = canonical_surface_patches(backend)
    ordered = sorted(
        patches,
        key=lambda p: (
            -min(p.horizontal_extent_m, 4.0)
            * min(p.vertical_extent_m, 3.0),
            p.voxel_offset,
        ),
    )
    if len(ordered) <= 128:
        selected = ordered
    else:
        broad = ordered[:64]
        remaining = ordered[64:]
        indices = np.linspace(0, len(remaining) - 1, 64, dtype=np.int64)
        selected = broad + [remaining[int(index)] for index in indices]
    admissible = 0
    two_sided = 0
    intervals = 0
    rejection = Counter()
    for patch in selected:
        center = np.asarray(patch.center)
        normal = np.asarray(patch.normal)
        face = center + normal * (.5 * backend.map.resolution)
        patch_has = False
        for standoff in STANDOFFS:
            camera = face + normal * standoff
            if backend.query_one(camera, .30)["collision"]:
                rejection["camera_collision_or_oob"] += 1
                continue
            distance = float(np.linalg.norm(center - camera))
            feasible = [
                feasibility_interval(
                    distance, patch.horizontal_extent_m,
                    patch.vertical_extent_m, 1, speed,
                    detection_distance_m=1.8,
                ) for speed in SPEEDS
            ]
            feasible = [value for value in feasible if value["feasible"]]
            if not feasible:
                rejection["v2_2_interval_empty"] += 1
                continue
            interval = max(
                feasible,
                key=lambda value:
                    value["upper_ratio"] - value["lower_ratio"],
            )
            ratio = .5 * (
                interval["lower_ratio"] + interval["upper_ratio"]
            )
            actor_anchor = camera - normal * (ratio * distance)
            if backend.query_one(actor_anchor, .30)["collision"]:
                rejection["actor_anchor_collision_or_oob"] += 1
                continue
            two_sided += 1
            intervals += 1
            patch_has = True
        admissible += int(patch_has)
    _, occupied_components = ndimage.label(
        occupied, structure=ndimage.generate_binary_structure(3, 1)
    )
    vertical = [float(p.vertical_extent_m) for p in selected]
    horizontal = [float(p.horizontal_extent_m) for p in selected]
    return {
        "map_uuid": row["map_uuid"],
        "maze_type": int(row["maze_type"]),
        "natural_type": TYPE_NAME[int(row["maze_type"])],
        "seed": int(row["seed"]),
        "profile_name": row["profile_name"],
        "resolved_parameters": row["resolved_parameters"],
        "authority_root": row["authority_root"],
        "authority_hash": row["authority_manifest_hash"],
        "raw_cloud_hash": row["raw_cloud_hash"],
        "raw_occupied_point_count":
            int(authority_metadata["source_point_count"]),
        "occupied_voxel_fraction": float(occupied.mean()),
        "occupied_connected_components": int(occupied_components),
        "largest_free_space_component_fraction_stride2":
            largest_free_fraction(occupied),
        "surface_patch_count": len(patches),
        "selected_patch_count": len(selected),
        "selected_horizontal_extent_max_m": max(horizontal, default=0.0),
        "selected_vertical_extent_max_m": max(vertical, default=0.0),
        "vertical_solid_patch_count": sum(value >= .6 for value in vertical),
        "two_sided_free_anchor_count": two_sided,
        "admissible_patch_count": admissible,
        "feasible_ratio_interval_count": intervals,
        "exact_physically_safe_trajectory_count":
            None,
        "cuda_projected_actor_cases": None,
        "cuda_partial_occlusion_cases": None,
        "cuda_full_occlusion_cases": None,
        "exact_gap1_count": None,
        "cuda_metrics_status":
            "DEFERRED_TO_BOUNDED_NEAR_MISS_AND_PROOF_STAGE",
        "rejection_counts": dict(rejection),
        "annex_used": False,
        "development_only": True,
    }


def summarize(rows, maze_type):
    values = [row for row in rows if row["maze_type"] == maze_type]
    grouped = {}
    for row in values:
        grouped.setdefault(row["profile_name"], []).append(row)
    return {
        "status": "PASS",
        "maze_type": maze_type,
        "natural_type": TYPE_NAME[maze_type],
        "profile_count": len(grouped),
        "seed_count": len({row["seed"] for row in values}),
        "maps": values,
        "profile_aggregates": [{
            "profile_name": name,
            "map_count": len(group),
            "parameters": group[0]["resolved_parameters"],
            "occupied_voxel_fraction_mean":
                float(np.mean([x["occupied_voxel_fraction"] for x in group])),
            "largest_free_space_component_fraction_mean": float(np.mean([
                x["largest_free_space_component_fraction_stride2"]
                for x in group
            ])),
            "admissible_patch_count_sum":
                sum(x["admissible_patch_count"] for x in group),
            "two_sided_free_anchor_count_sum":
                sum(x["two_sided_free_anchor_count"] for x in group),
            "feasible_ratio_interval_count_sum":
                sum(x["feasible_ratio_interval_count"] for x in group),
        } for name, group in sorted(grouped.items())],
        "bounded_grid": True,
        "no_seventh_profile": len(grouped) <= 6,
        "annex_used": False,
    }


def main():
    entry = json.loads(
        (REPORTS / "phase8jqv2_4mtc1_entry_gate.json").read_text()
    )
    if entry["status"] != "PASS":
        raise RuntimeError("MTC1 entry gate is not PASS")
    planned = planned_profiles()
    grid = {
        "status": "FROZEN_BEFORE_CUDA",
        "rows": [{
            "maze_type": row["maze_type"],
            "natural_type": row["natural_type"],
            "profile_index": row["profile_index"],
            "profile_name": row["profile"]["profile_name"],
            "seed": row["seed"],
            "parameters": row["profile"],
        } for row in planned],
        "counts": dict(Counter(row["natural_type"] for row in planned)),
        "maximum_profiles_per_type": 6,
        "seeds_per_profile": 2,
        "annex_used": False,
    }
    atomic_json(DIAGNOSTICS / "predeclared_grid.json", grid)
    started = time.perf_counter()
    generated = []
    for index, item in enumerate(planned):
        generated.append(build_one(
            OUTPUT, "development", item["profile"], item["seed"], index, True
        ))
        print(json.dumps({
            "status": "MAP_READY", "map": f"{index + 1}/{len(planned)}",
            "type": item["natural_type"], "seed": item["seed"],
        }), flush=True)
    measured = []
    # Geometry extraction is CPU-only and map-independent. Four spawn workers
    # keep memory bounded while using more than one host core.
    with mp.get_context("spawn").Pool(processes=4) as pool:
        iterator = pool.imap(metrics, generated)
        for index, (row, result) in enumerate(zip(generated, iterator)):
            measured.append(result)
            atomic_json(DIAGNOSTICS / f"{row['map_uuid']}.json", result)
            print(json.dumps({
                "status": "METRICS_READY",
                "map": f"{index + 1}/{len(generated)}",
                "type": result["natural_type"],
                "patches": result["surface_patch_count"],
                "admissible": result["admissible_patch_count"],
            }), flush=True)
    cave = summarize(measured, 1)
    cave["fill_values"] = [.06, .10, .14, .18, .22]
    occ = [
        x["occupied_voxel_fraction_mean"]
        for x in cave["profile_aggregates"]
    ]
    cave["occupancy_fraction_monotonic_non_decreasing"] = all(
        left <= right + 1e-12 for left, right in zip(occ, occ[1:])
    )
    cave["full_occlusion_monotonicity"] = "NOT_YET_ESTABLISHED"
    cave["free_space_tradeoff"] = "MEASURED_GEOMETRY_ONLY"
    cave["third_type_gate_implication"] = (
        "cave results cannot satisfy the required non-cave/forest third type"
    )
    atomic_json(REPORTS / "phase8jqv2_4mtc1_cave_fill_sensitivity.json",
                cave)
    for maze_type, filename in (
        (2, "phase8jqv2_4mtc1_pillar_parameter_sensitivity.json"),
        (6, "phase8jqv2_4mtc1_room_parameter_sensitivity.json"),
        (7, "phase8jqv2_4mtc1_wall_parameter_sensitivity.json"),
    ):
        atomic_json(REPORTS / filename, summarize(measured, maze_type))
    atomic_json(REPORTS / "phase8jqv2_4mtc1_cross_type_sensitivity.json", {
        "status": "PASS_GEOMETRY_STAGE",
        "grid_frozen_before_cuda": True,
        "map_count": len(measured),
        "types": {
            TYPE_NAME[value]: summarize(measured, value)["profile_aggregates"]
            for value in (1, 2, 6, 7)
        },
        "common_metrics": [
            "occupied_voxel_fraction", "occupied_connected_components",
            "largest_free_space_component_fraction_stride2",
            "surface_patch_count", "vertical_solid_patch_count",
            "admissible_patch_count", "two_sided_free_anchor_count",
            "feasible_ratio_interval_count",
        ],
        "cuda_metrics_pending": True,
        "detector_executed": False,
        "tracker_executed": False,
        "elapsed_seconds": time.perf_counter() - started,
    })
    conclusion = (
        "# MTC1 Cave fill conclusion\n\n"
        "This paired two-seed sweep is diagnostic and applies only to "
        "`maze_type=1`. Raw-cloud perturbation independently proved that "
        "`fill` leaves pillar, forest, room, and wall byte-identical. "
        f"Canonical occupancy was monotonic non-decreasing: "
        f"`{cave['occupancy_fraction_monotonic_non_decreasing']}`. "
        "Higher cave fill therefore cannot establish the required third "
        "non-cave/forest map type and cannot open the corpus gate. CUDA "
        "gap metrics remain reserved for the bounded near-miss/proof stage.\n"
    )
    path = REPORTS / "phase8jqv2_4mtc1_cave_fill_conclusion.md"
    if path.exists() and path.read_text() != conclusion:
        raise FileExistsError(path)
    path.write_text(conclusion)
    print(json.dumps({
        "status": "PASS_GEOMETRY_STAGE",
        "maps": len(measured),
        "elapsed_seconds": time.perf_counter() - started,
    }, indent=2))


if __name__ == "__main__":
    main()
