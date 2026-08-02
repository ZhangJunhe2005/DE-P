#!/usr/bin/env python3
"""Bounded geometry-only capability audit of the 30 RR1 development maps."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from authoritative_dataset.exact_backend_v1 import ExactAuthorityBVH
from authoritative_dataset.occlusion_constructor_v2_2 import (
    canonical_surface_patches, feasibility_interval,
)


INVENTORY = ROOT / "diagnostics/phase8jqv2_4rr1/corpus_generation/allowed_map_inventory.json"
REPORT = ROOT / "reports/phase8jqv2_4ce1_existing_map_capability.json"
TYPE_REPORT = ROOT / "reports/phase8jqv2_4ce1_map_type_capability.json"
DIAGNOSTICS = ROOT / "diagnostics/phase8jqv2_4ce1/map_capability"
TYPE_NAMES = {1: "cave", 2: "pillar", 5: "forest", 6: "room", 7: "wall"}
P0_PER_MAP = 128
STANDOFFS = (.355, .45, .60, .80, 1.00, 1.20)
SPEEDS = (1.40, 1.55, 1.70)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def choose_patches(patches):
    """Keep broad patches while preserving deterministic spatial diversity."""
    ordered = sorted(
        patches,
        key=lambda p: (
            -min(p.horizontal_extent_m, 4.0) * min(p.vertical_extent_m, 3.0),
            p.voxel_offset,
        ),
    )
    if len(ordered) <= P0_PER_MAP:
        return ordered
    broad = ordered[:P0_PER_MAP // 2]
    remaining = ordered[P0_PER_MAP // 2:]
    indices = np.linspace(
        0, len(remaining) - 1, P0_PER_MAP - len(broad), dtype=np.int64
    )
    return broad + [remaining[int(index)] for index in indices]


def audit_map(row):
    started = time.perf_counter()
    backend = ExactAuthorityBVH(row["authority_root"])
    patches = canonical_surface_patches(backend)
    selected = choose_patches(patches)
    occupancy = np.asarray(
        backend.map.flat, dtype=bool
    ).reshape(tuple(map(int, backend.map.dimensions)))
    _, connected_components = ndimage.label(
        occupancy, structure=ndimage.generate_binary_structure(3, 1)
    )
    candidate_rows = []
    rejection = Counter()
    for patch in selected:
        center = np.asarray(patch.center, dtype=np.float64)
        normal = np.asarray(patch.normal, dtype=np.float64)
        face = center + normal * (.5 * float(backend.map.resolution))
        patch_any = False
        for standoff in STANDOFFS:
            camera = face + normal * standoff
            camera_query = backend.query_one(camera, .30)
            if camera_query["collision"]:
                rejection["camera_sphere_collision_or_oob"] += 1
                continue
            distance = float(np.linalg.norm(center - camera))
            intervals = [
                feasibility_interval(
                    distance, patch.horizontal_extent_m,
                    patch.vertical_extent_m, 1, speed,
                    detection_distance_m=1.80,
                )
                for speed in SPEEDS
            ]
            feasible = [
                (speed, interval) for speed, interval in zip(SPEEDS, intervals)
                if interval["feasible"]
            ]
            if not feasible:
                rejection["v2_2_ratio_interval_empty"] += 1
                continue
            speed, interval = max(
                feasible,
                key=lambda item: item[1]["upper_ratio"] - item[1]["lower_ratio"],
            )
            ratio = .5 * (
                interval["lower_ratio"] + interval["upper_ratio"]
            )
            actor_anchor = camera + (-normal) * (ratio * distance)
            actor_query = backend.query_one(actor_anchor, .30)
            if actor_query["collision"]:
                rejection["actor_anchor_collision_or_oob"] += 1
                continue
            patch_any = True
            candidate_rows.append({
                "patch_voxel_offset": int(patch.voxel_offset),
                "patch_center": list(map(float, patch.center)),
                "patch_normal": list(map(float, patch.normal)),
                "horizontal_extent_m": float(patch.horizontal_extent_m),
                "vertical_extent_m": float(patch.vertical_extent_m),
                "camera_standoff_m": float(standoff),
                "camera_position": camera.tolist(),
                "speed_mps": float(speed),
                "ratio_interval": interval,
                "nominal_actor_anchor": actor_anchor.tolist(),
                "camera_minimum_gap_m": float(camera_query["minimum_gap_m"]),
                "actor_anchor_minimum_gap_m":
                    float(actor_query["minimum_gap_m"]),
            })
        if not patch_any:
            rejection["patch_has_no_admissible_standoff"] += 1
    # This is intentionally only a geometry broadphase, never a claim of a
    # raster-valid one-frame gap.
    horizontal_extents = [
        float(patch.horizontal_extent_m) for patch in selected
    ]
    vertical_extents = [
        float(patch.vertical_extent_m) for patch in selected
    ]
    ratios = [
        candidate["ratio_interval"] for candidate in candidate_rows
    ]
    camera_free = (
        len(selected) * len(STANDOFFS)
        - rejection["camera_sphere_collision_or_oob"]
    )
    if len(candidate_rows) >= 10:
        capability = "high_potential"
    elif candidate_rows:
        capability = "geometry_possible_search_insufficient"
    elif rejection["camera_sphere_collision_or_oob"] >= (
        len(selected) * len(STANDOFFS) * .8
    ):
        capability = "camera_access_limited"
    elif rejection["actor_anchor_collision_or_oob"]:
        capability = "actor_corridor_limited"
    elif patches:
        capability = "topology_incapable"
    else:
        capability = "no_connected_occluder_patch"
    result = {
        "map_uuid": row["map_uuid"],
        "maze_type": int(row["maze_type"]),
        "natural_type": TYPE_NAMES[int(row["maze_type"])],
        "map_seed": int(row["seed"]),
        "authority_root": row["authority_root"],
        "authority_hash": row["authority_hash"],
        "occupancy_hash": row["occupancy_hash"],
        "source": row["source"],
        "profile_name": row["profile_name"],
        "annex_used": bool(row["annex_enabled"]),
        "occupied_voxel_count": int(len(backend.occupied)),
        "occupied_connected_component_count": int(connected_components),
        "surface_patch_count": len(patches),
        "selected_patch_horizontal_extent_m": {
            "minimum": min(horizontal_extents, default=0.0),
            "median": float(np.median(horizontal_extents))
                if horizontal_extents else 0.0,
            "maximum": max(horizontal_extents, default=0.0),
        },
        "selected_patch_vertical_extent_m": {
            "minimum": min(vertical_extents, default=0.0),
            "median": float(np.median(vertical_extents))
                if vertical_extents else 0.0,
            "maximum": max(vertical_extents, default=0.0),
        },
        "p0_patch_checks": len(selected),
        "p0_budget": P0_PER_MAP,
        "camera_side_free_pose_count": int(camera_free),
        "actor_side_free_anchor_count": len(candidate_rows),
        "feasible_camera_standoffs_m": sorted({
            candidate["camera_standoff_m"] for candidate in candidate_rows
        }),
        "feasible_actor_depth_ratio_bounds": {
            "minimum_lower": min(
                (value["lower_ratio"] for value in ratios), default=None
            ),
            "maximum_upper": max(
                (value["upper_ratio"] for value in ratios), default=None
            ),
        },
        "straight_motion_length": {
            "status": "NOT_CERTIFIED_IN_BROADPHASE",
            "reason": "requires exact continuous trajectory stage",
        },
        "camera_motion_family_potential": (
            [
                "static", "yaw_left", "yaw_right",
                "lateral_linear_positive", "lateral_linear_negative",
                "retreat_linear", "approach_linear",
                "yaw_lateral", "yaw_forward",
            ] if candidate_rows else []
        ),
        "gap1_temporal_width_estimate": {
            "feasible_interval_count": len(ratios),
            "minimum_gap_duration_s": .1,
            "basis": "v2.2 feasibility_interval analytic bound",
        },
        "projected_actor_support_estimate": {
            "status": "POSITIVE_EXPECTED_NOT_RASTER_CERTIFIED"
                if candidate_rows else "NOT_ESTABLISHED",
            "exact_pixels": None,
        },
        "expected_pre_post_visibility_window": {
            "pre_frames": 4,
            "post_frames": 3,
            "status": "NOT_CERTIFIED_IN_BROADPHASE",
        },
        "admissible_patch_standoff_count": len(candidate_rows),
        "geometry_broadphase_capable": bool(candidate_rows),
        "capability_class": capability,
        "rejection_counts": dict(sorted(rejection.items())),
        "candidate_digest": hashlib.sha256(json.dumps(
            candidate_rows, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(DIAGNOSTICS / f"{row['map_uuid']}.json", {
        **result,
        "candidates": candidate_rows,
        "evidence_boundary": (
            "geometry broadphase only; no exact trajectory, continuous "
            "certificate, CUDA visibility, or representation probe"
        ),
    })
    return result


def main():
    inventory = json.loads(INVENTORY.read_text())
    if inventory["status"] != "PASS" or len(inventory["maps"]) != 30:
        raise RuntimeError("frozen 30-map RR1 inventory is invalid")
    started = time.perf_counter()
    rows = []
    for index, row in enumerate(inventory["maps"], 1):
        result = audit_map(row)
        rows.append(result)
        print(json.dumps({
            "map": f"{index}/30",
            "map_uuid": result["map_uuid"],
            "type": result["natural_type"],
            "broadphase_candidates":
                result["admissible_patch_standoff_count"],
            "elapsed_seconds": result["elapsed_seconds"],
        }), flush=True)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["natural_type"]].append(row)
    type_rows = {}
    for name, values in sorted(grouped.items()):
        type_rows[name] = {
            "map_count": len(values),
            "geometry_broadphase_capable_maps": sum(
                row["geometry_broadphase_capable"] for row in values
            ),
            "admissible_patch_standoff_count": sum(
                row["admissible_patch_standoff_count"] for row in values
            ),
            "surface_patch_count": sum(
                row["surface_patch_count"] for row in values
            ),
            "map_uuids": [row["map_uuid"] for row in values],
        }
    payload = {
        "status": "PASS",
        "audit_version": "phase8jqv2_4ce1_existing_map_capability_v1",
        "inventory_sha256": hashlib.sha256(INVENTORY.read_bytes()).hexdigest(),
        "map_count": len(rows),
        "p0_total_budget": 30 * P0_PER_MAP,
        "p0_total_used": sum(row["p0_patch_checks"] for row in rows),
        "geometry_broadphase_capable_maps": sum(
            row["geometry_broadphase_capable"] for row in rows
        ),
        "maps": rows,
        "elapsed_seconds": time.perf_counter() - started,
        "evidence_boundary": (
            "Capability means only frozen-authority geometry broadphase. "
            "Exact one-frame CUDA visibility remains to be established by E1."
        ),
        "annex_used": False,
        "sealed_holdout_accessed": False,
        "formal_generation_started": False,
        "training_started": False,
    }
    atomic_json(REPORT, payload)
    atomic_json(TYPE_REPORT, {
        "status": "PASS",
        "audit_version": payload["audit_version"],
        "types": type_rows,
        "evidence_boundary": payload["evidence_boundary"],
    })
    print(json.dumps({
        "status": "PASS",
        "maps": len(rows),
        "capable": payload["geometry_broadphase_capable_maps"],
        "elapsed_seconds": payload["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
