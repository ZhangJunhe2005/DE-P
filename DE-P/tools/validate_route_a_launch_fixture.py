#!/usr/bin/env python3
"""Validate deterministic Route-A closed-loop launch fixtures.

This validator is deliberately independent of RViz and ROS.  It reads the
canonical occupancy authority and checks that a fixture starts in the same
state-sampling domain used by upstream YOPO, has meaningful static clearance,
and has an initially observable route in the 90 degree forward camera sector.
It is a fixture-quality check, not a claim that the complete route is safe.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometry_authority.static_v1 import (
    CONTACT_TOLERANCE_M,
    StaticAuthorityMap,
)


CONTRACT_VERSION = "route_a_launch_fixture_v1"
OFFICIAL_SAMPLE_DOMAIN = {
    "x_m": [-20.0, 20.0],
    "y_m": [-20.0, 20.0],
    "z_m": [0.5, 4.0],
}
EXPECTED_CONTRACT = {
    "contract_version": CONTRACT_VERSION,
    "official_yopo_sample_domain": OFFICIAL_SAMPLE_DOMAIN,
    "uav_radius_m": 0.3,
    "minimum_center_clearance_m": 1.5,
    "minimum_uav_surface_gap_m": 1.0,
    "horizontal_fov_deg": 90.0,
    "fov_ray_count": 19,
    "fov_range_m": 6.0,
    "ray_step_m": 0.1,
    "minimum_fov_p10_open_range_m": 3.0,
    "minimum_open_first_segment_m": 3.0,
}


def _as_vector(value, label):
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{label} must be a finite XYZ vector")
    return vector


def _check_contract(contract):
    if not isinstance(contract, dict):
        raise ValueError("launch_fixture_contract must be an object")
    for key, expected in EXPECTED_CONTRACT.items():
        actual = contract.get(key)
        if actual != expected:
            raise ValueError(
                f"launch fixture contract {key} mismatch: "
                f"expected {expected!r}, got {actual!r}"
            )
    return contract


def _inside_official_sample_domain(point):
    point = _as_vector(point, "sample point")
    bounds = np.asarray([
        OFFICIAL_SAMPLE_DOMAIN["x_m"],
        OFFICIAL_SAMPLE_DOMAIN["y_m"],
        OFFICIAL_SAMPLE_DOMAIN["z_m"],
    ], dtype=np.float64)
    return bool(np.all(point >= bounds[:, 0]) and np.all(point <= bounds[:, 1]))


def _sphere_collides(authority, center, radius):
    """Exact local sphere-vs-occupied-voxel query without a global scan."""
    center = _as_vector(center, "sphere center")
    radius = float(radius)
    if radius < 0.0 or not math.isfinite(radius):
        raise ValueError("sphere radius must be finite and non-negative")
    if np.any(center - radius < authority.bounds_min - CONTACT_TOLERANCE_M):
        return True
    if np.any(center + radius > authority.bounds_max + CONTACT_TOLERANCE_M):
        return True

    lower = np.floor(
        (center - radius - authority.origin) / authority.resolution
    ).astype(np.int64)
    upper = np.floor(
        (center + radius - authority.origin) / authority.resolution
    ).astype(np.int64)
    lower = np.maximum(lower, 0)
    upper = np.minimum(upper, authority.dimensions - 1)
    grid = authority.flat.reshape(tuple(authority.dimensions))
    occupied = np.argwhere(grid[
        lower[0]:upper[0] + 1,
        lower[1]:upper[1] + 1,
        lower[2]:upper[2] + 1,
    ])
    if not len(occupied):
        return False
    occupied += lower
    box_min = authority.origin + occupied * authority.resolution
    box_max = box_min + authority.resolution
    delta = np.maximum(np.maximum(box_min - center, center - box_max), 0.0)
    squared_distance = np.einsum("ij,ij->i", delta, delta)
    return bool(np.any(
        squared_distance <= radius * radius + CONTACT_TOLERANCE_M
    ))


def _forward_fov_open_ranges(authority, start, goal, contract):
    direction = goal - start
    horizontal_norm = float(np.linalg.norm(direction[:2]))
    if horizontal_norm <= 1e-9:
        raise ValueError("launch fixture requires a non-zero horizontal route")
    heading = math.atan2(direction[1], direction[0])
    angles = np.linspace(
        -0.5 * float(contract["horizontal_fov_deg"]),
        0.5 * float(contract["horizontal_fov_deg"]),
        int(contract["fov_ray_count"]),
    )
    maximum = float(contract["fov_range_m"])
    step = float(contract["ray_step_m"])
    steps = int(math.floor(maximum / step + 1e-9))
    radius = float(contract["uav_radius_m"])
    ranges = []
    for angle_deg in angles:
        angle = heading + math.radians(float(angle_deg))
        ray = np.asarray([math.cos(angle), math.sin(angle), 0.0])
        open_range = 0.0
        for index in range(1, steps + 1):
            distance = index * step
            if _sphere_collides(authority, start + ray * distance, radius):
                break
            open_range = distance
        ranges.append(open_range)
    return np.asarray(ranges, dtype=np.float64)


def _rounded(value):
    return round(float(value), 9)


def validate_scene(scene_name, scene, contract, *, root=ROOT):
    contract = _check_contract(contract)
    start = _as_vector(scene.get("start"), f"{scene_name}.start")
    goal = _as_vector(
        scene.get("suggested_goal"), f"{scene_name}.suggested_goal"
    )
    if not _inside_official_sample_domain(start):
        raise ValueError(f"{scene_name}: start is outside official YOPO sample domain")
    if not _inside_official_sample_domain(goal):
        raise ValueError(f"{scene_name}: goal is outside official YOPO sample domain")

    authority_root = (root / scene["authority_root"]).resolve()
    authority = StaticAuthorityMap(
        authority_root, expected_map_uuid=str(scene["map_uuid"])
    )
    radius = float(contract["uav_radius_m"])
    start_query = authority.query(start, radius=radius)
    goal_query = authority.query(goal, radius=radius)
    if bool(start_query.collision) or bool(goal_query.collision):
        raise ValueError(f"{scene_name}: launch endpoint collides with authority")

    start_surface_gap = float(start_query.minimum_gap_m)
    goal_surface_gap = float(goal_query.minimum_gap_m)
    minimum_surface_gap = min(start_surface_gap, goal_surface_gap)
    minimum_center_clearance = minimum_surface_gap + radius
    if minimum_center_clearance + 1e-9 < float(
        contract["minimum_center_clearance_m"]
    ):
        raise ValueError(f"{scene_name}: insufficient center clearance")
    if minimum_surface_gap + 1e-9 < float(
        contract["minimum_uav_surface_gap_m"]
    ):
        raise ValueError(f"{scene_name}: insufficient UAV surface gap")

    open_ranges = _forward_fov_open_ranges(
        authority, start, goal, contract
    )
    fov_p10 = float(np.percentile(open_ranges, 10.0))
    center_range = float(open_ranges[len(open_ranges) // 2])
    if fov_p10 + 1e-9 < float(contract["minimum_fov_p10_open_range_m"]):
        raise ValueError(f"{scene_name}: forward 90-degree FOV is locally blocked")
    if center_range + 1e-9 < float(contract["minimum_open_first_segment_m"]):
        raise ValueError(f"{scene_name}: first route segment is blocked")

    declared_path = float(scene.get("reference_path_length_m", math.nan))
    direct_distance = float(np.linalg.norm(goal - start))
    if not math.isfinite(declared_path) or declared_path + 1e-9 < direct_distance:
        raise ValueError(
            f"{scene_name}: reference path must be declared and no shorter "
            "than the endpoint distance"
        )

    result = {
        "status": "PASS",
        "contract_version": CONTRACT_VERSION,
        "scene": scene_name,
        "map_uuid": str(scene["map_uuid"]),
        "authority_manifest_hash": authority.metadata[
            "artifact_manifest_hash"
        ],
        "official_sample_domain": True,
        "start_center_clearance_m": _rounded(start_surface_gap + radius),
        "goal_center_clearance_m": _rounded(goal_surface_gap + radius),
        "start_uav_surface_gap_m": _rounded(start_surface_gap),
        "goal_uav_surface_gap_m": _rounded(goal_surface_gap),
        "fov_p10_open_range_m": _rounded(fov_p10),
        "first_segment_open_range_m": _rounded(center_range),
        "reference_euclidean_distance_m": _rounded(direct_distance),
        "reference_path_length_m": _rounded(declared_path),
    }
    certificate = scene.get("launch_certificate")
    if not isinstance(certificate, dict):
        raise ValueError(f"{scene_name}: launch_certificate is required")
    for key in (
        "contract_version", "authority_manifest_hash",
        "start_center_clearance_m", "goal_center_clearance_m",
        "start_uav_surface_gap_m", "goal_uav_surface_gap_m",
        "fov_p10_open_range_m", "first_segment_open_range_m",
        "reference_euclidean_distance_m", "reference_path_length_m",
    ):
        expected = certificate.get(key)
        actual = result[key]
        if isinstance(actual, float):
            if expected is None or not math.isclose(
                float(expected), actual, rel_tol=0.0, abs_tol=1e-6
            ):
                raise ValueError(
                    f"{scene_name}: stale launch certificate field {key}"
                )
        elif expected != actual:
            raise ValueError(
                f"{scene_name}: stale launch certificate field {key}"
            )
    return result


def validate_scene_config(config_path, *, scenes=None, root=ROOT):
    config_path = Path(config_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    contract = payload.get("launch_fixture_contract")
    available = payload.get("scenes", {})
    selected = sorted(available) if scenes is None else list(scenes)
    results = {}
    for name in selected:
        if name not in available:
            raise ValueError(f"unknown scene: {name}")
        results[name] = validate_scene(
            name, available[name], contract, root=Path(root)
        )
    return {
        "status": "PASS",
        "contract_version": CONTRACT_VERSION,
        "scenes": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs/dep_interactive_demo_scenes_v4_6.json",
    )
    parser.add_argument("--scene", action="append", dest="scenes")
    args = parser.parse_args()
    print(json.dumps(
        validate_scene_config(args.config, scenes=args.scenes),
        indent=2, sort_keys=True,
    ))


if __name__ == "__main__":
    main()
