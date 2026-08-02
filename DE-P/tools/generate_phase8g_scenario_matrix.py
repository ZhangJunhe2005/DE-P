#!/usr/bin/env python3
"""Generate fresh Phase 8G scenarios, including never-observed controls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from ruamel.yaml import YAML
from scipy.spatial import cKDTree

from generate_phase8c_scenario_matrix import (
    actor,
    actor_id_base,
    point_and_tangent,
    scenario,
)


KINDS = (
    "no_target", "crossing", "head_on", "multi_target",
    "temporal_separation", "occluded_but_tracked",
    "always_behind", "always_outside_fov", "static_wall_occluded",
    "not_yet_active", "static_wall_edge", "visible_plus_never_nearby",
    "moving_camera_static_scene", "enters_after_sequence",
)
BASE = set(KINDS[:6])


def stationary(actor_id, position, *, start=0.0, end=8.5, radius=0.25):
    return actor(actor_id, "sphere", radius, position, {
        "type": "stationary", "start_time": start, "end_time": end,
    })


def dense_static_interior(entry, path_points):
    """Choose a PLY point surrounded by occupied samples in all directions."""
    cloud = np.asarray(o3d.io.read_point_cloud(entry["static_ply"]).points)
    tree = cKDTree(cloud)
    path = np.asarray(path_points, dtype=float)
    path = path[:min(len(path), 14)]
    stride = max(1, len(cloud) // 20000)
    candidates = cloud[::stride]
    distance_to_path = np.min(
        np.linalg.norm(candidates[:, None, :] - path[None, :, :], axis=2), axis=1
    )
    candidates = candidates[(distance_to_path >= 3.0) & (distance_to_path <= 15.0)]
    if not len(candidates):
        raise ValueError("no static interior candidates near camera path")
    directions = np.asarray([
        (x, y, z) for x in (-1, 0, 1)
        for y in (-1, 0, 1) for z in (-1, 0, 1)
    ], dtype=float)
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1.0)
    probes = candidates[:, None, :] + 0.28 * directions[None, :, :]
    distances = tree.query(probes.reshape(-1, 3), workers=-1)[0].reshape(
        len(candidates), len(directions)
    )
    worst = distances.max(axis=1)
    index = int(np.argmin(worst))
    if worst[index] > 0.14:
        raise ValueError(
            f"map has no sufficiently dense static interior (gap={worst[index]:.3f})"
        )
    return candidates[index]


def control_scenario(kind, seed, sequence_id, entry):
    points = np.asarray(entry["reachability"]["path_waypoints_world"], dtype=float)
    start, tangent, perpendicular = point_and_tangent(points, 0.5)
    center, center_tangent, center_perpendicular = point_and_tangent(points, 7.0)
    base_id = actor_id_base(seed)
    actors = []
    if kind == "always_behind":
        actors = [stationary(base_id + 1, start - 4.0 * tangent)]
    elif kind == "always_outside_fov":
        actors = [stationary(base_id + 1, center + 30.0 * center_perpendicular)]
    elif kind == "static_wall_occluded":
        position = dense_static_interior(entry, points)
        actors = [stationary(base_id + 1, position, radius=0.12)]
    elif kind == "static_wall_edge":
        cloud = np.asarray(o3d.io.read_point_cloud(entry["static_ply"]).points)
        _, index = cKDTree(cloud).query(center)
        position = cloud[int(index)].copy()
        position += 0.15 * center_perpendicular
        actors = [stationary(base_id + 1, position, radius=0.18)]
    elif kind == "not_yet_active":
        actors = [stationary(base_id + 1, center, start=20.0, end=30.0)]
    elif kind == "visible_plus_never_nearby":
        visible = scenario("crossing", 0, seed, sequence_id, entry)[
            "dynamic_scenario"
        ]["actors"][0]
        never_position = np.asarray(visible["initial_position_world"], dtype=float)
        actors = [visible, stationary(
            base_id + 2, never_position + 0.7 * center_tangent,
            start=20.0, end=30.0, radius=0.22,
        )]
    elif kind == "moving_camera_static_scene":
        actors = []
    elif kind == "enters_after_sequence":
        position = center - center_perpendicular * 5.0
        actors = [actor(base_id + 1, "sphere", 0.3, position, {
            "type": "delayed_linear",
            "velocity_world": (center_perpendicular * 1.0).round(6).tolist(),
            "start_time": 9.0, "end_time": 15.0,
        })]
    return {"dynamic_scenario": {
        "enabled": bool(actors), "scenario_id": sequence_id,
        "seed": int(seed), "actors": actors,
    }}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--phase-label", default="phase8g")
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    catalog = YAML(typ="safe").load(args.catalog)
    entries = [entry for entry in catalog["maps"] if entry["split"] == args.split]
    if not entries:
        raise ValueError(f"catalog has no split {args.split}")
    yaml = YAML()
    rows = []
    for map_number, entry in enumerate(entries):
        for kind_index, kind in enumerate(KINDS):
            sequence_id = (
                f"{args.phase_label}_{args.split}_{map_number:02d}_{kind_index:02d}"
            )
            seed = int(entry["actor_seed_base"]) + 100 + kind_index
            payload = (scenario(kind, 0, seed, sequence_id, entry)
                       if kind in BASE else
                       control_scenario(kind, seed, sequence_id, entry))
            path = root / f"{sequence_id}.yaml"
            with path.open("w", encoding="utf-8") as stream:
                yaml.dump(payload, stream)
            rows.append({
                "sequence_id": sequence_id, "split": args.split,
                "scenario_type": kind, "variant": 0, "actor_seed": seed,
                "map_id": entry["map_id"], "local_map_id": entry["local_map_id"],
                "map_seed": entry["map_seed"], "static_ply": entry["static_ply"],
                "static_map_sha256": entry["static_map_sha256"],
                "reachability_json": str(
                    Path(entry["static_ply"]).with_name("reachability_metadata.json")
                ),
                "goal": json.dumps(entry["reachability"]["goal"]),
                "scenario_file": str(path),
                "scenario_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "record_instance": True,
            })
    with (root / "matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": "PASS", "split": args.split,
                      "maps": len(entries), "sequences": len(rows),
                      "scenarios": list(KINDS)}, indent=2))


if __name__ == "__main__":
    main()
