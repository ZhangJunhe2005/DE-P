#!/usr/bin/env python3
"""Inflated 3-D connectivity and shortest-path validation for YOPO maps."""

from __future__ import annotations

import argparse
from collections import deque
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from ruamel.yaml import YAML
from scipy.ndimage import label
from scipy.spatial import cKDTree


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shortest_path_cells(free, start, goal):
    queue = deque([start])
    parents = {start: None}
    shape = free.shape
    while queue:
        current = queue.popleft()
        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = parents[current]
            return list(reversed(path))
        for axis in range(3):
            for delta in (-1, 1):
                neighbor = list(current); neighbor[axis] += delta; neighbor = tuple(neighbor)
                if (0 <= neighbor[axis] < shape[axis] and neighbor not in parents
                        and free[neighbor]):
                    parents[neighbor] = current
                    queue.append(neighbor)
    return None


def validate_map(ply_path, pose_path, *, grid_resolution, inflated_radius,
                 min_path_length):
    points = np.asarray(o3d.io.read_point_cloud(str(ply_path)).points)
    if not len(points):
        raise ValueError(f"blank point cloud: {ply_path}")
    with Path(pose_path).open(newline="", encoding="utf-8") as stream:
        poses = np.asarray([[float(row[key]) for key in ("px", "py", "pz")]
                            for row in csv.DictReader(stream)], dtype=float)
    if len(poses) < 2:
        raise ValueError(f"at least two poses are required: {pose_path}")
    minimum = np.minimum(points.min(0), poses.min(0)) - grid_resolution
    maximum = np.maximum(points.max(0), poses.max(0)) + grid_resolution
    shape = np.ceil((maximum - minimum) / grid_resolution).astype(int) + 1
    if int(np.prod(shape)) > 2_000_000:
        raise ValueError(f"reachability grid is unexpectedly large: {shape.tolist()}")
    axes = [minimum[i] + np.arange(shape[i]) * grid_resolution for i in range(3)]
    mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    distance = cKDTree(points).query(mesh, workers=-1)[0]
    free = (distance >= inflated_radius).reshape(tuple(shape))
    # scipy requires a centrosymmetric 6-connected structure.
    structure = np.zeros((3, 3, 3), dtype=int)
    structure[1, 1, :] = structure[1, :, 1] = structure[:, 1, 1] = 1
    components, component_count = label(free, structure=structure)
    pose_cells = np.rint((poses - minimum) / grid_resolution).astype(int)
    in_bounds = np.all((pose_cells >= 0) & (pose_cells < shape), axis=1)
    component_ids = np.zeros(len(poses), dtype=int)
    valid_indices = np.flatnonzero(in_bounds)
    component_ids[valid_indices] = components[tuple(pose_cells[valid_indices].T)]
    valid_indices = valid_indices[component_ids[valid_indices] > 0]
    if len(valid_indices) < 2:
        raise ValueError("fewer than two sampled poses lie in inflated free space")
    best = None
    for start_index in valid_indices:
        same = valid_indices[component_ids[valid_indices] == component_ids[start_index]]
        if len(same) < 2:
            continue
        distances = np.linalg.norm(poses[same] - poses[start_index], axis=1)
        goal_index = int(same[int(np.argmax(distances))])
        candidate = float(distances.max())
        if best is None or candidate > best[0]:
            best = (candidate, int(start_index), goal_index)
    if best is None:
        raise ValueError("no two poses share an inflated free-space component")
    _, start_index, goal_index = best
    path_cells = shortest_path_cells(
        free, tuple(pose_cells[start_index]), tuple(pose_cells[goal_index])
    )
    if path_cells is None:
        raise ValueError("component label and BFS reachability disagree")
    path_length = (len(path_cells) - 1) * grid_resolution
    if path_length < min_path_length:
        raise ValueError(f"map is too simple: shortest path {path_length:.3f} m")
    path_world = [
        (minimum + np.asarray(cell, dtype=float) * grid_resolution).tolist()
        for cell in path_cells
    ]
    # Retain every metre plus the exact endpoint. These waypoints follow the
    # validated 6-connected path and are consumed by the formal odometry node.
    waypoint_stride = max(1, int(round(1.0 / grid_resolution)))
    path_waypoints = path_world[::waypoint_stride]
    if path_waypoints[-1] != path_world[-1]:
        path_waypoints.append(path_world[-1])
    return {
        "ply_sha256": sha256(ply_path), "pose_csv_sha256": sha256(pose_path),
        "grid_resolution": grid_resolution, "uav_inflated_radius": inflated_radius,
        "component_count": int(component_count), "reachable_pose_count": int(len(valid_indices)),
        "start_pose_index": start_index, "goal_pose_index": goal_index,
        "start": poses[start_index].tolist(), "goal": poses[goal_index].tolist(),
        "shortest_path_length": float(path_length), "minimum_path_length": min_path_length,
        "path_waypoints_world": path_waypoints,
        "path_waypoint_spacing_max": float(waypoint_stride * grid_resolution),
        "reachability": "PASS",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--grid-resolution", type=float, default=0.5)
    parser.add_argument("--uav-radius", type=float, default=0.3)
    parser.add_argument("--clearance-margin", type=float, default=0.2)
    parser.add_argument("--min-path-length", type=float, default=5.0)
    args = parser.parse_args()
    metadata_path = args.dataset / "generation_metadata.yaml"
    metadata = YAML(typ="safe").load(metadata_path)
    if metadata.get("completion_status") != "complete":
        raise ValueError("generator output is incomplete")
    ply_files = sorted(args.dataset.glob("pointcloud-*.ply"))
    results = []
    for ply in ply_files:
        map_id = int(ply.stem.split("-")[-1])
        result = validate_map(
            ply, args.dataset / f"pose-{map_id}.csv",
            grid_resolution=args.grid_resolution,
            inflated_radius=args.uav_radius + args.clearance_margin,
            min_path_length=args.min_path_length,
        )
        result["map_id"] = map_id; results.append(result)
        with (args.dataset / f"start_goal-{map_id}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream); writer.writerow(("sx", "sy", "sz", "gx", "gy", "gz"))
            writer.writerow((*result["start"], *result["goal"]))
    if not results:
        raise ValueError("no generated pointcloud maps found")
    output = {"status": "PASS", "maps": results}
    (args.dataset / "reachability_metadata.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
