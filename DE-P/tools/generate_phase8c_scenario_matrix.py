#!/usr/bin/env python3
"""Generate 12 deterministic path-relative scenarios for every formal map."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from ruamel.yaml import YAML


SCENARIOS = (
    "no_target", "crossing", "head_on", "multi_target",
    "temporal_separation", "occluded_but_tracked",
)


def validate_catalog_map_entry(map_entry):
    local_id = int(map_entry["local_map_id"])
    reachability_id = int(map_entry["reachability"]["map_id"])
    ply_id = int(Path(map_entry["static_ply"]).stem.rsplit("-", 1)[1])
    if len({local_id, reachability_id, ply_id}) != 1:
        raise ValueError(
            "catalog map identity mismatch: "
            f"local_map_id={local_id}, reachability.map_id={reachability_id}, "
            f"static_ply_id={ply_id}"
        )


def point_and_tangent(points, distance):
    points = np.asarray(points, dtype=float)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    index = min(int(np.searchsorted(cumulative, distance, side="right") - 1), len(lengths) - 1)
    alpha = (distance - cumulative[index]) / max(lengths[index], 1e-9)
    point = points[index] * (1 - alpha) + points[index + 1] * alpha
    tangent = points[index + 1] - points[index]
    tangent[2] = 0.0
    if np.linalg.norm(tangent) < 1e-6:
        tangent = np.asarray([1.0, 0.0, 0.0])
    tangent /= np.linalg.norm(tangent)
    perpendicular = np.asarray([-tangent[1], tangent[0], 0.0])
    return point, tangent, perpendicular


def temporal_actor_motion(points, center_distance, speed):
    """Choose a delayed crossing direction with a geometric time margin."""
    center, _, perpendicular = point_and_tangent(points, center_distance)
    delay = 2.5
    crossing_duration = 1.0
    sample_times = np.arange(0.0, 6.01, 0.05)
    vehicle_positions = np.asarray([
        point_and_tangent(points, min(2.0 * time_value, center_distance + 10.0))[0]
        for time_value in sample_times
    ])
    candidates = []
    for sign in (-1.0, 1.0):
        velocity = perpendicular * speed * sign
        initial = center - velocity * crossing_duration
        actor_positions = np.asarray([
            initial + velocity * max(0.0, time_value - delay)
            for time_value in sample_times
        ])
        minimum_separation = float(np.linalg.norm(
            actor_positions - vehicle_positions, axis=1
        ).min())
        candidates.append((minimum_separation, initial, velocity))
    minimum_separation, initial, velocity = max(candidates, key=lambda item: item[0])
    if minimum_separation < 1.0:
        raise ValueError(
            f"cannot construct safe temporal-separation scenario: {minimum_separation:.3f} m"
        )
    return center, initial, velocity, delay, minimum_separation


def actor(actor_id, shape, radius, position, trajectory, height=None):
    result = {
        "id": actor_id, "enabled": True, "shape": shape, "radius": radius,
        "initial_position_world": np.asarray(position).round(6).tolist(),
        "trajectory": trajectory,
    }
    if height is not None:
        result["height"] = height
    return result


# The Simulator publishes instance IDs as ROS 32SC1, so visible IDs must fit
# signed int32 even though the internal C++ state stores uint32_t.
INSTANCE_ID_MAX = 2**31 - 1
MAX_ACTORS_PER_SCENARIO = 3


def actor_id_base(seed):
    """Reserve four signed-32-bit instance IDs per scenario."""
    seed = int(seed)
    maximum_seed = (INSTANCE_ID_MAX - MAX_ACTORS_PER_SCENARIO) // 4
    if seed < 0 or seed > maximum_seed:
        raise ValueError(
            f"actor seed {seed} cannot be encoded as ROS 32SC1 actor IDs; "
            f"expected 0..{maximum_seed}"
        )
    return seed * 4


def scenario(kind, variant, seed, sequence_id, map_entry):
    center, tangent, perpendicular = point_and_tangent(
        map_entry["reachability"]["path_waypoints_world"], 6.0 + variant
    )
    heldout = map_entry["split"] == "test"
    crossing_speed = (2.2 + 0.35 * variant) if heldout else (1.0 + 0.25 * variant)
    head_speed = (2.1 + 0.30 * variant) if heldout else (0.9 + 0.20 * variant)
    delay = (1.2 + 0.4 * variant) if heldout else (0.4 + 0.35 * variant)
    cross_time = 3.0
    z = float(center[2])
    base_id = actor_id_base(seed)
    actors = []
    if kind == "crossing":
        initial = center - perpendicular * crossing_speed * cross_time
        actors = [actor(base_id + 1, "sphere", 0.4, initial, {
            "type": "linear", "velocity_world": (perpendicular * crossing_speed).round(6).tolist(),
            "start_time": 0.0, "end_time": 8.5,
        })]
    elif kind == "head_on":
        initial = center + tangent * (head_speed * cross_time)
        initial[2] = z - 0.8
        actors = [actor(base_id + 1, "vertical_cylinder", 0.38, initial, {
            "type": "linear", "velocity_world": (-tangent * head_speed).round(6).tolist(),
            "start_time": 0.0, "end_time": 8.5,
        }, height=1.6)]
    elif kind == "multi_target":
        cross_initial = center - perpendicular * crossing_speed * cross_time
        head_initial = center + tangent * (head_speed * cross_time + 2.0)
        head_initial[2] = z - 0.75
        waypoint_center = center + tangent * 2.0
        actors = [
            actor(base_id + 1, "sphere", 0.38, cross_initial, {
                "type": "linear", "velocity_world": (perpendicular * crossing_speed).round(6).tolist(),
                "start_time": 0.0, "end_time": 8.5,
            }),
            actor(base_id + 2, "vertical_cylinder", 0.36, head_initial, {
                "type": "delayed_linear", "velocity_world": (-tangent * head_speed).round(6).tolist(),
                "start_time": delay, "end_time": 8.5,
            }, height=1.5),
            actor(base_id + 3, "sphere", 0.30, waypoint_center - perpendicular * 2.0, {
                "type": "waypoint_ping_pong",
                "waypoints_world": [
                    (waypoint_center - perpendicular * 2.0).round(6).tolist(),
                    (waypoint_center + perpendicular * 2.0).round(6).tolist(),
                ],
                "speed": 0.8 if not heldout else 2.3,
                "start_time": 0.3, "end_time": 8.5,
            }),
        ]
    elif kind == "temporal_separation":
        # Cross a future path point only after the vehicle has passed the
        # nearby approach. Both perpendicular signs are simulated first and
        # the safer one is selected, which also handles sharp 3-D path turns.
        _, initial, velocity, temporal_delay, _ = temporal_actor_motion(
            map_entry["reachability"]["path_waypoints_world"],
            14.0 + variant,
            crossing_speed,
        )
        actors = [actor(base_id + 1, "sphere", 0.38, initial, {
            "type": "delayed_linear", "velocity_world": velocity.round(6).tolist(),
            "start_time": temporal_delay, "end_time": 8.5,
        })]
    elif kind == "occluded_but_tracked":
        start = center + tangent * 2.0 - perpendicular * 1.5
        end = center - tangent * 3.0 + perpendicular * 1.5
        actors = [actor(base_id + 1, "sphere", 0.36, start, {
            "type": "waypoint_ping_pong",
            "waypoints_world": [start.round(6).tolist(), end.round(6).tolist()],
            "speed": 1.3 if not heldout else 2.4,
            "start_time": delay, "end_time": 8.5,
        })]
    return {"dynamic_scenario": {
        "enabled": bool(actors), "scenario_id": sequence_id,
        "seed": seed, "actors": actors,
    }}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    catalog = YAML(typ="safe").load(args.catalog)
    writer_yaml = YAML()
    rows = []
    split_counter = {split: 0 for split in ("train", "valid", "test")}
    for map_entry in catalog["maps"]:
        validate_catalog_map_entry(map_entry)
        split = map_entry["split"]
        for kind in SCENARIOS:
            for variant in range(2):
                split_counter[split] += 1
                sequence_id = f"phase8c_{split}_{split_counter[split]:04d}"
                seed = int(map_entry["actor_seed_base"]) + SCENARIOS.index(kind) * 10 + variant
                path = root / f"{sequence_id}.yaml"
                with path.open("w", encoding="utf-8") as stream:
                    writer_yaml.dump(scenario(kind, variant, seed, sequence_id, map_entry), stream)
                rows.append({
                    "sequence_id": sequence_id, "split": split,
                    "scenario_type": kind, "variant": variant, "actor_seed": seed,
                    "map_id": map_entry["map_id"], "local_map_id": map_entry["local_map_id"],
                    "map_seed": map_entry["map_seed"],
                    "static_ply": map_entry["static_ply"],
                    "static_map_sha256": map_entry["static_map_sha256"],
                    "reachability_json": str(
                        Path(map_entry["static_ply"]).with_name("reachability_metadata.json")
                    ),
                    "goal": json.dumps(map_entry["reachability"]["goal"]),
                    "scenario_file": str(path),
                    "scenario_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "heldout_actor_range": bool(split == "test"),
                })
    with (root / "matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    summary = {"status": "PASS", "sequences": len(rows), "split_counts": split_counter,
               "scenarios": list(SCENARIOS)}
    (root / "matrix_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
