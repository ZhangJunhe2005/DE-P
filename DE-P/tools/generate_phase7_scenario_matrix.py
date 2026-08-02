#!/usr/bin/env python3
"""Generate the deterministic 24-sequence Phase-7 scenario matrix."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ruamel.yaml import YAML


def actor(object_id, shape, radius, position, trajectory, height=None):
    value = {
        "id": object_id, "enabled": True, "shape": shape, "radius": radius,
        "initial_position_world": position, "trajectory": trajectory,
    }
    if height is not None:
        value["height"] = height
    return value


def scenario(kind, index, seed, scenario_id):
    if kind == "no_target":
        return {"dynamic_scenario": {
            "enabled": False, "scenario_id": scenario_id, "seed": seed, "actors": []
        }}
    delay = 0.35 + 0.1 * (index % 3)
    if kind == "crossing":
        direction = 1.0 if index % 2 == 0 else -1.0
        speed = 0.70 + 0.10 * index
        start_y = -3.0 if direction > 0 else 5.0
        actors = [actor(100 + index, "sphere", 0.38 + 0.02 * (index % 2),
                        [7.0 + 0.25 * index, start_y, 1.6], {
                            "type": "linear", "velocity_world": [0.0, direction * speed, 0.0],
                            "start_time": delay, "end_time": 8.5,
                        })]
        # Even variants add a time-offset low-risk crossing target.
        if index % 2 == 0:
            actors.append(actor(200 + index, "sphere", 0.30, [11.5, 4.5, 1.8], {
                "type": "delayed_linear", "velocity_world": [0.0, -0.45, 0.0],
                "start_time": 2.0 + 0.15 * index, "end_time": 8.5,
            }))
    elif kind == "head_on":
        speed = 0.55 + 0.09 * index
        actors = [actor(300 + index, "vertical_cylinder", 0.36,
                        [11.0 + 0.35 * index, 2.0 + 0.12 * (index - 2.5), 1.3], {
                            "type": "delayed_linear", "velocity_world": [-speed, 0.0, 0.0],
                            "start_time": delay + 0.25, "end_time": 8.5,
                        }, height=1.7)]
    else:
        direction = 1.0 if index % 2 == 0 else -1.0
        actors = [
            actor(400 + index, "sphere", 0.38, [6.5, -2.5 if direction > 0 else 4.5, 1.6], {
                "type": "linear", "velocity_world": [0.0, direction * (0.75 + 0.07 * index), 0.0],
                "start_time": delay, "end_time": 8.5,
            }),
            actor(500 + index, "vertical_cylinder", 0.34,
                  [11.5 + 0.2 * index, 2.0 + 0.1 * (index % 3), 1.3], {
                      "type": "delayed_linear", "velocity_world": [-(0.65 + 0.06 * index), 0.0, 0.0],
                      "start_time": 1.0 + 0.1 * index, "end_time": 8.5,
                  }, height=1.6),
            actor(600 + index, "sphere", 0.28, [9.0, 0.0, 2.0], {
                "type": "waypoint_ping_pong", "waypoints_world": [[9.0, 0.0, 2.0], [9.0, 4.0, 2.0]],
                "speed": 0.55 + 0.04 * index, "start_time": 0.5, "end_time": 8.5,
            }),
        ]
    return {"dynamic_scenario": {
        "enabled": True, "scenario_id": scenario_id, "seed": seed, "actors": actors
    }}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    yaml = YAML()
    rows = []
    for kind_index, kind in enumerate(("no_target", "crossing", "head_on", "multi_target")):
        for index in range(6):
            seed = 71000 + kind_index * 100 + index
            sequence_id = f"phase7_{kind}_{index + 1:02d}"
            split = "train" if index < 4 else ("valid" if index == 4 else "test")
            path = root / f"{sequence_id}.yaml"
            with path.open("w", encoding="utf-8") as stream:
                yaml.dump(scenario(kind, index, seed, sequence_id), stream)
            rows.append({"sequence_id": sequence_id, "scenario_type": kind,
                         "seed": seed, "split": split, "scenario_file": str(path)})
    with (root / "matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"generated {len(rows)} deterministic scenario files in {root}")


if __name__ == "__main__":
    main()
