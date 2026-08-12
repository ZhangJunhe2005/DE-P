#!/usr/bin/env python3
"""Evaluate the single external V4.2.9 closed-loop readiness layer."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ruamel.yaml import YAML


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision-report", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.collision_report.read_text(encoding="utf-8"))
    config = YAML(typ="safe").load(args.config)
    contract = config["validation"]["closed_loop_contract"]
    collision_count = int(report["any_collision_events"])
    arrived = bool(report.get("goal_arrived", False))
    path_efficiency = report.get("path_efficiency")
    observations = {
        "collision_events": collision_count,
        "goal_arrived": arrived,
        "path_efficiency": path_efficiency,
    }
    passed = (
        collision_count <= int(contract["collision_events_max"])
        and (arrived or not bool(contract["goal_arrival_required"]))
        and path_efficiency is not None
        and float(path_efficiency) <= float(contract["path_efficiency_max"])
    )
    result = {
        "status": "PASS" if passed else "FAIL",
        "layer": "closed_loop_result",
        "observations": observations,
        "contract": {
            "collision_events_max": int(contract["collision_events_max"]),
            "goal_arrival_required": bool(contract["goal_arrival_required"]),
            "path_efficiency_max": float(contract["path_efficiency_max"]),
        },
        "hardware_invariants": {
            "max_speed_mps": float(contract["max_speed_mps"]),
            "max_acceleration_mps2": float(contract["max_acceleration_mps2"]),
            "role": "non_relaxable_runtime_contract_not_an_extra_gate",
        },
        "collision_report": str(args.collision_report.resolve()),
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
