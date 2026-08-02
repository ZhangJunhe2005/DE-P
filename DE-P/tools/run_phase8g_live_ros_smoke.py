#!/usr/bin/env python3
"""Run seven finite live ROS/Simulator instance-perception scenarios."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from run_phase8c_formal_recording import (
    start, stop, wait_for_ros_master, wait_for_simulator_subscription,
    worker_environment,
)


ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = (
    "no_target", "crossing", "static_wall_occluded",
    "always_outside_fov", "visible_plus_never_nearby",
    "occluded_but_tracked", "temporal_separation",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ros-master-port", type=int, default=13500)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.matrix.open(newline="", encoding="utf-8")))
    for row in rows:
        row["scenario_file"] = str(
            args.matrix.parent / Path(row["scenario_file"]).name
        )
    selected = {}
    for scenario in SCENARIOS:
        selected[scenario] = next(
            row for row in rows
            if row["scenario_type"] == scenario
        )
    runtime = Path(tempfile.mkdtemp(prefix="phase8g-live-", dir="/tmp"))
    env = worker_environment(runtime, 0, args.ros_master_port)
    roscore = start(["roscore", "-p", str(args.ros_master_port)],
                    runtime / "roscore.log", env)
    results = {}
    try:
        wait_for_ros_master(env, roscore)
        for scenario, row in selected.items():
            collector = simulator = odom = None
            try:
                output = runtime / f"{scenario}_collector.log"
                collector = start([
                    sys.executable, str(ROOT / "tools/phase8g_live_collector.py"),
                    "--scenario", scenario, "--frames", "60", "--timeout", "25",
                ], output, env)
                simulator = start([
                    "rosrun", "sensor_simulator", "sensor_simulator_cuda",
                    f"_dynamic_scenario_file:={row['scenario_file']}",
                    "_random_map:=false", f"_ply_file:={row['static_ply']}",
                    "_render_lidar:=false", "_render_depth:=true",
                ], runtime / f"{scenario}_simulator.log", env)
                wait_for_simulator_subscription(env, simulator, scenario)
                odom = start([
                    "/usr/bin/python3", str(ROOT / "tools/publish_reachable_path_odom.py"),
                    "--reachability", row["reachability_json"],
                    "--map-id", row["local_map_id"], "--speed", "2.0",
                ], runtime / f"{scenario}_odom.log", env)
                if collector.wait(timeout=35) != 0:
                    raise RuntimeError(f"collector failed: {scenario}")
                collector._phase8c_log.close()
                text = output.read_text()
                results[scenario] = json.loads(text[text.index("{"):])
            finally:
                stop(collector); stop(simulator); stop(odom)
    finally:
        stop(roscore)
    payload = {
        "status": "PASS" if all(
            item["status"] == "PASS" for item in results.values()
        ) and set(results) == set(SCENARIOS) else "FAIL",
        "scenarios": results,
        "process_cleanup": "PASS",
        "runtime_instance_subscription_owner": "evaluator_collector_only",
        "dep_runtime_instance_input": False,
        "logs": str(runtime),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
