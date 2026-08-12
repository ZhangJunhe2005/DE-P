#!/usr/bin/env python3
"""Compact V4.5.10 report with the V4.5.9 initialization baseline."""

from __future__ import annotations

import json
from pathlib import Path

from ruamel.yaml import YAML


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs/route_a_static_yopo_v4_5_10_tail_aware_safety"
CONFIG = ROOT / "configs/route_a_v4_5_10_tail_aware_safety_shakedown.yaml"


def selected_metrics(value):
    names = (
        "unsafe_selection", "hardware_unsafe_selection",
        "oracle_collision_unsafe", "collision_free_candidate_count",
        "feasible_candidate_count", "collision_free_candidate_available",
        "physical_feasible_candidate_available",
        "conditional_collision_selection_error",
        "conditional_physical_selection_error",
        "selected_endpoint_distance", "selected_clearance",
        "selected_trajectory_max_speed", "selected_trajectory_max_acceleration",
        "selected_time_dilation", "score_oracle_regret",
        "static_safety_loss", "static_time_mean_cost",
        "static_worst_five_cost", "smoothness_loss", "guidance_loss",
        "kinematic_loss", "hover_selection",
    )
    return {name: float(value[name]) for name in names if name in value}


def compact(row):
    maps = row["validation_by_map_type"]
    return {
        "epoch": int(row["epoch"]),
        "metric": float(row["selection_metric"]),
        "score_top1_macro": sum(
            float(value["score_top1_label_agreement"])
            for value in maps.values()
        ) / len(maps),
        "learning_rates": row["learning_rates"],
        "validation": selected_metrics(row["validation_components"]),
        "validation_by_map_type": {
            name: selected_metrics(value) for name, value in maps.items()
        },
    }


def read_rows(run):
    path = run / "metrics.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()] if path.is_file() else []


def main():
    runs = sorted(path for path in RUN_ROOT.glob("*") if path.is_dir())
    if not runs:
        print(json.dumps({"status": "NOT_STARTED"}, indent=2))
        return
    run = runs[-1]
    rows = read_rows(run)
    completion_path = run / "training_complete.json"
    completion = json.loads(completion_path.read_text()) \
        if completion_path.is_file() else None
    state_path = run / "run_state.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else None
    if not rows:
        print(json.dumps({
            "status": "RUNNING_OR_FAILED_BEFORE_FIRST_EPOCH",
            "run": str(run), "run_state": state,
        }, indent=2, sort_keys=True))
        return

    config = YAML(typ="safe").load(CONFIG)
    parent_run = Path(config["parent_v4_5_9_run"])
    parent_rows = read_rows(parent_run)
    parent_best = min(parent_rows, key=lambda row: row["selection_metric"])
    best = min(rows, key=lambda row: row["selection_metric"])
    print(json.dumps({
        "status": (
            "DRY_RUN_ONLY" if completion
            and completion.get("status") == "DRY_RUN_PASS"
            else (completion or state or {}).get("status", "RUNNING")
        ),
        "role": "tail_aware_safety_shakedown",
        "run": str(run),
        "epochs_recorded": len(rows),
        "qualification_gate_count": 0,
        "aggregation": "90_percent_time_mean_plus_10_percent_worst_5_mean",
        "parent_v4_5_9_best": compact(parent_best),
        "first": compact(rows[0]),
        "best": compact(best),
        "last": compact(rows[-1]),
        "run_state": state,
        "completion": completion,
        "long_training_decision": (
            "REQUIRES_CLOSED_LOOP_ZERO_COLLISION_AND_NO_CANDIDATE_COLLAPSE"
        ),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
