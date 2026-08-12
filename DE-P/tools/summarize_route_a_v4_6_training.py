#!/usr/bin/env python3
"""Compact V4.6 four-scene fine-tuning summary."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs/route_a_static_yopo_v4_6_four_scene_finetune"


def metrics(row):
    names = (
        "collision_free_candidate_available", "collision_free_candidate_count",
        "conditional_collision_selection_error", "oracle_collision_unsafe",
        "unsafe_selection", "physical_feasible_candidate_available",
        "hardware_unsafe_selection", "score_oracle_regret",
        "selected_clearance", "selected_endpoint_distance", "hover_selection",
    )
    values = row["validation_components"]
    return {name: float(values[name]) for name in names if name in values}


def compact(row):
    return {
        "epoch": int(row["epoch"]),
        "metric": float(row["selection_metric"]),
        "learning_rates": row["learning_rates"],
        "validation": metrics(row),
        "validation_by_map_type": {
            name: {
                key: float(value[key]) for key in (
                    "total_loss", "collision_free_candidate_available",
                    "conditional_collision_selection_error", "unsafe_selection",
                    "selected_clearance", "selected_endpoint_distance",
                ) if key in value
            }
            for name, value in sorted(row["validation_by_map_type"].items())
        },
    }


def main():
    runs = sorted(path for path in RUN_ROOT.glob("*") if path.is_dir())
    if not runs:
        print(json.dumps({"status": "NOT_STARTED"}, indent=2))
        return
    run = runs[-1]
    metrics_path = run / "metrics.jsonl"
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines()
            if line.strip()] if metrics_path.is_file() else []
    state_path = run / "run_state.json"
    completion_path = run / "training_complete.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else None
    completion = json.loads(completion_path.read_text()) \
        if completion_path.is_file() else None
    if not rows:
        print(json.dumps({
            "status": "RUNNING_OR_FAILED_BEFORE_FIRST_EPOCH",
            "run": str(run), "run_state": state,
        }, indent=2, sort_keys=True))
        return
    best = min(rows, key=lambda row: row["selection_metric"])
    print(json.dumps({
        "status": (completion or state or {}).get("status", "RUNNING"),
        "run": str(run.resolve()),
        "map_types": sorted(best["validation_by_map_type"]),
        "room_present": "room" in best["validation_by_map_type"],
        "qualification_gate_count": 0,
        "epochs_recorded": len(rows),
        "first": compact(rows[0]),
        "best": compact(best),
        "last": compact(rows[-1]),
        "run_state": state,
        "completion": completion,
        "next_decision": "REQUIRES_FOUR_SCENE_FIXED_AB_CLOSED_LOOP",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
