#!/usr/bin/env python3
"""Summarize the ten-epoch V4.5.10 continuation against its parent best."""

from __future__ import annotations

import json
from pathlib import Path

from ruamel.yaml import YAML


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs/route_a_static_yopo_v4_5_10_controlled_continuation"
CONFIG = ROOT / "configs/route_a_v4_5_10_controlled_continuation.yaml"


METRICS = (
    "unsafe_selection", "oracle_collision_unsafe",
    "conditional_collision_selection_error", "score_oracle_regret",
    "collision_free_candidate_count", "collision_free_candidate_available",
    "feasible_candidate_count", "physical_feasible_candidate_available",
    "hardware_unsafe_selection", "selected_endpoint_distance",
    "selected_clearance", "hover_selection",
)


def read_rows(run):
    path = Path(run) / "metrics.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()] if path.is_file() else []


def compact(row):
    values = row["validation_components"]
    return {
        "epoch": int(row["epoch"]),
        "metric": float(row["selection_metric"]),
        "learning_rates": row["learning_rates"],
        **{name: float(values[name]) for name in METRICS if name in values},
    }


def main():
    runs = sorted(path for path in RUN_ROOT.glob("*") if path.is_dir())
    if not runs:
        print(json.dumps({"status": "NOT_STARTED"}, indent=2))
        return
    run = runs[-1]
    rows = read_rows(run)
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
    config = YAML(typ="safe").load(CONFIG)
    parent_rows = read_rows(config["parent_v4_5_10_run"])
    parent_best = min(parent_rows, key=lambda row: row["selection_metric"])
    best = min(rows, key=lambda row: row["selection_metric"])
    print(json.dumps({
        "status": (
            "DRY_RUN_ONLY" if completion
            and completion.get("status") == "DRY_RUN_PASS"
            else (completion or state or {}).get("status", "RUNNING")
        ),
        "role": "tail_aware_controlled_continuation",
        "run": str(run),
        "epochs_recorded": len(rows),
        "qualification_gate_count": 0,
        "parent_v4_5_10_best": compact(parent_best),
        "first": compact(rows[0]),
        "best": compact(best),
        "last": compact(rows[-1]),
        "completion": completion,
        "run_state": state,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
