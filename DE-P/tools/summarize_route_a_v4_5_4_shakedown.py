#!/usr/bin/env python3
"""Compact V4.5.4 independent-Score shakedown summary."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs/route_a_static_yopo_v4_5_4_independent_score_shakedown"


def summarize(metrics):
    names = (
        "clearance_barrier_loss", "score_oracle_regret",
        "hardware_unsafe_selection", "unsafe_selection",
        "collision_free_candidate_count", "hardware_feasible_candidate_count",
        "feasible_candidate_count", "collision_free_candidate_available",
        "physical_feasible_candidate_available",
        "selected_endpoint_distance", "selected_clearance",
    )
    result = {name: float(metrics[name]) for name in names}
    collision_available = result["collision_free_candidate_available"]
    physical_available = result["physical_feasible_candidate_available"]
    collision_numerator = float(
        metrics["conditional_collision_selection_error"]
    )
    physical_numerator = float(
        metrics["conditional_physical_selection_error"]
    )
    result["conditional_collision_selection_error_rate"] = (
        collision_numerator / collision_available
        if collision_available > 0.0 else None
    )
    result["conditional_physical_selection_error_rate"] = (
        physical_numerator / physical_available
        if physical_available > 0.0 else None
    )
    result["selected_physical_unsafe"] = (
        1.0 - physical_available + physical_numerator
    )
    return result


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
        "validation": summarize(row["validation_components"]),
    }


def candidate_stability(rows):
    names = (
        "collision_free_candidate_count",
        "hardware_feasible_candidate_count",
        "feasible_candidate_count",
        "collision_free_candidate_available",
        "physical_feasible_candidate_available",
        "clearance_barrier_loss",
    )
    first = rows[0]["validation_components"]
    result = {}
    for name in names:
        values = [float(row["validation_components"][name]) for row in rows]
        result[name] = {
            "epoch_0": float(first[name]),
            "maximum_absolute_drift": max(abs(value - float(first[name])) for value in values),
        }
    return result


def main():
    runs = sorted(path for path in RUN_ROOT.glob("*") if path.is_dir())
    if not runs:
        print(json.dumps({"status": "NOT_STARTED"}, indent=2))
        return
    run = runs[-1]
    metrics_path = run / "metrics.jsonl"
    rows = (
        [json.loads(line) for line in metrics_path.read_text().splitlines()
         if line.strip()]
        if metrics_path.is_file() else []
    )
    completion_path = run / "training_complete.json"
    completion = (
        json.loads(completion_path.read_text())
        if completion_path.is_file() else None
    )
    if not rows:
        state_path = run / "run_state.json"
        state = json.loads(state_path.read_text()) if state_path.is_file() else None
        print(json.dumps({
            "status": "RUNNING_OR_FAILED_BEFORE_FIRST_EPOCH",
            "run": str(run),
            "run_state": state,
        }, indent=2, sort_keys=True))
        return
    result = {
        "status": "PASS",
        "role": "diagnostic_only_not_production_qualification",
        "run": str(run),
        "epochs_recorded": len(rows),
        "qualification_gate_count": 0,
        "candidate_generator": "frozen_v4_5_2",
        "epochs": [compact(row) for row in rows],
        "best": compact(min(rows, key=lambda row: row["selection_metric"])),
        "last": compact(rows[-1]),
        "candidate_generator_stability": candidate_stability(rows),
        "completion": completion,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
