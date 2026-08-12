#!/usr/bin/env python3
"""Compact V4.5.5 dense-static-ESDF shakedown summary."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs/route_a_static_yopo_v4_5_5_dense_static_esdf_shakedown"


def metrics(value):
    names = (
        "unsafe_selection", "hardware_unsafe_selection",
        "collision_free_candidate_count", "feasible_candidate_count",
        "collision_free_candidate_available",
        "physical_feasible_candidate_available",
        "conditional_collision_selection_error",
        "conditional_physical_selection_error",
        "selected_endpoint_distance", "selected_clearance",
        "coarse_false_safe_candidate_count", "dense_clearance_drop",
        "score_oracle_regret",
    )
    return {name: float(value[name]) for name in names}


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
        "validation": metrics(row["validation_components"]),
        "validation_by_map_type": {
            name: metrics(value) for name, value in maps.items()
        },
    }


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
    dry_run_only = bool(
        completion and completion.get("status") == "DRY_RUN_PASS"
    )
    result = {
        "status": "DRY_RUN_ONLY" if dry_run_only else "PASS",
        "role": "diagnostic_only_not_production_qualification",
        "run": str(run),
        "epochs_recorded": len(rows),
        "qualification_gate_count": 0,
        "candidate_generator": "trainable_unified_from_v4_5_2",
        "static_esdf_samples": 81,
        "first": compact(rows[0]),
        "best": compact(min(rows, key=lambda row: row["selection_metric"])),
        "last": compact(rows[-1]),
        "completion": completion,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
