#!/usr/bin/env python3
"""Compact parent-baseline comparison for the V4.8.3 shakedown."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs/route_a_static_yopo_v4_8_3_candidate_only_shakedown"


def read_rows(run):
    path = run / "metrics.jsonl"
    return [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ] if path.is_file() else []


def compact(row):
    values = row["validation_components"]
    names = (
        "safe_sector_coverage_loss",
        "recovery_sample_fraction",
        "recovery_open_sector_soft_count",
        "recovery_mean_endpoint_distance",
        "recovery_max_endpoint_distance",
        "collision_free_candidate_count",
        "feasible_candidate_count",
        "selected_endpoint_distance",
        "selected_clearance",
        "unsafe_selection",
        "hardware_unsafe_selection",
        "conditional_collision_selection_error",
        "score_oracle_regret",
    )
    return {
        "epoch": int(row["epoch"]),
        "initial_checkpoint_validation": bool(
            row.get("initial_checkpoint_validation", False)
        ),
        "selection_metric": float(row["selection_metric"]),
        "ordinary_four_map_macro_loss": float(
            row["validation_ordinary_macro_map_type_total_static_loss"]
        ),
        "conditional_recovery_safe_sector_loss": float(
            row["validation_recovery_safe_sector_loss"]
        ),
        "learning_rates": row["learning_rates"],
        "validation": {
            name: float(values[name]) for name in names if name in values
        },
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
            "status": "RUNNING_OR_FAILED_BEFORE_INITIAL_VALIDATION",
            "run": str(run.resolve()),
            "run_state": state,
        }, indent=2, sort_keys=True))
        return
    baseline = next((row for row in rows if int(row["epoch"]) == -1), None)
    trained = [row for row in rows if int(row["epoch"]) >= 0]
    best_trained = min(trained, key=lambda row: row["selection_metric"]) \
        if trained else None
    selected = min(rows, key=lambda row: row["selection_metric"])
    result = {
        "status": (completion or state or {}).get("status", "RUNNING"),
        "role": "recovery_candidate_only_three_epoch_shakedown",
        "run": str(run.resolve()),
        "epochs_recorded": [int(row["epoch"]) for row in rows],
        "parent_baseline": compact(baseline) if baseline else None,
        "best_trained": compact(best_trained) if best_trained else None,
        "selected_best": compact(selected),
        "trained_checkpoint_beats_parent": (
            None if baseline is None or best_trained is None else
            float(best_trained["selection_metric"])
            < float(baseline["selection_metric"])
        ),
        "candidate_distance_preserved_vs_parent": (
            None if baseline is None or best_trained is None else
            float(best_trained["validation_components"][
                "recovery_mean_endpoint_distance"
            ]) >= float(baseline["validation_components"][
                "recovery_mean_endpoint_distance"
            ])
        ),
        "qualification_gate_count": 0,
        "next_decision": (
            "RUN_FIXED_SCENE_PARENT_VS_V483_CLOSED_LOOP_AB_ONLY_IF_TRAINED_"
            "CHECKPOINT_BEATS_PARENT"
        ),
        "run_state": state,
        "completion": completion,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
