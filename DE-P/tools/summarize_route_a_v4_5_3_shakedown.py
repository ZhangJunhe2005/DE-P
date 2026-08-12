#!/usr/bin/env python3
"""Compact V4.5.3 score-only versus joint-finetune summary."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs/route_a_static_yopo_v4_5_3_two_stage_shakedown"
SCORE_ONLY_EPOCHS = 3


def summarize(metrics):
    names = (
        "clearance_barrier_loss", "score_oracle_regret",
        "hardware_unsafe_selection", "unsafe_selection",
        "speed_unsafe_selection", "acceleration_unsafe_selection",
        "collision_free_candidate_count", "hardware_feasible_candidate_count",
        "feasible_candidate_count", "collision_free_candidate_available",
        "physical_feasible_candidate_available", "oracle_collision_unsafe",
        "oracle_hardware_unsafe", "oracle_physical_unsafe",
        "selected_endpoint_distance", "selected_clearance",
        "selected_absolute_vertical_displacement",
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


def main():
    runs = sorted(path for path in RUN_ROOT.glob("*") if path.is_dir())
    if not runs:
        print(json.dumps({"status": "NOT_STARTED"}, indent=2))
        return
    run = runs[-1]
    rows = [
        json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise RuntimeError(f"no epoch metrics: {run}")
    score_only = [row for row in rows if int(row["epoch"]) < SCORE_ONLY_EPOCHS]
    joint = [row for row in rows if int(row["epoch"]) >= SCORE_ONLY_EPOCHS]
    completion_path = run / "training_complete.json"
    completion = (
        json.loads(completion_path.read_text())
        if completion_path.is_file() else None
    )
    result = {
        "status": "PASS",
        "role": "diagnostic_only_not_production_qualification",
        "run": str(run),
        "epochs_recorded": len(rows),
        "qualification_gate_count": 0,
        "score_only": {
            "epochs": [int(row["epoch"]) for row in score_only],
            "best": compact(min(score_only, key=lambda row: row["selection_metric"]))
            if score_only else None,
            "last": compact(score_only[-1]) if score_only else None,
        },
        "joint_finetune": {
            "epochs": [int(row["epoch"]) for row in joint],
            "best": compact(min(joint, key=lambda row: row["selection_metric"]))
            if joint else None,
            "last": compact(joint[-1]) if joint else None,
        },
        "overall_best": compact(min(rows, key=lambda row: row["selection_metric"])),
        "completion": completion,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
