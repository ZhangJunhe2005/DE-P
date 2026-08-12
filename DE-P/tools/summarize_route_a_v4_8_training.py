#!/usr/bin/env python3
"""Compact report for the V4.8 recovery-capacity five-epoch shakedown."""

from __future__ import annotations

import json
from pathlib import Path

from ruamel.yaml import YAML


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs/route_a_static_yopo_v4_8_recovery_capacity_shakedown"
CONFIG = ROOT / "configs/route_a_v4_8_recovery_capacity_shakedown.yaml"


def selected_metrics(value):
    names = (
        "safe_sector_coverage_loss",
        "recovery_sample_fraction",
        "recovery_open_sector_soft_count",
        "recovery_mean_endpoint_distance",
        "recovery_max_endpoint_distance",
        "unsafe_selection",
        "hardware_unsafe_selection",
        "oracle_collision_unsafe",
        "collision_free_candidate_count",
        "feasible_candidate_count",
        "collision_free_candidate_available",
        "physical_feasible_candidate_available",
        "conditional_collision_selection_error",
        "conditional_physical_selection_error",
        "selected_endpoint_distance",
        "selected_clearance",
        "selected_trajectory_max_speed",
        "selected_trajectory_max_acceleration",
        "selected_time_dilation",
        "score_oracle_regret",
        "static_safety_loss",
        "static_time_mean_cost",
        "static_worst_five_cost",
        "smoothness_loss",
        "guidance_loss",
        "kinematic_loss",
        "hover_selection",
    )
    return {name: float(value[name]) for name in names if name in value}


def compact(row):
    maps = row.get("validation_by_map_type", {})
    result = {
        "epoch": int(row["epoch"]),
        "metric": float(row["selection_metric"]),
        "learning_rates": row["learning_rates"],
        "validation": selected_metrics(row["validation_components"]),
        "validation_by_map_type": {
            name: selected_metrics(value) for name, value in sorted(maps.items())
        },
    }
    agreements = [
        float(value["score_top1_label_agreement"])
        for value in maps.values()
        if "score_top1_label_agreement" in value
    ]
    if agreements:
        result["score_top1_macro"] = sum(agreements) / len(agreements)
    return result


def read_rows(run):
    path = run / "metrics.jsonl"
    return [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ] if path.is_file() else []


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
            "run": str(run.resolve()),
            "run_state": state,
        }, indent=2, sort_keys=True))
        return

    config = YAML(typ="safe").load(CONFIG)
    parent_run = Path(config["parent_v4_7_run"])
    parent_rows = read_rows(parent_run)
    parent_best = min(parent_rows, key=lambda row: row["selection_metric"]) \
        if parent_rows else None
    best = min(rows, key=lambda row: row["selection_metric"])
    result = {
        "status": (
            "DRY_RUN_ONLY" if completion
            and completion.get("status") == "DRY_RUN_PASS"
            else (completion or state or {}).get("status", "RUNNING")
        ),
        "role": "recovery_capacity_five_epoch_shakedown",
        "run": str(run.resolve()),
        "epochs_recorded": len(rows),
        "dataset_reused_from": "route_a_v4_7_static_yopo",
        "observation_contract": "route_a_recovery_state_v4_8",
        "validation_contract": "route_a_v4_8_recovery_capacity_v1",
        "qualification_gate_count": 0,
        "first": compact(rows[0]),
        "best": compact(best),
        "last": compact(rows[-1]),
        "run_state": state,
        "completion": completion,
        "next_decision": (
            "REQUIRES_FIXED_V47_SCENE_CLOSED_LOOP_RECOVERY_EVALUATION"
        ),
    }
    if parent_best is not None:
        result["parent_v4_7_best"] = compact(parent_best)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
