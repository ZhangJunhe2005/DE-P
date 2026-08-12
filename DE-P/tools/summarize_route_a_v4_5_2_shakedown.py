#!/usr/bin/env python3
"""Compact V4.5.2 diagnostic summary with feasibility decomposition."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs/route_a_static_yopo_v4_5_2_mixed_shakedown"


def summarize(metrics):
    names = (
        "score_oracle_regret", "hardware_unsafe_selection",
        "unsafe_selection", "speed_unsafe_selection",
        "acceleration_unsafe_selection", "collision_free_candidate_count",
        "hardware_feasible_candidate_count", "feasible_candidate_count",
        "collision_free_candidate_available",
        "physical_feasible_candidate_available",
        "oracle_collision_unsafe", "oracle_speed_unsafe",
        "oracle_acceleration_unsafe", "oracle_hardware_unsafe",
        "oracle_physical_unsafe", "selected_endpoint_distance",
        "selected_clearance", "selected_absolute_vertical_displacement",
    )
    result = {name: float(metrics[name]) for name in names}
    collision_available = result["collision_free_candidate_available"]
    physical_available = result["physical_feasible_candidate_available"]
    result["conditional_collision_selection_error_rate"] = (
        float(metrics["conditional_collision_selection_error"])
        / collision_available if collision_available > 0.0 else None
    )
    result["conditional_physical_selection_error_rate"] = (
        float(metrics["conditional_physical_selection_error"])
        / physical_available if physical_available > 0.0 else None
    )
    return result


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
    best = min(rows, key=lambda row: float(row["selection_metric"]))
    completion_path = run / "training_complete.json"
    completion = (
        json.loads(completion_path.read_text())
        if completion_path.is_file() else None
    )
    print(json.dumps({
        "status": "PASS",
        "role": "diagnostic_only_not_production_qualification",
        "run": str(run),
        "epochs_recorded": len(rows),
        "best_epoch": int(best["epoch"]),
        "best_metric": float(best["selection_metric"]),
        "score_top1_macro": sum(
            float(value["score_top1_label_agreement"])
            for value in best["validation_by_map_type"].values()
        ) / len(best["validation_by_map_type"]),
        "validation": summarize(best["validation_components"]),
        "validation_by_map_type": {
            map_type: {
                "score_top1_label_agreement": float(
                    metrics["score_top1_label_agreement"]
                ),
                **summarize(metrics),
            }
            for map_type, metrics in sorted(
                best["validation_by_map_type"].items()
            )
        },
        "completion": completion,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
