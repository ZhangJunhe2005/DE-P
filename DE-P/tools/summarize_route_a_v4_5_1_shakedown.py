#!/usr/bin/env python3
"""Compact, non-Gate summary of the latest V4.5.1 mixed shakedown."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs/route_a_static_yopo_v4_5_1_mixed_shakedown"


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
    metric_names = (
        "score_top1_label_agreement",
        "score_oracle_regret",
        "hardware_unsafe_selection",
        "unsafe_selection",
        "feasible_candidate_count",
        "selected_endpoint_distance",
        "selected_clearance",
        "kinematic_loss",
        "selected_absolute_vertical_displacement",
        "oracle_absolute_vertical_displacement",
        "selected_large_vertical_maneuver",
        "oracle_large_vertical_maneuver",
    )
    by_type = {
        map_type: {
            name: float(metrics[name]) for name in metric_names
        }
        for map_type, metrics in sorted(best["validation_by_map_type"].items())
    }
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
        "validation": {
            name: float(best["validation_components"][name])
            for name in metric_names if name != "score_top1_label_agreement"
        },
        "validation_by_map_type": by_type,
        "completion": completion,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
