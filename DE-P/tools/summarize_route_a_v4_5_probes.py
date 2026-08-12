#!/usr/bin/env python3
"""Summarize the five V4.5 convergence probes without creating a Gate."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAP_TYPES = ("cave", "forest", "pillar", "room", "wall")


def latest_run(root):
    values = sorted(path for path in root.glob("*") if path.is_dir())
    return values[-1] if values else None


def read_jsonl(path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main():
    probes = {}
    complete = True
    for map_type in MAP_TYPES:
        run = latest_run(
            ROOT / f"runs/route_a_static_yopo_v4_5_probe_{map_type}"
        )
        rows = [] if run is None else read_jsonl(run / "metrics.jsonl")
        if not rows:
            probes[map_type] = {"status": "NOT_STARTED"}
            complete = False
            continue
        best = min(rows, key=lambda row: row["selection_metric"])
        state_path = run / "run_state.json"
        state = json.load(open(state_path)) if state_path.is_file() else {}
        metrics = best["validation_components"]
        probes[map_type] = {
            "status": state.get("status", "UNKNOWN"),
            "run_dir": str(run),
            "best_epoch": int(best["epoch"]),
            "best_validation_metric": float(best["selection_metric"]),
            "score_oracle_regret": float(metrics["score_oracle_regret"]),
            "selected_clearance_m": float(metrics["selected_clearance"]),
            "dangerous_segment_loss": float(
                metrics["dangerous_segment_loss"]
            ),
            "selected_vertical_displacement_m": float(
                metrics["selected_vertical_displacement"]
            ),
            "oracle_vertical_displacement_m": float(
                metrics["oracle_vertical_displacement"]
            ),
            "selected_vertical_primitive_rate": float(
                metrics["selected_vertical_primitive"]
            ),
            "oracle_vertical_primitive_rate": float(
                metrics["oracle_vertical_primitive"]
            ),
            "selected_primitive_distribution": {
                "up": float(metrics["selected_upward_primitive"]),
                "level": float(metrics["selected_level_primitive"]),
                "down": float(metrics["selected_downward_primitive"]),
            },
            "oracle_primitive_distribution": {
                "up": float(metrics["oracle_upward_primitive"]),
                "level": float(metrics["oracle_level_primitive"]),
                "down": float(metrics["oracle_downward_primitive"]),
            },
        }
        complete &= state.get("status") == "TRAINING_COMPLETE"
    print(json.dumps({
        "status": "PASS" if complete else "INCOMPLETE",
        "role": "diagnostic_only_not_a_training_gate",
        "probes": probes,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
