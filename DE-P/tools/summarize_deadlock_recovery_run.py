#!/usr/bin/env python3
"""Summarize one interactive run without changing its evidence files."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    root = args.run.expanduser().resolve()
    telemetry = root / "safety_decisions.jsonl"
    rows = (
        [
            json.loads(line) for line in telemetry.read_text().splitlines()
            if line.strip()
        ]
        if telemetry.is_file() else []
    )
    modes = Counter(row.get("mode", "missing") for row in rows)
    transitions = Counter(
        row["recovery_transition"] for row in rows
        if row.get("recovery_transition")
    )
    replans = [row for row in rows if "feasible_candidate_count" in row]
    collision_report = root / "collision_report.json"
    collision = (
        json.loads(collision_report.read_text())
        if collision_report.is_file() else None
    )
    result = {
        "status": "PASS",
        "run": str(root),
        "telemetry_rows": len(rows),
        "telemetry_present": telemetry.is_file(),
        "replans": len(replans),
        "mode_counts": dict(modes),
        "transition_counts": dict(transitions),
        "zero_feasible_fraction": (
            sum(row["feasible_candidate_count"] == 0 for row in replans)
            / len(replans) if replans else None
        ),
        "mean_feasible_candidates": (
            sum(row["feasible_candidate_count"] for row in replans)
            / len(replans) if replans else None
        ),
        "collision_status": collision.get("status") if collision else "MISSING",
        "collision_events": (
            collision.get("any_collision_events") if collision else None
        ),
        "near_contact_events": (
            collision.get("near_contact_events") if collision else None
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
