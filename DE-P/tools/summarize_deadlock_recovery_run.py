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
    selection_modes = Counter(
        row.get("network_selection_mode", "missing") for row in replans
    )
    behavior_versions = Counter(
        row.get("runtime_behavior_version", "missing") for row in replans
    )
    scan_limits = Counter(
        str(row.get("recovery_current_scan_limit_deg", "missing"))
        for row in replans
        if row.get("mode") == "recovery_yaw_scan"
    )
    scan_rows = [
        row for row in replans if row.get("mode") == "recovery_yaw_scan"
    ]
    handoff_rows = [
        row for row in replans
        if row.get("recovery_handoff_confirmation_replans") is not None
    ]
    scan_sector_switches = 0
    previous_sector = None
    for row in scan_rows:
        sector = row.get(
            "recovery_selected_candidate_horizontal_sector_id"
        )
        eligible = bool(row.get("recovery_selected_candidate_eligible"))
        if not eligible or sector is None:
            previous_sector = None
            continue
        if previous_sector is not None and sector != previous_sector:
            scan_sector_switches += 1
        previous_sector = sector
    boundary_regions = Counter(
        row["flight_volume_state"]["region"] for row in replans
        if row.get("flight_volume_state") is not None
    )
    boundary_clearances = [
        float(row["flight_volume_state"]["minimum_signed_clearance_m"])
        for row in replans if row.get("flight_volume_state") is not None
    ]
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
        "network_selection_counts": dict(selection_modes),
        "runtime_behavior_versions": dict(behavior_versions),
        "recovery_scan_limit_counts": dict(scan_limits),
        "recovery_scan_horizontal_sector_switches": scan_sector_switches,
        "recovery_handoff_confirmation_replans": [
            row["recovery_handoff_confirmation_replans"]
            for row in handoff_rows
        ],
        "recovery_handoff_horizontal_sectors": [
            row.get("recovery_handoff_horizontal_sector_id")
            for row in handoff_rows
        ],
        "recovery_handoff_minimum_clearances_m": [
            row.get("recovery_selected_candidate_min_observed_clearance_m")
            for row in handoff_rows
        ],
        "flight_volume_region_counts": dict(boundary_regions),
        "boundary_escape_replans": sum(
            row.get("boundary_escape_candidate_count", 0) > 0
            for row in replans
        ),
        "minimum_boundary_clearance_m": (
            min(boundary_clearances) if boundary_clearances else None
        ),
        "zero_feasible_fraction": (
            sum(row["feasible_candidate_count"] == 0 for row in replans)
            / len(replans) if replans else None
        ),
        "mean_feasible_candidates": (
            sum(row["feasible_candidate_count"] for row in replans)
            / len(replans) if replans else None
        ),
        "network_best_rejected_fraction": (
            sum(bool(row.get("network_best_rejected")) for row in replans)
            / len(replans) if replans else None
        ),
        "maximum_preserved_zero_feasible_replans": (
            max(
                int(row.get("recovery_zero_feasible_replans", 0) or 0)
                for row in replans
            ) if replans else None
        ),
        "maximum_motion_stagnation_replans": (
            max(
                int(row.get("recovery_motion_stagnation_replans", 0) or 0)
                for row in replans
            ) if replans else None
        ),
        "minimum_motion_window_displacement_m": min(
            (
                float(row["recovery_motion_window_displacement_m"])
                for row in replans
                if row.get("recovery_motion_window_displacement_m") is not None
            ),
            default=None,
        ),
        "recovery_trigger_reasons": dict(Counter(
            row["recovery_trigger_reason"]
            for row in replans
            if row.get("recovery_transition") in {
                "network_to_braking", "network_stagnation_to_braking",
            }
            and row.get("recovery_trigger_reason")
        )),
        "collision_status": collision.get("status") if collision else "MISSING",
        "collision_events": (
            collision.get("any_collision_events") if collision else None
        ),
        "near_contact_events": (
            collision.get("near_contact_events") if collision else None
        ),
        "goal_arrived": collision.get("goal_arrived") if collision else None,
        "goal_path_length_m": (
            collision.get("goal_path_length_m") if collision else None
        ),
        "goal_minimum_distance_m": (
            collision.get("goal_minimum_distance_m") if collision else None
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
