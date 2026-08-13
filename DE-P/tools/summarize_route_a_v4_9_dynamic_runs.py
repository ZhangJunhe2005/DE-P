#!/usr/bin/env python3
"""Summarize V4.9 four-scene evidence without creating a qualification Gate."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs/dep_interactive_demo"
SCENES = ("cave", "forest", "pillar", "wall")
CHECKPOINT = (
    ROOT / "runs/route_a_static_yopo_v4_8_3_candidate_only_shakedown"
    / "20260813T050742Z-13178/checkpoints/best.pth"
).resolve()
RUNTIME_PROFILE = "v4_9_dynamic_motion_preserving_safety"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def matches_contract(manifest):
    checkpoint = Path(str(manifest.get("checkpoint", ""))).expanduser()
    return bool(
        checkpoint.is_absolute()
        and checkpoint.resolve() == CHECKPOINT
        and manifest.get("actors") == "multi_target"
        and int(manifest.get("actor_count", -1)) == 16
        and manifest.get("actor_layout") == "route_encounters"
        and int(manifest.get("actor_seed", -1)) == 8801
        and manifest.get("dynamic_mode") == "dynamic_safety"
        and manifest.get("dynamic_foreground_mode") == "range_image_hybrid"
        and manifest.get("runtime_profile") == RUNTIME_PROFILE
        and manifest.get("goal_mode") == "fixed-ab"
        and int(manifest.get("route_encounter_actor_count", -1)) == 16
        and int(manifest.get("route_contract_preserved_actor_count", -1)) == 16
    )


def fraction(rows, predicate):
    return None if not rows else sum(map(predicate, rows)) / len(rows)


def finite_quantiles(values):
    values = np.asarray([
        value for value in values
        if value is not None and np.isfinite(float(value))
    ], dtype=np.float64)
    if not len(values):
        return None
    return {
        "minimum": float(np.min(values)),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
    }


def summarize_run(run, report):
    rows = read_jsonl(run / "safety_decisions.jsonl")
    state_counts = Counter(str(row.get("control_state", "missing")) for row in rows)
    cause_counts = Counter(str(row.get("dynamic_blocking_cause", "missing")) for row in rows)
    selected_scales = Counter(
        str(row.get("selected_dynamic_time_scale", 1.0)) for row in rows
    )
    tracked_rows = [
        row for row in rows if int(row.get("predicted_dynamic_track_count", 0)) > 0
    ]
    collision_classes = report.get("dynamic_collision_observability_counts", {})
    return {
        "run": str(run.resolve()),
        "collision_report_status": report.get("status"),
        "goal_arrived": bool(report.get("goal_arrived", False)),
        "goal_path_length_m": report.get("goal_path_length_m"),
        "path_efficiency": report.get("path_efficiency"),
        "static_collision_events": int(report.get("static_collision_events", 0)),
        "dynamic_collision_events": int(report.get("dynamic_collision_events", 0)),
        "dynamic_collision_observability_counts": collision_classes,
        "replans": len(rows),
        "control_state_counts": dict(state_counts),
        "dynamic_blocking_cause_counts": dict(cause_counts),
        "dynamic_track_frame_fraction": fraction(
            rows,
            lambda row: int(row.get("predicted_dynamic_track_count", 0)) > 0,
        ),
        "moving_during_tracked_frame_fraction": fraction(
            tracked_rows,
            lambda row: float(row.get("current_speed_mps", 0.0)) > 0.25,
        ),
        "dynamic_risk_changed_selection_count": sum(
            bool(row.get("dynamic_risk_changed_selection", False)) for row in rows
        ),
        "dynamic_time_scaling_attempt_count": sum(
            bool(row.get("dynamic_time_scaling_attempted", False)) for row in rows
        ),
        "dynamic_time_scaling_accept_count": sum(
            bool(row.get("dynamic_time_scaling_accepted_scales", [])) for row in rows
        ),
        "selected_dynamic_time_scale_counts": dict(selected_scales),
        "selected_minimum_dynamic_ttc_s": finite_quantiles(
            row.get("selected_minimum_dynamic_ttc_s") for row in rows
        ),
        "selected_dynamic_risk_cost": finite_quantiles(
            row.get("selected_dynamic_risk_cost") for row in rows
        ),
        "dynamic_yield_replans": int(state_counts.get("dynamic_yield", 0)),
        "no_certified_dynamic_action_replans": sum(
            bool(row.get("no_certified_dynamic_action", False))
            for row in rows
        ),
        "stale_dynamic_plan_hold_replans": int(
            state_counts.get("stale_dynamic_plan_hold", 0)
        ),
        "certified_bounded_braking_replans": sum(
            row.get("braking_selection_mode")
            == "certified_bounded_braking"
            for row in rows
        ),
        "minimum_risk_braking_replans": sum(
            row.get("braking_selection_mode")
            == "minimum_risk_uncertified_bounded_braking"
            for row in rows
        ),
        "static_recovery_replans": int(state_counts.get("static_recovery", 0)),
        "mixed_recovery_replans": int(state_counts.get("mixed_recovery", 0)),
        "dynamic_veto_candidates_before_scaling": sum(
            int(row.get("dynamic_hard_veto_count_before_scaling", 0))
            for row in rows
        ),
        "dynamic_veto_candidates_after_scaling": sum(
            int(row.get("dynamic_hard_veto_count_after_scaling", 0))
            for row in rows
        ),
    }


def collect(run_root=RUN_ROOT):
    latest = {}
    for run in sorted(
        (path for path in Path(run_root).glob("*") if path.is_dir()),
        reverse=True,
    ):
        manifest_path = run / "manifest.json"
        report_path = run / "collision_report.json"
        if not (manifest_path.is_file() and report_path.is_file()):
            continue
        manifest = read_json(manifest_path)
        scene = str(manifest.get("scene", ""))
        if scene not in SCENES or scene in latest or not matches_contract(manifest):
            continue
        latest[scene] = summarize_run(run, read_json(report_path))
    complete = len(latest) == len(SCENES)
    return {
        "status": (
            "FOUR_SCENE_EVIDENCE_COMPLETE" if complete
            else "FOUR_SCENE_EVIDENCE_INCOMPLETE"
        ),
        "role": "diagnostic_only_no_gate_no_production_claim",
        "checkpoint": str(CHECKPOINT),
        "contract": {
            "actors": "multi_target",
            "actor_count": 16,
            "actor_layout": "route_encounters",
            "actor_seed": 8801,
            "runtime_profile": RUNTIME_PROFILE,
            "goal_mode": "fixed-ab",
        },
        "scenes": {
            scene: latest.get(scene, {"status": "MISSING"}) for scene in SCENES
        },
    }


def main():
    print(json.dumps(collect(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
