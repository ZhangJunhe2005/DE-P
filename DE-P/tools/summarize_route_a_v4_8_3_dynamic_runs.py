#!/usr/bin/env python3
"""Summarize the fixed V4.8.3 four-scene dynamic closed-loop matrix."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs/dep_interactive_demo"
SCENES = ("cave", "forest", "pillar", "wall")
CHECKPOINT = (
    ROOT / "runs/route_a_static_yopo_v4_8_3_candidate_only_shakedown"
    / "20260813T050742Z-13178/checkpoints/best.pth"
).resolve()
RUNTIME_PROFILE = "v4_8_2_universal_stagnation_recovery"


def matches_contract(manifest):
    checkpoint = Path(str(manifest.get("checkpoint", ""))).expanduser()
    return (
        checkpoint.is_absolute()
        and checkpoint.resolve() == CHECKPOINT
        and manifest.get("actors") == "multi_target"
        and int(manifest.get("actor_count", -1)) == 16
        and manifest.get("actor_layout") == "hybrid"
        and int(manifest.get("actor_seed", -1)) == 8801
        and float(manifest.get("actor_vertical_span", -1.0)) == 2.0
        and manifest.get("dynamic_mode") == "dynamic_safety"
        and manifest.get("dynamic_foreground_mode") == "range_image_hybrid"
        and manifest.get("runtime_profile") == RUNTIME_PROFILE
        and manifest.get("goal_mode") == "fixed-ab"
    )


def read_jsonl(path):
    return [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ] if path.is_file() else []


def summarize_run(run, report):
    rows = read_jsonl(run / "safety_decisions.jsonl")
    track_counts = [
        int(row.get("predicted_dynamic_track_count", 0)) for row in rows
    ]
    valid_context = [
        bool(row.get("dynamic_context_valid", False)) for row in rows
    ]
    dynamic_veto_replans = 0
    dynamic_veto_candidates = 0
    for row in rows:
        count = sum(
            "predicted_dynamic_clearance" in evaluation.get("reasons", [])
            for evaluation in row.get("evaluations", [])
        )
        dynamic_veto_candidates += count
        dynamic_veto_replans += int(count > 0)
    fallback_reasons = Counter(
        str(row["dynamic_context_fallback_reason"])
        for row in rows if row.get("dynamic_context_fallback_reason")
    )
    modes = Counter(str(row.get("mode", "missing")) for row in rows)
    replans = len(rows)
    return {
        "run": str(run.resolve()),
        "collision_status": report.get("status"),
        "goal_arrived": bool(report.get("goal_arrived", False)),
        "static_collision_events": int(report.get("static_collision_events", 0)),
        "dynamic_collision_events": int(report.get("dynamic_collision_events", 0)),
        "independent_dynamic_collision_events": int(
            report.get("independent_dynamic_collision_events", 0)
        ),
        "near_contact_events": int(report.get("near_contact_events", 0)),
        "goal_path_length_m": report.get("goal_path_length_m"),
        "path_efficiency": report.get("path_efficiency"),
        "replans": replans,
        "mode_counts": dict(modes),
        "dynamic_context_valid_fraction": (
            sum(valid_context) / replans if replans else None
        ),
        "frames_with_dynamic_tracks": sum(value > 0 for value in track_counts),
        "dynamic_track_frame_fraction": (
            sum(value > 0 for value in track_counts) / replans
            if replans else None
        ),
        "maximum_predicted_dynamic_tracks": max(track_counts, default=0),
        "mean_predicted_dynamic_tracks": (
            sum(track_counts) / replans if replans else None
        ),
        "dynamic_veto_replans": dynamic_veto_replans,
        "dynamic_veto_candidates": dynamic_veto_candidates,
        "dynamic_context_fallback_reasons": dict(fallback_reasons),
        "network_best_rejected_fraction": (
            sum(bool(row.get("network_best_rejected")) for row in rows) / replans
            if replans else None
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
        manifest = json.loads(manifest_path.read_text())
        scene = str(manifest.get("scene", ""))
        if scene not in SCENES or scene in latest or not matches_contract(manifest):
            continue
        latest[scene] = summarize_run(
            run, json.loads(report_path.read_text())
        )
    results = {
        scene: latest.get(scene, {"status": "MISSING"}) for scene in SCENES
    }
    complete = all(scene in latest for scene in SCENES)
    return {
        "status": (
            "FOUR_SCENE_DYNAMIC_EVIDENCE_COMPLETE"
            if complete else "FOUR_SCENE_DYNAMIC_EVIDENCE_INCOMPLETE"
        ),
        "role": "diagnostic_dynamic_closed_loop_summary_not_a_training_gate",
        "checkpoint": str(CHECKPOINT),
        "contract": {
            "actors": "multi_target",
            "actor_count": 16,
            "actor_layout": "hybrid",
            "actor_seed": 8801,
            "dynamic_mode": "dynamic_safety",
            "runtime_profile": RUNTIME_PROFILE,
        },
        "scenes": results,
    }


def main():
    print(json.dumps(collect(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
