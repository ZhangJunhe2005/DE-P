from __future__ import annotations

import json
from pathlib import Path

from tools.summarize_route_a_v4_8_3_dynamic_runs import (
    CHECKPOINT,
    RUNTIME_PROFILE,
    collect,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _manifest(scene):
    return {
        "scene": scene,
        "checkpoint": str(CHECKPOINT),
        "actors": "multi_target",
        "actor_count": 16,
        "actor_layout": "hybrid",
        "actor_seed": 8801,
        "actor_vertical_span": 2.0,
        "dynamic_mode": "dynamic_safety",
        "dynamic_foreground_mode": "range_image_hybrid",
        "runtime_profile": RUNTIME_PROFILE,
        "goal_mode": "fixed-ab",
    }


def test_dynamic_entry_uses_frozen_v483_and_uniform_four_scene_contract():
    script = (
        ROOT / "scripts/route_a_v4_8_3_dynamic_rviz_host.sh"
    ).read_text()
    assert "22e5c63c273d751c15479d70c99d9b85" in script
    assert "--actors multi_target" in script
    assert "--actor-count 16" in script
    assert "--actor-layout hybrid" in script
    assert "--actor-seed 8801" in script
    assert "--dynamic-mode dynamic_safety" in script
    assert "--dynamic-foreground-mode range_image_hybrid" in script
    assert "dynamic_attention" not in script
    assert "if [[ \"$SCENE\" ==" not in script


def test_dynamic_summary_requires_all_four_matching_runs(tmp_path):
    run = tmp_path / "20260813T000000Z-pillar"
    _write_json(run / "manifest.json", _manifest("pillar"))
    _write_json(run / "collision_report.json", {
        "status": "NO_COLLISION",
        "goal_arrived": True,
        "static_collision_events": 0,
        "dynamic_collision_events": 0,
        "independent_dynamic_collision_events": 0,
        "near_contact_events": 1,
    })
    (run / "safety_decisions.jsonl").write_text("\n".join((
        json.dumps({
            "mode": "network",
            "dynamic_context_valid": True,
            "predicted_dynamic_track_count": 2,
            "dynamic_context_fallback_reason": None,
            "network_best_rejected": True,
            "evaluations": [
                {"reasons": ["predicted_dynamic_clearance"]},
                {"reasons": []},
            ],
        }),
        json.dumps({
            "mode": "network",
            "dynamic_context_valid": False,
            "predicted_dynamic_track_count": 0,
            "dynamic_context_fallback_reason": "no_sensor_update",
            "network_best_rejected": False,
            "evaluations": [],
        }),
    )) + "\n")
    result = collect(tmp_path)
    assert result["status"] == "FOUR_SCENE_DYNAMIC_EVIDENCE_INCOMPLETE"
    pillar = result["scenes"]["pillar"]
    assert pillar["goal_arrived"] is True
    assert pillar["dynamic_context_valid_fraction"] == 0.5
    assert pillar["maximum_predicted_dynamic_tracks"] == 2
    assert pillar["dynamic_veto_replans"] == 1
    assert pillar["dynamic_veto_candidates"] == 1
    assert pillar["dynamic_context_fallback_reasons"] == {
        "no_sensor_update": 1
    }
