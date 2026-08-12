#!/usr/bin/env bash
set -euo pipefail

cd /home/zjh/YOPO/DE-P
conda run --no-capture-output -n yopo python - <<'PY'
import json
from pathlib import Path

root = Path("runs/dep_interactive_demo")
scene_config = json.loads(Path(
    "configs/dep_interactive_demo_scenes_v4_6.json"
).read_text())
expected = {
    name: {
        "map_uuid": row["map_uuid"],
        "start": row["start"],
        "suggested_goal": row["suggested_goal"],
    }
    for name, row in scene_config["scenes"].items()
}

def same_xyz(actual, wanted):
    if not isinstance(actual, list) or len(actual) != 3:
        return False
    return all(
        abs(float(left) - float(right)) <= 1e-9
        for left, right in zip(actual, wanted)
    )

latest = {}
for run in sorted(
    (path for path in root.glob("*") if path.is_dir()), reverse=True
):
    manifest_path = run / "manifest.json"
    report_path = run / "collision_report.json"
    if not manifest_path.is_file() or not report_path.is_file():
        continue
    manifest = json.loads(manifest_path.read_text())
    scene = manifest.get("scene")
    if scene not in expected:
        continue
    fixture = expected[scene]
    if manifest.get("map_uuid") != fixture["map_uuid"]:
        continue
    if not same_xyz(manifest.get("start"), fixture["start"]):
        continue
    if not same_xyz(
        manifest.get("suggested_goal"), fixture["suggested_goal"]
    ):
        continue
    checkpoint = str(manifest.get("checkpoint", ""))
    if "route_a_static_yopo_v4_6_four_scene_finetune" not in checkpoint:
        continue
    latest.setdefault(scene, (run, json.loads(report_path.read_text())))

results = {}
for scene in sorted(expected):
    if scene not in latest:
        results[scene] = {"status": "MISSING"}
        continue
    run, report = latest[scene]
    results[scene] = {
        "status": report.get("status"),
        "run": str(run.resolve()),
        "goal_arrived": bool(report.get("goal_arrived", False)),
        "goal_minimum_distance_m": report.get("goal_minimum_distance_m"),
        "static_collision_events": int(
            report.get("static_collision_events", 0)
        ),
        "dynamic_collision_events": int(
            report.get("dynamic_collision_events", 0)
        ),
        "near_contact_events": int(report.get("near_contact_events", 0)),
        "goal_path_length_m": report.get("goal_path_length_m"),
        "path_efficiency": report.get("path_efficiency"),
    }

print(json.dumps({
    "status": (
        "FOUR_SCENE_EVIDENCE_COMPLETE"
        if all(value.get("status") != "MISSING" for value in results.values())
        else "FOUR_SCENE_EVIDENCE_INCOMPLETE"
    ),
    "role": "diagnostic_closed_loop_summary_not_a_new_training_gate",
    "scenes": results,
}, indent=2, sort_keys=True))
PY
