from __future__ import annotations

import json
from pathlib import Path

from ruamel.yaml import YAML

import tools.prepare_route_a_v4_8_4_training_config as entry


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_v4_8_4_parent_requires_completed_v483_epoch_two(tmp_path):
    bad = tmp_path / "bad"
    good = tmp_path / "good"
    for run, epoch in ((bad, 1), (good, 2)):
        _write_json(run / "training_complete.json", {
            "status": entry.PARENT_COMPLETION_STATUS,
            "best_epoch": epoch,
        })
        checkpoint = run / "checkpoints/best.pth"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(str(epoch).encode())
    run, checkpoint, completion = entry.latest_parent_checkpoint(tmp_path)
    assert run == good
    assert checkpoint == good / "checkpoints/best.pth"
    assert completion["best_epoch"] == 2


def test_v4_8_4_requires_matching_zero_collision_closed_loop(tmp_path):
    checkpoint = (tmp_path / "parent/checkpoints/best.pth").resolve()
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"parent")
    failed = tmp_path / "runs/20260813T000000Z-pillar"
    passed = tmp_path / "runs/20260813T000001Z-pillar"
    for run, collisions in ((failed, 1), (passed, 0)):
        _write_json(run / "manifest.json", {
            "checkpoint": str(checkpoint),
            "runtime_profile": entry.RUNTIME_PROFILE,
            "goal_mode": "fixed-ab",
            "actors": "none",
        })
        _write_json(run / "collision_report.json", {
            "status": "NO_COLLISION" if collisions == 0 else "COLLISION",
            "any_collision_events": collisions,
            "goal_arrived": True,
        })
    run, _, collision = entry.latest_closed_loop_evidence(
        checkpoint, tmp_path / "runs"
    )
    assert run == passed
    assert collision["any_collision_events"] == 0


def test_v4_8_4_is_score_only_two_epoch_with_parent_baseline(tmp_path):
    parent = YAML(typ="safe").load(entry.PARENT)
    parent_run = tmp_path / "parent"
    checkpoint = parent_run / "checkpoints/best.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"v4.8.3 epoch two")
    source_manifest = tmp_path / "source.json"
    derived_manifest = tmp_path / "derived.json"
    _write_json(source_manifest, {"dataset_version": "source"})
    _write_json(derived_manifest, {"dataset_version": "derived"})
    config = entry.build_config(
        parent, parent_run, checkpoint, {"best_epoch": 2},
        tmp_path / "closed_loop", {
            "goal_arrived": True, "any_collision_events": 0,
        }, source_manifest, derived_manifest,
    )
    assert config["training"]["max_epochs"] == 2
    assert config["training"]["evaluate_initial_checkpoint"] is True
    assert config["training"]["trainable_groups"] == "score_head_only"
    assert config["optimizer"]["backbone_learning_rate"] == 0.0
    assert config["optimizer"]["candidate_head_learning_rate"] == 0.0
    assert config["optimizer"]["score_head_learning_rate"] == 5.0e-7
    assert config["static_yopo_v4_8"]["safe_sector_weight"] == 0.10
    assert config["route_a_v4_8_4"]["ordinary_sample_fraction"] == 0.80
    assert config["route_a_v4_8_4"]["recovery_sample_fraction"] == 0.20
    assert config["validation"]["primary_metric"] == (
        "v4_8_4_score_adaptation"
    )
    assert config["route_a_v4_8_4"]["qualification_gate_count"] == 0


def test_v4_8_4_shell_entries_reuse_data_and_runtime():
    train = (ROOT / "scripts/route_a_v4_8_4_train_host.sh").read_text()
    assert "prepare_route_a_v4_8_4_training_config.py" in train
    assert "validate_route_a_v4_7_dataset.py" in train
    assert "--verify-only" in train
    assert "--authorized" in train
    assert "generate_dataset" not in train
    rviz = (ROOT / "scripts/route_a_v4_8_4_rviz_host.sh").read_text()
    assert "route_a_v4_8_2_rviz_host.sh" in rviz
    assert "route_a_static_yopo_v4_8_4_score_adaptation" in rviz
