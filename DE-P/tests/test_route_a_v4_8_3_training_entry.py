from __future__ import annotations

import json
from pathlib import Path

from ruamel.yaml import YAML

import tools.prepare_route_a_v4_8_3_training_config as entry


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_v4_8_3_parent_requires_completed_v48_epoch_zero(tmp_path):
    bad = tmp_path / "bad"
    good = tmp_path / "good"
    for run, epoch in ((bad, 1), (good, 0)):
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
    assert completion["best_epoch"] == 0


def test_v4_8_3_is_candidate_only_three_epoch_with_parent_baseline(tmp_path):
    parent = YAML(typ="safe").load(entry.PARENT)
    parent_run = tmp_path / "parent"
    checkpoint = parent_run / "checkpoints/best.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"v4.8 epoch zero")
    source_manifest = tmp_path / "source.json"
    derived_manifest = tmp_path / "derived.json"
    _write_json(source_manifest, {"dataset_version": "source"})
    _write_json(derived_manifest, {"dataset_version": "derived"})
    config = entry.build_config(
        parent, parent_run, checkpoint, {"best_epoch": 0},
        source_manifest, derived_manifest,
    )
    assert config["training"]["max_epochs"] == 3
    assert config["training"]["evaluate_initial_checkpoint"] is True
    assert config["training"]["trainable_groups"] == "candidate_head_only"
    assert config["optimizer"]["backbone_learning_rate"] == 0.0
    assert config["optimizer"]["candidate_head_learning_rate"] == 5.0e-7
    assert config["optimizer"]["score_head_learning_rate"] == 0.0
    assert config["static_yopo_v4_8"]["safe_sector_weight"] == 0.10
    assert config["route_a_v4_8_3"]["ordinary_sample_fraction"] == 0.80
    assert config["route_a_v4_8_3"]["recovery_sample_fraction"] == 0.20
    assert config["validation"]["primary_metric"] == (
        "v4_8_3_normal_plus_recovery_safe_sector"
    )
    assert config["route_a_v4_8_3"]["qualification_gate_count"] == 0


def test_v4_8_3_shell_entries_do_not_generate_data_or_start_long_training():
    train = (ROOT / "scripts/route_a_v4_8_3_train_host.sh").read_text()
    assert "prepare_route_a_v4_8_3_training_config.py" in train
    assert "validate_route_a_v4_7_dataset.py" in train
    assert "--verify-only" in train
    assert "--authorized" in train
    assert "generate_dataset" not in train
    rviz = (ROOT / "scripts/route_a_v4_8_3_rviz_host.sh").read_text()
    assert "route_a_v4_8_2_rviz_host.sh" in rviz
    assert "route_a_static_yopo_v4_8_3_candidate_only_shakedown" in rviz
