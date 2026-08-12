from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from ruamel.yaml import YAML

import tools.prepare_route_a_v4_8_training_config as entry
from tools.run_dep_interactive_demo import UNGATED_STATIC_TRAINING_CONTRACTS


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _derived_manifest(**overrides):
    value = {
        "dataset_version": entry.DERIVED_VERSION,
        "status": "COMPLETE_FROZEN",
        "split_counts": {"train": 80, "validation": 40},
        "map_type_sample_counts": {
            split: {name: 10 for name in sorted(entry.EXPECTED_TYPES)}
            for split in ("train", "validation")
        },
    }
    value.update(overrides)
    return value


def _parent():
    return YAML(typ="safe").load(entry.PARENT)


def _build(tmp_path):
    parent_run = tmp_path / "parent_run"
    checkpoint = parent_run / "checkpoints/best.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"strict V4.7 best fixture")
    source_manifest = tmp_path / "source.json"
    derived_manifest = tmp_path / "derived.json"
    _write_json(source_manifest, {"dataset_version": entry.SOURCE_VERSION})
    _write_json(derived_manifest, _derived_manifest())
    return entry.build_config(
        _parent(),
        parent_run,
        checkpoint,
        {"best_epoch": 15},
        source_manifest,
        derived_manifest,
    )


def test_v4_8_reuses_exact_frozen_v4_7_dataset_identity(tmp_path):
    source = tmp_path / "route_a_v4_7_raw_static"
    derived = tmp_path / "route_a_v4_7_static_yopo"
    _write_json(source / "manifests/dataset_manifest.json", {
        "dataset_version": entry.SOURCE_VERSION,
    })
    _write_json(
        derived / "manifests/dataset_manifest.json", _derived_manifest()
    )
    _, _, manifest = entry.validate_dataset_identity(source, derived)
    assert manifest["status"] == "COMPLETE_FROZEN"
    assert set(manifest["map_type_sample_counts"]["train"]) == {
        "cave", "forest", "pillar", "wall",
    }
    assert entry.SOURCE.name == "route_a_v4_7_raw_static"
    assert entry.DERIVED.name == "route_a_v4_7_static_yopo"


@pytest.mark.parametrize("bad_status", ["INCOMPLETE", "COMPLETE"])
def test_v4_8_rejects_nonfrozen_v4_7_dataset(tmp_path, bad_status):
    source = tmp_path / "raw"
    derived = tmp_path / "derived"
    _write_json(source / "manifests/dataset_manifest.json", {
        "dataset_version": entry.SOURCE_VERSION,
    })
    _write_json(
        derived / "manifests/dataset_manifest.json",
        _derived_manifest(status=bad_status),
    )
    with pytest.raises(RuntimeError, match="not frozen"):
        entry.validate_dataset_identity(source, derived)


def test_v4_8_latest_parent_requires_completed_v4_7_best(tmp_path):
    older = tmp_path / "20260812T010000Z-1"
    newer_incomplete = tmp_path / "20260812T020000Z-2"
    newest = tmp_path / "20260812T030000Z-3"
    for run, status, epoch in (
        (older, entry.PARENT_COMPLETION_STATUS, 11),
        (newer_incomplete, "RUNNING", 12),
        (newest, entry.PARENT_COMPLETION_STATUS, 15),
    ):
        _write_json(run / "training_complete.json", {
            "status": status,
            "best_epoch": epoch,
        })
        checkpoint = run / "checkpoints/best.pth"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(str(epoch).encode())
    run, checkpoint, completion = entry.latest_parent_checkpoint(tmp_path)
    assert run == newest
    assert checkpoint == newest / "checkpoints/best.pth"
    assert completion["best_epoch"] == 15


def test_v4_8_build_rejects_nonbest_checkpoint(tmp_path):
    parent_run = tmp_path / "parent"
    wrong_checkpoint = parent_run / "checkpoints/epoch_015.pth"
    wrong_checkpoint.parent.mkdir(parents=True)
    wrong_checkpoint.write_bytes(b"wrong")
    source_manifest = tmp_path / "source.json"
    derived_manifest = tmp_path / "derived.json"
    _write_json(source_manifest, {"dataset_version": entry.SOURCE_VERSION})
    _write_json(derived_manifest, _derived_manifest())
    with pytest.raises(ValueError, match="best.pth"):
        entry.build_config(
            _parent(), parent_run, wrong_checkpoint, {"best_epoch": 15},
            source_manifest, derived_manifest,
        )


def test_v4_8_build_contract_is_five_epoch_low_lr_and_ungated(tmp_path):
    config = _build(tmp_path)
    assert config["contract_version"] == entry.TRAINING_CONTRACT
    assert config["experiment_role"] == (
        "recovery_capacity_five_epoch_shakedown"
    )
    assert config["source_dataset_root"] == str(entry.SOURCE)
    assert config["derived_dataset_root"] == str(entry.DERIVED)
    assert config["model"]["strict_load"] is True
    assert config["model"]["initial_checkpoint"].endswith("best.pth")
    assert config["observation"] == {
        "contract": "route_a_recovery_state_v4_8",
        "seed": 84710,
    }
    assert config["training"]["max_epochs"] == 5
    assert config["training"]["seed"] == 84801
    assert config["optimizer"]["backbone_learning_rate"] == 5.0e-8
    assert config["optimizer"]["candidate_head_learning_rate"] == 1.0e-6
    assert config["optimizer"]["score_head_learning_rate"] == 2.0e-6
    assert config["validation"]["contract_version"] == (
        "route_a_v4_8_recovery_capacity_v1"
    )
    assert config["route_a_v4_8"]["qualification_gate_count"] == 0


def test_v4_8_objective_inherits_v4510_and_adds_only_bounded_sector_fields(
    tmp_path,
):
    parent = _parent()
    original_parent = copy.deepcopy(parent)
    config = _build(tmp_path)
    assert parent == original_parent
    assert config["static_yopo_v4_5_10"]["enabled"] is False
    v48 = config["static_yopo_v4_8"]
    for name, value in original_parent["static_yopo_v4_5_10"].items():
        if name != "enabled":
            assert v48[name] == value
    inherited = dict(original_parent["static_yopo_v4_5_10"])
    inherited["enabled"] = True
    assert {
        name: value for name, value in v48.items()
        if not name.startswith("safe_sector_")
    } == inherited
    assert {
        name: v48[name] for name in (
            "safe_sector_weight",
            "safe_sector_ray_start_m",
            "safe_sector_ray_end_m",
            "safe_sector_ray_samples",
            "safe_sector_openness_center_m",
            "safe_sector_openness_softness_m",
            "safe_sector_target_length_m",
            "safe_sector_shortfall_softness_m",
        )
    } == {
        "safe_sector_weight": 0.05,
        "safe_sector_ray_start_m": 0.5,
        "safe_sector_ray_end_m": 4.0,
        "safe_sector_ray_samples": 12,
        "safe_sector_openness_center_m": 0.45,
        "safe_sector_openness_softness_m": 0.10,
        "safe_sector_target_length_m": 3.0,
        "safe_sector_shortfall_softness_m": 0.35,
    }


def test_v4_8_shell_entries_preserve_launch_contract():
    expected = {
        "route_a_v4_8_train_host.sh": (
            "route_a_v4_8_recovery_capacity_shakedown.yaml"
        ),
        "route_a_v4_8_training_status.sh": (
            "route_a_static_yopo_v4_8_recovery_capacity_shakedown"
        ),
        "route_a_v4_8_summary.sh": "summarize_route_a_v4_8_training.py",
        "route_a_v4_8_rviz_host.sh": (
            "dep_interactive_demo_scenes_v4_6.json"
        ),
    }
    for name, token in expected.items():
        text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert token in text

    train = (ROOT / "scripts/route_a_v4_8_train_host.sh").read_text()
    assert "validate_route_a_v4_7_dataset.py" in train
    assert "prepare_route_a_v4_8_training_config.py" in train
    assert "--verify-only" in train
    assert "--authorized" in train
    assert "generate_dataset" not in train

    rviz = (ROOT / "scripts/route_a_v4_8_rviz_host.sh").read_text()
    assert "validate_route_a_launch_fixture.py" in rviz
    assert "--runtime-profile v4_7_balanced_dynamic" in rviz
    assert '[[ "$SCENE" == "pillar" ]]' in rviz
    assert "--deadlock-recovery-profile bounded_scan_v3" in rviz
    assert "cave|forest|pillar|wall" in rviz


def test_v4_8_checkpoint_readiness_is_ungated():
    assert entry.TRAINING_CONTRACT in UNGATED_STATIC_TRAINING_CONTRACTS


def test_v4_8_dry_run_does_not_freeze_future_config_refresh(tmp_path):
    dry = tmp_path / "dry"
    formal = tmp_path / "formal"
    _write_json(dry / "run_state.json", {
        "status": "DRY_RUN_COMPLETE", "long_training_started": False,
    })
    assert entry.formal_training_started(tmp_path) is False
    _write_json(formal / "run_state.json", {
        "status": "RUNNING", "long_training_started": True,
    })
    assert entry.formal_training_started(tmp_path) is True
