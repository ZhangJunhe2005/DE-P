from __future__ import annotations

import json
from pathlib import Path

import pytest
from ruamel.yaml import YAML

import tools.prepare_route_a_v4_7_training_config as entry
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


def test_v4_7_dataset_identity_uses_new_roots_and_exact_four_types(tmp_path):
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
        "cave", "forest", "pillar", "wall"
    }


def test_v4_7_dataset_identity_rejects_room_or_unfrozen_data(tmp_path):
    source = tmp_path / "raw"
    derived = tmp_path / "derived"
    _write_json(source / "manifests/dataset_manifest.json", {
        "dataset_version": entry.SOURCE_VERSION,
    })
    invalid = _derived_manifest(status="INCOMPLETE")
    invalid["map_type_sample_counts"]["train"]["room"] = 1
    _write_json(derived / "manifests/dataset_manifest.json", invalid)
    with pytest.raises(RuntimeError):
        entry.validate_dataset_identity(source, derived)


def test_v4_7_entry_contract_is_balanced_low_lr_and_ungated(tmp_path):
    parent = YAML(typ="safe").load(entry.PARENT)
    checkpoint = tmp_path / "best.pth"
    checkpoint.write_bytes(b"checkpoint fixture")
    source_manifest = tmp_path / "source.json"
    derived_manifest = tmp_path / "derived.json"
    _write_json(source_manifest, {"dataset_version": entry.SOURCE_VERSION})
    _write_json(derived_manifest, _derived_manifest())
    config = entry.build_config(
        parent, tmp_path / "parent_run", checkpoint, {"best_epoch": 22},
        source_manifest, derived_manifest,
    )
    assert config["loader"]["sampling_strategy"] == "map_type_balanced"
    assert config["route_a_v4_7"]["qualification_gate_count"] == 0
    assert config["route_a_v4_7"]["loss_contract_changed"] is False
    assert config["training"]["max_epochs"] == 20
    assert config["optimizer"]["backbone_learning_rate"] == 5.0e-8
    assert config["optimizer"]["candidate_head_learning_rate"] == 5.0e-7
    assert config["optimizer"]["score_head_learning_rate"] == 1.0e-5
    assert config["validation"]["contract_version"] == (
        "route_a_v4_5_10_tail_aware_safety_v1"
    )
    assert config["static_yopo_v4_5_10"] == parent["static_yopo_v4_5_10"]


def test_v4_7_shell_entries_reference_only_v4_7_outputs():
    expected = {
        "route_a_v4_7_train_host.sh": "route_a_v4_7_original_density_finetune.yaml",
        "route_a_v4_7_training_status.sh": (
            "route_a_static_yopo_v4_7_original_density_finetune"
        ),
        "route_a_v4_7_summary.sh": "summarize_route_a_v4_7_training.py",
        # Keep one frozen held-out fixture for an interpretable V4.6/V4.7 A/B.
        "route_a_v4_7_rviz_host.sh": "dep_interactive_demo_scenes_v4_6.json",
    }
    for name, token in expected.items():
        text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert token in text
        assert "route_a_v4_6_rviz_host.sh" not in text
    train = (ROOT / "scripts/route_a_v4_7_train_host.sh").read_text()
    assert "validate_route_a_v4_7_dataset.py" in train


def test_v4_7_rviz_validates_frozen_fixture_and_accepts_explicit_checkpoint():
    text = (ROOT / "scripts/route_a_v4_7_rviz_host.sh").read_text()
    assert "validate_route_a_launch_fixture.py" in text
    assert "dep_interactive_demo_scenes_v4_6.json" in text
    assert "--checkpoint" in text
    assert "cave|forest|pillar|wall" in text
    assert '[[ "$SCENE" == "pillar" ]]' in text
    assert "--deadlock-recovery-profile bounded_scan_v3" in text


def test_v4_7_checkpoint_uses_pending_closed_loop_readiness_semantics():
    assert (
        "route_a_static_yopo_training_v4_7_original_density_finetune_v1"
        in UNGATED_STATIC_TRAINING_CONTRACTS
    )
