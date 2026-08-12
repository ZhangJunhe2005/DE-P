#!/usr/bin/env python3
"""Bind V4.5.10 to the four-scene, lower-density-wall V4.6 dataset."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.static_yopo_manifest_v1 import sha256_file
from tools.train_mixed_static_yopo_v1 import training_implementation_hash


PARENT = ROOT / "configs/route_a_v4_5_10_controlled_continuation.yaml"
PARENT_RUN_ROOT = ROOT / "runs/route_a_static_yopo_v4_5_10_controlled_continuation"
SOURCE = ROOT / "data/route_a_v4_6_raw_static"
DERIVED = ROOT / "data/route_a_v4_6_static_yopo"
OUTPUT = ROOT / "configs/route_a_v4_6_four_scene_finetune.yaml"
RUN_ROOT = ROOT / "runs/route_a_static_yopo_v4_6_four_scene_finetune"
EXPECTED_TYPES = {"cave", "pillar", "forest", "wall"}


def latest_parent_checkpoint():
    candidates = []
    for run in sorted(path for path in PARENT_RUN_ROOT.glob("*") if path.is_dir()):
        completion_path = run / "training_complete.json"
        checkpoint = run / "checkpoints/best.pth"
        if not (completion_path.is_file() and checkpoint.is_file()):
            continue
        completion = json.loads(completion_path.read_text())
        if completion.get("status") != "DIAGNOSTIC_COMPLETE_NOT_PRODUCTION":
            continue
        candidates.append((run, checkpoint, completion))
    if not candidates:
        raise RuntimeError("no completed V4.5.10 continuation checkpoint found")
    return candidates[-1]


def validate_dataset_identity():
    source_manifest = SOURCE / "manifests/dataset_manifest.json"
    derived_manifest = DERIVED / "manifests/dataset_manifest.json"
    if not source_manifest.is_file() or not derived_manifest.is_file():
        raise FileNotFoundError("generate and validate V4.6 before training")
    source = json.loads(source_manifest.read_text())
    derived = json.loads(derived_manifest.read_text())
    if source.get("dataset_version") != "route_a_v4_6_raw_static_v1":
        raise RuntimeError("V4.6 raw identity mismatch")
    if derived.get("dataset_version") != "route_a_v4_6_static_yopo" \
            or derived.get("status") != "COMPLETE_FROZEN":
        raise RuntimeError("V4.6 derived identity mismatch")
    for split in ("train", "validation"):
        actual = set(derived["map_type_sample_counts"][split])
        if actual != EXPECTED_TYPES:
            raise RuntimeError(f"V4.6 {split} is not exactly four-scene")
    return source_manifest, derived_manifest, derived


def main():
    source_manifest, derived_manifest, derived = validate_dataset_identity()
    parent_run, checkpoint, completion = latest_parent_checkpoint()
    yaml = YAML()
    config = copy.deepcopy(yaml.load(PARENT))
    config["contract_version"] = (
        "route_a_static_yopo_training_v4_6_four_scene_finetune_v1"
    )
    config["training_implementation_hash"] = training_implementation_hash()
    config["parent_v4_5_10_continuation_config_hash"] = sha256_file(PARENT)
    config["parent_v4_5_10_continuation_run"] = str(parent_run)
    config["parent_v4_5_10_continuation_best_epoch"] = int(
        completion["best_epoch"]
    )
    config["source_dataset_root"] = str(SOURCE)
    config["derived_dataset_root"] = str(DERIVED)
    config["source_dataset_manifest_hash"] = sha256_file(source_manifest)
    config["derived_dataset_manifest_hash"] = sha256_file(derived_manifest)
    config["output_root"] = str(RUN_ROOT)
    config["model"].update({
        "head_variant": "independent",
        "initial_checkpoint": str(checkpoint),
        "initial_checkpoint_sha256": sha256_file(checkpoint),
        "strict_load": True,
        "allow_unified_to_split": False,
    })
    # The previous run was a low-LR plateau rather than classical overfit.
    # Keep proposal drift small, allow the score branch to adapt to the new
    # four-scene distribution, and use early stopping instead of forcing 50.
    config["optimizer"] = {
        "name": "AdamW",
        "learning_rate": 2.0e-5,
        "backbone_learning_rate": 1.0e-7,
        "candidate_head_learning_rate": 1.0e-6,
        "score_head_learning_rate": 2.0e-5,
        "score_only_learning_rate": 2.0e-5,
        "weight_decay": 1.0e-5,
    }
    config["scheduler"] = {
        "name": "ReduceLROnPlateau",
        "factor": 0.5,
        "patience": 3,
        "threshold": 1.0e-4,
        "minimum_learning_rate": {
            "backbone": 2.5e-8,
            "candidate_head": 2.5e-7,
            "score_head": 2.5e-6,
        },
    }
    config["training"].update({
        "max_epochs": 30,
        "seed": 84610,
        "score_only_warmup_epochs": 0,
    })
    config["loader"]["sampling_strategy"] = "map_type_balanced"
    config["loader"].pop("map_type_filter", None)
    config["validation"].update({
        # Reuse the same single scalar objective. The macro is now computed
        # over the four types physically present in the frozen V4.6 dataset.
        "contract_version": "route_a_v4_5_10_tail_aware_safety_v1",
        "primary_metric": "macro_map_type_total_static_loss",
        "minimum_epoch": 10,
        "patience": 8,
    })
    config.pop("experiment_role", None)
    config["route_a_v4_6"] = {
        "enabled": True,
        "supported_map_types": sorted(EXPECTED_TYPES),
        "excluded_map_types": ["room"],
        "room_sample_count": 0,
        "wall_large_number": 100,
        "wall_narrow_number": 25,
        "map_type_balanced": True,
        "qualification_gate_count": 0,
    }
    config["implementation_hotfixes"] = list(
        config.get("implementation_hotfixes", [])
    ) + [
        "route_a_v4_6_exclude_room_from_data_train_and_validation_v1",
        "route_a_v4_6_restore_original_wall_density_v1",
        "route_a_v4_6_four_type_balanced_sampling_v1",
        "route_a_v4_6_thirty_epoch_bounded_finetune_v1",
        "route_a_v4_6_early_stop_without_new_gate_v1",
    ]
    if OUTPUT.exists():
        with OUTPUT.open(encoding="utf-8") as stream:
            existing = yaml.load(stream)
        if existing != config and any(RUN_ROOT.glob("*")):
            raise RuntimeError("refusing to change V4.6 config after training started")
    with OUTPUT.open("w", encoding="utf-8") as stream:
        yaml.dump(config, stream)
    print(json.dumps({
        "status": "PASS",
        "config": str(OUTPUT),
        "config_sha256": sha256_file(OUTPUT),
        "dataset": str(DERIVED),
        "samples": derived["split_counts"],
        "map_types": sorted(EXPECTED_TYPES),
        "room_samples": 0,
        "initial_checkpoint": str(checkpoint),
        "parent_best_epoch": int(completion["best_epoch"]),
        "maximum_epochs": 30,
        "earliest_early_stop_epoch": 10,
        "early_stop_patience": 8,
        "learning_rates": {
            "backbone": 1.0e-7,
            "candidate_head": 1.0e-6,
            "score_head": 2.0e-5,
        },
        "qualification_gate_count": 0,
        "training_started": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
