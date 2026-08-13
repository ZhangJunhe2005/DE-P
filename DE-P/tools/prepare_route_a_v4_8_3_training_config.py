#!/usr/bin/env python3
"""Prepare the bounded V4.8.3 candidate-only recovery shakedown."""

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
from tools.prepare_route_a_v4_8_training_config import (
    DERIVED,
    SOURCE,
    validate_dataset_identity,
)
from tools.train_mixed_static_yopo_v1 import training_implementation_hash


PARENT = ROOT / "configs/route_a_v4_8_recovery_capacity_shakedown.yaml"
PARENT_RUN_ROOT = (
    ROOT / "runs/route_a_static_yopo_v4_8_recovery_capacity_shakedown"
)
OUTPUT = ROOT / "configs/route_a_v4_8_3_candidate_only_shakedown.yaml"
RUN_ROOT = ROOT / "runs/route_a_static_yopo_v4_8_3_candidate_only_shakedown"
TRAINING_CONTRACT = (
    "route_a_static_yopo_training_v4_8_3_candidate_only_shakedown_v1"
)
VALIDATION_CONTRACT = "route_a_v4_8_3_candidate_only_recovery_v1"
EXPERIMENT_ROLE = "recovery_candidate_only_three_epoch_shakedown"
PARENT_COMPLETION_STATUS = "DIAGNOSTIC_COMPLETE_NOT_PRODUCTION"


def latest_parent_checkpoint(run_root: Path = PARENT_RUN_ROOT):
    candidates = []
    for run in sorted(path for path in run_root.glob("*") if path.is_dir()):
        completion_path = run / "training_complete.json"
        checkpoint = run / "checkpoints/best.pth"
        if not (completion_path.is_file() and checkpoint.is_file()):
            continue
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("status") != PARENT_COMPLETION_STATUS:
            continue
        if int(completion.get("best_epoch", -999)) != 0:
            continue
        candidates.append((run, checkpoint, completion))
    if not candidates:
        raise RuntimeError(
            "no completed V4.8 run with best_epoch=0; the validated parent "
            "checkpoint is required"
        )
    return candidates[-1]


def formal_training_started(run_root: Path = RUN_ROOT):
    for run in sorted(path for path in run_root.glob("*") if path.is_dir()):
        state_path = run / "run_state.json"
        if not state_path.is_file():
            continue
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if bool(state.get("long_training_started")):
            return True
    return False


def build_config(
    parent, parent_run, checkpoint, completion,
    source_manifest, derived_manifest,
):
    parent_run = Path(parent_run).resolve()
    checkpoint = Path(checkpoint).resolve()
    if checkpoint != parent_run / "checkpoints/best.pth":
        raise ValueError("V4.8.3 must initialize from the V4.8 best.pth")
    if int(completion.get("best_epoch", -999)) != 0:
        raise ValueError("V4.8.3 parent must be the validated V4.8 epoch 0")
    config = copy.deepcopy(parent)
    config["contract_version"] = TRAINING_CONTRACT
    config["training_implementation_hash"] = training_implementation_hash()
    config["experiment_role"] = EXPERIMENT_ROLE
    config["parent_v4_8_config_hash"] = sha256_file(PARENT)
    config["parent_v4_8_run"] = str(parent_run)
    config["parent_v4_8_best_epoch"] = int(completion["best_epoch"])
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
    config["optimizer"] = {
        "name": "AdamW",
        "learning_rate": 5.0e-7,
        "backbone_learning_rate": 0.0,
        "candidate_head_learning_rate": 5.0e-7,
        "score_head_learning_rate": 0.0,
        "score_only_learning_rate": 0.0,
        "weight_decay": 1.0e-5,
    }
    config["scheduler"] = {
        "name": "ReduceLROnPlateau",
        "factor": 0.5,
        "patience": 1,
        "threshold": 1.0e-4,
        "minimum_learning_rate": {
            "backbone": 0.0,
            "candidate_head": 1.25e-7,
            "score_head": 0.0,
        },
    }
    config["training"].update({
        "max_epochs": 3,
        "seed": 84803,
        "score_only_warmup_epochs": 0,
        "trainable_groups": "candidate_head_only",
        "evaluate_initial_checkpoint": True,
    })
    config["static_yopo_v4_8"]["safe_sector_weight"] = 0.10
    config["validation"].update({
        "contract_version": VALIDATION_CONTRACT,
        "primary_metric": "v4_8_3_normal_plus_recovery_safe_sector",
        "minimum_epoch": 3,
        "patience": 3,
    })
    config["route_a_v4_8_3"] = {
        "enabled": True,
        "initial_validation_epoch": -1,
        "ordinary_sample_fraction": 0.80,
        "recovery_sample_fraction": 0.20,
        "trainable_groups": ["candidate_head"],
        "frozen_groups": ["backbone", "score_head"],
        "maximum_epochs": 3,
        "qualification_gate_count": 0,
        "selection_metric": (
            "ordinary_four_map_macro_total_loss_plus_conditional_"
            "recovery_safe_sector_loss"
        ),
    }
    config["implementation_hotfixes"] = list(
        config.get("implementation_hotfixes", [])
    ) + [
        "route_a_v4_8_3_initialize_from_closed_loop_v48_epoch0_v1",
        "route_a_v4_8_3_record_epoch_minus_one_parent_baseline_v1",
        "route_a_v4_8_3_freeze_backbone_and_score_head_v1",
        "route_a_v4_8_3_candidate_lr_5e_7_v1",
        "route_a_v4_8_3_safe_sector_weight_0_10_v1",
        "route_a_v4_8_3_three_epoch_no_new_gate_v1",
    ]
    return config


def main():
    source_manifest, derived_manifest, derived = validate_dataset_identity()
    parent_run, checkpoint, completion = latest_parent_checkpoint()
    yaml = YAML()
    parent = yaml.load(PARENT)
    config = build_config(
        parent, parent_run, checkpoint, completion,
        source_manifest, derived_manifest,
    )
    if OUTPUT.exists():
        existing = YAML(typ="safe").load(OUTPUT)
        if existing != config and formal_training_started():
            raise RuntimeError(
                "refusing to change V4.8.3 config after formal training started"
            )
    with OUTPUT.open("w", encoding="utf-8") as stream:
        yaml.dump(config, stream)
    print(json.dumps({
        "status": "PASS",
        "config": str(OUTPUT),
        "config_sha256": sha256_file(OUTPUT),
        "dataset_reused_from": str(DERIVED),
        "samples": derived["split_counts"],
        "initial_checkpoint": str(checkpoint),
        "initial_checkpoint_epoch": -1,
        "parent_best_epoch": int(completion["best_epoch"]),
        "epochs": 3,
        "trainable_groups": ["candidate_head"],
        "frozen_groups": ["backbone", "score_head"],
        "ordinary_recovery_ratio": [80, 20],
        "qualification_gate_count": 0,
        "training_started": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
