#!/usr/bin/env python3
"""Prepare the bounded V4.8.4 Score-only adaptation."""

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


PARENT = ROOT / "configs/route_a_v4_8_3_candidate_only_shakedown.yaml"
PARENT_RUN_ROOT = (
    ROOT / "runs/route_a_static_yopo_v4_8_3_candidate_only_shakedown"
)
CLOSED_LOOP_ROOT = ROOT / "runs/dep_interactive_demo"
OUTPUT = ROOT / "configs/route_a_v4_8_4_score_adaptation.yaml"
RUN_ROOT = ROOT / "runs/route_a_static_yopo_v4_8_4_score_adaptation"
TRAINING_CONTRACT = "route_a_static_yopo_training_v4_8_4_score_adaptation_v1"
VALIDATION_CONTRACT = "route_a_v4_8_4_score_adaptation_v1"
EXPERIMENT_ROLE = "recovery_score_only_two_epoch_adaptation"
PARENT_COMPLETION_STATUS = "DIAGNOSTIC_COMPLETE_NOT_PRODUCTION"
PARENT_BEST_EPOCH = 2
RUNTIME_PROFILE = "v4_8_2_universal_stagnation_recovery"


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
        if int(completion.get("best_epoch", -999)) != PARENT_BEST_EPOCH:
            continue
        candidates.append((run, checkpoint, completion))
    if not candidates:
        raise RuntimeError(
            "no completed V4.8.3 run with best_epoch=2; the validated parent "
            "checkpoint is required"
        )
    return candidates[-1]


def latest_closed_loop_evidence(
    checkpoint: Path, closed_loop_root: Path = CLOSED_LOOP_ROOT,
):
    checkpoint = checkpoint.resolve()
    candidates = []
    for run in sorted(path for path in closed_loop_root.glob("*-pillar")
                      if path.is_dir()):
        manifest_path = run / "manifest.json"
        collision_path = run / "collision_report.json"
        if not (manifest_path.is_file() and collision_path.is_file()):
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        collision = json.loads(collision_path.read_text(encoding="utf-8"))
        recorded = Path(str(manifest.get("checkpoint", ""))).expanduser()
        if not recorded.is_absolute() or recorded.resolve() != checkpoint:
            continue
        if manifest.get("runtime_profile") != RUNTIME_PROFILE:
            continue
        if manifest.get("goal_mode") != "fixed-ab":
            continue
        if manifest.get("actors") != "none":
            continue
        if collision.get("status") != "NO_COLLISION":
            continue
        if int(collision.get("any_collision_events", -1)) != 0:
            continue
        if not bool(collision.get("goal_arrived")):
            continue
        candidates.append((run, manifest, collision))
    if not candidates:
        raise RuntimeError(
            "no zero-collision fixed Pillar closed-loop evidence for the "
            "V4.8.3 parent checkpoint"
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
    parent, parent_run, checkpoint, completion, closed_loop_run,
    closed_loop_collision, source_manifest, derived_manifest,
):
    parent_run = Path(parent_run).resolve()
    checkpoint = Path(checkpoint).resolve()
    closed_loop_run = Path(closed_loop_run).resolve()
    if checkpoint != parent_run / "checkpoints/best.pth":
        raise ValueError("V4.8.4 must initialize from V4.8.3 best.pth")
    if int(completion.get("best_epoch", -999)) != PARENT_BEST_EPOCH:
        raise ValueError("V4.8.4 parent must be V4.8.3 epoch 2")
    if not bool(closed_loop_collision.get("goal_arrived")):
        raise ValueError("V4.8.3 parent closed-loop evidence did not arrive")
    if int(closed_loop_collision.get("any_collision_events", -1)) != 0:
        raise ValueError("V4.8.3 parent closed-loop evidence contains collision")
    config = copy.deepcopy(parent)
    config["contract_version"] = TRAINING_CONTRACT
    config["training_implementation_hash"] = training_implementation_hash()
    config["experiment_role"] = EXPERIMENT_ROLE
    config["parent_v4_8_3_config_hash"] = sha256_file(PARENT)
    config["parent_v4_8_3_run"] = str(parent_run)
    config["parent_v4_8_3_best_epoch"] = int(completion["best_epoch"])
    config["parent_v4_8_3_closed_loop_run"] = str(closed_loop_run)
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
        "candidate_head_learning_rate": 0.0,
        "score_head_learning_rate": 5.0e-7,
        "score_only_learning_rate": 5.0e-7,
        "weight_decay": 1.0e-5,
    }
    config["scheduler"] = {
        "name": "ReduceLROnPlateau",
        "factor": 0.5,
        "patience": 1,
        "threshold": 1.0e-4,
        "minimum_learning_rate": {
            "backbone": 0.0,
            "candidate_head": 0.0,
            "score_head": 1.25e-7,
        },
    }
    config["training"].update({
        "max_epochs": 2,
        "seed": 84804,
        "score_only_warmup_epochs": 0,
        "trainable_groups": "score_head_only",
        "evaluate_initial_checkpoint": True,
    })
    if float(config["static_yopo_v4_8"]["safe_sector_weight"]) != 0.10:
        raise ValueError("V4.8.4 must preserve V4.8.3 safe-sector weight")
    config["validation"].update({
        "contract_version": VALIDATION_CONTRACT,
        "primary_metric": "v4_8_4_score_adaptation",
        "minimum_epoch": 2,
        "patience": 2,
    })
    config["route_a_v4_8_4"] = {
        "enabled": True,
        "initial_validation_epoch": -1,
        "ordinary_sample_fraction": 0.80,
        "recovery_sample_fraction": 0.20,
        "trainable_groups": ["score_head"],
        "frozen_groups": ["backbone", "candidate_head"],
        "maximum_epochs": 2,
        "score_head_learning_rate": 5.0e-7,
        "qualification_gate_count": 0,
        "candidate_contract": "frozen_v4_8_3_epoch2",
        "selection_metric": (
            "ordinary_four_map_macro_total_loss_plus_conditional_"
            "recovery_safe_sector_loss"
        ),
    }
    config["implementation_hotfixes"] = list(
        config.get("implementation_hotfixes", [])
    ) + [
        "route_a_v4_8_4_initialize_from_closed_loop_v483_epoch2_v1",
        "route_a_v4_8_4_record_epoch_minus_one_parent_baseline_v1",
        "route_a_v4_8_4_freeze_backbone_and_candidate_head_v1",
        "route_a_v4_8_4_score_lr_5e_7_v1",
        "route_a_v4_8_4_two_epoch_no_new_gate_v1",
    ]
    return config


def main():
    source_manifest, derived_manifest, derived = validate_dataset_identity()
    parent_run, checkpoint, completion = latest_parent_checkpoint()
    closed_loop_run, _, collision = latest_closed_loop_evidence(checkpoint)
    parent = YAML(typ="safe").load(PARENT)
    config = build_config(
        parent, parent_run, checkpoint, completion, closed_loop_run, collision,
        source_manifest, derived_manifest,
    )
    if OUTPUT.exists():
        existing = YAML(typ="safe").load(OUTPUT)
        if existing != config and formal_training_started():
            raise RuntimeError(
                "refusing to change V4.8.4 config after formal training started"
            )
    yaml = YAML()
    with OUTPUT.open("w", encoding="utf-8") as stream:
        yaml.dump(config, stream)
    print(json.dumps({
        "status": "PASS",
        "config": str(OUTPUT),
        "config_sha256": sha256_file(OUTPUT),
        "dataset_reused_from": str(DERIVED),
        "samples": derived["split_counts"],
        "initial_checkpoint": str(checkpoint),
        "parent_best_epoch": int(completion["best_epoch"]),
        "parent_closed_loop_run": str(closed_loop_run),
        "epochs": 2,
        "trainable_groups": ["score_head"],
        "frozen_groups": ["backbone", "candidate_head"],
        "ordinary_recovery_ratio": [80, 20],
        "qualification_gate_count": 0,
        "training_started": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
