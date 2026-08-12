#!/usr/bin/env python3
"""Freeze V4.2.7 recovery-state and split-head training."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.static_yopo_manifest_v1 import sha256_file
from tools.train_mixed_static_yopo_v1 import training_implementation_hash

PARENT = ROOT / "configs/route_a_v4_2_6_feasibility_score_training.yaml"
OUTPUT = ROOT / "configs/route_a_v4_2_7_recovery_split_training.yaml"
INITIAL_CHECKPOINT = (
    ROOT / "runs/route_a_static_yopo_v4_2_6_feasibility_score"
    / "20260804T154430Z-4268/checkpoints/best_unqualified.pth"
)


def main():
    if not INITIAL_CHECKPOINT.is_file():
        raise FileNotFoundError(f"missing V4.2.6 initialization: {INITIAL_CHECKPOINT}")
    yaml = YAML()
    config = yaml.load(PARENT)
    config["contract_version"] = (
        "route_a_static_yopo_training_v4_2_7_recovery_split"
    )
    config["parent_v4_2_6_config_hash"] = sha256_file(PARENT)
    config["training_implementation_hash"] = training_implementation_hash()
    config["output_root"] = str(
        ROOT / "runs/route_a_static_yopo_v4_2_7_recovery_split"
    )
    config["model"].update({
        "head_variant": "split",
        "allow_unified_to_split": True,
        "initial_checkpoint": str(INITIAL_CHECKPOINT),
        "initial_checkpoint_sha256": sha256_file(INITIAL_CHECKPOINT),
    })
    config["observation"].update({
        "contract": "route_a_recovery_state_v2",
        "seed": 82701,
        "recovery_sample_probability": 0.35,
        "recovery_speed_range_mps": [0.0, 0.75],
        "recovery_acceleration_range_mps2": [0.0, 1.0],
        "recovery_goal_yaw_range_deg": [-75.0, 75.0],
    })
    config["optimizer"].pop("head_learning_rate", None)
    config["optimizer"].update({
        "learning_rate": 1.0e-5,
        "backbone_learning_rate": 1.0e-7,
        "candidate_head_learning_rate": 3.0e-6,
        "score_head_learning_rate": 1.0e-5,
        "weight_decay": 5.0e-4,
    })
    config["training"].update({"max_epochs": 45, "seed": 82701})
    config["scheduler"].update({
        "factor": 0.5,
        "patience": 4,
        "threshold": 2.0e-4,
        "minimum_learning_rate": {
            "backbone": 1.0e-7,
            "candidate_head": 3.0e-7,
            "score_head": 1.0e-6,
        },
    })
    config["validation"].update({
        "primary_metric": "hard_safety_weighted_macro_loss",
        "hard_safety_metric_weights": {
            "unsafe_selection": 20.0,
            "hardware_unsafe_selection": 10.0,
            "hover_selection": 2.0,
        },
        "minimum_epoch": 10,
        "patience": 14,
    })
    config["implementation_hotfixes"] = [
        *config.get("implementation_hotfixes", []),
        "route_a_v4_2_7_recovery_state_observation_v2",
        "route_a_v4_2_7_exact_unified_to_split_head_migration_v1",
        "route_a_v4_2_7_independent_candidate_score_learning_rates_v1",
        "route_a_v4_2_7_hard_safety_weighted_checkpoint_selection_v1",
    ]

    if OUTPUT.exists():
        existing = yaml.load(OUTPUT)
        if dict(existing) != dict(config):
            raise FileExistsError(f"refusing to overwrite divergent config: {OUTPUT}")
    else:
        with OUTPUT.open("w", encoding="utf-8") as stream:
            yaml.dump(config, stream)
    print(json.dumps({
        "status": "PASS",
        "config": str(OUTPUT),
        "config_sha256": sha256_file(OUTPUT),
        "initial_checkpoint": str(INITIAL_CHECKPOINT),
        "head_migration": "unified_to_split_exact",
        "dataset_reused": str(config["derived_dataset_root"]),
        "dataset_generation_required": False,
        "training_started": False,
    }, indent=2))


if __name__ == "__main__":
    main()
