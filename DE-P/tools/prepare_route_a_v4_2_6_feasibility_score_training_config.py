#!/usr/bin/env python3
"""Freeze the V4.2.6 single feasibility-score training contract."""

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

PARENT = ROOT / "configs/route_a_v4_2_5_progress_recovery_training.yaml"
OUTPUT = ROOT / "configs/route_a_v4_2_6_feasibility_score_training.yaml"
INITIAL_CHECKPOINT = (
    ROOT / "runs/route_a_static_yopo_v4_2_5_safe_progress_recovery"
    / "20260804T100125Z-10690/checkpoints/best_unqualified.pth"
)


def main():
    if not INITIAL_CHECKPOINT.is_file():
        raise FileNotFoundError(
            "V4.2.6 requires the retained V4.2.5 checkpoint: "
            f"{INITIAL_CHECKPOINT}"
        )
    yaml = YAML()
    config = yaml.load(PARENT)
    config["contract_version"] = (
        "route_a_static_yopo_training_v4_2_6_feasibility_score"
    )
    config["parent_v4_2_5_config_hash"] = sha256_file(PARENT)
    config["training_implementation_hash"] = training_implementation_hash()
    config["output_root"] = str(
        ROOT / "runs/route_a_static_yopo_v4_2_6_feasibility_score"
    )
    config["model"]["initial_checkpoint"] = str(INITIAL_CHECKPOINT)
    config["model"]["initial_checkpoint_sha256"] = sha256_file(
        INITIAL_CHECKPOINT
    )
    # Preserve the learned visual representation and give the unified output
    # head enough freedom to repair candidate ordering.
    config["optimizer"].update({
        "learning_rate": 1.0e-5,
        "backbone_learning_rate": 2.0e-7,
        "head_learning_rate": 1.0e-5,
        "weight_decay": 5.0e-4,
    })
    config["training"].update({"max_epochs": 40, "seed": 82601})
    config["observation"]["seed"] = 82601
    config["scheduler"].update({
        "factor": 0.5,
        "patience": 4,
        "threshold": 2.0e-4,
        "minimum_learning_rate": {"backbone": 1.0e-7, "head": 1.0e-6},
    })
    config["validation"].update({"minimum_epoch": 8, "patience": 12})

    # Disable the three overlapping score-ranking paths.  Their continuous
    # candidate-generation losses remain active below.
    config["safety_first"]["ranking_weight"] = 0.0
    config["preventive_safety"]["ranking_weight"] = 0.0
    config["preventive_safety"]["label_weight"] = 0.0
    config["progress_safety"].update({
        "preferred_progress_m": 2.0,
        "loss_weight": 0.75,
        "ranking_weight": 0.0,
        "label_weight": 0.0,
        "minimum_progress_candidates": 2,
    })
    config["feasibility_score"] = {
        "enabled": True,
        "required_clearance_m": 0.65,
        "max_speed_mps": 6.0,
        "max_acceleration_mps2": 6.0,
        "infeasible_label_floor": 2.0,
        "regression_weight": 0.5,
        "listwise_weight": 2.0,
        "ranking_weight": 2.0,
        "ranking_margin": 1.0,
        "target_temperature": 0.35,
        "prediction_temperature": 1.0,
        "quality_epsilon": 1.0e-6,
    }

    # Only genuine hard-safety and anti-stall proxies block a checkpoint.
    # Candidate-capacity, preventive margin, and 3.5 m progress remain visible
    # diagnostics rather than an expanding family of release gates.
    config["validation"]["selection_gate"] = {
        "unsafe_selection_rate_max": 0.05,
        "hardware_unsafe_selection_rate_max": 0.05,
        "hover_selection_rate_max": 0.20,
    }
    config["implementation_hotfixes"] = [
        *config.get("implementation_hotfixes", []),
        "route_a_v4_2_6_single_hard_feasibility_score_contract_v1",
        "route_a_v4_2_6_argmin_aligned_listwise_score_loss_v1",
        "route_a_v4_2_6_diagnostic_only_candidate_progress_metrics_v1",
    ]

    if OUTPUT.exists():
        existing = yaml.load(OUTPUT)
        if dict(existing) != dict(config):
            raise FileExistsError(
                f"refusing to overwrite divergent V4.2.6 contract: {OUTPUT}"
            )
    else:
        with OUTPUT.open("w", encoding="utf-8") as stream:
            yaml.dump(config, stream)
    print(json.dumps({
        "status": "PASS",
        "config": str(OUTPUT),
        "config_sha256": sha256_file(OUTPUT),
        "initial_checkpoint": str(INITIAL_CHECKPOINT),
        "initial_checkpoint_sha256": sha256_file(INITIAL_CHECKPOINT),
        "dataset_reused": str(config["derived_dataset_root"]),
        "formal_dataset_generation_required": False,
        "training_started": False,
    }, indent=2))


if __name__ == "__main__":
    main()
