#!/usr/bin/env python3
"""Freeze V4.2.8 goal-directed mobility and safety fine-tuning."""

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

PARENT = ROOT / "configs/route_a_v4_2_7_recovery_split_training.yaml"
OUTPUT = ROOT / "configs/route_a_v4_2_8_goal_mobility_training.yaml"
INITIAL_CHECKPOINT = (
    ROOT / "runs/route_a_static_yopo_v4_2_3_ego_feasibility"
    / "20260802T141747Z-35778/checkpoints/best_unqualified.pth"
)


def main():
    if not INITIAL_CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"missing V4.2.3 mobility initialization: {INITIAL_CHECKPOINT}"
        )
    yaml = YAML()
    config = yaml.load(PARENT)
    config["contract_version"] = (
        "route_a_static_yopo_training_v4_2_8_goal_mobility"
    )
    config["parent_v4_2_7_config_hash"] = sha256_file(PARENT)
    config["training_implementation_hash"] = training_implementation_hash()
    config["output_root"] = str(
        ROOT / "runs/route_a_static_yopo_v4_2_8_goal_mobility"
    )
    config["model"].update({
        "head_variant": "split",
        "allow_unified_to_split": True,
        "initial_checkpoint": str(INITIAL_CHECKPOINT),
        "initial_checkpoint_sha256": sha256_file(INITIAL_CHECKPOINT),
    })
    # V4.2.3 reached goals with this distribution. Recovery-specific states
    # return only after signed goal progress is proven in closed loop.
    config["observation"].update({
        "contract": "route_a_wide_state_v1",
        "seed": 82801,
    })
    for key in list(config["observation"]):
        if key.startswith("recovery_"):
            del config["observation"][key]
    config["objective"]["local_goal_horizon_m"] = 10.0

    config["optimizer"].update({
        "learning_rate": 1.0e-5,
        "backbone_learning_rate": 1.0e-7,
        "candidate_head_learning_rate": 1.0e-6,
        "score_head_learning_rate": 1.0e-5,
        "weight_decay": 2.0e-4,
    })
    config["training"].update({
        "max_epochs": 50,
        "seed": 82801,
        "score_only_warmup_epochs": 3,
    })
    config["scheduler"].update({
        "factor": 0.5,
        "patience": 5,
        "threshold": 2.0e-4,
        "minimum_learning_rate": {
            "backbone": 1.0e-8,
            "candidate_head": 1.0e-7,
            "score_head": 1.0e-6,
        },
    })

    # Return the proposal gradients to the V4.2.3 scale. Hard 6/6 limits are
    # unchanged; they are no longer allowed to dominate goal progress 8:1.
    config["safety_first"].update({
        "unsafe_label_priority": 8.0,
        "safety_cvar_weight": 0.5,
        "kinematic_weight": 2.0,
        "ranking_weight": 0.0,
    })
    config["kinodynamic_v2"].update({
        "soft_limit_ratio": 0.90,
        "normal_acceleration_soft_ratio": 0.75,
        "dense_norm_weight": 1.0,
        "axis_weight": 0.25,
        "normal_acceleration_weight": 0.5,
        "time_dilation_weight": 1.0,
        "candidate_mean_weight": 0.5,
        "candidate_cvar_weight": 0.5,
        "feasible_coverage_weight": 0.5,
        "minimum_feasible_candidates": 3,
        "label_weight": 1.0,
    })
    config["preventive_safety"].update({
        "base_clearance_m": 1.0,
        "speed_clearance_gain_s": 0.08,
        "maximum_clearance_m": 1.5,
        "loss_weight": 0.5,
        "ranking_weight": 0.0,
        "label_weight": 0.0,
        "candidate_mean_weight": 0.25,
        "candidate_cvar_weight": 0.5,
        "feasible_coverage_weight": 0.5,
        "minimum_clear_candidates": 2,
        "maximum_sample_weight": 2.0,
    })
    config["progress_safety"] = {
        **dict(config["progress_safety"]),
        "enabled": False,
    }
    config["goal_progress_v2"] = {
        "enabled": True,
        "minimum_path_length_m": 1.0,
        "minimum_goal_progress_m": 0.25,
        "preferred_goal_progress_m": 3.0,
        "maximum_reverse_progress_m": 0.05,
        "softplus_temperature_m": 0.25,
        "loss_weight": 2.0,
        "reverse_weight": 4.0,
        "ranking_weight": 1.0,
        "ranking_margin": 0.75,
        "label_weight": 1.0,
        "minimum_goal_progress_candidates": 3,
    }
    config["feasibility_score"].update({
        "regression_weight": 0.5,
        "listwise_weight": 1.0,
        "ranking_weight": 1.0,
        "ranking_margin": 1.0,
    })
    config["validation"].update({
        "primary_metric": "hard_safety_weighted_macro_loss",
        "minimum_epoch": 12,
        "patience": 16,
        "selection_gate": {
            "unsafe_selection_rate_max": 0.15,
            "hardware_unsafe_selection_rate_max": 0.15,
            "hover_selection_rate_max": 0.10,
            "feasible_candidate_count_mean_min": 3.0,
            "selected_goal_progress_mean_min": 1.5,
            "selected_goal_alignment_mean_min": 0.30,
            "reverse_selection_rate_max": 0.02,
            "insufficient_progress_selection_rate_max": 0.20,
        },
        "hard_safety_metric_weights": {
            "unsafe_selection": 10.0,
            "hardware_unsafe_selection": 8.0,
            "hover_selection": 4.0,
            "reverse_selection": 20.0,
            "insufficient_progress_selection": 5.0,
        },
    })
    for metric in (
        "selected_goal_progress", "selected_goal_alignment",
        "reverse_selection",
    ):
        if metric not in config["validation"]["predefined_metrics"]:
            config["validation"]["predefined_metrics"].append(metric)
    config["implementation_hotfixes"] = [
        *config.get("implementation_hotfixes", []),
        "route_a_v4_2_8_v423_candidate_mobility_rebase_v1",
        "route_a_v4_2_8_signed_goal_progress_v2",
        "route_a_v4_2_8_goal_aware_recovery_release_v1",
        "route_a_v4_2_8_score_only_then_joint_finetune_v1",
    ]

    if OUTPUT.exists():
        existing = yaml.load(OUTPUT)
        if dict(existing) != dict(config):
            raise FileExistsError(
                f"refusing to overwrite divergent config: {OUTPUT}"
            )
    else:
        with OUTPUT.open("w", encoding="utf-8") as stream:
            yaml.dump(config, stream)
    print(json.dumps({
        "status": "PASS",
        "config": str(OUTPUT),
        "config_sha256": sha256_file(OUTPUT),
        "initial_checkpoint": str(INITIAL_CHECKPOINT),
        "mobility_baseline": "V4.2.3",
        "head_migration": "unified_to_split_exact",
        "dataset_reused": str(config["derived_dataset_root"]),
        "dataset_generation_required": False,
        "training_started": False,
    }, indent=2))


if __name__ == "__main__":
    main()
