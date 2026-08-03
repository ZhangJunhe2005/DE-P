#!/usr/bin/env python3
"""Freeze V4.2.4 preventive safety training without mutating V4.2.3."""

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

PARENT = ROOT / "configs/route_a_v4_2_3_ego_feasibility_training.yaml"
OUTPUT = ROOT / "configs/route_a_v4_2_4_preventive_safety_training.yaml"
INITIAL_CHECKPOINT = (
    ROOT / "runs/route_a_static_yopo_v4_2_2_safety_first"
    / "20260802T115843Z-16376/checkpoints/best.pth"
)


def main():
    if not INITIAL_CHECKPOINT.is_file():
        raise FileNotFoundError(
            "V4.2.4 requires the empirically safer V4.2.2 initialization: "
            f"{INITIAL_CHECKPOINT}"
        )
    yaml = YAML()
    config = yaml.load(PARENT)
    config["contract_version"] = (
        "route_a_static_yopo_training_v4_2_4_preventive_safety"
    )
    config["parent_v4_2_3_config_hash"] = sha256_file(PARENT)
    config["training_implementation_hash"] = training_implementation_hash()
    config["output_root"] = str(
        ROOT / "runs/route_a_static_yopo_v4_2_4_preventive_safety"
    )
    config["model"]["initial_checkpoint"] = str(INITIAL_CHECKPOINT)
    config["model"]["initial_checkpoint_sha256"] = sha256_file(
        INITIAL_CHECKPOINT
    )

    # Ten metres in 1.7 s consumes virtually the complete 6 m/s envelope.
    # Seven metres preserves continuous progress while leaving real curvature
    # and braking authority for avoidance.
    config["objective"]["local_goal_horizon_m"] = 7.0
    config["optimizer"].update({
        "learning_rate": 1.0e-5,
        "backbone_learning_rate": 1.0e-6,
        "head_learning_rate": 1.0e-5,
        "weight_decay": 5.0e-4,
    })
    config["training"]["amp"] = False
    config["numerics"]["cnn_amp"] = False
    config["scheduler"].update({
        "factor": 0.5, "patience": 4, "threshold": 2.0e-4,
        "minimum_learning_rate": {"backbone": 2.5e-7, "head": 1.0e-6},
    })
    config["validation"].update({"minimum_epoch": 15, "patience": 10})
    config["safety_first"].update({
        "required_clearance_m": 0.65,
        "unsafe_label_priority": 6.0,
        "ranking_margin": 1.0,
        "ranking_weight": 0.75,
        "safety_cvar_weight": 0.75,
        "kinematic_weight": 8.0,
    })
    config["kinodynamic_v2"].update({
        "soft_limit_ratio": 0.85,
        "normal_acceleration_soft_ratio": 0.65,
        "dense_norm_weight": 1.5,
        "axis_weight": 0.25,
        "normal_acceleration_weight": 0.75,
        "time_dilation_weight": 2.0,
        "candidate_mean_weight": 0.50,
        "candidate_cvar_weight": 0.75,
        "feasible_coverage_weight": 2.0,
        "minimum_feasible_candidates": 4,
        "label_weight": 2.0,
    })
    config["preventive_safety"] = {
        "enabled": True,
        "base_clearance_m": 1.20,
        "speed_clearance_gain_s": 0.12,
        "maximum_clearance_m": 2.00,
        "softplus_temperature_m": 0.20,
        "loss_weight": 2.0,
        "ranking_weight": 1.0,
        "ranking_margin": 0.50,
        "label_weight": 2.0,
        "candidate_mean_weight": 0.25,
        "candidate_cvar_weight": 0.75,
        "candidate_cvar_fraction": 1.0 / 3.0,
        "feasible_coverage_weight": 1.0,
        "minimum_clear_candidates": 3,
        "approach_depth_m": 5.0,
        "near_depth_m": 3.0,
        "near_fraction_reference": 0.10,
        "maximum_sample_weight": 4.0,
        "depth_max_m": 20.0,
    }
    gate = config["validation"]["selection_gate"]
    gate.update({
        "selected_endpoint_distance_mean_min": 3.5,
        "selected_endpoint_speed_mean_min": 2.0,
        "hover_selection_rate_max": 0.20,
        "unsafe_selection_rate_max": 0.03,
        "hardware_unsafe_selection_rate_max": 0.05,
        "feasible_candidate_count_mean_min": 3.0,
        "selected_time_dilation_mean_max": 1.02,
        "anticipatory_unsafe_selection_rate_max": 0.20,
        "selected_clearance_mean_min": 1.20,
        "clear_candidate_count_mean_min": 3.0,
    })
    config["validation"]["predefined_metrics"] = list(dict.fromkeys([
        *config["validation"]["predefined_metrics"],
        "preventive_safety_loss", "preventive_ranking_loss",
        "preventive_required_clearance", "preventive_sample_weight",
        "clear_candidate_count", "selected_clearance",
        "anticipatory_unsafe_selection",
    ]))
    config["implementation_hotfixes"] = [
        *config.get("implementation_hotfixes", []),
        "route_a_v4_2_4_seven_metre_maneuver_budget_v1",
        "route_a_v4_2_4_speed_aware_preventive_clearance_v1",
        "route_a_v4_2_4_near_field_adaptive_weight_v1",
        "route_a_v4_2_4_clearance_score_ranking_v1",
        "route_a_v4_2_4_causal_dynamic_track_safety_v1",
    ]

    if OUTPUT.exists():
        existing = yaml.load(OUTPUT)
        if dict(existing) != dict(config):
            raise FileExistsError(
                f"refusing to overwrite divergent V4.2.4 contract: {OUTPUT}"
            )
    else:
        with OUTPUT.open("w", encoding="utf-8") as stream:
            yaml.dump(config, stream)
    print(json.dumps({
        "status": "PASS",
        "config": str(OUTPUT),
        "config_sha256": sha256_file(OUTPUT),
        "parent_sha256": sha256_file(PARENT),
        "initial_checkpoint": str(INITIAL_CHECKPOINT),
        "initial_checkpoint_sha256": sha256_file(INITIAL_CHECKPOINT),
        "local_goal_horizon_m": config["objective"]["local_goal_horizon_m"],
        "preventive_safety": dict(config["preventive_safety"]),
        "training_started": False,
    }, indent=2))


if __name__ == "__main__":
    main()
