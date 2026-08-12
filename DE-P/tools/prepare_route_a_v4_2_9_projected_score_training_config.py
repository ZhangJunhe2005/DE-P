#!/usr/bin/env python3
"""Freeze Route-A V4.2.9 projected-score training without rebuilding data."""

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

PARENT = ROOT / "configs/route_a_v4_2_8_goal_mobility_training.yaml"
OUTPUT = ROOT / "configs/route_a_v4_2_9_projected_score_training.yaml"
INITIAL_CHECKPOINT = (
    ROOT / "runs/route_a_static_yopo_v4_2_8_goal_mobility"
    / "20260806T125140Z-12283/checkpoints/best_unqualified.pth"
)


def main():
    if not INITIAL_CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"missing frozen V4.2.8 mobility checkpoint: {INITIAL_CHECKPOINT}"
        )
    yaml = YAML()
    config = yaml.load(PARENT)
    config["contract_version"] = (
        "route_a_static_yopo_training_v4_2_9_projected_score_v1"
    )
    config["parent_v4_2_8_config_hash"] = sha256_file(PARENT)
    config["training_implementation_hash"] = training_implementation_hash()
    config["output_root"] = str(
        ROOT / "runs/route_a_static_yopo_v4_2_9_projected_score"
    )
    config["model"].update({
        "initial_checkpoint": str(INITIAL_CHECKPOINT),
        "initial_checkpoint_sha256": sha256_file(INITIAL_CHECKPOINT),
        "head_variant": "split",
        "allow_unified_to_split": False,
    })

    # Preserve V4.2.8's mobile candidate generator during score alignment,
    # then permit only a very small proposal/backbone fine-tune.
    config["optimizer"].update({
        "learning_rate": 1.0e-5,
        "backbone_learning_rate": 2.0e-8,
        "candidate_head_learning_rate": 5.0e-7,
        "score_head_learning_rate": 1.0e-5,
        "weight_decay": 2.0e-4,
    })
    config["training"].update({
        "max_epochs": 40,
        "seed": 82901,
        "score_only_warmup_epochs": 8,
    })
    config["scheduler"].update({
        "factor": 0.5,
        "patience": 4,
        "threshold": 1.0e-4,
        "minimum_learning_rate": {
            "backbone": 1.0e-9,
            "candidate_head": 5.0e-8,
            "score_head": 5.0e-7,
        },
    })

    # The candidate head keeps differentiable motion/safety/progress losses.
    # Every legacy score-label path is disabled so there is exactly one score
    # authority: the runtime-equivalent projected trajectory.
    config["safety_first"]["ranking_weight"] = 0.0
    config["preventive_safety"].update({
        "ranking_weight": 0.0,
        "label_weight": 0.0,
    })
    config["progress_safety"]["ranking_weight"] = 0.0
    config["progress_safety"]["label_weight"] = 0.0
    config["goal_progress_v2"].update({
        "ranking_weight": 0.0,
        "label_weight": 0.0,
    })
    config["feasibility_score"] = {
        **dict(config["feasibility_score"]),
        "enabled": False,
    }
    config["projected_score_v3"] = {
        "enabled": True,
        "max_speed_mps": 6.0,
        "max_acceleration_mps2": 6.0,
        "projection_min_scale": 0.15,
        "projection_steps": 18,
        "trajectory_samples": 81,
        "limit_tolerance": 0.001,
        "vehicle_radius_m": 0.30,
        "tracking_margin_m": 0.35,
        "reaction_time_s": 0.12,
        "braking_acceleration_mps2": 6.0,
        "clearance_sensor_tolerance_m": 0.08,
        "horizontal_fov_deg": 90.0,
        "vertical_fov_deg": 60.0,
        "visibility_margin_deg": 5.0,
        "minimum_path_length_m": 1.0,
        "minimum_goal_progress_m": 0.25,
        "preferred_goal_progress_m": 3.0,
        "clearance_saturation_m": 1.0,
        "unsafe_label_floor": 4.0,
        "progress_weight": 1.0,
        "detour_weight": 0.25,
        "smoothness_weight": 0.05,
        "regression_weight": 0.25,
        "listwise_weight": 1.0,
        "safety_ranking_weight": 2.0,
        "quality_ranking_weight": 1.0,
        "ranking_margin": 1.0,
        "quality_pair_gap": 0.10,
        "target_temperature": 0.35,
        "prediction_temperature": 1.0,
        "candidate_trajectory_samples": 30,
        "candidate_stopping_loss_weight": 0.25,
        "candidate_stopping_temperature_m": 0.25,
        "candidate_stopping_mean_weight": 0.25,
        "candidate_stopping_cvar_weight": 0.50,
        "candidate_stopping_coverage_weight": 1.0,
        "candidate_stopping_cvar_fraction": 1.0 / 3.0,
        "candidate_stopping_minimum_candidates": 3,
    }

    config["validation"] = {
        "contract_version": "route_a_v4_2_9_three_layer_v1",
        "primary_metric": "projected_contract_macro_loss",
        "direction": "minimize",
        "minimum_epoch": 12,
        "patience": 10,
        # Exactly two offline Gate decisions. Do not add map-specific or
        # overlapping booleans here; map breakdowns remain diagnostics.
        "offline_gate": {
            "candidate_availability_rate_min": 0.75,
            "conditional_selection_error_rate_max": 0.10,
        },
        "metric_weights": {
            "candidate_unavailability": 10.0,
            "conditional_selection_error": 20.0,
        },
        "closed_loop_contract": {
            "collision_events_max": 0,
            "goal_arrival_required": True,
            "path_efficiency_max": 1.75,
            "max_speed_mps": 6.0,
            "max_acceleration_mps2": 6.0,
        },
        "predefined_metrics": [
            "macro_map_type_total_static_loss",
            "per_map_type_total_static_loss",
            "projected_candidate_availability",
            "projected_safe_candidate_count",
            "projected_conditional_selection_error_rate",
            "projected_selected_stopping_reserve",
            "projected_selected_visible",
            "projected_selection_regret",
        ],
    }
    config["implementation_hotfixes"] = [
        *config.get("implementation_hotfixes", []),
        "route_a_v4_2_9_runtime_equivalent_projection_v1",
        "route_a_v4_2_9_physical_stopping_distance_v1",
        "route_a_v4_2_9_camera_visibility_and_flight_volume_v1",
        "route_a_v4_2_9_three_layer_gate_contract_v1",
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
        "dataset_reused": str(config["derived_dataset_root"]),
        "dataset_generation_required": False,
        "offline_gate_count": len(config["validation"]["offline_gate"]),
        "closed_loop_evidence": "external_rviz_report",
        "training_started": False,
    }, indent=2))


if __name__ == "__main__":
    main()
