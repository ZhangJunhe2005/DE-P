#!/usr/bin/env python3
"""Freeze V4.4 static-parity training while reusing actor-free V4.2 data."""

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


PARENT = ROOT / "configs/route_a_v4_3_simple_training.yaml"
OUTPUT = ROOT / "configs/route_a_v4_4_static_parity_training.yaml"
INITIAL_CHECKPOINT = ROOT / "saved/DEP_0/epoch10.pth"


def main():
    if not PARENT.is_file():
        raise FileNotFoundError(PARENT)
    if not INITIAL_CHECKPOINT.is_file():
        raise FileNotFoundError(INITIAL_CHECKPOINT)
    yaml = YAML()
    config = yaml.load(PARENT)
    config["contract_version"] = (
        "route_a_static_yopo_training_v4_4_static_parity_v1"
    )
    config["parent_v4_3_config_hash"] = sha256_file(PARENT)
    config["training_implementation_hash"] = training_implementation_hash()
    config["output_root"] = str(
        ROOT / "runs/route_a_static_yopo_v4_4_static_parity"
    )
    config["model"].update({
        "initial_checkpoint": str(INITIAL_CHECKPOINT),
        "initial_checkpoint_sha256": sha256_file(INITIAL_CHECKPOINT),
        "head_variant": "unified",
        "allow_unified_to_split": False,
    })
    config["observation"] = {
        "contract": "route_a_original_yopo_state_v4_4",
        "seed": 82501,
        "velocity_distribution": "original_yopo_forward_biased",
        "acceleration_distribution": "original_yopo_zero_centered_gaussian",
        "goal_distribution": "original_yopo_forward_10m_with_near_goal_tail",
    }
    config["optimizer"] = {
        "name": "AdamW",
        "learning_rate": 1.0e-4,
        "backbone_learning_rate": 1.0e-5,
        "head_learning_rate": 1.0e-4,
        "weight_decay": 1.0e-5,
    }
    config["scheduler"] = {
        "name": "ReduceLROnPlateau",
        "factor": 0.5,
        "patience": 6,
        "threshold": 1.0e-4,
        "minimum_learning_rate": {
            "backbone": 5.0e-7,
            "head": 5.0e-6,
        },
    }
    config["training"].update({
        "max_epochs": 50,
        "seed": 84401,
        "amp": False,
        "score_only_warmup_epochs": 0,
    })
    config["numerics"].update({
        "cnn_amp": False,
        "trajectory_loss_precision": "float32",
    })

    for name in (
        "safety_first", "kinodynamic_v2", "preventive_safety",
        "progress_safety", "feasibility_score", "goal_progress_v2",
        "projected_score_v3", "simple_yopo_v4_3",
    ):
        config[name] = {"enabled": False}
    config["static_yopo_v4_4"] = {
        "enabled": True,
        "derivative_samples": 81,
        "jerk_unit_weight": 10.0,
        "acceleration_unit_weight": 1.0,
        "safety_weight": 1.0,
        "guidance_weight": 0.15,
        "guidance_perpendicular_weight": 0.50,
        "score_regression_weight": 1.0,
        "relative_order_weight": 0.25,
        "relative_order_temperature": 0.75,
        "training_speed_mps": 6.0,
        "max_speed_mps": 6.0,
        "max_acceleration_mps2": 6.0,
        "vehicle_radius_m": 0.30,
        "clear_distance_m": 1.20,
    }
    config["validation"] = {
        "contract_version": "route_a_v4_4_static_parity_v1",
        "primary_metric": "macro_map_type_total_static_loss",
        "direction": "minimize",
        # Original YOPO always ran 50 epochs.  V4.4 keeps all 50 so a short
        # plateau cannot silently terminate the parity experiment.
        "minimum_epoch": 49,
        "patience": 50,
        "predefined_metrics": [
            "macro_map_type_total_static_loss",
            "per_map_type_total_static_loss",
            "score_top1_label_agreement",
            "score_oracle_regret",
            "selected_endpoint_distance",
            "selected_endpoint_speed",
            "selected_clearance",
            "hardware_unsafe_selection_rate",
            "feasible_candidate_count",
        ],
    }
    config["implementation_hotfixes"] = [
        "route_a_v4_4_original_state_distribution_v1",
        "route_a_v4_4_original_analytic_smoothness_v1",
        "route_a_v4_4_original_full_static_esdf_mean_v1",
        "route_a_v4_4_single_relative_score_calibration_v1",
        "route_a_v4_4_no_offline_qualification_gate_v1",
        "route_a_v4_4_dynamic_track_filter_preserved_v1",
        "route_a_v4_4_rare_scan_only_recovery_v1",
    ]

    if OUTPUT.exists():
        existing = yaml.load(OUTPUT)
        if existing.get("contract_version") != config["contract_version"]:
            raise FileExistsError(
                f"refusing to overwrite foreign config: {OUTPUT}"
            )
    with OUTPUT.open("w", encoding="utf-8") as stream:
        yaml.dump(config, stream)
    print(json.dumps({
        "status": "PASS",
        "config": str(OUTPUT),
        "config_sha256": sha256_file(OUTPUT),
        "initial_checkpoint": str(INITIAL_CHECKPOINT),
        "dataset_reused": str(config["derived_dataset_root"]),
        "dataset_actor_input_used": False,
        "dataset_generation_required": False,
        "qualification_gate_count": 0,
        "training_started": False,
    }, indent=2))


if __name__ == "__main__":
    main()
