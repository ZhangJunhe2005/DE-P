#!/usr/bin/env python3
"""Freeze V4.3 single-cost training while reusing the V4.2 static dataset."""

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
OUTPUT = ROOT / "configs/route_a_v4_3_simple_training.yaml"
INITIAL_CHECKPOINT = ROOT / "saved/DEP_0/epoch10.pth"


def main():
    if not INITIAL_CHECKPOINT.is_file():
        raise FileNotFoundError(INITIAL_CHECKPOINT)
    yaml = YAML()
    config = yaml.load(PARENT)
    config["contract_version"] = "route_a_static_yopo_training_v4_3_single_cost_v1"
    config["parent_v4_2_3_config_hash"] = sha256_file(PARENT)
    config["training_implementation_hash"] = training_implementation_hash()
    config["output_root"] = str(ROOT / "runs/route_a_static_yopo_v4_3_simple")
    config["model"].update({
        "initial_checkpoint": str(INITIAL_CHECKPOINT),
        "initial_checkpoint_sha256": sha256_file(INITIAL_CHECKPOINT),
        "head_variant": "unified",
        "allow_unified_to_split": False,
    })
    config["optimizer"] = {
        "name": "AdamW",
        "learning_rate": 2.0e-5,
        "backbone_learning_rate": 2.0e-6,
        "head_learning_rate": 2.0e-5,
        "weight_decay": 2.0e-4,
    }
    config["scheduler"] = {
        "name": "ReduceLROnPlateau",
        "factor": 0.5,
        "patience": 5,
        "threshold": 1.0e-4,
        "minimum_learning_rate": {
            "backbone": 2.0e-7,
            "head": 2.0e-6,
        },
    }
    config["training"].update({
        "max_epochs": 50,
        "seed": 84301,
        "amp": False,
        "score_only_warmup_epochs": 0,
    })
    config["numerics"].update({
        "cnn_amp": False,
        "trajectory_loss_precision": "float32",
    })

    # Every legacy qualification/ranking/projection path is explicitly off.
    config["safety_first"] = {"enabled": False}
    config["kinodynamic_v2"] = {"enabled": False}
    config["preventive_safety"] = {"enabled": False}
    config["progress_safety"] = {"enabled": False}
    config["feasibility_score"] = {"enabled": False}
    config["goal_progress_v2"] = {"enabled": False}
    config["projected_score_v3"] = {"enabled": False}
    config["simple_yopo_v4_3"] = {
        "enabled": True,
        "derivative_samples": 81,
        "jerk_unit_weight": 10.0,
        "acceleration_unit_weight": 1.0,
        "safety_weight": 1.0,
        "guidance_weight": 0.15,
        "guidance_perpendicular_weight_max": 0.50,
        "guidance_goal_length_m": 10.0,
        "hardware_weight": 4.0,
        "score_weight": 1.0,
        "training_speed_mps": 6.0,
        "max_speed_mps": 6.0,
        "max_acceleration_mps2": 6.0,
    }
    config["validation"] = {
        "contract_version": "route_a_v4_3_single_loss_v1",
        "primary_metric": "macro_map_type_total_static_loss",
        "direction": "minimize",
        "minimum_epoch": 20,
        "patience": 12,
        "predefined_metrics": [
            "macro_map_type_total_static_loss",
            "per_map_type_total_static_loss",
            "score_top1_label_agreement",
            "selected_endpoint_distance",
            "selected_endpoint_speed",
            "selected_trajectory_max_speed",
            "selected_trajectory_max_acceleration",
            "hardware_unsafe_selection_rate",
            "feasible_candidate_count",
        ],
    }
    config["implementation_hotfixes"] = [
        "route_a_v4_3_single_total_cost_v1",
        "route_a_v4_3_full_trajectory_static_esdf_v1",
        "route_a_v4_3_all_candidate_gradient_v1",
        "route_a_v4_3_runtime_minimal_physical_filter_v1",
        "route_a_v4_3_dynamic_track_filter_preserved_v1",
    ]

    if OUTPUT.exists():
        existing = yaml.load(OUTPUT)
        if existing.get("contract_version") != config["contract_version"]:
            raise FileExistsError(f"refusing to overwrite foreign config: {OUTPUT}")
    with OUTPUT.open("w", encoding="utf-8") as stream:
        yaml.dump(config, stream)
    print(json.dumps({
        "status": "PASS",
        "config": str(OUTPUT),
        "config_sha256": sha256_file(OUTPUT),
        "initial_checkpoint": str(INITIAL_CHECKPOINT),
        "dataset_reused": str(config["derived_dataset_root"]),
        "dataset_generation_required": False,
        "qualification_gate_count": 0,
        "training_started": False,
    }, indent=2))


if __name__ == "__main__":
    main()
