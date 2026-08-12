#!/usr/bin/env python3
"""Prepare V4.5.1 mixed shakedown and deferred full-training configs."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.static_yopo_manifest_v1 import sha256_file
from tools.train_mixed_static_yopo_v1 import training_implementation_hash


PARENT = ROOT / "configs/route_a_v4_5_bounded_danger_training.yaml"
FULL = ROOT / "configs/route_a_v4_5_1_relative_kinematic_training.yaml"
SHAKEDOWN = ROOT / "configs/route_a_v4_5_1_mixed_shakedown.yaml"


def objective_mapping():
    return {
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
        "relative_label_min_scale": 1.0e-3,
        "training_speed_mps": 6.0,
        "max_speed_mps": 6.0,
        "max_acceleration_mps2": 6.0,
        # One continuous objective, not a boolean feasibility Gate.
        "kinematic_speed_weight": 1.0,
        "kinematic_acceleration_weight": 1.0,
        "kinematic_softness": 0.05,
        "vehicle_radius_m": 0.30,
        "clear_distance_m": 1.20,
        "dangerous_segment_weight": 0.15,
        "dangerous_segment_window": 5,
        "dangerous_segment_focus": 8.0,
        "large_vertical_displacement_m": 1.0,
    }


def common(parent):
    config = copy.deepcopy(parent)
    config["contract_version"] = (
        "route_a_static_yopo_training_v4_5_1_relative_kinematic_v1"
    )
    config["parent_v4_5_config_hash"] = sha256_file(PARENT)
    config["training_implementation_hash"] = training_implementation_hash()
    config["output_root"] = str(
        ROOT / "runs/route_a_static_yopo_v4_5_1_relative_kinematic"
    )
    config["static_yopo_v4_5"] = {"enabled": False}
    config["static_yopo_v4_5_1"] = objective_mapping()
    config["loader"].pop("map_type_filter", None)
    config["validation"].update({
        "contract_version": "route_a_v4_5_1_relative_kinematic_v1",
        "primary_metric": "macro_map_type_total_static_loss",
        "direction": "minimize",
        "minimum_epoch": 49,
        "patience": 50,
        "predefined_metrics": [
            "macro_map_type_total_static_loss",
            "per_map_type_total_static_loss",
            "score_top1_label_agreement",
            "score_oracle_regret",
            "score_label_scale",
            "kinematic_loss",
            "hardware_unsafe_selection",
            "dangerous_segment_loss",
            "selected_clearance",
            "selected_endpoint_distance",
            "selected_absolute_vertical_displacement",
            "oracle_absolute_vertical_displacement",
            "selected_large_vertical_maneuver",
            "oracle_large_vertical_maneuver",
        ],
    })
    config["implementation_hotfixes"] = [
        "route_a_v4_5_1_within_sample_relative_score_label_v1",
        "route_a_v4_5_1_continuous_kinematic_excess_gradient_v1",
        "route_a_v4_5_1_absolute_vertical_activity_diagnostics_v1",
        "route_a_v4_5_1_diagnostic_nonproduction_state_semantics_v1",
        "route_a_v4_5_1_no_new_qualification_gate_v1",
    ]
    return config


def write_config(path, config, yaml):
    if path.exists():
        existing = yaml.load(path)
        if existing.get("contract_version") != config["contract_version"]:
            raise FileExistsError(f"refusing to overwrite foreign config: {path}")
    with path.open("w", encoding="utf-8") as stream:
        yaml.dump(config, stream)


def main():
    if not PARENT.is_file():
        raise FileNotFoundError(PARENT)
    yaml = YAML()
    parent = yaml.load(PARENT)
    full = common(parent)
    write_config(FULL, full, yaml)

    shakedown = copy.deepcopy(full)
    shakedown["contract_version"] = (
        "route_a_static_yopo_training_v4_5_1_mixed_shakedown_v1"
    )
    shakedown["experiment_role"] = "mixed_five_epoch_shakedown"
    shakedown["training"].update({"max_epochs": 5, "seed": 84511})
    shakedown["validation"].update({"minimum_epoch": 4, "patience": 5})
    shakedown["output_root"] = str(
        ROOT / "runs/route_a_static_yopo_v4_5_1_mixed_shakedown"
    )
    write_config(SHAKEDOWN, shakedown, yaml)

    print(json.dumps({
        "status": "PASS",
        "full_config": str(FULL),
        "full_config_sha256": sha256_file(FULL),
        "shakedown_config": str(SHAKEDOWN),
        "shakedown_config_sha256": sha256_file(SHAKEDOWN),
        "dataset_reused": str(full["derived_dataset_root"]),
        "dataset_generation_required": False,
        "qualification_gate_count": 0,
        "training_started": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
