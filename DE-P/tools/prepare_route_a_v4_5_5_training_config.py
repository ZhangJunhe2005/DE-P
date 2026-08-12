#!/usr/bin/env python3
"""Prepare the bounded V4.5.5 dense-static-ESDF shakedown."""

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


PARENT = ROOT / "configs/route_a_v4_5_2_mixed_shakedown.yaml"
OUTPUT = ROOT / "configs/route_a_v4_5_5_dense_static_esdf_shakedown.yaml"
V452_RUN_ROOT = ROOT / "runs/route_a_static_yopo_v4_5_2_mixed_shakedown"


def latest_v452_checkpoint():
    candidates = []
    for run in sorted(path for path in V452_RUN_ROOT.glob("*") if path.is_dir()):
        completion_path = run / "training_complete.json"
        checkpoint = run / "checkpoints/best.pth"
        if not completion_path.is_file() or not checkpoint.is_file():
            continue
        completion = json.loads(completion_path.read_text())
        if completion.get("status") != "DIAGNOSTIC_COMPLETE_NOT_PRODUCTION":
            continue
        candidates.append((run, checkpoint, completion))
    if not candidates:
        raise RuntimeError("no complete V4.5.2 diagnostic best checkpoint found")
    return candidates[-1]


def objective_mapping():
    return {
        "enabled": True,
        "derivative_samples": 81,
        "static_safety_samples": 81,
        "jerk_unit_weight": 10.0,
        "acceleration_unit_weight": 1.0,
        "safety_weight": 1.0,
        "guidance_weight": 0.15,
        "guidance_perpendicular_weight": 0.50,
        "score_regression_weight": 1.0,
        "relative_order_weight": 1.0,
        "relative_order_temperature": 0.75,
        "relative_label_min_scale": 1.0e-3,
        "training_speed_mps": 6.0,
        "max_speed_mps": 6.0,
        "max_acceleration_mps2": 6.0,
        "kinematic_speed_weight": 1.0,
        "kinematic_acceleration_weight": 1.0,
        "kinematic_softness": 0.05,
        "vehicle_radius_m": 0.30,
        "clear_distance_m": 1.20,
        "dangerous_segment_weight": 0.15,
        "dangerous_segment_window": 14,
        "dangerous_segment_focus": 8.0,
        "large_vertical_displacement_m": 1.0,
    }


def main():
    if not PARENT.is_file():
        raise FileNotFoundError(PARENT)
    run, checkpoint, completion = latest_v452_checkpoint()
    yaml = YAML()
    config = copy.deepcopy(yaml.load(PARENT))
    config["contract_version"] = (
        "route_a_static_yopo_training_v4_5_5_dense_static_esdf_shakedown_v1"
    )
    config["parent_v4_5_2_config_hash"] = sha256_file(PARENT)
    config["parent_v4_5_2_run"] = str(run)
    config["parent_v4_5_2_best_epoch"] = int(completion["best_epoch"])
    config["training_implementation_hash"] = training_implementation_hash()
    config["experiment_role"] = "dense_static_esdf_five_epoch_shakedown"
    config["output_root"] = str(
        ROOT / "runs/route_a_static_yopo_v4_5_5_dense_static_esdf_shakedown"
    )
    config["model"].update({
        "head_variant": "unified",
        "initial_checkpoint": str(checkpoint),
        "initial_checkpoint_sha256": sha256_file(checkpoint),
        "strict_load": True,
        "allow_unified_to_split": False,
    })
    config["optimizer"] = {
        "name": "AdamW",
        "learning_rate": 5.0e-5,
        "backbone_learning_rate": 5.0e-6,
        "head_learning_rate": 5.0e-5,
        "weight_decay": 1.0e-5,
    }
    config["scheduler"] = {
        "name": "ReduceLROnPlateau",
        "factor": 0.5,
        "patience": 3,
        "threshold": 1.0e-4,
        "minimum_learning_rate": {
            "backbone": 5.0e-7,
            "head": 5.0e-6,
        },
    }
    config["training"].update({
        "max_epochs": 5,
        "seed": 84551,
        "score_only_warmup_epochs": 0,
    })
    config["static_yopo_v4_5_2"] = {"enabled": False}
    config["static_yopo_v4_5_3"] = {"enabled": False}
    config["static_yopo_v4_5_4"] = {"enabled": False}
    config["static_yopo_v4_5_5"] = objective_mapping()
    config["validation"].update({
        "contract_version": "route_a_v4_5_5_dense_static_esdf_v1",
        "primary_metric": "macro_map_type_total_static_loss",
        "direction": "minimize",
        "minimum_epoch": 4,
        "patience": 5,
        "predefined_metrics": [
            "macro_map_type_total_static_loss",
            "per_map_type_total_static_loss",
            "score_top1_label_agreement",
            "score_oracle_regret",
            "collision_free_candidate_count",
            "hardware_feasible_candidate_count",
            "feasible_candidate_count",
            "collision_free_candidate_available",
            "physical_feasible_candidate_available",
            "conditional_collision_selection_error",
            "conditional_physical_selection_error",
            "hardware_unsafe_selection",
            "unsafe_selection",
            "selected_endpoint_distance",
            "selected_clearance",
            "coarse_false_safe_candidate_count",
            "dense_clearance_drop",
        ],
    })
    config["implementation_hotfixes"] = [
        "route_a_v4_5_5_restore_unified_head_from_v452_v1",
        "route_a_v4_5_5_dense_81_point_static_esdf_v1",
        "route_a_v4_5_5_preserve_v45_danger_time_window_v1",
        "route_a_v4_5_5_candidate_and_score_same_continuous_total_v1",
        "route_a_v4_5_5_no_clearance_barrier_no_gate_v1",
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
        "initial_checkpoint": str(checkpoint),
        "initial_checkpoint_sha256": sha256_file(checkpoint),
        "initial_checkpoint_best_epoch": int(completion["best_epoch"]),
        "head_variant": "unified",
        "candidate_generator": "trainable_from_v4_5_2",
        "static_esdf_samples": 81,
        "qualification_gate_count": 0,
        "dataset_generation_required": False,
        "training_started": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

