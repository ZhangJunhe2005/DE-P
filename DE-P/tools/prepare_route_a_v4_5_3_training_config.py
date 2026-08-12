#!/usr/bin/env python3
"""Prepare the bounded V4.5.3 two-stage mixed-map shakedown."""

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
OUTPUT = ROOT / "configs/route_a_v4_5_3_two_stage_shakedown.yaml"
V452_RUN_ROOT = ROOT / "runs/route_a_static_yopo_v4_5_2_mixed_shakedown"


def latest_v452_checkpoint():
    candidates = []
    for run in sorted(path for path in V452_RUN_ROOT.glob("*") if path.is_dir()):
        completion_path = run / "training_complete.json"
        checkpoint = run / "checkpoints/best.pth"
        metrics = run / "metrics.jsonl"
        if not completion_path.is_file() or not checkpoint.is_file() \
                or not metrics.is_file():
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
        "dangerous_segment_window": 5,
        "dangerous_segment_focus": 8.0,
        "large_vertical_displacement_m": 1.0,
        # Continuous and unbounded; it is not a Boolean inflated boundary.
        "clearance_barrier_weight": 1.0,
        "clearance_barrier_softness_m": 0.05,
    }


def main():
    if not PARENT.is_file():
        raise FileNotFoundError(PARENT)
    run, checkpoint, completion = latest_v452_checkpoint()
    yaml = YAML()
    config = copy.deepcopy(yaml.load(PARENT))
    config["contract_version"] = (
        "route_a_static_yopo_training_v4_5_3_two_stage_shakedown_v1"
    )
    config["parent_v4_5_2_config_hash"] = sha256_file(PARENT)
    config["parent_v4_5_2_run"] = str(run)
    config["parent_v4_5_2_best_epoch"] = int(completion["best_epoch"])
    config["training_implementation_hash"] = training_implementation_hash()
    config["experiment_role"] = "mixed_two_stage_eight_epoch_shakedown"
    config["output_root"] = str(
        ROOT / "runs/route_a_static_yopo_v4_5_3_two_stage_shakedown"
    )
    config["model"].update({
        "head_variant": "split",
        "initial_checkpoint": str(checkpoint),
        "initial_checkpoint_sha256": sha256_file(checkpoint),
        "strict_load": True,
        "allow_unified_to_split": True,
    })
    config["optimizer"] = {
        "name": "AdamW",
        "learning_rate": 2.0e-5,
        "backbone_learning_rate": 5.0e-7,
        "candidate_head_learning_rate": 5.0e-6,
        "score_head_learning_rate": 2.0e-5,
        "score_only_learning_rate": 1.0e-4,
        "weight_decay": 1.0e-5,
    }
    config["scheduler"] = {
        "name": "ReduceLROnPlateau",
        "factor": 0.5,
        "patience": 3,
        "threshold": 1.0e-4,
        "minimum_learning_rate": {
            "backbone": 1.0e-7,
            "candidate_head": 1.0e-6,
            "score_head": 5.0e-6,
        },
    }
    config["training"].update({
        "max_epochs": 8,
        "seed": 84531,
        "score_only_warmup_epochs": 3,
    })
    config["static_yopo_v4_5_2"] = {"enabled": False}
    config["static_yopo_v4_5_3"] = objective_mapping()
    config["validation"].update({
        "contract_version": (
            "route_a_v4_5_3_continuous_clearance_two_stage_v1"
        ),
        "primary_metric": "macro_map_type_total_static_loss",
        "direction": "minimize",
        "minimum_epoch": 7,
        "patience": 8,
        "predefined_metrics": [
            "macro_map_type_total_static_loss",
            "per_map_type_total_static_loss",
            "clearance_barrier_loss",
            "score_top1_label_agreement",
            "score_oracle_regret",
            "speed_unsafe_selection",
            "acceleration_unsafe_selection",
            "collision_free_candidate_count",
            "hardware_feasible_candidate_count",
            "feasible_candidate_count",
            "collision_free_candidate_available",
            "physical_feasible_candidate_available",
            "conditional_collision_selection_error",
            "conditional_physical_selection_error",
            "oracle_collision_unsafe",
            "oracle_hardware_unsafe",
            "oracle_physical_unsafe",
            "selected_endpoint_distance",
            "selected_clearance",
        ],
    })
    config["implementation_hotfixes"] = [
        "route_a_v4_5_3_continuous_physical_radius_clearance_barrier_v1",
        "route_a_v4_5_3_exact_unified_to_split_initialization_v1",
        "route_a_v4_5_3_three_epoch_score_only_then_five_epoch_joint_v1",
        "route_a_v4_5_3_no_new_qualification_gate_v1",
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
        "head_migration": "unified_to_split_exact",
        "score_only_epochs": 3,
        "joint_finetune_epochs": 5,
        "dataset_reused": str(config["derived_dataset_root"]),
        "dataset_generation_required": False,
        "qualification_gate_count": 0,
        "training_started": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
