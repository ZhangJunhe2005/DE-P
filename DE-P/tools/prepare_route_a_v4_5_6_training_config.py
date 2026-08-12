#!/usr/bin/env python3
"""Prepare the focused V4.5.6 continuous-safety Score calibration."""

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


PARENT = ROOT / "configs/route_a_v4_5_5_dense_static_esdf_shakedown.yaml"
OUTPUT = ROOT / "configs/route_a_v4_5_6_continuous_score_safety_training.yaml"
V455_RUN_ROOT = ROOT / "runs/route_a_static_yopo_v4_5_5_dense_static_esdf_shakedown"


def latest_v455_checkpoint():
    candidates = []
    for run in sorted(path for path in V455_RUN_ROOT.glob("*") if path.is_dir()):
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
        raise RuntimeError("no completed V4.5.5 diagnostic best checkpoint found")
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
        # These terms supervise only the independent Score branch. They do
        # not reject candidates and never back-propagate into the proposal.
        "clearance_score_weight": 1.0,
        "clearance_score_softness_m": 0.08,
        "clearance_pairwise_weight": 1.0,
        "clearance_pairwise_margin": 0.50,
    }


def main():
    if not PARENT.is_file():
        raise FileNotFoundError(PARENT)
    run, checkpoint, completion = latest_v455_checkpoint()
    yaml = YAML()
    config = copy.deepcopy(yaml.load(PARENT))
    config["contract_version"] = (
        "route_a_static_yopo_training_v4_5_6_continuous_score_safety_v1"
    )
    config["parent_v4_5_5_config_hash"] = sha256_file(PARENT)
    config["parent_v4_5_5_run"] = str(run)
    config["parent_v4_5_5_best_epoch"] = int(completion["best_epoch"])
    config["training_implementation_hash"] = training_implementation_hash()
    config["experiment_role"] = "continuous_safety_score_calibration"
    config["output_root"] = str(
        ROOT / "runs/route_a_static_yopo_v4_5_6_continuous_score_safety"
    )
    config["model"].update({
        "head_variant": "independent",
        "initial_checkpoint": str(checkpoint),
        "initial_checkpoint_sha256": sha256_file(checkpoint),
        "strict_load": True,
        # Unified -> independent copies both towers exactly, so epoch zero
        # starts with bit-identical candidate and Score outputs.
        "allow_unified_to_split": True,
    })
    config["optimizer"] = {
        "name": "AdamW",
        "learning_rate": 1.0e-4,
        "backbone_learning_rate": 5.0e-7,
        "candidate_head_learning_rate": 5.0e-6,
        "score_head_learning_rate": 1.0e-4,
        "score_only_learning_rate": 1.0e-4,
        "weight_decay": 1.0e-5,
    }
    # The proposal is deliberately frozen for the whole bounded run. A fixed
    # Score LR is easier to audit than a scheduler that never observes a joint
    # stage; early stopping still selects the best validation epoch.
    config["scheduler"] = {
        "name": "ReduceLROnPlateau",
        "factor": 0.5,
        "patience": 4,
        "threshold": 1.0e-4,
        "minimum_learning_rate": {
            "backbone": 0.0,
            "candidate_head": 0.0,
            "score_head": 1.0e-5,
        },
    }
    config["training"].update({
        "max_epochs": 20,
        "seed": 84561,
        # Equal to max_epochs: backbone and proposal remain frozen.
        "score_only_warmup_epochs": 20,
    })
    for name in (
        "static_yopo_v4_5_2", "static_yopo_v4_5_3",
        "static_yopo_v4_5_4", "static_yopo_v4_5_5",
    ):
        config[name] = {"enabled": False}
    config["static_yopo_v4_5_6"] = objective_mapping()
    config["validation"].update({
        "contract_version": "route_a_v4_5_6_continuous_score_safety_v1",
        "primary_metric": "macro_map_type_total_static_loss",
        "direction": "minimize",
        "minimum_epoch": 5,
        # Equal to max_epochs: collect the complete 20-epoch calibration
        # curve. best.pth is still selected from held-out validation loss.
        "patience": 20,
        "predefined_metrics": [
            "macro_map_type_total_static_loss",
            "per_map_type_total_static_loss",
            "clearance_barrier_loss",
            "clearance_pairwise_ranking_loss",
            "score_top1_label_agreement",
            "score_oracle_regret",
            "collision_free_candidate_count",
            "hardware_feasible_candidate_count",
            "feasible_candidate_count",
            "collision_free_candidate_available",
            "physical_feasible_candidate_available",
            "oracle_collision_unsafe",
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
        "route_a_v4_5_6_initialize_from_v455_best_v1",
        "route_a_v4_5_6_exact_unified_to_independent_initialization_v1",
        "route_a_v4_5_6_freeze_backbone_and_candidate_for_full_run_v1",
        "route_a_v4_5_6_detached_continuous_physical_clearance_target_v1",
        "route_a_v4_5_6_continuous_clearance_pairwise_order_v1",
        "route_a_v4_5_6_no_candidate_rejection_no_qualification_gate_v1",
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
        "head_migration": "unified_to_independent_exact",
        "candidate_generator": "frozen_v4_5_5_best",
        "score_only_epochs_max": 20,
        "early_stopping_patience": 20,
        "dataset_reused": str(config["derived_dataset_root"]),
        "dataset_generation_required": False,
        "qualification_gate_count": 0,
        "training_started": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
