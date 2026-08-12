#!/usr/bin/env python3
"""Prepare the bounded V4.5.10 tail-aware five-epoch comparison."""

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


PARENT = ROOT / "configs/route_a_v4_5_9_time_mean_safety_shakedown.yaml"
OUTPUT = ROOT / "configs/route_a_v4_5_10_tail_aware_safety_shakedown.yaml"
PARENT_RUN_ROOT = ROOT / "runs/route_a_static_yopo_v4_5_9_time_mean_safety"


def latest_parent_checkpoint():
    candidates = []
    for run in sorted(path for path in PARENT_RUN_ROOT.glob("*") if path.is_dir()):
        completion_path = run / "training_complete.json"
        checkpoint = run / "checkpoints/best.pth"
        metrics = run / "metrics.jsonl"
        if not (completion_path.is_file() and checkpoint.is_file()
                and metrics.is_file()):
            continue
        completion = json.loads(completion_path.read_text())
        if completion.get("status") != "DIAGNOSTIC_COMPLETE_NOT_PRODUCTION":
            continue
        candidates.append((run, checkpoint, completion))
    if not candidates:
        raise RuntimeError("no completed V4.5.9 diagnostic checkpoint found")
    return candidates[-1]


def main():
    run, checkpoint, completion = latest_parent_checkpoint()
    yaml = YAML()
    config = copy.deepcopy(yaml.load(PARENT))
    config["contract_version"] = (
        "route_a_static_yopo_training_v4_5_10_tail_aware_safety_v1"
    )
    config["training_implementation_hash"] = training_implementation_hash()
    config["experiment_role"] = "tail_aware_safety_shakedown"
    config["parent_v4_5_9_config_hash"] = sha256_file(PARENT)
    config["parent_v4_5_9_run"] = str(run)
    config["parent_v4_5_9_best_epoch"] = int(completion["best_epoch"])
    config["output_root"] = str(
        ROOT / "runs/route_a_static_yopo_v4_5_10_tail_aware_safety"
    )
    config["model"].update({
        "head_variant": "independent",
        "initial_checkpoint": str(checkpoint),
        "initial_checkpoint_sha256": sha256_file(checkpoint),
        "strict_load": True,
        "allow_unified_to_split": False,
    })
    # Let the score branch adapt normally while limiting candidate drift.  The
    # candidate path still receives the continuous safety gradient; it is not
    # frozen and there is no extra feasibility Gate.
    config["optimizer"] = {
        "name": "AdamW",
        "learning_rate": 3.0e-5,
        "backbone_learning_rate": 1.0e-7,
        "candidate_head_learning_rate": 1.0e-6,
        "score_head_learning_rate": 3.0e-5,
        "score_only_learning_rate": 3.0e-5,
        "weight_decay": 1.0e-5,
    }
    config["scheduler"] = {
        "name": "ReduceLROnPlateau",
        "factor": 0.5,
        "patience": 2,
        "threshold": 1.0e-4,
        "minimum_learning_rate": {
            "backbone": 2.5e-8,
            "candidate_head": 2.5e-7,
            "score_head": 5.0e-6,
        },
    }
    config["training"].update({
        "max_epochs": 5,
        "seed": 84510,
        "score_only_warmup_epochs": 0,
    })
    config["static_yopo_v4_5_9"]["enabled"] = False
    config["static_yopo_v4_5_10"] = {
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
        "kinematic_speed_weight": 0.10,
        "kinematic_acceleration_weight": 0.10,
        "kinematic_softness": 0.05,
        "vehicle_radius_m": 0.30,
        "clear_distance_m": 0.50,
        "dangerous_segment_weight": 0.0,
        "dangerous_segment_window": 14,
        "dangerous_segment_focus": 8.0,
        "large_vertical_displacement_m": 1.0,
        "clearance_score_weight": 0.0,
        "clearance_score_softness_m": 0.08,
        "clearance_pairwise_weight": 0.0,
        "clearance_pairwise_margin": 0.0,
        "far_clearance_cost": 0.001,
        "far_decay_m": 0.10,
        "radius_boundary_cost": 0.10,
        "collision_surrogate_cost": 3.0,
        "penetration_scale_m": 0.05,
        "maximum_training_cost": 6.0,
        "time_mean_weight": 0.90,
        "worst_sample_weight": 0.10,
        "worst_sample_count": 5,
    }
    config["validation"].update({
        "contract_version": "route_a_v4_5_10_tail_aware_safety_v1",
        "minimum_epoch": 1,
        "patience": 5,
    })
    config["implementation_hotfixes"] = [
        "route_a_v4_5_10_initialize_from_v459_best_v1",
        "route_a_v4_5_10_single_90_mean_10_worst_five_safety_v1",
        "route_a_v4_5_10_limited_candidate_drift_v1",
        "route_a_v4_5_10_runtime_0_35_representation_floor_v1",
        "route_a_v4_5_10_keep_next_feasible_and_retiming_v1",
        "route_a_v4_5_10_no_new_qualification_gate_v1",
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
        "initial_checkpoint_best_epoch": int(completion["best_epoch"]),
        "initial_checkpoint_sha256": sha256_file(checkpoint),
        "aggregation": {
            "time_mean_weight": 0.90,
            "worst_sample_weight": 0.10,
            "worst_sample_count": 5,
            "samples": 81,
        },
        "runtime_collision_floor_m": 0.35,
        "epochs": 5,
        "dataset_generation_required": False,
        "qualification_gate_count": 0,
        "training_started": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
