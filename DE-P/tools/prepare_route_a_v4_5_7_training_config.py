#!/usr/bin/env python3
"""Prepare the bounded V4.5.7 high-safety joint shakedown."""

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


PARENT = ROOT / "configs/route_a_v4_5_6_continuous_score_safety_training.yaml"
OUTPUT = ROOT / "configs/route_a_v4_5_7_high_safety_retimed_shakedown.yaml"
PARENT_RUN_ROOT = (
    ROOT / "runs/route_a_static_yopo_v4_5_6_continuous_score_safety"
)


def latest_parent_checkpoint():
    candidates = []
    for run in sorted(path for path in PARENT_RUN_ROOT.glob("*") if path.is_dir()):
        completion_path = run / "training_complete.json"
        checkpoint = run / "checkpoints/best.pth"
        if not completion_path.is_file() or not checkpoint.is_file():
            continue
        completion = json.loads(completion_path.read_text())
        if completion.get("status") != "DIAGNOSTIC_COMPLETE_NOT_PRODUCTION":
            continue
        candidates.append((run, checkpoint, completion))
    if not candidates:
        raise RuntimeError("no completed V4.5.6 diagnostic checkpoint found")
    return candidates[-1]


def main():
    if not PARENT.is_file():
        raise FileNotFoundError(PARENT)
    run, checkpoint, completion = latest_parent_checkpoint()
    yaml = YAML()
    config = copy.deepcopy(yaml.load(PARENT))
    config["contract_version"] = (
        "route_a_static_yopo_training_v4_5_7_high_safety_retimed_v1"
    )
    config["training_implementation_hash"] = training_implementation_hash()
    config["experiment_role"] = "high_safety_joint_retiming_shakedown"
    config["parent_v4_5_6_config_hash"] = sha256_file(PARENT)
    config["parent_v4_5_6_run"] = str(run)
    config["parent_v4_5_6_best_epoch"] = int(completion["best_epoch"])
    config["output_root"] = str(
        ROOT / "runs/route_a_static_yopo_v4_5_7_high_safety_retimed"
    )
    config["model"].update({
        "head_variant": "independent",
        "initial_checkpoint": str(checkpoint),
        "initial_checkpoint_sha256": sha256_file(checkpoint),
        "strict_load": True,
        "allow_unified_to_split": False,
    })
    config["optimizer"] = {
        "name": "AdamW",
        "learning_rate": 3.0e-5,
        "backbone_learning_rate": 2.0e-7,
        "candidate_head_learning_rate": 3.0e-6,
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
            "backbone": 5.0e-8,
            "candidate_head": 5.0e-7,
            "score_head": 5.0e-6,
        },
    }
    config["training"].update({
        "max_epochs": 5,
        "seed": 84571,
        "score_only_warmup_epochs": 0,
    })
    objective = config["static_yopo_v4_5_6"]
    objective.update({
        # At the V4.5.6 best epoch the mean weighted static loss was about
        # 1.96 versus 0.31 smoothness and 0.65 guidance.  A 5x multiplier
        # makes collision avoidance dominate their sum by roughly an order of
        # magnitude while retaining smooth gradients.
        "safety_weight": 5.0,
        "dangerous_segment_weight": 1.0,
        "clearance_score_weight": 5.0,
        "clearance_pairwise_weight": 2.0,
        # Hardware excess remains a weak preference during learning because
        # runtime performs certified time retiming instead of rejecting an
        # otherwise collision-free geometric route.
        "kinematic_speed_weight": 0.10,
        "kinematic_acceleration_weight": 0.10,
    })
    config["validation"].update({
        "contract_version": "route_a_v4_5_7_high_safety_retimed_v1",
        "minimum_epoch": 1,
        "patience": 5,
    })
    config["implementation_hotfixes"] = [
        "route_a_v4_5_7_initialize_from_v456_best_v1",
        "route_a_v4_5_7_joint_candidate_and_score_training_v1",
        "route_a_v4_5_7_static_safety_weight_5x_v1",
        "route_a_v4_5_7_dangerous_segment_weight_1x_v1",
        "route_a_v4_5_7_weak_kinematic_preference_for_runtime_retiming_v1",
        "route_a_v4_5_7_no_new_training_qualification_gate_v1",
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
        "safety_weight": objective["safety_weight"],
        "dangerous_segment_weight": objective["dangerous_segment_weight"],
        "kinematic_weights": {
            "speed": objective["kinematic_speed_weight"],
            "acceleration": objective["kinematic_acceleration_weight"],
        },
        "epochs": config["training"]["max_epochs"],
        "dataset_generation_required": False,
        "qualification_gate_count": 0,
        "training_started": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
