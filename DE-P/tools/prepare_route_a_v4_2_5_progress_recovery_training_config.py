#!/usr/bin/env python3
"""Freeze V4.2.5 safe-progress training without mutating V4.2.4."""

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

PARENT = ROOT / "configs/route_a_v4_2_4_preventive_safety_training.yaml"
OUTPUT = ROOT / "configs/route_a_v4_2_5_progress_recovery_training.yaml"
INITIAL_CHECKPOINT = (
    ROOT / "runs/route_a_static_yopo_v4_2_4_preventive_safety"
    / "20260803T133235Z-20679/checkpoints/best_unqualified.pth"
)


def main():
    if not INITIAL_CHECKPOINT.is_file():
        raise FileNotFoundError(
            "V4.2.5 requires the retained V4.2.4 candidate-capacity checkpoint: "
            f"{INITIAL_CHECKPOINT}"
        )
    yaml = YAML()
    config = yaml.load(PARENT)
    config["contract_version"] = (
        "route_a_static_yopo_training_v4_2_5_safe_progress_recovery"
    )
    config["parent_v4_2_4_config_hash"] = sha256_file(PARENT)
    config["training_implementation_hash"] = training_implementation_hash()
    config["output_root"] = str(
        ROOT / "runs/route_a_static_yopo_v4_2_5_safe_progress_recovery"
    )
    config["model"]["initial_checkpoint"] = str(INITIAL_CHECKPOINT)
    config["model"]["initial_checkpoint_sha256"] = sha256_file(
        INITIAL_CHECKPOINT
    )
    config["optimizer"].update({
        "learning_rate": 1.0e-5,
        "backbone_learning_rate": 5.0e-7,
        "head_learning_rate": 1.0e-5,
        "weight_decay": 5.0e-4,
    })
    config["training"].update({"max_epochs": 45, "seed": 82502})
    config["observation"]["seed"] = 82502
    config["scheduler"].update({
        "factor": 0.5,
        "patience": 4,
        "threshold": 2.0e-4,
        "minimum_learning_rate": {"backbone": 1.0e-7, "head": 1.0e-6},
    })
    config["validation"].update({"minimum_epoch": 12, "patience": 12})

    # Safety remains lexicographically first.  The new objective only orders
    # candidates inside the intersection of hardware-safe and preventive-clear.
    config["safety_first"].update({
        "unsafe_label_priority": 10.0,
        "ranking_weight": 2.0,
    })
    config["preventive_safety"].update({
        "ranking_weight": 2.0,
        "label_weight": 3.0,
    })
    config["progress_safety"] = {
        "enabled": True,
        "minimum_progress_m": 1.0,
        "preferred_progress_m": 3.5,
        "softplus_temperature_m": 0.25,
        "loss_weight": 1.5,
        "ranking_weight": 2.0,
        "ranking_margin": 0.75,
        "label_weight": 3.0,
        "minimum_progress_candidates": 3,
    }
    gate = config["validation"]["selection_gate"]
    gate.update({
        "safe_progress_candidate_count_mean_min": 3.0,
        "preferred_progress_candidate_count_mean_min": 2.0,
        "insufficient_progress_selection_rate_max": 0.03,
    })
    config["validation"]["predefined_metrics"] = list(dict.fromkeys([
        *config["validation"]["predefined_metrics"],
        "progress_safety_loss",
        "progress_ranking_loss",
        "safe_progress_candidate_count",
        "preferred_progress_candidate_count",
        "insufficient_progress_selection",
    ]))
    config["implementation_hotfixes"] = [
        *config.get("implementation_hotfixes", []),
        "route_a_v4_2_5_safety_subordinate_progress_labels_v1",
        "route_a_v4_2_5_safe_progress_pairwise_ranking_v1",
        "route_a_v4_2_5_safe_progress_candidate_coverage_v1",
        "deadlock_recovery_v2_bounded_scan_v1",
        "deadlock_recovery_v2_observed_escape_certification_v1",
    ]

    if OUTPUT.exists():
        existing = yaml.load(OUTPUT)
        if dict(existing) != dict(config):
            raise FileExistsError(
                f"refusing to overwrite divergent V4.2.5 contract: {OUTPUT}"
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
        "progress_safety": dict(config["progress_safety"]),
        "training_started": False,
    }, indent=2))


if __name__ == "__main__":
    main()
