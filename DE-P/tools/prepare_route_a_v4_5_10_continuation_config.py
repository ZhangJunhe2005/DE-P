#!/usr/bin/env python3
"""Prepare ten additional bounded V4.5.10 epochs from its best shakedown."""

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


PARENT = ROOT / "configs/route_a_v4_5_10_tail_aware_safety_shakedown.yaml"
OUTPUT = ROOT / "configs/route_a_v4_5_10_controlled_continuation.yaml"
PARENT_RUN_ROOT = ROOT / "runs/route_a_static_yopo_v4_5_10_tail_aware_safety"


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
        raise RuntimeError("no completed V4.5.10 shakedown checkpoint found")
    return candidates[-1]


def main():
    run, checkpoint, completion = latest_parent_checkpoint()
    yaml = YAML()
    config = copy.deepcopy(yaml.load(PARENT))
    config["contract_version"] = (
        "route_a_static_yopo_training_v4_5_10_controlled_continuation_v1"
    )
    config["training_implementation_hash"] = training_implementation_hash()
    config["experiment_role"] = "tail_aware_controlled_continuation"
    config["parent_v4_5_10_config_hash"] = sha256_file(PARENT)
    config["parent_v4_5_10_run"] = str(run)
    config["parent_v4_5_10_best_epoch"] = int(completion["best_epoch"])
    config["output_root"] = str(
        ROOT / "runs/route_a_static_yopo_v4_5_10_controlled_continuation"
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
        "learning_rate": 1.0e-5,
        "backbone_learning_rate": 1.0e-7,
        "candidate_head_learning_rate": 1.0e-6,
        "score_head_learning_rate": 1.0e-5,
        "score_only_learning_rate": 1.0e-5,
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
            "score_head": 2.5e-6,
        },
    }
    config["training"].update({
        # These are ten additional epochs in a new run; the parent five-epoch
        # optimizer history is intentionally not rewritten or impersonated.
        "max_epochs": 10,
        "seed": 84511,
        "score_only_warmup_epochs": 0,
    })
    config["validation"].update({
        "contract_version": "route_a_v4_5_10_tail_aware_safety_v1",
        "minimum_epoch": 9,
        "patience": 10,
    })
    config["implementation_hotfixes"] = list(
        config.get("implementation_hotfixes", [])
    ) + [
        "route_a_v4_5_10_continue_from_shakedown_best_v1",
        "route_a_v4_5_10_ten_additional_epochs_v1",
        "route_a_v4_5_10_lower_score_learning_rate_v1",
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
        "initial_checkpoint_sha256": sha256_file(checkpoint),
        "parent_best_epoch": int(completion["best_epoch"]),
        "additional_epochs": 10,
        "learning_rates": {
            "backbone": 1.0e-7,
            "candidate_head": 1.0e-6,
            "score_head": 1.0e-5,
        },
        "dataset_generation_required": False,
        "qualification_gate_count": 0,
        "training_started": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
