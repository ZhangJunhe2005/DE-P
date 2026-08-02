#!/usr/bin/env python3
"""Freeze the corrected V4.1 local-goal training contract."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.static_yopo_manifest_v1 import sha256_file
from policy.static_yopo_objective_contract_v1 import (
    LOCAL_GOAL_OBJECTIVE_CONTRACT_V1_HASH,
)
from tools.train_mixed_static_yopo_v1 import training_implementation_hash

SOURCE = ROOT / "configs/route_a_v4_static_yopo_training.yaml"
OUTPUT = ROOT / "configs/route_a_v4_1_local_goal_training.yaml"


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite frozen config: {OUTPUT}")
    yaml = YAML()
    config = yaml.load(SOURCE)
    config["contract_version"] = "route_a_static_yopo_training_v4_1"
    config["parent_v4_config_hash"] = sha256_file(SOURCE)
    config["training_implementation_hash"] = training_implementation_hash()
    config["objective_contract_hash"] = (
        LOCAL_GOAL_OBJECTIVE_CONTRACT_V1_HASH
    )
    config["objective"] = {
        "route_goal_range_m": [10.0, 40.0],
        "local_goal_horizon_m": 10.0,
        "score_label_goal": "local_horizon_projection",
    }
    config["optimizer"]["learning_rate"] = 5.0e-5
    config["optimizer"]["backbone_learning_rate"] = 5.0e-6
    config["optimizer"]["head_learning_rate"] = 5.0e-5
    config["optimizer"]["weight_decay"] = 1.0e-4
    config["scheduler"]["patience"] = 8
    config["scheduler"]["threshold"] = 5.0e-4
    config["training"]["gradient_clip_norm"] = 5.0
    config["validation"]["minimum_epoch"] = 30
    config["validation"]["patience"] = 12
    predefined = list(config["validation"]["predefined_metrics"])
    for name in ("route_goal_distance", "objective_goal_distance"):
        if name not in predefined:
            predefined.append(name)
    config["validation"]["predefined_metrics"] = predefined
    config["output_root"] = str(
        ROOT / "runs/route_a_static_yopo_v4_1_local_goal"
    )
    config["resume"]["automatic"] = False
    config["implementation_hotfixes"] = [
        *config.get("implementation_hotfixes", []),
        "route_a_v4_1_local_goal_observability_repair_v1",
        "route_a_v4_1_gradient_clip_and_scheduler_rebaseline_v1",
    ]
    with open(OUTPUT, "w", encoding="utf-8") as stream:
        yaml.dump(config, stream)
    print(json.dumps({
        "status": "PASS",
        "config": str(OUTPUT),
        "config_hash": sha256_file(OUTPUT),
        "objective_contract_hash": LOCAL_GOAL_OBJECTIVE_CONTRACT_V1_HASH,
        "initial_checkpoint": config["model"]["initial_checkpoint"],
        "output_root": config["output_root"],
        "training_started": False,
    }, indent=2))


if __name__ == "__main__":
    main()
