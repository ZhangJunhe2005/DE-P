#!/usr/bin/env python3
"""Create the immutable V4.2.1 regularized training contract."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.static_yopo_manifest_v1 import sha256_file
from tools.train_mixed_static_yopo_v1 import training_implementation_hash

PARENT = ROOT / "configs/route_a_v4_2_scene_relative_training.yaml"
OUTPUT = ROOT / "configs/route_a_v4_2_1_regularized_training.yaml"


def main():
    yaml = YAML()
    config = yaml.load(PARENT)
    config["contract_version"] = "route_a_static_yopo_training_v4_2_1"
    config["parent_v4_2_config_hash"] = sha256_file(PARENT)
    config["training_implementation_hash"] = training_implementation_hash()

    # Rebaseline from the clean legacy checkpoint. Do not continue from the
    # over-fitted V4.2 epoch-30 state or its optimizer moments.
    optimizer = config["optimizer"]
    optimizer["learning_rate"] = 2.0e-5
    optimizer["backbone_learning_rate"] = 2.0e-6
    optimizer["head_learning_rate"] = 2.0e-5
    optimizer["weight_decay"] = 5.0e-4

    scheduler = config["scheduler"]
    scheduler["patience"] = 5
    scheduler["factor"] = 0.5

    validation = config["validation"]
    validation["minimum_epoch"] = 12
    validation["patience"] = 8

    config["output_root"] = str(
        ROOT / "runs/route_a_static_yopo_v4_2_1_regularized"
    )
    config["resume"]["automatic"] = False
    config["implementation_hotfixes"] = [
        *config.get("implementation_hotfixes", []),
        "route_a_v4_2_1_score_head_regularization_v1",
        "route_a_v4_2_1_early_stop_contract_v1",
    ]

    if OUTPUT.exists():
        existing = yaml.load(OUTPUT)
        if dict(existing) != dict(config):
            raise FileExistsError(
                f"refusing to overwrite divergent frozen config: {OUTPUT}"
            )
    else:
        with OUTPUT.open("w", encoding="utf-8") as stream:
            yaml.dump(config, stream)

    print(json.dumps({
        "status": "PASS",
        "config": str(OUTPUT),
        "config_sha256": sha256_file(OUTPUT),
        "parent_v4_2_config_sha256": sha256_file(PARENT),
        "initial_checkpoint": config["model"]["initial_checkpoint"],
        "optimizer": dict(config["optimizer"]),
        "scheduler": dict(config["scheduler"]),
        "validation_stop": {
            "minimum_epoch": validation["minimum_epoch"],
            "patience": validation["patience"],
        },
        "training_started": False,
    }, indent=2))


if __name__ == "__main__":
    main()
