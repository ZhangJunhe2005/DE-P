#!/usr/bin/env python3
"""Create the V4.2.2 safety-first contract without mutating V4.2.1."""

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

PARENT = ROOT / "configs/route_a_v4_2_1_regularized_training.yaml"
OUTPUT = ROOT / "configs/route_a_v4_2_2_safety_first_training.yaml"


def main():
    yaml = YAML()
    config = yaml.load(PARENT)
    config["contract_version"] = "route_a_static_yopo_training_v4_2_2_safety_first"
    config["parent_v4_2_1_config_hash"] = sha256_file(PARENT)
    config["training_implementation_hash"] = training_implementation_hash()
    config["safety_first"] = {
        "enabled": True,
        "required_clearance_m": 0.65,
        "unsafe_label_priority": 5.0,
        "ranking_margin": 1.0,
        "ranking_weight": 0.5,
        "safety_cvar_fraction": 1.0 / 3.0,
        "safety_cvar_weight": 0.5,
        "max_speed_mps": 6.0,
        "max_acceleration_mps2": 6.0,
        "kinematic_weight": 0.5,
    }
    # Do not inherit the over-fitted model or optimizer moments.  V4.2.2 is a
    # clean rebaseline from the original legacy checkpoint declared by parent.
    config["output_root"] = str(
        ROOT / "runs/route_a_static_yopo_v4_2_2_safety_first"
    )
    config["validation"]["selection_gate"]["unsafe_selection_rate_max"] = 0.05
    config["validation"]["selection_gate"][
        "hardware_unsafe_selection_rate_max"
    ] = 0.05
    config["validation"]["predefined_metrics"] = list(dict.fromkeys([
        *config["validation"]["predefined_metrics"],
        "unsafe_selection_rate", "hardware_unsafe_selection_rate",
        "selected_trajectory_max_speed", "selected_trajectory_max_acceleration",
        "ranking_loss", "safety_cvar_loss", "kinematic_loss",
    ]))
    config["implementation_hotfixes"] = [
        *config.get("implementation_hotfixes", []),
        "runtime_trajectory_safety_v1",
        "route_a_v4_2_2_safety_first_score_ranking_v1",
        "route_a_v4_2_2_static_safety_cvar_v1",
        "route_a_v4_2_2_anti_hover_contract_preserved_v1",
    ]

    if OUTPUT.exists():
        existing = yaml.load(OUTPUT)
        if dict(existing) != dict(config):
            raise FileExistsError(
                f"refusing to overwrite divergent V4.2.2 contract: {OUTPUT}"
            )
    else:
        with OUTPUT.open("w", encoding="utf-8") as stream:
            yaml.dump(config, stream)
    print(json.dumps({
        "status": "PASS",
        "config": str(OUTPUT),
        "config_sha256": sha256_file(OUTPUT),
        "parent_sha256": sha256_file(PARENT),
        "training_implementation_hash": config["training_implementation_hash"],
        "safety_first": dict(config["safety_first"]),
        "anti_hover_gate": dict(config["validation"]["selection_gate"]),
        "training_started": False,
    }, indent=2))


if __name__ == "__main__":
    main()
