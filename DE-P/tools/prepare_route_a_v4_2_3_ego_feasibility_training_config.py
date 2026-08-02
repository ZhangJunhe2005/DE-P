#!/usr/bin/env python3
"""Freeze V4.2.3 EGO-inspired kinodynamic training without mutating V4.2.2."""

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

PARENT = ROOT / "configs/route_a_v4_2_2_safety_first_training.yaml"
OUTPUT = ROOT / "configs/route_a_v4_2_3_ego_feasibility_training.yaml"


def main():
    yaml = YAML()
    config = yaml.load(PARENT)
    config["contract_version"] = (
        "route_a_static_yopo_training_v4_2_3_ego_feasibility"
    )
    config["parent_v4_2_2_config_hash"] = sha256_file(PARENT)
    config["training_implementation_hash"] = training_implementation_hash()
    config["output_root"] = str(
        ROOT / "runs/route_a_static_yopo_v4_2_3_ego_feasibility"
    )

    # The outer coefficient gives the feasibility gradient comparable weight
    # to trajectory/guidance loss.  Internal terms are dimensionless and
    # normalized by the actual 6 m/s and 6 m/s^2 hardware limits.
    config["safety_first"]["kinematic_weight"] = 2.0
    config["kinodynamic_v2"] = {
        "enabled": True,
        "max_speed_mps": 6.0,
        "max_acceleration_mps2": 6.0,
        "soft_limit_ratio": 0.90,
        "normal_acceleration_soft_ratio": 0.75,
        "dense_norm_weight": 1.0,
        "axis_weight": 0.25,
        "normal_acceleration_weight": 0.50,
        "time_dilation_weight": 1.0,
        "candidate_mean_weight": 0.50,
        "candidate_cvar_weight": 0.50,
        "candidate_cvar_fraction": 1.0 / 3.0,
        "feasible_coverage_weight": 0.50,
        "minimum_feasible_candidates": 3,
        "label_weight": 1.0,
        "numerical_epsilon": 1.0e-6,
    }
    gate = config["validation"]["selection_gate"]
    gate["feasible_candidate_count_mean_min"] = 3.0
    gate["selected_time_dilation_mean_max"] = 1.02
    config["validation"]["predefined_metrics"] = list(dict.fromkeys([
        *config["validation"]["predefined_metrics"],
        "feasible_candidate_count", "candidate_time_dilation_mean",
        "selected_time_dilation", "selected_normal_acceleration",
    ]))
    config["implementation_hotfixes"] = [
        *config.get("implementation_hotfixes", []),
        "route_a_v4_2_3_dense_full_horizon_kinodynamic_gradient_v2",
        "route_a_v4_2_3_ego_time_dilation_supervision_v1",
        "route_a_v4_2_3_tangent_normal_anisotropic_acceleration_v1",
        "route_a_v4_2_3_all_unsafe_continuous_score_labels_v1",
    ]

    if OUTPUT.exists():
        existing = yaml.load(OUTPUT)
        if dict(existing) != dict(config):
            raise FileExistsError(
                f"refusing to overwrite divergent V4.2.3 contract: {OUTPUT}"
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
        "kinodynamic_v2": dict(config["kinodynamic_v2"]),
        "selection_gate": dict(gate),
        "training_started": False,
    }, indent=2))


if __name__ == "__main__":
    main()
