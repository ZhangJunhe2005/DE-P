#!/usr/bin/env python3
"""Prepare V4.5 full training and five bounded map-type probes."""

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


PARENT = ROOT / "configs/route_a_v4_4_static_parity_training.yaml"
OUTPUT = ROOT / "configs/route_a_v4_5_bounded_danger_training.yaml"
PROBE_TYPES = ("cave", "forest", "pillar", "room", "wall")


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
        "relative_order_weight": 0.25,
        "relative_order_temperature": 0.75,
        "training_speed_mps": 6.0,
        "max_speed_mps": 6.0,
        "max_acceleration_mps2": 6.0,
        "vehicle_radius_m": 0.30,
        "clear_distance_m": 1.20,
        # The complete added candidate cost is mathematically bounded to this
        # small value.  A zero-weight ablation is covered by unit tests.
        "dangerous_segment_weight": 0.15,
        "dangerous_segment_window": 5,
        "dangerous_segment_focus": 8.0,
    }


def common(parent):
    config = copy.deepcopy(parent)
    config["contract_version"] = (
        "route_a_static_yopo_training_v4_5_bounded_danger_v1"
    )
    config["parent_v4_4_config_hash"] = sha256_file(PARENT)
    config["training_implementation_hash"] = training_implementation_hash()
    config["output_root"] = str(
        ROOT / "runs/route_a_static_yopo_v4_5_bounded_danger"
    )
    config["static_yopo_v4_4"] = {"enabled": False}
    config["static_yopo_v4_5"] = objective_mapping()
    config["loader"].pop("map_type_filter", None)
    config["validation"].update({
        "contract_version": "route_a_v4_5_bounded_danger_v1",
        "primary_metric": "macro_map_type_total_static_loss",
        "direction": "minimize",
        "minimum_epoch": 49,
        "patience": 50,
        "predefined_metrics": [
            "macro_map_type_total_static_loss",
            "per_map_type_total_static_loss",
            "score_top1_label_agreement",
            "score_oracle_regret",
            "dangerous_segment_loss",
            "selected_clearance",
            "selected_vertical_displacement",
            "oracle_vertical_displacement",
            "selected_primitive_row",
            "oracle_primitive_row",
            "selected_vertical_primitive",
            "oracle_vertical_primitive",
            "selected_upward_primitive",
            "selected_level_primitive",
            "selected_downward_primitive",
            "oracle_upward_primitive",
            "oracle_level_primitive",
            "oracle_downward_primitive",
        ],
    })
    config["implementation_hotfixes"] = [
        "route_a_v4_5_physical_outer_boundary_floor_v1",
        "route_a_v4_5_preferred_outer_envelope_diagnostic_only_v1",
        "route_a_v4_5_network_score_preserved_near_outer_boundary_v1",
        "route_a_v4_5_bounded_smooth_dangerous_segment_v1",
        "route_a_v4_5_vertical_behavior_diagnostics_only_v1",
        "route_a_v4_5_no_new_qualification_gate_v1",
    ]
    return config


def write_config(path, config, yaml):
    if path.exists():
        existing = yaml.load(path)
        if existing.get("contract_version") != config["contract_version"]:
            raise FileExistsError(f"refusing to overwrite foreign config: {path}")
    with path.open("w", encoding="utf-8") as stream:
        yaml.dump(config, stream)


def main():
    if not PARENT.is_file():
        raise FileNotFoundError(PARENT)
    yaml = YAML()
    parent = yaml.load(PARENT)
    full = common(parent)
    write_config(OUTPUT, full, yaml)

    probes = {}
    for map_type in PROBE_TYPES:
        config = common(parent)
        config["contract_version"] = (
            f"route_a_static_yopo_training_v4_5_probe_{map_type}_v1"
        )
        config["experiment_role"] = "map_type_convergence_probe"
        config["loader"]["map_type_filter"] = map_type
        config["loader"]["sampling_strategy"] = "uniform"
        config["training"].update({
            "max_epochs": 5,
            # Each type starts from exactly the same original epoch10 weights.
            "seed": 84501,
        })
        config["validation"].update({"minimum_epoch": 4, "patience": 5})
        config["output_root"] = str(
            ROOT / f"runs/route_a_static_yopo_v4_5_probe_{map_type}"
        )
        path = ROOT / f"configs/route_a_v4_5_probe_{map_type}.yaml"
        write_config(path, config, yaml)
        probes[map_type] = {
            "config": str(path), "sha256": sha256_file(path),
        }

    print(json.dumps({
        "status": "PASS",
        "full_config": str(OUTPUT),
        "full_config_sha256": sha256_file(OUTPUT),
        "probe_configs": probes,
        "dataset_reused": str(full["derived_dataset_root"]),
        "dataset_generation_required": False,
        "qualification_gate_count": 0,
        "training_started": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
