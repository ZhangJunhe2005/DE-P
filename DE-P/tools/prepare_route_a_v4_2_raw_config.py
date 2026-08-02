#!/usr/bin/env python3
"""Create the versioned Route-A V4.2 scene-relative spatial config."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "configs/route_a_v4_raw_static_resolved.yaml"
TARGET = ROOT / "configs/route_a_v4_2_raw_static_resolved.yaml"
OUTPUT = ROOT / "data/route_a_v4_2_raw_static"


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()


def contract():
    value = {
        "version": "route_a_scene_spatial_sampling_v1",
        "altitude_reference": (
            "normalized_within_authority_bounds_after_0.4m_center_margin"
        ),
        "collision_radius_m": 0.3,
        "maximum_sequence_resamples": 32,
        "map_types": {
            "cave": {
                "interpretation": "enclosed_volume_full_height_coverage",
                "altitude_bands": [
                    {"name": "low", "relative_min": 0.05,
                     "relative_max": 0.33, "weight": 0.34},
                    {"name": "middle", "relative_min": 0.33,
                     "relative_max": 0.67, "weight": 0.33},
                    {"name": "high", "relative_min": 0.67,
                     "relative_max": 0.95, "weight": 0.33},
                ],
                "near_depth_m": 10.0,
                "minimum_return_fraction": 0.35,
                "minimum_near_obstacle_fraction": 0.20,
            },
            "pillar": {
                "interpretation": "lower_and_middle_obstacle_field_primary",
                "altitude_bands": [
                    {"name": "low", "relative_min": 0.06,
                     "relative_max": 0.38, "weight": 0.58},
                    {"name": "middle", "relative_min": 0.38,
                     "relative_max": 0.66, "weight": 0.32},
                    {"name": "high_observable", "relative_min": 0.66,
                     "relative_max": 0.80, "weight": 0.10},
                ],
                "near_depth_m": 10.0,
                "minimum_return_fraction": 0.34,
                "minimum_near_obstacle_fraction": 0.18,
            },
            "forest": {
                "interpretation": "trunk_layer_primary_canopy_secondary",
                "altitude_bands": [
                    {"name": "trunk", "relative_min": 0.06,
                     "relative_max": 0.38, "weight": 0.62},
                    {"name": "lower_canopy", "relative_min": 0.38,
                     "relative_max": 0.66, "weight": 0.30},
                    {"name": "upper_observable", "relative_min": 0.66,
                     "relative_max": 0.78, "weight": 0.08},
                ],
                "near_depth_m": 10.0,
                "minimum_return_fraction": 0.28,
                "minimum_near_obstacle_fraction": 0.14,
            },
            "room": {
                "interpretation": "enclosed_walls_full_height_coverage",
                "altitude_bands": [
                    {"name": "low", "relative_min": 0.05,
                     "relative_max": 0.33, "weight": 0.34},
                    {"name": "middle", "relative_min": 0.33,
                     "relative_max": 0.67, "weight": 0.33},
                    {"name": "high", "relative_min": 0.67,
                     "relative_max": 0.95, "weight": 0.33},
                ],
                "near_depth_m": 10.0,
                "minimum_return_fraction": 0.65,
                "minimum_near_obstacle_fraction": 0.50,
            },
            "wall": {
                "interpretation": "lower_and_middle_wall_field_primary",
                "altitude_bands": [
                    {"name": "low", "relative_min": 0.06,
                     "relative_max": 0.40, "weight": 0.54},
                    {"name": "middle", "relative_min": 0.40,
                     "relative_max": 0.68, "weight": 0.36},
                    {"name": "high_observable", "relative_min": 0.68,
                     "relative_max": 0.82, "weight": 0.10},
                ],
                "near_depth_m": 10.0,
                "minimum_return_fraction": 0.38,
                "minimum_near_obstacle_fraction": 0.20,
            },
        },
    }
    value["contract_hash"] = hashlib.sha256(canonical(value)).hexdigest()
    return value


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    value = yaml.safe_load(SOURCE.read_text())
    value["dataset_version"] = "route_a_v4_2_raw_static_v1"
    value["output_root"] = str(OUTPUT)
    value["random_seeds"]["sequence_base"] = 826000000
    value["state_sampling_contract"]["semantics_version"] = (
        "route_a_v4_2_scene_relative_camera_pose_sampling_v1"
    )
    value["scene_spatial_sampling_contract"] = contract()
    value["smoke_settings"].update({
        "frames_per_split": 1000,
        "frames_per_sequence": 20,
        "map_count": 5,
    })
    encoded = yaml.safe_dump(value, sort_keys=False)
    if TARGET.exists() and TARGET.read_text() != encoded:
        raise RuntimeError(
            f"refusing to overwrite different V4.2 config: {TARGET}"
        )
    TARGET.write_text(encoded)
    print(json.dumps({
        "status": "PASS",
        "config": str(TARGET),
        "config_sha256": hashlib.sha256(TARGET.read_bytes()).hexdigest(),
        "spatial_contract_hash":
            value["scene_spatial_sampling_contract"]["contract_hash"],
        "output": str(OUTPUT),
        "frames": {"train": 300000, "valid": 60000},
        "map_authority_reused": "route_a_v4_mixed_maps",
        "training_started": False,
    }, indent=2))


if __name__ == "__main__":
    main()
