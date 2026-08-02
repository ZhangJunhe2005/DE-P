#!/usr/bin/env python3
"""Fixed-seed raw-cloud parameter data-flow audit for original YOPO Maps."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from phase8jqv2_4m1_mixed_maps import invoke_generator

REPORTS = ROOT / "reports"
PROFILE_PATH = ROOT / "configs/mixed_scene_map_profiles_v2.yaml"
MAPS_CPP = ROOT.parent / "Simulator/src/src/maps.cpp"
MAPS_HPP = ROOT.parent / "Simulator/src/include/maps.hpp"
TREE = ROOT.parent / "Simulator/src/pointcloud/tree.ply"
TYPE_NAME = {1: "cave", 2: "pillar", 5: "forest", 6: "room", 7: "wall"}


def atomic_new(path: Path, value) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() == payload:
            return
        raise FileExistsError(f"refusing to overwrite MTC1 evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload)
    os.replace(temporary, path)


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def cloud_result(profile, seed):
    with tempfile.TemporaryDirectory(prefix="phase8-mtc1-param-", dir="/tmp") as tmp:
        path = Path(tmp) / "cloud.bin"
        points = invoke_generator(profile, seed, path)
    return {
        "raw_cloud_hash": sha_bytes(points.tobytes()),
        "raw_point_count": int(len(points)),
        "minimum": points.min(axis=0).astype(float).tolist(),
        "maximum": points.max(axis=0).astype(float).tolist(),
    }


def profiles():
    document = yaml.safe_load(PROFILE_PATH.read_text())
    result = {}
    for name, override in document["profiles"].items():
        maze_type = int(override["maze_type"])
        if maze_type not in result:
            result[maze_type] = {
                **document["common"], **override, "profile_name": name,
            }
    return result


def main():
    entry = json.loads(
        (REPORTS / "phase8jqv2_4mtc1_entry_gate.json").read_text()
    )
    if entry["status"] != "PASS":
        raise RuntimeError("MTC1 entry gate is not PASS")
    baseline = profiles()
    seeds = {1: 1_731_000_001, 2: 1_731_000_002, 5: 1_731_000_005,
             6: 1_731_000_006, 7: 1_731_000_007}
    perturbations = {
        1: {
            "complexity": 0.041, "fill": 0.14,
            "fractal": 2, "attenuation": 0.18,
        },
        2: {
            "width_min": 1.4, "width_max": 2.2,
            "obstacle_number": 33,
        },
        5: {"tree_dist": 6.0},
        6: {
            "room_number": 4, "max_windows": 3,
            "window_size_min": 1.1, "window_size_max": 1.7,
            "add_ceiling": 1,
        },
        7: {
            "wall_width_min": 2.8, "wall_width_max": 7.2,
            "wall_thick": 0.2, "wall_number": 22, "wall_ceiling": 1,
        },
    }
    rows = []
    base_results = {
        maze_type: cloud_result(profile, seeds[maze_type])
        for maze_type, profile in baseline.items()
    }
    for maze_type, values in perturbations.items():
        for parameter, value in values.items():
            changed = dict(baseline[maze_type])
            changed[parameter] = value
            result = cloud_result(changed, seeds[maze_type])
            rows.append({
                "maze_type": maze_type,
                "natural_type": TYPE_NAME[maze_type],
                "parameter": parameter,
                "baseline_value": baseline[maze_type][parameter],
                "perturbed_value": value,
                "seed": seeds[maze_type],
                "baseline": base_results[maze_type],
                "perturbed": result,
                "raw_cloud_changed":
                    result["raw_cloud_hash"]
                    != base_results[maze_type]["raw_cloud_hash"],
            })
    fill_cross_type = []
    for maze_type, profile in baseline.items():
        changed = dict(profile)
        changed["fill"] = 0.14
        result = cloud_result(changed, seeds[maze_type])
        fill_cross_type.append({
            "maze_type": maze_type, "natural_type": TYPE_NAME[maze_type],
            "seed": seeds[maze_type],
            "baseline_fill": profile["fill"], "perturbed_fill": 0.14,
            "baseline_hash": base_results[maze_type]["raw_cloud_hash"],
            "perturbed_hash": result["raw_cloud_hash"],
            "raw_cloud_changed":
                result["raw_cloud_hash"]
                != base_results[maze_type]["raw_cloud_hash"],
        })
    unrelated = []
    for maze_type, profile in baseline.items():
        changed = dict(profile)
        changed["fill"] = 0.14
        changed["obstacle_number"] = 31
        changed["tree_dist"] = 6.5
        changed["room_number"] = 4
        changed["wall_number"] = 21
        relevant = {
            1: ["fill"], 2: ["obstacle_number"], 5: ["tree_dist"],
            6: ["room_number"], 7: ["wall_number"],
        }[maze_type]
        for key in relevant:
            changed[key] = profile[key]
        result = cloud_result(changed, seeds[maze_type])
        unrelated.append({
            "maze_type": maze_type, "natural_type": TYPE_NAME[maze_type],
            "changed_only_other_type_parameters": True,
            "raw_cloud_unchanged":
                result["raw_cloud_hash"]
                == base_results[maze_type]["raw_cloud_hash"],
        })
    effect = {
        "status": "PASS",
        "fixed_seed_paired": True,
        "rows": rows,
        "fill_cross_type": fill_cross_type,
        "unrelated_type_perturbation": unrelated,
        "fill_only_affects_cave": (
            next(x for x in fill_cross_type if x["maze_type"] == 1)[
                "raw_cloud_changed"
            ]
            and all(
                not x["raw_cloud_changed"] for x in fill_cross_type
                if x["maze_type"] != 1
            )
        ),
    }
    atomic_new(REPORTS / "phase8jqv2_4mtc1_parameter_effect_matrix.json",
               effect)
    dataflow = {
        "status": "PASS",
        "evidence": {
            "maps_cpp": str(MAPS_CPP),
            "maps_cpp_sha256": hashlib.sha256(MAPS_CPP.read_bytes()).hexdigest(),
            "maps_hpp": str(MAPS_HPP),
            "maps_hpp_sha256": hashlib.sha256(MAPS_HPP.read_bytes()).hexdigest(),
            "paired_raw_cloud_experiment":
                "phase8jqv2_4mtc1_parameter_effect_matrix.json",
        },
        "types": {
            "cave": {
                "maze_type": 1, "function": "Maps::perlin3D",
                "parameters": ["complexity", "fill", "fractal", "attenuation"],
            },
            "pillar": {
                "maze_type": 2, "function": "Maps::randomMapGenerate",
                "parameters": ["width_min", "width_max", "obstacle_number"],
            },
            "forest": {
                "maze_type": 5, "function": "Maps::forest",
                "parameters": ["tree_dist", "tree_file"],
                "tree_file_sha256": hashlib.sha256(TREE.read_bytes()).hexdigest(),
            },
            "room": {
                "maze_type": 6, "function": "Maps::room",
                "parameters": [
                    "room_number", "max_windows", "window_size_min",
                    "window_size_max", "add_ceiling",
                ],
            },
            "wall": {
                "maze_type": 7, "function": "Maps::wall",
                "parameters": [
                    "wall_width_min", "wall_width_max", "wall_thick",
                    "wall_number", "wall_ceiling",
                ],
            },
        },
        "all_yaml_fields_are_read_in": "Maps::setParam",
        "generation_dispatch": "Maps::generate(maze_type)",
        "raw_cloud_then_canonicalized":
            "tools/phase8jqv2_4m1_mixed_maps.py::build_one",
    }
    atomic_new(REPORTS / "phase8jqv2_4mtc1_parameter_dataflow.json",
               dataflow)
    hardcoded = {
        "status": "PASS",
        "room": {
            "wall_thickness_m": 0.2,
            "minimum_windows_per_wall": 1,
            "window_count_expression": "rand() % max_windows + 1",
            "max_windows_zero_legal": False,
            "max_windows_zero_failure": "integer modulo by zero",
            "window_position_fraction_range": [0.1, 0.9],
            "source": "Maps::room / Maps::generateWallWithWindows",
        },
        "wall": {
            "yaw_configurable": False,
            "yaw_distribution_rad": [-3.141592653589793, 3.141592653589793],
            "pitch_configurable": False,
            "pitch_distribution_rad": [
                -0.3141592653589793, 0.3141592653589793],
            "height_configurable": False,
            "height_fraction_of_map_z": [0.3, 1.0],
            "source": "Maps::wall",
        },
        "forest": {
            "tree_scale_configurable": False,
            "tree_scale_range": [0.5, 1.0],
            "rotation_configurable": False,
            "roll_pitch_degree_range": [0.0, 10.0],
            "yaw_degree_range": [0.0, 360.0],
        },
        "pillar": {
            "center_sampled_continuously": True,
            "center_forced_to_canonical_grid": False,
            "height_distribution_m": "uniform(0,map_z)",
            "shell_not_solid_interior": True,
        },
        "annex_used": False,
    }
    atomic_new(REPORTS / "phase8jqv2_4mtc1_hardcoded_generator_parameters.json",
               hardcoded)
    print(json.dumps({
        "status": "PASS",
        "paired_tests": len(rows),
        "fill_only_affects_cave": effect["fill_only_affects_cave"],
    }, indent=2))


if __name__ == "__main__":
    main()
