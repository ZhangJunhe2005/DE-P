#!/usr/bin/env python3
"""CE1 E3 bounded original-parameter profile variant map generation."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from geometry_authority.static_v1 import StaticAuthorityMap
from phase8jqv2_4m1_mixed_maps import (
    BINARY, SIM, YOPO, build_one, profile_hash, source_hash,
)


CONFIG = ROOT / "configs/natural_gap1_corpus_profiles_v1.yaml"
E1 = ROOT / "reports/phase8jqv2_4ce1_existing_map_search.json"
E2 = ROOT / "reports/phase8jqv2_4ce1_new_map_search.json"
OUTPUT = ROOT / "data/phase8_natural_representation_audit_v2_staging/profile_variant_maps"
MANIFEST = ROOT / "reports/phase8jqv2_4ce1_e3_map_manifest.json"
PROFILE_REPORT = ROOT / "reports/phase8jqv2_4ce1_profile_variants.json"
SEED_REPORT = ROOT / "reports/phase8jqv2_4ce1_e3_seed_registry.json"
TYPE_NAMES = {2: "pillar", 6: "room", 7: "wall"}
SEED_BASES = {2: 1_620_200_000, 6: 1_620_600_000, 7: 1_620_700_000}
ALLOWED_PARAMETER_KEYS = {
    "resolution", "complexity", "fill", "fractal", "attenuation",
    "width_min", "width_max", "obstacle_number", "road_width",
    "add_wall_x", "add_wall_y", "tree_file", "tree_dist",
    "room_number", "max_windows", "window_size_min",
    "window_size_max", "add_ceiling", "wall_width_min",
    "wall_width_max", "wall_thick", "wall_number", "wall_ceiling",
    "occlusion_annex", "maze_type", "semantic_name", "x_length",
    "y_length", "z_length", "post_translate", "profile_name",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main():
    e1, e2 = json.loads(E1.read_text()), json.loads(E2.read_text())
    if e1["accepted_case_count"] or e2["accepted_case_count"]:
        raise RuntimeError("E3 is not allowed after corpus success")
    document = yaml.safe_load(CONFIG.read_text())
    if document["profiles_version"] != "natural_gap1_corpus_profiles_v1":
        raise RuntimeError("E3 profile version mismatch")
    profiles = []
    for name, overrides in sorted(document["profiles"].items()):
        profile = {**document["common"], **overrides, "profile_name": name}
        extra = set(profile) - ALLOWED_PARAMETER_KEYS
        if extra:
            raise RuntimeError(f"non-original profile parameters: {extra}")
        if profile.get("occlusion_annex"):
            raise RuntimeError("annex is forbidden")
        profiles.append(profile)
    counts = Counter(int(row["maze_type"]) for row in profiles)
    if counts != Counter({2: 2, 6: 2, 7: 2}):
        raise RuntimeError("E3 profile budget mismatch")
    plans = []
    for profile in profiles:
        maze_type = int(profile["maze_type"])
        same_type = [
            value for value in profiles
            if int(value["maze_type"]) == maze_type
        ]
        variant_index = same_type.index(profile)
        for seed_offset in range(2):
            plans.append({
                "profile": profile,
                "profile_name": profile["profile_name"],
                "profile_hash": profile_hash(profile),
                "maze_type": maze_type,
                "natural_type": TYPE_NAMES[maze_type],
                "seed": (
                    SEED_BASES[maze_type] + variant_index * 10 + seed_offset
                ),
            })
    atomic_json(SEED_REPORT, {
        "status": "PASS",
        "registry_version":
            "natural_gap1_corpus_expansion_e3_seed_registry_v1",
        "frozen_before_generation": True,
        "profile_config_hash": sha(CONFIG),
        "seeds": [{
            key: row[key] for key in (
                "seed", "maze_type", "natural_type",
                "profile_name", "profile_hash",
            )
        } for row in plans],
        "seed_count": 12,
        "formal_test_blind_seed_reuse": False,
    })
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    started = time.perf_counter()
    maps = []
    for index, plan in enumerate(plans):
        row = build_one(
            OUTPUT, "development", plan["profile"],
            plan["seed"], index, True,
        )
        StaticAuthorityMap(
            row["authority_root"], expected_map_uuid=row["map_uuid"]
        )
        maps.append({
            **row,
            "natural_type": plan["natural_type"],
            "tier": "E3_bounded_original_parameter_variant",
            "seed_namespace":
                "natural_gap1_corpus_expansion_e3_seed_registry_v1",
        })
        print(json.dumps({
            "map": f"{index + 1}/12",
            "map_uuid": row["map_uuid"],
            "type": plan["natural_type"],
            "profile": plan["profile_name"],
            "seed": plan["seed"],
            "status": row["status"],
        }), flush=True)
    value = {
        "status": "PASS" if all(row["status"] == "PASS" for row in maps)
            else "FAIL",
        "map_count": len(maps),
        "maps": maps,
        "profile_config": str(CONFIG),
        "profile_config_hash": sha(CONFIG),
        "profiles_per_type": 2,
        "seeds_per_profile": 2,
        "maximum_maps": 12,
        "original_yopo_generator_reused": True,
        "generator_strategy":
            "compile_link_original_maps_cpp_and_perlinnoise_cpp",
        "generator_binary": str(BINARY),
        "generator_source_hash": source_hash(),
        "original_yopo_root": str(YOPO),
        "simulator_source_root": str(SIM),
        "annex_used": False,
        "new_primitive_added": False,
        "occupancy_postprocessing": False,
        "detector_output_used": False,
        "elapsed_seconds": time.perf_counter() - started,
        "formal_generation_started": False,
        "test_accessed": False,
        "blind_accessed": False,
        "training_started": False,
    }
    atomic_json(MANIFEST, value)
    atomic_json(PROFILE_REPORT, {
        **value,
        "e3_executed": True,
        "trigger": {
            "e1_hard_gate_failed": True,
            "e2_hard_gate_failed": True,
            "underrepresented_types_with_geometry_potential":
                ["pillar", "room", "wall"],
        },
        "only_original_generator_parameters_changed": True,
        "profiles_frozen_before_cuda": True,
    })
    print(json.dumps({
        "status": value["status"],
        "maps": len(maps),
        "elapsed_seconds": value["elapsed_seconds"],
    }, indent=2))
    if value["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
