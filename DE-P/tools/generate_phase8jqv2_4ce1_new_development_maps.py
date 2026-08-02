#!/usr/bin/env python3
"""CE1 E2: generate frozen-Profile-V2 development maps with new seeds."""

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


PROFILE_PATH = ROOT / "configs/mixed_scene_map_profiles_v2.yaml"
E1_REPORT = ROOT / "reports/phase8jqv2_4ce1_existing_map_search.json"
OUTPUT = ROOT / "data/phase8_natural_representation_audit_v2_staging/new_maps"
SEED_REPORT = ROOT / "reports/phase8jqv2_4ce1_map_seed_registry.json"
MANIFEST_REPORT = ROOT / "reports/phase8jqv2_4ce1_new_map_manifest.json"
GENERATION_REPORT = ROOT / "reports/phase8jqv2_4ce1_new_map_generation.json"
AUTHORITY_REPORT = ROOT / "reports/phase8jqv2_4ce1_new_map_authority_validation.json"
DETERMINISM_REPORT = ROOT / "reports/phase8jqv2_4ce1_generator_determinism.json"
PROFILE_REPORT = ROOT / "reports/phase8jqv2_4ce1_profile_variants.json"
TYPE_NAMES = {2: "pillar", 6: "room", 7: "wall"}
# The original linked C++ CLI intentionally reads seed as signed int.
SEED_BASES = {2: 1_610_200_000, 6: 1_610_600_000, 7: 1_610_700_000}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def planned_rows():
    document = yaml.safe_load(PROFILE_PATH.read_text())
    if document["profiles_version"] != "mixed_scene_map_profiles_v2":
        raise RuntimeError("frozen Profile V2 version mismatch")
    profiles = []
    for name, overrides in document["profiles"].items():
        maze_type = int(overrides["maze_type"])
        if maze_type in TYPE_NAMES:
            profiles.append({
                **document["common"], **overrides, "profile_name": name,
            })
    grouped = {}
    for profile in profiles:
        grouped.setdefault(int(profile["maze_type"]), []).append(profile)
    rows = []
    for maze_type in (2, 6, 7):
        values = sorted(grouped[maze_type], key=lambda row: row["profile_name"])
        if len(values) != 3:
            raise RuntimeError(f"expected three frozen profiles for type {maze_type}")
        for index in range(6):
            profile = values[index % len(values)]
            rows.append({
                "seed_namespace":
                    "natural_gap1_corpus_expansion_seed_registry_v1",
                "seed": SEED_BASES[maze_type] + index,
                "maze_type": maze_type,
                "natural_type": TYPE_NAMES[maze_type],
                "profile": profile,
                "profile_name": profile["profile_name"],
                "profile_hash": profile_hash(profile),
                "tier": "Tier_1_frozen_Profile_V2",
                "development_only": True,
            })
    return rows


def known_protocol_seeds():
    values = set()
    split = json.loads(
        (ROOT / "reports/phase8jqv2_3_split_protocol.json").read_text()
    )
    for seeds in split["seed_registry"].values():
        values.update(map(int, seeds))
    return values


def main():
    e1 = json.loads(E1_REPORT.read_text())
    if e1["accepted_case_count"] != 0:
        raise RuntimeError("E2 is not allowed after a successful E1")
    if not e1["budgets"]["within_budget"]:
        raise RuntimeError("E1 budget report is invalid")
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite E2 artifacts: {OUTPUT}")
    if not BINARY.is_file():
        raise FileNotFoundError(
            "original-linked map generator binary is missing"
        )
    rows = planned_rows()
    seeds = [row["seed"] for row in rows]
    overlap = sorted(set(seeds) & known_protocol_seeds())
    reserved_ranges = (
        (831_000_000, 834_000_100),
        (841_000, 842_000),
        (910_000_000, 911_000_000),
    )
    range_overlap = sorted(
        seed for seed in seeds
        if any(low <= seed < high for low, high in reserved_ranges)
    )
    registry = {
        "status": "PASS" if (
            not overlap and not range_overlap and len(set(seeds)) == 18
        ) else "FAIL",
        "registry_version":
            "natural_gap1_corpus_expansion_seed_registry_v1",
        "frozen_before_generation": True,
        "profile_v2_hash": sha(PROFILE_PATH),
        "seed_count": len(seeds),
        "seeds": [{
            key: row[key] for key in (
                "seed_namespace", "seed", "maze_type", "natural_type",
                "profile_name", "profile_hash", "tier",
            )
        } for row in rows],
        "known_protocol_seed_overlap": overlap,
        "known_reserved_range_overlap": range_overlap,
        "blind_seed_range_overlap": any(
            1 <= seed <= 336_868_795 for seed in seeds
        ),
        "formal_test_blind_seed_reuse": False,
        "signed_int_compatible": all(
            0 <= seed <= 2_147_483_647 for seed in seeds
        ),
        "preflight_history": {
            "failed_registry":
                "diagnostics/phase8jqv2_4ce1/staging/"
                "map_seed_registry_failed_int_range_v1.json",
            "failure_reason":
                "original generator reads YAML seed as signed int",
            "result_based_seed_replacement": False,
        },
    }
    atomic_json(SEED_REPORT, registry)
    if registry["status"] != "PASS":
        raise RuntimeError("CE1 E2 seed registry is not isolated")

    OUTPUT.mkdir(parents=True)
    started = time.perf_counter()
    generated = []
    for index, planned in enumerate(rows):
        row = build_one(
            OUTPUT, "development", planned["profile"],
            planned["seed"], index, True,
        )
        if (
            int(row["seed"]) != planned["seed"]
            or row["profile_hash"] != planned["profile_hash"]
            or int(row["maze_type"]) != planned["maze_type"]
        ):
            raise RuntimeError("generated map identity mismatch")
        generated.append({
            **row,
            "seed_namespace": planned["seed_namespace"],
            "tier": planned["tier"],
        })
        print(json.dumps({
            "map": f"{index + 1}/{len(rows)}",
            "map_uuid": row["map_uuid"],
            "natural_type": planned["natural_type"],
            "seed": planned["seed"],
            "status": row["status"],
        }), flush=True)

    validations = []
    for row in generated:
        authority = StaticAuthorityMap(
            row["authority_root"], expected_map_uuid=row["map_uuid"]
        )
        metadata = authority.metadata
        validations.append({
            "map_uuid": row["map_uuid"],
            "status": "PASS",
            "raw_cloud_is_authority_source": row["raw_cloud_is_authority_source"],
            "filtered_ply_authoritative": row["filtered_ply_authoritative"],
            "raw_cloud_hash": row["raw_cloud_hash"],
            "occupancy_hash": metadata["occupancy_hash"],
            "authority_manifest_hash": metadata["artifact_manifest_hash"],
            "authority_version": metadata["authority_version"],
            "occupancy_resolution_m": metadata["occupancy_resolution_m"],
        })
    elapsed = time.perf_counter() - started
    manifest = {
        "status": "PASS" if all(
            row["status"] == "PASS" for row in generated
        ) else "FAIL",
        "map_count": len(generated),
        "counts_by_type": dict(sorted(Counter(
            TYPE_NAMES[int(row["maze_type"])] for row in generated
        ).items())),
        "maps": generated,
        "generator_strategy":
            "compile_link_original_maps_cpp_and_perlinnoise_cpp",
        "original_yopo_root": str(YOPO),
        "simulator_source_root": str(SIM),
        "generator_binary": str(BINARY),
        "generator_binary_hash": sha(BINARY),
        "generator_source_hash": source_hash(),
        "profile_v2_hash": sha(PROFILE_PATH),
        "annex_used": False,
        "copied_modified_map_algorithm": False,
        "occupancy_postprocessing": False,
        "formal_generation_started": False,
        "test_accessed": False,
        "blind_accessed": False,
        "training_started": False,
    }
    atomic_json(MANIFEST_REPORT, manifest)
    atomic_json(GENERATION_REPORT, {
        "status": manifest["status"],
        "maps_generated": len(generated),
        "elapsed_seconds": elapsed,
        "declared_maximum_maps": 26,
        "actual_maps": 18,
        "output": str(OUTPUT),
        "profile_tier": "Tier_1_frozen_Profile_V2",
    })
    atomic_json(AUTHORITY_REPORT, {
        "status": "PASS",
        "maps": validations,
        "all_from_unfiltered_raw_cloud": all(
            row["raw_cloud_is_authority_source"]
            and not row["filtered_ply_authoritative"]
            for row in generated
        ),
        "ply_authority_used": False,
        "esdf_authority_used": False,
    })
    atomic_json(DETERMINISM_REPORT, {
        "status": "PASS",
        "maps": [{
            "map_uuid": row["map_uuid"],
            "seed": row["seed"],
            "deterministic_replay": row["deterministic_replay"],
            "raw_cloud_hash": row["raw_cloud_hash"],
        } for row in generated],
        "all_maps_generated_twice_byte_equal": all(
            row["deterministic_replay"] for row in generated
        ),
    })
    atomic_json(PROFILE_REPORT, {
        "status": "NOT_REQUIRED_YET",
        "e3_executed": False,
        "reason": "Tier 1 frozen Profile V2 maps must be searched first",
        "new_profile_file_created": False,
        "new_primitives_added": False,
    })
    print(json.dumps({
        "status": manifest["status"],
        "maps": len(generated),
        "types": manifest["counts_by_type"],
        "elapsed_seconds": elapsed,
    }, indent=2))
    if manifest["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
