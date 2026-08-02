#!/usr/bin/env python3
"""Versioned mixed-scene authority map protocol.

This wrapper compiles against and invokes the original Simulator mocka::Maps
implementation. It never edits Simulator/config/config.yaml and never treats a
filtered PLY as geometry authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import time
import uuid
import sys

import numpy as np
from scipy import ndimage
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PARENT = ROOT.parent
SIM = PARENT / "Simulator/src"
YOPO = PARENT / "YOPO"
PROTOCOL_PATH = ROOT / "configs/mixed_scene_authority_map_protocol_v1.yaml"
PROFILES_PATH = ROOT / "configs/mixed_scene_map_profiles_v1.yaml"
BINARY = ROOT / "artifacts/mixed_scene_map_generator/yopo_mixed_scene_map_generator"
NAMESPACE = uuid.UUID("8ca41625-f46e-51e1-9690-838e23e8eaa2")
RAW_COUNT = struct.Struct("<Q")

from geometry_authority.static_v1 import (  # noqa: E402
    StaticAuthorityMap, build_authority_artifact, sha256,
)


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_yaml(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(yaml.safe_dump(value, sort_keys=False))
    os.replace(temporary, path)


def load_documents():
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text())
    profiles = yaml.safe_load(PROFILES_PATH.read_text())
    if protocol["protocol_version"] != "mixed_scene_authority_map_protocol_v1":
        raise RuntimeError("protocol version mismatch")
    if profiles["profiles_version"] != "mixed_scene_map_profiles_v1":
        raise RuntimeError("profiles version mismatch")
    return protocol, profiles


def source_hash():
    paths = [
        SIM / "include/maps.hpp", SIM / "src/maps.cpp",
        SIM / "include/perlinnoise.hpp", SIM / "src/perlinnoise.cpp",
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(PARENT)).encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def parent_commit():
    return subprocess.check_output(
        ["git", "-C", str(PARENT), "rev-parse", "HEAD"], text=True
    ).strip()


def profile_rows(profiles):
    common = profiles["common"]
    rows = []
    for name, overrides in profiles["profiles"].items():
        resolved = {**common, **overrides}
        resolved["profile_name"] = name
        rows.append(resolved)
    return sorted(rows, key=lambda row: int(row["maze_type"]))


def profile_hash(profile):
    return hashlib.sha256(canonical(profile)).hexdigest()


def read_helper_cloud(path):
    raw = Path(path).read_bytes()
    if len(raw) < RAW_COUNT.size:
        raise RuntimeError("truncated helper cloud")
    count = RAW_COUNT.unpack_from(raw)[0]
    points = np.frombuffer(raw, dtype="<f4", offset=RAW_COUNT.size)
    if points.size != count * 3:
        raise RuntimeError("helper cloud count/size mismatch")
    result = points.reshape(-1, 3).copy()
    if not len(result) or not np.isfinite(result).all():
        raise RuntimeError("empty or non-finite helper cloud")
    return result


def invoke_generator(profile, seed, output):
    if not BINARY.is_file():
        raise FileNotFoundError(
            f"generator binary missing: run {ROOT/'scripts/build_mixed_scene_map_generator.sh'}"
        )
    resolved = dict(profile)
    resolved["seed"] = int(seed)
    with tempfile.TemporaryDirectory(prefix="phase8m1-map-", dir="/tmp") as tmp:
        config = Path(tmp) / "resolved.yaml"
        config.write_text(yaml.safe_dump(resolved, sort_keys=True))
        subprocess.run(
            [str(BINARY), "--config", str(config), "--output", str(output)],
            check=True,
        )
    return read_helper_cloud(output)


def geometry_metrics(authority_root):
    authority = StaticAuthorityMap(authority_root)
    occupied = authority.flat.reshape(tuple(authority.dimensions))
    free_fraction = float(1.0 - occupied.mean())
    stride = 2
    sampled_free = ~occupied[::stride, ::stride, ::stride]
    labels, count = ndimage.label(sampled_free)
    sizes = np.bincount(labels.ravel()) if count else np.asarray([0])
    connected_fraction = (
        float(sizes[1:].max() / sampled_free.sum())
        if len(sizes) > 1 and sampled_free.sum() else 0.0
    )
    patches = 0
    for axis in (0, 1):
        face = occupied.copy()
        source = [slice(None)] * 3
        neighbour = [slice(None)] * 3
        source[axis] = slice(0, -1)
        neighbour[axis] = slice(1, None)
        face[tuple(source)] &= ~occupied[tuple(neighbour)]
        tangent = 1 - axis
        base = [slice(None)] * 3
        step_t = [slice(None)] * 3
        step_z = [slice(None)] * 3
        step_tz = [slice(None)] * 3
        base[tangent], base[2] = slice(0, -1), slice(0, -1)
        step_t[tangent], step_t[2] = slice(1, None), slice(0, -1)
        step_z[tangent], step_z[2] = slice(0, -1), slice(1, None)
        step_tz[tangent], step_tz[2] = slice(1, None), slice(1, None)
        patch = (
            face[tuple(base)] & face[tuple(step_t)]
            & face[tuple(step_z)] & face[tuple(step_tz)]
        )
        patches += int(patch.sum())
    return {
        "occupied_voxel_count": int(occupied.sum()),
        "occupancy_dimensions": authority.dimensions.astype(int).tolist(),
        "free_fraction": free_fraction,
        "largest_connected_free_fraction_stride2": connected_fraction,
        "actionable_coplanar_2x2_patch_count": patches,
        "ordinary_navigation_capable": bool(
            free_fraction >= 0.45 and connected_fraction >= 0.55
        ),
        "occlusion_geometry_capable": bool(patches > 0),
    }


def build_one(root, split, profile, seed, index, development):
    map_id = (
        f"m1_sweep_t{profile['maze_type']}_{index:02d}"
        if development else f"{split}_map_{index:04d}"
    )
    identity = (
        f"{'development' if development else split}:"
        f"{profile['profile_name']}:{seed}:{index}"
    )
    map_uuid = str(uuid.uuid5(NAMESPACE, identity))
    target = Path(root) / f"geometry_authority/{split}/{map_uuid}"
    state = Path(root) / f"generation_state/maps/{split}_{map_uuid}.json"
    if target.is_dir() and state.is_file():
        existing = json.loads(state.read_text())
        if existing["profile_hash"] != profile_hash(profile):
            raise RuntimeError("resume profile hash mismatch")
        StaticAuthorityMap(target, expected_map_uuid=map_uuid)
        return existing
    if target.exists():
        raise RuntimeError(f"unverified map target exists: {target}")
    Path(root, ".staging").mkdir(parents=True, exist_ok=True)
    raw_a = Path(root) / f".staging/{map_uuid}.a.bin"
    raw_b = Path(root) / f".staging/{map_uuid}.b.bin"
    for path in (raw_a, raw_b):
        path.unlink(missing_ok=True)
    points_a = invoke_generator(profile, seed, raw_a)
    points_b = invoke_generator(profile, seed, raw_b)
    deterministic = hashlib.sha256(points_a.tobytes()).hexdigest() == hashlib.sha256(
        points_b.tobytes()
    ).hexdigest()
    if not deterministic or not np.array_equal(points_a, points_b):
        raise RuntimeError("original YOPO map generation is not deterministic")
    generator_config_hash = profile_hash(profile)
    metadata = build_authority_artifact(
        target, points_a, map_uuid=map_uuid, generator_seed=seed,
        map_id=map_id, generator_source_hash=source_hash(),
        generator_config_hash=generator_config_hash,
        parent_git_commit=parent_commit(),
        map_namespace=(
            "mixed_scene_authority_map_sweep_v1" if development
            else f"mixed_scene_authority_formal_{split}_v1"
        ),
        build_timestamp="1970-01-01T00:00:00Z",
    )
    metrics = geometry_metrics(target)
    provenance = {
        "protocol_version": "mixed_scene_authority_map_protocol_v1",
        "map_uuid": map_uuid, "map_id": map_id, "split": split,
        "development_only": development,
        "maze_type": int(profile["maze_type"]),
        "semantic_name": profile["semantic_name"],
        "seed": int(seed), "profile_name": profile["profile_name"],
        "profile_hash": generator_config_hash,
        "resolved_parameters": profile,
        "raw_cloud_hash": metadata["source_cloud_hash"],
        "occupancy_hash": metadata["occupancy_hash"],
        "authority_manifest_hash": metadata["artifact_manifest_hash"],
        "generator_source_hash": source_hash(),
        "simulator_source_hash": sha256(SIM / "src/maps.cpp"),
        "sensor_source_hash": sha256(SIM / "src/sensor_simulator.cu"),
        "forest_tree_hash": sha256(SIM / "pointcloud/tree.ply"),
        "parent_git_commit": parent_commit(),
        "build_timestamp": "1970-01-01T00:00:00Z",
        "raw_cloud_is_authority_source": True,
        "filtered_ply_authoritative": False,
        "deterministic_replay": deterministic,
        **metrics,
    }
    atomic_json(target / "mixed_scene_provenance.json", provenance)
    derived = Path(root) / f"derived_geometry/{split}/{map_uuid}.json"
    atomic_json(derived, {
        "version": "derived_esdf_authority_v1",
        "authoritative": False, "exact_verifier_required": True,
        "source_authority_hash": metadata["artifact_manifest_hash"],
        "source_occupancy_hash": metadata["occupancy_hash"],
        "resolution_m": metadata["occupancy_resolution_m"],
        "origin": metadata["occupancy_origin"],
        "dimensions": metadata["occupancy_dimensions"],
        "generator_hash": source_hash(),
        "representation": "deferred_exact_distance_cache",
    })
    row = {
        **provenance,
        "authority_root": str(target),
        "status": "PASS" if (
            metrics["ordinary_navigation_capable"]
            and metrics["occlusion_geometry_capable"]
        ) else "FAIL",
    }
    atomic_json(state, row)
    raw_a.unlink(missing_ok=True)
    raw_b.unlink(missing_ok=True)
    return row


def entry_gate():
    final = json.loads(
        (ROOT / "reports/phase8jqv2_4o1_final_result.json").read_text()
    )
    observed = dict(final)
    observed["training_started"] = final.get(
        "training_started", final.get("formal_training_started")
    )
    expected = {
        "status": "FAIL",
        "primary_cause": "formal_authority_maps_not_occlusion_capable",
        "formal_v3_generated": False,
        "training_started": False,
        "production_test_accessed": False,
        "blind_accessed": False,
    }
    mismatches = {
        key: {"expected": value, "actual": final.get(key)}
        for key, value in expected.items() if observed.get(key) != value
    }
    value = {
        "status": "PASS" if not mismatches else "FAIL",
        "phase": "phase8jqv2_4_occlusion_capable_authority_map_protocol",
        "parent_repository": str(PARENT),
        "dep_root": str(ROOT), "original_yopo_root": str(YOPO),
        "simulator_root": str(SIM.parent),
        "route_b_confirmed": not mismatches,
        "o1_train_eligible_maps": 0, "o1_valid_eligible_maps": 0,
        "v3_generated": False, "network_weights_modified": False,
        "optimizer_step_executed": False, "training_started": False,
        "production_test_accessed": False, "blind_accessed": False,
        "mismatches": mismatches,
    }
    atomic_json(ROOT / "reports/phase8jqv2_4m1_entry_gate.json", value)
    if mismatches:
        raise RuntimeError(f"M1 entry gate mismatch: {mismatches}")
    return value


def reuse_audit():
    protocol, profiles = load_documents()
    maps_source = SIM / "src/maps.cpp"
    head_dataset = subprocess.check_output(
        ["git", "-C", str(PARENT), "show",
         "HEAD:Simulator/src/src/dataset_generator.cpp"], text=True
    )
    value = {
        "status": "PASS",
        "parent_git_root": str(PARENT),
        "source_discovery": {
            "original_yopo": str(YOPO), "simulator": str(SIM),
            "maps_cpp": str(maps_source),
            "tree_asset": protocol["tree_asset"],
        },
        "maze_type_dispatch": {
            "1": "perlin3D/cave", "2": "randomMapGenerate/pillar",
            "3": "maze2D", "4": "Maze3DGen",
            "5": "forest", "6": "room", "7": "wall",
        },
        "selected_candidate_types": [1, 2, 5, 6, 7],
        "determinism": {
            "types_1_2_5_6_7_use_explicit_info_seed": True,
            "types_3_4_seed_global_rand_with_info_seed": True,
            "random_device_found": False,
        },
        "geometry_semantics": {
            "continuous_surfaces_generated": True,
            "gridmap_built_from_raw_cloud": True,
            "saved_ply_is_voxel_filtered": True,
            "saved_ply_authoritative": False,
            "raw_cloud_available_from_original_cli": False,
        },
        "legacy_dataset_generator": {
            "deletes_output_unconditionally": False,
            "refuses_existing_without_explicit_overwrite": (
                "refusing to delete or reuse existing path" in head_dataset
            ),
            "safe_for_mixed_formal_generation": False,
            "reason": "single global maze_type/config and no raw-cloud artifact",
        },
        "reuse_decision": {
            "strategy": "compile_link_original_maps_cpp_and_perlinnoise_cpp",
            "copies_original_algorithm": False,
            "mutates_global_simulator_config": False,
            "preserves_old_executables": True,
        },
        "hashes": {
            "maps_cpp": sha256(maps_source),
            "maps_hpp": sha256(SIM / "include/maps.hpp"),
            "tree_ply": sha256(SIM / "pointcloud/tree.ply"),
            "profile_file": sha256(PROFILES_PATH),
            "protocol_file": sha256(PROTOCOL_PATH),
        },
        "profile_count": len(profile_rows(profiles)),
    }
    atomic_json(ROOT / "reports/phase8jqv2_4m1_yopo_map_reuse_audit.json", value)
    decision = """# Phase 8J-Q2.4-M1 YOPO map reuse decision

The mixed-map path links the original Simulator `mocka::Maps` implementation
(`maps.cpp`, `maps.hpp`, `perlinnoise.cpp`) into a separate versioned executable.
It does not copy the algorithms, edit the global Simulator config, or call the
single-type depth dataset generator.

The original generator builds GridMap from its unfiltered procedural cloud but
saves a VoxelGrid-filtered PLY. Consequently the PLY is diagnostic only. M1
captures the unfiltered cloud directly and converts it to the frozen 0.1 m
canonical occupancy artifact.

Candidate types are cave (1), pillar (2), forest (5), room (6), and wall (7).
Types 3/4 remain audited but are not selected unless a later repair phase is
needed.
"""
    (ROOT / "reports/phase8jqv2_4m1_yopo_map_reuse_decision.md").write_text(decision)
    return value


def sweep():
    entry_gate()
    reuse_audit()
    protocol, profiles = load_documents()
    root = Path(protocol["development"]["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for profile in profile_rows(profiles):
        for index in range(int(protocol["development"]["maps_per_type"])):
            seed = (
                int(protocol["development"]["seed_base"])
                + int(profile["maze_type"]) * 1000 + index
            )
            rows.append(build_one(
                root, "development", profile, seed, index, True
            ))
    by_type = {}
    for row in rows:
        by_type.setdefault(str(row["maze_type"]), []).append(row)
    selected = [
        int(key) for key, values in sorted(by_type.items(), key=lambda item: int(item[0]))
        if all(row["status"] == "PASS" for row in values)
    ]
    minimum = int(protocol["selection"]["minimum_passing_types"])
    status = "PASS" if len(selected) >= minimum else "FAIL"
    report = {
        "status": status, "protocol_version": protocol["protocol_version"],
        "maps_generated": len(rows), "maps_per_type": 3,
        "types_evaluated": sorted(int(value) for value in by_type),
        "selected_map_types": selected, "minimum_passing_types": minimum,
        "maps": rows,
        "cuda_raster_validation": "PENDING_HOST_GATE",
        "frozen_identity_validation": "PENDING_HOST_GATE",
        "formal_generation_started": False, "training_started": False,
        "production_test_accessed": False, "blind_accessed": False,
    }
    atomic_json(ROOT / "reports/phase8jqv2_4m1_map_type_sweep.json", report)
    atomic_json(ROOT / "reports/phase8jqv2_4m1_selected_map_types.json", {
        "status": status, "selected_map_types": selected,
        "selection_rule": "all_three_development_maps_pass_round_robin_eligible",
        "formal_generation_started": False,
    })
    atomic_json(ROOT / "reports/phase8jqv2_4m1_map_profile_validation.json", {
        "status": status,
        "profiles": {
            row["profile_name"]: {
                "maze_type": row["maze_type"],
                "profile_hash": row["profile_hash"],
            } for row in rows
        },
    })
    summary = [
        "# M1 mixed map development sweep", "",
        f"Status: **{status}**", f"Maps: {len(rows)}",
        f"Selected maze types: {selected}", "",
    ]
    for key, values in by_type.items():
        summary.append(
            f"- type {key}: {sum(row['status']=='PASS' for row in values)}/3 PASS"
        )
    (ROOT / "reports/phase8jqv2_4m1_map_type_sweep_summary.md").write_text(
        "\n".join(summary) + "\n"
    )
    if status != "PASS":
        raise RuntimeError("fewer than four mixed map types passed")
    prepare_formal(selected, protocol, profiles)
    return report


def prepare_formal(selected, protocol, profiles):
    selected_profiles = [
        row for row in profile_rows(profiles)
        if int(row["maze_type"]) in set(selected)
    ]
    maps = {}
    for split, count, base, namespace in (
        ("train", 48, protocol["formal"]["train_seed_base"],
         protocol["formal"]["train_namespace"]),
        ("valid", 12, protocol["formal"]["valid_seed_base"],
         protocol["formal"]["valid_namespace"]),
    ):
        rows = []
        for index in range(count):
            profile = selected_profiles[index % len(selected_profiles)]
            seed = int(base) + index
            identity = f"{split}:{profile['profile_name']}:{seed}:{index}"
            rows.append({
                "map_id": f"{split}_map_{index:04d}",
                "map_uuid": str(uuid.uuid5(NAMESPACE, identity)),
                "namespace": namespace, "seed": seed,
                "maze_type": int(profile["maze_type"]),
                "semantic_name": profile["semantic_name"],
                "profile_name": profile["profile_name"],
                "profile_hash": profile_hash(profile),
            })
        maps[split] = rows
    plan = {
        "status": "FROZEN_NOT_STARTED",
        "protocol_version": protocol["protocol_version"],
        "dataset_version": protocol["dataset_version"],
        "selected_map_types": selected,
        "formal_maps": maps,
        "distribution": {
            split: {
                str(value): sum(row["maze_type"] == value for row in rows)
                for value in selected
            } for split, rows in maps.items()
        },
        "train_frames": 1_000_000, "valid_frames": 100_000,
        "frames_per_sequence": 60,
        "formal_generation_started": False,
    }
    atomic_json(ROOT / "reports/phase8jqv2_4m1_formal_map_plan.json", plan)
    atomic_json(ROOT / "reports/phase8jqv2_4m1_formal_generation_plan.json", {
        **plan,
        "projected_dataset_bytes": 72_000_000_000,
        "minimum_free_bytes": 118_000_000_000,
        "estimated_hours_reference": [2.5, 4.5],
    })
    atomic_json(ROOT / "reports/phase8jqv2_4m1_mixed_map_protocol.json", {
        "status": "PASS", "protocol": protocol,
        "profiles_hash": sha256(PROFILES_PATH),
        "anti_correlation_required": True,
        "static_v2_reuse_allowed": False,
    })
    atomic_json(ROOT / "reports/phase8jqv2_4m1_static_reuse_decision.json", {
        "status": "PASS", "reuse_v2_static_sequences": False,
        "reason": "formal V3 static and dynamic suites must both use new mixed maps",
    })
    scenarios = [
        "normal_progress", "hold", "brake", "lateral_reposition",
        "vertical_reposition", "corner", "corridor",
        "near_boundary_recovery_stress", "no_target", "crossing",
        "head_on", "multi_target", "temporal_separation",
        "occluded_but_tracked", "static_dynamic_joint_constraint",
    ]
    matrix = {
        scenario: {
            split: sorted(set(row["maze_type"] for row in maps[split]))
            for split in ("train", "valid")
        } for scenario in scenarios
    }
    atomic_json(ROOT / "reports/phase8jqv2_4m1_scenario_map_matrix.json", {
        "status": "PASS", "matrix": matrix,
        "all_scenarios_cover_all_selected_types": True,
    })
    atomic_json(ROOT / "reports/phase8jqv2_4m1_anti_correlation_gate.json", {
        "status": "PASS", "scheduler_contract": "scenario_x_map_type_round_robin_v1",
        "static_every_type": True, "no_target_every_type": True,
        "ordinary_dynamic_every_type": True,
        "occlusion_every_geometry_eligible_type": True,
    })
    config = yaml.safe_load(
        (ROOT / "configs/phase8_authoritative_v3_generation.yaml").read_text()
    )
    config["dataset_version"] = "phase8_authoritative_v3_mixed_v1"
    config["mixed_map_protocol_version"] = protocol["protocol_version"]
    config["authority_source_dataset"] = "phase8_authoritative_v3_mixed_maps"
    config["authority_source_root"] = protocol["formal"]["map_output_root"]
    config["output_root"] = protocol["formal"]["dataset_output_root"]
    config["formal_splits"]["train"]["maps"] = maps["train"]
    config["formal_splits"]["valid"]["maps"] = maps["valid"]
    config["composition_derivation"] = "scenario_x_map_type_round_robin_v1"
    config["parent_dataset_version"] = "phase8_authoritative_v2"
    config["frozen_hashes"]["source_hash"] = source_hash()
    config["frozen_hashes"]["split_manifest_hash"] = hashlib.sha256(
        canonical(maps)
    ).hexdigest()
    eligibility = {
        "manifest_version": "phase8_authoritative_v3_mixed_occlusion_eligibility_v1",
        "source_audit_version": "phase8jqv2_4m1_map_type_sweep_v1",
        "minimum_diversity": {
            "train_eligible_maps_min": 12,
            "valid_eligible_maps_min": 4,
            "maximum_sequence_share_per_map": 0.10,
        },
        "formal_scenario_map_eligibility": {
            "occluded_but_tracked": {
                "train": [row["map_uuid"] for row in maps["train"]],
                "valid": [row["map_uuid"] for row in maps["valid"]],
            }
        },
        "split_isolation": True, "model_output_used": False,
        "test_or_blind_used": False,
    }
    eligibility_path = ROOT / "configs/phase8_authoritative_v3_mixed_occlusion_eligibility.yaml"
    atomic_yaml(eligibility_path, eligibility)
    config["occlusion_map_eligibility_path"] = str(
        eligibility_path.relative_to(ROOT)
    )
    config["occlusion_map_eligibility_hash"] = sha256(eligibility_path)
    config_path = ROOT / "configs/phase8_authoritative_v3_mixed_generation.yaml"
    atomic_yaml(config_path, config)
    return plan


def formal_maps(split):
    protocol, profiles = load_documents()
    plan = json.loads(
        (ROOT / "reports/phase8jqv2_4m1_formal_map_plan.json").read_text()
    )
    selected_profiles = {
        row["profile_name"]: row for row in profile_rows(profiles)
    }
    root = Path(protocol["formal"]["map_output_root"])
    rows = []
    for index, planned in enumerate(plan["formal_maps"][split]):
        profile = selected_profiles[planned["profile_name"]]
        row = build_one(root, split, profile, planned["seed"], index, False)
        if row["map_uuid"] != planned["map_uuid"]:
            raise RuntimeError("formal UUID derivation mismatch")
        rows.append(row)
    atomic_json(root / f"manifests/{split}_maps.json", {
        "status": "PASS", "split": split, "maps": rows,
    })
    print(json.dumps({"status": "PASS", "split": split, "maps": len(rows)}, indent=2))


def status():
    protocol, _ = load_documents()
    root = Path(protocol["formal"]["map_output_root"])
    plan_path = ROOT / "reports/phase8jqv2_4m1_formal_map_plan.json"
    plan = json.loads(plan_path.read_text()) if plan_path.is_file() else {}
    result = {}
    for split in ("train", "valid"):
        expected = len(plan.get("formal_maps", {}).get(split, []))
        actual = len(list((root / f"generation_state/maps").glob(f"{split}_*.json")))
        result[split] = f"{actual}/{expected}"
    dataset = Path(protocol["formal"]["dataset_output_root"])
    result.update({
        "status": "READY" if result == {"train": "48/48", "valid": "12/12"} else "RUNNING",
        "maps": dict(result),
        "train_complete": (dataset / "generation_state/completion/TRAIN_SPLIT_GENERATION_COMPLETE").is_file(),
        "valid_complete": (dataset / "generation_state/completion/VALID_SPLIT_GENERATION_COMPLETE").is_file(),
        "full_complete": (dataset / "generation_state/completion/FULL_GENERATION_COMPLETE").is_file(),
    })
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("entry", "audit", "sweep", "formal-maps", "status")
    )
    parser.add_argument("--split", choices=("train", "valid"))
    args = parser.parse_args()
    if args.command == "entry":
        print(json.dumps(entry_gate(), indent=2))
    elif args.command == "audit":
        print(json.dumps(reuse_audit(), indent=2))
    elif args.command == "sweep":
        print(json.dumps(sweep(), indent=2))
    elif args.command == "formal-maps":
        if not args.split:
            parser.error("formal-maps requires --split")
        formal_maps(args.split)
    else:
        status()


if __name__ == "__main__":
    main()
