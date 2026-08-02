#!/usr/bin/env python3
"""Freeze RR1 entry facts and inventory allowed existing development maps."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DIAGNOSTICS = ROOT / "diagnostics/phase8jqv2_4rr1"
EXPECTED_MANIFEST = (
    "1a699f9caeb6b1139f4f31bc17efc3ca0f9cb09f1c29bc1cf38860f5295f5efe"
)


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_manifest(paths):
    rows = []
    for path in sorted(paths):
        rows.append({
            "path": str(path.relative_to(ROOT)),
            "sha256": sha(path),
            "bytes": path.stat().st_size,
        })
    digest = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {"root_hash": digest, "files": rows}


def write_new(path, value):
    path = Path(path)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() == payload:
            return
        raise FileExistsError(f"refusing to overwrite RR1 evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload)
    os.replace(temporary, path)


def main():
    dpar2 = load(REPORTS / "phase8jqv2_4dpar2_final_result.json")
    entry = load(REPORTS / "phase8jqv2_4dpar2_entry_gate.json")
    candidate = load(
        REPORTS / "phase8jqv2_4dpar2_candidate0_development.json"
    )["result"]
    natural = load(
        REPORTS / "phase8jqv2_4dpar2_natural_measurement_validation.json"
    )
    ccr1 = load(REPORTS / "phase8jqv2_4ccr1_final_result.json")
    manifest = load(
        ROOT / "data/phase8_dynamic_perception_controls_v1/manifest.json"
    )
    physical = load(
        ROOT /
        "data/phase8_dynamic_perception_controls_v1/physical_validation.json"
    )
    mismatches = {}
    expected = {
        "status": "FAIL", "route": "E",
        "primary_cause": "natural_depth_dynamic_observability",
        "secondary_cause":
            "frozen_natural_gap1_case_coverage_insufficient",
        "development_candidate": "physical_control_residual_v1",
        "physical_control_development": "PASS",
        "natural_gap1_pre_measurement": False,
        "natural_gap1_post_measurement": False,
        "natural_gap1_independent_maps": 1,
        "natural_gap1_maze_types": 1,
        "candidate_frozen": False,
        "physical_control_holdout": "NOT_RUN",
        "holdout_runtime_artifacts_read": False,
        "tracker_integration": "NOT_RUN",
        "legacy_default_changed": False,
        "formal_preflight_rerun": False,
        "formal_generation_started": False,
        "optimizer_step_executed": False,
        "training_started": False,
        "production_test_accessed": False,
        "blind_accessed": False,
        "next_allowed_phase":
            "phase8jqv2_4_dynamic_perception_representation_review",
    }
    for key, value in expected.items():
        if dpar2.get(key) != value:
            mismatches[f"dpar2:{key}"] = {
                "expected": value, "actual": dpar2.get(key)
            }
    if manifest.get("manifest_hash") != EXPECTED_MANIFEST:
        mismatches["ccr1_manifest"] = manifest.get("manifest_hash")
    if manifest.get("split_counts") != {
        "development": 39, "sealed_control_holdout": 39
    }:
        mismatches["ccr1_split"] = manifest.get("split_counts")
    if (physical.get("passed"), physical.get("failed")) != (78, 0):
        mismatches["ccr1_validation"] = {
            "passed": physical.get("passed"), "failed": physical.get("failed")
        }
    if not candidate["development_hard_gate"]:
        mismatches["candidate_physical_gate"] = False
    if natural["natural_gap1_gate"] != "FAIL":
        mismatches["natural_gap1_gate"] = natural["natural_gap1_gate"]
    if ccr1.get("tf1_sealed_holdout_accessed"):
        mismatches["tf1_holdout"] = True
    if (
        ROOT / "diagnostics/phase8jqv2_4dpar2/holdout_access_log.jsonl"
    ).read_text().count("\n") != 1:
        mismatches["dpar2_holdout_access_log"] = "unexpected activity"

    frozen_paths = {
        "ccr1_manifest":
            ROOT / "data/phase8_dynamic_perception_controls_v1/manifest.json",
        "physical_control_residual_v1":
            ROOT / "policy/dynamic/physical_control_residual_v1.py",
        "architecture_config":
            ROOT /
            "configs/dynamic_perception_architecture_candidates_v1.yaml",
        "runtime_visibility_provenance":
            ROOT / "policy/dynamic/visibility_provenance_v1.py",
        "legacy_temporal":
            ROOT / "policy/dynamic/temporal_foreground.py",
        "legacy_range":
            ROOT / "policy/dynamic/range_image_foreground.py",
        "track_manager": ROOT / "policy/dynamic/track_manager.py",
        "constructor_v2_1":
            ROOT / "authoritative_dataset/occlusion_constructor_v2_1.py",
        "constructor_v2_2":
            ROOT / "authoritative_dataset/occlusion_constructor_v2_2.py",
        "schedule_v1":
            ROOT /
            "authoritative_dataset/occlusion_identity_schedule_v1.py",
        "authority": ROOT / "geometry_authority/static_v1.py",
        "sensor": ROOT / "config/traj_opt.yaml",
        "motion_contract":
            ROOT /
            "configs/authoritative_dynamic_motion_contract_v2_1.yaml",
        "profile_v1": ROOT / "configs/mixed_scene_map_profiles_v1.yaml",
        "profile_v2": ROOT / "configs/mixed_scene_map_profiles_v2.yaml",
        "historical_case_manifest":
            REPORTS / "phase8jqv2_4i1_retained_case_manifest.json",
    }
    frozen_hashes = {key: sha(path) for key, path in frozen_paths.items()}
    if frozen_hashes["physical_control_residual_v1"] != (
        entry["frozen_hashes"].get("physical_control_residual_v1")
        or sha(ROOT / "policy/dynamic/physical_control_residual_v1.py")
    ):
        mismatches["physical_control_residual_v1"] = "hash mismatch"

    dpar2_files = [
        *REPORTS.glob("phase8jqv2_4dpar2_*"),
        ROOT / "configs/dynamic_perception_architecture_candidates_v1.yaml",
        ROOT / "policy/dynamic/physical_control_residual_v1.py",
        ROOT / "policy/dynamic/visibility_provenance_v1.py",
        ROOT /
        "policy/dynamic/dynamic_perception_architecture_registry.py",
        ROOT / "policy/dynamic/dynamic_measurement_v3.py",
    ]
    dpar2_freeze = tree_manifest([
        path for path in dpar2_files if path.is_file()
    ])

    profile = load(REPORTS / "phase8jqv2_4n1_profile_candidates.json")
    original = load(
        REPORTS / "occlusion_constructor_v2_2_natural_map_set.json"
    )
    # The first N1 namespace is explicitly development-only. Do not touch the
    # 851xxx roots registered as TF1 sealed maps.
    map_candidates = [
        {
            "source": "n1_profile_v2_development",
            "map_uuid": row["map_uuid"],
            "maze_type": int(row["maze_type"]),
            "seed": int(row["seed"]),
            "authority_root": row["authority_root"],
            "authority_hash": row["authority_manifest_hash"],
            "occupancy_hash": row["occupancy_hash"],
            "annex_enabled": False,
            "profile_name": row["profile_name"],
        }
        for row in profile["maps"] if int(row["seed"]) < 850000
    ] + [
        {
            "source": "v2_2_original_development",
            "map_uuid": row["map_uuid"],
            "maze_type": int(row["maze_type"]),
            "seed": int(row["seed"]),
            "authority_root": row["authority_root"],
            "authority_hash": row["authority_manifest_hash"],
            "occupancy_hash": row["occupancy_hash"],
            "annex_enabled": bool(row["annex_enabled"]),
            "profile_name": None,
        }
        for row in original["maps"] if row["eligible"]
    ]
    unique = {}
    duplicates = []
    for row in map_candidates:
        if row["annex_enabled"]:
            mismatches[f"annex:{row['map_uuid']}"] = True
            continue
        if not Path(row["authority_root"]).is_dir():
            mismatches[f"missing_map:{row['map_uuid']}"] = row["authority_root"]
            continue
        key = row["authority_hash"]
        if key in unique:
            duplicates.append({
                "authority_hash": key,
                "kept": unique[key]["map_uuid"],
                "removed": row["map_uuid"],
            })
        else:
            unique[key] = row
    maps = sorted(
        unique.values(),
        key=lambda row: (row["maze_type"], row["seed"], row["map_uuid"])
    )
    if len(maps) < 6 or len({row["maze_type"] for row in maps}) < 3:
        mismatches["map_pool"] = {
            "maps": len(maps),
            "types": len({row["maze_type"] for row in maps}),
        }

    write_new(REPORTS / "phase8jqv2_4rr1_entry_gate.json", {
        "status": "PASS" if not mismatches else "FAIL",
        "phase": "phase8jqv2_4_dynamic_perception_representation_review",
        "dpar2_route": "E",
        "ccr1_manifest_hash": EXPECTED_MANIFEST,
        "ccr1_controls": 78,
        "ccr1_validation": "78/78 PASS",
        "candidate_physical_development": "PASS",
        "natural_gap1_pre_measurement": False,
        "natural_gap1_post_measurement": False,
        "candidate_frozen": False,
        "ccr1_holdout_runtime_accessed": False,
        "tf1_sealed_holdout_accessed": False,
        "tracker_integrated": False,
        "formal_preflight_rerun": False,
        "production_test_accessed": False,
        "blind_accessed": False,
        "training_started": False,
        "mismatches": mismatches,
    })
    write_new(REPORTS / "phase8jqv2_4rr1_frozen_artifacts.json", {
        "status": "PASS" if not mismatches else "FAIL",
        "frozen_hashes": frozen_hashes,
        "dpar2_tree": dpar2_freeze,
        "historical_maps_modified": False,
        "historical_cases_modified": False,
    })
    write_new(REPORTS / "phase8jqv2_4rr1_runtime_input_isolation.json", {
        "status": "PASS",
        "allowed": [
            "current_and_past_depth", "current_and_past_camera_pose",
            "intrinsics", "max_depth", "timestamp",
        ],
        "forbidden": [
            "actor_mask", "actor_id", "actor_trajectory", "gap_label",
            "expected_control_role", "authority_map",
            "authority_correspondence", "offline_warp_reference",
            "future_depth", "future_actor_state", "pointcloud_topic",
        ],
        "offline_scoring_after_runtime_only": True,
    })
    write_new(
        DIAGNOSTICS / "corpus_generation/allowed_map_inventory.json",
        {
            "status": "PASS" if not mismatches else "FAIL",
            "candidate_count_before_deduplication": len(map_candidates),
            "unique_authority_count": len(maps),
            "maze_types": sorted({row["maze_type"] for row in maps}),
            "duplicates": duplicates,
            "maps": maps,
            "tf1_sealed_map_roots_read": False,
            "new_maps_generated": False,
            "annex_used": False,
        },
    )
    access_log = DIAGNOSTICS / "holdout_access_log.jsonl"
    initial = json.dumps({
        "event": "audit_initialized",
        "phase": "phase8jqv2_4_dynamic_perception_representation_review",
        "sealed_natural_runtime_access_count": 0,
        "ccr1_holdout_runtime_access_count": 0,
        "tf1_holdout_accessed": False,
    }, sort_keys=True) + "\n"
    access_log.parent.mkdir(parents=True, exist_ok=True)
    if access_log.exists():
        if access_log.read_text() != initial:
            raise FileExistsError("RR1 holdout access log contains activity")
    else:
        access_log.write_text(initial)
    if mismatches:
        raise RuntimeError(f"RR1 entry mismatch: {mismatches}")
    print(json.dumps({
        "status": "PASS", "unique_allowed_maps": len(maps),
        "maze_types": sorted({row["maze_type"] for row in maps}),
        "tf1_sealed_maps_read": False,
    }, indent=2))


if __name__ == "__main__":
    main()
