#!/usr/bin/env python3
"""Freeze TF1 entry facts and evaluation split before candidate evaluation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) == value:
            return
        raise FileExistsError(f"refusing to overwrite different evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def main():
    n1 = json.loads(
        (ROOT / "reports/phase8jqv2_4n1_final_result.json").read_text()
    )
    maps = json.loads(
        (ROOT / "reports/phase8jqv2_4n1_profile_candidates.json").read_text()
    )
    retained = json.loads(
        (ROOT / "reports/phase8jqv2_4i1_retained_case_manifest.json").read_text()
    )
    expected = {
        "status": "FAIL",
        "primary_cause":
            "frozen_temporal_foreground_natural_observability",
        "gap_1_identity": "FAIL",
        "gap_2_identity": "NOT_RUN_BLOCKED_GAP1",
        "gap_3": "CONTRACT_REVIEW_REQUIRED",
        "next_allowed_phase":
            "phase8jqv2_4_temporal_foreground_observability_contract_review",
    }
    mismatches = {
        key: {"expected": value, "actual": n1.get(key)}
        for key, value in expected.items() if n1.get(key) != value
    }
    if maps["map_count"] != 30:
        mismatches["profile_maps"] = {"expected": 30, "actual": maps["map_count"]}
    if len({row["profile_name"] for row in maps["maps"]}) != 15:
        mismatches["profile_count"] = {"expected": 15}
    if len({row["maze_type"] for row in maps["maps"]}) != 5:
        mismatches["maze_types"] = {"expected": 5}
    frozen = {
        "temporal_foreground.py":
            sha(ROOT / "policy/dynamic/temporal_foreground.py"),
        "range_image_foreground.py":
            sha(ROOT / "policy/dynamic/range_image_foreground.py"),
        "track_manager.py": sha(ROOT / "policy/dynamic/track_manager.py"),
        "dynamic_perception.py":
            sha(ROOT / "policy/dynamic/dynamic_perception.py"),
        "perception_probe_v2.py":
            sha(ROOT / "authoritative_dataset/perception_probe_v2.py"),
        "occlusion_constructor_v2_1.py": sha(
            ROOT / "authoritative_dataset/occlusion_constructor_v2_1.py"
        ),
        "occlusion_constructor_v2_2.py": sha(
            ROOT / "authoritative_dataset/occlusion_constructor_v2_2.py"
        ),
        "occlusion_identity_schedule_v1.py": sha(
            ROOT / "authoritative_dataset/occlusion_identity_schedule_v1.py"
        ),
        "motion_contract_v2_1.yaml": sha(
            ROOT / "configs/authoritative_dynamic_motion_contract_v2_1.yaml"
        ),
        "authority_static_v1.py":
            sha(ROOT / "geometry_authority/static_v1.py"),
        "cuda_renderer_v1.py": sha(
            ROOT / "authoritative_dataset/cuda_renderer_v1.py"
        ),
        "sensor_traj_opt.yaml": sha(ROOT / "config/traj_opt.yaml"),
        "mixed_scene_map_profiles_v1.yaml": sha(
            ROOT / "configs/mixed_scene_map_profiles_v1.yaml"
        ),
        "mixed_scene_map_profiles_v2.yaml": sha(
            ROOT / "configs/mixed_scene_map_profiles_v2.yaml"
        ),
        "n1_profile_case_manifest": sha(
            ROOT / "reports/phase8jqv2_4n1_profile_candidates.json"
        ),
        "i1_retained_case_manifest": sha(
            ROOT / "reports/phase8jqv2_4i1_retained_case_manifest.json"
        ),
    }
    entry = {
        "status": "PASS" if not mismatches else "FAIL",
        "phase":
            "phase8jqv2_4_temporal_foreground_observability_contract_review",
        "n1_status": n1["status"],
        "primary_cause": n1["primary_cause"],
        "gap_1_identity": n1["gap_1_identity"],
        "gap_1_witnesses": 0,
        "gap_2_identity": n1["gap_2_identity"],
        "gap_3": n1["gap_3"],
        "profile_maps": maps["map_count"],
        "profile_count": len({row["profile_name"] for row in maps["maps"]}),
        "maze_types": len({row["maze_type"] for row in maps["maps"]}),
        "frozen_hashes": frozen,
        "mismatches": mismatches,
        "annex_used": False, "new_maps_generated": False,
        "formal_preflight_rerun": False,
        "formal_v3_entry_created": False,
        "formal_generation_started": False, "training_started": False,
        "test_accessed": False, "blind_accessed": False,
        "next_allowed_phase":
            "phase8jqv2_4_temporal_foreground_observability_contract_review",
    }
    write_new(ROOT / "reports/phase8jqv2_4tf1_entry_gate.json", entry)
    if mismatches:
        raise RuntimeError(f"TF1 entry mismatch: {mismatches}")
    development_maps = sorted(
        (
            {
                "map_uuid": row["map_uuid"],
                "profile_name": row["profile_name"],
                "maze_type": row["maze_type"], "seed": row["seed"],
                "authority_root": row["authority_root"],
            }
            for row in maps["maps"]
            if row["seed_namespace"] == "profile_debug"
        ),
        key=lambda row: (row["maze_type"], row["profile_name"]),
    )
    holdout_maps = sorted(
        (
            {
                "map_uuid": row["map_uuid"],
                "profile_name": row["profile_name"],
                "maze_type": row["maze_type"], "seed": row["seed"],
                "authority_root": row["authority_root"],
            }
            for row in maps["maps"]
            if row["seed_namespace"] == "profile_holdout"
        ),
        key=lambda row: (row["maze_type"], row["profile_name"]),
    )
    cases = sorted(
        (
            {
                "case_id": row["case_id"], "map_uuid": row["map_uuid"],
                "requested_gap": row["requested_gap"],
                "seed": row["map_seed"],
            }
            for row in retained["cases"]
        ),
        key=lambda row: hashlib.sha256(row["case_id"].encode()).hexdigest(),
    )
    fixed_case = (
        "natural_forest_81064183_seed831005002_gap1"
    )
    fixed = next(row for row in cases if row["case_id"] == fixed_case)
    remaining = [row for row in cases if row["case_id"] != fixed_case]
    development_cases = [fixed, *remaining[:3]]
    holdout_cases = remaining[3:]
    split = {
        "status": "FROZEN_BEFORE_CANDIDATE_EVALUATION",
        "version": "phase8jqv2_4tf1_evaluation_split_v1",
        "development_maps": development_maps,
        "sealed_holdout_maps": holdout_maps,
        "development_natural_cases": development_cases,
        "sealed_holdout_natural_cases": holdout_cases,
        "development_map_count": len(development_maps),
        "holdout_map_count": len(holdout_maps),
        "fixed_383_case_role": "development_required_control",
        "holdout_results_accessed": False,
        "formal_maps_used": False, "test_used": False, "blind_used": False,
        "new_maps_generated": False,
    }
    split["split_hash"] = hashlib.sha256(json.dumps(
        split, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    write_new(ROOT / "reports/phase8jqv2_4tf1_evaluation_split.json", split)
    print(json.dumps({
        "status": "PASS", "development_maps": len(development_maps),
        "sealed_holdout_maps": len(holdout_maps),
        "development_cases": len(development_cases),
        "sealed_holdout_cases": len(holdout_cases),
    }, indent=2))


if __name__ == "__main__":
    main()
