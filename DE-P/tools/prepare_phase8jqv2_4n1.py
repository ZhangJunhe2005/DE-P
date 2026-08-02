#!/usr/bin/env python3
"""Validate the N1 entry and freeze its read-only implementation baselines."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_new(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    i1 = json.loads(
        (ROOT / "reports/phase8jqv2_4i1_final_result.json").read_text()
    )
    manifest = json.loads(
        (ROOT / "reports/phase8jqv2_4i1_retained_case_manifest.json").read_text()
    )
    expected = {
        "status": "FAIL",
        "primary_cause": "natural_occlusion_detection_support",
        "gap_1": "FAIL",
        "gap_2": "NOT_RUN_BLOCKED_STAGE1",
        "gap_3": "NOT_RUN_BLOCKED_STAGE1",
        "probe_lifecycle": "PASS",
        "next_allowed_phase": "phase8jqv2_4_natural_map_profile_repair",
    }
    mismatches = {
        key: {"expected": value, "actual": i1.get(key)}
        for key, value in expected.items() if i1.get(key) != value
    }
    frozen = {
        "occlusion_constructor_v2_1.py": sha(
            ROOT / "authoritative_dataset/occlusion_constructor_v2_1.py"
        ),
        "occlusion_constructor_v2_2.py": sha(
            ROOT / "authoritative_dataset/occlusion_constructor_v2_2.py"
        ),
        "occlusion_identity_schedule_v1.py": sha(
            ROOT / "authoritative_dataset/occlusion_identity_schedule_v1.py"
        ),
        "temporal_foreground.py": sha(
            ROOT / "policy/dynamic/temporal_foreground.py"
        ),
        "range_image_foreground.py": sha(
            ROOT / "policy/dynamic/range_image_foreground.py"
        ),
        "image_foreground_components.py": sha(
            ROOT / "policy/dynamic/image_foreground_components.py"
        ),
        "track_manager.py": sha(ROOT / "policy/dynamic/track_manager.py"),
        "dynamic_perception.py": sha(
            ROOT / "policy/dynamic/dynamic_perception.py"
        ),
        "perception_probe_v2.py": sha(
            ROOT / "authoritative_dataset/perception_probe_v2.py"
        ),
        "dynamic_motion_v2.py": sha(
            ROOT / "authoritative_dataset/dynamic_motion_v2.py"
        ),
        "motion_contract_v2_1.yaml": sha(
            ROOT / "configs/authoritative_dynamic_motion_contract_v2_1.yaml"
        ),
        "authority_static_v1.py": sha(
            ROOT / "geometry_authority/static_v1.py"
        ),
        "cuda_renderer_v1.py": sha(
            ROOT / "authoritative_dataset/cuda_renderer_v1.py"
        ),
        "traj_opt.yaml": sha(ROOT / "config/traj_opt.yaml"),
        "mixed_scene_map_profiles_v1.yaml": sha(
            ROOT / "configs/mixed_scene_map_profiles_v1.yaml"
        ),
    }
    expected_hashes = {
        "occlusion_constructor_v2_1.py":
            "266b621a90a562f7a9056d5949838fc066c0ca6dc2c35febffd06351cb75f759",
        "occlusion_constructor_v2_2.py":
            "cedb080f09c407621c57abcba77fcd7f5076bb9e0d0e7d954b295989487f9892",
        "motion_contract_v2_1.yaml":
            "5a6bc337d73617aa42b8b30d1a33386b5cfe876acb43c73667bdf32c45b8e7a4",
        "mixed_scene_map_profiles_v1.yaml":
            "dad3e6a9fc5fcaef8a277765f39003be89eea90c7b90ff00217f93006a4395bd",
    }
    for key, expected_hash in expected_hashes.items():
        if frozen[key] != expected_hash:
            mismatches[f"hash:{key}"] = {
                "expected": expected_hash, "actual": frozen[key]
            }
    report = {
        "status": "PASS" if not mismatches else "FAIL",
        "phase": "phase8jqv2_4_natural_map_profile_repair",
        "i1_status": i1["status"],
        "primary_cause": i1["primary_cause"],
        "gap_1": i1["gap_1"],
        "gap_2": i1["gap_2"],
        "gap_3": i1["gap_3"],
        "probe_lifecycle": i1["probe_lifecycle"],
        "natural_geometry_pass_count": manifest["case_count"],
        "frozen_hashes": frozen,
        "mismatches": mismatches,
        "formal_preflight_rerun": False,
        "formal_v3_entry_created": False,
        "formal_generation_started": False,
        "training_started": False,
        "annex_used": False,
        "test_accessed": False,
        "blind_accessed": False,
        "next_allowed_phase": "phase8jqv2_4_natural_map_profile_repair",
    }
    atomic_new(ROOT / "reports/phase8jqv2_4n1_entry_gate.json", report)
    atomic_new(ROOT / "reports/phase8jqv2_4n1_problem_statement.json", {
        "status": "CONFIRMED",
        "best_gap1_continuous_candidates": 5,
        "pre_gap_visible_pixels_each_candidate": 383,
        "pre_gap_temporal_foreground_pixels_each_candidate": 0,
        "pre_gap_measurement_valid": False,
        "first_post_gap_measurement_valid": False,
        "larger_time_shifts": "continuous_motion_contract_rejection",
        "proven": [
            "pre-roll alone cannot repair the fixed trajectory",
            "visible pixels are not sufficient for a temporal foreground measurement",
            "CUDA natural visibility is not a frozen detection observability certificate",
        ],
        "not_primary_causes": [
            "actor_enters_too_late", "gap_starts_too_early",
            "TrackManager_cannot_hold_one_missing_frame",
            "v2_2_geometry_infeasible", "no_natural_occlusion",
        ],
    })
    print(json.dumps({"status": report["status"], "mismatches": mismatches}, indent=2))
    if mismatches:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
