#!/usr/bin/env python3
"""Validate and freeze the DPAR2 entry without reading holdout artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DATASET = ROOT / "data/phase8_dynamic_perception_controls_v1"
DIAGNOSTICS = ROOT / "diagnostics/phase8jqv2_4dpar2"
EXPECTED_MANIFEST_HASH = (
    "1a699f9caeb6b1139f4f31bc17efc3ca0f9cb09f1c29bc1cf38860f5295f5efe"
)


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() == payload:
            return
        raise FileExistsError(f"refusing to overwrite DPAR2 evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload)
    os.replace(temporary, path)


def main():
    manifest = load(DATASET / "manifest.json")
    physical = load(DATASET / "physical_validation.json")
    ccr1 = load(REPORTS / "phase8jqv2_4ccr1_final_result.json")
    frozen_paths = {
        "physical_manifest": DATASET / "manifest.json",
        "physical_validation": DATASET / "physical_validation.json",
        "ccr1_generator":
            ROOT / "tools/generate_dynamic_perception_controls_v1.py",
        "ccr1_validator":
            ROOT / "tools/validate_dynamic_perception_controls_v1.py",
        "ccr1_warp_reference":
            ROOT / "authoritative_dataset/warp_visibility_reference_v1.py",
        "legacy_temporal":
            ROOT / "policy/dynamic/temporal_foreground.py",
        "legacy_range":
            ROOT / "policy/dynamic/range_image_foreground.py",
        "tf1_candidate":
            ROOT / "policy/dynamic/range_image_foreground_v2_1.py",
        "tf1_registry":
            ROOT / "policy/dynamic/foreground_contract_registry.py",
        "tf1_config":
            ROOT / "configs/temporal_foreground_contract_v2_1_candidates.yaml",
        "track_manager": ROOT / "policy/dynamic/track_manager.py",
        "motion_contract":
            ROOT / "configs/authoritative_dynamic_motion_contract_v2_1.yaml",
        "constructor_v2_1":
            ROOT / "authoritative_dataset/occlusion_constructor_v2_1.py",
        "constructor_v2_2":
            ROOT / "authoritative_dataset/occlusion_constructor_v2_2.py",
        "schedule":
            ROOT / "authoritative_dataset/occlusion_identity_schedule_v1.py",
        "profile_v1": ROOT / "configs/mixed_scene_map_profiles_v1.yaml",
        "profile_v2": ROOT / "configs/mixed_scene_map_profiles_v2.yaml",
        "authority": ROOT / "geometry_authority/static_v1.py",
        "sensor": ROOT / "config/traj_opt.yaml",
    }
    frozen_hashes = {name: sha(path) for name, path in frozen_paths.items()}
    failed_roots = {
        "sparse_plane":
            ROOT / "data/phase8_dynamic_perception_controls_v1_failed_sparse_plane_v1",
        "combined_motion":
            ROOT / "data/phase8_dynamic_perception_controls_v1_failed_combined_motion_v1",
    }
    mismatches = {}
    expected_ccr1 = {
        "status": "PASS",
        "physical_control_contract": "PASS",
        "control_count": 78,
        "holdout_architecture_evaluated": False,
        "tf1_sealed_holdout_accessed": False,
        "formal_preflight_rerun": False,
        "formal_generation_started": False,
        "optimizer_step_executed": False,
        "training_started": False,
        "next_allowed_phase":
            "phase8jqv2_4_dynamic_perception_architecture_review_rerun",
    }
    for key, expected in expected_ccr1.items():
        if ccr1.get(key) != expected:
            mismatches[f"ccr1:{key}"] = {
                "expected": expected, "actual": ccr1.get(key),
            }
    expected_manifest = {
        "manifest_hash": EXPECTED_MANIFEST_HASH,
        "split_counts": {
            "development": 39, "sealed_control_holdout": 39,
        },
        "architecture_executed": False,
        "detector_executed": False,
        "tf1_holdout_accessed": False,
        "test_accessed": False,
        "blind_accessed": False,
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            mismatches[f"manifest:{key}"] = {
                "expected": expected, "actual": manifest.get(key),
            }
    if physical.get("status") != "PASS" or (
        physical.get("passed"), physical.get("failed")
    ) != (78, 0):
        mismatches["physical_validation"] = {
            "expected": "78/78 PASS",
            "actual": {
                "status": physical.get("status"),
                "passed": physical.get("passed"),
                "failed": physical.get("failed"),
            },
        }
    if physical.get("architecture_executed") or physical.get(
        "detector_executed"
    ):
        mismatches["physical_runtime_state"] = "architecture/detector executed"
    for name, path in failed_roots.items():
        if not path.is_dir():
            mismatches[f"failed_root:{name}"] = "missing"

    entry = {
        "status": "PASS" if not mismatches else "FAIL",
        "phase":
            "phase8jqv2_4_dynamic_perception_architecture_review_rerun",
        "ccr1_route": "A",
        "physical_control_manifest_hash": manifest["manifest_hash"],
        "control_count": 78,
        "development_count": 39,
        "sealed_control_holdout_count": 39,
        "physical_validation": {
            "status": physical["status"],
            "passed": physical["passed"],
            "failed": physical["failed"],
        },
        "holdout_runtime_artifacts_read_pre_freeze": False,
        "holdout_runtime_access_count_pre_freeze": 0,
        "tf1_sealed_holdout_accessed": False,
        "historical_failed_roots": {
            name: str(path.relative_to(ROOT))
            for name, path in failed_roots.items()
        },
        "frozen_hashes": frozen_hashes,
        "mismatches": mismatches,
    }
    write_new(REPORTS / "phase8jqv2_4dpar2_entry_gate.json", entry)
    if mismatches:
        raise RuntimeError(f"DPAR2 entry mismatch: {mismatches}")

    write_new(REPORTS / "phase8jqv2_4dpar2_control_integrity.json", {
        "status": "PASS",
        "suite_modified": False,
        "manifest_hash": manifest["manifest_hash"],
        "manifest_file_sha256": frozen_hashes["physical_manifest"],
        "physical_validation_sha256": frozen_hashes["physical_validation"],
        "control_count": 78,
        "physical_validation": "78/78 PASS",
        "failed_control_evidence_preserved": True,
        "new_maps_generated": False,
        "annex_used": False,
    })
    write_new(REPORTS / "phase8jqv2_4dpar2_evaluation_split.json", {
        "status": "FROZEN_BEFORE_ARCHITECTURE_EVALUATION",
        "development_count": 39,
        "sealed_control_holdout_count": 39,
        "development_runtime_access_allowed": True,
        "holdout_pre_freeze_allowed_fields": [
            "split_count", "root_hash", "file_hashes", "freeze_metadata",
        ],
        "holdout_runtime_access_before_freeze": False,
        "tf1_sealed_holdout_accessed": False,
        "production_test_accessed": False,
        "blind_accessed": False,
    })
    write_new(REPORTS / "phase8jqv2_4dpar2_runtime_input_isolation.json", {
        "status": "FROZEN_BEFORE_EVALUATION",
        "allowed_runtime_inputs": [
            "current_depth", "past_depth", "current_camera_pose",
            "past_camera_poses", "intrinsics", "max_depth", "timestamp",
            "existing_track_states_reacquisition_only",
        ],
        "forbidden_runtime_inputs": [
            "actor_trajectory", "actor_mask", "actor_id",
            "authority_correspondence", "expected_role",
            "control_semantic_class", "reference_provenance_masks",
            "future_frame", "future_actor_state", "holdout_label",
        ],
        "offline_validation_after_runtime_only": True,
        "runtime_gt_used": False,
        "future_used": False,
    })
    access_log = DIAGNOSTICS / "holdout_access_log.jsonl"
    access_log.parent.mkdir(parents=True, exist_ok=True)
    initial = json.dumps({
        "event": "audit_initialized",
        "phase": entry["phase"],
        "pre_freeze_runtime_access_count": 0,
        "holdout_runtime_artifacts_read": False,
    }, sort_keys=True) + "\n"
    if access_log.exists():
        if access_log.read_text() != initial:
            raise FileExistsError("holdout access log already contains activity")
    else:
        access_log.write_text(initial)
    print(json.dumps({
        "status": "PASS",
        "manifest_hash": manifest["manifest_hash"],
        "development": 39,
        "holdout": 39,
        "pre_freeze_holdout_runtime_access": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
