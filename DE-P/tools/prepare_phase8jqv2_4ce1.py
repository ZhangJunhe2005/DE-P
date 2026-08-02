#!/usr/bin/env python3
"""Validate CE1 entry and freeze all RR1/historical artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DIAGNOSTICS = ROOT / "diagnostics/phase8jqv2_4ce1"
V1 = ROOT / "data/phase8_natural_representation_audit_v1"
EXPECTED_V1 = "16b67a873a21f668de55d2d32be425af4f208da07bb0262162cbc42fff964eba"
EXPECTED_CCR1 = (
    "1a699f9caeb6b1139f4f31bc17efc3ca0f9cb09f1c29bc1cf38860f5295f5efe"
)


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_digest(root):
    root = Path(root)
    rows = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        rows.append({
            "path": str(path.relative_to(root)),
            "sha256": sha(path), "bytes": path.stat().st_size,
        })
    return {
        "path": str(root), "file_count": len(rows),
        "bytes": sum(row["bytes"] for row in rows),
        "tree_hash": hashlib.sha256(json.dumps(
            rows, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest(),
        "files": rows,
    }


def write_new(path, value):
    path = Path(path)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() == payload:
            return
        raise FileExistsError(f"refusing to overwrite CE1 evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload)
    os.replace(temporary, path)


def main():
    rr1 = load(REPORTS / "phase8jqv2_4rr1_final_result.json")
    rr1_manifest_report = load(
        REPORTS / "phase8jqv2_4rr1_natural_corpus_manifest.json"
    )
    rr1_integrity = load(
        REPORTS / "phase8jqv2_4rr1_natural_corpus_integrity.json"
    )
    manifest = load(V1 / "manifest.json")
    ccr1 = load(
        ROOT / "data/phase8_dynamic_perception_controls_v1/manifest.json"
    )
    attempts_path = V1 / "generation_attempts.json"
    attempts = load(attempts_path)
    staging_roots = sorted(
        ROOT.glob("data/.phase8_natural_representation_audit_v1.staging-*")
    )
    mismatches = {}
    expected_rr1 = {
        "status": "FAIL", "route": "B",
        "primary_cause":
            "natural_gap1_representation_corpus_insufficient",
        "representation_review": "FAIL",
        "natural_corpus_maps": 3, "natural_corpus_types": 2,
        "natural_corpus_seeds": 3, "required_maps": 6,
        "required_types": 3, "representation_candidates_created": False,
        "candidate_frozen": False, "natural_holdout_accessed": False,
        "ccr1_holdout_accessed": False,
        "tf1_sealed_holdout_accessed": False,
        "tracker_integration_executed": False,
        "formal_preflight_rerun": False,
        "formal_v3_entry_created": False,
        "formal_v3_generation_started": False,
        "production_test_accessed": False, "blind_accessed": False,
        "optimizer_step_executed": False, "training_started": False,
        "next_allowed_phase":
            "phase8jqv2_4_natural_gap1_representation_corpus_expansion",
    }
    for key, expected in expected_rr1.items():
        if rr1.get(key) != expected:
            mismatches[f"rr1:{key}"] = {
                "expected": expected, "actual": rr1.get(key)
            }
    if manifest.get("root_manifest_hash") != EXPECTED_V1:
        mismatches["v1_root_hash"] = manifest.get("root_manifest_hash")
    if rr1_manifest_report.get("root_manifest_hash") != EXPECTED_V1:
        mismatches["v1_report_hash"] = rr1_manifest_report.get(
            "root_manifest_hash"
        )
    if (
        rr1_integrity.get("accepted_case_count"),
        rr1_integrity.get("independent_map_count"),
        rr1_integrity.get("maze_type_count"),
        rr1_integrity.get("trajectory_seed_count"),
    ) != (3, 3, 2, 3):
        mismatches["v1_counts"] = "not 3 cases/maps, 2 types, 3 seeds"
    if len(attempts) != 58:
        mismatches["attempt_count"] = len(attempts)
    if not staging_roots:
        mismatches["rr1_staging"] = "missing"
    if ccr1.get("manifest_hash") != EXPECTED_CCR1:
        mismatches["ccr1_manifest_hash"] = ccr1.get("manifest_hash")
    for forbidden in (
        ROOT / "policy/dynamic/dynamic_representation_v1.py",
        ROOT / "tools/evaluate_dynamic_representation_v1.py",
    ):
        if forbidden.exists():
            mismatches[f"forbidden:{forbidden.name}"] = "exists"
    if manifest.get("detector_output_used_for_selection"):
        mismatches["detector_selection"] = True
    for key in (
        "annex_used", "new_maps_generated", "tf1_sealed_holdout_accessed",
        "production_test_accessed", "blind_accessed",
        "formal_generation_started", "training_started",
    ):
        if manifest.get(key):
            mismatches[f"manifest:{key}"] = True

    v1_tree = tree_digest(V1)
    staging = [tree_digest(path) for path in staging_roots]
    accepted = list(manifest["cases"]) + list(
        manifest["unused_valid_cases"]
    )
    case_rows = []
    for row in accepted:
        case_root = V1 / "cases" / row["case_id"]
        case_rows.append({
            "case_id": row["case_id"],
            "case_hash": row["case_hash"],
            "case_json_sha256": sha(case_root / "case.json"),
            "artifact_hashes": row["artifact_hashes"],
            "map_uuid": row["map_uuid"], "map_seed": row["map_seed"],
            "maze_type": row["maze_type"],
            "camera_motion": row["camera_motion"],
            "authority_hash": row["authority_hash"],
        })
    frozen_paths = {
        "constructor_v2_1":
            ROOT / "authoritative_dataset/occlusion_constructor_v2_1.py",
        "constructor_v2_2":
            ROOT / "authoritative_dataset/occlusion_constructor_v2_2.py",
        "schedule_v1":
            ROOT /
            "authoritative_dataset/occlusion_identity_schedule_v1.py",
        "authority": ROOT / "geometry_authority/static_v1.py",
        "renderer": ROOT / "authoritative_dataset/cuda_renderer_v1.py",
        "continuous": ROOT / "authoritative_dataset/continuous_v1.py",
        "motion_contract":
            ROOT /
            "configs/authoritative_dynamic_motion_contract_v2_1.yaml",
        "sensor": ROOT / "config/traj_opt.yaml",
        "profile_v1": ROOT / "configs/mixed_scene_map_profiles_v1.yaml",
        "profile_v2": ROOT / "configs/mixed_scene_map_profiles_v2.yaml",
        "physical_control_residual_v1":
            ROOT / "policy/dynamic/physical_control_residual_v1.py",
        "track_manager": ROOT / "policy/dynamic/track_manager.py",
        "rr1_generator":
            ROOT / "tools/generate_phase8jqv2_4rr1_corpus.py",
        "rr1_validator":
            ROOT / "tools/validate_phase8jqv2_4rr1_corpus.py",
        "rr1_attempts": attempts_path,
    }
    frozen_hashes = {key: sha(path) for key, path in frozen_paths.items()}
    if frozen_hashes["rr1_generator"] != manifest["generator_hash"]:
        mismatches["rr1_generator_hash"] = {
            "manifest": manifest["generator_hash"],
            "current": frozen_hashes["rr1_generator"],
        }

    write_new(REPORTS / "phase8jqv2_4ce1_entry_gate.json", {
        "status": "PASS" if not mismatches else "FAIL",
        "phase":
            "phase8jqv2_4_natural_gap1_representation_corpus_expansion",
        "rr1_route": "B", "v1_root_manifest_hash": EXPECTED_V1,
        "accepted_case_count": 3, "map_count": 3,
        "maze_type_count": 2, "seed_count": 3,
        "attempt_count": 58, "representation_candidates_exist": False,
        "holdout_accessed": False, "detector_used_for_selection": False,
        "tracker_executed": False, "formal_executed": False,
        "production_test_accessed": False, "blind_accessed": False,
        "training_started": False, "mismatches": mismatches,
    })
    write_new(REPORTS / "phase8jqv2_4ce1_rr1_artifact_integrity.json", {
        "status": "PASS" if not mismatches else "FAIL",
        "v1_root_manifest_hash": EXPECTED_V1,
        "v1_tree": v1_tree,
        "accepted_cases": case_rows,
        "attempt_count": len(attempts),
        "attempt_log_sha256": sha(attempts_path),
        "staging": staging,
        "staging_preserved": bool(staging),
        "read_only_preservation": True,
    })
    write_new(REPORTS / "phase8jqv2_4ce1_frozen_hashes.json", {
        "status": "PASS" if not mismatches else "FAIL",
        "frozen_hashes": frozen_hashes,
        "v1_tree_hash": v1_tree["tree_hash"],
        "staging_tree_hashes": {
            row["path"]: row["tree_hash"] for row in staging
        },
    })
    if mismatches:
        raise RuntimeError(f"CE1 entry mismatch: {mismatches}")
    print(json.dumps({
        "status": "PASS", "v1_hash": EXPECTED_V1,
        "attempts": len(attempts), "staging": [{
            "path": row["path"], "bytes": row["bytes"],
            "tree_hash": row["tree_hash"],
        } for row in staging],
    }, indent=2))


if __name__ == "__main__":
    main()
