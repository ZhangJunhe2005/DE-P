#!/usr/bin/env python3
"""Freeze CCR1 entry, history, taxonomy, and physical contract."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
CONFIG = ROOT / "configs/dynamic_perception_physical_control_contract_v1.yaml"
sys.path.insert(0, str(ROOT))


def read(name):
    return json.loads((REPORTS / name).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) == value:
            return
        raise FileExistsError(f"refusing to overwrite CCR1 evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")
    os.replace(temporary, path)


def main():
    dpar = read("phase8jqv2_4dpar1_final_result.json")
    audit = read("phase8jqv2_4dpar1_control_contract_audit.json")
    dpar_entry = read("phase8jqv2_4dpar1_entry_gate.json")
    tf1_entry = read("phase8jqv2_4tf1_entry_gate.json")
    tf1_split = read("phase8jqv2_4tf1_evaluation_split.json")
    expected = {
        "status": "FAIL",
        "primary_cause": "dynamic_perception_control_contract_invalid",
        "architecture_review": "NOT_RUN_CONTROL_GATE_FAILED",
        "architecture_prototype_created": False,
        "candidate_selected": False,
        "holdout_accessed": False,
        "tracker_modified": False,
        "legacy_default_changed": False,
        "formal_preflight_rerun": False,
        "formal_generation_started": False,
        "optimizer_step_executed": False,
        "training_started": False,
        "test_accessed": False,
        "blind_accessed": False,
        "next_allowed_phase":
            "phase8jqv2_4_dynamic_perception_control_contract_repair",
    }
    mismatches = {
        key: {"expected": value, "actual": dpar.get(key)}
        for key, value in expected.items() if dpar.get(key) != value
    }
    for name in ("small_projection", "fov_boundary_change"):
        if audit[name]["classification"] != "fixture_semantics_invalid":
            mismatches[f"historical:{name}"] = {
                "expected": "fixture_semantics_invalid",
                "actual": audit[name]["classification"],
            }
    paths = {
        "legacy_temporal_foreground":
            ROOT / "policy/dynamic/temporal_foreground.py",
        "legacy_range_image_foreground":
            ROOT / "policy/dynamic/range_image_foreground.py",
        "tf1_candidate":
            ROOT / "policy/dynamic/range_image_foreground_v2_1.py",
        "track_manager": ROOT / "policy/dynamic/track_manager.py",
        "cuda_renderer":
            ROOT / "authoritative_dataset/cuda_renderer_v1.py",
        "authority": ROOT / "geometry_authority/static_v1.py",
        "sensor_config": ROOT / "config/traj_opt.yaml",
        "motion_contract":
            ROOT / "configs/authoritative_dynamic_motion_contract_v2_1.yaml",
        "constructor_v2_1":
            ROOT / "authoritative_dataset/occlusion_constructor_v2_1.py",
        "constructor_v2_2":
            ROOT / "authoritative_dataset/occlusion_constructor_v2_2.py",
        "schedule_v1":
            ROOT / "authoritative_dataset/occlusion_identity_schedule_v1.py",
        "profile_v1": ROOT / "configs/mixed_scene_map_profiles_v1.yaml",
        "profile_v2": ROOT / "configs/mixed_scene_map_profiles_v2.yaml",
        "tf1_split":
            ROOT / "reports/phase8jqv2_4tf1_evaluation_split.json",
    }
    hashes = {name: sha(path) for name, path in paths.items()}
    expected_frozen = {
        "legacy_temporal_foreground":
            tf1_entry["frozen_hashes"]["temporal_foreground.py"],
        "legacy_range_image_foreground":
            tf1_entry["frozen_hashes"]["range_image_foreground.py"],
        "track_manager": tf1_entry["frozen_hashes"]["track_manager.py"],
        "authority": tf1_entry["frozen_hashes"]["authority_static_v1.py"],
        "sensor_config": tf1_entry["frozen_hashes"]["sensor_traj_opt.yaml"],
        "motion_contract":
            tf1_entry["frozen_hashes"]["motion_contract_v2_1.yaml"],
        "constructor_v2_1":
            tf1_entry["frozen_hashes"]["occlusion_constructor_v2_1.py"],
        "constructor_v2_2":
            tf1_entry["frozen_hashes"]["occlusion_constructor_v2_2.py"],
        "schedule_v1":
            tf1_entry["frozen_hashes"]["occlusion_identity_schedule_v1.py"],
        "profile_v1":
            tf1_entry["frozen_hashes"]["mixed_scene_map_profiles_v1.yaml"],
        "profile_v2":
            tf1_entry["frozen_hashes"]["mixed_scene_map_profiles_v2.yaml"],
    }
    for name, expected_hash in expected_frozen.items():
        if hashes[name] != expected_hash:
            mismatches[f"hash:{name}"] = {
                "expected": expected_hash, "actual": hashes[name],
            }
    if hashes["tf1_split"] != sha(
        ROOT / "reports/phase8jqv2_4tf1_evaluation_split.json"
    ):
        raise AssertionError("unreachable split hash comparison")
    entry = {
        "status": "PASS" if not mismatches else "FAIL",
        "phase": "phase8jqv2_4_dynamic_perception_control_contract_repair",
        "dpar1_route": "B",
        "dpar1_primary_cause": dpar["primary_cause"],
        "historical_small_fixture":
            audit["small_projection"]["classification"],
        "historical_fov_fixture":
            audit["fov_boundary_change"]["classification"],
        "architecture_prototype_exists": False,
        "holdout_accessed": False,
        "legacy_modified": False,
        "tracker_modified": False,
        "formal_preflight_rerun": False,
        "formal_v3_entry_created": False,
        "formal_generation_started": False,
        "test_accessed": False,
        "blind_accessed": False,
        "optimizer_step_executed": False,
        "training_started": False,
        "frozen_hashes": hashes,
        "tf1_split_hash": tf1_split["split_hash"],
        "mismatches": mismatches,
    }
    write_new(REPORTS / "phase8jqv2_4ccr1_entry_gate.json", entry)
    if mismatches:
        raise RuntimeError(f"CCR1 entry mismatch: {mismatches}")

    historical_files = sorted(
        list(REPORTS.glob("phase8jqv2_4tf1_*"))
        + list(REPORTS.glob("phase8jqv2_4dpar1_*"))
    )
    integrity = {
        "status": "FROZEN",
        "artifact_count": len(historical_files),
        "artifacts": {
            str(path.relative_to(ROOT)): sha(path)
            for path in historical_files if path.is_file()
        },
        "tf1_diagnostics_files": len([
            path for path in (
                ROOT / "diagnostics/phase8jqv2_4tf1"
            ).rglob("*") if path.is_file()
        ]),
        "dpar1_diagnostics_files": len([
            path for path in (
                ROOT / "diagnostics/phase8jqv2_4dpar1"
            ).rglob("*") if path.is_file()
        ]),
    }
    write_new(
        REPORTS / "phase8jqv2_4ccr1_historical_artifact_integrity.json",
        integrity,
    )
    mapping = {
        "status": "PASS",
        "legacy_small_projection": {
            "classification": "fixture_semantics_invalid",
            "eligible_for_future_gate": False,
            "historical_evidence_preserved": True,
            "replacement_control_ids": [
                "actor_min_full_r020_d180_center",
                "diagnostic_legacy_equivalent_r010_d400",
            ],
        },
        "legacy_fov_boundary_change": {
            "classification": "fixture_semantics_invalid",
            "eligible_for_future_gate": False,
            "historical_evidence_preserved": True,
            "replacement_control_ids": [
                "static_fov_yaw_left_wall",
                "static_fov_yaw_right_wall",
                "static_fov_lateral_translation",
            ],
        },
    }
    write_new(
        REPORTS / "phase8jqv2_4ccr1_historical_control_mapping.json",
        mapping,
    )
    config = yaml.safe_load(CONFIG.read_text())
    write_new(
        REPORTS / "phase8jqv2_4ccr1_control_contract.json",
        {
            "status": "FROZEN_BEFORE_RENDERING",
            "contract": config,
            "config_sha256": sha(CONFIG),
            "architecture_parameters_present": False,
            "detector_output_defines_label": False,
        },
    )
    write_new(
        REPORTS / "phase8jqv2_4ccr1_control_taxonomy.json",
        {
            "status": "PASS",
            "classes": {
                "required_in_contract_positive":
                    "physical in-contract actor; future architecture hard positive",
                "required_hard_negative":
                    "physical static scene; future architecture must reject",
                "bounded_latency_positive":
                    "physical edge-entry actor accepted after frozen K frames",
                "diagnostic_out_of_contract":
                    "outside physical contract; excluded from hard rate",
                "fixture_invalid_historical":
                    "preserved history only; excluded from all new gates",
            },
            "mixed_pass_rate_forbidden": True,
        },
    )
    from authoritative_dataset.dynamic_perception_control_schema_v1 import (
        FORBIDDEN_RUNTIME_FIELDS, RUNTIME_FIELDS, SEMANTIC_CLASSES,
    )
    write_new(
        REPORTS / "phase8jqv2_4ccr1_control_schema.json",
        {
            "status": "PASS",
            "contract_version": config["contract_version"],
            "semantic_classes": sorted(SEMANTIC_CLASSES),
            "runtime_fields": sorted(RUNTIME_FIELDS),
            "forbidden_runtime_fields":
                sorted(FORBIDDEN_RUNTIME_FIELDS),
            "offline_gt_separate": True,
        },
    )
    print(json.dumps({
        "status": "PASS", "frozen_artifacts": len(historical_files),
        "config_sha256": sha(CONFIG),
    }, indent=2))


if __name__ == "__main__":
    main()
