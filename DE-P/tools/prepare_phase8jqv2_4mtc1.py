#!/usr/bin/env python3
"""Fail-closed MTC1 entry gate and immutable historical-artifact snapshot."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
CE1_DATA = ROOT / "data/phase8_natural_representation_audit_v1"
PARENT = ROOT.parent
PREFIX = "phase8jqv2_4mtc1_"


def load(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_digest(path: Path) -> dict:
    rows = []
    for item in sorted(value for value in path.rglob("*") if value.is_file()):
        rows.append({
            "path": str(item.relative_to(path)),
            "bytes": item.stat().st_size,
            "sha256": sha(item),
        })
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return {
        "path": str(path),
        "file_count": len(rows),
        "bytes": sum(row["bytes"] for row in rows),
        "tree_hash": hashlib.sha256(payload).hexdigest(),
    }


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


def main() -> None:
    final = load(REPORTS / "phase8jqv2_4ce1_final_result.json")
    entry = load(REPORTS / "phase8jqv2_4ce1_entry_gate.json")
    frozen = load(REPORTS / "phase8jqv2_4ce1_frozen_hashes.json")
    v1_manifest = load(CE1_DATA / "manifest.json")
    v2 = load(REPORTS / "phase8jqv2_4ce1_corpus_v2_manifest.json")
    split = load(REPORTS / "phase8jqv2_4ce1_corpus_split.json")
    expected = {
        "status": "FAIL",
        "route": "B",
        "primary_cause": "natural_gap1_map_type_capability_insufficient",
        "secondary_cause": "natural_gap1_corpus_map_count_insufficient",
        "e1_new_cases": 0,
        "e2_new_cases": 0,
        "e3_new_cases": 0,
        "combined_maps": 3,
        "combined_types": 2,
        "combined_seeds": 3,
        "observed_types": ["cave", "forest"],
        "corpus_v2_created": False,
        "split_created": False,
        "representation_candidates_created": False,
        "detector_executed": False,
        "tracker_executed": False,
        "formal_preflight_rerun": False,
        "formal_v3_entry_created": False,
        "formal_generation_started": False,
        "optimizer_step_executed": False,
        "training_started": False,
        "production_test_accessed": False,
        "blind_accessed": False,
        "next_allowed_phase":
            "phase8jqv2_4_natural_gap1_map_type_capability_review",
    }
    mismatches = {}
    for key, value in expected.items():
        if final.get(key) != value:
            mismatches[f"ce1_final:{key}"] = {
                "expected": value, "actual": final.get(key),
            }
    if entry.get("status") != "PASS":
        mismatches["ce1_entry"] = entry.get("status")
    if v2.get("exists") is not False:
        mismatches["corpus_v2"] = v2
    if split.get("split_created") is not False:
        mismatches["split"] = split
    if v1_manifest.get("root_manifest_hash") != entry.get(
        "v1_root_manifest_hash"
    ):
        mismatches["corpus_v1_manifest_hash"] = {
            "manifest": v1_manifest.get("root_manifest_hash"),
            "entry": entry.get("v1_root_manifest_hash"),
        }

    frozen_paths = {
        "constructor_v2_1":
            ROOT / "authoritative_dataset/occlusion_constructor_v2_1.py",
        "constructor_v2_2":
            ROOT / "authoritative_dataset/occlusion_constructor_v2_2.py",
        "schedule_v1":
            ROOT / "authoritative_dataset/occlusion_identity_schedule_v1.py",
        "authority": ROOT / "geometry_authority/static_v1.py",
        "renderer": ROOT / "authoritative_dataset/cuda_renderer_v1.py",
        "continuous": ROOT / "authoritative_dataset/continuous_v1.py",
        "motion_contract":
            ROOT / "configs/authoritative_dynamic_motion_contract_v2_1.yaml",
        "sensor": ROOT / "config/traj_opt.yaml",
        "profile_v1": ROOT / "configs/mixed_scene_map_profiles_v1.yaml",
        "profile_v2": ROOT / "configs/mixed_scene_map_profiles_v2.yaml",
        "simulator_maps_hpp": PARENT / "Simulator/src/include/maps.hpp",
        "simulator_maps_cpp": PARENT / "Simulator/src/src/maps.cpp",
        "simulator_dataset_generator":
            PARENT / "Simulator/src/src/dataset_generator.cpp",
        "simulator_config": PARENT / "Simulator/src/config/config.yaml",
    }
    current_hashes = {key: sha(path) for key, path in frozen_paths.items()}
    for key, value in frozen["frozen_hashes"].items():
        if key in current_hashes and current_hashes[key] != value:
            mismatches[f"frozen:{key}"] = {
                "expected": value, "actual": current_hashes[key],
            }
    v1_tree = tree_digest(CE1_DATA)
    if v1_tree["tree_hash"] != frozen["v1_tree_hash"]:
        mismatches["corpus_v1_tree_hash"] = {
            "expected": frozen["v1_tree_hash"],
            "actual": v1_tree["tree_hash"],
        }
    staging = {}
    for path, expected_hash in frozen["staging_tree_hashes"].items():
        digest = tree_digest(Path(path))
        staging[path] = digest
        if digest["tree_hash"] != expected_hash:
            mismatches[f"ce1_staging:{path}"] = {
                "expected": expected_hash, "actual": digest["tree_hash"],
            }

    status = "PASS" if not mismatches else "FAIL"
    snapshot = {
        "status": status,
        "phase": "phase8jqv2_4_natural_gap1_map_type_capability_review",
        "ce1_route": final["route"],
        "ce1_new_cases": {
            "e1": final["e1_new_cases"],
            "e2": final["e2_new_cases"],
            "e3": final["e3_new_cases"],
        },
        "corpus_v1_root_manifest_hash": v1_manifest["root_manifest_hash"],
        "corpus_v1_tree": v1_tree,
        "source_hashes": current_hashes,
        "ce1_staging": staging,
        "mismatches": mismatches,
    }
    atomic_new(REPORTS / f"{PREFIX}entry_gate.json", {
        "status": status,
        "phase": snapshot["phase"],
        "ce1_route": final["route"],
        "e1_new_cases": final["e1_new_cases"],
        "e2_new_cases": final["e2_new_cases"],
        "e3_new_cases": final["e3_new_cases"],
        "corpus_v2_created": False,
        "split_created": False,
        "detector_executed": False,
        "tracker_executed": False,
        "formal_generation_started": False,
        "production_test_accessed": False,
        "blind_accessed": False,
        "optimizer_step_executed": False,
        "training_started": False,
        "mismatches": mismatches,
    })
    atomic_new(REPORTS / f"{PREFIX}frozen_artifacts.json", snapshot)
    atomic_new(REPORTS / f"{PREFIX}historical_integrity.json", {
        "status": status,
        "rr1_corpus_v1_modified": False if not mismatches else None,
        "ce1_artifacts_modified": False if not mismatches else None,
        "original_yopo_maps_cpp_modified": False if not mismatches else None,
        "original_yopo_config_modified": False if not mismatches else None,
        "checked_hashes": current_hashes,
        "mismatches": mismatches,
    })
    print(json.dumps({
        "status": status,
        "v1_tree_hash": v1_tree["tree_hash"],
        "source_count": len(current_hashes),
        "mismatches": mismatches,
    }, indent=2))
    if mismatches:
        raise RuntimeError("MTC1 fail-closed entry mismatch")


if __name__ == "__main__":
    main()
