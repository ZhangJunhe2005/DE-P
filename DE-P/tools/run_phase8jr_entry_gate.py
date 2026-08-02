#!/usr/bin/env python3
"""Fail-closed Phase 8J-R entry and frozen-baseline verification."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SIMULATOR = ROOT.parent / "Simulator"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    reports = ROOT / "reports"
    phase8h = load(reports / "phase8h_final_perception_gate.json")
    phase8i = load(reports / "phase8i_final_result.json")
    phase8i_cache = load(reports / "phase8i_estimated_cache_validation.json")
    phase8j = load(reports / "phase8j_final_result.json")
    phase8j_gate = load(reports / "phase8j_coverage_gate.json")
    prior_entry = load(reports / "phase8j_entry_gate.json")
    blind_lock = load(reports / ".phase8h_blind_once.lock")

    perception_sources = [
        ROOT / "policy/dynamic/dynamic_perception.py",
        ROOT / "policy/dynamic/range_image_foreground.py",
        ROOT / "policy/dynamic/track_manager.py",
        ROOT / "policy/dynamic/types.py",
        ROOT / "policy/dynamic/instance_evaluation.py",
        ROOT / "tools/evaluate_phase8g_instance_perception.py",
        ROOT / "tools/generate_phase8c_scenario_matrix.py",
        ROOT / "tools/generate_phase8g_scenario_matrix.py",
        ROOT / "tools/run_phase8c_formal_recording.py",
        ROOT / "tools/run_phase8g_instance_recording.py",
        ROOT / "tools/phase8h_blind_protocol.py",
        SIMULATOR / "src/include/dynamic_actor.hpp",
        SIMULATOR / "src/src/dynamic_actor.cpp",
        SIMULATOR / "src/src/test_simulator_cuda.cpp",
    ]
    current_frozen = {
        "perception_implementation_sha256": aggregate_sha256(perception_sources),
        "perception_config_sha256": sha256(ROOT / "config/traj_opt.yaml"),
        "evaluator_sha256": sha256(
            ROOT / "tools/evaluate_phase8g_instance_perception.py"
        ),
        "simulator_binary_sha256": sha256(
            SIMULATOR / "devel/lib/sensor_simulator/sensor_simulator_cuda"
        ),
    }
    expected_frozen = prior_entry["frozen_hashes"]
    frozen_matches = {
        key: current_frozen[key] == expected_frozen[key]
        for key in current_frozen
    }

    cache_index = Path(phase8i_cache["index"])
    manifest = ROOT / "data/phase8_dynamic_production/dataset_manifest.yaml"
    static_catalog = ROOT / "configs/static_map_catalog.yaml"
    dynamic_catalog = ROOT / "data/phase8_dynamic_production/map_catalog.yaml"
    risk_manifest = ROOT / "diagnostics/phase8c_production_risk_set_manifest.json"
    current_data_hashes = {
        "production_dataset_manifest_sha256": sha256(manifest),
        "static_map_catalog_sha256": sha256(static_catalog),
        "dynamic_map_catalog_sha256": sha256(dynamic_catalog),
        "risk_set_manifest_sha256": sha256(risk_manifest),
    }
    expected_data_hashes = prior_entry["data_hashes"]
    data_matches = {
        key: current_data_hashes[key] == expected_data_hashes[key]
        for key in current_data_hashes
    }

    representative = phase8j_gate["representative"]
    c_k3_checkpoint = Path(representative["checkpoint"])
    matrix = ROOT / "configs/phase8j_coverage_matrix.yaml"
    three_seed_results = [
        load(
            ROOT / "runs/phase8j_coverage"
            / f"C_k3_seed{seed}_e3/result.json"
        )
        for seed in (8511, 8512, 8513)
    ]
    failures = []
    required_checks = {
        "phase8h_perception_gate_pass": phase8h.get("status") == "PASS",
        "phase8h_blind_run_count_one": (
            phase8h.get("phase8h_blind_run_count") == 1
            and blind_lock.get("phase8h_blind_run_count") == 1
        ),
        "phase8i_route_c": (
            phase8i.get("status") == "PASS"
            and phase8i.get("route") == "C"
            and phase8i.get("perception_frozen") is True
        ),
        "phase8j_failed_coverage": (
            phase8j.get("status") == "FAIL"
            and phase8j.get("coverage_ready") is False
            and phase8j.get("score_stage_executed") is False
        ),
        "production_test_unused": (
            phase8i.get("production_test_used") is False
            and phase8j.get("production_test_used") is False
        ),
        "perception_frozen_hashes_match": all(frozen_matches.values()),
        "cache_index_hash_matches": (
            sha256(cache_index) == phase8i_cache["index_sha256"]
            == prior_entry["phase8i_artifacts"]["cache_index_sha256"]
        ),
        "cache_is_range_image_hybrid": (
            phase8i_cache.get("effective_foreground_mode") == "range_image_hybrid"
        ),
        "data_hashes_match": all(data_matches.values()),
        "c_k3_checkpoint_hash_matches": (
            sha256(c_k3_checkpoint) == representative["checkpoint_sha256"]
        ),
        "score_branch_hash_unchanged": all(
            result.get("score_parameter_hash_unchanged") is True
            for result in three_seed_results
        ),
    }
    failures.extend(name for name, passed in required_checks.items() if not passed)

    entry = {
        "status": "PASS" if not failures else "FAIL",
        "phase": "8J-R",
        "required_state": {
            "phase8h_perception_gate": phase8h.get("status"),
            "perception_frozen": phase8i.get("perception_frozen"),
            "phase8i_route": phase8i.get("route"),
            "phase8j_status": phase8j.get("status"),
            "coverage_ready": phase8j.get("coverage_ready"),
            "score_stage_executed": phase8j.get("score_stage_executed"),
            "production_test_used": phase8j.get("production_test_used"),
        },
        "checks": required_checks,
        "frozen_hashes": {
            "expected": expected_frozen,
            "current": current_frozen,
            "matches": frozen_matches,
        },
        "cache": {
            "root": phase8i_cache["cache_root"],
            "index": str(cache_index),
            "index_sha256": sha256(cache_index),
            "index_content_hash": phase8i_cache["index_content_hash"],
            "indexed_windows": phase8i_cache["indexed_cache_file_count"],
            "unindexed_stale_files": phase8i_cache["unindexed_stale_file_count"],
            "foreground_mode": phase8i_cache["effective_foreground_mode"],
        },
        "data_hashes": {
            "expected": expected_data_hashes,
            "current": current_data_hashes,
            "matches": data_matches,
        },
        "phase8j_c_k3": {
            "matrix": str(matrix),
            "matrix_sha256": sha256(matrix),
            "checkpoint": str(c_k3_checkpoint),
            "checkpoint_sha256": sha256(c_k3_checkpoint),
            "seed": representative["seed"],
            "selected_epoch": representative["selected_epoch"],
            "score_branch_hash_unchanged": all(
                result["score_parameter_hash_unchanged"]
                for result in three_seed_results
            ),
        },
        "entry_failures": failures,
        "production_test_allowed": False,
        "score_training_allowed": False,
        "long_training_allowed": False,
    }
    atomic_json(reports / "phase8jr_entry_gate.json", entry)
    if failures:
        raise RuntimeError(f"Phase 8J-R entry Gate failed: {failures}")

    def selection_key(item):
        value = (
            item["selection_key"]
            if "selection_key" in item
            else item["selection_keys"]["estimated"]
        )
        return tuple(value)

    selected_epochs = [
        min(result["history"], key=selection_key)
        for result in three_seed_results
    ]
    baseline = {
        "status": "PASS",
        "phase8i": {
            "checkpoint": prior_entry["decision_baseline"]["checkpoint_path"],
            "checkpoint_sha256": prior_entry["decision_baseline"]["checkpoint_sha256"],
            "suite": "valid_estimated",
            "windows": 2052,
            "dynamic_coverage_failure_fraction": prior_entry[
                "decision_baseline"
            ]["dynamic_coverage_failure_fraction"],
            "joint_coverage_failure_fraction": prior_entry[
                "decision_baseline"
            ]["joint_coverage_failure_fraction"],
        },
        "phase8j_c_k3": {
            "dynamic_coverage_failure_fractions": [
                run["validation"]["valid_estimated"][
                    "dynamic_coverage_failure_fraction"
                ]["point_estimate"]
                for run in selected_epochs
            ],
            "joint_coverage_failure_fractions": [
                run["validation"]["valid_estimated"][
                    "joint_coverage_failure_fraction"
                ]["point_estimate"]
                for run in selected_epochs
            ],
            "joint_safe_q10": [
                run["validation"]["valid_estimated"][
                    "joint_safe_candidate_count"
                ]["q10"]
                for run in selected_epochs
            ],
            "no_target_dynamic_cost_max_abs": [
                run["validation"]["valid_estimated"][
                    "no_target_dynamic_cost_max_abs"
                ]
                for run in selected_epochs
            ],
            "score_branch_hash_unchanged": all(
                result["score_parameter_hash_unchanged"]
                for result in three_seed_results
            ),
            "interpretation": (
                "The three-seed variation is baseline-level noise, not a "
                "candidate-coverage improvement."
            ),
        },
        "production_test_used": False,
        "network_weights_modified": False,
    }
    atomic_json(reports / "phase8jr_baseline_reproduction.json", baseline)
    print(json.dumps({
        "status": "PASS",
        "entry_gate": str(reports / "phase8jr_entry_gate.json"),
        "baseline": str(reports / "phase8jr_baseline_reproduction.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
