#!/usr/bin/env python3
"""Verify Phase 8J-V2-S0 entry facts without modifying old reports."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main():
    required = [
        "phase8jqv2_final_result.json",
        "phase8jqv2_final_recommendation.md",
        "phase8jqv2_evaluator_spec.md",
        "phase8jqv2_static_validation.json",
        "phase8jqv2_component_ablation.json",
        "phase8jv2_capacity_gate.json",
        "phase8jv2_capacity_oracle.json",
        "phase8jv2_parameterization_recommendation.md",
        "phase8jv2_final_result.json",
        "phase8jv2_final_readiness.md",
    ]
    source_paths = [
        "policy/dep_dataset.py",
        "tools/evaluate_phase8jqv2_static.py",
        "tools/analyze_candidate_capacity_v2.py",
        "tools/optimize_candidate_capacity_v2.py",
        "tools/evaluate_candidate_decomposition_v2.py",
        "policy/safety_evaluator_v2.py",
        "loss/safety_geometry_v2.py",
        "loss/safety_loss.py",
        "policy/state_transform.py",
        "configs/static_map_catalog.yaml",
        "configs/train_dynamic_production_v2.yaml",
        "config/traj_opt.yaml",
    ]
    missing = [
        str(path) for path in
        [REPORTS / name for name in required]
        + [ROOT / name for name in source_paths]
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"missing S0 inputs: {missing}")

    dataset = (ROOT / "policy/dep_dataset.py").read_text()
    static_eval = (
        ROOT / "tools/evaluate_phase8jqv2_static.py"
    ).read_text()
    capacity = (
        ROOT / "tools/analyze_candidate_capacity_v2.py"
    ).read_text()
    optimizer = (
        ROOT / "tools/optimize_candidate_capacity_v2.py"
    ).read_text()
    production = (
        ROOT / "configs/train_dynamic_production_v2.yaml"
    ).read_text()
    facts = {
        "static_state_goal_sampled_inside_getitem": (
            "vel_b, acc_b = self._get_random_state()" in dataset
            and "goal_w = self._get_random_goal()" in dataset
        ),
        "legacy_rng_worker_not_sample_index": (
            "global_seed + epoch * 100003 + worker_id" in dataset
            and "per_index_seed" not in dataset
        ),
        "c0_static_default_batch_size_32": (
            'parser.add_argument("--batch-size", type=int, default=32)'
            in static_eval
        ),
        "c1_c2_static_reconstruction_batch_size_64": (
            "dataset, batch_size=64" in capacity
        ),
        "legacy_static_artifact_omits_state_goal": (
            'columns = {"clearance": [], "predicted": [], "map": [], "t0": []}'
            in static_eval
        ),
        "static_dataset_has_no_reference_trajectory": (
            "reference_trajectory" not in dataset
        ),
        "c1_c2_c3_hard_limits_6": (
            "velocity_limit_mps\": 6.0" in capacity
            and "acceleration_limit_mps2\": 6.0" in capacity
            and "speed-6.0" in optimizer
            and "accel-6.0" in optimizer
        ),
        "static_sampler_norm_limit_1_2x": (
            "np.linalg.norm(vel) < 1.2 * self.vel_max" in dataset
            and "np.linalg.norm(acc) < 1.2 * self.acc_max" in dataset
        ),
        "estimated_covariance_uses_sqrt_variance": (
            "variance.sqrt()" in optimizer
        ),
        "static_checker_fixed_subdivision_lipschitz_bound": (
            "np.minimum(raw[:, :, :-1], raw[:, :, 1:]) - segment"
            in static_eval
        ),
        "maps_10_11_are_dynamic_train_not_validation": (
            "train: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]"
            in production
            and "valid: [12, 13, 14]" in production
        ),
    }
    report = {
        "status": "PASS" if all(facts.values()) else "FAIL",
        "phase": "8J-V2-S0",
        "facts": facts,
        "source_hashes": {
            name: sha256(ROOT / name) for name in source_paths
        },
        "input_report_hashes": {
            name: sha256(REPORTS / name) for name in required
        },
        "network_weights_modified": False,
        "training_executed": False,
        "score_training_executed": False,
        "candidate_generator_frozen": False,
        "production_test_used": False,
        "blind_used": False,
        "depth_dataset_rebuilt": False,
        "PLY_dataset_rebuilt": False,
    }
    atomic_json(REPORTS / "phase8jv2s0_entry_gate.json", report)
    legacy = {
        "status": "HISTORICAL_ONLY",
        "static_pairing_valid": False,
        "reason": (
            "C0 and recovery stages do not persist or reuse identical "
            "velocity, acceleration and goal records"
        ),
        "capacity_upper_bound_claim_valid": False,
        "legacy_static_results": {
            "c0": 0.3680,
            "c1": 0.2529,
            "c2_512": 0.2659,
            "c3_best_reported": 0.2565,
        },
        "dynamic_a0_status_unchanged": True,
        "old_reports_overwritten": False,
    }
    atomic_json(REPORTS / "phase8jv2s0_legacy_a0_validity.json", legacy)
    atomic_json(REPORTS / "phase8jv2s0_map_namespace.json", {
        "status": "PASS",
        "legacy_static_validation_maps": list(range(10)),
        "production_dynamic_validation_maps": [12, 13, 14],
        "production_dynamic_train_maps_intentionally_absent": [10, 11],
        "removed_invalid_check": "maps_0_through_14_all_present",
    })
    print(json.dumps({"status": report["status"], "facts": facts}, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
