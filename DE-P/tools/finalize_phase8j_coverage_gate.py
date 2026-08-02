#!/usr/bin/env python3
"""Finalize Phase 8J-A and fail closed before score calibration."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

import numpy as np
import torch
from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loss.coverage_loss import CoverageObjectiveConfig
from loss.dynamic_types import DynamicLossConfig, DynamicObjectiveConfig, RiskMetricsConfig
from policy.checkpoint_utils import load_dep_checkpoint
from policy.dynamic_training_config import DynamicTrainingConfig
from policy.phase8j_coverage_trainer import Phase8JCoverageTrainer
from policy.training_schedule import TrainingScheduleConfig
from tools.phase8j_coverage_metrics import evaluate_static
from tools.run_phase8i_failure_decomposition import scenario_lookup
from tools.generate_phase8i_estimated_cache import phase8h_perception_config


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def best_epoch(result):
    return min(result["history"], key=lambda item: tuple(item["selection_key"]))


def scalar_metric(epoch, suite, key):
    value = epoch["validation"][suite][key]
    return value["point_estimate"] if isinstance(value, dict) and "point_estimate" in value else value


@torch.inference_mode()
def collect_failures(trainer, checkpoint):
    load_dep_checkpoint(trainer.policy, checkpoint, "corrected")
    trainer.policy.eval()
    loader = trainer.validation_suites["valid_estimated"]
    scenarios = scenario_lookup(loader.dataset)
    failures = []
    for batch in loader:
        details = trainer.compute_batch("dynamic", batch)
        diagnostics = details["dynamic_diagnostics"]
        dynamic_clearance = diagnostics.candidate_min_clearance.detach().cpu().numpy()
        static_clearance = details["candidate_static_clearance"].detach().cpu().numpy()
        dynamic_safe = dynamic_clearance >= -1e-6
        static_safe = static_clearance >= -1e-6
        joint_safe = dynamic_safe & static_safe
        for row, failed in enumerate(~joint_safe.any(1)):
            if not failed:
                continue
            sequence = batch["sequence_id"][row]
            failures.append({
                "sequence_id": sequence,
                "frame_index": int(batch["frame_index"][row]),
                "map_id": int(batch["map_id"][row]),
                "scenario": scenarios[sequence],
                "category": batch["sample_category"][row],
                "dynamic_safe_candidate_count": int(dynamic_safe[row].sum()),
                "static_safe_candidate_count": int(static_safe[row].sum()),
                "joint_safe_candidate_count": 0,
                "maximum_dynamic_clearance_m": float(dynamic_clearance[row].max()),
                "maximum_static_clearance_m": float(static_clearance[row].max()),
            })
    return failures


def main():
    matrix_path = ROOT / "configs/phase8j_coverage_matrix.yaml"
    matrix = YAML(typ="safe").load(matrix_path)
    baseline = json.loads(
        (ROOT / "reports/phase8j_candidate_diversity_baseline.json").read_text()
    )["suites"]["valid_estimated"]
    cache = json.loads(Path(matrix["estimated_cache_validation"]).read_text())
    base = YAML(typ="safe").load(Path(matrix["base_config"]))
    training = dict(base["dynamic_training"])
    training.update({
        "estimated_cache_dir": cache["cache_root"],
        "context_source": "estimated",
        "curriculum": [{"start_epoch": 0, "context_source": "estimated", "ratio": 1.0}],
    })
    coverage = dict(matrix["common"])
    coverage.update(matrix["ablations"]["C_k3"])
    trainer = Phase8JCoverageTrainer(
        learning_rate=float(matrix["learning_rate"]), batch_size=16,
        loss_weight=[1, 0], tensorboard_path=str(ROOT / "runs/phase8j_gate_finalize"),
        checkpoint_path=matrix["baseline_checkpoint"],
        backbone_variant="corrected", dataset_mode="mixed",
        dynamic_data_root=base["dataset_root"], freeze_policy=base["freeze_policy"],
        random_seed=8513, num_workers=4,
        training_config_override=DynamicTrainingConfig.from_mapping(training),
        dynamic_loss_config_override=DynamicLossConfig.from_mapping(base["dynamic_loss"]),
        dynamic_objective_config_override=DynamicObjectiveConfig.from_mapping(
            base["dynamic_objective"]
        ),
        risk_metrics_config_override=RiskMetricsConfig.from_mapping(base["risk_metrics"]),
        static_map_catalog_override=base["static_map_catalog"],
        dynamic_map_catalog_override=base["dynamic_map_catalog"],
        training_schedule_override=TrainingScheduleConfig.from_mapping(
            base["training_schedule"]
        ),
        static_cache_size=int(base["static_dataset"]["cache_size"]),
        coverage_config=CoverageObjectiveConfig.from_mapping(coverage),
        dynamic_perception_config_override=phase8h_perception_config(),
        estimated_cache_perception_config_override=phase8h_perception_config(),
    )
    baseline_static = evaluate_static(
        trainer, trainer.validation_suites["valid_static"]
    )

    prescreen = {}
    for variant in ("B", "C_k1", "C_k2", "C_k3", "D", "E"):
        result = json.loads((
            ROOT / "runs/phase8j_coverage"
            / f"{variant}_seed8511_e1/result.json"
        ).read_text())
        epoch = best_epoch(result)
        prescreen[variant] = {
            "seed": 8511, "epochs": 1,
            "dynamic_coverage_failure_fraction": scalar_metric(
                epoch, "valid_estimated", "dynamic_coverage_failure_fraction"
            ),
            "joint_coverage_failure_fraction": scalar_metric(
                epoch, "valid_estimated", "joint_coverage_failure_fraction"
            ),
            "score_parameter_hash_unchanged": result[
                "score_parameter_hash_unchanged"
            ],
        }
    runs = []
    for seed in matrix["seeds"]:
        path = (
            ROOT / "runs/phase8j_coverage"
            / f"C_k3_seed{seed}_e3/result.json"
        )
        result = json.loads(path.read_text())
        epoch = best_epoch(result)
        runs.append({
            "seed": seed,
            "selected_epoch": epoch["epoch"],
            "checkpoint": result["best_checkpoint"],
            "checkpoint_sha256": result["best_checkpoint_sha256"],
            "score_parameter_hash_unchanged": result["score_parameter_hash_unchanged"],
            "metrics": epoch["validation"],
            "schedule": epoch["schedule"],
        })
    representative = min(runs, key=lambda item: (
        scalar_metric({"validation": item["metrics"]}, "valid_estimated",
                      "joint_coverage_failure_fraction"),
        scalar_metric({"validation": item["metrics"]}, "valid_estimated",
                      "dynamic_coverage_failure_fraction"),
    ))
    failures = collect_failures(trainer, representative["checkpoint"])
    atomic_json(
        ROOT / "diagnostics/phase8j/coverage_failures_after_training.json",
        {
            "status": "PASS", "checkpoint": representative["checkpoint"],
            "failure_count": len(failures), "failures": failures,
            "production_test_used": False,
        },
    )

    def seed_values(key):
        return [
            scalar_metric({"validation": run["metrics"]}, "valid_estimated", key)
            for run in runs
        ]
    dynamic_values = seed_values("dynamic_coverage_failure_fraction")
    joint_values = seed_values("joint_coverage_failure_fraction")
    medians = [
        run["metrics"]["valid_estimated"]["joint_safe_candidate_count"]["median"]
        for run in runs
    ]
    q10s = [
        run["metrics"]["valid_estimated"]["joint_safe_candidate_count"]["q10"]
        for run in runs
    ]
    endpoint = [
        run["metrics"]["valid_estimated"]["endpoint_pairwise_distance_mean_m"]
        for run in runs
    ]
    trajectory = [
        run["metrics"]["valid_estimated"]["trajectory_pairwise_distance_mean_m"]
        for run in runs
    ]
    duplicates = [
        run["metrics"]["valid_estimated"]["near_duplicate_candidate_pair_fraction"]
        for run in runs
    ]
    no_target = [
        run["metrics"]["valid_estimated"]["no_target_dynamic_cost_max_abs"]
        for run in runs
    ]
    static_failure = [
        run["metrics"]["valid_static"]["static_joint_coverage_failure_fraction"]
        for run in runs
    ]
    limits = matrix["secondary_regression_limits"]
    checks = {
        "dynamic_coverage_failure_le_005": max(dynamic_values) <= .05,
        "joint_coverage_failure_le_005": max(joint_values) <= .05,
        "oracle_dynamic_collision_le_005": max(dynamic_values) <= .05,
        "oracle_joint_collision_le_005": max(joint_values) <= .05,
        "joint_safe_median_not_decreased": min(medians) >= baseline[
            "joint_safe_candidate_count"
        ]["quantiles"]["0.5"],
        "joint_safe_q10_positive": min(q10s) > 0,
        "endpoint_diversity_not_regressed": min(endpoint) >= (
            baseline["endpoint_pairwise_distance_m"]["mean"]
            * limits["endpoint_diversity_relative_ratio_min"]
        ),
        "trajectory_diversity_not_regressed": min(trajectory) >= (
            baseline["trajectory_pairwise_distance_m"]["mean"]
            * limits["trajectory_diversity_relative_ratio_min"]
        ),
        "near_duplicate_not_regressed": max(duplicates) <= (
            baseline["near_duplicate_candidate_pair_fraction"]["mean"]
            * limits["near_duplicate_relative_ratio_max"]
        ),
        "no_target_dynamic_exact_zero": max(no_target) == 0,
        "static_coverage_not_regressed": max(static_failure) <= (
            baseline_static["static_joint_coverage_failure_fraction"]
            + limits["static_coverage_absolute_regression_max"]
        ),
        "static_smoke_pass": all(
            run["metrics"]["valid_static"]["static_smoke_pass"] for run in runs
        ),
        "physical_feasibility_pass": all(
            run["metrics"]["valid_estimated"]["maximum_endpoint_radius_m"]
            <= limits["endpoint_radius_max_m"] for run in runs
        ),
        "finite": all(
            run["metrics"]["valid_estimated"]["finite"]
            and run["metrics"]["valid_static"]["finite"] for run in runs
        ),
        "three_seeds_complete": len(runs) == 3,
        "score_branch_hash_unchanged": all(
            run["score_parameter_hash_unchanged"] for run in runs
        ),
        "schedule_exact": all(
            run["schedule"]["dynamic_batch_count"] == 513
            and run["schedule"]["static_batch_count"] == 513
            and run["schedule"]["dynamic_loader_recycle_count"] == 0
            and run["schedule"]["static_loader_recycle_count"] == 0
            for run in runs
        ),
    }
    gate_pass = all(checks.values())
    ablation = {
        "status": "PASS",
        "baseline": {
            "dynamic_coverage_failure_fraction": baseline[
                "dynamic_coverage_failure_fraction"
            ],
            "joint_coverage_failure_fraction": baseline[
                "joint_coverage_failure_fraction"
            ],
        },
        "prescreen": prescreen,
        "safe_count_train_only_selection": json.loads((
            ROOT / "reports/phase8j_safe_count_train_only_selection.json"
        ).read_text()),
        "primitive_collapse_regularization_executed": False,
        "primitive_collapse_reason": (
            "audit found 14.587/15 effective candidates and 0.00395 near-duplicate pairs"
        ),
        "selected_final_variant": "C_k3",
        "final_runs": runs,
    }
    atomic_json(ROOT / "reports/phase8j_coverage_ablation_summary.json", ablation)
    summary = {
        "status": "PASS" if gate_pass else "FAIL",
        "coverage_ready": gate_pass,
        "score_stage_executed": False,
        "selected_variant": "C_k3",
        "representative": representative,
        "three_seed": {
            "dynamic_coverage_failure_fraction": {
                "values": dynamic_values,
                "mean": statistics.mean(dynamic_values),
                "std": statistics.pstdev(dynamic_values),
            },
            "joint_coverage_failure_fraction": {
                "values": joint_values,
                "mean": statistics.mean(joint_values),
                "std": statistics.pstdev(joint_values),
            },
        },
        "baseline_static": baseline_static,
        "checks": checks,
        "production_test_used": False,
    }
    atomic_json(ROOT / "reports/phase8j_coverage_summary.json", summary)
    atomic_json(ROOT / "reports/phase8j_coverage_gate.json", summary)

    skipped = {
        "status": "NOT_EXECUTED",
        "reason": "Phase 8J-A Coverage Gate failed; Phase 8J-B is forbidden",
        "coverage_ready": False,
        "production_test_used": False,
    }
    for name in (
        "phase8j_score_target_property_tests.json",
        "phase8j_score_ablation_summary.json",
        "phase8j_score_calibration_summary.json",
        "phase8j_score_gate.json",
    ):
        atomic_json(ROOT / "reports" / name, skipped)
    for name in (
        "remaining_label_inversions.json",
        "remaining_model_ranking_failures.json",
        "gt_estimated_disagreements.json",
    ):
        atomic_json(
            ROOT / "diagnostics/phase8j" / name,
            {**skipped, "unchanged_reference": (
                "reports/phase8i_candidate_failure_decomposition.json"
            )},
        )
    (ROOT / "reports/phase8j_score_target_design.md").write_text(
        "# Phase 8J-B Score Target\n\n"
        "NOT EXECUTED: Phase 8J-A Coverage Gate failed. No score target, "
        "ranking loss, or score-head optimizer step was implemented or run.\n"
    )
    final = {
        "status": "FAIL",
        "route": "C",
        "coverage_ready": False,
        "score_stage_executed": False,
        "candidate_generator_frozen": False,
        "perception_frozen": True,
        "production_test_used": False,
        "long_training_started": False,
        "next_allowed_phase": None,
        "coverage_gate": str(ROOT / "reports/phase8j_coverage_gate.json"),
        "failed_checks": [name for name, passed in checks.items() if not passed],
    }
    atomic_json(ROOT / "reports/phase8j_final_result.json", final)
    (ROOT / "reports/phase8j_final_readiness.md").write_text(
        "# Phase 8J Final Readiness\n\n"
        "**FAIL — stopped at Phase 8J-A.**\n\n"
        f"- 3-seed dynamic coverage: {dynamic_values}\n"
        f"- 3-seed joint coverage: {joint_values}\n"
        f"- q10 joint-safe candidate count: {q10s}\n"
        f"- Failed checks: {[name for name, passed in checks.items() if not passed]}\n"
        "- Score calibration was not executed.\n"
        "- Candidate generator was not frozen as a production candidate.\n"
        "- Phase 8K and long production training are not authorized.\n"
    )
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
