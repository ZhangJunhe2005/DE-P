#!/usr/bin/env python3
"""Finite Phase 8J-A coverage training; never launches production training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

import torch
from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loss.coverage_loss import CoverageObjectiveConfig
from loss.dynamic_types import DynamicLossConfig, DynamicObjectiveConfig, RiskMetricsConfig
from policy.dynamic_training_config import DynamicTrainingConfig
from policy.phase8j_coverage_trainer import Phase8JCoverageTrainer
from policy.training_schedule import TrainingScheduleConfig
from tools.phase8j_coverage_metrics import evaluate_all
from train_dep import configure_random_seed
from tools.generate_phase8i_estimated_cache import phase8h_perception_config


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_hash(parameters):
    digest = hashlib.sha256()
    for name, value in sorted(parameters):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=(
        ROOT / "configs/phase8j_coverage_matrix.yaml"
    ))
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if args.epochs < 1 or args.epochs > 5:
        raise ValueError("Phase 8J-A permits only 1..5 bounded epochs")
    matrix = YAML(typ="safe").load(args.matrix)
    if args.variant not in matrix["ablations"]:
        raise ValueError(f"unknown ablation {args.variant}")
    if args.seed not in matrix["seeds"]:
        raise ValueError("seed is outside the predeclared three-seed set")
    base = YAML(typ="safe").load(Path(matrix["base_config"]))
    cache_report = json.loads(Path(matrix["estimated_cache_validation"]).read_text())
    if cache_report["status"] != "PASS":
        raise RuntimeError("Phase 8I estimated cache validation is not PASS")
    training = dict(base["dynamic_training"])
    training["estimated_cache_dir"] = cache_report["cache_root"]
    training["context_source"] = "estimated"
    training["noise_seed"] = args.seed
    training["curriculum"] = [{
        "start_epoch": 0, "context_source": "estimated", "ratio": 1.0,
    }]
    coverage_mapping = dict(matrix["common"])
    coverage_mapping.update(matrix["ablations"][args.variant])
    coverage = CoverageObjectiveConfig.from_mapping(coverage_mapping)
    run_dir = args.run_dir or (
        ROOT / "runs/phase8j_coverage"
        / f"{args.variant}_seed{args.seed}_e{args.epochs}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    configure_random_seed(args.seed)
    trainer = Phase8JCoverageTrainer(
        learning_rate=float(matrix["learning_rate"]),
        batch_size=int(base["batch_size"]),
        loss_weight=[1.0, 0.0],
        tensorboard_path=str(run_dir / "tensorboard"),
        checkpoint_path=None if args.resume else matrix["baseline_checkpoint"],
        backbone_variant="corrected",
        dataset_mode="mixed",
        dynamic_data_root=base["dataset_root"],
        freeze_policy=base["freeze_policy"],
        random_seed=args.seed,
        num_workers=args.num_workers,
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
        coverage_config=coverage,
        dynamic_perception_config_override=phase8h_perception_config(),
        estimated_cache_perception_config_override=phase8h_perception_config(),
    )
    trainer.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        trainer.optimizer, mode="min", factor=.5, patience=2, min_lr=1e-6
    )
    if args.resume:
        trainer.resume_training(args.resume)
    score_named = [
        (name, value) for name, value in trainer.policy.named_parameters()
        if name.startswith("dep_head.score_head.")
    ]
    if not score_named or any(value.requires_grad for _, value in score_named):
        raise RuntimeError("score branch is not independently frozen")
    initial_score_hash = parameter_hash(score_named)
    config_snapshot = {
        "phase": "8J-A", "variant": args.variant, "seed": args.seed,
        "epochs": args.epochs, "matrix": str(args.matrix.resolve()),
        "matrix_sha256": sha256(args.matrix),
        "baseline_checkpoint": matrix["baseline_checkpoint"],
        "baseline_checkpoint_sha256": sha256(matrix["baseline_checkpoint"]),
        "coverage": coverage_mapping,
        "secondary_regression_limits": matrix["secondary_regression_limits"],
        "steps_per_epoch": 1026, "dynamic_batches_per_epoch": 513,
        "static_batches_per_epoch": 513, "loader_recycling": False,
        "estimated_cache_root": cache_report["cache_root"],
        "estimated_cache_index_sha256": cache_report["index_sha256"],
        "production_test_used": False,
    }
    atomic_json(run_dir / "config.json", config_snapshot)
    history = []
    best_keys = {
        "estimated": None,
        "gt": None,
        "balanced": None,
    }
    start_epoch = trainer.start_epoch
    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs):
        trainer.epoch_i = epoch
        trainer._apply_curriculum(epoch)
        trainer.set_training_mode()
        batch_times = []
        aggregates = {}
        count = 0
        for kind, batch in trainer._iter_training_batches():
            tick = time.perf_counter()
            details = trainer.optimize_batch(kind, batch)
            batch_times.append(time.perf_counter() - tick)
            count += 1
            trainer.global_step += 1
            for name in (
                "total_loss", "coverage_loss", "coverage_dynamic_cvar",
                "coverage_best_safe", "coverage_safe_count",
                "coverage_endpoint_diversity", "coverage_trajectory_diversity",
                "guidance_loss", "smooth_loss", "static_safety_loss",
                "pcgrad_conflict", "pcgrad_activated",
            ):
                aggregates[name] = aggregates.get(name, 0.0) + float(
                    details[name].detach().cpu()
                )
        if count != 1026:
            raise RuntimeError(f"epoch contains {count} steps, expected 1026")
        validation = evaluate_all(trainer)
        estimated = validation["valid_estimated"]
        gt = validation["valid_gt"]
        estimated_key = (
            estimated["joint_coverage_failure_fraction"]["point_estimate"],
            estimated["dynamic_coverage_failure_fraction"]["point_estimate"],
            -estimated["joint_safe_candidate_count"]["q10"],
            -estimated["joint_safe_candidate_count"]["median"],
        )
        gt_key = (
            gt["joint_coverage_failure_fraction"]["point_estimate"],
            gt["dynamic_coverage_failure_fraction"]["point_estimate"],
            -gt["joint_safe_candidate_count"]["q10"],
            -gt["joint_safe_candidate_count"]["median"],
        )
        balanced_key = (
            max(estimated_key[0], gt_key[0]),
            max(estimated_key[1], gt_key[1]),
            estimated_key[0] + gt_key[0],
            estimated_key[1] + gt_key[1],
            min(estimated_key[2], gt_key[2]),
            min(estimated_key[3], gt_key[3]),
        )
        trainer.scheduler.step(estimated_key[0])
        latest = run_dir / "checkpoints/latest.pt"
        trainer.save_checkpoint(latest, {
            "phase8j_variant": args.variant,
            "phase8j_matrix_sha256": sha256(args.matrix),
        })
        selections = {
            "estimated": estimated_key,
            "gt": gt_key,
            "balanced": balanced_key,
        }
        for selection, selection_key in selections.items():
            if (best_keys[selection] is None
                    or selection_key < best_keys[selection]):
                best_keys[selection] = selection_key
                metadata = {
                    "phase8j_variant": args.variant,
                    "selection_context": selection,
                    "selection_key": list(selection_key),
                }
                trainer.save_checkpoint(
                    run_dir / f"checkpoints/best_coverage_{selection}.pt",
                    metadata,
                )
                if selection == "estimated":
                    trainer.save_checkpoint(
                        run_dir / "checkpoints/best_dynamic_safety.pt",
                        metadata,
                    )
                elif selection == "balanced":
                    trainer.save_checkpoint(
                        run_dir / "checkpoints/best_balanced.pt",
                        metadata,
                    )
        epoch_result = {
            "epoch": epoch + 1, "global_step": trainer.global_step,
            "train": {name: value / count for name, value in aggregates.items()},
            "validation": validation,
            "selection_keys": {
                name: list(value) for name, value in selections.items()
            },
            "batch_latency_ms": {
                "median": statistics.median(batch_times) * 1000,
                "p95": float(torch.quantile(torch.tensor(batch_times), .95)) * 1000,
            },
            "gpu_peak_bytes": torch.cuda.max_memory_allocated(),
            "schedule": trainer.last_epoch_schedule_stats,
        }
        history.append(epoch_result)
        atomic_json(run_dir / "history.json", history)
        print(json.dumps({
            "variant": args.variant, "seed": args.seed, "epoch": epoch + 1,
            "selection_key": estimated_key,
        }))
    final_score_hash = parameter_hash(score_named)
    if final_score_hash != initial_score_hash:
        raise RuntimeError("frozen score branch parameter hash changed")
    best = run_dir / "checkpoints/best_coverage_estimated.pt"
    result = {
        "status": "PASS",
        **config_snapshot,
        "run_dir": str(run_dir.resolve()),
        "best_checkpoint": str(best.resolve()),
        "best_checkpoint_sha256": sha256(best),
        "score_parameter_hash_before": initial_score_hash,
        "score_parameter_hash_after": final_score_hash,
        "score_parameter_hash_unchanged": True,
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "long_training_started": False,
    }
    atomic_json(run_dir / "result.json", result)
    print(json.dumps({
        "status": "PASS", "run_dir": str(run_dir),
        "score_parameter_hash_unchanged": True,
    }, indent=2))


if __name__ == "__main__":
    main()
