#!/usr/bin/env python3
"""Measure Phase 8J-A objective gradients before choosing a PCGrad boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loss.coverage_loss import CoverageObjectiveConfig
from loss.dynamic_types import DynamicLossConfig, DynamicObjectiveConfig, RiskMetricsConfig
from policy.dynamic_training_config import DynamicTrainingConfig
from policy.phase8j_coverage_trainer import Phase8JCoverageTrainer
from policy.training_schedule import TrainingScheduleConfig
from train_dep import configure_random_seed
from tools.generate_phase8i_estimated_cache import phase8h_perception_config


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gradient_vector(loss, parameters):
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=True, allow_unused=True
    )
    return torch.cat([
        (torch.zeros_like(parameter) if gradient is None else gradient).reshape(-1)
        for parameter, gradient in zip(parameters, gradients)
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=(
        ROOT / "runs/phase8d_shakedown/fixed_050_seed8403/config.yaml"
    ))
    parser.add_argument("--checkpoint", type=Path, default=(
        ROOT / "runs/phase8d_shakedown/fixed_050_seed8403/checkpoints/best_dynamic.pt"
    ))
    parser.add_argument("--output", type=Path, default=(
        ROOT / "reports/phase8j_coverage_gradient_diagnostics.json"
    ))
    args = parser.parse_args()
    config = YAML(typ="safe").load(args.config)
    cache_report = json.loads(
        (ROOT / "reports/phase8i_estimated_cache_validation.json").read_text()
    )
    indexed_root = cache_report["cache_root"]
    training = dict(config["dynamic_training"])
    training["estimated_cache_dir"] = str(indexed_root)
    training["context_source"] = "estimated"
    coverage = CoverageObjectiveConfig.from_mapping({
        "enabled": True,
        "dynamic_cvar_weight": 1.0,
        "best_safe_weight": 1.0,
        "safe_count_weight": 0.25,
        "safe_count_k": 2,
        "endpoint_diversity_weight": 0.05,
        "trajectory_diversity_weight": 0.05,
    })
    configure_random_seed(8500)
    trainer = Phase8JCoverageTrainer(
        learning_rate=float(config["learning_rate"]),
        batch_size=int(config["batch_size"]),
        loss_weight=[1.0, 0.0],
        tensorboard_path=str(ROOT / "runs/phase8j_gradient_diagnostics"),
        checkpoint_path=str(args.checkpoint),
        backbone_variant="corrected",
        dataset_mode="dynamic",
        dynamic_data_root=config["dataset_root"],
        freeze_policy=config["freeze_policy"],
        random_seed=8500,
        num_workers=0,
        training_config_override=DynamicTrainingConfig.from_mapping(training),
        dynamic_loss_config_override=DynamicLossConfig.from_mapping(config["dynamic_loss"]),
        dynamic_objective_config_override=DynamicObjectiveConfig.from_mapping(
            config["dynamic_objective"]
        ),
        risk_metrics_config_override=RiskMetricsConfig.from_mapping(config["risk_metrics"]),
        dynamic_map_catalog_override=config["dynamic_map_catalog"],
        coverage_config=coverage,
        dynamic_perception_config_override=phase8h_perception_config(),
        estimated_cache_perception_config_override=phase8h_perception_config(),
    )
    trainer.set_training_mode()
    chosen = None
    for batch in trainer.train_loaders["dynamic"]:
        details = trainer.compute_batch("dynamic", batch)
        diagnostics = details["dynamic_diagnostics"]
        if bool((diagnostics.dynamic_obstacle_count > 0).any()):
            chosen = details
            break
    if chosen is None:
        raise RuntimeError("no target-bearing dynamic batch found")
    parameters = [
        parameter for parameter in trainer.policy.parameters() if parameter.requires_grad
    ]
    objectives = {
        "guidance": chosen["guidance_loss"],
        "smoothness": chosen["smooth_loss"],
        "static": chosen["static_safety_loss"],
        "dynamic_mean": chosen["dynamic_training_objective"],
        "dynamic_cvar": chosen["coverage_dynamic_cvar"],
        "diversity": (
            chosen["coverage_endpoint_diversity"]
            + chosen["coverage_trajectory_diversity"]
        ),
        "coverage": chosen["coverage_loss"],
    }
    vectors = {name: gradient_vector(value, parameters).detach()
               for name, value in objectives.items()}
    norms = {name: float(torch.linalg.vector_norm(value).cpu())
             for name, value in vectors.items()}
    cosines = {}
    conflicts = 0
    pairs = 0
    for left, left_vector in vectors.items():
        cosines[left] = {}
        for right, right_vector in vectors.items():
            denominator = (
                torch.linalg.vector_norm(left_vector)
                * torch.linalg.vector_norm(right_vector)
            )
            value = (
                float(torch.dot(left_vector, right_vector).div(denominator).cpu())
                if float(denominator) > 0 else 0.0
            )
            cosines[left][right] = value
            if left < right:
                pairs += 1
                conflicts += int(value < 0)
    safety = vectors["dynamic_mean"] + vectors["coverage"]
    non_dynamic = vectors["guidance"] + vectors["smoothness"] + vectors["static"]
    denominator = torch.linalg.vector_norm(safety) * torch.linalg.vector_norm(non_dynamic)
    group_cosine = float(torch.dot(safety, non_dynamic).div(denominator).cpu())
    result = {
        "status": "PASS",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "coverage_config": coverage.__dict__,
        "gradient_norms": norms,
        "pairwise_cosine": cosines,
        "conflict_fraction": conflicts / max(pairs, 1),
        "candidate_grouping": {
            "selected": "coverage_plus_dynamic_as_safety_group",
            "safety_vs_non_dynamic_cosine": group_cosine,
            "reason": (
                "coverage contains dynamic/static joint violation and must retain "
                "dynamic-priority PCGrad while preserving guidance/smoothness/static gradients"
            ),
        },
        "zero_gradient": [name for name, value in norms.items() if value <= 1e-12],
        "optimizer_step_executed": False,
        "production_test_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
