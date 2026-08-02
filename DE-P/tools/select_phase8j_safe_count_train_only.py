#!/usr/bin/env python3
"""Choose K=1/2/3 using only the frozen production-train split."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import torch
from ruamel.yaml import YAML
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loss.coverage_loss import CoverageObjectiveConfig
from loss.dynamic_types import DynamicLossConfig, DynamicObjectiveConfig, RiskMetricsConfig
from policy.checkpoint_utils import load_dep_checkpoint
from policy.dep_dataset import seed_dataset_worker
from policy.dynamic_collate import dynamic_sequence_collate
from policy.dynamic_training_config import DynamicTrainingConfig
from policy.phase8j_coverage_trainer import Phase8JCoverageTrainer
from tools.phase8j_coverage_metrics import evaluate_dynamic
from tools.generate_phase8i_estimated_cache import phase8h_perception_config


def main():
    matrix = YAML(typ="safe").load(ROOT / "configs/phase8j_coverage_matrix.yaml")
    base = YAML(typ="safe").load(Path(matrix["base_config"]))
    cache = json.loads(Path(matrix["estimated_cache_validation"]).read_text())
    training = dict(base["dynamic_training"])
    training.update({
        "estimated_cache_dir": cache["cache_root"],
        "context_source": "estimated",
        "curriculum": [{"start_epoch": 0, "context_source": "estimated", "ratio": 1.0}],
    })
    coverage = dict(matrix["common"])
    coverage.update(matrix["ablations"]["C_k2"])
    trainer = Phase8JCoverageTrainer(
        learning_rate=float(matrix["learning_rate"]), batch_size=32,
        loss_weight=[1, 0], tensorboard_path=str(ROOT / "runs/phase8j_k_selection"),
        backbone_variant="corrected", dataset_mode="dynamic",
        dynamic_data_root=base["dataset_root"], freeze_policy=base["freeze_policy"],
        random_seed=8511, num_workers=4,
        training_config_override=DynamicTrainingConfig.from_mapping(training),
        dynamic_loss_config_override=DynamicLossConfig.from_mapping(base["dynamic_loss"]),
        dynamic_objective_config_override=DynamicObjectiveConfig.from_mapping(
            base["dynamic_objective"]
        ),
        risk_metrics_config_override=RiskMetricsConfig.from_mapping(base["risk_metrics"]),
        dynamic_map_catalog_override=base["dynamic_map_catalog"],
        coverage_config=CoverageObjectiveConfig.from_mapping(coverage),
        dynamic_perception_config_override=phase8h_perception_config(),
        estimated_cache_perception_config_override=phase8h_perception_config(),
    )
    dataset = trainer.train_loaders["dynamic"].dataset
    loader = DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=4,
        pin_memory=True, persistent_workers=True,
        worker_init_fn=seed_dataset_worker,
        collate_fn=lambda samples: dynamic_sequence_collate(samples, max_obstacles=16),
    )
    results = {}
    for variant in ("C_k1", "C_k2", "C_k3"):
        checkpoint = (
            ROOT / "runs/phase8j_coverage"
            / f"{variant}_seed8511_e1/checkpoints/best_coverage_estimated.pt"
        )
        load_dep_checkpoint(trainer.policy, checkpoint, "corrected")
        results[variant] = evaluate_dynamic(trainer, loader)
    # Predeclared train-only order: coverage, q10, median, then smallest K.
    order = sorted(results, key=lambda name: (
        results[name]["joint_coverage_failure_fraction"]["point_estimate"],
        results[name]["dynamic_coverage_failure_fraction"]["point_estimate"],
        -results[name]["joint_safe_candidate_count"]["q10"],
        -results[name]["joint_safe_candidate_count"]["median"],
        int(name[-1]),
    ))
    payload = {
        "status": "PASS",
        "selection_data": "production_train_8208_only",
        "validation_used_for_k_selection": False,
        "test_used": False,
        "results": results,
        "selected": order[0],
        "selection_order": order,
    }
    output = ROOT / "reports/phase8j_safe_count_train_only_selection.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "selected": order[0], "order": order}, indent=2))


if __name__ == "__main__":
    main()
