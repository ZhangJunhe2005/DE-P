#!/usr/bin/env python3
"""Finite host-GPU smoke for Phase 8D map, schedule, validation and lazy data."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import torch
from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loss.dynamic_types import DynamicLossConfig, DynamicObjectiveConfig, RiskMetricsConfig
from policy.dep_trainer import DepTrainer
from policy.dynamic_training_config import DynamicTrainingConfig
from policy.training_schedule import TrainingScheduleConfig
from train_dep import configure_random_seed


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA full-access is required")
    config = YAML(typ="safe").load(ROOT / "configs/train_dynamic_production_v2.yaml")
    configure_random_seed(8801)
    with tempfile.TemporaryDirectory(prefix="phase8d-mixed-smoke-") as directory:
        trainer = DepTrainer(
            learning_rate=config["learning_rate"], batch_size=config["batch_size"],
            loss_weight=config["loss_weight"], tensorboard_path=directory,
            checkpoint_path=config["initialization_checkpoint"],
            backbone_variant="corrected", dataset_mode="mixed",
            dynamic_data_root=config["dataset_root"], freeze_policy="late",
            num_workers=0, random_seed=8801,
            training_config_override=DynamicTrainingConfig.from_mapping(
                config["dynamic_training"]
            ),
            dynamic_loss_config_override=DynamicLossConfig.from_mapping(
                config["dynamic_loss"]
            ),
            dynamic_objective_config_override=DynamicObjectiveConfig.from_mapping(
                config["dynamic_objective"]
            ),
            risk_metrics_config_override=RiskMetricsConfig.from_mapping(
                config["risk_metrics"]
            ),
            static_map_catalog_override=config["static_map_catalog"],
            dynamic_map_catalog_override=config["dynamic_map_catalog"],
            training_schedule_override=TrainingScheduleConfig.from_mapping(
                config["training_schedule"]
            ),
            static_cache_size=config["static_dataset"]["cache_size"],
        )
        trainer._apply_curriculum(0)
        static_batch = next(iter(trainer.train_loaders["static"]))
        dynamic_batch = next(iter(trainer.train_loaders["dynamic"]))
        static = trainer.optimize_batch("static", static_batch)
        dynamic = trainer.optimize_batch("dynamic", dynamic_batch)
        static_paths = [Path(path).resolve() for _, path in trainer.static_dep_loss.safety_loss.map_files]
        dynamic_paths = [Path(path).resolve() for _, path in trainer.dynamic_dep_loss.safety_loss.map_files]
        schedule_counts = trainer.training_schedule.batch_counts()
        result = {
            "status": "PASS",
            "device": torch.cuda.get_device_name(0),
            "static_catalog_hash": trainer.static_map_catalog_hash,
            "dynamic_catalog_hash": trainer.dynamic_map_catalog_hash,
            "static_maps_only_legacy": all(
                path.parent == Path("/home/zjh/YOPO/dataset") for path in static_paths
            ),
            "dynamic_maps_only_formal": all(
                "phase8c_static_maps" in str(path) for path in dynamic_paths
            ),
            "static_dynamic_loss": float(static["dynamic_safety_loss"].detach()),
            "dynamic_loss_finite": bool(torch.isfinite(dynamic["total_loss"])),
            "static_loss_finite": bool(torch.isfinite(static["total_loss"])),
            "steps_per_epoch": trainer._mixed_epoch_length(),
            "batch_counts": schedule_counts,
            "dynamic_loader_batches": len(trainer.train_loaders["dynamic"]),
            "static_loader_batches": len(trainer.train_loaders["static"]),
            "dynamic_recycling": False,
            "static_dataset_cache_count": trainer.train_loaders["static"].dataset.cached_depth_count,
            "validation_contexts": {
                name: loader.dataset._curriculum_context_source
                for name, loader in trainer.validation_suites.items()
                if name != "valid_static"
            },
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        }
        checks = (
            result["static_maps_only_legacy"], result["dynamic_maps_only_formal"],
            result["static_dynamic_loss"] == 0.0, result["dynamic_loss_finite"],
            result["static_loss_finite"], result["steps_per_epoch"] == 1026,
            schedule_counts == {"static": 513, "dynamic": 513},
            result["dynamic_loader_batches"] == 513,
            result["static_loader_batches"] == 513,
            result["validation_contexts"] == {
                "valid_gt": "ground_truth", "valid_estimated": "estimated",
            },
        )
        result["status"] = "PASS" if all(checks) else "FAIL"
        output = ROOT / "reports/phase8d_mixed_smoke.json"
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("HOST_PHASE8D_MIXED_SMOKE_RESULT")
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
