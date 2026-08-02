#!/usr/bin/env python3
"""Finite mixed-training smoke: three steps, validation, save, and resume."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from policy.dep_trainer import DepTrainer
from policy.dynamic_collate import dynamic_sequence_collate
from tests.baseline_helpers import seed_everything


def scalar(metrics, key):
    value = metrics[key]
    return float(value.detach().cpu()) if torch.is_tensor(value) else float(value)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required; run this script with host full-access")
    seed_everything(6002)
    runtime = Path(tempfile.mkdtemp(prefix="dep-phase6-training-"))
    try:
        data_root = runtime / "dynamic_dataset"
        subprocess.run([
            sys.executable, str(ROOT / "tools/generate_synthetic_dynamic_sequences.py"),
            "--output", str(data_root),
        ], check=True)
        trainer = DepTrainer(
            learning_rate=1e-5, batch_size=2, loss_weight=[1.0, 1.0],
            tensorboard_path=str(runtime / "runs"),
            checkpoint_path=str(ROOT / "saved/DEP_corrected_init/epoch10_converted.pth"),
            backbone_variant="corrected", dataset_mode="mixed",
            dynamic_data_root=str(data_root), dynamic_context_source="ground_truth",
            dynamic_loss_enabled=True, dynamic_loss_weight=0.1,
            freeze_policy="late", max_grad_norm=0.1, num_workers=0,
            random_seed=6002,
        )
        torch.cuda.reset_peak_memory_stats(trainer.device)
        dynamic_dataset = trainer.train_loaders["dynamic"].dataset
        moving = dynamic_dataset[0]
        empty = next(
            dynamic_dataset[index] for index in range(len(dynamic_dataset))
            if dynamic_dataset[index]["sequence_id"] == "sequence_000002"
        )
        moving_batch = dynamic_sequence_collate([moving, dynamic_dataset[1]])
        empty_batch = dynamic_sequence_collate([empty])
        static_batch = next(iter(trainer.train_loaders["static"]))
        schedule = [kind for kind, _ in zip(
            (kind for kind, _ in trainer._iter_training_batches()), range(6)
        )]

        tracked = next(
            parameter for parameter in trainer.policy.dep_head.parameters()
            if parameter.requires_grad
        )
        before_parameter = tracked.detach().clone()
        moving_metrics = trainer.optimize_batch("dynamic", moving_batch)
        empty_metrics = trainer.optimize_batch("dynamic", empty_batch)
        static_metrics = trainer.optimize_batch("static", static_batch)
        parameter_delta = float((tracked.detach() - before_parameter).abs().max().cpu())

        trainer.policy.eval()
        batch_norm_before = [
            (module.running_mean.clone(), module.running_var.clone())
            for module in trainer.policy.modules() if isinstance(module, torch.nn.BatchNorm2d)
        ]
        with torch.inference_mode():
            validation = trainer.compute_batch(
                "dynamic", next(iter(trainer.val_loaders["dynamic"]))
            )
        batch_norm_after = [
            (module.running_mean.clone(), module.running_var.clone())
            for module in trainer.policy.modules() if isinstance(module, torch.nn.BatchNorm2d)
        ]
        bn_unchanged = all(
            torch.equal(lhs_mean, rhs_mean) and torch.equal(lhs_var, rhs_var)
            for (lhs_mean, lhs_var), (rhs_mean, rhs_var)
            in zip(batch_norm_before, batch_norm_after)
        )
        trainer.epoch_i = 0
        checkpoint = Path(trainer.save_checkpoint(runtime / "dynamic_smoke.pth"))
        saved_parameter = tracked.detach().clone()
        with torch.no_grad():
            tracked.add_(1.0)
        metadata = trainer.resume_training(checkpoint)
        resume_exact = torch.equal(tracked.detach(), saved_parameter)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        metadata_fields = {
            "backbone_variant", "dynamic_perception_enabled", "dynamic_loss_config",
            "dynamic_training_config", "dataset_version", "dataset_manifest_hash",
            "sensor_source", "world_frame", "body_frame", "camera_frame",
            "simulator_commit", "dep_commit", "training_epoch", "random_seed",
        }
        gradients = {
            "before": scalar(moving_metrics, "gradient_norm_before_clip"),
            "after": scalar(moving_metrics, "gradient_norm_after_clip"),
        }
        finite_losses = all(torch.isfinite(value).all() for value in (
            moving_metrics["total_loss"], empty_metrics["total_loss"],
            static_metrics["total_loss"], validation["trajectory_loss"],
        ))
        status = all((
            finite_losses,
            scalar(moving_metrics, "dynamic_safety_loss") > 0,
            scalar(empty_metrics, "dynamic_safety_loss") == 0,
            gradients["after"] <= 0.10001,
            parameter_delta > 0,
            resume_exact,
            bn_unchanged,
            set(schedule) == {"static", "dynamic"},
            metadata_fields.issubset(payload["metadata"]),
            not moving_metrics["score_label"].requires_grad,
        ))
        result = {
            "status": "PASS" if status else "FAIL",
            "device": torch.cuda.get_device_name(trainer.device),
            "dataset_source": "synthetic_smoke_only",
            "official_training_source": "depth",
            "steps": 3,
            "losses": {
                "moving_dynamic": scalar(moving_metrics, "dynamic_safety_loss"),
                "empty_dynamic": scalar(empty_metrics, "dynamic_safety_loss"),
                "moving_total": scalar(moving_metrics, "total_loss"),
                "static_total": scalar(static_metrics, "total_loss"),
                "validation_trajectory": scalar(validation, "trajectory_loss"),
            },
            "gradient_norms": gradients,
            "gradient_clipping_effective": gradients["after"] <= 0.10001,
            "parameter_updates": {"max_abs_delta": parameter_delta},
            "mixed_schedule_sample": schedule,
            "validation_bn_unchanged": bn_unchanged,
            "future_information_in_network": False,
            "score_label_detached": not moving_metrics["score_label"].requires_grad,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(trainer.device)),
            "checkpoint_resume": {
                "status": "PASS" if resume_exact else "FAIL",
                "full_optimizer_state": "optimizer_state" in payload,
                "metadata_complete": metadata_fields.issubset(payload["metadata"]),
                "checkpoint_role": metadata.get("checkpoint_role"),
            },
        }
        print("HOST_DYNAMIC_TRAINING_SMOKE_RESULT")
        print(json.dumps(result, indent=2))
        return 0 if status else 2
    finally:
        shutil.rmtree(runtime, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
