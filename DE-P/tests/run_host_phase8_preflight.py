#!/usr/bin/env python3
"""Host-GPU Phase-8 gates: overfit, bounded run, resume, sanity, resources."""

from __future__ import annotations

import json
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time

import numpy as np
from scipy.stats import kendalltau, spearmanr
import torch
from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loss.dynamic_types import DynamicLossConfig
from policy.dep_trainer import DepTrainer
from policy.dynamic_collate import dynamic_sequence_collate
from policy.dynamic_training_config import DynamicTrainingConfig
from tests.baseline_helpers import seed_everything


def value(metrics, key):
    item = metrics[key]
    return float(item.detach().cpu()) if torch.is_tensor(item) else float(item)


def model_metrics(trainer, batch):
    trainer.policy.eval()
    with torch.inference_mode():
        details = trainer.compute_batch("dynamic", batch)
        _, scores = trainer.policy.inference(
            batch["current_depth"].to(trainer.device),
            batch["observation_9d"].to(trainer.device),
            dynamic_context=batch["dynamic_context"].to(trainer.device),
        )
    predicted = scores.reshape(-1).detach().cpu().numpy()
    risk = details["dynamic_score_label"].detach().cpu().numpy()
    top = int(np.argmin(predicted))
    return {
        "total": value(details, "trajectory_loss") + value(details, "score_loss"),
        "score": value(details, "score_loss"),
        "dynamic": value(details, "dynamic_safety_loss"),
        "static": value(details, "static_safety_loss"),
        "spearman": float(np.nan_to_num(spearmanr(predicted, risk).statistic)),
        "kendall": float(np.nan_to_num(kendalltau(predicted, risk).statistic)),
        "top1_risk": float(risk[top]),
        "oracle_regret": float(risk[top] - risk.min()),
        "min_dynamic_distance": value(details, "min_dynamic_distance"),
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run with full access")
    seed_everything(8008)
    config_path = ROOT / "configs/train_dynamic_preflight.yaml"
    config = YAML(typ="safe").load(config_path)
    training_config = DynamicTrainingConfig.from_mapping(config["dynamic_training"])
    loss_config = DynamicLossConfig.from_mapping(config["dynamic_loss"])
    runtime = Path(tempfile.mkdtemp(prefix="dep-phase8-preflight-"))
    trainer = DepTrainer(
        learning_rate=config["learning_rate"], batch_size=4,
        tensorboard_path=str(runtime / "tb"), checkpoint_path=config["initialization_checkpoint"],
        backbone_variant="corrected", dataset_mode="dynamic",
        dynamic_data_root=config["dataset_root"], freeze_policy="late",
        num_workers=0, random_seed=8008, training_config_override=training_config,
        dynamic_loss_config_override=loss_config,
    )
    dataset = trainer.train_loaders["dynamic"].dataset
    wanted = ["no_target", "visible_high_risk", "visible_high_risk", "multi_target"]
    indices, used = [], set()
    for wanted_category in wanted:
        index = next(index for index, category in enumerate(dataset.sample_categories)
                     if category == wanted_category and index not in used)
        indices.append(index); used.add(index)
    batch = dynamic_sequence_collate([dataset[index] for index in indices])
    no_target_index = next(index for index, category in enumerate(dataset.sample_categories)
                           if category == "no_target")
    no_target_batch = dynamic_sequence_collate([dataset[no_target_index]])
    torch.cuda.reset_peak_memory_stats()
    baseline = model_metrics(trainer, batch)
    no_target_before = model_metrics(trainer, no_target_batch)
    tracked = next(parameter for parameter in trainer.policy.dep_head.parameters()
                   if parameter.requires_grad)
    parameter_before = tracked.detach().clone()
    losses = []
    memory_samples = []
    rss_samples = []
    fd_samples = []
    started = time.monotonic()
    for step in range(30):
        trainer.set_training_mode()
        metrics = trainer.optimize_batch("dynamic", batch)
        trainer.global_step += 1
        losses.append(value(metrics, "total_loss"))
        if step >= 5:
            memory_samples.append(torch.cuda.memory_allocated())
            rss_samples.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
            fd_samples.append(len(list(Path("/proc/self/fd").iterdir())))
    elapsed = time.monotonic() - started
    final = model_metrics(trainer, batch)
    no_target_after = model_metrics(trainer, no_target_batch)
    parameter_delta = float((tracked.detach() - parameter_before).abs().max().cpu())
    trainer.epoch_i = 0
    checkpoint = Path(trainer.save_checkpoint(runtime / "resume.pt"))
    saved = tracked.detach().clone()
    with torch.no_grad(): tracked.add_(1)
    trainer.resume_training(checkpoint)
    resume_exact = torch.equal(saved, tracked.detach())

    bounded = subprocess.run([
        sys.executable, str(ROOT / "tools/run_managed_dynamic_training.py"),
        "--config", str(config_path), "--yes",
    ], text=True, capture_output=True, timeout=300)
    bounded_json = None
    if bounded.returncode == 0:
        start = bounded.stdout.rfind("{\n")
        bounded_json = json.loads(bounded.stdout[start:])
    bounded_run = Path(bounded_json["run_dir"]) if bounded_json else None
    bounded_artifacts = bool(
        bounded_run and (bounded_run / "checkpoints/latest.pt").is_file()
        and (bounded_run / "checkpoints/best.pt").is_file()
        and (bounded_run / "metrics.csv").is_file()
        and json.loads((bounded_run / "status.json").read_text())["state"] == "complete"
    )
    memory_growth = max(memory_samples[-5:]) - min(memory_samples[:5])
    rss_growth = max(rss_samples[-5:]) - min(rss_samples[:5])
    fd_growth = max(fd_samples[-5:]) - min(fd_samples[:5])
    gates = {
        "small_set_total_loss_decreased": final["total"] < baseline["total"] * 0.99,
        "small_set_score_loss_decreased": final["score"] < baseline["score"],
        "small_set_dynamic_loss_decreased": final["dynamic"] < baseline["dynamic"],
        "no_target_dynamic_exact_zero": no_target_before["dynamic"] == 0.0 and no_target_after["dynamic"] == 0.0,
        "model_parameter_updated": parameter_delta > 0,
        "finite_training": bool(np.isfinite(losses).all()),
        "checkpoint_resume_exact": resume_exact,
        "bounded_one_epoch": bounded.returncode == 0 and bounded_artifacts,
        "static_regression_bounded": final["static"] <= max(baseline["static"] * 1.5, baseline["static"] + 1e-5),
        "score_risk_not_worse": final["spearman"] >= baseline["spearman"] - 0.05,
        "oracle_regret_not_worse": final["oracle_regret"] <= baseline["oracle_regret"] + 0.05,
        "gpu_memory_stable": memory_growth <= config["abort_thresholds"]["gpu_memory_growth_bytes"],
        "cpu_memory_stable": rss_growth <= 128 * 2**20,
        "file_descriptors_stable": fd_growth <= 2,
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    result = {
        "status": status, "production_ready": status == "PASS",
        "device": torch.cuda.get_device_name(0), "compute_capability": torch.cuda.get_device_capability(0),
        "steps": 30, "seconds": elapsed, "steps_per_second": 30 / elapsed,
        "baseline": baseline, "after_overfit": final,
        "no_target_before": no_target_before, "no_target_after": no_target_after,
        "parameter_max_abs_delta": parameter_delta,
        "checkpoint_resume_exact": resume_exact,
        "bounded_run": str(bounded_run) if bounded_run else None,
        "bounded_stdout_tail": bounded.stdout[-3000:], "bounded_stderr_tail": bounded.stderr[-3000:],
        "resources": {"peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
                      "gpu_memory_growth_bytes": memory_growth,
                      "cpu_rss_growth_bytes": rss_growth, "fd_growth": fd_growth},
        "gates": gates,
    }
    output = ROOT / "reports/phase8_preflight_result.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("PHASE8_PREFLIGHT_RESULT")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if status == "PASS" else 2)


if __name__ == "__main__":
    main()
