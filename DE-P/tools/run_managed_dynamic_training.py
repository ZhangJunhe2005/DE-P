#!/usr/bin/env python3
"""Managed finite/production training with atomic state and checkpoints."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
from scipy.stats import kendalltau, spearmanr
import torch
from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loss.dynamic_types import DynamicLossConfig, DynamicObjectiveConfig, RiskMetricsConfig
from policy.dep_trainer import DepTrainer
from policy.dynamic_training_config import DynamicTrainingConfig
from policy.training_schedule import TrainingScheduleConfig
from train_dep import configure_random_seed
from tools.phase8c_checkpoint_selection import CheckpointSelector
from tools.phase8c_validation_metrics import evaluate_validation
from tools.phase8d_production_gate import validate_phase8d_gate


STOP_REQUESTED = False


class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)
    def flush(self):
        for stream in self.streams:
            stream.flush()


def atomic_json(path, payload):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def git_state(path):
    revision = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                              capture_output=True, text=True)
    diff = subprocess.run(["git", "-C", str(path), "diff", "--binary"],
                          capture_output=True)
    return {"commit": revision.stdout.strip() if revision.returncode == 0 else "unknown",
            "diff_sha256": hashlib.sha256(diff.stdout).hexdigest(),
            "dirty": bool(diff.stdout)}


def scalar(value):
    return float(value.detach().cpu()) if torch.is_tensor(value) else float(value)


def current_rss_bytes():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return 0


def worker_rss_bytes():
    """Return aggregate RSS for direct DataLoader worker children."""
    total = 0
    children = Path(f"/proc/{os.getpid()}/task/{os.getpid()}/children")
    if not children.is_file():
        return 0
    for value in children.read_text().split():
        status = Path("/proc") / value / "status"
        if not status.is_file():
            continue
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                total += int(line.split()[1]) * 1024
                break
    return total


def static_cache_entries(trainer):
    loader = trainer.train_loaders.get("static")
    dataset = loader.dataset if loader is not None else None
    cache = getattr(dataset, "_depth_cache", None)
    return len(cache) if cache is not None else 0


def training_risk_metrics(kind, details, cvar_fraction):
    result = {
        "raw_dynamic_mean": 0.0, "dynamic_cvar": 0.0,
        "top1_collision_fraction": 0.0, "selected_min_clearance": 0.0,
        "safe_candidate_fraction": scalar(details["safe_candidate_fraction"]),
        "oracle_regret": 0.0, "spearman": 0.0, "kendall": 0.0,
        "static_cost": scalar(details["static_safety_loss"]),
        "guidance": scalar(details["guidance_loss"]),
        "smoothness": scalar(details["smooth_loss"]),
        "score_loss": scalar(details["score_loss"]),
    }
    if kind != "dynamic" or details["dynamic_diagnostics"] is None:
        return result
    diagnostics = details["dynamic_diagnostics"]
    target = (diagnostics.dynamic_obstacle_count > 0).detach().cpu().numpy()
    if not bool(target.any()):
        return result
    raw = details["candidate_dynamic_cost_raw"].reshape(-1, 15).detach().cpu().numpy()[target]
    score = details["predicted_score"].reshape(-1, 15).detach().cpu().numpy()[target]
    clearance = diagnostics.candidate_min_clearance.detach().cpu().numpy()[target]
    selected = score.argmin(axis=1)
    rows = np.arange(len(raw))
    selected_clearance = clearance[rows, selected]
    top_count = max(1, int(np.ceil(raw.shape[1] * float(cvar_fraction))))
    correlations, kendalls = [], []
    for predicted, risk in zip(score, raw):
        correlations.append(float(np.nan_to_num(spearmanr(predicted, risk).statistic)))
        kendalls.append(float(np.nan_to_num(kendalltau(predicted, risk).statistic)))
    finite = selected_clearance[np.isfinite(selected_clearance)]
    result.update({
        "raw_dynamic_mean": float(raw.mean()),
        "dynamic_cvar": float(np.sort(raw, axis=1)[:, -top_count:].mean()),
        "top1_collision_fraction": float((selected_clearance < 0).mean()),
        "selected_min_clearance": float(finite.min()) if finite.size else -float("inf"),
        "safe_candidate_fraction": float((clearance >= 0).mean()),
        "oracle_regret": float((raw[rows, selected] - raw.min(axis=1)).mean()),
        "spearman": float(np.mean(correlations)), "kendall": float(np.mean(kendalls)),
    })
    return result


def install_signals():
    def request_stop(signum, _frame):
        global STOP_REQUESTED
        STOP_REQUESTED = True
        print(f"safe stop requested by signal {signum}; checkpoint after current batch")
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)


def unique_run_dir(base, config_version):
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for suffix in range(1000):
        name = f"{stamp}-{config_version}" + (f"-{suffix}" if suffix else "")
        candidate = base / name
        try:
            candidate.mkdir(parents=True)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError("could not allocate a unique run directory")


def require_production_authorization(confirmed, config_path, gate_path=None):
    """Production recognizes only the complete fail-closed Phase 8D gate."""
    result = validate_phase8d_gate(config_path, gate_path)
    if not confirmed:
        raise RuntimeError("production requires explicit --yes after reviewing Phase 8D gate")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    config = YAML(typ="safe").load(config_path)
    if config["run_kind"] == "production":
        authorization = require_production_authorization(args.yes, config_path)
        print(json.dumps({"phase8d_authorization": authorization}, indent=2))
    manifest = Path(config["dataset_manifest"])
    checkpoint = Path(config["initialization_checkpoint"])
    if not manifest.is_file() or not checkpoint.is_file():
        raise FileNotFoundError("dataset manifest or explicit initialization checkpoint is missing")
    if not torch.cuda.is_available():
        raise RuntimeError("managed dynamic training requires host CUDA full-access")
    configure_random_seed(int(config["random_seed"]))
    run_dir = (args.run_dir.expanduser().resolve() if args.run_dir else
               unique_run_dir(ROOT / "runs/dynamic", config["config_version"]))
    for directory in ("tensorboard", "checkpoints"):
        (run_dir / directory).mkdir(parents=True, exist_ok=True)
    log_stream = (run_dir / "train.log").open("a", encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_stream)
    sys.stderr = Tee(sys.__stderr__, log_stream)
    if not (run_dir / "config.yaml").exists():
        shutil.copy2(config_path, run_dir / "config.yaml")
    elif args.resume:
        shutil.copy2(config_path, run_dir / "resume_config.yaml")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    atomic_json(run_dir / "environment.json", {
        "python": sys.version, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "capability": torch.cuda.get_device_capability(0),
    })
    atomic_json(run_dir / "git_state.json", {
        "dep": git_state(ROOT), "simulator": git_state(ROOT.parent / "Simulator")
    })
    (run_dir / "pid").write_text(str(os.getpid()) + "\n", encoding="ascii")
    (run_dir / "pid_start_time").write_text(
        Path("/proc/self/stat").read_text().split()[21] + "\n", encoding="ascii"
    )
    install_signals()

    training_config = DynamicTrainingConfig.from_mapping(config["dynamic_training"])
    training_schedule = TrainingScheduleConfig.from_mapping(config["training_schedule"])
    loss_config = DynamicLossConfig.from_mapping(config["dynamic_loss"])
    objective_config = (DynamicObjectiveConfig.from_mapping(config["dynamic_objective"])
                        if "dynamic_objective" in config else DynamicObjectiveConfig.from_global_config())
    risk_config = (RiskMetricsConfig.from_mapping(config["risk_metrics"])
                   if "risk_metrics" in config else RiskMetricsConfig.from_global_config())
    trainer = DepTrainer(
        learning_rate=float(config["learning_rate"]), batch_size=int(config["batch_size"]),
        loss_weight=config["loss_weight"], tensorboard_path=str(run_dir / "tensorboard"),
        checkpoint_path=None if args.resume else str(checkpoint), save_on_exit=False,
        backbone_variant=config["architecture"], dataset_mode=config["dataset_mode"],
        dynamic_data_root=config["dataset_root"], freeze_policy=config["freeze_policy"],
        random_seed=int(config["random_seed"]), num_workers=int(config["num_workers"]),
        training_config_override=training_config, dynamic_loss_config_override=loss_config,
        dynamic_objective_config_override=objective_config,
        risk_metrics_config_override=risk_config,
        static_map_catalog_override=config["static_map_catalog"],
        dynamic_map_catalog_override=config["dynamic_map_catalog"],
        training_schedule_override=training_schedule,
        static_cache_size=int(config["static_dataset"]["cache_size"]),
    )
    scheduler_config = config["scheduler"]
    trainer.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        trainer.optimizer, mode="min", factor=float(scheduler_config["factor"]),
        patience=int(scheduler_config["patience"]), min_lr=float(scheduler_config["min_lr"]),
    )
    if args.resume:
        trainer.resume_training(args.resume)
    metrics_path = run_dir / "metrics.csv"
    fields = [
        "epoch", "split", "total_loss", "trajectory_loss", "score_loss",
        "dynamic_safety_loss", "raw_dynamic_mean", "dynamic_cvar",
        "top1_collision_fraction", "selected_min_clearance", "safe_candidate_fraction",
        "oracle_regret", "spearman", "kendall", "no_target_dynamic_mean",
        "static_safety_loss", "static_cost", "guidance", "smoothness",
        "pcgrad_dynamic_gradient_norm", "pcgrad_non_dynamic_gradient_norm",
        "pcgrad_activated", "pcgrad_fallback_no_target", "pcgrad_fallback_safe",
        "pcgrad_fallback_small_gradient", "samples_per_second",
    ]
    if not metrics_path.exists():
        with metrics_path.open("w", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=fields).writeheader()
    step_metrics_path = run_dir / "step_metrics.csv"
    step_fields = [
        "global_step", "epoch", "raw_dynamic_mean", "dynamic_cvar",
        "top1_collision_fraction", "selected_min_clearance", "safe_candidate_fraction",
        "oracle_regret", "spearman", "kendall", "static_cost", "guidance",
        "smoothness", "score_loss", "pcgrad_dynamic_gradient_norm",
        "pcgrad_non_dynamic_gradient_norm", "pcgrad_applied_ratio", "pcgrad_activated",
        "pcgrad_fallback_no_target", "pcgrad_fallback_safe",
        "pcgrad_fallback_small_gradient", "batch_latency_p50_ms", "batch_latency_p95_ms",
        "gpu_allocated_bytes", "gpu_reserved_bytes", "cpu_rss_bytes", "file_descriptors",
        "worker_rss_bytes", "static_cache_entries", "static_cache_capacity",
    ]
    if not step_metrics_path.exists():
        with step_metrics_path.open("w", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=step_fields).writeheader()
    selection_config = config["checkpoint_selection"]
    validation_limit = config["validation"].get("max_batches_per_suite")
    bad_epochs = 0
    started = time.monotonic()
    status_path = run_dir / "status.json"
    baseline_path = run_dir / "selection_baseline.json"
    if baseline_path.is_file():
        selection_baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    else:
        selection_baseline = evaluate_validation(
            trainer, validation_limit
        )
        atomic_json(baseline_path, selection_baseline)
    selector = CheckpointSelector(selection_config, selection_baseline)
    if args.resume and status_path.is_file():
        previous_status = json.loads(status_path.read_text(encoding="utf-8"))
        if previous_status.get("checkpoint_selection"):
            selector.load_state_dict(previous_status["checkpoint_selection"])
    state = "running"
    last_checkpoint = None
    interval_sums, interval_count, interval_latencies = {}, 0, []
    try:
        for epoch in range(trainer.start_epoch, int(config["epochs"])):
            trainer.epoch_i = epoch
            trainer._apply_curriculum(epoch)
            trainer.set_training_mode()
            sums, count, epoch_start = {}, 0, time.monotonic()
            epoch_latencies = []
            category_counts = Counter()
            for kind, batch in trainer._iter_training_batches():
                wait = time.monotonic() - epoch_start if count == 0 else 0.0
                if wait > float(config["abort_thresholds"]["dataloader_stall_seconds"]):
                    raise TimeoutError("DataLoader initial batch stall threshold exceeded")
                batch_started = time.monotonic()
                metrics = trainer.optimize_batch(kind, batch)
                batch_latency = time.monotonic() - batch_started
                epoch_latencies.append(batch_latency)
                interval_latencies.append(batch_latency)
                trainer.global_step += 1
                count += 1
                interval_count += 1
                if kind == "dynamic":
                    category_counts.update(batch.get("sample_category", []))
                for key in ("total_loss", "trajectory_loss", "score_loss",
                            "dynamic_safety_loss", "static_safety_loss",
                            "pcgrad_dynamic_gradient_norm",
                            "pcgrad_non_dynamic_gradient_norm", "pcgrad_activated",
                            "pcgrad_applied_ratio",
                            "pcgrad_fallback_no_target", "pcgrad_fallback_safe",
                            "pcgrad_fallback_small_gradient"):
                    sums[key] = sums.get(key, 0.0) + scalar(metrics[key])
                risk_values = training_risk_metrics(
                    kind, metrics, trainer.risk_metrics_config.cvar_fraction
                )
                for key, value in {**risk_values, **{
                    "pcgrad_dynamic_gradient_norm": scalar(metrics["pcgrad_dynamic_gradient_norm"]),
                    "pcgrad_non_dynamic_gradient_norm": scalar(metrics["pcgrad_non_dynamic_gradient_norm"]),
                    "pcgrad_applied_ratio": scalar(metrics["pcgrad_applied_ratio"]),
                    "pcgrad_activated": scalar(metrics["pcgrad_activated"]),
                    "pcgrad_fallback_no_target": scalar(metrics["pcgrad_fallback_no_target"]),
                    "pcgrad_fallback_safe": scalar(metrics["pcgrad_fallback_safe"]),
                    "pcgrad_fallback_small_gradient": scalar(metrics["pcgrad_fallback_small_gradient"]),
                }}.items():
                    interval_sums[key] = interval_sums.get(key, 0.0) + float(value)
                if trainer.global_step % 100 == 0:
                    row = {key: interval_sums.get(key, 0.0) / interval_count
                           for key in step_fields}
                    row.update({
                        "global_step": trainer.global_step, "epoch": epoch,
                        "batch_latency_p50_ms": float(np.percentile(interval_latencies, 50) * 1000),
                        "batch_latency_p95_ms": float(np.percentile(interval_latencies, 95) * 1000),
                        "gpu_allocated_bytes": torch.cuda.memory_allocated(),
                        "gpu_reserved_bytes": torch.cuda.memory_reserved(),
                        "cpu_rss_bytes": current_rss_bytes(),
                        "worker_rss_bytes": worker_rss_bytes(),
                        "static_cache_entries": static_cache_entries(trainer),
                        "static_cache_capacity": int(config["static_dataset"]["cache_size"]),
                        "file_descriptors": len(tuple(Path("/proc/self/fd").iterdir())),
                    })
                    with step_metrics_path.open("a", newline="", encoding="utf-8") as stream:
                        csv.DictWriter(stream, fieldnames=step_fields).writerow(row)
                    interval_sums, interval_count, interval_latencies = {}, 0, []
                if scalar(metrics["gradient_norm_before_clip"]) > float(
                        config["abort_thresholds"]["max_gradient_norm"]):
                    raise FloatingPointError("gradient watchdog threshold exceeded")
                max_steps = config.get("max_steps_per_epoch")
                if STOP_REQUESTED or (max_steps is not None and count >= int(max_steps)):
                    break
            train_row = {key: sums[key] / count for key in sums}
            train_row.update(epoch=epoch, split="train",
                             samples_per_second=count * int(config["batch_size"]) /
                             max(time.monotonic() - epoch_start, 1e-6))
            validation_started = time.monotonic()
            validation = evaluate_validation(
                trainer, validation_limit
            )
            validation_seconds = time.monotonic() - validation_started
            valid_row = dict(validation)
            valid_row.update(epoch=epoch, split="valid", samples_per_second=0.0)
            with metrics_path.open("a", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writerow({key: train_row.get(key, 0.0) for key in fields})
                writer.writerow({key: valid_row.get(key, 0.0) for key in fields})
            decision = selector.consider(validation)
            scheduler_metric = (
                decision["balanced_score"] if decision["balanced_feasible"]
                else validation["dynamic_cvar"]
            )
            trainer.scheduler.step(scheduler_metric)
            selection_metadata = {
                "selection_validation_metrics": validation,
                "selection_decision": decision,
                "selection_rules": selection_config,
                "selection_baseline": selection_baseline,
            }
            latest = run_dir / "checkpoints/latest.pt"
            checkpoint_started = time.monotonic()
            last_checkpoint = trainer.save_checkpoint(
                latest, {**selection_metadata, "selection_role": "latest"}
            )
            if (epoch + 1) % int(config["save_interval"]) == 0:
                trainer.save_checkpoint(
                    run_dir / f"checkpoints/epoch_{epoch + 1:04d}.pt",
                    {**selection_metadata, "selection_role": "epoch"},
                )
            if decision["dynamic_improved"]:
                trainer.save_checkpoint(
                    run_dir / "checkpoints/best_dynamic.pt",
                    {**selection_metadata, "selection_role": "best_dynamic"},
                )
            if decision["balanced_improved"]:
                trainer.save_checkpoint(
                    run_dir / "checkpoints/best_balanced.pt",
                    {**selection_metadata, "selection_role": "best_balanced"},
                )
            selection_improved = (
                decision["balanced_improved"] if decision["balanced_feasible"]
                else decision["dynamic_improved"]
            )
            bad_epochs = 0 if selection_improved else bad_epochs + 1
            free_gib = shutil.disk_usage(run_dir).free / 2**30
            if free_gib < float(config["abort_thresholds"]["min_free_disk_gib"]):
                raise OSError("free disk watchdog threshold exceeded")
            elapsed = time.monotonic() - started
            atomic_json(status_path, {
                "state": "stopping" if STOP_REQUESTED else "running", "epoch": epoch + 1,
                "global_step": trainer.global_step, "train_loss": train_row["total_loss"],
                "valid_loss": valid_row["total_loss"],
                "dynamic_loss": train_row["dynamic_safety_loss"],
                "static_loss": train_row["static_safety_loss"],
                "score_loss": train_row["score_loss"],
                "validation_metrics": validation,
                "checkpoint_decision": decision,
                "checkpoint_selection": selector.state_dict(),
                "scheduler_metric": scheduler_metric,
                "learning_rate": trainer.optimizer.param_groups[0]["lr"],
                "gpu_memory": torch.cuda.max_memory_allocated(),
                "samples_per_second": train_row["samples_per_second"],
                "validation_seconds": validation_seconds,
                "last_checkpoint": last_checkpoint,
                "estimated_remaining_time": elapsed / (epoch + 1) * (int(config["epochs"]) - epoch - 1),
                "curriculum_stage": trainer.current_curriculum,
                "sample_category_counts": dict(sorted(category_counts.items())),
                "batch_latency_p50_ms": float(np.percentile(epoch_latencies, 50) * 1000),
                "batch_latency_p95_ms": float(np.percentile(epoch_latencies, 95) * 1000),
                "cpu_rss_bytes": current_rss_bytes(),
                "worker_rss_bytes": worker_rss_bytes(),
                "static_cache_entries": static_cache_entries(trainer),
                "static_cache_capacity": int(config["static_dataset"]["cache_size"]),
                "file_descriptors": len(tuple(Path("/proc/self/fd").iterdir())),
                "gpu_reserved_bytes": torch.cuda.memory_reserved(),
                "checkpoint_write_seconds": time.monotonic() - checkpoint_started,
                "steps_per_epoch": trainer._mixed_epoch_length(),
                "epoch_schedule": trainer.last_epoch_schedule_stats,
                "validation_suite_version": validation["validation_suite_version"],
            })
            if STOP_REQUESTED or bad_epochs >= int(config["early_stopping"]["patience"]):
                state = "stopped" if STOP_REQUESTED else "early_stopped"
                break
        else:
            state = "complete"
    except Exception as error:
        state = "failed"
        atomic_json(status_path, {"state": state, "epoch": getattr(trainer, "epoch_i", -1),
                    "global_step": trainer.global_step, "error": repr(error),
                    "last_checkpoint": last_checkpoint})
        raise
    finally:
        if state != "failed":
            current = json.loads(status_path.read_text()) if status_path.exists() else {}
            current["state"] = state
            atomic_json(status_path, current)
        trainer.tensorboard_log.close()
        (run_dir / "pid").unlink(missing_ok=True)
        (run_dir / "pid_start_time").unlink(missing_ok=True)
    print(json.dumps({"status": "PASS", "state": state, "run_dir": str(run_dir),
                      "last_checkpoint": last_checkpoint}, indent=2))


if __name__ == "__main__":
    main()
