#!/usr/bin/env python3
"""Aggregate the Phase 8D three-strategy, three-seed formal shakedown."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics


DYNAMIC_METRICS = (
    "raw_dynamic_mean", "dynamic_cvar", "top1_collision_fraction",
    "selected_min_clearance", "safe_candidate_fraction", "oracle_regret",
    "spearman", "kendall", "score_loss",
)
STATIC_METRICS = (
    "static_cost", "guidance", "score_loss", "selected_min_clearance",
    "selected_clearance_mean", "offline_safe_selected_fraction",
    "static_score_label_mae", "selected_guidance_to_median_ratio",
)
EXPECTED_SEEDS = (8401, 8402, 8403)
EXPECTED_STEPS_PER_EPOCH = 1026
EXPECTED_EPOCHS = 2


def aggregate(items, metrics):
    return {metric: {
        "mean": statistics.fmean(float(item[metric]) for item in items),
        "min": min(float(item[metric]) for item in items),
        "max": max(float(item[metric]) for item in items),
    } for metric in metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--estimated-quality", type=Path, required=True)
    args = parser.parse_args()
    failures, runs = [], []
    for directory in sorted(path for path in args.runs.iterdir()
                            if path.is_dir() and "_seed" in path.name):
        status_path = directory / "status.json"
        if not status_path.is_file():
            failures.append(f"{directory.name}: status.json missing")
            continue
        status = json.loads(status_path.read_text())
        validation = status.get("validation_metrics", {})
        suites = validation.get("suites", {})
        if set(suites) != {"valid_gt", "valid_estimated", "valid_static"}:
            failures.append(f"{directory.name}: fixed validation suites missing")
            continue
        parts = directory.name.rsplit("_seed", 1)
        strategy, seed = parts[0], int(parts[1])
        expected_steps = EXPECTED_STEPS_PER_EPOCH * EXPECTED_EPOCHS
        if status.get("state") != "complete" or int(status.get("global_step", -1)) != expected_steps:
            failures.append(f"{directory.name}: expected complete at {expected_steps} steps")
        schedule = status.get("epoch_schedule", {})
        required_schedule = {
            "static_batch_count": 513, "dynamic_batch_count": 513,
            "static_loader_recycle_count": 0, "dynamic_loader_recycle_count": 0,
        }
        for key, expected in required_schedule.items():
            if schedule.get(key) != expected:
                failures.append(f"{directory.name}: {key}={schedule.get(key)!r}, expected {expected}")
        if abs(float(schedule.get("dynamic_equivalent_passes", -1)) - 1.0) > 1e-12:
            failures.append(f"{directory.name}: dynamic equivalent passes is not 1.0")
        if not bool(suites["valid_static"].get("static_smoke_pass")):
            failures.append(f"{directory.name}: real static smoke failed")
        if float(suites["valid_estimated"].get("no_target_dynamic_max_abs", 1)) != 0.0:
            failures.append(f"{directory.name}: no-target dynamic cost is nonzero")
        checkpoints = {}
        for name in ("latest.pt", "best_dynamic.pt", "best_balanced.pt"):
            path = directory / "checkpoints" / name
            checkpoints[name] = bool(
                path.is_file() and path.with_suffix(path.suffix + ".sha256").is_file()
            )
        if not checkpoints["latest.pt"] or not checkpoints["best_dynamic.pt"]:
            failures.append(f"{directory.name}: latest/best_dynamic checkpoint missing")
        with (directory / "step_metrics.csv").open(newline="") as stream:
            resources_rows = list(csv.DictReader(stream))
        resources = {"records": len(resources_rows)}
        if resources_rows:
            first, last = resources_rows[0], resources_rows[-1]
            resources.update({
                "cpu_rss_growth_bytes": int(float(last["cpu_rss_bytes"]) - float(first["cpu_rss_bytes"])),
                "worker_rss_peak_bytes": max(int(float(row.get("worker_rss_bytes", 0))) for row in resources_rows),
                "gpu_reserved_growth_bytes": int(float(last["gpu_reserved_bytes"]) - float(first["gpu_reserved_bytes"])),
                "file_descriptor_growth": int(float(last["file_descriptors"]) - float(first["file_descriptors"])),
                "file_descriptor_peak_above_first": max(
                    int(float(row["file_descriptors"])) for row in resources_rows
                ) - int(float(first["file_descriptors"])),
                "static_cache_capacity": int(float(last.get("static_cache_capacity", 0))),
                "batch_latency_p95_ms_max": max(float(row["batch_latency_p95_ms"]) for row in resources_rows),
            })
            if resources["cpu_rss_growth_bytes"] > 512 * 2**20:
                failures.append(f"{directory.name}: CPU RSS growth exceeds 512 MiB")
            if resources["gpu_reserved_growth_bytes"] > 256 * 2**20:
                failures.append(f"{directory.name}: GPU reserved growth exceeds 256 MiB")
            # Worker iterator rollover opens/closes a bounded group of pipes,
            # so endpoint deltas are phase-sensitive. Fail only if the observed
            # envelope grows beyond one eight-worker pipe group.
            if resources["file_descriptor_peak_above_first"] > 32:
                failures.append(f"{directory.name}: file descriptor envelope exceeds 32")
            if resources["static_cache_capacity"] != 128:
                failures.append(f"{directory.name}: unexpected static cache capacity")
        else:
            failures.append(f"{directory.name}: resource records missing")
        runs.append({
            "name": directory.name, "strategy": strategy, "seed": seed,
            "global_step": int(status.get("global_step", -1)),
            "valid_estimated": suites["valid_estimated"],
            "valid_gt": suites["valid_gt"], "valid_static": suites["valid_static"],
            "estimated_gt_dynamic_cvar_gap": validation["estimated_gt_dynamic_cvar_gap"],
            "estimated_gt_top1_collision_gap": validation["estimated_gt_top1_collision_gap"],
            "checkpoint_state": checkpoints, "resources": resources,
            "training": {
                "train_loss": status.get("train_loss"),
                "dynamic_loss": status.get("dynamic_loss"),
                "static_loss": status.get("static_loss"),
                "score_loss": status.get("score_loss"),
                "samples_per_second": status.get("samples_per_second"),
                "pcgrad_last_interval": resources_rows[-1] if resources_rows else None,
            },
        })

    grouped = defaultdict(list)
    for run in runs:
        grouped[run["strategy"]].append(run)
    summaries = {}
    for strategy, values in sorted(grouped.items()):
        seeds = sorted(item["seed"] for item in values)
        if seeds != list(EXPECTED_SEEDS):
            failures.append(f"{strategy}: expected seeds {EXPECTED_SEEDS}, got {seeds}")
        estimated = aggregate([item["valid_estimated"] for item in values], DYNAMIC_METRICS)
        gt = aggregate([item["valid_gt"] for item in values], DYNAMIC_METRICS)
        static = aggregate([item["valid_static"] for item in values], STATIC_METRICS)
        summaries[strategy] = {
            "seeds": seeds, "valid_estimated": estimated, "valid_gt": gt,
            "valid_static": static,
            "estimated_gt_dynamic_cvar_gap_mean": statistics.fmean(
                float(item["estimated_gt_dynamic_cvar_gap"]) for item in values
            ),
            "estimated_gt_top1_collision_gap_mean": statistics.fmean(
                float(item["estimated_gt_top1_collision_gap"]) for item in values
            ),
            "all_static_smoke_pass": all(item["valid_static"]["static_smoke_pass"] for item in values),
            "all_required_checkpoints_present": all(
                item["checkpoint_state"]["latest.pt"] and item["checkpoint_state"]["best_dynamic.pt"]
                for item in values
            ),
        }
    selected = None
    if len(summaries) == 3:
        feasible = [
            (name, item) for name, item in summaries.items()
            if item["valid_estimated"]["top1_collision_fraction"]["mean"] <= 0.05
            and item["estimated_gt_dynamic_cvar_gap_mean"] <= 0.25
            and item["estimated_gt_top1_collision_gap_mean"] <= 0.10
            and item["all_static_smoke_pass"]
        ]
        if feasible:
            selected = min(feasible, key=lambda pair: (
                pair[1]["valid_estimated"]["top1_collision_fraction"]["mean"],
                pair[1]["valid_estimated"]["dynamic_cvar"]["mean"],
                -pair[1]["valid_estimated"]["selected_min_clearance"]["mean"],
                pair[1]["valid_estimated"]["oracle_regret"]["mean"],
            ))[0]
        else:
            failures.append("no strategy meets fixed-validation hard constraints")
    resume_path = args.runs / "fixed_025_seed8401" / "resume_audit_before.json"
    resume = json.loads(resume_path.read_text()) if resume_path.is_file() else {"status": "FAIL"}
    resume_pass = bool(
        resume.get("status") == "PASS"
        and next((run for run in runs if run["name"] == "fixed_025_seed8401"), {}).get("global_step")
        == EXPECTED_STEPS_PER_EPOCH * EXPECTED_EPOCHS
    )
    if not resume_pass:
        failures.append("epoch-boundary latest checkpoint resume audit failed")
    estimated_quality = json.loads(args.estimated_quality.read_text())
    estimated_allowed = bool(
        estimated_quality.get("status") == "PASS"
        and estimated_quality.get("recommendation", {}).get(
            "estimated_context_training_sufficient"
        )
    )
    recommendation = None if selected is None else {
        "pcgrad_strategy": selected,
        "steps_per_epoch": EXPECTED_STEPS_PER_EPOCH,
        "dynamic_passes_per_epoch": 1.0,
        "estimated_context_start_epoch": (
            estimated_quality["recommendation"]["recommended_start_epoch"]
            if estimated_allowed else None
        ),
        "estimated_context_ratio": (
            estimated_quality["recommendation"]["recommended_ratio"]
            if estimated_allowed else 0.0
        ),
        "learning_rate": 0.00015,
        "recommended_epochs": 30,
        "early_stopping_patience": 8,
        "selection_source": "fixed validation suites, never training loss",
        "estimated_context_blocked_by_quality_gate": not estimated_allowed,
    }
    payload = {
        "status": "PASS" if not failures else "FAIL",
        "run_count": len(runs), "epochs_per_run": EXPECTED_EPOCHS,
        "steps_per_epoch": EXPECTED_STEPS_PER_EPOCH,
        "optimizer_steps_total": sum(max(run["global_step"], 0) for run in runs),
        "seed_count_per_strategy": 3, "runs": runs, "strategies": summaries,
        "selected_strategy": selected, "recommendation": recommendation,
        "resume_audit": resume, "resume_passed": resume_pass,
        "estimated_context_quality_status": estimated_quality.get("status"),
        "test_split_used_for_selection": False,
        "scalar_training_loss_used_for_selection": False,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
