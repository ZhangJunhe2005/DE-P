#!/usr/bin/env python3
"""Aggregate the 4-strategy x 3-seed bounded training comparison."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics


METRICS = (
    "raw_dynamic_mean", "dynamic_cvar", "top1_collision_fraction",
    "selected_min_clearance", "safe_candidate_fraction", "oracle_regret",
    "spearman", "kendall", "static_cost", "guidance", "smoothness", "score_loss",
)


def mean(values):
    return statistics.fmean(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    failures, runs = [], []
    for directory in sorted(path for path in args.runs.iterdir() if path.is_dir()):
        status = json.loads((directory / "status.json").read_text())
        baseline = json.loads((directory / "selection_baseline.json").read_text())
        with (directory / "metrics.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        valid = [row for row in rows if row["split"] == "valid"][-1]
        with (directory / "step_metrics.csv").open(newline="") as stream:
            step_rows = list(csv.DictReader(stream))
        parts = directory.name.rsplit("_seed", 1)
        strategy, seed = parts[0], int(parts[1])
        checkpoints = directory / "checkpoints"
        checkpoint_state = {}
        for name in ("latest.pt", "best_dynamic.pt", "best_balanced.pt"):
            path = checkpoints / name
            checkpoint_state[name] = bool(path.is_file() and path.with_suffix(path.suffix + ".sha256").is_file())
        if status["state"] != "complete" or int(status["global_step"]) != 500:
            failures.append(f"{directory.name}: not complete at 500 steps")
        if len(step_rows) != 5:
            failures.append(f"{directory.name}: expected five 100-step resource records")
        if not checkpoint_state["latest.pt"] or not checkpoint_state["best_dynamic.pt"]:
            failures.append(f"{directory.name}: required checkpoint missing")
        if not checkpoint_state["best_balanced.pt"]:
            failures.append(f"{directory.name}: no hard-constraint-feasible balanced checkpoint")
        resources = {}
        if step_rows:
            resources = {
                "cpu_rss_growth_bytes": int(float(step_rows[-1]["cpu_rss_bytes"]) - float(step_rows[0]["cpu_rss_bytes"])),
                "gpu_reserved_growth_bytes": int(float(step_rows[-1]["gpu_reserved_bytes"]) - float(step_rows[0]["gpu_reserved_bytes"])),
                "file_descriptor_growth": int(float(step_rows[-1]["file_descriptors"]) - float(step_rows[0]["file_descriptors"])),
                "batch_latency_p95_ms_max": max(float(row["batch_latency_p95_ms"]) for row in step_rows),
            }
            if resources["cpu_rss_growth_bytes"] > 256 * 2**20:
                failures.append(f"{directory.name}: CPU RSS grew over 256 MiB")
            if resources["gpu_reserved_growth_bytes"] > 256 * 2**20:
                failures.append(f"{directory.name}: GPU reserved memory grew over 256 MiB")
            if resources["file_descriptor_growth"] > 8:
                failures.append(f"{directory.name}: file descriptors grew by more than 8")
        runs.append({
            "name": directory.name, "strategy": strategy, "seed": seed,
            "global_step": int(status["global_step"]),
            "validation": {key: float(valid[key]) for key in METRICS},
            "baseline": {key: float(baseline[key]) for key in METRICS},
            "checkpoint_state": checkpoint_state,
            "best_dynamic": str(checkpoints / "best_dynamic.pt"),
            "best_balanced": str(checkpoints / "best_balanced.pt"),
            "resource_stability": resources,
            "step_metrics": step_rows,
            "curriculum_stage": status.get("curriculum_stage"),
        })
    grouped = defaultdict(list)
    for run in runs:
        grouped[run["strategy"]].append(run)
    summaries = {}
    for strategy, values in sorted(grouped.items()):
        if len(values) != 3 or sorted(item["seed"] for item in values) != [8010, 8011, 8012]:
            failures.append(f"{strategy}: missing required three seeds")
        aggregate = {metric: {
            "mean": mean([item["validation"][metric] for item in values]),
            "min": min(item["validation"][metric] for item in values),
            "max": max(item["validation"][metric] for item in values),
        } for metric in METRICS}
        baseline = {metric: mean([item["baseline"][metric] for item in values]) for metric in METRICS}
        feasible = (
            aggregate["top1_collision_fraction"]["mean"] <= baseline["top1_collision_fraction"]
            and aggregate["dynamic_cvar"]["mean"] <= baseline["dynamic_cvar"]
            and aggregate["selected_min_clearance"]["mean"] >= baseline["selected_min_clearance"]
            and aggregate["static_cost"]["mean"] <= baseline["static_cost"] * 1.15
        )
        summaries[strategy] = {"seeds": [item["seed"] for item in values],
                               "validation": aggregate, "baseline": baseline,
                               "pareto_constraints_met": feasible}
    feasible = [(name, values) for name, values in summaries.items()
                if values["pareto_constraints_met"]]
    selected = None
    if feasible:
        selected = min(feasible, key=lambda item: (
            item[1]["validation"]["top1_collision_fraction"]["mean"],
            item[1]["validation"]["dynamic_cvar"]["mean"],
            -item[1]["validation"]["selected_min_clearance"]["mean"],
            item[1]["validation"]["raw_dynamic_mean"]["mean"],
        ))[0]
    else:
        failures.append("no strategy met validation Pareto/static constraints")
    representative = None
    if selected:
        target = summaries[selected]["validation"]
        candidates = grouped[selected]
        representative = min(candidates, key=lambda item: sum(abs(
            item["validation"][metric] - target[metric]["mean"]
        ) for metric in ("top1_collision_fraction", "dynamic_cvar", "selected_min_clearance")))
    resume_run = next((item for item in runs if item["name"] == "c_risk_gated_025_seed8010"), None)
    resume_pass = bool(resume_run and (args.runs / resume_run["name"] / "resume_config.yaml").is_file())
    if not resume_pass:
        failures.append("designated exact-resume audit did not run")
    payload = {
        "status": "PASS" if not failures else "FAIL", "run_count": len(runs),
        "steps_per_run": 500, "seed_count_per_strategy": 3,
        "strategies": summaries, "selected_strategy": selected,
        "representative": ({key: representative[key] for key in
                            ("name", "seed", "best_dynamic", "best_balanced")}
                           if representative else None),
        "resume_audit_pass": resume_pass,
        "scalar_total_loss_used_for_selection": False,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
