#!/usr/bin/env python3
"""Five-seed immutable Phase-8B Gate; never launches production work."""

from __future__ import annotations

import json
import gc
from pathlib import Path
import resource
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

from loss.dynamic_types import DynamicObjectiveConfig
from phase8b_common import load_config, load_fixed_samples
from run_dynamic_objective_ablation import MATRIX, run


def seed_gates(result):
    before, after = result["before"], result["after"]
    no_before, no_after = result["no_target_before"], result["no_target_after"]
    return {
        "mean_dynamic_not_increased": after["raw_dynamic_mean"] <= before["raw_dynamic_mean"] + 1e-7,
        "dynamic_cvar_decreased": after["dynamic_cvar"] < before["dynamic_cvar"] * 0.99,
        "safe_candidate_fraction_not_decreased": after["safe_candidate_fraction"] >= before["safe_candidate_fraction"],
        "hard_top1_risk_decreased": after["top1_raw_risk"] < before["top1_raw_risk"],
        "hard_top1_collision_fraction_decreased": after["top1_collision_fraction"] < before["top1_collision_fraction"],
        "minimum_clearance_not_decreased": after["minimum_clearance"] >= before["minimum_clearance"],
        "score_risk_correlation_not_decreased": after["spearman"] >= before["spearman"] - 1e-7,
        "oracle_regret_not_worse": after["oracle_regret"] <= before["oracle_regret"] + 1e-7,
        "no_target_dynamic_exact_zero": no_before["raw_dynamic_mean"] == 0.0 == no_after["raw_dynamic_mean"],
        "no_target_static_regression_bounded": no_after["static"] <= no_before["static"] * 1.2,
        "static_regression_bounded": after["static"] <= before["static"] * 1.2,
        "finite": result["finite"],
        "checkpoint_resume_exact": result["checkpoint_resume_exact"],
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Phase-8B preflight requires host CUDA full-access")
    config = load_config(ROOT / "configs/phase8b_preflight.yaml")
    objective = DynamicObjectiveConfig.from_mapping(config["dynamic_objective"])
    hard = load_fixed_samples(config, ROOT / "diagnostics/fixed_hard_risk.json", training_only=True)
    no_target = load_fixed_samples(config, ROOT / "diagnostics/fixed_no_target.json", training_only=True)
    configured_root = Path(config["dataset_root"]).resolve()
    no_target = [sample for sample in no_target
                 if Path(sample["fixed_entry"]["dataset_root"]).resolve() == configured_root]
    fd_before = len(list(Path("/proc/self/fd").iterdir()))
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    torch.cuda.reset_peak_memory_stats()
    results = []
    resource_samples = []
    for seed in config["seeds"]:
        result = run(config, f"phase8b_seed_{seed}", MATRIX["F_full"], hard, no_target,
                     int(config["steps"]), int(seed), objective=objective)
        result["gates"] = seed_gates(result)
        result["passed"] = all(result["gates"].values())
        results.append(result)
        gc.collect(); torch.cuda.empty_cache()
        status = Path("/proc/self/status").read_text()
        rss_kib = int(next(line.split()[1] for line in status.splitlines()
                           if line.startswith("VmRSS:")))
        resource_samples.append({
            "seed": int(seed), "fd": len(list(Path("/proc/self/fd").iterdir())),
            "current_rss_bytes": rss_kib * 1024,
            "gpu_allocated_bytes": torch.cuda.memory_allocated(),
        })
    pass_count = sum(item["passed"] for item in results)
    numeric_keys = ("raw_dynamic_mean", "dynamic_cvar", "safe_candidate_fraction",
                    "top1_raw_risk", "minimum_clearance", "spearman", "oracle_regret", "static")
    aggregate = {}
    for key in numeric_keys:
        values = np.asarray([item["after"][key] for item in results], dtype=float)
        aggregate[key] = {"mean": float(values.mean()), "std": float(values.std()),
                          "median": float(np.median(values)), "values": values.tolist()}
    before = results[0]["before"]
    aggregate_gates = {
        "mean_dynamic_not_increased": aggregate["raw_dynamic_mean"]["mean"] <= before["raw_dynamic_mean"] + 1e-7,
        "dynamic_cvar_decreased": aggregate["dynamic_cvar"]["mean"] < before["dynamic_cvar"] * .99,
        "safe_candidate_fraction_not_decreased": aggregate["safe_candidate_fraction"]["mean"] >= before["safe_candidate_fraction"],
        "hard_top1_risk_decreased": aggregate["top1_raw_risk"]["mean"] < before["top1_raw_risk"],
        "minimum_clearance_not_decreased": aggregate["minimum_clearance"]["mean"] >= before["minimum_clearance"],
        "score_risk_correlation_not_decreased": aggregate["spearman"]["mean"] >= before["spearman"] - 1e-7,
        "oracle_regret_not_worse": aggregate["oracle_regret"]["mean"] <= before["oracle_regret"] + 1e-7,
    }
    dynamic_only = json.loads((ROOT / "reports/phase8b_gradient_diagnostics/objective_ablation.json").read_text())["results"][0]
    dynamic_only_gate = (
        dynamic_only["name"] == "A_dynamic_only"
        and dynamic_only["after"]["raw_dynamic_mean"] < dynamic_only["before"]["raw_dynamic_mean"] * .9
        and dynamic_only["finite"]
    )
    map_gate_path = ROOT / "reports/phase8b_map_generator_validation.json"
    map_gate = (map_gate_path.is_file()
                and json.loads(map_gate_path.read_text()).get("status") == "PASS")
    stable_samples = resource_samples[1:]
    global_gates = {
        "dynamic_only_sanity": dynamic_only_gate,
        "at_least_four_of_five_seeds": pass_count >= 4,
        "aggregate_metrics_pass": all(aggregate_gates.values()),
        "map_generator_hardening_pass": map_gate,
        "gpu_memory_finite": torch.cuda.max_memory_allocated() > 0,
        "file_descriptors_stable": max(item["fd"] for item in stable_samples)
                                     - min(item["fd"] for item in stable_samples) <= 2,
        "cpu_memory_stable_after_warmup": max(item["current_rss_bytes"] for item in stable_samples)
                                          - min(item["current_rss_bytes"] for item in stable_samples) <= 128 * 2**20,
        "gpu_memory_stable_after_warmup": max(item["gpu_allocated_bytes"] for item in stable_samples)
                                          - min(item["gpu_allocated_bytes"] for item in stable_samples) <= 64 * 2**20,
    }
    status = "PASS" if all(global_gates.values()) else "FAIL"
    payload = {
        "status": status, "production_ready": status == "PASS",
        "device": torch.cuda.get_device_name(0),
        "compute_capability": torch.cuda.get_device_capability(0),
        "immutable_fixed_set": str((ROOT / "diagnostics/fixed_hard_risk.json").resolve()),
        "objective": config["dynamic_objective"], "seeds": results,
        "seed_pass_count": pass_count, "aggregate": aggregate,
        "aggregate_gates": aggregate_gates, "global_gates": global_gates,
        "resources": {"peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
                      "cpu_peak_rss_delta_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 - rss_before,
                      "fd_delta_from_process_start": len(list(Path("/proc/self/fd").iterdir())) - fd_before,
                      "post_seed_samples": resource_samples},
        "production_training_started": False, "production_data_recording_started": False,
    }
    output = ROOT / "reports/phase8b_preflight_result.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("PHASE8B_PREFLIGHT_RESULT")
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if status == "PASS" else 2)


if __name__ == "__main__":
    main()
