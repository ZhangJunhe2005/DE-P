#!/usr/bin/env python3
"""Create the bounded 4-strategy x 3-seed Phase-8C comparison configs."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from ruamel.yaml import YAML


STRATEGIES = {
    "a_fixed_025": {"risk_gate": False, "ratio": 0.25, "switch": -1, "late": 0.25},
    "b_fixed_050": {"risk_gate": False, "ratio": 0.50, "switch": -1, "late": 0.50},
    "c_risk_gated_025": {"risk_gate": True, "ratio": 0.25, "switch": -1, "late": 0.25},
    "d_scheduled_025_050": {"risk_gate": False, "ratio": 0.25, "switch": 250, "late": 0.50},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("configs/train_dynamic_production.yaml"))
    parser.add_argument("--output", type=Path, default=Path("configs/phase8c_ablation"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace generated configs: {args.output}")
    yaml = YAML()
    base = YAML(typ="safe").load(args.base)
    args.output.mkdir(parents=True)
    for strategy_name, strategy in STRATEGIES.items():
        for seed in (8010, 8011, 8012):
            config = deepcopy(base)
            config["config_version"] = f"phase8c_{strategy_name}_seed{seed}"
            config["run_kind"] = "bounded_shakedown"
            config["random_seed"] = seed
            config["epochs"] = 2
            config["max_steps_per_epoch"] = 250
            config["max_validation_batches"] = 16
            config["num_workers"] = 2
            config["save_interval"] = 1
            config["early_stopping"]["patience"] = 20
            config["dynamic_training"]["noise_seed"] = seed
            config["dynamic_training"]["curriculum"] = [
                {"start_epoch": 0, "context_source": "ground_truth", "ratio": 1.0},
                {"start_epoch": 1, "context_source": "estimated", "ratio": 0.5},
            ]
            objective = config["dynamic_objective"]
            objective["pcgrad_risk_gate"] = strategy["risk_gate"]
            objective["pcgrad_other_norm_ratio"] = strategy["ratio"]
            objective["pcgrad_ratio_switch_step"] = strategy["switch"]
            objective["pcgrad_late_other_norm_ratio"] = strategy["late"]
            config["abort_thresholds"]["min_free_disk_gib"] = 10.0
            path = args.output / f"{strategy_name}_seed{seed}.yaml"
            with path.open("w", encoding="utf-8") as stream:
                yaml.dump(config, stream)
    print(f"generated {len(STRATEGIES) * 3} bounded configs in {args.output}")


if __name__ == "__main__":
    main()
