#!/usr/bin/env python3
"""Generate the locked three-strategy/three-seed Phase 8D matrix."""

from copy import deepcopy
from pathlib import Path

from ruamel.yaml import YAML


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (8401, 8402, 8403)
STRATEGIES = {
    "fixed_025": {"early": 0.25, "switch": -1, "late": 0.25},
    "scheduled_025_050": {"early": 0.25, "switch": 1026, "late": 0.5},
    "fixed_050": {"early": 0.5, "switch": -1, "late": 0.5},
}


def main():
    yaml = YAML()
    source = YAML(typ="safe").load(ROOT / "configs/train_dynamic_production_v2.yaml")
    output = ROOT / "configs/phase8d_shakedown"
    output.mkdir(parents=True, exist_ok=True)
    for strategy, values in STRATEGIES.items():
        for seed in SEEDS:
            config = deepcopy(source)
            config["config_version"] = f"phase8d_{strategy}_seed{seed}"
            config["run_kind"] = "bounded_shakedown"
            config["epochs"] = 2
            config["random_seed"] = seed
            config["dynamic_training"]["noise_seed"] = seed
            # The formal Phase-8D perception audit currently hard-fails. Keep
            # training context fixed to GT so A/B/C isolates only PCGrad ratio;
            # fixed valid_estimated remains active as the production-facing gate.
            config["dynamic_training"]["curriculum"] = [
                {"start_epoch": 0, "context_source": "ground_truth", "ratio": 1.0},
            ]
            objective = config["dynamic_objective"]
            objective["pcgrad_other_norm_ratio"] = values["early"]
            objective["pcgrad_ratio_switch_step"] = values["switch"]
            objective["pcgrad_late_other_norm_ratio"] = values["late"]
            config["validation"]["max_batches_per_suite"] = 32
            config["early_stopping"]["patience"] = 20
            config["abort_thresholds"]["min_free_disk_gib"] = 10.0
            path = output / f"{strategy}_seed{seed}.yaml"
            with path.open("w", encoding="utf-8") as stream:
                yaml.dump(config, stream)
    print(f"generated {len(SEEDS) * len(STRATEGIES)} configs in {output}")


if __name__ == "__main__":
    main()
