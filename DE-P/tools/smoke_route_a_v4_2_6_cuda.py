#!/usr/bin/env python3
"""One-batch host CUDA smoke for the V4.2.6 training graph."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch
from ruamel.yaml import YAML
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.static_yopo_dataset_v1 import StaticYOPODatasetV1
from policy.static_yopo_training_v1 import (
    MixedSceneStaticYOPOObjectiveV1,
    MixedSceneStaticYOPOV1,
)
from policy.static_yopo_wide_state_v1 import StaticYOPOWideStateDatasetV1
from policy.static_yopo_recovery_state_v2 import StaticYOPORecoveryStateDatasetV2
from tools.train_mixed_static_yopo_v1 import build_optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs/route_a_v4_2_6_feasibility_score_training.yaml",
    )
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    config = YAML(typ="safe").load(config_path)
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA smoke requires CUDA")
    device = torch.device("cuda:0")
    dataset = StaticYOPODatasetV1(
        Path(config["derived_dataset_root"]), "train", mmap_cache_size=1
    )
    observation_contract = config["observation"]["contract"]
    wrapper = {
        "route_a_wide_state_v1": StaticYOPOWideStateDatasetV1,
        "route_a_recovery_state_v2": StaticYOPORecoveryStateDatasetV2,
    }.get(observation_contract)
    if wrapper is None:
        raise ValueError(f"unsupported smoke observation: {observation_contract}")
    dataset = wrapper(dataset, seed=int(config["training"]["seed"]))
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    model = MixedSceneStaticYOPOV1(
        config["model"]["initial_checkpoint"],
        head_variant=config["model"].get("head_variant", "unified"),
        allow_unified_to_split=bool(
            config["model"].get("allow_unified_to_split", False)
        ),
    ).to(device).train()
    full_catalog_path = (
        Path(config["derived_dataset_root"]) / "map_catalog.yaml"
    )
    yaml = YAML(typ="safe")
    full_catalog = yaml.load(full_catalog_path)
    used_map_ids = sorted(set(int(value) for value in batch["map_id"].tolist()))
    limited_catalog = dict(full_catalog)
    limited_catalog["maps"] = [
        row for row in full_catalog["maps"]
        if int(row["map_id"]) in used_map_ids
    ]
    if len(limited_catalog["maps"]) != len(used_map_ids):
        raise RuntimeError("smoke map catalog does not cover the sampled batch")
    with tempfile.TemporaryDirectory(prefix="route_a_v428_smoke_") as directory:
        limited_catalog_path = Path(directory) / "map_catalog.yaml"
        writer = YAML()
        with limited_catalog_path.open("w", encoding="utf-8") as stream:
            writer.dump(limited_catalog, stream)
        objective = MixedSceneStaticYOPOObjectiveV1(
            limited_catalog_path,
            local_goal_horizon_m=config["objective"]["local_goal_horizon_m"],
            safety_first_config=config.get("safety_first"),
            kinodynamic_config=config.get("kinodynamic_v2"),
            preventive_safety_config=config.get("preventive_safety"),
            progress_safety_config=config.get("progress_safety"),
            feasibility_score_config=config.get("feasibility_score"),
            goal_progress_config=config.get("goal_progress_v2"),
        ).to(device)
        optimizer = build_optimizer(model, config)
        optimizer.zero_grad(set_to_none=True)
        details = objective(model, batch)
        details["total_loss"].backward()
    gradients = [
        value.grad for value in model.parameters() if value.grad is not None
    ]
    if not gradients or not all(torch.isfinite(value).all() for value in gradients):
        raise RuntimeError("V4.2.6 produced missing or nonfinite gradients")
    optimizer.step()
    print(json.dumps({
        "status": "PASS",
        "device": torch.cuda.get_device_name(0),
        "checkpoint_strict_load": bool(
            model.initial_checkpoint_result["strict"]
            if isinstance(model.initial_checkpoint_result, dict)
            else True
        ),
        "head_variant": model.network.head_variant,
        "head_migrated": bool(
            model.initial_checkpoint_result.get("head_migrated", False)
        ),
        "observation_contract": observation_contract,
        "optimizer_groups": [
            value.get("group_name", str(index))
            for index, value in enumerate(optimizer.param_groups)
        ],
        "batch_size": 2,
        "map_ids": used_map_ids,
        "total_loss": float(details["total_loss"].detach()),
        "score_loss": float(details["score_loss"].detach()),
        "unsafe_selection_rate": float(
            details["per_sample_unsafe_selection"].mean()
        ),
        "selected_goal_progress_mean": float(
            details["per_sample_selected_goal_progress"].mean()
        ),
        "reverse_selection_rate": float(
            details["per_sample_reverse_selection"].mean()
        ),
        "gradient_tensor_count": len(gradients),
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
    }, indent=2))


if __name__ == "__main__":
    main()
