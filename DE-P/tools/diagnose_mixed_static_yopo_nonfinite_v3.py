#!/usr/bin/env python3
"""Exact epoch-0 replay of the legacy AMP boundary, stopping at first NaN/Inf."""

from __future__ import annotations

import json
import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.static_yopo_dataset_v1 import StaticYOPODatasetV1
from data.static_yopo_loader_v1 import make_static_yopo_loader_v1
from policy.static_yopo_training_v1 import (
    MixedSceneStaticYOPOObjectiveV1,
    MixedSceneStaticYOPOV1,
)

CONFIG = ROOT / "configs/phase8jqv2_5_mixed_static_yopo_training_v3.yaml"
REPORT = ROOT / "reports/phase8jqv2_5_training_nonfinite_replay.json"


def finite(value):
    return bool(torch.isfinite(value).all())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-batches", type=int)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA is required")
    config = YAML(typ="safe").load(CONFIG)
    seed = int(config["training"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    derived = Path(config["derived_dataset_root"])
    dataset = StaticYOPODatasetV1(
        derived, "train", mmap_cache_size=config["loader"]["mmap_cache_size"]
    )
    loader, sampler = make_static_yopo_loader_v1(
        dataset, config["training"]["batch_size"], seed,
        config["loader"]["num_workers"], True, config["loader"]["prefetch_factor"],
        config["loader"]["pin_memory"], True,
    )
    sampler.set_epoch(0)
    device = torch.device("cuda:0")
    model = MixedSceneStaticYOPOV1(config["model"]["initial_checkpoint"]).to(device)
    objective = MixedSceneStaticYOPOObjectiveV1(derived / "map_catalog.yaml").to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["optimizer"]["learning_rate"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    model.train()
    started = time.perf_counter()
    component_names = (
        "total_loss", "trajectory_loss", "score_loss", "smoothness_loss",
        "static_safety_loss", "guidance_loss", "dynamic_safety_loss",
        "candidate_smooth_cost", "candidate_static_cost",
        "candidate_guidance_cost", "score_label",
    )
    result = None
    for batch_no, batch in enumerate(loader):
        batch = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=True):
            details = objective(model, batch)
        status = {name: finite(details[name]) for name in component_names}
        if not all(status.values()):
            result = {
                "status": "REPRODUCED",
                "epoch": 0,
                "batch": batch_no,
                "global_step_if_executed": batch_no + 1,
                "elapsed_seconds": time.perf_counter() - started,
                "finite_components": status,
                "component_dtype": {
                    name: str(details[name].dtype) for name in component_names
                },
                "sample_id": list(batch["sample_id"]),
                "map_id": batch["map_id"].detach().cpu().tolist(),
                "depth_min": float(batch["depth"].amin()),
                "depth_max": float(batch["depth"].amax()),
                "observation_min": float(batch["observation"].amin()),
                "observation_max": float(batch["observation"].amax()),
            }
            break
        scaler.scale(details["total_loss"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config["training"]["gradient_clip_norm"])
        )
        scaler.step(optimizer)
        scaler.update()
        if (batch_no + 1) % 500 == 0:
            print(json.dumps({
                "status": "REPLAYING", "epoch": 0, "batch": batch_no + 1,
                "batches_total": len(loader),
                "elapsed_seconds": time.perf_counter() - started,
            }), flush=True)
        if args.max_batches and batch_no + 1 >= args.max_batches:
            break
    if result is None:
        result = {
            "status": "NOT_REPRODUCED",
            "epoch": 0,
            "batches_scanned": min(len(loader), args.max_batches or len(loader)),
            "elapsed_seconds": time.perf_counter() - started,
        }
    temporary = REPORT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, REPORT)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
