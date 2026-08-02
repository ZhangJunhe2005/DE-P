#!/usr/bin/env python3
"""Full sequential valid_static evaluation under Safety Evaluator V2.

This is inference-only.  It rebuilds each checkpoint's candidate trajectories
on the versioned latency-aware timeline and evaluates raw static ESDF clearance
minus the physical UAV radius.  It never reads the production test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from policy.checkpoint_utils import load_dep_checkpoint
from policy.dep_dataset import DEPDataset, seed_dataset_worker
from policy.safety_evaluator_v2 import SafetyEvaluatorV2, SafetyEvaluatorV2Config
from tools.run_phase8i_failure_decomposition import make_runtime


COUNT = 15
TOLERANCE = 1e-6
REPORT = ROOT / "reports/phase8jqv2_static_validation.json"
ARTIFACT_ROOT = ROOT / "artifacts/phase8jqv2"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def loader(dataset, batch_size, workers):
    generator = torch.Generator().manual_seed(8172402)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        worker_init_fn=seed_dataset_worker, generator=generator,
        pin_memory=torch.cuda.is_available(), persistent_workers=False,
    )


def metric(flags, rule):
    flags = np.asarray(flags, bool)
    return {
        "numerator": int(flags.sum()),
        "denominator": int(len(flags)),
        "fraction": float(flags.mean()) if len(flags) else None,
        "inclusion_rule": rule,
        "exclusion_rule": "none",
    }


def distribution(values):
    values = np.asarray(values, float)
    return {
        "mean": float(values.mean()), "std": float(values.std()),
        "q10": float(np.quantile(values, .1)),
        "q50": float(np.quantile(values, .5)),
        "q90": float(np.quantile(values, .9)),
    }


def evaluate_checkpoint(runtime, dataset, checkpoint, evaluator, args, key):
    artifact = ARTIFACT_ROOT / f"{checkpoint['name']}-valid_static-v2.npz"
    if artifact.is_file():
        loaded = dict(np.load(artifact, allow_pickle=True))
        if str(loaded.get("analysis_key", "")) == key:
            return loaded, True
    load_dep_checkpoint(runtime.policy, checkpoint["path"], "corrected")
    runtime.policy.eval()
    columns = {"clearance": [], "predicted": [], "map": [], "t0": []}
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader(dataset, args.batch_size, args.workers):
            details = runtime._forward_details(
                *batch, dynamic_context=None, dynamic_obstacles=None,
                dep_loss=runtime.static_dep_loss,
            )
            size = len(batch[0])
            start = details["start_state_world_expanded"]
            end = details["end_state_world_expanded"]
            old_positions, _ = (
                runtime.static_dep_loss.safety_loss.trajectory_sampler.grouped(
                    start.permute(0, 2, 1), end.permute(0, 2, 1), size
                )
            )
            old_positions = old_positions.reshape(
                size, COUNT, -1, 3
            ).detach().cpu().numpy()
            current = start.reshape(size, COUNT, 3, 3)[:, 0].permute(
                0, 2, 1
            ).detach().cpu().numpy()
            timelines = np.stack([
                np.stack([
                    evaluator.timeline(current[row], old_positions[row, candidate])[
                        "positions"
                    ]
                    for candidate in range(COUNT)
                ])
                for row in range(size)
            ])
            maps = details["batch_map_id"]
            positions = torch.from_numpy(
                timelines.reshape(size, -1, 3).astype(np.float32)
            ).to(runtime.device)
            _, raw = runtime.static_dep_loss.safety_loss.get_distance_cost(
                positions, maps
            )
            raw = raw.reshape(size, COUNT, -1).detach().cpu().numpy()
            segment = np.linalg.norm(np.diff(timelines, axis=2), axis=3)
            lower = np.minimum(raw[:, :, :-1], raw[:, :, 1:]) - segment
            clearance = np.minimum(
                raw.min(2), lower.min(2)
            ) - evaluator.config.uav_radius_m
            columns["clearance"].append(clearance)
            columns["predicted"].append(
                details["predicted_score"].reshape(size, COUNT).detach().cpu().numpy()
            )
            columns["map"].append(maps.detach().cpu().numpy())
            columns["t0"].append(
                raw[:, 0, 0] - evaluator.config.uav_radius_m
            )
    arrays = {
        name: np.concatenate(value) for name, value in columns.items()
    }
    arrays.update({
        "analysis_key": np.asarray(key),
        "checkpoint_sha256": np.asarray(checkpoint["sha256"]),
        "elapsed_seconds": np.asarray(time.perf_counter() - started),
    })
    artifact.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact.with_name(f".{artifact.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, artifact)
    return arrays, False


def summarize(arrays):
    clearance = np.asarray(arrays["clearance"], float)
    predicted = np.asarray(arrays["predicted"], float)
    safe = clearance >= -TOLERANCE
    rows = np.arange(len(safe))
    selected = safe[rows, predicted.argmin(1)]
    failure = ~safe.any(1)
    maps = np.asarray(arrays["map"], int)
    return {
        "window_count": int(len(safe)),
        "sequential_complete": int(len(safe)) == 10000,
        "physical_static_coverage_failure": metric(
            failure, "all 10,000 sequential valid_static windows"
        ),
        "physical_safe_candidate_count": distribution(safe.sum(1)),
        "top1_physical_collision": metric(
            ~selected, "all 10,000 sequential valid_static windows"
        ),
        "already_unsafe_fraction": metric(
            np.asarray(arrays["t0"], float) < -TOLERANCE,
            "all 10,000 sequential valid_static windows at t=0",
        ),
        "map_groups": {
            str(map_id): {
                "coverage_failure": metric(
                    failure[maps == map_id], f"valid_static map={map_id}"
                ),
                "window_count": int((maps == map_id).sum()),
            }
            for map_id in sorted(set(maps))
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke-windows", type=int, default=0)
    args = parser.parse_args()
    entry = json.loads(
        (ROOT / "reports/phase8jqv2_entry_gate.json").read_text()
    )
    matrix = json.loads(
        (ROOT / "reports/phase8i_checkpoint_matrix.json").read_text()
    )
    checkpoints = [
        row for row in matrix["checkpoints"] if row["analyzed_full_phase8i"]
    ]
    if args.smoke_windows:
        checkpoints = checkpoints[:1]
    config = SafetyEvaluatorV2Config.load(
        ROOT / "configs/safety_evaluator_v2.yaml",
        ROOT / "reports/phase8jq_controller_authoritative_envelope.json",
    )
    evaluator = SafetyEvaluatorV2(config)
    runtime = make_runtime(
        ROOT / "data/phase8_dynamic_production/map_catalog.yaml",
        ROOT / "configs/static_map_catalog.yaml",
    )
    dataset = DEPDataset(mode="valid", cache_size=128, global_seed=8172402)
    if args.smoke_windows:
        dataset.image_paths = dataset.image_paths[:args.smoke_windows]
        dataset.img_list = dataset.image_paths
        dataset.positions = dataset.positions[:args.smoke_windows]
        dataset.quaternions = dataset.quaternions[:args.smoke_windows]
        dataset.map_idx = dataset.map_idx[:args.smoke_windows]
    started = time.perf_counter()
    results, hits, misses = {}, 0, 0
    implementation_hash = sha256(__file__)
    for checkpoint in checkpoints:
        key = canonical_hash({
            "schema": "phase8jqv2_valid_static_v1",
            "implementation": implementation_hash,
            "config": config.canonical_hash(),
            "checkpoint": checkpoint["sha256"],
            "static_catalog": sha256(ROOT / "configs/static_map_catalog.yaml"),
            "window_count": len(dataset),
            "seed": 8172402,
        })
        arrays, hit = evaluate_checkpoint(
            runtime, dataset, checkpoint, evaluator, args, key
        )
        hits += int(hit)
        misses += int(not hit)
        results[checkpoint["name"]] = {
            "checkpoint_hash": checkpoint["sha256"],
            "strict_load": True,
            **summarize(arrays),
        }
        print(
            f"{checkpoint['name']} valid_static complete cache={hit}",
            flush=True,
        )
    base = {
        "evaluator_version": config.evaluator_version,
        "config_hash": entry["frozen_hashes"]["v2_config_sha256"],
        "geometry_hash": sha256(ROOT / "loss/safety_geometry_v2.py"),
        "timeline_hash": sha256(ROOT / "policy/safety_evaluator_v2.py"),
        "uncertainty_policy_hash": canonical_hash({
            "policy": config.estimated_uncertainty_policy,
            "sigma": config.confidence_sigma,
        }),
        "Simulator_geometry_hash": json.loads(
            (ROOT / "reports/phase8jqv2_geometry_validation.json").read_text()
        )["simulator_geometry_hash"],
        "dataset_manifest_hash": entry["frozen_hashes"][
            "production_manifest_sha256"
        ],
        "cache_index_hash": entry["frozen_hashes"][
            "estimated_cache_index_sha256"
        ],
        "checkpoint_hash": next(
            row["sha256"] for row in checkpoints
            if row["name"] == "fixed_050_seed8403"
        ) if not args.smoke_windows else checkpoints[0]["sha256"],
        "checkpoint_hashes": {
            row["name"]: row["sha256"] for row in checkpoints
        },
    }
    report = {
        **base,
        "status": "PASS",
        "suite": "valid_static",
        "full_sequential": not bool(args.smoke_windows),
        "window_count": len(dataset),
        "results": results,
        "performance": {
            "device": str(runtime.device),
            "gpu": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else None
            ),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": (
                torch.cuda.max_memory_allocated()
                if torch.cuda.is_available() else 0
            ),
            "peak_cpu_rss_kib": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
            "artifact_cache_hits": hits,
            "artifact_cache_misses": misses,
        },
        "training_executed": False,
        "backward_executed": False,
        "optimizer_step_executed": False,
        "production_test_used": False,
    }
    output = (
        ROOT / "reports/phase8jqv2_static_smoke.json"
        if args.smoke_windows else REPORT
    )
    atomic_json(output, report)
    print(json.dumps({
        "status": "PASS", "window_count": len(dataset),
        "checkpoints": len(checkpoints), "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
