#!/usr/bin/env python3
"""Validate the actual lazy Formal V2 train/valid loader."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from authoritative_dataset.loader_v1 import (
    AuthoritativeFormalDataset, authoritative_formal_collate,
)


EXPECTED_ROOT = (
    "56d7118862afc22d8df8672d747c930bc4fdcfcb1fe0795c19e5ceef47989461"
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def tensor_hash(batch):
    digest = hashlib.sha256()
    for name in (
        "current_depth", "observation_9d", "position_world",
        "velocity_world", "acceleration_world", "goal_world", "goal_body",
        "map_index", "stress", "recoverable", "unknown", "actor_count",
    ):
        digest.update(batch[name].contiguous().cpu().numpy().tobytes())
    digest.update("\n".join(batch["sequence_id"]).encode())
    digest.update(json.dumps(batch["frame_index"]).encode())
    return digest.hexdigest()


def check_batch(batch, batch_size):
    tensor_shapes = {
        name: list(value.shape)
        for name, value in batch.items() if torch.is_tensor(value)
    }
    finite = all(
        bool(torch.isfinite(value).all())
        for value in batch.values()
        if torch.is_tensor(value) and value.is_floating_point()
    )
    return {
        "batch_size": int(batch["current_depth"].shape[0]),
        "expected_batch_size": batch_size,
        "shapes": tensor_shapes,
        "depth_dtype": str(batch["current_depth"].dtype),
        "observation_dtype": str(batch["observation_9d"].dtype),
        "finite": finite,
        "nonempty": bool(batch["current_depth"].numel()),
    }


def first_batch(dataset, workers, batch_size, persistent):
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent if workers else False,
        prefetch_factor=2 if workers else None,
        collate_fn=authoritative_formal_collate,
    )
    iterator = iter(loader)
    first = next(iterator)
    second = next(iterator)
    del iterator, loader
    return first, second


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path,
        default=ROOT / "data/phase8_authoritative_v2",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "reports/phase8jqv2_5_v2_loader_validation.json",
    )
    args = parser.parse_args()
    manifest_path = args.root / "manifests/dataset_manifest.json"
    before = sha256(manifest_path)
    failures = []
    split_results = {}
    for split, expected in (("train", 1_000_000), ("valid", 100_000)):
        dataset = AuthoritativeFormalDataset(
            args.root, split, expected_root_hash=EXPECTED_ROOT,
            cache_size=4, verify_files=True, max_depth_m=20.0,
        )
        if len(dataset) != expected:
            failures.append(f"{split} length mismatch")
        indices = [0, len(dataset)//2, len(dataset)-1]
        probe = []
        for index in indices:
            first_hash = dataset.semantic_hash(index)
            second_hash = dataset.semantic_hash(index)
            sample = dataset[index]
            probe.append({
                "index": index,
                "semantic_hash": first_hash,
                "repeat_hash": second_hash,
                "match": first_hash == second_hash,
                "depth_shape": list(sample["current_depth"].shape),
                "observation_shape": list(sample["observation_9d"].shape),
                "depth_dtype": str(sample["current_depth"].dtype),
                "observation_dtype": str(
                    sample["observation_9d"].dtype
                ),
                "sequence_id": sample["sequence_id"],
                "frame_index": sample["frame_index"],
            })
        if not all(row["match"] for row in probe):
            failures.append(f"{split} single-sample nondeterminism")
        dataset._cache.clear()
        serial_first, serial_second = first_batch(
            dataset, 0, args.batch_size, False
        )
        dataset._cache.clear()
        worker_first, worker_second = first_batch(
            dataset, args.workers, args.batch_size, True
        )
        dataset._cache.clear()
        restarted_first, _ = first_batch(
            dataset, args.workers, args.batch_size, True
        )
        hashes = {
            "num_workers_0_first": tensor_hash(serial_first),
            "num_workers_0_second": tensor_hash(serial_second),
            "multiworker_first": tensor_hash(worker_first),
            "multiworker_second": tensor_hash(worker_second),
            "restarted_multiworker_first": tensor_hash(restarted_first),
        }
        deterministic = (
            hashes["num_workers_0_first"] == hashes["multiworker_first"]
            == hashes["restarted_multiworker_first"]
        )
        if not deterministic:
            failures.append(f"{split} worker/restart nondeterminism")
        batch_check = check_batch(worker_first, args.batch_size)
        if not batch_check["finite"] or not batch_check["nonempty"]:
            failures.append(f"{split} invalid batch")
        device_transfer = {"status": "NOT_RUN", "device": None}
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{int(args.device)}")
            transferred = {
                name: value.to(device, non_blocking=True)
                for name, value in worker_first.items()
                if torch.is_tensor(value)
            }
            torch.cuda.synchronize(device)
            if not all(value.device == device for value in transferred.values()):
                failures.append(f"{split} device transfer failed")
            device_transfer = {
                "status": "PASS",
                "device": str(device),
                "tensor_count": len(transferred),
            }
        split_results[split] = {
            "length": len(dataset),
            "single_sample_probes": probe,
            "batch": batch_check,
            "batch_hashes": hashes,
            "worker_restart_deterministic": deterministic,
            "num_workers_tested": [0, args.workers],
            "persistent_workers_tested": True,
            "prefetch_factor": 2,
            "device_transfer": device_transfer,
        }
    after = sha256(manifest_path)
    if before != after:
        failures.append("loader modified root manifest")
    result = {
        "status": "PASS" if not failures else "FAIL",
        "loader_version": "authoritative_formal_loader_v1",
        "dataset_version": "phase8_authoritative_v2",
        "root_manifest_hash": after,
        "splits": split_results,
        "failures": failures,
        "manifest_unchanged_during_validation": before == after,
        "production_test_used": False,
        "blind_used": False,
        "test_files_loaded": False,
        "optimizer_step_executed": False,
        "network_weights_modified": False,
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
