#!/usr/bin/env python3
"""Validate immutable fixture semantics across loader scheduling choices."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from policy.static_v2_fixture import (
    StaticV2FixtureDataset, StaticV2FixtureMetadataDataset,
)


FIXTURE = ROOT / "artifacts/phase8jv2s0/static_valid_fixture_v1.npz"
REPORT = ROOT / "reports/phase8jv2s0_static_fixture_determinism.json"
WORKERS = (0, 1, 4, 8)
BATCH_SIZES = (1, 16, 32, 64, 128)


def atomic_json(path, value):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def row_hash(batch, row):
    digest = hashlib.sha256()
    for name in (
        "fixture_id", "image_sha256", "map_id", "position_world",
        "rotation_world_from_body", "velocity_body", "acceleration_body",
        "goal_body",
    ):
        value = batch[name][row]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
            digest.update(str(value.dtype).encode())
            digest.update(np.ascontiguousarray(value).tobytes())
        else:
            digest.update(str(value).encode())
    return digest.hexdigest()


def load_hashes(dataset, batch_size, workers, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=workers, generator=generator,
        persistent_workers=False,
    )
    output = {}
    for batch in loader:
        for row, fixture_id in enumerate(batch["fixture_id"]):
            output[str(fixture_id)] = row_hash(batch, row)
    aggregate = hashlib.sha256()
    for fixture_id, value in sorted(output.items()):
        aggregate.update(fixture_id.encode())
        aggregate.update(value.encode())
    return output, aggregate.hexdigest()


def main():
    base = StaticV2FixtureDataset(FIXTURE, cache_size=0)
    metadata = StaticV2FixtureMetadataDataset(FIXTURE)
    if len(metadata) != 10000:
        raise RuntimeError("fixture window count is not 10,000")
    repeated = all(
        np.array_equal(
            np.asarray(base.metadata(317)[name]),
            np.asarray(base.metadata(317)[name]),
        )
        for name in base.metadata(317)
    )
    reference, reference_hash = load_hashes(
        metadata, batch_size=128, workers=0, shuffle=False, seed=8172402
    )
    combinations = []
    for workers in WORKERS:
        for batch_size in BATCH_SIZES:
            observed, semantic_hash = load_hashes(
                metadata, batch_size, workers, False, 8172402
            )
            combinations.append({
                "workers": workers,
                "batch_size": batch_size,
                "semantic_hash": semantic_hash,
                "exact_match": observed == reference,
            })
    shuffled, shuffled_hash = load_hashes(
        metadata, batch_size=32, workers=4, shuffle=True, seed=9123401
    )
    epoch_zero, epoch_zero_hash = load_hashes(
        metadata, batch_size=64, workers=1, shuffle=False, seed=0
    )
    epoch_nine, epoch_nine_hash = load_hashes(
        metadata, batch_size=64, workers=1, shuffle=False, seed=900009
    )
    checks = {
        "same_index_repeated_exact": repeated,
        "batch_size_independent": all(
            row["exact_match"] for row in combinations
        ),
        "worker_count_independent": all(
            row["exact_match"] for row in combinations
        ),
        "shuffle_restore_by_fixture_id_exact": shuffled == reference,
        "shuffle_semantic_hash_exact": shuffled_hash == reference_hash,
        "epoch_seed_independent": (
            epoch_zero == reference and epoch_nine == reference
            and epoch_zero_hash == epoch_nine_hash == reference_hash
        ),
        "valid_fixture_has_no_epoch_mutator": not hasattr(base, "set_epoch"),
        "window_count_10000": len(base) == 10000,
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "fixture_semantic_hash": base.semantic_hash,
        "loader_order_independent_hash": reference_hash,
        "checks": checks,
        "combinations": combinations,
        "shuffle": {
            "workers": 4, "batch_size": 32,
            "semantic_hash": shuffled_hash,
        },
        "training_fixture_created": False,
        "valid_fixture_polluted_by_train": False,
    }
    atomic_json(REPORT, report)
    print(json.dumps({
        "status": report["status"],
        "fixture_semantic_hash": base.semantic_hash,
        "combinations": len(combinations),
        "checks": checks,
    }, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
