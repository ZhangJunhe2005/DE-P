#!/usr/bin/env python3
"""Local AABB surrogate/exact authority alignment Gate."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from authoritative_dataset.exact_backend_v1 import ExactAuthorityBVH
from authoritative_dataset.loader_v1 import AuthoritativeFormalDataset
from loss.authoritative_surrogate_v2_1 import (
    SURROGATE_VERSION,
    local_aabb_signed_gap,
    nearest_voxel_bounds,
)


ROOT_HASH = "56d7118862afc22d8df8672d747c930bc4fdcfcb1fe0795c19e5ceef47989461"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=120)
    args = parser.parse_args()
    dataset = AuthoritativeFormalDataset(
        ROOT / "data/phase8_authoritative_v2",
        "valid",
        expected_root_hash=ROOT_HASH,
        verify_files=False,
    )
    indices = np.linspace(
        0, len(dataset)-1, args.samples, dtype=np.int64
    ).tolist()
    by_map = {}
    exact_values = []
    surrogate_values = []
    map_counts = Counter()
    gradient_failures = 0
    for index in indices:
        sample = dataset[index]
        sequence_index, _ = dataset._location(index)
        _, sequence = dataset._sequences[sequence_index]
        map_uuid = sequence["map_uuid"]
        backend = by_map.setdefault(
            map_uuid,
            ExactAuthorityBVH(
                ROOT / "data/phase8_authoritative_v2/geometry_authority"
                / "valid" / map_uuid
            ),
        )
        point_np = sample["position_world"].numpy().astype(np.float64)[None]
        exact = backend.query_one(point_np[0], 0.3)
        minimum, maximum, _ = nearest_voxel_bounds(backend, point_np)
        point = torch.tensor(point_np, dtype=torch.float64, requires_grad=True)
        surrogate = local_aabb_signed_gap(
            point,
            torch.tensor(minimum, dtype=torch.float64),
            torch.tensor(maximum, dtype=torch.float64),
        )
        surrogate.sum().backward()
        gradient_failures += int(not bool(torch.isfinite(point.grad).all()))
        exact_values.append(float(exact["minimum_gap_m"]))
        surrogate_values.append(float(surrogate.detach()[0]))
        map_counts[map_uuid] += 1
    exact_values = np.asarray(exact_values)
    surrogate_values = np.asarray(surrogate_values)
    errors = np.abs(exact_values-surrogate_values)
    exact_safe = exact_values > 1e-6
    surrogate_safe = surrogate_values > 1e-6
    local_pass = (
            float(errors.max()) <= 1e-9
            and not gradient_failures
            and not int((surrogate_safe & ~exact_safe).sum())
            and len(map_counts) == 12
    )
    result = {
        "status": "PRELIMINARY_PASS" if local_pass else "FAIL",
        "full_alignment_gate": False,
        "remaining_scope": [
            "continuous_candidate_trajectory_outcomes",
            "dynamic_surrogate_alignment",
            "scenario_map_speed_acceleration_breakdowns",
        ],
        "surrogate_version": SURROGATE_VERSION,
        "role": "differentiable_local_surrogate_exact_verifier_required",
        "dataset_version": "phase8_authoritative_v2",
        "root_manifest_hash": ROOT_HASH,
        "sample_count": len(indices),
        "valid_map_count": len(map_counts),
        "map_breakdown": dict(sorted(map_counts.items())),
        "clearance_mae_m": float(errors.mean()),
        "clearance_max_abs_error_m": float(errors.max()),
        "safe_unsafe_agreement": float((exact_safe == surrogate_safe).mean()),
        "false_safe": int((surrogate_safe & ~exact_safe).sum()),
        "false_unsafe": int((~surrogate_safe & exact_safe).sum()),
        "gradient_finite_fraction": 1.0-gradient_failures/len(indices),
        "continuous_outcome_policy":
            "all final outcomes require exact continuous authority verification",
        "production_test_used": False,
        "blind_used": False,
        "optimizer_step_executed": False,
        "source_sha256": hashlib.sha256(
            (ROOT / "loss/authoritative_surrogate_v2_1.py").read_bytes()
        ).hexdigest(),
    }
    output = ROOT / "reports/phase8jqv2_5_surrogate_alignment.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if local_pass else 2)


if __name__ == "__main__":
    main()
