#!/usr/bin/env python3
"""Synthetic CUDA positive/negative controls for occlusion constructor v2.2."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import uuid

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from authoritative_dataset.cuda_renderer_v1 import CudaAuthorityRenderer  # noqa
from authoritative_dataset.dynamic_motion_v2 import (  # noqa
    MotionSamplingError,
    load_motion_contract,
)
from authoritative_dataset.exact_backend_v1 import ExactAuthorityBVH  # noqa
from authoritative_dataset.occlusion_constructor_v2_2 import (  # noqa
    CONSTRUCTOR_VERSION,
    build_occluded_actor_specs_v2_2,
    sample_occlusion_uav_sequence_v2_2,
)
from geometry_authority.static_v1 import build_authority_artifact  # noqa


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def build_fixture(root, name, points):
    identifier = str(uuid.uuid5(
        uuid.NAMESPACE_URL, f"occlusion-constructor-v2.2:{name}"
    ))
    target = Path(root) / identifier
    build_authority_artifact(
        target, np.asarray(points, dtype=np.float32).reshape(-1, 3),
        map_uuid=identifier, generator_seed=2200,
        map_id=name,
        generator_source_hash=hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        generator_config_hash="2" * 64,
        parent_git_commit="synthetic-control",
        map_namespace="occlusion_constructor_v2_2_synthetic_controls",
        bounds_min=[-20, -10, 0], bounds_max=[20, 15, 6],
    )
    return ExactAuthorityBVH(target)


def main():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA is mandatory for v2.2 raster controls")
    sensor = {
        "height": 96, "width": 160,
        "intrinsics": [80.0, 80.0, 80.0, 45.0],
        "max_depth_m": 20.0, "ray_step_m": .1,
        "frame_period_ns": 100000000,
    }
    renderer = CudaAuthorityRenderer(sensor, "cuda:0")
    contract = load_motion_contract(
        ROOT / "configs/authoritative_dynamic_motion_contract_v2_1.yaml"
    )
    frame_times = np.arange(60, dtype=np.float64) * .1
    # A 0.4 m wide by 1.2 m high finite panel. It is deliberately narrow
    # enough to permit 1/2/3-frame passages, unlike an infinite wall.
    positive_points = [
        [2.0, y / 10.0, z / 10.0]
        for y in range(28, 32) for z in range(19, 31)
    ]
    negative_points = [[2.0, 3.0, 2.5]]
    positives = []
    negatives = []
    with tempfile.TemporaryDirectory(
        prefix="occlusion-v22-controls-", dir="/tmp"
    ) as temporary:
        positive = build_fixture(
            temporary, "finite_connected_panel", positive_points
        )
        requested_gaps = tuple(int(value) for value in os.environ.get(
            "V22_CONTROL_GAPS", "1,2,3"
        ).split(","))
        for gap in requested_gaps:
            rng = np.random.default_rng(220000 + gap)
            try:
                uav = sample_occlusion_uav_sequence_v2_2(
                    positive, rng, frame_times, gap
                )
                construction = build_occluded_actor_specs_v2_2(
                    positive, renderer, rng, uav.position_world, uav.yaw,
                    frame_times, contract, gap,
                    maximum_candidates=512,
                )
                passed = (
                    construction.gap_end
                    - construction.gap_start + 1 == gap
                )
                positives.append({
                    "name": f"cuda_connected_panel_gap_{gap}",
                    "status": "PASS" if passed else "FAIL",
                    "requested_gap_frames": gap,
                    "observed_gap": [
                        construction.gap_start, construction.gap_end
                    ],
                    "attempts": construction.attempts,
                    "authority_hash":
                        construction.evidence["authority_hash"],
                })
            except Exception as error:
                positives.append({
                    "name": f"cuda_connected_panel_gap_{gap}",
                    "status": "FAIL", "requested_gap_frames": gap,
                    "error": repr(error),
                })
        negative = build_fixture(
            temporary, "single_voxel_insufficient_patch", negative_points
        )
        try:
            sample_occlusion_uav_sequence_v2_2(
                negative, np.random.default_rng(229999),
                frame_times, 3,
            )
            negatives.append({
                "name": "single_voxel_insufficient_patch",
                "status": "FAIL",
                "reason": "constructor accepted an infeasible patch",
            })
        except MotionSamplingError as error:
            negatives.append({
                "name": "single_voxel_insufficient_patch",
                "status": "PASS", "bounded_rejection": str(error),
            })
    rows = positives + negatives
    report = {
        "status": "PASS" if all(
            row["status"] == "PASS" for row in rows
        ) else "FAIL",
        "constructor_version": CONSTRUCTOR_VERSION,
        "device": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "synthetic_positive_controls": positives,
        "synthetic_negative_controls": negatives,
        "validator_inputs": [
            "canonical occupancy", "camera state", "actor state",
            "CUDA per-actor visibility",
        ],
        "annex_provenance_read": False,
        "formal_generation_started": False,
        "training_executed": False,
        "test_accessed": False,
        "blind_accessed": False,
    }
    atomic_json(
        ROOT / "reports/occlusion_constructor_v2_2_cuda_controls.json",
        report,
    )
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
