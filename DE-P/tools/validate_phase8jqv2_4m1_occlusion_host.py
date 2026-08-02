#!/usr/bin/env python3
"""Bounded natural-occlusion and frozen-identity Gate for M1 sweep maps."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from authoritative_dataset.cuda_renderer_v1 import CudaAuthorityRenderer  # noqa
from authoritative_dataset.dynamic_motion_v2 import (  # noqa
    actor_position, load_motion_contract,
)
from authoritative_dataset.exact_backend_v1 import ExactAuthorityBVH  # noqa
from authoritative_dataset.occlusion_constructor_v2_1 import (  # noqa
    build_occluded_actor_specs_v2_1,
    sample_occlusion_uav_sequence_v2_1,
)
from authoritative_dataset.perception_probe_v2 import run_frozen_perception_probe  # noqa


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA required")
    sweep = json.loads(
        (ROOT / "reports/phase8jqv2_4m1_map_type_sweep.json").read_text()
    )
    contract = load_motion_contract(
        ROOT / "configs/authoritative_dynamic_motion_contract_v2_1.yaml"
    )
    sensor = {
        "height": 96, "width": 160,
        "intrinsics": [80.0, 80.0, 80.0, 45.0],
        "max_depth_m": 20.0, "ray_step_m": 0.1,
        "frame_period_ns": 100000000,
    }
    renderer = CudaAuthorityRenderer(sensor, "cuda:0")
    frame_times = np.arange(60, dtype=np.float64) * .1
    requested_types = os.environ.get("PHASE8_M1_TYPES", "").strip()
    selected_types = (
        [int(value) for value in requested_types.split(",")]
        if requested_types else sweep["selected_map_types"]
    )
    rows = []
    for order, maze_type in enumerate(selected_types):
        started = time.perf_counter()
        map_row = next(
            row for row in sweep["maps"] if row["maze_type"] == maze_type
        )
        backend = ExactAuthorityBVH(map_row["authority_root"])
        rng = np.random.default_rng(842000000 + maze_type)
        gap = 1 + order % 3
        try:
            uav = sample_occlusion_uav_sequence_v2_1(
                backend, rng, frame_times, gap
            )
            construction = build_occluded_actor_specs_v2_1(
                backend, renderer, rng, uav.position_world, uav.yaw,
                frame_times, contract, gap, method="deterministic",
                maximum_candidates=64,
            )
            actors = np.empty((60, len(construction.actors), 3), np.float64)
            for actor_index, actor in enumerate(construction.actors):
                actors[:, actor_index] = actor_position(actor, frame_times)
            diagnostics = construction.diagnostics
            visible = diagnostics["per_actor_visible_pixel_count"][:, 0] > 0
            minimum_frames = int(np.ceil(
                contract["sampling"]["minimum_sustained_dynamic_duration_s"] / .1
            ))
            probe = run_frozen_perception_probe(
                diagnostics["composed_depth"], uav.position_world, uav.yaw,
                actors, frame_times, sensor,
                minimum_sustained_frames=minimum_frames,
                visibility_mask=visible, require_occlusion_identity=True,
                prediction_position_error_max_m=float(
                    contract["validation"]["prediction_position_error_max_m"]
                ),
            )
            row = {
                "maze_type": maze_type, "map_uuid": map_row["map_uuid"],
                "requested_gap_frames": gap,
                "constructed_gap": [
                    construction.gap_start, construction.gap_end
                ],
                "constructor_attempts": construction.attempts,
                "constructor_method": construction.method,
                "frozen_perception": probe,
                "elapsed_seconds": time.perf_counter() - started,
                "status": probe["status"],
            }
        except Exception as error:
            row = {
                "maze_type": maze_type, "map_uuid": map_row["map_uuid"],
                "requested_gap_frames": gap,
                "elapsed_seconds": time.perf_counter() - started,
                "status": "FAIL", "error": repr(error),
            }
        rows.append(row)
        print(json.dumps({
            "maze_type": maze_type, "status": row["status"],
            "elapsed_seconds": row["elapsed_seconds"],
            "error": row.get("error"),
        }), flush=True)
    report = {
        "status": "PASS" if all(row["status"] == "PASS" for row in rows)
        else "FAIL",
        "map_types": rows,
        "passing_type_count": sum(row["status"] == "PASS" for row in rows),
        "required_passing_type_count": (
            4 if not requested_types else len(selected_types)
        ),
        "gaps_requested": sorted(set(row["requested_gap_frames"] for row in rows)),
        "frozen_perception_modified": False,
        "formal_generation_started": False,
        "production_test_accessed": False, "blind_accessed": False,
    }
    atomic_json(
        ROOT / "reports/phase8jqv2_4m1_occlusion_host_validation.json",
        report,
    )
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
