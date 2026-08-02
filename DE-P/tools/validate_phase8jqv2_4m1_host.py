#!/usr/bin/env python3
"""Bounded host CUDA validation for the M1 development map sweep."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from authoritative_dataset.cuda_renderer_v1 import (  # noqa: E402
    CudaAuthorityRenderer, RENDERER_VERSION,
)
from authoritative_dataset.exact_backend_v1 import ExactAuthorityBVH  # noqa: E402
from authoritative_dataset.generate_v1 import depth_image_cpu, safe_position  # noqa: E402
from authoritative_dataset.perception_probe_v2 import (  # noqa: E402
    run_frozen_perception_probe,
)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA is required; do not accept sandbox skip")
    sweep = json.loads(
        (ROOT / "reports/phase8jqv2_4m1_map_type_sweep.json").read_text()
    )
    sensor = {
        "height": 96, "width": 160,
        "intrinsics": [80.0, 80.0, 80.0, 45.0],
        "max_depth_m": 20.0, "ray_step_m": 0.1,
    }
    renderer = CudaAuthorityRenderer(sensor, "cuda:0")
    rows = []
    for maze_type in sweep["selected_map_types"]:
        map_row = next(
            row for row in sweep["maps"] if row["maze_type"] == maze_type
        )
        backend = ExactAuthorityBVH(map_row["authority_root"])
        rng = np.random.default_rng(841000000 + maze_type)
        position, safe_query = safe_position(backend, rng)
        yaw = float(rng.uniform(-np.pi, np.pi))
        actor_positions = np.empty((1, 0, 3), dtype=np.float64)
        result = renderer.render_with_actor_diagnostics(
            backend, position[None], [yaw], actor_positions, []
        )
        cuda_depth = result["static_depth"][0]
        cpu_depth = depth_image_cpu(backend, position, yaw, sensor)
        difference = np.abs(cuda_depth - cpu_depth)
        no_target = run_frozen_perception_probe(
            np.repeat(cuda_depth[None], 20, axis=0),
            np.repeat(position[None], 20, axis=0),
            np.repeat(yaw, 20),
            np.empty((20, 0, 3), dtype=np.float64),
            np.arange(20, dtype=np.float64) * 0.1,
            sensor, minimum_sustained_frames=3,
        )
        row = {
            "maze_type": maze_type,
            "map_uuid": map_row["map_uuid"],
            "depth_shape": list(cuda_depth.shape),
            "finite": bool(np.isfinite(cuda_depth).all()),
            "cpu_cuda_max_abs_m": float(difference.max()),
            "cpu_cuda_p99_abs_m": float(np.quantile(difference, .99)),
            "cpu_cuda_within_one_ray_step": bool(
                np.quantile(difference, .99) <= .100001
            ),
            "no_target_false_attention_frames":
                no_target["no_target_false_attention_frames"],
            "no_target_gate": no_target["status"],
            "exact_backend_query": safe_query,
        }
        row["status"] = "PASS" if (
            row["finite"] and row["cpu_cuda_within_one_ray_step"]
            and row["no_target_false_attention_frames"] == 0
            and row["no_target_gate"] == "PASS"
            and not row["exact_backend_query"]["collision"]
        ) else "FAIL"
        rows.append(row)
    report = {
        "status": "PASS" if all(row["status"] == "PASS" for row in rows)
        else "FAIL",
        "device": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "renderer": RENDERER_VERSION,
        "map_types": rows,
        "production_test_accessed": False,
        "blind_accessed": False,
        "optimizer_step_executed": False,
        "formal_generation_started": False,
    }
    atomic_json(
        ROOT / "reports/phase8jqv2_4m1_map_host_validation.json", report
    )
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
