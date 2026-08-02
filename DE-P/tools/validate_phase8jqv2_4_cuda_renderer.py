#!/usr/bin/env python3
"""Short host-only CPU/CUDA renderer and actor-compositing regression."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from authoritative_dataset.cuda_renderer_v1 import (
    CudaAuthorityRenderer, RENDERER_VERSION,
)
from authoritative_dataset.exact_backend_v1 import ExactAuthorityBVH
from authoritative_dataset.generate_v1 import depth_image_cpu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    smoke = ROOT/"data/phase8_authoritative_generation_smoke_v1"
    sequence_path = next((smoke/"manifests/sequences").glob("*.json"))
    sequence = json.loads(sequence_path.read_text())
    frame_path = smoke / (
        f"{sequence['suite']}/{sequence['split']}/"
        f"{sequence['sequence_id']}/frames.jsonl")
    frame = json.loads(frame_path.read_text().splitlines()[0])
    map_root = next(
        (smoke/"geometry_authority").rglob(frame["map_uuid"]))
    backend = ExactAuthorityBVH(map_root)
    sensor = yaml.safe_load(
        (ROOT/"configs/phase8_authoritative_v1_generation.yaml").read_text()
    )["sensor_settings"]
    position = np.asarray(frame["position_world"], dtype=np.float64)
    quaternion = frame["quaternion_world_from_body"]
    yaw = 2*np.arctan2(quaternion[3], quaternion[0])
    cpu = depth_image_cpu(backend, position, yaw, sensor)
    renderer = CudaAuthorityRenderer(sensor, "cuda:0")
    empty = np.empty((1, 0, 3), dtype=np.float32)
    gpu, static, no_actor_count = renderer.render(
        backend, position[None], np.asarray([yaw]), empty, [])
    actor = position + np.asarray([np.cos(yaw)*2, np.sin(yaw)*2, 0])
    composed, _, actor_count = renderer.render(
        backend, position[None], np.asarray([yaw]),
        actor.reshape(1, 1, 3), [.2])
    q23 = json.loads(
        (ROOT/"reports/phase8jqv2_3_final_result.json").read_text())
    report = {
        **{key: q23[key] for key in (
            "Simulator_geometry_hash", "cache_index_hash",
            "checkpoint_hash", "config_hash", "dataset_manifest_hash",
            "evaluator_version", "geometry_hash", "timeline_hash",
            "uncertainty_policy_hash")},
        "status": "PASS",
        "renderer_version": RENDERER_VERSION,
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "cpu_cuda_static_exact_equal": bool(np.array_equal(cpu, gpu[0])),
        "cpu_cuda_static_max_abs_difference":
            float(np.max(np.abs(cpu-gpu[0]))),
        "cpu_cuda_static_tolerance_m": 1e-6,
        "no_actor_pixel_count": int(no_actor_count[0]),
        "actor_pixel_count": int(actor_count[0]),
        "actor_reduces_depth": bool(np.any(composed[0] < static[0])),
    }
    report["cpu_cuda_static_within_tolerance"] = (
        report["cpu_cuda_static_max_abs_difference"]
        <= report["cpu_cuda_static_tolerance_m"])
    report["status"] = "PASS" if (
        report["cpu_cuda_static_within_tolerance"]
        and report["no_actor_pixel_count"] == 0
        and report["actor_pixel_count"] > 0
        and report["actor_reduces_depth"]
    ) else "FAIL"
    if not args.no_write:
        path = ROOT/"reports/phase8jqv2_4_cuda_renderer_validation.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
