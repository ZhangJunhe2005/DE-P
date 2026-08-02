#!/usr/bin/env python3
"""Host GPU validation for the stage-4 dynamic perception device boundary."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policy.dynamic.attention import build_dynamic_attention
from policy.dynamic.camera_model import depth_to_pointcloud
from policy.dynamic.dynamic_perception import DynamicPerception
from tests.dynamic_helpers import (
    camera_cloud_from_world,
    camera_model,
    dynamic_track,
    pose,
    test_config,
    world_cluster,
)


def fail(message, code=2):
    print(f"HOST DYNAMIC PERCEPTION VALIDATION FAILED: {message}", file=sys.stderr)
    return code


def main():
    print("torch_version:", torch.__version__)
    print("torch_cuda_build:", torch.version.cuda)
    print("cuda_available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        return fail("NOT MEASURED IN WORKSPACE SANDBOX: CUDA device is not visible")
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    _ = torch.empty(1, device=device)  # initialize the CUDA context before memory-stat APIs
    torch.cuda.reset_peak_memory_stats(0)
    camera = camera_model()

    depth = torch.full(
        (camera.height, camera.width), 5.0, dtype=torch.float32, device=device
    )
    depth[0, 0] = float("nan")
    pointcloud_start = time.perf_counter()
    points_cuda = depth_to_pointcloud(depth, camera, stride=2)
    torch.cuda.synchronize(device)
    pointcloud_ms = (time.perf_counter() - pointcloud_start) * 1000
    if points_cuda.device.type != "cuda" or points_cuda.shape[1:] != (3,):
        return fail(f"depth back-projection crossed device or shape boundary: {points_cuda.shape}")
    if not torch.isfinite(points_cuda).all():
        return fail("CUDA depth back-projection contains NaN/Inf")

    config = test_config(
        cluster_eps=0.15,
        cluster_min_samples=5,
        association_distance_threshold=1.0,
        association_mahalanobis_threshold=100.0,
        min_confirmed_hits=2,
        dynamic_min_confirmed_hits=3,
        dynamic_max_velocity_std=2.0,
    )
    direct_attention = build_dynamic_attention(
        [dynamic_track()], pose(), camera, (3, 5), config=config, device=device
    )
    if direct_attention.device.type != "cuda" or not torch.isfinite(direct_attention).all():
        return fail("direct CUDA attention is on the wrong device or non-finite")

    perception = DynamicPerception(config, feature_shape=(3, 5), attention_device=device)
    offsets = world_cluster([0, 0, 0], seed=44, count=35, scale=0.02)
    result = None
    pipeline_start = time.perf_counter()
    for frame in range(12):
        timestamp = frame * 0.1
        current_pose = pose(timestamp=timestamp)
        center = np.array([0.04 * frame, 0.0, 5.0])
        cloud_camera = camera_cloud_from_world(offsets + center, current_pose)
        result = perception.update(cloud_camera, current_pose, timestamp, camera)
    torch.cuda.synchronize(device)
    pipeline_ms = (time.perf_counter() - pipeline_start) * 1000
    if len(result.dynamic_tracks) != 1 or len(result.projected_dynamic_tracks) != 1:
        return fail("end-to-end pipeline did not produce one projected dynamic track")
    if result.attention_map.device.type != "cuda" or not torch.isfinite(result.attention_map).all():
        return fail("pipeline attention is on the wrong device or non-finite")
    if not (0 < float(result.attention_map.max()) <= config.attention_max):
        return fail("pipeline attention is empty or exceeds its configured bound")
    peak_memory = int(torch.cuda.max_memory_allocated(device))
    if peak_memory <= 0:
        return fail("peak GPU memory is not positive")

    summary = {
        "status": "PASS",
        "device": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "torch_version": str(torch.__version__),
        "torch_cuda_build": torch.version.cuda,
        "depth_pointcloud_shape": list(points_cuda.shape),
        "depth_pointcloud_device": str(points_cuda.device),
        "depth_pointcloud_ms": pointcloud_ms,
        "direct_attention_shape": list(direct_attention.shape),
        "direct_attention_max": float(direct_attention.max()),
        "pipeline_frames": 12,
        "pipeline_total_ms": pipeline_ms,
        "dynamic_track_count": len(result.dynamic_tracks),
        "projected_dynamic_track_count": len(result.projected_dynamic_tracks),
        "pipeline_attention_shape": list(result.attention_map.shape),
        "pipeline_attention_device": str(result.attention_map.device),
        "pipeline_attention_max": float(result.attention_map.max()),
        "peak_gpu_memory_bytes": peak_memory,
        "cpu_gpu_boundary": "DBSCAN/KF NumPy CPU; attention and Torch depth projection CUDA",
    }
    print("HOST_DYNAMIC_PERCEPTION_VALIDATION_RESULT")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
