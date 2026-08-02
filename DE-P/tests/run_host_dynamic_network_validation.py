#!/usr/bin/env python3
"""One-shot host CUDA validation for phase-5 network integration."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policy.checkpoint_utils import load_dep_checkpoint
from policy.dep_network import DepNetwork
from policy.dynamic.context import DynamicContext
from policy.dynamic.dynamic_perception import DynamicPerception
from policy.dynamic.types import DynamicPerceptionConfig
from tests.dynamic_helpers import camera_cloud_from_world, camera_model, pose, world_cluster


def synchronize():
    torch.cuda.synchronize()


def benchmark(model, depth, obs, context=None, warmup=20, repeats=100):
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(depth, obs, dynamic_context=context)
        synchronize()
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            output = model(depth, obs, dynamic_context=context)
            synchronize()
            samples.append((time.perf_counter() - start) * 1000.0)
    return output, {
        "average_ms": statistics.mean(samples),
        "p95_ms": float(np.percentile(samples, 95)),
    }


def assert_finite(output):
    if not all(bool(torch.isfinite(value).all()) for value in output):
        raise RuntimeError("network output contains NaN/Inf")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("HOST CUDA validation requires a visible NVIDIA GPU")
    device = torch.device("cuda:0")
    torch.manual_seed(20260721)
    torch.cuda.manual_seed_all(20260721)
    static_config = DynamicPerceptionConfig.from_global_config()
    dynamic_config = replace(static_config, enabled=True, use_attention=True)
    dynamic_config.validate()
    depth = torch.linspace(0, 1, 96 * 160, device=device).reshape(1, 1, 96, 160)
    obs = torch.linspace(-0.5, 0.5, 9 * 15, device=device).reshape(1, 9, 3, 5)
    # Initialize the CUDA context before querying/resetting allocator statistics.
    torch.cuda.reset_peak_memory_stats(device)

    checkpoints = {
        "legacy": ROOT / "saved/DEP_0/epoch10.pth",
        "corrected": ROOT / "saved/DEP_corrected_init/epoch10_converted.pth",
    }
    models = {}
    static_results = {}
    for variant, checkpoint in checkpoints.items():
        model = DepNetwork(
            backbone_variant=variant, dynamic_config=static_config
        ).to(device).eval()
        load_dep_checkpoint(model, str(checkpoint), variant)
        output, latency = benchmark(model, depth, obs)
        assert_finite(output)
        models[variant] = model
        static_results[variant] = {
            "checkpoint": str(checkpoint),
            "strict_load": True,
            "endstate_shape": list(output[0].shape),
            "score_shape": list(output[1].shape),
            "latency": latency,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        }

    dynamic_model = DepNetwork(
        backbone_variant="corrected", dynamic_config=dynamic_config
    ).to(device).eval()
    load_dep_checkpoint(dynamic_model, str(checkpoints["corrected"]), "corrected")
    static_reference = dynamic_model._forward_impl(depth, obs)
    missing_output = dynamic_model(depth, obs, dynamic_context=None)
    if not all(torch.equal(a, b) for a, b in zip(static_reference, missing_output)):
        raise RuntimeError("missing DynamicContext did not fall back exactly to static")

    perception = DynamicPerception(
        dynamic_config, feature_shape=(3, 5), attention_device="cpu"
    )
    base_cluster = world_cluster([0, 0, 0], seed=55, count=40, scale=0.02)
    result = None
    generation_samples = []
    for frame in range(12):
        timestamp = frame * 0.1
        current_pose = pose(timestamp=timestamp)
        cloud = camera_cloud_from_world(
            base_cluster + np.asarray([0.04 * frame, 0.0, 5.0]), current_pose
        )
        start = time.perf_counter()
        result = perception.update(cloud, current_pose, timestamp, camera_model())
        generation_samples.append((time.perf_counter() - start) * 1000.0)
    cpu_context = DynamicContext.from_perception_result(result, 1.1, "depth")
    transfer_start = time.perf_counter()
    attention_cuda = result.attention_map.to(device, non_blocking=True)
    synchronize()
    transfer_ms = (time.perf_counter() - transfer_start) * 1000.0
    cuda_context = DynamicContext(
        attention_maps_by_level={"backbone_output": attention_cuda},
        dynamic_tracks=cpu_context.dynamic_tracks,
        timestamp=cpu_context.timestamp,
        source="depth",
        valid=True,
        diagnostics=cpu_context.diagnostics,
    )
    zero_context = DynamicContext(
        attention_maps_by_level={"backbone_output": torch.zeros_like(attention_cuda)},
        timestamp=1.1, source="depth", valid=True,
    )
    zero_output = dynamic_model(depth, obs, zero_context)
    if not all(torch.equal(a, b) for a, b in zip(static_reference, zero_output)):
        raise RuntimeError("zero attention is not exactly static-equivalent")

    target_output, dynamic_latency = benchmark(
        dynamic_model, depth, obs, cuda_context
    )
    assert_finite(target_output)
    if torch.equal(static_reference[0], target_output[0]):
        raise RuntimeError("non-zero dynamic attention did not affect network output")

    dynamic_model.train()
    optimizer = torch.optim.AdamW(dynamic_model.parameters(), lr=1e-5)
    optimizer.zero_grad(set_to_none=True)
    train_output = dynamic_model(depth, obs, cuda_context)
    loss = train_output[0].square().mean() + train_output[1].mean()
    loss.backward()
    gradients = [p.grad for p in dynamic_model.parameters() if p.grad is not None]
    if not gradients or not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
        raise RuntimeError("dynamic forward/backward produced missing or invalid gradients")
    optimizer.step()
    synchronize()

    report = {
        "status": "PASS",
        "device": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch_version": torch.__version__,
        "static": static_results,
        "dynamic_corrected": {
            "empty_context_exact_static_fallback": True,
            "zero_attention_exact_static": True,
            "dynamic_track_count": len(result.dynamic_tracks),
            "attention_shape": list(attention_cuda.shape),
            "attention_min": float(attention_cuda.min()),
            "attention_max": float(attention_cuda.max()),
            "attention_generation_average_ms": statistics.mean(generation_samples),
            "attention_transfer_ms": transfer_ms,
            "endstate_max_abs_delta_from_static": float(
                (target_output[0] - static_reference[0]).abs().max()
            ),
            "score_max_abs_delta_from_static": float(
                (target_output[1] - static_reference[1]).abs().max()
            ),
            "latency": dynamic_latency,
            "forward_backward": "PASS",
            "gradient_tensor_count": len(gradients),
            "loss": float(loss.detach()),
        },
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(0),
        "all_outputs_finite": True,
    }
    print("HOST_DYNAMIC_NETWORK_VALIDATION_RESULT")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
