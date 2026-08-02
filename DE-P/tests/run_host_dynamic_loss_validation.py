#!/usr/bin/env python3
"""Short CUDA validation of corrected DEP attention and future collision loss."""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.config import cfg
from loss.dynamic_safety_loss import DynamicCollisionLoss
from loss.dynamic_types import DynamicLossConfig, DynamicObstacleBatch
from loss.trajectory_sampler import QuinticTrajectorySampler
from policy.checkpoint_utils import load_dep_checkpoint
from policy.dep_network import DepNetwork
from policy.dynamic.context import ATTENTION_BACKBONE_OUTPUT, DynamicContext
from policy.dynamic.types import DynamicPerceptionConfig
from tests.baseline_helpers import reset_lattice_singleton, seed_everything


def coefficient_map(duration, device):
    matrix = torch.zeros(6, 6, dtype=torch.float32, device=device)
    for derivative in range(3):
        matrix[2 * derivative, derivative] = math.factorial(derivative)
        for power in range(derivative, 6):
            matrix[2 * derivative + 1, power] = (
                math.factorial(power) / math.factorial(power - derivative)
                * duration ** (power - derivative)
            )
    permutation = torch.zeros(6, 6, device=device)
    permutation[[0, 2, 4, 1, 3, 5], [0, 1, 2, 3, 4, 5]] = 1
    return torch.linalg.inv(matrix) @ permutation


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required; run this script with host full-access")
    seed_everything(6001)
    device = torch.device("cuda:0")
    checkpoint = ROOT / "saved/DEP_corrected_init/epoch10_converted.pth"
    perception = replace(
        DynamicPerceptionConfig.from_global_config(), enabled=True, use_attention=True
    )
    reset_lattice_singleton()
    model = DepNetwork(backbone_variant="corrected", dynamic_config=perception).to(device)
    load_dep_checkpoint(model, checkpoint, "corrected")
    model.train()
    depth = torch.rand(1, 1, 96, 160, device=device)
    observation = torch.tensor(
        [[0.2, 0, 0, 0, 0, 0, 8, 0, 0]], dtype=torch.float32, device=device
    )
    context = DynamicContext(
        attention_maps_by_level={
            ATTENTION_BACKBONE_OUTPUT: torch.zeros(1, 1, 3, 5, device=device)
        }, source="depth", valid=True,
    )
    context.attention_maps_by_level[ATTENTION_BACKBONE_OUTPUT][0, 0, 1, 2] = 1.0
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    endstate, score = model.inference(depth, observation, dynamic_context=context)

    trajectory_count = cfg["traj_num"]
    start = torch.zeros(trajectory_count, 3, 3, device=device)
    start[:, 2, 0] = 2.0
    predicted = endstate.permute(0, 2, 3, 1).reshape(trajectory_count, 3, 3).transpose(1, 2)
    sampler = QuinticTrajectorySampler(
        coefficient_map(cfg["sgm_time"], device), cfg["sgm_time"], 30
    ).to(device)
    loss_config = replace(
        DynamicLossConfig.from_global_config(), enabled=True,
        target_source="constant_velocity",
    )
    collision = DynamicCollisionLoss(sampler, loss_config).to(device)
    obstacles = DynamicObstacleBatch(
        positions_world=torch.tensor([[[2.5, 0.0, 2.0]]], device=device),
        velocities_world=torch.tensor([[[-1.0, 0.0, 0.0]]], device=device),
        position_covariances=torch.eye(3, device=device).reshape(1, 1, 3, 3) * 0.01,
        radii=torch.tensor([[0.35]], device=device),
        track_timestamps=torch.tensor([[1.0]], device=device),
        sample_timestamps=torch.tensor([1.0], device=device),
        confidence=torch.ones(1, 1, device=device),
        valid_mask=torch.ones(1, 1, dtype=torch.bool, device=device),
        dynamic_mask=torch.ones(1, 1, dtype=torch.bool, device=device),
    )
    dynamic_cost, diagnostics = collision(start, predicted, obstacles)
    score_label = dynamic_cost.reshape(-1).detach()
    total = dynamic_cost.mean() + torch.nn.functional.smooth_l1_loss(
        score.reshape(-1), score_label
    )
    total.backward()
    torch.cuda.synchronize(device)
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    elapsed_ms = (time.perf_counter() - started) * 1000
    status = (
        tuple(endstate.shape) == (1, 9, 3, 5)
        and tuple(score.shape) == (1, 3, 5)
        and bool(torch.isfinite(dynamic_cost).all())
        and float(dynamic_cost.max()) > 0
        and gradients and all(bool(torch.isfinite(g).all()) for g in gradients)
    )
    result = {
        "status": "PASS" if status else "FAIL",
        "device": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "checkpoint": str(checkpoint),
        "dynamic_context_shape": [1, 1, 3, 5],
        "endstate_shape": list(endstate.shape),
        "score_shape": list(score.shape),
        "dynamic_loss_mean": float(dynamic_cost.mean().detach().cpu()),
        "score_label_includes_dynamic_cost": True,
        "score_label_requires_grad": score_label.requires_grad,
        "dynamic_gradient_tensor_count": len(gradients),
        "min_dynamic_distance": float(diagnostics.min_dynamic_distance.mean().cpu()),
        "risky_trajectory_ratio": float(diagnostics.risky_trajectory_ratio.mean().cpu()),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "elapsed_ms": elapsed_ms,
    }
    print("HOST_DYNAMIC_LOSS_VALIDATION_RESULT")
    print(json.dumps(result, indent=2))
    return 0 if status else 2


if __name__ == "__main__":
    raise SystemExit(main())
