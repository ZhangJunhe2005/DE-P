#!/usr/bin/env python3
"""Finite-gradient Gate for the estimated dynamic training surrogate."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.config import cfg
from loss.dynamic_safety_loss import DynamicCollisionLoss
from loss.dynamic_types import DynamicLossConfig, DynamicObstacleBatch
from loss.trajectory_sampler import QuinticTrajectorySampler


def coefficient_map(duration, dtype=torch.float64):
    matrix = torch.zeros(6, 6, dtype=dtype)
    for derivative in range(3):
        matrix[2 * derivative, derivative] = math.factorial(derivative)
        for power in range(derivative, 6):
            matrix[2 * derivative + 1, power] = (
                math.factorial(power) / math.factorial(power-derivative)
                * duration ** (power-derivative)
            )
    permutation = torch.zeros(6, 6, dtype=dtype)
    permutation[[0, 2, 4, 1, 3, 5], range(6)] = 1
    return torch.linalg.inv(matrix) @ permutation


def obstacle_batch(batch, covariance, valid, positions=None):
    obstacles = covariance.shape[1]
    positions = (
        torch.zeros(batch, obstacles, 3, dtype=covariance.dtype)
        if positions is None else positions
    )
    return DynamicObstacleBatch(
        positions_world=positions,
        velocities_world=torch.zeros_like(positions),
        position_covariances=covariance,
        radii=torch.full((batch, obstacles), 0.2, dtype=covariance.dtype),
        track_timestamps=torch.zeros(batch, obstacles, dtype=covariance.dtype),
        sample_timestamps=torch.zeros(batch, dtype=covariance.dtype),
        confidence=torch.ones(batch, obstacles, dtype=covariance.dtype),
        valid_mask=valid,
        dynamic_mask=torch.ones_like(valid),
        observable_mask=valid.clone(),
    )


def finite_backward(loss, fixed, predicted, obstacles):
    candidate = predicted.detach().clone().requires_grad_(True)
    value = loss(fixed, candidate, obstacles)[0].mean()
    value.backward()
    return {
        "loss": float(value.detach()),
        "gradient_finite": bool(torch.isfinite(candidate.grad).all()),
        "gradient_norm": float(candidate.grad.norm()),
    }


def main():
    torch.manual_seed(8251001)
    duration = float(cfg["sgm_time"])
    sampler = QuinticTrajectorySampler(
        coefficient_map(duration), duration, 30
    )
    config = replace(
        DynamicLossConfig.from_global_config(),
        enabled=True,
        target_source="constant_velocity",
        covariance_growth_rate=0.0,
    )
    loss = DynamicCollisionLoss(sampler, config).double()
    fixed = torch.zeros(30, 3, 3, dtype=torch.float64)
    predicted = torch.zeros_like(fixed)
    predicted[:, 0, 0] = torch.linspace(0.5, 2.0, 30)
    positions = torch.tensor(
        [[[0.0, 0.0, 0.0], [100.0, 100.0, 100.0]],
         [[1.0, 0.2, 0.0], [100.0, 100.0, 100.0]]],
        dtype=torch.float64,
    )
    zero_covariance = torch.zeros(2, 2, 3, 3, dtype=torch.float64)
    valid = torch.tensor([[True, False], [True, False]])
    cases = {
        "zero_covariance_zero_relative_displacement": obstacle_batch(
            2, zero_covariance, valid, positions
        ),
        "padded_inactive_actor": obstacle_batch(
            2, zero_covariance, valid, positions
        ),
        "positive_covariance": obstacle_batch(
            2,
            torch.eye(3, dtype=torch.float64).reshape(1, 1, 3, 3)
            .expand(2, 2, -1, -1).clone() * 0.01,
            valid,
            positions,
        ),
    }
    results = {
        name: finite_backward(loss, fixed, predicted, obstacles)
        for name, obstacles in cases.items()
    }

    # Central finite difference and autograd must agree in sign and magnitude.
    direction = torch.randn_like(predicted)
    direction /= direction.norm()
    variable = predicted.detach().clone().requires_grad_(True)
    objective = loss(fixed, variable, cases["positive_covariance"])[0].mean()
    gradient = torch.autograd.grad(objective, variable)[0]
    automatic = float((gradient * direction).sum())
    epsilon = 1e-5
    with torch.no_grad():
        plus = float(loss(
            fixed, predicted + epsilon * direction,
            cases["positive_covariance"],
        )[0].mean())
        minus = float(loss(
            fixed, predicted - epsilon * direction,
            cases["positive_covariance"],
        )[0].mean())
    finite_difference = (plus-minus) / (2*epsilon)
    relative_error = abs(automatic-finite_difference) / max(
        abs(automatic), abs(finite_difference), 1e-12
    )
    finite_difference_pass = (
        math.isfinite(automatic)
        and math.isfinite(finite_difference)
        and relative_error < 5e-4
    )

    empty = DynamicObstacleBatch.empty(2, timestamp=0.0)
    empty_result = finite_backward(loss, fixed, predicted, empty)
    checks = {
        "all_backward_gradients_finite": all(
            value["gradient_finite"] for value in results.values()
        ),
        "two_batch_backward": all(
            results[name]["gradient_finite"]
            for name in (
                "zero_covariance_zero_relative_displacement",
                "positive_covariance",
            )
        ),
        "finite_difference": finite_difference_pass,
        "empty_actor_set_exact_zero": empty_result["loss"] == 0.0,
        "no_optimizer_step": True,
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "implementation": "DynamicCollisionLoss",
        "cases": results,
        "empty_actor_set": empty_result,
        "finite_difference": {
            "autograd_directional_derivative": automatic,
            "central_difference": finite_difference,
            "relative_error": relative_error,
            "status": "PASS" if finite_difference_pass else "FAIL",
        },
        "checks": checks,
        "network_weights_modified": False,
        "optimizer_step_executed": False,
        "production_test_used": False,
        "blind_used": False,
        "source_sha256": hashlib.sha256(
            (ROOT / "loss/dynamic_safety_loss.py").read_bytes()
        ).hexdigest(),
    }
    output = ROOT / "reports/phase8jqv2_5_estimated_gradient_integrity.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    (ROOT / "reports/phase8jqv2_5_backward_smoke.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
