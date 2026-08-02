"""Shared quintic trajectory sampling for static and dynamic collision costs."""

from __future__ import annotations

import torch
from torch import nn


class QuinticTrajectorySampler(nn.Module):
    """Sample positions from fixed/start and predicted/end PVA derivatives.

    Inputs use ``[K,3,3]`` in ``[xyz, pva]`` order, matching SafetyLoss after
    its historical permutation. The implementation deliberately preserves the
    original matmul and time grid so phase-2A static costs remain unchanged.
    """

    def __init__(self, coefficient_map, duration: float, eval_points: int = 30):
        super().__init__()
        if coefficient_map.shape != (6, 6):
            raise ValueError("coefficient_map must have shape [6,6]")
        if duration <= 0 or eval_points <= 0:
            raise ValueError("duration and eval_points must be positive")
        self.register_buffer("coefficient_map", coefficient_map.detach().clone())
        self.duration = float(duration)
        self.eval_points = int(eval_points)

    def coefficients(self, fixed_derivatives, predicted_derivatives):
        if fixed_derivatives.shape != predicted_derivatives.shape:
            raise ValueError("fixed and predicted derivative shapes must match")
        if fixed_derivatives.ndim != 3 or fixed_derivatives.shape[1:] != (3, 3):
            raise ValueError("derivatives must have shape [K,3,3] in [xyz,pva] order")
        count = fixed_derivatives.shape[0]
        coefficient_map = self.coefficient_map.to(
            device=fixed_derivatives.device, dtype=fixed_derivatives.dtype
        ).unsqueeze(0).expand(count, -1, -1)
        coefficient = torch.zeros(
            count, 18, device=fixed_derivatives.device, dtype=fixed_derivatives.dtype
        )
        for axis in range(3):
            derivative = torch.cat(
                [fixed_derivatives[:, axis, :], predicted_derivatives[:, axis, :]], dim=1
            ).unsqueeze(-1)
            axis_coefficient = (coefficient_map @ derivative).squeeze(-1)
            coefficient[:, 6 * axis:6 * (axis + 1)] = axis_coefficient
        return coefficient

    def forward(self, fixed_derivatives, predicted_derivatives):
        coefficient = self.coefficients(fixed_derivatives, predicted_derivatives)
        dt = self.duration / self.eval_points
        sample_times = torch.linspace(
            dt, self.duration, self.eval_points,
            device=fixed_derivatives.device, dtype=fixed_derivatives.dtype,
        )
        time = sample_times.view(1, -1, 1).expand(coefficient.shape[0], -1, -1)
        time_power = torch.stack(
            [torch.ones_like(time), time, time ** 2, time ** 3, time ** 4, time ** 5],
            dim=-1,
        ).squeeze(-2)
        positions = []
        for axis in range(3):
            axis_coefficient = coefficient[:, 6 * axis:6 * (axis + 1)]
            positions.append(torch.sum(time_power * axis_coefficient.unsqueeze(1), dim=-1))
        return torch.stack(positions, dim=-1), sample_times

    def grouped(self, fixed_derivatives, predicted_derivatives, batch_size):
        positions, sample_times = self(fixed_derivatives, predicted_derivatives)
        if batch_size <= 0 or positions.shape[0] % batch_size:
            raise ValueError("flat trajectory count must be divisible by batch size")
        trajectory_count = positions.shape[0] // batch_size
        return positions.reshape(batch_size, trajectory_count, self.eval_points, 3), sample_times
