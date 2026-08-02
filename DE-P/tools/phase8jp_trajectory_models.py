"""Differentiable offline trajectory models for Phase 8J-P."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TrajectorySamples:
    position: torch.Tensor
    velocity: torch.Tensor
    acceleration: torch.Tensor
    jerk: torch.Tensor
    times: torch.Tensor


def _boundary_matrix(duration, device, dtype):
    t = torch.as_tensor(duration, device=device, dtype=dtype)
    zero = torch.zeros((), device=device, dtype=dtype)
    one = torch.ones((), device=device, dtype=dtype)
    return torch.stack((
        torch.stack((one, zero, zero, zero, zero, zero)),
        torch.stack((zero, one, zero, zero, zero, zero)),
        torch.stack((zero, zero, 2 * one, zero, zero, zero)),
        torch.stack((one, t, t**2, t**3, t**4, t**5)),
        torch.stack((zero, one, 2*t, 3*t**2, 4*t**3, 5*t**4)),
        torch.stack((zero, zero, 2*one, 6*t, 12*t**2, 20*t**3)),
    ))


def quintic_coefficients(start, end, duration):
    """Return [..., xyz, coefficient] for [..., xyz, pva] boundaries."""
    if start.shape != end.shape or start.shape[-2:] != (3, 3):
        raise ValueError("start/end states must be [...,3,3] in [xyz,pva]")
    matrix = _boundary_matrix(duration, start.device, start.dtype)
    boundary = torch.cat((start, end), dim=-1)
    return torch.linalg.solve(matrix, boundary.unsqueeze(-1)).squeeze(-1)


def sample_quintic(start, end, duration, points, time_offset=0.0):
    if points <= 0:
        raise ValueError("points must be positive")
    coefficient = quintic_coefficients(start, end, duration)
    dt = float(duration) / points
    local = torch.linspace(
        dt, float(duration), points, device=start.device, dtype=start.dtype
    )
    powers = torch.stack((
        torch.ones_like(local), local, local**2, local**3, local**4, local**5
    ), dim=-1)
    velocity_powers = torch.stack((
        torch.zeros_like(local), torch.ones_like(local), 2*local,
        3*local**2, 4*local**3, 5*local**4,
    ), dim=-1)
    acceleration_powers = torch.stack((
        torch.zeros_like(local), torch.zeros_like(local),
        2*torch.ones_like(local), 6*local, 12*local**2, 20*local**3,
    ), dim=-1)
    jerk_powers = torch.stack((
        torch.zeros_like(local), torch.zeros_like(local),
        torch.zeros_like(local), 6*torch.ones_like(local),
        24*local, 60*local**2,
    ), dim=-1)
    return TrajectorySamples(
        position=torch.einsum("...xc,tc->...tx", coefficient, powers),
        velocity=torch.einsum(
            "...xc,tc->...tx", coefficient, velocity_powers
        ),
        acceleration=torch.einsum(
            "...xc,tc->...tx", coefficient, acceleration_powers
        ),
        jerk=torch.einsum(
            "...xc,tc->...tx", coefficient, jerk_powers
        ),
        times=local + float(time_offset),
    )


def sample_piecewise_quintic(
    start, midpoint, end, first_duration, second_duration, points=30
):
    if first_duration <= 0 or second_duration <= 0:
        raise ValueError("segment durations must be positive")
    first_points = max(
        1, round(points * first_duration / (first_duration + second_duration))
    )
    second_points = points - first_points
    if second_points <= 0:
        first_points -= 1
        second_points = 1
    first = sample_quintic(
        start, midpoint, first_duration, first_points, 0.0
    )
    second = sample_quintic(
        midpoint, end, second_duration, second_points, first_duration
    )
    return TrajectorySamples(
        position=torch.cat((first.position, second.position), dim=-2),
        velocity=torch.cat((first.velocity, second.velocity), dim=-2),
        acceleration=torch.cat(
            (first.acceleration, second.acceleration), dim=-2
        ),
        jerk=torch.cat((first.jerk, second.jerk), dim=-2),
        times=torch.cat((first.times, second.times)),
    )


def integrate_piecewise_constant_jerk(
    start, jerk, horizon, samples_per_interval=4
):
    """Exact constant-jerk integration with continuous P/V/A state."""
    if start.shape[-2:] != (3, 3) or jerk.shape[-1] != 3:
        raise ValueError("invalid start or jerk shape")
    intervals = jerk.shape[-2]
    if intervals <= 0 or samples_per_interval <= 0:
        raise ValueError("interval/sample counts must be positive")
    dt = float(horizon) / intervals
    position = start[..., :, 0]
    velocity = start[..., :, 1]
    acceleration = start[..., :, 2]
    positions, velocities, accelerations, jerks, times = [], [], [], [], []
    elapsed = 0.0
    for interval in range(intervals):
        control = jerk[..., interval, :]
        for step in range(1, samples_per_interval + 1):
            tau = dt * step / samples_per_interval
            positions.append(
                position + velocity*tau + 0.5*acceleration*tau**2
                + control*tau**3/6.0
            )
            velocities.append(
                velocity + acceleration*tau + 0.5*control*tau**2
            )
            accelerations.append(acceleration + control*tau)
            jerks.append(control)
            times.append(elapsed + tau)
        position = (
            position + velocity*dt + 0.5*acceleration*dt**2
            + control*dt**3/6.0
        )
        velocity = velocity + acceleration*dt + 0.5*control*dt**2
        acceleration = acceleration + control*dt
        elapsed += dt
    return TrajectorySamples(
        position=torch.stack(positions, dim=-2),
        velocity=torch.stack(velocities, dim=-2),
        acceleration=torch.stack(accelerations, dim=-2),
        jerk=torch.stack(jerks, dim=-2),
        times=torch.tensor(times, device=start.device, dtype=start.dtype),
    )
