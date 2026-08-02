"""Dense, differentiable kinodynamic supervision for quintic YOPO candidates.

The design adapts two ideas from EGO-Planner to YOPO's single-segment
quintics without pretending that a B-spline control-point formula applies to
terminal P/V/A states:

* velocity and acceleration feasibility are optimized directly, at every
  trajectory sample, so gradients reach the candidate (trajectory) head;
* the EGO time-stretch diagnostic
  ``max(v / v_max, sqrt(a / a_max))`` is supervised continuously.

Acceleration is also resolved into tangent and normal components.  Normal
acceleration receives an earlier soft margin at high speed, which teaches the
network to slow through sharp bends while preserving progress and candidate
direction.  Hard 6 m/s and 6 m/s^2 limits remain validation/runtime contracts.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class KinodynamicFeasibilityConfigV2:
    enabled: bool = False
    max_speed_mps: float = 6.0
    max_acceleration_mps2: float = 6.0
    soft_limit_ratio: float = 0.90
    normal_acceleration_soft_ratio: float = 0.75
    dense_norm_weight: float = 1.0
    axis_weight: float = 0.25
    normal_acceleration_weight: float = 0.50
    time_dilation_weight: float = 1.0
    candidate_mean_weight: float = 0.50
    candidate_cvar_weight: float = 0.50
    candidate_cvar_fraction: float = 1.0 / 3.0
    feasible_coverage_weight: float = 0.50
    minimum_feasible_candidates: int = 3
    label_weight: float = 1.0
    numerical_epsilon: float = 1.0e-6

    @classmethod
    def from_mapping(cls, value):
        value = {} if value is None else dict(value)
        unknown = sorted(set(value) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown kinodynamic_v2 keys: {unknown}")
        result = cls(**value)
        result.validate()
        return result

    def validate(self):
        if self.max_speed_mps <= 0 or self.max_acceleration_mps2 <= 0:
            raise ValueError("kinodynamic limits must be positive")
        if not 0 < self.soft_limit_ratio <= 1:
            raise ValueError("soft_limit_ratio must be in (0, 1]")
        if not 0 < self.normal_acceleration_soft_ratio <= 1:
            raise ValueError(
                "normal_acceleration_soft_ratio must be in (0, 1]"
            )
        weights = (
            self.dense_norm_weight, self.axis_weight,
            self.normal_acceleration_weight, self.time_dilation_weight,
            self.candidate_mean_weight, self.candidate_cvar_weight,
            self.feasible_coverage_weight, self.label_weight,
        )
        if min(weights) < 0:
            raise ValueError("kinodynamic weights must be non-negative")
        if not 0 < self.candidate_cvar_fraction <= 1:
            raise ValueError("candidate_cvar_fraction must be in (0, 1]")
        if self.minimum_feasible_candidates < 1:
            raise ValueError("minimum_feasible_candidates must be positive")
        if self.numerical_epsilon <= 0:
            raise ValueError("numerical_epsilon must be positive")


def _sample_quintic_pva(sampler, fixed_derivatives, predicted_derivatives):
    """Return dense position/velocity/acceleration, including both endpoints."""
    coefficients = sampler.coefficients(
        fixed_derivatives, predicted_derivatives
    )
    times = torch.linspace(
        0.0, sampler.duration, sampler.eval_points + 1,
        device=coefficients.device, dtype=coefficients.dtype,
    ).view(1, -1)
    positions, velocities, accelerations = [], [], []
    for axis in range(3):
        c = coefficients[:, 6 * axis:6 * (axis + 1)]
        positions.append(
            c[:, 0:1] + c[:, 1:2] * times + c[:, 2:3] * times ** 2
            + c[:, 3:4] * times ** 3 + c[:, 4:5] * times ** 4
            + c[:, 5:6] * times ** 5
        )
        velocities.append(
            c[:, 1:2] + 2.0 * c[:, 2:3] * times
            + 3.0 * c[:, 3:4] * times ** 2
            + 4.0 * c[:, 4:5] * times ** 3
            + 5.0 * c[:, 5:6] * times ** 4
        )
        accelerations.append(
            2.0 * c[:, 2:3] + 6.0 * c[:, 3:4] * times
            + 12.0 * c[:, 4:5] * times ** 2
            + 20.0 * c[:, 5:6] * times ** 3
        )
    return (
        torch.stack(positions, dim=2),
        torch.stack(velocities, dim=2),
        torch.stack(accelerations, dim=2),
        times.squeeze(0),
    )


def dense_quintic_kinodynamic_objective_v2(
    sampler, fixed_derivatives, predicted_derivatives,
    batch_size, candidate_count, config: KinodynamicFeasibilityConfigV2,
):
    """Compute candidate and per-sample physical feasibility objectives.

    The returned loss is differentiable with respect to all terminal P/V/A
    components.  Boolean masks and hard maxima are diagnostics only.
    """
    config.validate()
    _, velocity, acceleration, sample_times = _sample_quintic_pva(
        sampler, fixed_derivatives, predicted_derivatives
    )
    expected = int(batch_size) * int(candidate_count)
    if velocity.shape[0] != expected:
        raise ValueError("kinodynamic candidate count mismatch")

    eps = config.numerical_epsilon
    speed = velocity.norm(dim=2)
    acceleration_norm = acceleration.norm(dim=2)
    speed_ratio = speed / config.max_speed_mps
    acceleration_ratio = acceleration_norm / config.max_acceleration_mps2

    # Hard feasibility is based on vector norms, matching the controller.
    maximum_speed = speed.amax(dim=1)
    maximum_acceleration = acceleration_norm.amax(dim=1)
    hard_time_dilation = torch.maximum(
        maximum_speed / config.max_speed_mps,
        torch.sqrt((maximum_acceleration / config.max_acceleration_mps2).clamp_min(0.0)),
    ).clamp_min(1.0)

    # Dense vector-norm barrier spreads gradient across the trajectory rather
    # than only through the single argmax sample.
    norm_barrier = (
        F.relu(speed_ratio - config.soft_limit_ratio).square()
        + F.relu(acceleration_ratio - config.soft_limit_ratio).square()
    ).mean(dim=1)

    # EGO evaluates feasibility per coordinate of its B-spline control-point
    # derivatives.  This term transfers that directional sensitivity while
    # the norm barrier above enforces the controller's actual spherical bound.
    axis_speed_ratio = velocity.abs() / config.max_speed_mps
    axis_acceleration_ratio = acceleration.abs() / config.max_acceleration_mps2
    axis_barrier = (
        F.relu(axis_speed_ratio - config.soft_limit_ratio).square().mean((1, 2))
        + F.relu(
            axis_acceleration_ratio - config.soft_limit_ratio
        ).square().mean((1, 2))
    )

    # Trajectory-relative anisotropy: normal acceleration bends the flight
    # direction, whereas tangent acceleration changes speed.  Penalizing the
    # normal component earlier at high speed creates a differentiable
    # "slow-down through corners" signal without rewarding hover.
    tangent = velocity / speed.clamp_min(eps).unsqueeze(2)
    tangent_acceleration = (acceleration * tangent).sum(dim=2)
    normal_acceleration = (
        acceleration - tangent_acceleration.unsqueeze(2) * tangent
    ).norm(dim=2)
    normal_ratio = normal_acceleration / config.max_acceleration_mps2
    high_speed_weight = speed_ratio.clamp(0.0, 1.5).square()
    normal_barrier = (
        F.relu(
            normal_ratio - config.normal_acceleration_soft_ratio
        ).square() * high_speed_weight
    ).mean(dim=1)

    time_dilation_barrier = (hard_time_dilation - 1.0).square()
    candidate_loss = (
        config.dense_norm_weight * norm_barrier
        + config.axis_weight * axis_barrier
        + config.normal_acceleration_weight * normal_barrier
        + config.time_dilation_weight * time_dilation_barrier
    ).reshape(batch_size, candidate_count)

    maximum_speed = maximum_speed.reshape(batch_size, candidate_count)
    maximum_acceleration = maximum_acceleration.reshape(
        batch_size, candidate_count
    )
    hard_time_dilation = hard_time_dilation.reshape(
        batch_size, candidate_count
    )
    feasible = (
        (maximum_speed <= config.max_speed_mps)
        & (maximum_acceleration <= config.max_acceleration_mps2)
    )

    top_count = max(
        1, int(candidate_count * config.candidate_cvar_fraction + 0.999999)
    )
    candidate_mean = candidate_loss.mean(dim=1)
    candidate_cvar = candidate_loss.topk(
        top_count, dim=1, largest=True
    ).values.mean(dim=1)

    # A smooth coverage deficit keeps at least several candidates close to the
    # feasible set even when an early checkpoint has zero hard-feasible paths.
    # Unlike safe-vs-unsafe ranking, it therefore never loses its gradient.
    coverage_count = min(config.minimum_feasible_candidates, candidate_count)
    best_candidate_losses = candidate_loss.topk(
        coverage_count, dim=1, largest=False
    ).values
    coverage_loss = best_candidate_losses.mean(dim=1)
    per_sample_loss = (
        config.candidate_mean_weight * candidate_mean
        + config.candidate_cvar_weight * candidate_cvar
        + config.feasible_coverage_weight * coverage_loss
    )

    return {
        "per_sample_loss": per_sample_loss,
        "candidate_loss": candidate_loss,
        "candidate_label_cost": config.label_weight * candidate_loss.detach(),
        "maximum_speed": maximum_speed,
        "maximum_acceleration": maximum_acceleration,
        "time_dilation_ratio": hard_time_dilation,
        "feasible_mask": feasible,
        "feasible_candidate_count": feasible.sum(dim=1),
        "sample_times": sample_times,
        "maximum_normal_acceleration": normal_acceleration.amax(dim=1).reshape(
            batch_size, candidate_count
        ),
    }
