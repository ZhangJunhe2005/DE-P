"""Versioned safety-first score supervision without weakening anti-hover gates."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class StaticSafetyFirstConfigV1:
    enabled: bool = False
    required_clearance_m: float = 0.65
    unsafe_label_priority: float = 5.0
    ranking_margin: float = 1.0
    ranking_weight: float = 0.5
    safety_cvar_fraction: float = 1.0 / 3.0
    safety_cvar_weight: float = 0.5
    max_speed_mps: float = 6.0
    max_acceleration_mps2: float = 6.0
    kinematic_weight: float = 0.5

    @classmethod
    def from_mapping(cls, value):
        value = {} if value is None else dict(value)
        unknown = sorted(set(value) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown safety_first keys: {unknown}")
        result = cls(**value)
        result.validate()
        return result

    def validate(self):
        if self.required_clearance_m <= 0 or self.unsafe_label_priority <= 0:
            raise ValueError("safety-first clearance/priority must be positive")
        if self.ranking_margin <= 0 or self.ranking_weight < 0:
            raise ValueError("invalid safety-first ranking contract")
        if not 0 < self.safety_cvar_fraction <= 1 or self.safety_cvar_weight < 0:
            raise ValueError("invalid safety CVaR contract")
        if self.max_speed_mps <= 0 or self.max_acceleration_mps2 <= 0:
            raise ValueError("kinematic limits must be positive")
        if self.kinematic_weight < 0:
            raise ValueError("kinematic_weight must be non-negative")


def sample_quintic_kinematic_maxima_v1(sampler, fixed_derivatives,
                                       predicted_derivatives, batch_size,
                                       candidate_count):
    coefficient = sampler.coefficients(fixed_derivatives, predicted_derivatives)
    times = torch.linspace(
        sampler.duration / sampler.eval_points, sampler.duration,
        sampler.eval_points, device=coefficient.device, dtype=coefficient.dtype,
    ).view(1, -1)
    velocity_axes = []
    acceleration_axes = []
    for axis in range(3):
        values = coefficient[:, 6 * axis:6 * (axis + 1)]
        velocity_axes.append(
            values[:, 1:2] + 2 * values[:, 2:3] * times
            + 3 * values[:, 3:4] * times ** 2
            + 4 * values[:, 4:5] * times ** 3
            + 5 * values[:, 5:6] * times ** 4
        )
        acceleration_axes.append(
            2 * values[:, 2:3] + 6 * values[:, 3:4] * times
            + 12 * values[:, 4:5] * times ** 2
            + 20 * values[:, 5:6] * times ** 3
        )
    speed = torch.stack(velocity_axes, dim=2).norm(dim=2)
    acceleration = torch.stack(acceleration_axes, dim=2).norm(dim=2)
    expected = batch_size * candidate_count
    if speed.shape[0] != expected:
        raise ValueError("kinematic candidate count mismatch")
    return (
        speed.amax(dim=1).reshape(batch_size, candidate_count),
        acceleration.amax(dim=1).reshape(batch_size, candidate_count),
    )


def safety_first_score_objective_v1(predicted_scores, base_labels,
                                    static_min_distance, static_safety_cost,
                                    config: StaticSafetyFirstConfigV1,
                                    trajectory_max_speed=None,
                                    trajectory_max_acceleration=None):
    """Build lexicographic labels and safe-vs-unsafe pairwise ranking loss."""
    config.validate()
    if predicted_scores.shape != base_labels.shape:
        raise ValueError("predicted score and label shapes differ")
    if static_min_distance.shape != base_labels.shape:
        raise ValueError("static clearance and label shapes differ")
    static_unsafe = static_min_distance < config.required_clearance_m
    if trajectory_max_speed is None or trajectory_max_acceleration is None:
        trajectory_max_speed = torch.zeros_like(static_min_distance)
        trajectory_max_acceleration = torch.zeros_like(static_min_distance)
    if (trajectory_max_speed.shape != base_labels.shape
            or trajectory_max_acceleration.shape != base_labels.shape):
        raise ValueError("kinematic maxima and label shapes differ")
    hardware_unsafe = (
        (trajectory_max_speed > config.max_speed_mps)
        | (trajectory_max_acceleration > config.max_acceleration_mps2)
    )
    unsafe = static_unsafe | hardware_unsafe
    labels = base_labels + unsafe.to(base_labels.dtype) * config.unsafe_label_priority
    regression_per_sample = F.smooth_l1_loss(
        predicted_scores, labels.detach(), reduction="none"
    ).mean(dim=1)

    # Every safe candidate must rank below every unsafe candidate. Samples
    # without both classes contribute exactly zero rather than fabricating a pair.
    pair_loss = F.relu(
        predicted_scores[:, :, None] - predicted_scores[:, None, :]
        + config.ranking_margin
    )
    pair_mask = (~unsafe)[:, :, None] & unsafe[:, None, :]
    pair_count = pair_mask.sum(dim=(1, 2))
    ranking_per_sample = (
        (pair_loss * pair_mask).sum(dim=(1, 2))
        / pair_count.clamp(min=1).to(pair_loss.dtype)
    )
    ranking_per_sample = torch.where(
        pair_count > 0, ranking_per_sample, torch.zeros_like(ranking_per_sample)
    )

    top_count = max(
        1, int(static_safety_cost.shape[1] * config.safety_cvar_fraction + 0.999999)
    )
    safety_cvar_per_sample = static_safety_cost.topk(
        top_count, dim=1, largest=True
    ).values.mean(dim=1)
    speed_violation = F.relu(
        trajectory_max_speed / config.max_speed_mps - 1.0
    ).square()
    acceleration_violation = F.relu(
        trajectory_max_acceleration / config.max_acceleration_mps2 - 1.0
    ).square()
    kinematic_per_sample = (speed_violation + acceleration_violation).mean(dim=1)
    selected = predicted_scores.argmin(dim=1)
    selected_unsafe = unsafe[
        torch.arange(unsafe.shape[0], device=unsafe.device), selected
    ]
    return {
        "labels": labels.detach(),
        "regression_per_sample": regression_per_sample,
        "ranking_per_sample": ranking_per_sample,
        "safety_cvar_per_sample": safety_cvar_per_sample,
        "kinematic_per_sample": kinematic_per_sample,
        "unsafe_mask": unsafe,
        "selected_unsafe": selected_unsafe,
        "hardware_unsafe_mask": hardware_unsafe,
        "selected_hardware_unsafe": hardware_unsafe[
            torch.arange(unsafe.shape[0], device=unsafe.device), selected
        ],
    }
