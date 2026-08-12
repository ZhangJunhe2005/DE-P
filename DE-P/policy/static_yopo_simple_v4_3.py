"""Versioned V4.3 static YOPO objective.

This restores one learnable cost for every one of the 15 proposals.  It does
not create score-ranking classes, project endpoints, or turn camera visibility
and minimum-progress heuristics into labels.  Dynamic actors are intentionally
absent: they remain a deterministic causal runtime filter.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class SimpleYOPOConfigV43:
    enabled: bool = False
    derivative_samples: int = 81
    jerk_unit_weight: float = 10.0
    acceleration_unit_weight: float = 1.0
    safety_weight: float = 1.0
    guidance_weight: float = 0.15
    guidance_perpendicular_weight_max: float = 0.50
    guidance_goal_length_m: float = 10.0
    hardware_weight: float = 4.0
    score_weight: float = 1.0
    training_speed_mps: float = 6.0
    max_speed_mps: float = 6.0
    max_acceleration_mps2: float = 6.0

    @classmethod
    def from_mapping(cls, value):
        value = {} if value is None else dict(value)
        unknown = sorted(set(value) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown simple_yopo_v4_3 keys: {unknown}")
        result = cls(**value)
        result.validate()
        return result

    def validate(self):
        if self.derivative_samples < 3:
            raise ValueError("derivative_samples must be at least three")
        weights = (
            self.jerk_unit_weight, self.acceleration_unit_weight,
            self.safety_weight, self.guidance_weight, self.hardware_weight,
            self.score_weight,
        )
        if min(weights) < 0.0:
            raise ValueError("V4.3 objective weights must be non-negative")
        if not 0.0 <= self.guidance_perpendicular_weight_max <= 1.0:
            raise ValueError("guidance_perpendicular_weight_max must be in [0,1]")
        if min(self.training_speed_mps, self.max_speed_mps,
               self.max_acceleration_mps2, self.guidance_goal_length_m) <= 0.0:
            raise ValueError("V4.3 speed and acceleration scales must be positive")

    def contract(self):
        return {
            "version": "static_yopo_single_total_cost_v4_3",
            "score_label": "detached_candidate_total_cost",
            "candidate_gradient": "all_15_candidates",
            "static_safety": "full_trajectory_esdf_integral",
            "qualification_gate": "none",
            "dynamic_training": False,
            **asdict(self),
        }


def _sample_quintic_derivatives(sampler, fixed, predicted, sample_count):
    coefficient = sampler.coefficients(fixed, predicted)
    times = torch.linspace(
        0.0, sampler.duration, int(sample_count),
        device=fixed.device, dtype=fixed.dtype,
    )
    velocity_power = torch.stack([
        torch.ones_like(times), 2.0 * times, 3.0 * times ** 2,
        4.0 * times ** 3, 5.0 * times ** 4,
    ], dim=1)
    acceleration_power = torch.stack([
        2.0 * torch.ones_like(times), 6.0 * times, 12.0 * times ** 2,
        20.0 * times ** 3,
    ], dim=1)
    jerk_power = torch.stack([
        6.0 * torch.ones_like(times), 24.0 * times, 60.0 * times ** 2,
    ], dim=1)
    velocities, accelerations, jerks = [], [], []
    for axis in range(3):
        axis_coefficient = coefficient[:, 6 * axis:6 * (axis + 1)]
        velocities.append(axis_coefficient[:, 1:] @ velocity_power.T)
        accelerations.append(axis_coefficient[:, 2:] @ acceleration_power.T)
        jerks.append(axis_coefficient[:, 3:] @ jerk_power.T)
    return (
        torch.stack(velocities, dim=2),
        torch.stack(accelerations, dim=2),
        torch.stack(jerks, dim=2),
        times,
    )


def simple_yopo_objective_v4_3(
    *, sampler, safety_loss, fixed, predicted, predicted_scores, goal_world,
    map_id, batch_size, candidate_count, route_goal_distance, config,
):
    """Compute the V4.3 objective and trainer-compatible diagnostics."""
    config.validate()
    if fixed.shape != predicted.shape or fixed.shape[1:] != (3, 3):
        raise ValueError("fixed/predicted derivatives must be [B*N,3,3]")
    if predicted_scores.shape != (batch_size, candidate_count):
        raise ValueError("predicted score shape violates V4.3 contract")

    velocity, acceleration, jerk, times = _sample_quintic_derivatives(
        sampler, fixed, predicted, config.derivative_samples
    )
    maximum_speed = velocity.norm(dim=2).amax(dim=1).reshape(
        batch_size, candidate_count
    )
    maximum_acceleration = acceleration.norm(dim=2).amax(dim=1).reshape(
        batch_size, candidate_count
    )
    jerk_integral = torch.trapezoid(
        jerk.square().sum(dim=2), times, dim=1
    ).reshape(batch_size, candidate_count)
    acceleration_integral = torch.trapezoid(
        acceleration.square().sum(dim=2), times, dim=1
    ).reshape(batch_size, candidate_count)
    smooth_cost = (
        config.jerk_unit_weight / config.training_speed_mps ** 5
        * jerk_integral
        + config.acceleration_unit_weight / config.training_speed_mps ** 3
        * acceleration_integral
    )

    sampled_position, _ = sampler(fixed, predicted)
    static_sample_cost, static_distance = safety_loss.get_distance_cost(
        sampled_position.reshape(batch_size, -1, 3), map_id
    )
    static_sample_cost = static_sample_cost.reshape(
        batch_size, candidate_count, sampler.eval_points
    )
    static_distance = static_distance.reshape(
        batch_size, candidate_count, sampler.eval_points
    )
    # The historical YOPO safety objective is an integral over the complete
    # trajectory.  No first-collision truncation or CVaR is used in V4.3.
    static_cost = config.safety_weight * sampler.duration * static_sample_cost.mean(
        dim=2
    )
    minimum_clearance = static_distance.amin(dim=2)

    current_position = fixed[:, :, 0].reshape(
        batch_size, candidate_count, 3
    )[:, 0]
    endpoint = predicted[:, :, 0].reshape(batch_size, candidate_count, 3)
    trajectory_delta = endpoint - current_position[:, None, :]
    goal_delta = goal_world.reshape(batch_size, candidate_count, 3)[:, 0] \
        - current_position
    goal_length = goal_delta.norm(dim=1)
    goal_direction = goal_delta / goal_length[:, None].clamp_min(1.0e-8)
    goal_progress = (trajectory_delta * goal_direction[:, None, :]).sum(dim=2)
    perpendicular = (
        trajectory_delta - goal_progress[:, :, None] * goal_direction[:, None, :]
    ).norm(dim=2)
    perpendicular_weight = (
        1.0 - goal_length / config.guidance_goal_length_m
    ).clamp(0.0, 1.0) * config.guidance_perpendicular_weight_max
    guidance_raw = (
        (1.0 - perpendicular_weight[:, None])
        * (goal_length[:, None] - goal_progress).abs()
        + perpendicular_weight[:, None] * perpendicular
    )
    guidance_cost = config.guidance_weight * guidance_raw

    speed_excess = F.relu(maximum_speed / config.max_speed_mps - 1.0)
    acceleration_excess = F.relu(
        maximum_acceleration / config.max_acceleration_mps2 - 1.0
    )
    hardware_cost = config.hardware_weight * (
        speed_excess.square() + acceleration_excess.square()
    )
    candidate_total = smooth_cost + static_cost + guidance_cost + hardware_cost
    label = candidate_total.detach()
    score_per_candidate = F.smooth_l1_loss(
        predicted_scores, label, reduction="none"
    )
    per_sample_score = score_per_candidate.mean(dim=1)
    per_sample_trajectory = candidate_total.mean(dim=1)
    per_sample_total = per_sample_trajectory + config.score_weight * per_sample_score

    row = torch.arange(batch_size, device=fixed.device)
    selected_index = predicted_scores.argmin(dim=1)
    selected_hardware_unsafe = (
        (maximum_speed > config.max_speed_mps)
        | (maximum_acceleration > config.max_acceleration_mps2)
    )[row, selected_index]
    hardware_feasible = (
        (maximum_speed <= config.max_speed_mps)
        & (maximum_acceleration <= config.max_acceleration_mps2)
    )
    endpoint_distance = trajectory_delta.norm(dim=2)
    endpoint_speed = predicted[:, :, 1].reshape(
        batch_size, candidate_count, 3
    ).norm(dim=2)
    time_dilation = torch.maximum(
        maximum_speed / config.max_speed_mps,
        torch.sqrt((maximum_acceleration / config.max_acceleration_mps2).clamp_min(0.0)),
    ).clamp_min(1.0)
    selected_clearance = minimum_clearance[row, selected_index]
    zeros = per_sample_total * 0.0
    zero_candidates = candidate_total * 0.0

    # Keep the established trainer/telemetry schema while giving the old
    # fields neutral values.  V4.3 has one scalar validation objective, not a
    # hidden collection of qualification gates.
    result = {
        "total_loss": per_sample_total.mean(),
        "trajectory_loss": per_sample_trajectory.mean(),
        "score_loss": per_sample_score.mean(),
        "smoothness_loss": smooth_cost.mean(),
        "static_safety_loss": static_cost.mean(),
        "guidance_loss": guidance_cost.mean(),
        "dynamic_safety_loss": zeros.mean(),
        "ranking_loss": zeros.mean(),
        "safety_cvar_loss": zeros.mean(),
        "kinematic_loss": hardware_cost.mean(),
        "preventive_safety_loss": zeros.mean(),
        "preventive_ranking_loss": zeros.mean(),
        "progress_safety_loss": zeros.mean(),
        "progress_ranking_loss": zeros.mean(),
        "stopping_distance_loss": zeros.mean(),
        "score_label": label.reshape(-1),
        "candidate_smooth_cost": smooth_cost.reshape(-1),
        "candidate_static_cost": static_cost.reshape(-1),
        "candidate_guidance_cost": guidance_cost.reshape(-1),
        "candidate_kinodynamic_cost": hardware_cost.reshape(-1),
        "candidate_preventive_cost": zero_candidates,
        "candidate_progress_cost": zero_candidates,
        "candidate_projected_safe_mask": hardware_feasible,
        "candidate_stopping_distance_cost": zero_candidates.reshape(-1),
        "per_sample_total_loss": per_sample_total,
        "per_sample_trajectory_loss": per_sample_trajectory,
        "per_sample_score_loss": per_sample_score,
        "per_sample_kinematic_loss": hardware_cost.mean(dim=1),
        "per_sample_smoothness_loss": smooth_cost.mean(dim=1),
        "per_sample_static_safety_loss": static_cost.mean(dim=1),
        "per_sample_guidance_loss": guidance_cost.mean(dim=1),
        "per_sample_selected_endpoint_distance": endpoint_distance[row, selected_index],
        "per_sample_selected_endpoint_speed": endpoint_speed[row, selected_index],
        "per_sample_hover_selection": endpoint_distance[row, selected_index].lt(0.75).float(),
        "per_sample_route_goal_distance": route_goal_distance,
        "per_sample_objective_goal_distance": goal_length,
        "per_sample_score_top1_match": selected_index.eq(label.argmin(dim=1)),
        "per_sample_unsafe_selection": selected_hardware_unsafe.float(),
        "per_sample_hardware_unsafe_selection": selected_hardware_unsafe.float(),
        "per_sample_selected_trajectory_max_speed": maximum_speed[row, selected_index],
        "per_sample_selected_trajectory_max_acceleration": maximum_acceleration[row, selected_index],
        "per_sample_feasible_candidate_count": hardware_feasible.sum(dim=1).to(fixed.dtype),
        "per_sample_candidate_time_dilation_mean": time_dilation.mean(dim=1),
        "per_sample_selected_time_dilation": time_dilation[row, selected_index],
        "per_sample_selected_normal_acceleration": zeros,
        "per_sample_selected_clearance": selected_clearance,
        "per_sample_clear_candidate_count": minimum_clearance.ge(0.65).sum(dim=1).to(fixed.dtype),
        "per_sample_selected_goal_progress": goal_progress[row, selected_index],
        "per_sample_selected_goal_alignment": (
            goal_progress[row, selected_index]
            / endpoint_distance[row, selected_index].clamp_min(1.0e-8)
        ),
        "per_sample_reverse_selection": goal_progress[row, selected_index].lt(0.0).float(),
    }
    neutral_names = (
        "ranking_loss", "safety_cvar_loss",
        "preventive_safety_loss", "preventive_ranking_loss",
        "progress_safety_loss", "progress_ranking_loss",
        "stopping_distance_loss", "stoppable_candidate_count",
        "preventive_required_clearance", "anticipatory_unsafe_selection",
        "safe_progress_candidate_count", "preferred_progress_candidate_count",
        "insufficient_progress_selection", "projected_candidate_availability",
        "projected_safe_candidate_count",
        "projected_conditional_selection_error_numerator",
        "projected_selected_stopping_reserve", "projected_selected_visible",
        "projected_selection_regret", "selected_normal_acceleration",
    )
    for name in neutral_names:
        result.setdefault(f"per_sample_{name}", zeros)
    result["per_sample_preventive_sample_weight"] = zeros + 1.0
    return result
