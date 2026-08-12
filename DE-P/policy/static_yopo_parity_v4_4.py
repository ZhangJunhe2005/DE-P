"""Original-YOPO static objective with one relative score calibration term.

V4.4 deliberately keeps the MobileNetV3/15-primitive network unchanged.  It
restores the original analytic jerk/acceleration costs, full-trajectory static
ESDF mean, and fixed guidance geometry.  A single listwise KL term teaches the
score channel the within-sample ordering of the same detached total cost; it
does not create a qualification Gate or a second safety policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class StaticYOPOParityConfigV44:
    enabled: bool = False
    derivative_samples: int = 81
    jerk_unit_weight: float = 10.0
    acceleration_unit_weight: float = 1.0
    safety_weight: float = 1.0
    guidance_weight: float = 0.15
    guidance_perpendicular_weight: float = 0.50
    score_regression_weight: float = 1.0
    relative_order_weight: float = 0.25
    relative_order_temperature: float = 0.75
    training_speed_mps: float = 6.0
    max_speed_mps: float = 6.0
    max_acceleration_mps2: float = 6.0
    vehicle_radius_m: float = 0.30
    clear_distance_m: float = 1.20

    @classmethod
    def from_mapping(cls, value):
        value = {} if value is None else dict(value)
        unknown = sorted(set(value) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown static_yopo_v4_4 keys: {unknown}")
        result = cls(**value)
        result.validate()
        return result

    def validate(self):
        if self.derivative_samples < 3:
            raise ValueError("derivative_samples must be at least three")
        nonnegative = (
            self.jerk_unit_weight, self.acceleration_unit_weight,
            self.safety_weight, self.guidance_weight,
            self.guidance_perpendicular_weight, self.score_regression_weight,
            self.relative_order_weight, self.vehicle_radius_m,
            self.clear_distance_m,
        )
        if min(nonnegative) < 0.0:
            raise ValueError("V4.4 weights/distances must be non-negative")
        positive = (
            self.relative_order_temperature, self.training_speed_mps,
            self.max_speed_mps, self.max_acceleration_mps2,
        )
        if min(positive) <= 0.0:
            raise ValueError("V4.4 scales and limits must be positive")

    def contract(self):
        return {
            "version": "static_yopo_original_parity_v4_4",
            "network": "unchanged_dep_mobilenetv3_15_primitive_unified_head",
            "candidate_objective": "original_yopo_analytic_total_cost",
            "score_label": "detached_original_candidate_total_cost",
            "score_calibration": "single_within_sample_listwise_kl",
            "qualification_gate": "none",
            "dynamic_training": False,
            **asdict(self),
        }


def derivative_quadratic_matrix(duration, derivative_order, *, device, dtype):
    matrix = torch.zeros((6, 6), device=device, dtype=dtype)
    for row in range(derivative_order, 6):
        row_scale = math.factorial(row) / math.factorial(row - derivative_order)
        for column in range(derivative_order, 6):
            column_scale = (
                math.factorial(column)
                / math.factorial(column - derivative_order)
            )
            power = row + column - 2 * derivative_order + 1
            matrix[row, column] = (
                row_scale * column_scale * float(duration) ** power / power
            )
    return matrix


def exact_derivative_integral(coefficient, duration, derivative_order):
    if coefficient.ndim != 3 or coefficient.shape[1:] != (3, 6):
        raise ValueError("coefficient must be [K,3,6]")
    matrix = derivative_quadratic_matrix(
        duration, derivative_order,
        device=coefficient.device, dtype=coefficient.dtype,
    )
    return torch.einsum("kai,ij,kaj->k", coefficient, matrix, coefficient)


def _sample_velocity_acceleration(sampler, fixed, predicted, sample_count):
    coefficient = sampler.coefficients(fixed, predicted).reshape(-1, 3, 6)
    times = torch.linspace(
        0.0, sampler.duration, int(sample_count),
        device=fixed.device, dtype=fixed.dtype,
    )
    velocity_power = torch.stack([
        torch.ones_like(times), 2.0 * times, 3.0 * times ** 2,
        4.0 * times ** 3, 5.0 * times ** 4,
    ])
    acceleration_power = torch.stack([
        2.0 * torch.ones_like(times), 6.0 * times,
        12.0 * times ** 2, 20.0 * times ** 3,
    ])
    velocity = torch.einsum(
        "kai,it->kat", coefficient[:, :, 1:], velocity_power
    ).permute(0, 2, 1)
    acceleration = torch.einsum(
        "kai,it->kat", coefficient[:, :, 2:], acceleration_power
    ).permute(0, 2, 1)
    return coefficient, velocity, acceleration


def _relative_order_kl(predicted_scores, detached_cost, temperature):
    relative_cost = detached_cost - detached_cost.amin(dim=1, keepdim=True)
    scale = relative_cost.std(dim=1, unbiased=False, keepdim=True).clamp_min(1.0e-3)
    target_probability = torch.softmax(
        -(relative_cost / scale) / float(temperature), dim=1
    )
    network_log_probability = torch.log_softmax(
        -predicted_scores / float(temperature), dim=1
    )
    target_log_probability = target_probability.clamp_min(1.0e-12).log()
    return (
        target_probability * (target_log_probability - network_log_probability)
    ).sum(dim=1).clamp_min(0.0)


def static_yopo_parity_objective_v4_4(
    *, sampler, safety_loss, fixed, predicted, predicted_scores, goal_world,
    map_id, batch_size, candidate_count, route_goal_distance, config,
):
    config.validate()
    if fixed.shape != predicted.shape or fixed.shape[1:] != (3, 3):
        raise ValueError("fixed/predicted derivatives must be [B*N,3,3]")
    if predicted_scores.shape != (batch_size, candidate_count):
        raise ValueError("predicted score shape violates V4.4 contract")

    coefficient, velocity, acceleration = _sample_velocity_acceleration(
        sampler, fixed, predicted, config.derivative_samples
    )
    jerk_integral = exact_derivative_integral(
        coefficient, sampler.duration, derivative_order=3
    ).reshape(batch_size, candidate_count)
    acceleration_integral = exact_derivative_integral(
        coefficient, sampler.duration, derivative_order=2
    ).reshape(batch_size, candidate_count)
    smooth_cost = (
        config.jerk_unit_weight / config.training_speed_mps ** 5 * jerk_integral
        + config.acceleration_unit_weight / config.training_speed_mps ** 3
        * acceleration_integral
    )

    sampled_position, _ = sampler(fixed, predicted)
    static_samples, static_distance = safety_loss.get_distance_cost(
        sampled_position.reshape(batch_size, -1, 3), map_id
    )
    static_samples = static_samples.reshape(
        batch_size, candidate_count, sampler.eval_points
    )
    static_distance = static_distance.reshape(
        batch_size, candidate_count, sampler.eval_points
    )
    # Original YOPO's implementation used the sampled full-trajectory mean;
    # it did not multiply that mean by segment duration.
    static_cost = config.safety_weight * static_samples.mean(dim=2)
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
    guidance_raw = F.smooth_l1_loss(
        goal_progress, goal_length[:, None].expand_as(goal_progress),
        reduction="none",
    ) + config.guidance_perpendicular_weight * perpendicular
    guidance_cost = config.guidance_weight * guidance_raw

    candidate_total = smooth_cost + static_cost + guidance_cost
    label = candidate_total.detach()
    score_regression = F.smooth_l1_loss(
        predicted_scores, label, reduction="none"
    ).mean(dim=1)
    relative_order = _relative_order_kl(
        predicted_scores, label, config.relative_order_temperature
    )
    per_sample_trajectory = candidate_total.mean(dim=1)
    per_sample_score = config.score_regression_weight * score_regression
    per_sample_ranking = config.relative_order_weight * relative_order
    per_sample_total = (
        per_sample_trajectory + per_sample_score + per_sample_ranking
    )

    maximum_speed = velocity.norm(dim=2).amax(dim=1).reshape(
        batch_size, candidate_count
    )
    maximum_acceleration = acceleration.norm(dim=2).amax(dim=1).reshape(
        batch_size, candidate_count
    )
    hardware_feasible = (
        (maximum_speed <= config.max_speed_mps)
        & (maximum_acceleration <= config.max_acceleration_mps2)
    )
    collision_free = minimum_clearance >= config.vehicle_radius_m
    physical_feasible = hardware_feasible & collision_free
    endpoint_distance = trajectory_delta.norm(dim=2)
    endpoint_speed = predicted[:, :, 1].reshape(
        batch_size, candidate_count, 3
    ).norm(dim=2)
    time_dilation = torch.maximum(
        maximum_speed / config.max_speed_mps,
        torch.sqrt(
            (maximum_acceleration / config.max_acceleration_mps2).clamp_min(0.0)
        ),
    ).clamp_min(1.0)
    row = torch.arange(batch_size, device=fixed.device)
    selected = predicted_scores.argmin(dim=1)
    oracle = label.argmin(dim=1)
    oracle_regret = label[row, selected] - label[row, oracle]
    selected_hardware_unsafe = ~hardware_feasible[row, selected]
    selected_collision_unsafe = ~collision_free[row, selected]
    zeros = per_sample_total * 0.0
    zero_candidates = candidate_total * 0.0

    result = {
        "total_loss": per_sample_total.mean(),
        "trajectory_loss": per_sample_trajectory.mean(),
        "score_loss": per_sample_score.mean(),
        "smoothness_loss": smooth_cost.mean(),
        "static_safety_loss": static_cost.mean(),
        "guidance_loss": guidance_cost.mean(),
        "dynamic_safety_loss": zeros.mean(),
        "ranking_loss": per_sample_ranking.mean(),
        "safety_cvar_loss": zeros.mean(),
        "kinematic_loss": zeros.mean(),
        "preventive_safety_loss": zeros.mean(),
        "preventive_ranking_loss": zeros.mean(),
        "progress_safety_loss": zeros.mean(),
        "progress_ranking_loss": zeros.mean(),
        "stopping_distance_loss": zeros.mean(),
        "score_label": label.reshape(-1),
        "candidate_smooth_cost": smooth_cost.reshape(-1),
        "candidate_static_cost": static_cost.reshape(-1),
        "candidate_guidance_cost": guidance_cost.reshape(-1),
        "candidate_kinodynamic_cost": zero_candidates.reshape(-1),
        "candidate_preventive_cost": zero_candidates,
        "candidate_progress_cost": zero_candidates,
        "candidate_projected_safe_mask": physical_feasible,
        "candidate_stopping_distance_cost": zero_candidates.reshape(-1),
        "per_sample_total_loss": per_sample_total,
        "per_sample_trajectory_loss": per_sample_trajectory,
        "per_sample_score_loss": per_sample_score,
        "per_sample_ranking_loss": per_sample_ranking,
        "per_sample_kinematic_loss": zeros,
        "per_sample_smoothness_loss": smooth_cost.mean(dim=1),
        "per_sample_static_safety_loss": static_cost.mean(dim=1),
        "per_sample_guidance_loss": guidance_cost.mean(dim=1),
        "per_sample_selected_endpoint_distance": endpoint_distance[row, selected],
        "per_sample_selected_endpoint_speed": endpoint_speed[row, selected],
        "per_sample_hover_selection": endpoint_distance[row, selected].lt(0.75).float(),
        "per_sample_route_goal_distance": route_goal_distance,
        "per_sample_objective_goal_distance": goal_length,
        "per_sample_score_top1_match": selected.eq(oracle),
        "per_sample_score_oracle_regret": oracle_regret,
        "per_sample_unsafe_selection": selected_collision_unsafe.float(),
        "per_sample_hardware_unsafe_selection": selected_hardware_unsafe.float(),
        "per_sample_selected_trajectory_max_speed": maximum_speed[row, selected],
        "per_sample_selected_trajectory_max_acceleration": maximum_acceleration[row, selected],
        "per_sample_feasible_candidate_count": physical_feasible.sum(dim=1).to(fixed.dtype),
        "per_sample_candidate_time_dilation_mean": time_dilation.mean(dim=1),
        "per_sample_selected_time_dilation": time_dilation[row, selected],
        "per_sample_selected_normal_acceleration": zeros,
        "per_sample_selected_clearance": minimum_clearance[row, selected],
        "per_sample_clear_candidate_count": minimum_clearance.ge(
            config.clear_distance_m
        ).sum(dim=1).to(fixed.dtype),
        "per_sample_selected_goal_progress": goal_progress[row, selected],
        "per_sample_selected_goal_alignment": (
            goal_progress[row, selected]
            / endpoint_distance[row, selected].clamp_min(1.0e-8)
        ),
        "per_sample_reverse_selection": goal_progress[row, selected].lt(0.0).float(),
        # Trainer compatibility: retain the old telemetry field while making
        # its V4.4 meaning explicit through score_oracle_regret above.
        "per_sample_projected_selection_regret": oracle_regret,
    }
    neutral_names = (
        "safety_cvar_loss", "preventive_safety_loss",
        "preventive_ranking_loss", "progress_safety_loss",
        "progress_ranking_loss", "stopping_distance_loss",
        "stoppable_candidate_count", "preventive_required_clearance",
        "anticipatory_unsafe_selection", "safe_progress_candidate_count",
        "preferred_progress_candidate_count", "insufficient_progress_selection",
        "projected_candidate_availability", "projected_safe_candidate_count",
        "projected_conditional_selection_error_numerator",
        "projected_selected_stopping_reserve", "projected_selected_visible",
        "selected_normal_acceleration",
    )
    for name in neutral_names:
        result.setdefault(f"per_sample_{name}", zeros)
    result["per_sample_preventive_sample_weight"] = zeros + 1.0
    return result
