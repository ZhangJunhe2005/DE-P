"""V4.5 static YOPO objective with one bounded continuous danger term.

The network, 15 primitive lattice and original YOPO smooth/safety/guidance
objective stay unchanged.  V4.5 only prevents a short near-collision segment
from being diluted by the full-trajectory safety mean.  The added term is a
smooth log-mean-exp over local trajectory windows and is strictly bounded by
``dangerous_segment_weight``; it is neither a feasibility Gate nor a second
ranking policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch.nn import functional as F

from policy.static_yopo_parity_v4_4 import (
    _relative_order_kl,
    _sample_velocity_acceleration,
    exact_derivative_integral,
)


@dataclass(frozen=True)
class StaticYOPOParityConfigV45:
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
    dangerous_segment_weight: float = 0.15
    dangerous_segment_window: int = 5
    dangerous_segment_focus: float = 8.0

    @classmethod
    def from_mapping(cls, value):
        value = {} if value is None else dict(value)
        unknown = sorted(set(value) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown static_yopo_v4_5 keys: {unknown}")
        result = cls(**value)
        result.validate()
        return result

    def validate(self):
        if self.derivative_samples < 3:
            raise ValueError("derivative_samples must be at least three")
        if self.dangerous_segment_window < 1:
            raise ValueError("dangerous_segment_window must be positive")
        nonnegative = (
            self.jerk_unit_weight, self.acceleration_unit_weight,
            self.safety_weight, self.guidance_weight,
            self.guidance_perpendicular_weight, self.score_regression_weight,
            self.relative_order_weight, self.vehicle_radius_m,
            self.clear_distance_m, self.dangerous_segment_weight,
        )
        if min(nonnegative) < 0.0:
            raise ValueError("V4.5 weights/distances must be non-negative")
        positive = (
            self.relative_order_temperature, self.training_speed_mps,
            self.max_speed_mps, self.max_acceleration_mps2,
            self.dangerous_segment_focus,
        )
        if min(positive) <= 0.0:
            raise ValueError("V4.5 scales and limits must be positive")

    def contract(self):
        return {
            "version": "static_yopo_original_parity_v4_5",
            "network": "unchanged_dep_mobilenetv3_15_primitive_unified_head",
            "candidate_objective": (
                "original_yopo_analytic_total_cost_plus_bounded_smooth_"
                "dangerous_segment"
            ),
            "score_label": "detached_same_candidate_total_cost",
            "score_calibration": "single_within_sample_listwise_kl",
            "qualification_gate": "none",
            "dynamic_training": False,
            "dangerous_segment_bound": self.dangerous_segment_weight,
            **asdict(self),
        }


def bounded_dangerous_segment_cost_v4_5(
    static_samples, *, weight, window, focus,
):
    """Return a smooth candidate cost in ``[0, weight]``.

    ``static_samples`` is the existing non-negative ESDF cost [B,N,T].  The
    rational compression bounds each point hazard, local average pooling makes
    the term respond to a segment rather than one noisy point, and normalized
    log-sum-exp supplies a smooth maximum whose gradient is a probability
    distribution instead of an unbounded spike.
    """
    if static_samples.ndim != 3:
        raise ValueError("static_samples must be [B,N,T]")
    if window < 1 or window > static_samples.shape[2]:
        raise ValueError("dangerous segment window exceeds trajectory samples")
    hazard = static_samples.clamp_min(0.0)
    hazard = hazard / (1.0 + hazard)
    flat = hazard.reshape(-1, 1, hazard.shape[2])
    segment = F.avg_pool1d(flat, kernel_size=int(window), stride=1)
    segment = segment.reshape(*hazard.shape[:2], -1)
    count = segment.shape[2]
    smooth_max = (
        torch.logsumexp(float(focus) * segment, dim=2)
        - math.log(count)
    ) / float(focus)
    # The mathematical result is already within [0,1].  Clamp only protects
    # the public numerical contract from a few ulps of round-off.
    return float(weight) * smooth_max.clamp(0.0, 1.0)


def static_yopo_parity_objective_v4_5(
    *, sampler, safety_loss, fixed, predicted, predicted_scores, goal_world,
    map_id, batch_size, candidate_count, route_goal_distance, config,
):
    config.validate()
    if fixed.shape != predicted.shape or fixed.shape[1:] != (3, 3):
        raise ValueError("fixed/predicted derivatives must be [B*N,3,3]")
    if predicted_scores.shape != (batch_size, candidate_count):
        raise ValueError("predicted score shape violates V4.5 contract")
    if candidate_count % 3:
        raise ValueError("V4.5 vertical diagnostics require three lattice rows")

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
    static_cost = config.safety_weight * static_samples.mean(dim=2)
    dangerous_segment_cost = bounded_dangerous_segment_cost_v4_5(
        static_samples,
        weight=config.dangerous_segment_weight,
        window=config.dangerous_segment_window,
        focus=config.dangerous_segment_focus,
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
        trajectory_delta
        - goal_progress[:, :, None] * goal_direction[:, None, :]
    ).norm(dim=2)
    guidance_raw = F.smooth_l1_loss(
        goal_progress, goal_length[:, None].expand_as(goal_progress),
        reduction="none",
    ) + config.guidance_perpendicular_weight * perpendicular
    guidance_cost = config.guidance_weight * guidance_raw

    candidate_total = (
        smooth_cost + static_cost + dangerous_segment_cost + guidance_cost
    )
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
    sample = torch.arange(batch_size, device=fixed.device)
    selected = predicted_scores.argmin(dim=1)
    oracle = label.argmin(dim=1)
    oracle_regret = label[sample, selected] - label[sample, oracle]
    selected_hardware_unsafe = ~hardware_feasible[sample, selected]
    selected_collision_unsafe = ~collision_free[sample, selected]
    zeros = per_sample_total * 0.0
    zero_candidates = candidate_total * 0.0

    columns_per_row = candidate_count // 3
    primitive_row = torch.div(
        torch.arange(candidate_count, device=fixed.device), columns_per_row,
        rounding_mode="floor",
    )
    selected_row = primitive_row[selected]
    oracle_row = primitive_row[oracle]

    result = {
        "total_loss": per_sample_total.mean(),
        "trajectory_loss": per_sample_trajectory.mean(),
        "score_loss": per_sample_score.mean(),
        "smoothness_loss": smooth_cost.mean(),
        "static_safety_loss": static_cost.mean(),
        "dangerous_segment_loss": dangerous_segment_cost.mean(),
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
        "candidate_dangerous_segment_cost": dangerous_segment_cost.reshape(-1),
        "candidate_guidance_cost": guidance_cost.reshape(-1),
        "candidate_kinodynamic_cost": zero_candidates.reshape(-1),
        # Diagnostic tensors used by later versioned objectives.  Exposing
        # already-computed values does not change the frozen V4.5 loss.
        "candidate_minimum_clearance": minimum_clearance.reshape(-1),
        # Preserve the already-computed dense ESDF samples for later
        # versioned objectives.  This is diagnostic data only for V4.5 and
        # avoids a second ESDF query when a descendant changes only the
        # temporal aggregation of the same safety samples.
        "candidate_static_distance_samples": static_distance,
        "candidate_maximum_speed": maximum_speed.reshape(-1),
        "candidate_maximum_acceleration": maximum_acceleration.reshape(-1),
        "candidate_preventive_cost": zero_candidates,
        "candidate_progress_cost": zero_candidates,
        "candidate_projected_safe_mask": physical_feasible,
        "candidate_stopping_distance_cost": zero_candidates.reshape(-1),
        "per_sample_total_loss": per_sample_total,
        "per_sample_trajectory_loss": per_sample_trajectory,
        "per_sample_score_loss": per_sample_score,
        "per_sample_ranking_loss": per_sample_ranking,
        "per_sample_dangerous_segment_loss": dangerous_segment_cost.mean(dim=1),
        "per_sample_kinematic_loss": zeros,
        "per_sample_smoothness_loss": smooth_cost.mean(dim=1),
        "per_sample_static_safety_loss": static_cost.mean(dim=1),
        "per_sample_guidance_loss": guidance_cost.mean(dim=1),
        "per_sample_selected_endpoint_distance": endpoint_distance[sample, selected],
        "per_sample_selected_endpoint_speed": endpoint_speed[sample, selected],
        "per_sample_hover_selection": endpoint_distance[sample, selected].lt(0.75).float(),
        "per_sample_route_goal_distance": route_goal_distance,
        "per_sample_objective_goal_distance": goal_length,
        "per_sample_score_top1_match": selected.eq(oracle),
        "per_sample_score_oracle_regret": oracle_regret,
        "per_sample_unsafe_selection": selected_collision_unsafe.float(),
        "per_sample_hardware_unsafe_selection": selected_hardware_unsafe.float(),
        "per_sample_selected_trajectory_max_speed": maximum_speed[sample, selected],
        "per_sample_selected_trajectory_max_acceleration": maximum_acceleration[sample, selected],
        "per_sample_feasible_candidate_count": physical_feasible.sum(dim=1).to(fixed.dtype),
        "per_sample_candidate_time_dilation_mean": time_dilation.mean(dim=1),
        "per_sample_selected_time_dilation": time_dilation[sample, selected],
        "per_sample_selected_normal_acceleration": zeros,
        "per_sample_selected_clearance": minimum_clearance[sample, selected],
        "per_sample_clear_candidate_count": minimum_clearance.ge(
            config.clear_distance_m
        ).sum(dim=1).to(fixed.dtype),
        "per_sample_selected_goal_progress": goal_progress[sample, selected],
        "per_sample_selected_goal_alignment": (
            goal_progress[sample, selected]
            / endpoint_distance[sample, selected].clamp_min(1.0e-8)
        ),
        "per_sample_reverse_selection": goal_progress[sample, selected].lt(0.0).float(),
        "per_sample_selected_vertical_displacement": trajectory_delta[
            sample, selected, 2
        ],
        "per_sample_oracle_vertical_displacement": trajectory_delta[
            sample, oracle, 2
        ],
        "per_sample_selected_primitive_row": selected_row.to(fixed.dtype),
        "per_sample_oracle_primitive_row": oracle_row.to(fixed.dtype),
        "per_sample_selected_vertical_primitive": selected_row.ne(1).to(fixed.dtype),
        "per_sample_oracle_vertical_primitive": oracle_row.ne(1).to(fixed.dtype),
        "per_sample_selected_upward_primitive": selected_row.eq(0).to(fixed.dtype),
        "per_sample_selected_level_primitive": selected_row.eq(1).to(fixed.dtype),
        "per_sample_selected_downward_primitive": selected_row.eq(2).to(fixed.dtype),
        "per_sample_oracle_upward_primitive": oracle_row.eq(0).to(fixed.dtype),
        "per_sample_oracle_level_primitive": oracle_row.eq(1).to(fixed.dtype),
        "per_sample_oracle_downward_primitive": oracle_row.eq(2).to(fixed.dtype),
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
