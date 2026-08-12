"""Localized static-clearance objective for the V4.5.8 recovery shakedown.

V4.5.7 multiplied several overlapping safety terms and collapsed useful
candidate length.  This version returns to one score target and one static
safety term.  Clearance outside 0.50 m is nearly free, the warning band from
0.50 m to the 0.30 m vehicle radius rises exponentially, and penetration of
the physical radius uses a finite differentiable training surrogate.  Runtime
still treats penetration as infinite cost by rejecting it in the physical
collision check; literal infinities are deliberately excluded from backprop.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math

import torch
from torch.nn import functional as F

from policy.static_yopo_parity_v4_5_1 import (
    standardize_relative_cost_v4_5_1,
)
from policy.static_yopo_parity_v4_5_6 import (
    StaticYOPOParityConfigV456,
    static_yopo_parity_objective_v4_5_6,
)
from policy.static_yopo_parity_v4_4 import _relative_order_kl


@dataclass(frozen=True)
class StaticYOPOParityConfigV458:
    enabled: bool = False
    derivative_samples: int = 81
    static_safety_samples: int = 81
    jerk_unit_weight: float = 10.0
    acceleration_unit_weight: float = 1.0
    safety_weight: float = 1.0
    guidance_weight: float = 0.15
    guidance_perpendicular_weight: float = 0.50
    score_regression_weight: float = 1.0
    relative_order_weight: float = 1.0
    relative_order_temperature: float = 0.75
    relative_label_min_scale: float = 1.0e-3
    training_speed_mps: float = 6.0
    max_speed_mps: float = 6.0
    max_acceleration_mps2: float = 6.0
    kinematic_speed_weight: float = 0.10
    kinematic_acceleration_weight: float = 0.10
    kinematic_softness: float = 0.05
    vehicle_radius_m: float = 0.30
    clear_distance_m: float = 0.50
    dangerous_segment_weight: float = 0.0
    dangerous_segment_window: int = 14
    dangerous_segment_focus: float = 8.0
    large_vertical_displacement_m: float = 1.0
    clearance_score_weight: float = 0.0
    clearance_score_softness_m: float = 0.08
    clearance_pairwise_weight: float = 0.0
    clearance_pairwise_margin: float = 0.0
    far_clearance_cost: float = 0.001
    far_decay_m: float = 0.10
    radius_boundary_cost: float = 1.0
    collision_surrogate_cost: float = 3.0
    penetration_scale_m: float = 0.05
    maximum_training_cost: float = 6.0

    @classmethod
    def from_mapping(cls, value):
        value = {} if value is None else dict(value)
        unknown = sorted(set(value) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown static_yopo_v4_5_8 keys: {unknown}")
        result = cls(**value)
        result.validate()
        return result

    def validate(self):
        self.as_v456().validate()
        if not self.vehicle_radius_m < self.clear_distance_m:
            raise ValueError("clear distance must exceed vehicle radius")
        positive = (
            self.far_clearance_cost,
            self.far_decay_m,
            self.radius_boundary_cost,
            self.collision_surrogate_cost,
            self.penetration_scale_m,
            self.maximum_training_cost,
        )
        if min(positive) <= 0.0:
            raise ValueError("localized safety costs/scales must be positive")
        if self.radius_boundary_cost <= self.far_clearance_cost:
            raise ValueError("warning band must rise toward the vehicle radius")
        if self.collision_surrogate_cost <= self.radius_boundary_cost:
            raise ValueError("collision surrogate must exceed warning-band cost")
        if self.maximum_training_cost < self.collision_surrogate_cost:
            raise ValueError("maximum cost must cover the collision surrogate")

    def as_v456(self):
        names = {field.name for field in fields(StaticYOPOParityConfigV456)}
        return StaticYOPOParityConfigV456(**{
            name: value for name, value in asdict(self).items()
            if name in names
        })

    def contract(self):
        return {
            "version": "static_yopo_original_parity_v4_5_8",
            "parent": "static_yopo_original_parity_v4_5_6",
            "candidate_objective": (
                "original_yopo_smooth_guidance_plus_one_localized_static_"
                "clearance_cost_plus_weak_kinematic_preference"
            ),
            "score_target": "detached_same_single_candidate_total_cost",
            "static_clearance_semantics": {
                "clear": "near_zero_above_clear_distance",
                "warning": "exponential_between_clear_distance_and_radius",
                "collision_training": "finite_differentiable_surrogate",
                "collision_runtime": "positive_infinity_reject",
            },
            "extra_clearance_score_barrier": False,
            "extra_clearance_pairwise_ranking": False,
            "qualification_gate": "none",
            "dynamic_training": False,
            **asdict(self),
        }


def localized_static_clearance_cost_v4_5_8(
    clearance,
    *,
    vehicle_radius,
    clear_distance,
    far_cost,
    far_decay,
    radius_boundary_cost,
    collision_surrogate_cost,
    penetration_scale,
    maximum_training_cost,
):
    """Return finite training cost and the deployment-infinite collision mask.

    The two finite branches are continuous at ``clear_distance``.  The jump at
    ``vehicle_radius`` encodes the physical body boundary.  Values at and below
    that radius remain finite only so gradient descent and AMP cannot be
    poisoned by literal ``inf``; runtime safety rejects the same mask.
    """
    if not torch.is_floating_point(clearance):
        raise TypeError("clearance must be floating point")
    radius = float(vehicle_radius)
    clear = float(clear_distance)
    far_value = float(far_cost)
    far_scale = float(far_decay)
    radius_value = float(radius_boundary_cost)
    collision_value = float(collision_surrogate_cost)
    penetration = float(penetration_scale)
    maximum = float(maximum_training_cost)
    if not radius < clear:
        raise ValueError("vehicle radius must be below clear distance")

    far_exponent = ((clear - clearance) / far_scale).clamp(max=0.0)
    far_branch = far_value * torch.exp(far_exponent)

    warning_fraction = ((clear - clearance) / (clear - radius)).clamp(0.0, 1.0)
    warning_log_ratio = math.log(radius_value / far_value)
    warning_branch = far_value * torch.exp(warning_log_ratio * warning_fraction)

    penetration_depth = (radius - clearance).clamp_min(0.0)
    # Bounded saturation keeps the training scale controlled while retaining
    # a finite outward gradient even for a deeply penetrating proposal.
    collision_branch = collision_value + (maximum - collision_value) * (
        1.0 - torch.exp(-penetration_depth / penetration)
    )
    collision_mask = clearance <= radius
    cost = torch.where(
        clearance >= clear,
        far_branch,
        torch.where(collision_mask, collision_branch, warning_branch),
    )
    return cost, collision_mask


def static_yopo_parity_objective_v4_5_8(
    *, sampler, safety_loss, fixed, predicted, predicted_scores, goal_world,
    map_id, batch_size, candidate_count, route_goal_distance, config,
):
    config.validate()
    # Reuse the dense, analytically checked V4.5.6 implementation for geometry,
    # derivatives and diagnostics.  Its old static/danger terms are removed
    # exactly below, and both overlapping clearance-score additions are zero.
    result = static_yopo_parity_objective_v4_5_6(
        sampler=sampler,
        safety_loss=safety_loss,
        fixed=fixed,
        predicted=predicted,
        predicted_scores=predicted_scores,
        goal_world=goal_world,
        map_id=map_id,
        batch_size=batch_size,
        candidate_count=candidate_count,
        route_goal_distance=route_goal_distance,
        config=config.as_v456(),
    )

    def candidates(name):
        return result[name].reshape(batch_size, candidate_count)

    minimum_clearance = candidates("candidate_minimum_clearance")
    localized_raw, collision_mask = localized_static_clearance_cost_v4_5_8(
        minimum_clearance,
        vehicle_radius=config.vehicle_radius_m,
        clear_distance=config.clear_distance_m,
        far_cost=config.far_clearance_cost,
        far_decay=config.far_decay_m,
        radius_boundary_cost=config.radius_boundary_cost,
        collision_surrogate_cost=config.collision_surrogate_cost,
        penetration_scale=config.penetration_scale_m,
        maximum_training_cost=config.maximum_training_cost,
    )
    localized_static = config.safety_weight * localized_raw
    old_total = candidates("candidate_raw_total_cost")
    old_static = candidates("candidate_static_cost")
    old_danger = candidates("candidate_dangerous_segment_cost")
    candidate_total = old_total - old_static - old_danger + localized_static

    relative_label, label_scale = standardize_relative_cost_v4_5_1(
        candidate_total, config.relative_label_min_scale
    )
    score_regression = F.smooth_l1_loss(
        predicted_scores, relative_label, reduction="none"
    ).mean(dim=1)
    relative_order = _relative_order_kl(
        predicted_scores, candidate_total.detach(),
        config.relative_order_temperature,
    )
    per_sample_trajectory = candidate_total.mean(dim=1)
    per_sample_score = config.score_regression_weight * score_regression
    per_sample_ranking = config.relative_order_weight * relative_order
    per_sample_total = (
        per_sample_trajectory + per_sample_score + per_sample_ranking
    )

    rows = torch.arange(batch_size, device=fixed.device)
    selected = predicted_scores.argmin(dim=1)
    oracle = candidate_total.detach().argmin(dim=1)
    collision_free = ~collision_mask
    collision_available = collision_free.any(dim=1)
    selected_collision = collision_mask[rows, selected]
    maximum_speed = candidates("candidate_maximum_speed")
    maximum_acceleration = candidates("candidate_maximum_acceleration")
    hardware_feasible = (
        (maximum_speed <= config.max_speed_mps)
        & (maximum_acceleration <= config.max_acceleration_mps2)
    )
    physical_feasible = collision_free & hardware_feasible
    physical_available = physical_feasible.any(dim=1)
    oracle_regret = (
        candidate_total.detach()[rows, selected]
        - candidate_total.detach()[rows, oracle]
    )
    zero_candidates = localized_static * 0.0
    zeros = per_sample_total * 0.0

    current_position = fixed[:, :, 0].reshape(
        batch_size, candidate_count, 3
    )[:, 0]
    endpoint = predicted[:, :, 0].reshape(batch_size, candidate_count, 3)
    vertical_displacement = endpoint[:, :, 2] - current_position[:, None, 2]
    oracle_vertical = vertical_displacement[rows, oracle]
    columns_per_row = candidate_count // 3
    primitive_row = torch.div(
        torch.arange(candidate_count, device=fixed.device), columns_per_row,
        rounding_mode="floor",
    )

    result.update({
        "total_loss": per_sample_total.mean(),
        "trajectory_loss": per_sample_trajectory.mean(),
        "score_loss": per_sample_score.mean(),
        "ranking_loss": per_sample_ranking.mean(),
        "static_safety_loss": localized_static.mean(),
        "dangerous_segment_loss": zeros.mean(),
        "clearance_barrier_loss": zeros.mean(),
        "clearance_pairwise_ranking_loss": zeros.mean(),
        "score_label": relative_label.reshape(-1),
        "candidate_static_cost": localized_static.reshape(-1),
        "candidate_dangerous_segment_cost": zero_candidates.reshape(-1),
        "candidate_clearance_barrier_cost": zero_candidates.reshape(-1),
        "candidate_proposal_total_cost": candidate_total.reshape(-1),
        "candidate_score_target_cost": candidate_total.reshape(-1),
        "candidate_raw_total_cost": candidate_total.reshape(-1),
        "candidate_projected_safe_mask": physical_feasible,
        "candidate_runtime_infinite_collision_mask": collision_mask,
        "per_sample_total_loss": per_sample_total,
        "per_sample_trajectory_loss": per_sample_trajectory,
        "per_sample_score_loss": per_sample_score,
        "per_sample_ranking_loss": per_sample_ranking,
        "per_sample_static_safety_loss": localized_static.mean(dim=1),
        "per_sample_dangerous_segment_loss": zeros,
        "per_sample_clearance_barrier_loss": zeros,
        "per_sample_clearance_pairwise_ranking_loss": zeros,
        "per_sample_score_label_scale": label_scale,
        "per_sample_score_oracle_regret": oracle_regret,
        "per_sample_projected_selection_regret": oracle_regret,
        "per_sample_score_top1_match": selected.eq(oracle),
        "per_sample_unsafe_selection": selected_collision.to(fixed.dtype),
        "per_sample_collision_free_candidate_count": collision_free.sum(
            dim=1
        ).to(fixed.dtype),
        "per_sample_collision_free_candidate_available": collision_available.to(
            fixed.dtype
        ),
        "per_sample_conditional_collision_selection_error": (
            collision_available & selected_collision
        ).to(fixed.dtype),
        "per_sample_physical_feasible_candidate_available": physical_available.to(
            fixed.dtype
        ),
        "per_sample_conditional_physical_selection_error": (
            physical_available
            & (selected_collision | ~hardware_feasible[rows, selected])
        ).to(fixed.dtype),
        "per_sample_oracle_collision_unsafe": collision_mask[
            rows, oracle
        ].to(fixed.dtype),
        "per_sample_oracle_physical_unsafe": (~physical_feasible[
            rows, oracle
        ]).to(fixed.dtype),
        "per_sample_oracle_vertical_displacement": oracle_vertical,
        "per_sample_oracle_absolute_vertical_displacement": oracle_vertical.abs(),
        "per_sample_oracle_large_vertical_maneuver": oracle_vertical.abs().ge(
            config.large_vertical_displacement_m
        ).to(fixed.dtype),
        "per_sample_oracle_primitive_row": primitive_row[oracle].to(fixed.dtype),
        "per_sample_oracle_vertical_primitive": primitive_row[oracle].ne(1).to(
            fixed.dtype
        ),
        "per_sample_oracle_upward_primitive": primitive_row[oracle].eq(0).to(
            fixed.dtype
        ),
        "per_sample_oracle_level_primitive": primitive_row[oracle].eq(1).to(
            fixed.dtype
        ),
        "per_sample_oracle_downward_primitive": primitive_row[oracle].eq(2).to(
            fixed.dtype
        ),
    })
    return result
