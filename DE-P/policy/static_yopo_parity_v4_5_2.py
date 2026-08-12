"""V4.5.2 score calibration and decomposed feasibility diagnostics.

This version keeps the V4.5.1 relative score contract, changes only the
near-limit kinodynamic penalty from a squared to a linear softplus hinge, and
raises the existing single listwise ordering weight.  It adds observability,
not qualification Gates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import torch
from torch.nn import functional as F

from policy.static_yopo_parity_v4_4 import _relative_order_kl
from policy.static_yopo_parity_v4_5_1 import (
    StaticYOPOParityConfigV451,
    standardize_relative_cost_v4_5_1,
    static_yopo_parity_objective_v4_5_1,
)


@dataclass(frozen=True)
class StaticYOPOParityConfigV452:
    enabled: bool = False
    derivative_samples: int = 81
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
    kinematic_speed_weight: float = 1.0
    kinematic_acceleration_weight: float = 1.0
    kinematic_softness: float = 0.05
    vehicle_radius_m: float = 0.30
    clear_distance_m: float = 1.20
    dangerous_segment_weight: float = 0.15
    dangerous_segment_window: int = 5
    dangerous_segment_focus: float = 8.0
    large_vertical_displacement_m: float = 1.0

    @classmethod
    def from_mapping(cls, value):
        value = {} if value is None else dict(value)
        unknown = sorted(set(value) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown static_yopo_v4_5_2 keys: {unknown}")
        result = cls(**value)
        result.validate()
        return result

    def validate(self):
        legacy = self.as_v451()
        legacy.validate()
        if self.relative_order_weight <= 0.0:
            raise ValueError("V4.5.2 must retain one positive ranking objective")

    def as_v451(self):
        names = {field.name for field in fields(StaticYOPOParityConfigV451)}
        values = {
            name: value for name, value in asdict(self).items()
            if name in names
        }
        return StaticYOPOParityConfigV451(**values)

    def contract(self):
        return {
            "version": "static_yopo_original_parity_v4_5_2",
            "parent": "static_yopo_original_parity_v4_5_1",
            "candidate_objective": (
                "v4_5_danger_plus_linear_softplus_kinematic_excess"
            ),
            "score_label": "detached_within_sample_standardized_total_cost",
            "score_calibration": "relative_regression_plus_one_listwise_kl",
            "qualification_gate": "none",
            "diagnostics": "geometry_speed_acceleration_oracle_decomposition",
            "dynamic_training": False,
            **asdict(self),
        }


def linear_kinematic_excess_cost_v4_5_2(
    maximum_speed, maximum_acceleration, *, max_speed, max_acceleration,
    speed_weight, acceleration_weight, softness,
):
    """Smooth linear hinge with useful gradient at the physical limit."""
    if maximum_speed.shape != maximum_acceleration.shape:
        raise ValueError("speed/acceleration candidate shapes must match")
    speed_ratio = maximum_speed / float(max_speed) - 1.0
    acceleration_ratio = maximum_acceleration / float(max_acceleration) - 1.0
    speed_excess = float(softness) * F.softplus(
        speed_ratio / float(softness)
    )
    acceleration_excess = float(softness) * F.softplus(
        acceleration_ratio / float(softness)
    )
    return (
        float(speed_weight) * speed_excess
        + float(acceleration_weight) * acceleration_excess
    )


def static_yopo_parity_objective_v4_5_2(
    *, sampler, safety_loss, fixed, predicted, predicted_scores, goal_world,
    map_id, batch_size, candidate_count, route_goal_distance, config,
):
    config.validate()
    result = static_yopo_parity_objective_v4_5_1(
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
        config=config.as_v451(),
    )

    def candidates(name):
        return result[name].reshape(batch_size, candidate_count)

    maximum_speed = candidates("candidate_maximum_speed")
    maximum_acceleration = candidates("candidate_maximum_acceleration")
    minimum_clearance = candidates("candidate_minimum_clearance")
    old_kinematic = candidates("candidate_kinodynamic_cost")
    base_total = candidates("candidate_raw_total_cost") - old_kinematic
    kinematic_cost = linear_kinematic_excess_cost_v4_5_2(
        maximum_speed,
        maximum_acceleration,
        max_speed=config.max_speed_mps,
        max_acceleration=config.max_acceleration_mps2,
        speed_weight=config.kinematic_speed_weight,
        acceleration_weight=config.kinematic_acceleration_weight,
        softness=config.kinematic_softness,
    )
    candidate_total = base_total + kinematic_cost
    relative_label, label_scale = standardize_relative_cost_v4_5_1(
        candidate_total, config.relative_label_min_scale
    )
    score_regression = F.smooth_l1_loss(
        predicted_scores, relative_label, reduction="none"
    ).mean(dim=1)
    relative_order = _relative_order_kl(
        predicted_scores,
        candidate_total.detach(),
        config.relative_order_temperature,
    )
    per_sample_trajectory = candidate_total.mean(dim=1)
    per_sample_score = config.score_regression_weight * score_regression
    per_sample_ranking = config.relative_order_weight * relative_order
    per_sample_total = (
        per_sample_trajectory + per_sample_score + per_sample_ranking
    )

    speed_unsafe = maximum_speed > config.max_speed_mps
    acceleration_unsafe = maximum_acceleration > config.max_acceleration_mps2
    hardware_feasible = ~(speed_unsafe | acceleration_unsafe)
    collision_free = minimum_clearance >= config.vehicle_radius_m
    physical_feasible = collision_free & hardware_feasible
    rows = torch.arange(batch_size, device=fixed.device)
    selected = predicted_scores.argmin(dim=1)
    oracle = candidate_total.detach().argmin(dim=1)
    selected_collision_unsafe = ~collision_free[rows, selected]
    selected_hardware_unsafe = ~hardware_feasible[rows, selected]
    collision_available = collision_free.any(dim=1)
    physical_available = physical_feasible.any(dim=1)
    oracle_regret = (
        candidate_total.detach()[rows, selected]
        - candidate_total.detach()[rows, oracle]
    )

    current_position = fixed[:, :, 0].reshape(
        batch_size, candidate_count, 3
    )[:, 0]
    endpoint = predicted[:, :, 0].reshape(batch_size, candidate_count, 3)
    vertical_displacement = endpoint[:, :, 2] - current_position[:, None, 2]
    oracle_vertical = vertical_displacement[rows, oracle]
    columns_per_row = candidate_count // 3
    primitive_row = torch.div(
        torch.arange(candidate_count, device=fixed.device),
        columns_per_row,
        rounding_mode="floor",
    )

    result.update({
        "total_loss": per_sample_total.mean(),
        "trajectory_loss": per_sample_trajectory.mean(),
        "score_loss": per_sample_score.mean(),
        "ranking_loss": per_sample_ranking.mean(),
        "kinematic_loss": kinematic_cost.mean(),
        "score_label": relative_label.reshape(-1),
        "candidate_kinodynamic_cost": kinematic_cost.reshape(-1),
        "candidate_raw_total_cost": candidate_total.reshape(-1),
        "candidate_projected_safe_mask": physical_feasible,
        "per_sample_total_loss": per_sample_total,
        "per_sample_trajectory_loss": per_sample_trajectory,
        "per_sample_score_loss": per_sample_score,
        "per_sample_ranking_loss": per_sample_ranking,
        "per_sample_kinematic_loss": kinematic_cost.mean(dim=1),
        "per_sample_score_label_scale": label_scale,
        "per_sample_score_oracle_regret": oracle_regret,
        "per_sample_projected_selection_regret": oracle_regret,
        "per_sample_unsafe_selection": selected_collision_unsafe.to(fixed.dtype),
        "per_sample_hardware_unsafe_selection": selected_hardware_unsafe.to(
            fixed.dtype
        ),
        "per_sample_speed_unsafe_selection": speed_unsafe[rows, selected].to(
            fixed.dtype
        ),
        "per_sample_acceleration_unsafe_selection": acceleration_unsafe[
            rows, selected
        ].to(fixed.dtype),
        "per_sample_collision_free_candidate_count": collision_free.sum(
            dim=1
        ).to(fixed.dtype),
        "per_sample_hardware_feasible_candidate_count": hardware_feasible.sum(
            dim=1
        ).to(fixed.dtype),
        "per_sample_feasible_candidate_count": physical_feasible.sum(dim=1).to(
            fixed.dtype
        ),
        "per_sample_collision_free_candidate_available": collision_available.to(
            fixed.dtype
        ),
        "per_sample_physical_feasible_candidate_available": physical_available.to(
            fixed.dtype
        ),
        "per_sample_conditional_collision_selection_error": (
            collision_available & selected_collision_unsafe
        ).to(fixed.dtype),
        "per_sample_conditional_physical_selection_error": (
            physical_available
            & (selected_collision_unsafe | selected_hardware_unsafe)
        ).to(fixed.dtype),
        "per_sample_oracle_collision_unsafe": (~collision_free[rows, oracle]).to(
            fixed.dtype
        ),
        "per_sample_oracle_speed_unsafe": speed_unsafe[rows, oracle].to(
            fixed.dtype
        ),
        "per_sample_oracle_acceleration_unsafe": acceleration_unsafe[
            rows, oracle
        ].to(fixed.dtype),
        "per_sample_oracle_hardware_unsafe": (~hardware_feasible[rows, oracle]).to(
            fixed.dtype
        ),
        "per_sample_oracle_physical_unsafe": (~physical_feasible[rows, oracle]).to(
            fixed.dtype
        ),
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
