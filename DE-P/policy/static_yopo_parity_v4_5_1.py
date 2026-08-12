"""V4.5.1 relative-score and continuous kinodynamic objective.

V4.5 remains frozen.  This version reuses its original-YOPO trajectory
objective and bounded dangerous-segment term, then makes two narrow changes:

* the score head regresses a per-sample standardized relative candidate cost;
* speed/acceleration excess above the physical 6/6 contract contributes a
  smooth differentiable candidate cost instead of being telemetry-only.

Neither change introduces a boolean model-qualification Gate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import torch
from torch.nn import functional as F

from policy.static_yopo_parity_v4_4 import (
    _relative_order_kl,
    _sample_velocity_acceleration,
)
from policy.static_yopo_parity_v4_5 import (
    StaticYOPOParityConfigV45,
    static_yopo_parity_objective_v4_5,
)


@dataclass(frozen=True)
class StaticYOPOParityConfigV451:
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
            raise ValueError(f"unknown static_yopo_v4_5_1 keys: {unknown}")
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
            self.relative_order_weight, self.kinematic_speed_weight,
            self.kinematic_acceleration_weight, self.vehicle_radius_m,
            self.clear_distance_m, self.dangerous_segment_weight,
            self.large_vertical_displacement_m,
        )
        if min(nonnegative) < 0.0:
            raise ValueError("V4.5.1 weights/distances must be non-negative")
        positive = (
            self.relative_order_temperature, self.relative_label_min_scale,
            self.training_speed_mps, self.max_speed_mps,
            self.max_acceleration_mps2, self.kinematic_softness,
            self.dangerous_segment_focus,
        )
        if min(positive) <= 0.0:
            raise ValueError("V4.5.1 scales and limits must be positive")

    def as_v45(self):
        names = {field.name for field in fields(StaticYOPOParityConfigV45)}
        values = {
            name: value for name, value in asdict(self).items()
            if name in names
        }
        return StaticYOPOParityConfigV45(**values)

    def contract(self):
        return {
            "version": "static_yopo_original_parity_v4_5_1",
            "parent": "static_yopo_original_parity_v4_5",
            "candidate_objective": (
                "v4_5_plus_smooth_relative_speed_acceleration_excess"
            ),
            "score_label": "detached_within_sample_standardized_total_cost",
            "score_calibration": "relative_regression_plus_single_listwise_kl",
            "qualification_gate": "none",
            "dynamic_training": False,
            **asdict(self),
        }


def standardize_relative_cost_v4_5_1(cost, minimum_scale=1.0e-3):
    """Remove per-sample offset/scale while preserving candidate ordering."""
    if cost.ndim != 2:
        raise ValueError("candidate cost must be [B,N]")
    detached = cost.detach()
    mean = detached.mean(dim=1, keepdim=True)
    scale = detached.std(dim=1, unbiased=False, keepdim=True).clamp_min(
        float(minimum_scale)
    )
    return (detached - mean) / scale, scale.squeeze(1)


def smooth_kinematic_excess_cost_v4_5_1(
    maximum_speed, maximum_acceleration, *, max_speed, max_acceleration,
    speed_weight, acceleration_weight, softness,
):
    """Smooth candidate penalty for exceeding speed/acceleration limits.

    The dimensionless softplus hinge is effectively zero well below each
    limit, remains differentiable at the limit, and grows quadratically above
    it.  It changes the training gradient but never rejects a candidate.
    """
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
        float(speed_weight) * speed_excess.square()
        + float(acceleration_weight) * acceleration_excess.square()
    )


def static_yopo_parity_objective_v4_5_1(
    *, sampler, safety_loss, fixed, predicted, predicted_scores, goal_world,
    map_id, batch_size, candidate_count, route_goal_distance, config,
):
    config.validate()
    if predicted_scores.shape != (batch_size, candidate_count):
        raise ValueError("predicted score shape violates V4.5.1 contract")
    if candidate_count % 3:
        raise ValueError("V4.5.1 vertical diagnostics require three rows")

    # Preserve the frozen V4.5 objective and diagnostics as the base contract.
    result = static_yopo_parity_objective_v4_5(
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
        config=config.as_v45(),
    )

    _, velocity, acceleration = _sample_velocity_acceleration(
        sampler, fixed, predicted, config.derivative_samples
    )
    maximum_speed = velocity.norm(dim=2).amax(dim=1).reshape(
        batch_size, candidate_count
    )
    maximum_acceleration = acceleration.norm(dim=2).amax(dim=1).reshape(
        batch_size, candidate_count
    )
    kinematic_cost = smooth_kinematic_excess_cost_v4_5_1(
        maximum_speed,
        maximum_acceleration,
        max_speed=config.max_speed_mps,
        max_acceleration=config.max_acceleration_mps2,
        speed_weight=config.kinematic_speed_weight,
        acceleration_weight=config.kinematic_acceleration_weight,
        softness=config.kinematic_softness,
    )

    def candidates(name):
        return result[name].reshape(batch_size, candidate_count)

    candidate_total = (
        candidates("candidate_smooth_cost")
        + candidates("candidate_static_cost")
        + candidates("candidate_dangerous_segment_cost")
        + candidates("candidate_guidance_cost")
        + kinematic_cost
    )
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

    rows = torch.arange(batch_size, device=fixed.device)
    selected = predicted_scores.argmin(dim=1)
    oracle = candidate_total.detach().argmin(dim=1)
    oracle_regret = (
        candidate_total.detach()[rows, selected]
        - candidate_total.detach()[rows, oracle]
    )
    current_position = fixed[:, :, 0].reshape(
        batch_size, candidate_count, 3
    )[:, 0]
    endpoint = predicted[:, :, 0].reshape(batch_size, candidate_count, 3)
    vertical_displacement = endpoint[:, :, 2] - current_position[:, None, 2]
    columns_per_row = candidate_count // 3
    primitive_row = torch.div(
        torch.arange(candidate_count, device=fixed.device),
        columns_per_row,
        rounding_mode="floor",
    )
    selected_vertical = vertical_displacement[rows, selected]
    oracle_vertical = vertical_displacement[rows, oracle]

    result.update({
        "total_loss": per_sample_total.mean(),
        "trajectory_loss": per_sample_trajectory.mean(),
        "score_loss": per_sample_score.mean(),
        "ranking_loss": per_sample_ranking.mean(),
        "kinematic_loss": kinematic_cost.mean(),
        "score_label": relative_label.reshape(-1),
        "candidate_kinodynamic_cost": kinematic_cost.reshape(-1),
        "candidate_raw_total_cost": candidate_total.reshape(-1),
        "per_sample_total_loss": per_sample_total,
        "per_sample_trajectory_loss": per_sample_trajectory,
        "per_sample_score_loss": per_sample_score,
        "per_sample_ranking_loss": per_sample_ranking,
        "per_sample_kinematic_loss": kinematic_cost.mean(dim=1),
        "per_sample_score_label_scale": label_scale,
        "per_sample_score_oracle_regret": oracle_regret,
        "per_sample_projected_selection_regret": oracle_regret,
        "per_sample_oracle_vertical_displacement": oracle_vertical,
        "per_sample_selected_absolute_vertical_displacement": (
            selected_vertical.abs()
        ),
        "per_sample_oracle_absolute_vertical_displacement": (
            oracle_vertical.abs()
        ),
        "per_sample_selected_large_vertical_maneuver": selected_vertical.abs().ge(
            config.large_vertical_displacement_m
        ).to(fixed.dtype),
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
