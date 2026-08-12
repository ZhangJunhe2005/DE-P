"""V4.5.5 dense static-ESDF supervision for useful YOPO proposals.

V4.5.4 could only reorder the frozen V4.5.2 candidates.  V4.5.5 returns to
the unified head and lets the proposal generator learn again, while retaining
the single original-YOPO-style continuous objective.  The only semantic
change from V4.5.2 is that static clearance and the already bounded dangerous
segment term are evaluated on a denser trajectory grid.

There is deliberately no Boolean candidate Gate and no minimum-clearance
barrier.  Unsafe candidates remain differentiable, and useful trajectory
length is still negotiated by the original smooth/safety/guidance total.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import torch
from policy.static_yopo_parity_v4_5_2 import (
    StaticYOPOParityConfigV452,
    static_yopo_parity_objective_v4_5_2,
)


@dataclass(frozen=True)
class StaticYOPOParityConfigV455:
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
    kinematic_speed_weight: float = 1.0
    kinematic_acceleration_weight: float = 1.0
    kinematic_softness: float = 0.05
    vehicle_radius_m: float = 0.30
    clear_distance_m: float = 1.20
    dangerous_segment_weight: float = 0.15
    # 14/81 ~= 5/30: preserve the V4.5 physical time window.
    dangerous_segment_window: int = 14
    dangerous_segment_focus: float = 8.0
    large_vertical_displacement_m: float = 1.0

    @classmethod
    def from_mapping(cls, value):
        value = {} if value is None else dict(value)
        unknown = sorted(set(value) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown static_yopo_v4_5_5 keys: {unknown}")
        result = cls(**value)
        result.validate()
        return result

    def validate(self):
        self.as_v452().validate()
        if self.static_safety_samples < 3:
            raise ValueError("static_safety_samples must be at least three")
        if not 1 <= self.dangerous_segment_window <= self.static_safety_samples:
            raise ValueError(
                "dangerous_segment_window must fit the dense safety grid"
            )

    def as_v452(self):
        names = {field.name for field in fields(StaticYOPOParityConfigV452)}
        return StaticYOPOParityConfigV452(**{
            name: value for name, value in asdict(self).items()
            if name in names
        })

    def contract(self):
        return {
            "version": "static_yopo_original_parity_v4_5_5",
            "parent": "static_yopo_original_parity_v4_5_2",
            "network": "mobilenetv3_unified_candidate_and_score_head",
            "candidate_objective": (
                "v4_5_2_total_with_dense_continuous_static_esdf"
            ),
            "score_label": "detached_standardized_same_dense_total_cost",
            "qualification_gate": "none",
            "dynamic_training": False,
            **asdict(self),
        }


def sample_quintic_positions_dense_v4_5_5(
    sampler, fixed, predicted, sample_count,
):
    """Sample the existing quintic without constructing a module per batch."""
    coefficient = sampler.coefficients(fixed, predicted).reshape(-1, 3, 6)
    duration = float(sampler.duration)
    times = torch.linspace(
        duration / int(sample_count), duration, int(sample_count),
        device=fixed.device, dtype=fixed.dtype,
    )
    powers = torch.stack([
        torch.ones_like(times), times, times ** 2, times ** 3,
        times ** 4, times ** 5,
    ])
    return torch.einsum("kai,it->kta", coefficient, powers)


class _DenseSamplerViewV455:
    """Present the frozen quintic sampler on a denser position grid.

    Derivative maxima continue to use the sampler's existing analytic/dense
    derivative path.  Only the static ESDF trajectory samples are replaced.
    This lets the complete V4.5.2 objective run once instead of querying the
    ESDF on both the legacy 30-point and the new 81-point grids every batch.
    """

    def __init__(self, sampler, sample_count):
        self.sampler = sampler
        self.duration = sampler.duration
        self.eval_points = int(sample_count)

    def coefficients(self, fixed, predicted):
        return self.sampler.coefficients(fixed, predicted)

    def __call__(self, fixed, predicted):
        position = sample_quintic_positions_dense_v4_5_5(
            self.sampler, fixed, predicted, self.eval_points
        )
        times = torch.linspace(
            float(self.duration) / self.eval_points,
            float(self.duration),
            self.eval_points,
            device=fixed.device,
            dtype=fixed.dtype,
        )
        return position, times


def static_yopo_parity_objective_v4_5_5(
    *, sampler, safety_loss, fixed, predicted, predicted_scores, goal_world,
    map_id, batch_size, candidate_count, route_goal_distance, config,
):
    config.validate()
    dense_sampler = _DenseSamplerViewV455(
        sampler, config.static_safety_samples
    )
    result = static_yopo_parity_objective_v4_5_2(
        sampler=dense_sampler,
        safety_loss=safety_loss,
        fixed=fixed,
        predicted=predicted,
        predicted_scores=predicted_scores,
        goal_world=goal_world,
        map_id=map_id,
        batch_size=batch_size,
        candidate_count=candidate_count,
        route_goal_distance=route_goal_distance,
        config=config.as_v452(),
    )

    dense_minimum_clearance = result["candidate_minimum_clearance"].reshape(
        batch_size, candidate_count
    )
    # The 30-point comparison is diagnostic-only.  Validation runs under
    # no_grad/inference_mode, so training never pays for a second ESDF query.
    if torch.is_grad_enabled():
        coarse_minimum_clearance = dense_minimum_clearance.detach()
        false_safe_count = torch.zeros(
            batch_size, device=fixed.device, dtype=fixed.dtype
        )
        dense_clearance_drop = torch.zeros_like(false_safe_count)
    else:
        coarse_position, _ = sampler(fixed, predicted)
        _, coarse_distance = safety_loss.get_distance_cost(
            coarse_position.reshape(batch_size, -1, 3), map_id
        )
        coarse_minimum_clearance = coarse_distance.reshape(
            batch_size, candidate_count, -1
        ).amin(dim=2)
        dense_collision_free = (
            dense_minimum_clearance >= config.vehicle_radius_m
        )
        coarse_collision_free = (
            coarse_minimum_clearance >= config.vehicle_radius_m
        )
        false_safe_count = (
            coarse_collision_free & ~dense_collision_free
        ).sum(dim=1).to(fixed.dtype)
        dense_clearance_drop = (
            coarse_minimum_clearance - dense_minimum_clearance
        ).clamp_min(0.0).mean(dim=1)

    result.update({
        "candidate_coarse_minimum_clearance": (
            coarse_minimum_clearance.reshape(-1)
        ),
        "per_sample_coarse_false_safe_candidate_count": false_safe_count,
        "per_sample_dense_clearance_drop": dense_clearance_drop,
    })
    return result
