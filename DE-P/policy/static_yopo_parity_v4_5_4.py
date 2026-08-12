"""V4.5.4 independent-score supervision with a frozen V4.5.2 proposal.

The proposal objective is byte-for-byte the V4.5.2 candidate total.  The
physical-radius clearance barrier is detached and used only to construct the
score label.  Consequently it cannot shorten or otherwise deform candidate
trajectories, even if proposal parameters are accidentally left trainable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import torch
from torch.nn import functional as F

from policy.static_yopo_parity_v4_4 import _relative_order_kl
from policy.static_yopo_parity_v4_5_1 import standardize_relative_cost_v4_5_1
from policy.static_yopo_parity_v4_5_2 import (
    StaticYOPOParityConfigV452,
    static_yopo_parity_objective_v4_5_2,
)
from policy.static_yopo_parity_v4_5_3 import (
    continuous_clearance_barrier_v4_5_3,
)


@dataclass(frozen=True)
class StaticYOPOParityConfigV454:
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
    clearance_barrier_weight: float = 1.0
    clearance_barrier_softness_m: float = 0.05

    @classmethod
    def from_mapping(cls, value):
        value = {} if value is None else dict(value)
        unknown = sorted(set(value) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown static_yopo_v4_5_4 keys: {unknown}")
        result = cls(**value)
        result.validate()
        return result

    def validate(self):
        self.as_v452().validate()
        if self.clearance_barrier_weight < 0.0:
            raise ValueError("clearance_barrier_weight must be non-negative")
        if self.clearance_barrier_softness_m <= 0.0:
            raise ValueError("clearance_barrier_softness_m must be positive")

    def as_v452(self):
        names = {field.name for field in fields(StaticYOPOParityConfigV452)}
        return StaticYOPOParityConfigV452(**{
            name: value for name, value in asdict(self).items()
            if name in names
        })

    def contract(self):
        return {
            "version": "static_yopo_original_parity_v4_5_4",
            "parent": "static_yopo_original_parity_v4_5_2",
            "candidate_objective": "unchanged_v4_5_2_total_cost",
            "score_target": (
                "detach(v4_5_2_total + physical_radius_clearance_barrier)"
            ),
            "clearance_barrier_formula": (
                "weight*softplus((vehicle_radius-minimum_clearance)/softness)"
            ),
            "score_feature_branch": "independent_1x1_feature_tower",
            "training_schedule": "five_epoch_score_only_no_joint_finetune",
            "qualification_gate": "none",
            "dynamic_training": False,
            **asdict(self),
        }


def static_yopo_parity_objective_v4_5_4(
    *, sampler, safety_loss, fixed, predicted, predicted_scores, goal_world,
    map_id, batch_size, candidate_count, route_goal_distance, config,
):
    """Keep V4.5.2 proposal gradients and supervise only Score with barrier."""
    config.validate()
    result = static_yopo_parity_objective_v4_5_2(
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
        config=config.as_v452(),
    )

    def candidates(name):
        return result[name].reshape(batch_size, candidate_count)

    proposal_total = candidates("candidate_raw_total_cost")
    minimum_clearance = candidates("candidate_minimum_clearance")
    barrier = continuous_clearance_barrier_v4_5_3(
        minimum_clearance,
        vehicle_radius=config.vehicle_radius_m,
        weight=config.clearance_barrier_weight,
        softness=config.clearance_barrier_softness_m,
    )
    # Both terms are label data.  No score loss can send a gradient into the
    # proposal, ESDF sampler, or clearance computation.
    score_target = proposal_total.detach() + barrier.detach()
    relative_label, label_scale = standardize_relative_cost_v4_5_1(
        score_target, config.relative_label_min_scale
    )
    score_regression = F.smooth_l1_loss(
        predicted_scores, relative_label, reduction="none"
    ).mean(dim=1)
    relative_order = _relative_order_kl(
        predicted_scores,
        score_target,
        config.relative_order_temperature,
    )

    # Crucial V4.5.4 boundary: proposal supervision is exactly V4.5.2.
    per_sample_trajectory = proposal_total.mean(dim=1)
    per_sample_score = config.score_regression_weight * score_regression
    per_sample_ranking = config.relative_order_weight * relative_order
    per_sample_total = (
        per_sample_trajectory + per_sample_score + per_sample_ranking
    )

    maximum_speed = candidates("candidate_maximum_speed")
    maximum_acceleration = candidates("candidate_maximum_acceleration")
    speed_unsafe = maximum_speed > config.max_speed_mps
    acceleration_unsafe = maximum_acceleration > config.max_acceleration_mps2
    hardware_feasible = ~(speed_unsafe | acceleration_unsafe)
    collision_free = minimum_clearance >= config.vehicle_radius_m
    physical_feasible = collision_free & hardware_feasible
    collision_available = collision_free.any(dim=1)
    physical_available = physical_feasible.any(dim=1)

    rows = torch.arange(batch_size, device=fixed.device)
    selected = predicted_scores.argmin(dim=1)
    oracle = score_target.argmin(dim=1)
    selected_collision_unsafe = ~collision_free[rows, selected]
    selected_hardware_unsafe = ~hardware_feasible[rows, selected]
    oracle_regret = score_target[rows, selected] - score_target[rows, oracle]

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
        "clearance_barrier_loss": barrier.mean(),
        "score_label": relative_label.reshape(-1),
        "candidate_proposal_total_cost": proposal_total.reshape(-1),
        "candidate_score_target_cost": score_target.reshape(-1),
        "candidate_clearance_barrier_cost": barrier.reshape(-1),
        # Retain the established name for proposal diagnostics and explicitly
        # do not replace it with the score-only target as V4.5.3 did.
        "candidate_raw_total_cost": proposal_total.reshape(-1),
        "candidate_projected_safe_mask": physical_feasible,
        "per_sample_total_loss": per_sample_total,
        "per_sample_trajectory_loss": per_sample_trajectory,
        "per_sample_score_loss": per_sample_score,
        "per_sample_ranking_loss": per_sample_ranking,
        "per_sample_clearance_barrier_loss": barrier.mean(dim=1),
        "per_sample_score_label_scale": label_scale,
        "per_sample_score_oracle_regret": oracle_regret,
        "per_sample_projected_selection_regret": oracle_regret,
        "per_sample_score_top1_match": selected.eq(oracle),
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
        "per_sample_feasible_candidate_count": physical_feasible.sum(
            dim=1
        ).to(fixed.dtype),
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
