"""V4.5.6 continuous safety calibration for a frozen V4.5.5 proposal.

The original-YOPO candidate objective and the dense 81-point ESDF proposal
gradient remain unchanged.  Only Score supervision is strengthened: its
detached target adds a smooth physical-radius clearance barrier, and one
continuous pairwise term teaches lower-risk candidates to receive lower
scores.  No candidate is rejected and no qualification Gate is introduced.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import torch
from torch.nn import functional as F

from policy.static_yopo_parity_v4_4 import _relative_order_kl
from policy.static_yopo_parity_v4_5_1 import standardize_relative_cost_v4_5_1
from policy.static_yopo_parity_v4_5_3 import continuous_clearance_barrier_v4_5_3
from policy.static_yopo_parity_v4_5_5 import (
    StaticYOPOParityConfigV455,
    static_yopo_parity_objective_v4_5_5,
)


@dataclass(frozen=True)
class StaticYOPOParityConfigV456:
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
    dangerous_segment_window: int = 14
    dangerous_segment_focus: float = 8.0
    large_vertical_displacement_m: float = 1.0
    clearance_score_weight: float = 1.0
    clearance_score_softness_m: float = 0.08
    clearance_pairwise_weight: float = 1.0
    clearance_pairwise_margin: float = 0.50

    @classmethod
    def from_mapping(cls, value):
        value = {} if value is None else dict(value)
        unknown = sorted(set(value) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown static_yopo_v4_5_6 keys: {unknown}")
        result = cls(**value)
        result.validate()
        return result

    def validate(self):
        self.as_v455().validate()
        nonnegative = (
            self.clearance_score_weight,
            self.clearance_pairwise_weight,
            self.clearance_pairwise_margin,
        )
        if min(nonnegative) < 0.0:
            raise ValueError("V4.5.6 safety-score weights must be non-negative")
        if self.clearance_score_softness_m <= 0.0:
            raise ValueError("clearance_score_softness_m must be positive")

    def as_v455(self):
        names = {field.name for field in fields(StaticYOPOParityConfigV455)}
        return StaticYOPOParityConfigV455(**{
            name: value for name, value in asdict(self).items()
            if name in names
        })

    def contract(self):
        return {
            "version": "static_yopo_original_parity_v4_5_6",
            "parent": "static_yopo_original_parity_v4_5_5",
            "candidate_objective": "unchanged_dense_v4_5_5_total_cost",
            "score_target": (
                "detach(v4_5_5_total + continuous_physical_clearance_cost)"
            ),
            "score_ranking": (
                "one_listwise_total_order_plus_one_continuous_clearance_order"
            ),
            "candidate_rejection": "none",
            "qualification_gate": "none",
            "dynamic_training": False,
            **asdict(self),
        }


def continuous_clearance_pairwise_ranking_v4_5_6(
    predicted_scores, minimum_clearance, *, vehicle_radius, softness, margin,
):
    """Continuously rank lower collision risk ahead of higher collision risk.

    The detached sigmoid risk avoids a Boolean safe/unsafe label.  Candidate
    pairs near the physical radius receive the strongest relative weight;
    pairs with indistinguishable risk contribute zero.  Lower network score
    remains better, matching the YOPO inference convention.
    """
    if predicted_scores.shape != minimum_clearance.shape:
        raise ValueError("score and clearance candidate shapes must match")
    if predicted_scores.ndim != 2:
        raise ValueError("score and clearance must be [B,N]")
    if float(vehicle_radius) <= 0.0 or float(softness) <= 0.0:
        raise ValueError("vehicle radius and softness must be positive")
    if float(margin) < 0.0:
        raise ValueError("pairwise margin must be non-negative")

    risk = torch.sigmoid(
        (float(vehicle_radius) - minimum_clearance.detach())
        / float(softness)
    )
    # [B,i,j]: positive means candidate j is riskier than candidate i.
    risk_delta = risk[:, None, :] - risk[:, :, None]
    score_delta = predicted_scores[:, None, :] - predicted_scores[:, :, None]
    direction = risk_delta.sign()
    importance = risk_delta.abs()
    pair_mask = torch.triu(
        torch.ones_like(importance, dtype=torch.bool), diagonal=1
    )
    importance = importance * pair_mask.to(importance.dtype)
    pair_loss = importance * F.softplus(
        float(margin) - direction * score_delta
    )
    return pair_loss.sum(dim=(1, 2)) / importance.sum(
        dim=(1, 2)
    ).clamp_min(1.0e-8)


def static_yopo_parity_objective_v4_5_6(
    *, sampler, safety_loss, fixed, predicted, predicted_scores, goal_world,
    map_id, batch_size, candidate_count, route_goal_distance, config,
):
    config.validate()
    result = static_yopo_parity_objective_v4_5_5(
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
        config=config.as_v455(),
    )

    def candidates(name):
        return result[name].reshape(batch_size, candidate_count)

    proposal_total = candidates("candidate_raw_total_cost")
    minimum_clearance = candidates("candidate_minimum_clearance")
    maximum_speed = candidates("candidate_maximum_speed")
    maximum_acceleration = candidates("candidate_maximum_acceleration")
    clearance_cost = continuous_clearance_barrier_v4_5_3(
        minimum_clearance,
        vehicle_radius=config.vehicle_radius_m,
        weight=config.clearance_score_weight,
        softness=config.clearance_score_softness_m,
    )
    score_target = proposal_total.detach() + clearance_cost.detach()
    relative_label, label_scale = standardize_relative_cost_v4_5_1(
        score_target, config.relative_label_min_scale
    )
    score_regression = F.smooth_l1_loss(
        predicted_scores, relative_label, reduction="none"
    ).mean(dim=1)
    listwise_order = _relative_order_kl(
        predicted_scores, score_target, config.relative_order_temperature
    )
    clearance_order = continuous_clearance_pairwise_ranking_v4_5_6(
        predicted_scores,
        minimum_clearance,
        vehicle_radius=config.vehicle_radius_m,
        softness=config.clearance_score_softness_m,
        margin=config.clearance_pairwise_margin,
    )

    per_sample_trajectory = proposal_total.mean(dim=1)
    per_sample_score = config.score_regression_weight * score_regression
    per_sample_ranking = (
        config.relative_order_weight * listwise_order
        + config.clearance_pairwise_weight * clearance_order
    )
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
    oracle = score_target.argmin(dim=1)
    oracle_regret = score_target[rows, selected] - score_target[rows, oracle]

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
        "clearance_barrier_loss": clearance_cost.mean(),
        "clearance_pairwise_ranking_loss": clearance_order.mean(),
        "score_label": relative_label.reshape(-1),
        "candidate_proposal_total_cost": proposal_total.reshape(-1),
        "candidate_score_target_cost": score_target.reshape(-1),
        "candidate_clearance_barrier_cost": clearance_cost.reshape(-1),
        # Candidate generation continues to optimize the unmodified V4.5.5
        # objective; the safety emphasis belongs only to Score supervision.
        "candidate_raw_total_cost": proposal_total.reshape(-1),
        "candidate_projected_safe_mask": physical_feasible,
        "per_sample_total_loss": per_sample_total,
        "per_sample_trajectory_loss": per_sample_trajectory,
        "per_sample_score_loss": per_sample_score,
        "per_sample_ranking_loss": per_sample_ranking,
        "per_sample_clearance_barrier_loss": clearance_cost.mean(dim=1),
        "per_sample_clearance_pairwise_ranking_loss": clearance_order,
        "per_sample_score_label_scale": label_scale,
        "per_sample_score_oracle_regret": oracle_regret,
        "per_sample_projected_selection_regret": oracle_regret,
        "per_sample_score_top1_match": selected.eq(oracle),
        "per_sample_oracle_collision_unsafe": (
            ~collision_free[rows, oracle]
        ).to(fixed.dtype),
        "per_sample_oracle_speed_unsafe": speed_unsafe[rows, oracle].to(
            fixed.dtype
        ),
        "per_sample_oracle_acceleration_unsafe": acceleration_unsafe[
            rows, oracle
        ].to(fixed.dtype),
        "per_sample_oracle_hardware_unsafe": (
            ~hardware_feasible[rows, oracle]
        ).to(fixed.dtype),
        "per_sample_oracle_physical_unsafe": (
            ~physical_feasible[rows, oracle]
        ).to(fixed.dtype),
        "per_sample_oracle_vertical_displacement": oracle_vertical,
        "per_sample_oracle_absolute_vertical_displacement": (
            oracle_vertical.abs()
        ),
        "per_sample_oracle_large_vertical_maneuver": oracle_vertical.abs().ge(
            config.large_vertical_displacement_m
        ).to(fixed.dtype),
        "per_sample_oracle_primitive_row": primitive_row[oracle].to(
            fixed.dtype
        ),
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
