"""V4.5.10 bounded tail-aware static-safety objective.

V4.5.9 restored the original YOPO time-mean aggregation and recovered useful
candidate length, but a very short near-collision segment could again be
diluted by the other 76--80 safe samples.  V4.5.10 keeps that time mean as
90 percent of the *single* static-safety term and assigns the remaining 10
percent to the five highest-cost samples.  This is deliberately not a new
boolean Gate and not a minimum-clearance barrier.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch.nn import functional as F

from policy.static_yopo_parity_v4_4 import _relative_order_kl
from policy.static_yopo_parity_v4_5_1 import standardize_relative_cost_v4_5_1
from policy.static_yopo_parity_v4_5_9 import (
    StaticYOPOParityConfigV459,
    static_yopo_parity_objective_v4_5_9,
)


@dataclass(frozen=True)
class StaticYOPOParityConfigV4510(StaticYOPOParityConfigV459):
    time_mean_weight: float = 0.90
    worst_sample_weight: float = 0.10
    worst_sample_count: int = 5

    def validate(self):
        super().validate()
        if self.time_mean_weight < 0.0 or self.worst_sample_weight < 0.0:
            raise ValueError("V4.5.10 aggregation weights must be non-negative")
        if abs(self.time_mean_weight + self.worst_sample_weight - 1.0) > 1e-9:
            raise ValueError("V4.5.10 aggregation weights must sum to one")
        if not 1 <= self.worst_sample_count <= self.static_safety_samples:
            raise ValueError("worst_sample_count must fit static safety samples")

    def contract(self):
        return {
            "version": "static_yopo_original_parity_v4_5_10",
            "parent": "static_yopo_original_parity_v4_5_9",
            "candidate_objective": (
                "original_yopo_smooth_guidance_plus_one_81_sample_"
                "tail_aware_static_safety_plus_weak_kinematic_preference"
            ),
            "score_target": "detached_same_single_candidate_total_cost",
            "static_clearance_semantics": {
                "point_cost": "v4_5_9_localized_continuous_clearance_curve",
                "trajectory_aggregation": (
                    "90_percent_time_mean_plus_10_percent_worst_5_mean"
                ),
                "collision_training": "finite_differentiable_surrogate",
                "collision_runtime": "0_35_m_representation_aware_reject",
            },
            "qualification_gate": "none",
            "dynamic_training": False,
            **asdict(self),
        }


def tail_aware_static_cost_v4_5_10(point_cost, *, config):
    """Aggregate ``[batch,candidate,time]`` point costs without a hard min."""
    config.validate()
    if point_cost.ndim != 3:
        raise ValueError("point cost must be [batch,candidate,time]")
    if point_cost.shape[2] != config.static_safety_samples:
        raise ValueError("point-cost sample count violates V4.5.10 contract")
    mean_cost = point_cost.mean(dim=2)
    worst_cost = point_cost.topk(
        k=config.worst_sample_count, dim=2, largest=True, sorted=False,
    ).values.mean(dim=2)
    combined = config.safety_weight * (
        config.time_mean_weight * mean_cost
        + config.worst_sample_weight * worst_cost
    )
    return combined, config.safety_weight * mean_cost, \
        config.safety_weight * worst_cost


def static_yopo_parity_objective_v4_5_10(
    *, sampler, safety_loss, fixed, predicted, predicted_scores, goal_world,
    map_id, batch_size, candidate_count, route_goal_distance, config,
):
    config.validate()
    # Reuse V4.5.9 for the dense geometry, derivatives and all selected-candidate
    # diagnostics.  Replace only its time-mean static term and the score label.
    result = static_yopo_parity_objective_v4_5_9(
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
        config=config,
    )

    def candidates(name):
        return result[name].reshape(batch_size, candidate_count)

    point_cost = result["candidate_static_point_cost"]
    tail_aware, mean_cost, worst_cost = tail_aware_static_cost_v4_5_10(
        point_cost, config=config,
    )
    v459_total = candidates("candidate_raw_total_cost")
    v459_static = candidates("candidate_static_cost")
    candidate_total = v459_total - v459_static + tail_aware

    relative_label, label_scale = standardize_relative_cost_v4_5_1(
        candidate_total, config.relative_label_min_scale,
    )
    score_regression = F.smooth_l1_loss(
        predicted_scores, relative_label, reduction="none",
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
    collision_mask = result["candidate_runtime_infinite_collision_mask"]
    maximum_speed = candidates("candidate_maximum_speed")
    maximum_acceleration = candidates("candidate_maximum_acceleration")
    speed_unsafe = maximum_speed > config.max_speed_mps
    acceleration_unsafe = maximum_acceleration > config.max_acceleration_mps2
    hardware_feasible = ~(speed_unsafe | acceleration_unsafe)
    physical_feasible = (~collision_mask) & hardware_feasible
    oracle_regret = (
        candidate_total.detach()[rows, selected]
        - candidate_total.detach()[rows, oracle]
    )

    current_position = fixed[:, :, 0].reshape(
        batch_size, candidate_count, 3,
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
        "static_safety_loss": tail_aware.mean(),
        "static_time_mean_cost": mean_cost.mean(),
        "static_worst_five_cost": worst_cost.mean(),
        "score_label": relative_label.reshape(-1),
        "candidate_static_cost": tail_aware.reshape(-1),
        "candidate_static_time_mean_cost": mean_cost.reshape(-1),
        "candidate_static_worst_five_cost": worst_cost.reshape(-1),
        "candidate_proposal_total_cost": candidate_total.reshape(-1),
        "candidate_score_target_cost": candidate_total.reshape(-1),
        "candidate_raw_total_cost": candidate_total.reshape(-1),
        "candidate_projected_safe_mask": physical_feasible,
        "per_sample_total_loss": per_sample_total,
        "per_sample_trajectory_loss": per_sample_trajectory,
        "per_sample_score_loss": per_sample_score,
        "per_sample_ranking_loss": per_sample_ranking,
        "per_sample_static_safety_loss": tail_aware.mean(dim=1),
        "per_sample_static_time_mean_cost": mean_cost.mean(dim=1),
        "per_sample_static_worst_five_cost": worst_cost.mean(dim=1),
        "per_sample_score_label_scale": label_scale,
        "per_sample_score_oracle_regret": oracle_regret,
        "per_sample_projected_selection_regret": oracle_regret,
        "per_sample_score_top1_match": selected.eq(oracle),
        "per_sample_oracle_collision_unsafe": collision_mask[
            rows, oracle
        ].to(fixed.dtype),
        "per_sample_oracle_speed_unsafe": speed_unsafe[
            rows, oracle
        ].to(fixed.dtype),
        "per_sample_oracle_acceleration_unsafe": acceleration_unsafe[
            rows, oracle
        ].to(fixed.dtype),
        "per_sample_oracle_hardware_unsafe": (~hardware_feasible[
            rows, oracle
        ]).to(fixed.dtype),
        "per_sample_oracle_physical_unsafe": (~physical_feasible[
            rows, oracle
        ]).to(fixed.dtype),
        "per_sample_oracle_vertical_displacement": oracle_vertical,
        "per_sample_oracle_absolute_vertical_displacement": (
            oracle_vertical.abs()
        ),
        "per_sample_oracle_large_vertical_maneuver": oracle_vertical.abs().ge(
            config.large_vertical_displacement_m,
        ).to(fixed.dtype),
        "per_sample_oracle_primitive_row": primitive_row[oracle].to(
            fixed.dtype,
        ),
        "per_sample_oracle_vertical_primitive": primitive_row[oracle].ne(1).to(
            fixed.dtype,
        ),
        "per_sample_oracle_upward_primitive": primitive_row[oracle].eq(0).to(
            fixed.dtype,
        ),
        "per_sample_oracle_level_primitive": primitive_row[oracle].eq(1).to(
            fixed.dtype,
        ),
        "per_sample_oracle_downward_primitive": primitive_row[oracle].eq(2).to(
            fixed.dtype,
        ),
    })
    return result


__all__ = [
    "StaticYOPOParityConfigV4510",
    "tail_aware_static_cost_v4_5_10",
    "static_yopo_parity_objective_v4_5_10",
]
