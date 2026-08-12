"""V4.5.9 original-YOPO time-mean safety with a physical collision floor.

V4.5.8 intentionally localized the clearance curve, but applied that curve to
the minimum clearance of an entire candidate.  That made a single close sample
price the whole trajectory and rewarded short candidates that never reached a
nearby passage.  V4.5.9 keeps the same 81-point geometry and one score target,
then restores the original YOPO temporal-mean aggregation.

The 0.30 m vehicle radius remains authoritative.  Training uses a finite,
differentiable collision surrogate; deployment continues to reject the same
physical collision mask.  No progress, FOV, qualification, or extra ranking
gate is introduced here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch.nn import functional as F

from policy.static_yopo_parity_v4_4 import _relative_order_kl
from policy.static_yopo_parity_v4_5_1 import standardize_relative_cost_v4_5_1
from policy.static_yopo_parity_v4_5_6 import static_yopo_parity_objective_v4_5_6
from policy.static_yopo_parity_v4_5_8 import (
    StaticYOPOParityConfigV458,
    localized_static_clearance_cost_v4_5_8,
)


@dataclass(frozen=True)
class StaticYOPOParityConfigV459(StaticYOPOParityConfigV458):
    """The V4.5.8 contract with a deliberately weak safe-side shoulder."""

    radius_boundary_cost: float = 0.10

    def contract(self):
        return {
            "version": "static_yopo_original_parity_v4_5_9",
            "parent": "static_yopo_original_parity_v4_5_8",
            "candidate_objective": (
                "original_yopo_smooth_guidance_plus_81_sample_time_mean_"
                "localized_static_safety_plus_weak_kinematic_preference"
            ),
            "score_target": "detached_same_single_candidate_total_cost",
            "static_clearance_semantics": {
                "clear": "near_zero_above_clear_distance",
                "warning": "weak_exponential_between_clear_distance_and_radius",
                "collision_training": "finite_differentiable_surrogate",
                "collision_runtime": "positive_infinity_reject",
                "trajectory_aggregation": "time_mean_over_81_samples",
            },
            "extra_clearance_score_barrier": False,
            "extra_clearance_pairwise_ranking": False,
            "qualification_gate": "none",
            "dynamic_training": False,
            **asdict(self),
        }


def time_mean_localized_static_cost_v4_5_9(distance_samples, *, config):
    """Return per-candidate time-mean cost and physical collision mask.

    ``distance_samples`` is ``[batch, candidate, time]`` and must come from
    the same dense sampler used by the rest of the objective.  The minimum is
    retained only for physical diagnostics and the deployment-equivalent
    collision mask; it is not used as the differentiable trajectory cost.
    """
    config.validate()
    if distance_samples.ndim != 3:
        raise ValueError("distance samples must be [batch,candidate,time]")
    if distance_samples.shape[2] != config.static_safety_samples:
        raise ValueError("distance sample count violates V4.5.9 contract")
    point_cost, point_collision = localized_static_clearance_cost_v4_5_8(
        distance_samples,
        vehicle_radius=config.vehicle_radius_m,
        clear_distance=config.clear_distance_m,
        far_cost=config.far_clearance_cost,
        far_decay=config.far_decay_m,
        radius_boundary_cost=config.radius_boundary_cost,
        collision_surrogate_cost=config.collision_surrogate_cost,
        penetration_scale=config.penetration_scale_m,
        maximum_training_cost=config.maximum_training_cost,
    )
    candidate_cost = config.safety_weight * point_cost.mean(dim=2)
    collision_mask = point_collision.any(dim=2)
    minimum_clearance = distance_samples.amin(dim=2)
    return candidate_cost, collision_mask, minimum_clearance, point_cost


def static_yopo_parity_objective_v4_5_9(
    *, sampler, safety_loss, fixed, predicted, predicted_scores, goal_world,
    map_id, batch_size, candidate_count, route_goal_distance, config,
):
    config.validate()
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

    distance_samples = result["candidate_static_distance_samples"]
    if distance_samples.shape[:2] != (batch_size, candidate_count):
        raise RuntimeError("dense static distance sample shape mismatch")
    localized_static, collision_mask, minimum_clearance, point_cost = (
        time_mean_localized_static_cost_v4_5_9(
            distance_samples, config=config,
        )
    )
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
        "candidate_static_point_cost": point_cost,
        "candidate_minimum_clearance": minimum_clearance.reshape(-1),
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


__all__ = [
    "StaticYOPOParityConfigV459",
    "time_mean_localized_static_cost_v4_5_9",
    "static_yopo_parity_objective_v4_5_9",
]
