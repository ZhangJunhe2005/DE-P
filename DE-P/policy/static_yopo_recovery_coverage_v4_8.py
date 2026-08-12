"""V4.8 recovery-state proposal-capacity objective.

V4.7 can rotate the camera far enough to reveal an exit, but the network was
never trained on that low-speed, strongly off-axis state.  This objective
keeps the complete V4.5.10 loss and score target unchanged, then adds one weak
continuous proposal term on the versioned recovery subset only: when a
canonical YOPO lattice ray is open in the authoritative ESDF, the matching
candidate is discouraged from collapsing to very short progress along that
lattice direction.

This is a differentiable training objective, not a Boolean feasibility Gate.
It does not reject candidates and it does not alter deployment safety checks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch.nn import functional as F

from policy.primitive import LatticePrimitive
from policy.static_yopo_parity_v4_5_10 import (
    StaticYOPOParityConfigV4510,
    static_yopo_parity_objective_v4_5_10,
)


@dataclass(frozen=True)
class StaticYOPORecoveryCoverageConfigV48(StaticYOPOParityConfigV4510):
    safe_sector_weight: float = 0.05
    safe_sector_ray_start_m: float = 0.50
    safe_sector_ray_end_m: float = 4.00
    safe_sector_ray_samples: int = 12
    safe_sector_openness_center_m: float = 0.45
    safe_sector_openness_softness_m: float = 0.10
    safe_sector_target_length_m: float = 3.00
    safe_sector_shortfall_softness_m: float = 0.35

    def validate(self):
        super().validate()
        if self.safe_sector_weight < 0.0:
            raise ValueError("safe_sector_weight must be non-negative")
        if self.safe_sector_ray_start_m < 0.0:
            raise ValueError("safe-sector ray start must be non-negative")
        if self.safe_sector_ray_end_m <= self.safe_sector_ray_start_m:
            raise ValueError("safe-sector ray end must exceed its start")
        if self.safe_sector_ray_samples < 2:
            raise ValueError("safe-sector ray requires at least two samples")
        positive = (
            self.safe_sector_openness_softness_m,
            self.safe_sector_target_length_m,
            self.safe_sector_shortfall_softness_m,
        )
        if min(positive) <= 0.0:
            raise ValueError("safe-sector scales and target must be positive")

    def contract(self):
        return {
            "version": "static_yopo_recovery_coverage_v4_8",
            "parent": "static_yopo_original_parity_v4_5_10",
            "candidate_objective": (
                "unchanged_v4_5_10_plus_one_recovery_only_continuous_"
                "open_lattice_axial_progress_term"
            ),
            "score_target": "byte_equivalent_v4_5_10_detached_total_cost",
            "candidate_rejection": "none",
            "qualification_gate": "none",
            "dynamic_training": False,
            **asdict(self),
        }


def _network_order_lattice_directions(*, device, dtype, candidate_count):
    primitive = LatticePrimitive.get_instance()
    directions = primitive.getStateLattice().to(
        device=device, dtype=dtype,
    ).flip(0)
    if directions.shape != (candidate_count, 3):
        raise RuntimeError(
            "safe-sector lattice does not match network candidate order"
        )
    return directions / directions.norm(dim=1, keepdim=True).clamp_min(1.0e-8)


def recovery_safe_sector_terms_v4_8(
    *, safety_loss, current_position, rotation_world_from_body, predicted,
    map_id, recovery_state, batch_size, candidate_count, config,
):
    """Return the sole V4.8 proposal loss and its continuous diagnostics."""
    config.validate()
    current_position = current_position.reshape(batch_size, 3)
    rotation_world_from_body = rotation_world_from_body.reshape(
        batch_size, 3, 3,
    )
    if recovery_state.dtype != torch.bool:
        raise ValueError("recovery_state must be a boolean tensor")
    recovery = recovery_state.to(
        device=predicted.device, dtype=predicted.dtype,
    ).reshape(batch_size)

    body_direction = _network_order_lattice_directions(
        device=predicted.device, dtype=predicted.dtype,
        candidate_count=candidate_count,
    )
    world_direction = torch.einsum(
        "bij,nj->bni", rotation_world_from_body, body_direction,
    )
    ray_distance = torch.linspace(
        config.safe_sector_ray_start_m,
        config.safe_sector_ray_end_m,
        config.safe_sector_ray_samples,
        device=predicted.device,
        dtype=predicted.dtype,
    )
    ray_points = (
        current_position[:, None, None, :]
        + world_direction[:, :, None, :] * ray_distance[None, None, :, None]
    )
    ray_cost, ray_clearance = safety_loss.get_distance_cost(
        ray_points.reshape(batch_size, -1, 3), map_id,
    )
    ray_cost = ray_cost.reshape(
        batch_size, candidate_count, config.safe_sector_ray_samples,
    )
    ray_clearance = ray_clearance.reshape(
        batch_size, candidate_count, config.safe_sector_ray_samples,
    )
    # ESDF geometry is detached: it weights which existing lattice sectors are
    # useful, while gradients flow only through the corresponding candidate
    # endpoint progress.  There is no learned map shortcut or deployment
    # feasibility Gate.  SafetyLoss marks authority-out-of-bounds queries with
    # out_of_bounds_cost; such rays (and equivalently severe penetrations) must
    # never be rewarded as open merely because their returned distance is 0.
    soft_clearance = ray_clearance.amin(dim=2).detach()
    openness = torch.sigmoid((
        soft_clearance - config.safe_sector_openness_center_m
    ) / config.safe_sector_openness_softness_m)
    out_of_bounds_cost = getattr(safety_loss, "out_of_bounds_cost", None)
    if out_of_bounds_cost is not None:
        closed_ray = ray_cost.detach().amax(dim=2).ge(
            float(out_of_bounds_cost) - 1.0e-6,
        )
        openness = openness.masked_fill(closed_ray, 0.0)

    endpoint = predicted[:, :, 0].reshape(batch_size, candidate_count, 3)
    endpoint_delta = endpoint - current_position[:, None, :]
    endpoint_length = endpoint_delta.norm(dim=2)
    # Axial progress (rather than Euclidean length) gives a useful non-zero
    # gradient even at a fully collapsed endpoint and prevents lateral motion
    # from satisfying the matching lattice anchor by accident.
    endpoint_progress = (endpoint_delta * world_direction).sum(dim=2)
    shortfall = config.safe_sector_shortfall_softness_m * F.softplus((
        config.safe_sector_target_length_m - endpoint_progress
    ) / config.safe_sector_shortfall_softness_m)
    candidate_cost = (
        config.safe_sector_weight
        * recovery[:, None]
        * openness
        * shortfall
    )
    per_sample_loss = candidate_cost.mean(dim=1)
    return {
        "candidate_cost": candidate_cost,
        "per_sample_loss": per_sample_loss,
        "openness": openness,
        "soft_clearance": soft_clearance,
        "endpoint_length": endpoint_length,
        "endpoint_progress": endpoint_progress,
        "recovery": recovery,
    }


def static_yopo_recovery_coverage_objective_v4_8(
    *, sampler, safety_loss, fixed, predicted, predicted_scores, goal_world,
    map_id, batch_size, candidate_count, route_goal_distance, recovery_state,
    rotation_world_from_body, config,
):
    """Add recovery proposal capacity without changing V4.5.10 Score labels."""
    config.validate()
    result = static_yopo_parity_objective_v4_5_10(
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
    current_position = fixed[:, :, 0].reshape(
        batch_size, candidate_count, 3,
    )[:, 0]
    terms = recovery_safe_sector_terms_v4_8(
        safety_loss=safety_loss,
        current_position=current_position,
        rotation_world_from_body=rotation_world_from_body,
        predicted=predicted,
        map_id=map_id,
        recovery_state=recovery_state,
        batch_size=batch_size,
        candidate_count=candidate_count,
        config=config,
    )
    per_sample_loss = terms["per_sample_loss"]
    per_sample_total = result["per_sample_total_loss"] + per_sample_loss
    per_sample_trajectory = (
        result["per_sample_trajectory_loss"] + per_sample_loss
    )
    recovery = terms["recovery"]
    endpoint_length = terms["endpoint_length"]
    candidate_proposal_total = result[
        "candidate_proposal_total_cost"
    ].reshape(batch_size, candidate_count) + terms["candidate_cost"]
    recovery_count = recovery.sum().clamp_min(1.0)

    def recovery_mean(values):
        return (recovery * values).sum() / recovery_count

    # Deliberately do not replace score_label, score_loss, ranking_loss,
    # candidate_score_target_cost, or candidate_raw_total_cost.  Only the
    # proposal diagnostic follows the additional candidate gradient.
    result.update({
        "total_loss": per_sample_total.mean(),
        "trajectory_loss": per_sample_trajectory.mean(),
        "safe_sector_coverage_loss": per_sample_loss.mean(),
        "candidate_proposal_total_cost": candidate_proposal_total.reshape(-1),
        "candidate_safe_sector_coverage_cost": terms[
            "candidate_cost"
        ].reshape(-1),
        "safe_sector_anchor_openness": terms["openness"],
        "safe_sector_anchor_soft_clearance": terms["soft_clearance"],
        "safe_sector_anchor_progress": terms["endpoint_progress"],
        "recovery_sample_fraction": recovery.mean(),
        "recovery_open_sector_soft_count": recovery_mean(
            terms["openness"].sum(dim=1)
        ),
        "recovery_mean_endpoint_distance": recovery_mean(
            endpoint_length.mean(dim=1)
        ),
        "recovery_max_endpoint_distance": recovery_mean(
            endpoint_length.amax(dim=1)
        ),
        "per_sample_total_loss": per_sample_total,
        "per_sample_trajectory_loss": per_sample_trajectory,
        "per_sample_safe_sector_coverage_loss": per_sample_loss,
        "per_sample_recovery_sample_fraction": recovery,
        "per_sample_recovery_open_sector_soft_count": (
            recovery * terms["openness"].sum(dim=1)
        ),
        "per_sample_recovery_mean_endpoint_distance": (
            recovery * endpoint_length.mean(dim=1)
        ),
        "per_sample_recovery_max_endpoint_distance": (
            recovery * endpoint_length.amax(dim=1)
        ),
    })
    return result


__all__ = [
    "StaticYOPORecoveryCoverageConfigV48",
    "recovery_safe_sector_terms_v4_8",
    "static_yopo_recovery_coverage_objective_v4_8",
]
