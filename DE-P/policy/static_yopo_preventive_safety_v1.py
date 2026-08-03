"""Differentiable early-clearance supervision for Route-A static YOPO.

The historical collision objective becomes large only near geometry.  This
versioned objective adds an earlier, speed-aware clearance margin while keeping
the anti-hover and hard 6 m/s / 6 m/s^2 contracts independent.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class PreventiveSafetyConfigV1:
    enabled: bool = False
    base_clearance_m: float = 1.20
    speed_clearance_gain_s: float = 0.12
    maximum_clearance_m: float = 2.00
    softplus_temperature_m: float = 0.20
    loss_weight: float = 2.0
    ranking_weight: float = 1.0
    ranking_margin: float = 0.50
    label_weight: float = 2.0
    candidate_mean_weight: float = 0.25
    candidate_cvar_weight: float = 0.75
    candidate_cvar_fraction: float = 1.0 / 3.0
    feasible_coverage_weight: float = 1.0
    minimum_clear_candidates: int = 3
    approach_depth_m: float = 5.0
    near_depth_m: float = 3.0
    near_fraction_reference: float = 0.10
    maximum_sample_weight: float = 4.0
    depth_max_m: float = 20.0

    @classmethod
    def from_mapping(cls, value):
        value = {} if value is None else dict(value)
        unknown = sorted(set(value) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown preventive_safety keys: {unknown}")
        result = cls(**value)
        result.validate()
        return result

    def validate(self):
        positive = (
            self.base_clearance_m, self.maximum_clearance_m,
            self.softplus_temperature_m, self.ranking_margin,
            self.approach_depth_m, self.near_depth_m,
            self.near_fraction_reference, self.maximum_sample_weight,
            self.depth_max_m,
        )
        if min(positive) <= 0:
            raise ValueError("preventive safety distances/scales must be positive")
        if self.maximum_clearance_m < self.base_clearance_m:
            raise ValueError("maximum clearance must cover base clearance")
        if self.near_depth_m >= self.approach_depth_m:
            raise ValueError("near depth must be below approach depth")
        weights = (
            self.speed_clearance_gain_s, self.loss_weight,
            self.ranking_weight, self.label_weight,
            self.candidate_mean_weight, self.candidate_cvar_weight,
            self.feasible_coverage_weight,
        )
        if min(weights) < 0:
            raise ValueError("preventive safety weights must be non-negative")
        if not 0 < self.candidate_cvar_fraction <= 1:
            raise ValueError("candidate_cvar_fraction must be in (0,1]")
        if self.minimum_clear_candidates < 1:
            raise ValueError("minimum_clear_candidates must be positive")
        if self.maximum_sample_weight < 1:
            raise ValueError("maximum_sample_weight must be at least one")


def preventive_safety_objective_v1(
    predicted_scores, static_min_distance, current_speed_mps,
    normalized_depth, config: PreventiveSafetyConfigV1,
):
    """Return continuous clearance gradients and score-order supervision.

    The required clearance grows mildly with current speed.  Near-field image
    occupancy only changes the sample weight; it never supplies labels or actor
    identity to the network.
    """
    config.validate()
    if predicted_scores.shape != static_min_distance.shape:
        raise ValueError("preventive score and clearance shapes differ")
    batch_size, candidate_count = predicted_scores.shape
    speed = current_speed_mps.reshape(batch_size, 1)
    required = (
        config.base_clearance_m + config.speed_clearance_gain_s * speed
    ).clamp(max=config.maximum_clearance_m)

    # Softplus retains a useful gradient before crossing the desired margin.
    normalized_deficit = (
        F.softplus(
            (required - static_min_distance)
            / config.softplus_temperature_m
        )
        * config.softplus_temperature_m
        / required.clamp_min(1.0e-6)
    )
    candidate_loss = normalized_deficit.square()

    depth_m = normalized_depth.float() * config.depth_max_m
    if depth_m.ndim != 4 or depth_m.shape[0] != batch_size:
        raise ValueError("normalized depth must be [B,1,H,W]")
    approach_fraction = (depth_m < config.approach_depth_m).float().mean((1, 2, 3))
    near_fraction = (depth_m < config.near_depth_m).float().mean((1, 2, 3))
    sample_weight = (
        1.0
        + approach_fraction / config.near_fraction_reference
        + 2.0 * near_fraction / config.near_fraction_reference
    ).clamp(max=config.maximum_sample_weight).detach()

    top_count = max(
        1, int(candidate_count * config.candidate_cvar_fraction + 0.999999)
    )
    coverage_count = min(config.minimum_clear_candidates, candidate_count)
    candidate_mean = candidate_loss.mean(dim=1)
    candidate_cvar = candidate_loss.topk(
        top_count, dim=1, largest=True
    ).values.mean(dim=1)
    coverage = candidate_loss.topk(
        coverage_count, dim=1, largest=False
    ).values.mean(dim=1)
    per_sample_loss = sample_weight * (
        config.candidate_mean_weight * candidate_mean
        + config.candidate_cvar_weight * candidate_cvar
        + config.feasible_coverage_weight * coverage
    )

    clear = static_min_distance >= required
    pair_loss = F.relu(
        predicted_scores[:, :, None] - predicted_scores[:, None, :]
        + config.ranking_margin
    )
    pair_mask = clear[:, :, None] & (~clear)[:, None, :]
    pair_count = pair_mask.sum((1, 2))
    ranking = (
        (pair_loss * pair_mask).sum((1, 2))
        / pair_count.clamp(min=1).to(pair_loss.dtype)
    )
    ranking = torch.where(pair_count > 0, ranking, ranking * 0.0)
    selected = predicted_scores.argmin(dim=1)
    row = torch.arange(batch_size, device=predicted_scores.device)
    selected_clearance = static_min_distance[row, selected]
    return {
        "per_sample_loss": per_sample_loss,
        "ranking_per_sample": ranking,
        "candidate_loss": candidate_loss,
        "candidate_label_cost": config.label_weight * candidate_loss.detach(),
        "required_clearance": required.squeeze(1),
        "sample_weight": sample_weight,
        "clear_mask": clear,
        "clear_candidate_count": clear.sum(dim=1),
        "selected_clearance": selected_clearance,
        "selected_below_margin": selected_clearance < required.squeeze(1),
    }


__all__ = ["PreventiveSafetyConfigV1", "preventive_safety_objective_v1"]
