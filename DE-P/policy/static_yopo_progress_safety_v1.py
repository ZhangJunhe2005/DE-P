"""Progress-aware supervision inside the set of safety-qualified proposals.

Safety remains lexicographically authoritative.  This objective never makes an
unsafe proposal preferable because progress cost and ranking are applied only
to candidates already certified by the static, preventive, and hardware
training masks.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class ProgressSafetyConfigV1:
    enabled: bool = False
    minimum_progress_m: float = 1.0
    preferred_progress_m: float = 3.5
    softplus_temperature_m: float = 0.25
    loss_weight: float = 1.0
    ranking_weight: float = 1.0
    ranking_margin: float = 0.75
    label_weight: float = 2.5
    minimum_progress_candidates: int = 3

    @classmethod
    def from_mapping(cls, value):
        value = {} if value is None else dict(value)
        unknown = sorted(set(value) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown progress_safety keys: {unknown}")
        result = cls(**value)
        result.validate()
        return result

    def validate(self):
        if self.minimum_progress_m <= 0:
            raise ValueError("minimum progress must be positive")
        if self.preferred_progress_m < self.minimum_progress_m:
            raise ValueError("preferred progress must cover runtime minimum")
        if self.softplus_temperature_m <= 0 or self.ranking_margin <= 0:
            raise ValueError("progress temperature/margin must be positive")
        if min(self.loss_weight, self.ranking_weight, self.label_weight) < 0:
            raise ValueError("progress weights must be non-negative")
        if self.minimum_progress_candidates < 1:
            raise ValueError("minimum progress candidate count must be positive")


def progress_safety_objective_v1(
    predicted_scores, endpoint_distance_m, safety_qualified_mask,
    config: ProgressSafetyConfigV1,
):
    """Prefer meaningful motion, but only among already-safe candidates."""
    config.validate()
    if predicted_scores.shape != endpoint_distance_m.shape:
        raise ValueError("progress score and endpoint shapes differ")
    if safety_qualified_mask.shape != endpoint_distance_m.shape:
        raise ValueError("progress safety mask and endpoint shapes differ")
    if safety_qualified_mask.dtype != torch.bool:
        raise TypeError("progress safety mask must be boolean")

    batch_size, candidate_count = predicted_scores.shape
    safe = safety_qualified_mask.detach()
    preferred = torch.as_tensor(
        config.preferred_progress_m,
        device=endpoint_distance_m.device,
        dtype=endpoint_distance_m.dtype,
    )
    deficit = (
        F.softplus(
            (preferred - endpoint_distance_m)
            / config.softplus_temperature_m
        )
        * config.softplus_temperature_m
        / preferred
    ).square()
    safe_deficit = torch.where(safe, deficit, torch.zeros_like(deficit))

    # Candidate generation needs only a bounded number of useful escape
    # options.  Optimising the best K safe proposals avoids forcing every
    # primitive to fly straight/far and preserves lateral/braking diversity.
    coverage_count = min(config.minimum_progress_candidates, candidate_count)
    sentinel = torch.full_like(deficit, 1.0e6)
    masked = torch.where(safe, deficit, sentinel)
    values, indices = masked.topk(coverage_count, dim=1, largest=False)
    selected_safe = safe.gather(1, indices)
    selected_values = torch.where(
        selected_safe, values, torch.zeros_like(values)
    )
    selected_count = selected_safe.sum(dim=1)
    per_sample_loss = selected_values.sum(dim=1) / selected_count.clamp(
        min=1
    ).to(selected_values.dtype)
    per_sample_loss = torch.where(
        selected_count > 0, per_sample_loss, per_sample_loss * 0.0
    )

    sufficient = safe & endpoint_distance_m.ge(config.minimum_progress_m)
    too_short = safe & ~sufficient
    pair_loss = F.relu(
        predicted_scores[:, :, None] - predicted_scores[:, None, :]
        + config.ranking_margin
    )
    pair_mask = sufficient[:, :, None] & too_short[:, None, :]
    pair_count = pair_mask.sum((1, 2))
    ranking = (
        (pair_loss * pair_mask).sum((1, 2))
        / pair_count.clamp(min=1).to(pair_loss.dtype)
    )
    ranking = torch.where(pair_count > 0, ranking, ranking * 0.0)

    selected_index = predicted_scores.argmin(dim=1)
    rows = torch.arange(batch_size, device=predicted_scores.device)
    selected_progress = endpoint_distance_m[rows, selected_index]
    return {
        "per_sample_loss": per_sample_loss,
        "ranking_per_sample": ranking,
        "candidate_loss": safe_deficit,
        "candidate_label_cost": (
            config.label_weight * safe_deficit.detach()
        ),
        "safe_mask": safe,
        "safe_progress_candidate_count": sufficient.sum(dim=1),
        "preferred_progress_candidate_count": (
            safe & endpoint_distance_m.ge(config.preferred_progress_m)
        ).sum(dim=1),
        "selected_progress": selected_progress,
        "selected_insufficient_progress": (
            selected_progress.lt(config.minimum_progress_m)
        ),
    }


__all__ = ["ProgressSafetyConfigV1", "progress_safety_objective_v1"]
