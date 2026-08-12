"""Single-contract score supervision for static YOPO candidates.

The score head has one job: rank physically feasible candidates ahead of
infeasible candidates.  Guidance, preventive clearance, and progress only
order candidates *inside* that hard-feasible set.  Lower scores are better.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class FeasibilityScoreConfigV1:
    enabled: bool = False
    required_clearance_m: float = 0.65
    max_speed_mps: float = 6.0
    max_acceleration_mps2: float = 6.0
    infeasible_label_floor: float = 2.0
    regression_weight: float = 0.5
    listwise_weight: float = 2.0
    ranking_weight: float = 2.0
    ranking_margin: float = 1.0
    target_temperature: float = 0.35
    prediction_temperature: float = 1.0
    quality_epsilon: float = 1.0e-6

    @classmethod
    def from_mapping(cls, value):
        value = {} if value is None else dict(value)
        unknown = sorted(set(value) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown feasibility_score keys: {unknown}")
        result = cls(**value)
        result.validate()
        return result

    def validate(self):
        positive = (
            self.required_clearance_m, self.max_speed_mps,
            self.max_acceleration_mps2, self.infeasible_label_floor,
            self.ranking_margin, self.target_temperature,
            self.prediction_temperature, self.quality_epsilon,
        )
        if min(positive) <= 0:
            raise ValueError("feasibility score scales must be positive")
        if min(
            self.regression_weight, self.listwise_weight, self.ranking_weight
        ) < 0:
            raise ValueError("feasibility score weights must be non-negative")
        if self.infeasible_label_floor <= 1.0:
            raise ValueError("infeasible labels must be above feasible labels")


def _masked_unit_interval(values, mask, epsilon):
    """Normalize each row on ``mask``; singleton/constant sets map to zero."""
    large = torch.finfo(values.dtype).max
    low = torch.where(mask, values, large).amin(dim=1, keepdim=True)
    high = torch.where(mask, values, -large).amax(dim=1, keepdim=True)
    span = (high - low).clamp_min(epsilon)
    normalized = (values - low) / span
    return torch.where(mask, normalized.clamp(0.0, 1.0), torch.zeros_like(values))


def feasibility_score_objective_v1(
    predicted_scores,
    quality_cost,
    static_min_distance,
    trajectory_max_speed,
    trajectory_max_acceleration,
    config: FeasibilityScoreConfigV1,
):
    """Return lexicographic labels and an argmin-aligned score loss."""
    config.validate()
    tensors = (
        quality_cost, static_min_distance, trajectory_max_speed,
        trajectory_max_acceleration,
    )
    if any(value.shape != predicted_scores.shape for value in tensors):
        raise ValueError("feasibility score candidate shapes differ")

    hard_feasible = (
        static_min_distance.ge(config.required_clearance_m)
        & trajectory_max_speed.le(config.max_speed_mps)
        & trajectory_max_acceleration.le(config.max_acceleration_mps2)
    ).detach()
    has_feasible = hard_feasible.any(dim=1)
    feasible_quality = _masked_unit_interval(
        quality_cost.detach(), hard_feasible, config.quality_epsilon
    )

    clearance_violation = F.relu(
        config.required_clearance_m - static_min_distance
    ) / config.required_clearance_m
    speed_violation = F.relu(
        trajectory_max_speed / config.max_speed_mps - 1.0
    )
    acceleration_violation = F.relu(
        trajectory_max_acceleration / config.max_acceleration_mps2 - 1.0
    )
    violation = (
        clearance_violation + speed_violation + acceleration_violation
    ).detach()
    infeasible = ~hard_feasible
    violation_rank = _masked_unit_interval(
        violation, infeasible, config.quality_epsilon
    )
    labels = torch.where(
        hard_feasible,
        feasible_quality,
        config.infeasible_label_floor + violation_rank,
    )
    # If every candidate violates a hard constraint, retain a continuous
    # least-violation target instead of presenting fifteen equivalent labels.
    all_infeasible_labels = violation_rank + 0.05 * _masked_unit_interval(
        quality_cost.detach(), infeasible, config.quality_epsilon
    )
    labels = torch.where(has_feasible[:, None], labels, all_infeasible_labels)

    regression = F.smooth_l1_loss(
        predicted_scores, labels, reduction="none"
    ).mean(dim=1)
    target_logits = -labels / config.target_temperature
    # When a hard-feasible action exists, the listwise target has exactly zero
    # probability on every infeasible action.  A large finite sentinel keeps
    # this stable under mixed precision and avoids relying on infinity.
    target_logits = torch.where(
        has_feasible[:, None] & (~hard_feasible),
        torch.full_like(target_logits, -1.0e4),
        target_logits,
    )
    target_probability = target_logits.softmax(dim=1).detach()
    prediction_log_probability = F.log_softmax(
        -predicted_scores / config.prediction_temperature, dim=1
    )
    listwise = -(target_probability * prediction_log_probability).sum(dim=1)

    pair_loss = F.relu(
        predicted_scores[:, :, None] - predicted_scores[:, None, :]
        + config.ranking_margin
    )
    pair_mask = hard_feasible[:, :, None] & (~hard_feasible)[:, None, :]
    pair_count = pair_mask.sum((1, 2))
    ranking = (
        (pair_loss * pair_mask).sum((1, 2))
        / pair_count.clamp(min=1).to(pair_loss.dtype)
    )
    ranking = torch.where(pair_count > 0, ranking, ranking * 0.0)
    per_sample_loss = (
        config.regression_weight * regression
        + config.listwise_weight * listwise
        + config.ranking_weight * ranking
    )

    selected = predicted_scores.argmin(dim=1)
    row = torch.arange(predicted_scores.shape[0], device=predicted_scores.device)
    return {
        "labels": labels.detach(),
        "per_sample_loss": per_sample_loss,
        "regression_per_sample": regression,
        "listwise_per_sample": listwise,
        "ranking_per_sample": ranking,
        "hard_feasible_mask": hard_feasible,
        "hard_feasible_candidate_count": hard_feasible.sum(dim=1),
        "selected_hard_infeasible": (~hard_feasible[row, selected]),
        "has_hard_feasible_candidate": has_feasible,
    }


__all__ = ["FeasibilityScoreConfigV1", "feasibility_score_objective_v1"]
