"""Goal-directed mobility supervision for Route-A static YOPO.

V1 treated ``||p_end - p_start||`` as progress.  That makes lateral motion
and motion directly away from the goal indistinguishable from useful flight.
V2 keeps path length as an anti-hover diagnostic, but optimises the signed
projection of candidate displacement onto the current local-goal direction.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class GoalProgressConfigV2:
    enabled: bool = False
    minimum_path_length_m: float = 1.0
    minimum_goal_progress_m: float = 0.25
    preferred_goal_progress_m: float = 3.0
    maximum_reverse_progress_m: float = 0.05
    softplus_temperature_m: float = 0.25
    loss_weight: float = 1.0
    reverse_weight: float = 2.0
    ranking_weight: float = 1.0
    ranking_margin: float = 0.75
    label_weight: float = 1.0
    minimum_goal_progress_candidates: int = 3

    @classmethod
    def from_mapping(cls, value):
        value = {} if value is None else dict(value)
        unknown = sorted(set(value) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown goal_progress_v2 keys: {unknown}")
        result = cls(**value)
        result.validate()
        return result

    def validate(self):
        positive = (
            self.minimum_path_length_m,
            self.preferred_goal_progress_m,
            self.softplus_temperature_m,
            self.ranking_margin,
        )
        if min(positive) <= 0:
            raise ValueError("goal progress distances/scales must be positive")
        if self.minimum_goal_progress_m < 0:
            raise ValueError("minimum goal progress must be non-negative")
        if self.preferred_goal_progress_m < self.minimum_goal_progress_m:
            raise ValueError("preferred goal progress must cover the minimum")
        if self.maximum_reverse_progress_m < 0:
            raise ValueError("reverse tolerance must be non-negative")
        if min(
            self.loss_weight, self.reverse_weight,
            self.ranking_weight, self.label_weight,
        ) < 0:
            raise ValueError("goal progress weights must be non-negative")
        if self.minimum_goal_progress_candidates < 1:
            raise ValueError("goal progress candidate count must be positive")


def goal_directed_progress_objective_v2(
    predicted_scores,
    endpoint_displacement_body,
    goal_body,
    static_clear_mask,
    config: GoalProgressConfigV2,
):
    """Preserve useful motion while leaving hard safety independently binding.

    Candidate-generation gradients are applied to statically clear proposals,
    including proposals that are temporarily outside the hardware envelope.
    This is intentional: the simultaneous kinodynamic loss can then slow or
    time-stretch a useful direction instead of making every proposal shorter.
    """
    config.validate()
    if endpoint_displacement_body.ndim != 3 \
            or endpoint_displacement_body.shape[2] != 3:
        raise ValueError("candidate displacement must be [B,N,3]")
    batch_size, candidate_count, _ = endpoint_displacement_body.shape
    if predicted_scores.shape != (batch_size, candidate_count):
        raise ValueError("goal-progress score shape differs from candidates")
    if goal_body.shape != (batch_size, 3):
        raise ValueError("goal body vector must be [B,3]")
    if static_clear_mask.shape != predicted_scores.shape \
            or static_clear_mask.dtype != torch.bool:
        raise ValueError("static clear mask must be boolean [B,N]")

    path_length = endpoint_displacement_body.norm(dim=2)
    goal_direction = goal_body / goal_body.norm(
        dim=1, keepdim=True
    ).clamp_min(1.0e-6)
    signed_goal_progress = (
        endpoint_displacement_body * goal_direction[:, None, :]
    ).sum(dim=2)
    goal_alignment = signed_goal_progress / path_length.clamp_min(1.0e-6)

    preferred = torch.as_tensor(
        config.preferred_goal_progress_m,
        device=signed_goal_progress.device,
        dtype=signed_goal_progress.dtype,
    )
    progress_deficit = (
        F.softplus(
            (preferred - signed_goal_progress)
            / config.softplus_temperature_m
        )
        * config.softplus_temperature_m
        / preferred
    ).square()
    reverse_limit = torch.as_tensor(
        -config.maximum_reverse_progress_m,
        device=signed_goal_progress.device,
        dtype=signed_goal_progress.dtype,
    )
    reverse_cost = F.relu(
        reverse_limit - signed_goal_progress
    ).square() / preferred.square()
    candidate_loss = torch.where(
        static_clear_mask,
        progress_deficit + config.reverse_weight * reverse_cost,
        torch.zeros_like(progress_deficit),
    )

    coverage_count = min(
        config.minimum_goal_progress_candidates, candidate_count
    )
    sentinel = torch.full_like(candidate_loss, 1.0e6)
    masked = torch.where(static_clear_mask, candidate_loss, sentinel)
    values, indices = masked.topk(coverage_count, dim=1, largest=False)
    selected_clear = static_clear_mask.gather(1, indices)
    selected_values = torch.where(
        selected_clear, values, torch.zeros_like(values)
    )
    selected_count = selected_clear.sum(dim=1)
    per_sample_loss = selected_values.sum(dim=1) / selected_count.clamp(
        min=1
    ).to(selected_values.dtype)
    per_sample_loss = torch.where(
        selected_count > 0, per_sample_loss, per_sample_loss * 0.0
    )

    useful = (
        static_clear_mask
        & path_length.ge(config.minimum_path_length_m)
        & signed_goal_progress.ge(config.minimum_goal_progress_m)
    )
    non_useful = static_clear_mask & ~useful
    pair_loss = F.relu(
        predicted_scores[:, :, None] - predicted_scores[:, None, :]
        + config.ranking_margin
    )
    pair_mask = useful[:, :, None] & non_useful[:, None, :]
    pair_count = pair_mask.sum((1, 2))
    ranking = (
        (pair_loss * pair_mask).sum((1, 2))
        / pair_count.clamp(min=1).to(pair_loss.dtype)
    )
    ranking = torch.where(pair_count > 0, ranking, ranking * 0.0)

    selected_index = predicted_scores.argmin(dim=1)
    rows = torch.arange(batch_size, device=predicted_scores.device)
    selected_path_length = path_length[rows, selected_index]
    selected_goal_progress = signed_goal_progress[rows, selected_index]
    selected_alignment = goal_alignment[rows, selected_index]
    return {
        "per_sample_loss": per_sample_loss,
        "ranking_per_sample": ranking,
        "candidate_loss": candidate_loss,
        "candidate_label_cost": config.label_weight * candidate_loss.detach(),
        "path_length": path_length,
        "signed_goal_progress": signed_goal_progress,
        "goal_alignment": goal_alignment,
        "useful_mask": useful,
        "goal_progress_candidate_count": useful.sum(dim=1),
        "preferred_goal_progress_candidate_count": (
            static_clear_mask
            & signed_goal_progress.ge(config.preferred_goal_progress_m)
        ).sum(dim=1),
        "selected_path_length": selected_path_length,
        "selected_goal_progress": selected_goal_progress,
        "selected_goal_alignment": selected_alignment,
        "selected_insufficient_goal_progress": (
            selected_path_length.lt(config.minimum_path_length_m)
            | selected_goal_progress.lt(config.minimum_goal_progress_m)
        ),
        "selected_reverse": selected_goal_progress.lt(
            -config.maximum_reverse_progress_m
        ),
        # Compatibility aliases keep the formal trainer's historical metric
        # plumbing readable while V4.2.8 adds the signed metrics below.
        "safe_progress_candidate_count": useful.sum(dim=1),
        "preferred_progress_candidate_count": (
            static_clear_mask
            & signed_goal_progress.ge(config.preferred_goal_progress_m)
        ).sum(dim=1),
        "selected_progress": selected_goal_progress,
        "selected_insufficient_progress": (
            selected_path_length.lt(config.minimum_path_length_m)
            | selected_goal_progress.lt(config.minimum_goal_progress_m)
        ),
    }


__all__ = ["GoalProgressConfigV2", "goal_directed_progress_objective_v2"]
