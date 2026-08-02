"""Differentiable objectives for Phase 8J candidate-set coverage.

The module operates on all 15 candidates.  It deliberately contains no score
selection and therefore cannot train the score branch.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class CoverageObjectiveConfig:
    enabled: bool = False
    dynamic_cvar_weight: float = 0.0
    dynamic_cvar_fraction: float = 4.0 / 15.0
    best_safe_weight: float = 0.0
    softmin_temperature: float = 0.10
    safe_count_weight: float = 0.0
    safe_count_k: int = 2
    safe_count_temperature: float = 0.10
    endpoint_diversity_weight: float = 0.0
    endpoint_margin_normalized: float = 0.025
    trajectory_diversity_weight: float = 0.0
    trajectory_margin_normalized: float = 0.015
    position_scale_m: float = 10.0
    violation_clip_m: float = 3.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]):
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"coverage objective unknown keys: {sorted(unknown)}")
        result = cls(**values)
        result.validate()
        return result

    def validate(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("coverage enabled must be boolean")
        for name in (
            "dynamic_cvar_weight", "best_safe_weight", "safe_count_weight",
            "endpoint_diversity_weight", "trajectory_diversity_weight",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in (
            "dynamic_cvar_fraction", "softmin_temperature",
            "safe_count_temperature", "endpoint_margin_normalized",
            "trajectory_margin_normalized", "position_scale_m", "violation_clip_m",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.dynamic_cvar_fraction > 1:
            raise ValueError("dynamic_cvar_fraction must be <= 1")
        if not isinstance(self.safe_count_k, int) or not 1 <= self.safe_count_k <= 15:
            raise ValueError("safe_count_k must be an integer in [1,15]")


@dataclass(frozen=True)
class CoverageLossOutput:
    total: torch.Tensor
    dynamic_cvar: torch.Tensor
    best_safe: torch.Tensor
    safe_count: torch.Tensor
    endpoint_diversity: torch.Tensor
    trajectory_diversity: torch.Tensor
    joint_violation: torch.Tensor
    soft_safe_count: torch.Tensor


class CandidateCoverageLoss(nn.Module):
    def __init__(self, config: CoverageObjectiveConfig):
        super().__init__()
        config.validate()
        self.config = config

    @staticmethod
    def _pairwise_repulsion(values, margin):
        # values: [B,C,...]; C is fixed at 15, so the bounded O(C^2) work is small.
        batch, candidates = values.shape[:2]
        flat = values.reshape(batch, candidates, -1)
        distances = torch.cdist(flat, flat, p=2) / math.sqrt(flat.shape[-1])
        upper = torch.triu(
            torch.ones(candidates, candidates, dtype=torch.bool, device=values.device),
            diagonal=1,
        )
        return F.relu(margin - distances[:, upper]).square().mean()

    def forward(
        self,
        dynamic_candidate_cost,
        dynamic_min_clearance,
        static_min_clearance,
        trajectories,
        has_dynamic_target,
    ):
        if dynamic_candidate_cost.ndim != 2 or dynamic_candidate_cost.shape[1] != 15:
            raise ValueError("dynamic_candidate_cost must have shape [B,15]")
        expected = dynamic_candidate_cost.shape
        if dynamic_min_clearance.shape != expected or static_min_clearance.shape != expected:
            raise ValueError("clearance tensors must have shape [B,15]")
        if trajectories.shape[:2] != expected or trajectories.shape[-1] != 3:
            raise ValueError("trajectories must have shape [B,15,T,3]")
        target = has_dynamic_target.to(
            device=dynamic_candidate_cost.device, dtype=dynamic_candidate_cost.dtype
        )
        if target.shape != (expected[0],):
            raise ValueError("has_dynamic_target must have shape [B]")

        count = max(1, math.ceil(15 * self.config.dynamic_cvar_fraction))
        dynamic_cvar_per_sample = dynamic_candidate_cost.topk(
            count, dim=1, largest=True
        ).values.mean(1)
        # Multiplication by the target mask makes no-target dynamic coverage
        # exactly zero, including when the batch contains only no-target rows.
        dynamic_cvar = (dynamic_cvar_per_sample * target).sum() / target.sum().clamp_min(1)

        dynamic_violation = F.relu(-dynamic_min_clearance).clamp_max(
            self.config.violation_clip_m
        )
        dynamic_violation = dynamic_violation * target[:, None]
        static_violation = F.relu(-static_min_clearance).clamp_max(
            self.config.violation_clip_m
        )
        joint_violation = dynamic_violation + static_violation

        temperature = self.config.softmin_temperature
        # Boltzmann-weighted soft minimum is non-negative, exactly zero when
        # all candidates are safe, and approaches the true minimum as tau
        # decreases without the candidate-count offset of raw log-sum-exp.
        weights = torch.softmax(-joint_violation / temperature, dim=1)
        soft_best = (weights * joint_violation).sum(1)
        best_safe = soft_best.mean()

        soft_safe_count = torch.sigmoid(
            -joint_violation / self.config.safe_count_temperature
        ).sum(1)
        # sigmoid(0)=0.5; scaling by two makes a boundary-safe candidate count
        # as one while retaining continuous gradients on the unsafe side.
        soft_safe_count = 2.0 * soft_safe_count
        safe_count = F.relu(
            dynamic_candidate_cost.new_tensor(float(self.config.safe_count_k))
            - soft_safe_count
        ).square().mean()

        normalized = trajectories / self.config.position_scale_m
        endpoint_diversity = self._pairwise_repulsion(
            normalized[:, :, -1], self.config.endpoint_margin_normalized
        )
        trajectory_diversity = self._pairwise_repulsion(
            normalized, self.config.trajectory_margin_normalized
        )
        total = (
            self.config.dynamic_cvar_weight * dynamic_cvar
            + self.config.best_safe_weight * best_safe
            + self.config.safe_count_weight * safe_count
            + self.config.endpoint_diversity_weight * endpoint_diversity
            + self.config.trajectory_diversity_weight * trajectory_diversity
        )
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("candidate coverage loss contains NaN/Inf")
        return CoverageLossOutput(
            total=total,
            dynamic_cvar=dynamic_cvar,
            best_safe=best_safe,
            safe_count=safe_count,
            endpoint_diversity=endpoint_diversity,
            trajectory_diversity=trajectory_diversity,
            joint_violation=joint_violation,
            soft_safe_count=soft_safe_count,
        )
