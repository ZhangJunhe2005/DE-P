"""Strongly typed labels and configuration for future dynamic collision loss."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch


@dataclass(frozen=True)
class DynamicLossConfig:
    enabled: bool
    weight: float
    uav_radius: float
    default_obstacle_radius: float
    covariance_sigma: float
    covariance_growth_rate: float
    temperature: float
    time_discount: float
    max_track_age: float
    obstacle_aggregation: str
    eval_points: int
    target_source: str

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]):
        fields = set(cls.__dataclass_fields__)
        unknown, missing = set(values) - fields, fields - set(values)
        if unknown or missing:
            raise ValueError(f"dynamic_loss config unknown={sorted(unknown)}, missing={sorted(missing)}")
        result = cls(**values)
        result.validate()
        return result

    @classmethod
    def from_global_config(cls):
        from config.config import cfg
        return cls.from_mapping(cfg["dynamic_loss"])

    def validate(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("dynamic_loss.enabled must be boolean")
        nonnegative = ("weight", "uav_radius", "default_obstacle_radius",
                       "covariance_sigma", "covariance_growth_rate", "time_discount",
                       "max_track_age")
        for name in nonnegative:
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"dynamic_loss.{name} must be finite and non-negative")
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("dynamic_loss.temperature must be finite and positive")
        if not isinstance(self.eval_points, int) or self.eval_points <= 0:
            raise ValueError("dynamic_loss.eval_points must be a positive integer")
        if self.obstacle_aggregation not in {"max"}:
            raise ValueError("dynamic_loss.obstacle_aggregation currently supports only 'max'")
        if self.target_source not in {"recorded_future_gt", "constant_velocity"}:
            raise ValueError("dynamic_loss.target_source is invalid")


@dataclass(frozen=True)
class RiskMetricsConfig:
    """Auditable, physical thresholds used only for reporting and Gates.

    ``cost_epsilon`` is the unweighted, per-time softplus-squared cost at the
    collision boundary for unit confidence: ``softplus(0)^2 = log(2)^2``.
    ``hard_window_threshold`` is a positive clearance margin, not a collision
    threshold.  The default 0.15 m is half the configured 0.30 m UAV radius.
    """

    clearance_threshold: float
    cost_epsilon: float
    cvar_fraction: float
    hard_window_threshold: float

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]):
        fields = set(cls.__dataclass_fields__)
        unknown, missing = set(values) - fields, fields - set(values)
        if unknown or missing:
            raise ValueError(f"risk_metrics config unknown={sorted(unknown)}, missing={sorted(missing)}")
        result = cls(**values)
        result.validate()
        return result

    @classmethod
    def from_global_config(cls):
        from config.config import cfg
        return cls.from_mapping(cfg["risk_metrics"])

    def validate(self):
        for name in ("clearance_threshold", "cost_epsilon", "hard_window_threshold"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"risk_metrics.{name} must be finite")
        if self.cost_epsilon < 0:
            raise ValueError("risk_metrics.cost_epsilon must be non-negative")
        if not np.isfinite(self.cvar_fraction) or not 0 < self.cvar_fraction <= 1:
            raise ValueError("risk_metrics.cvar_fraction must be in (0,1]")


@dataclass(frozen=True)
class DynamicObjectiveConfig:
    """Candidate aggregation for training; reporting always retains raw mean."""

    mean_coefficient: float
    cvar_coefficient: float
    cvar_fraction: float
    reference_scale: float
    gradient_strategy: str
    pcgrad_other_norm_ratio: float
    pcgrad_risk_gate: bool = False
    pcgrad_activation_clearance: float = 0.15
    pcgrad_dynamic_grad_epsilon: float = 1e-12
    pcgrad_ratio_switch_step: int = -1
    pcgrad_late_other_norm_ratio: float = 0.25

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]):
        fields = set(cls.__dataclass_fields__)
        unknown = set(values) - fields
        required = {
            "mean_coefficient", "cvar_coefficient", "cvar_fraction",
            "reference_scale", "gradient_strategy", "pcgrad_other_norm_ratio",
        }
        missing = required - set(values)
        if unknown or missing:
            raise ValueError(
                f"dynamic_objective config unknown={sorted(unknown)}, missing={sorted(missing)}"
            )
        result = cls(**values)
        result.validate()
        return result

    @classmethod
    def from_global_config(cls):
        from config.config import cfg
        return cls.from_mapping(cfg["dynamic_objective"])

    def validate(self):
        for name in ("mean_coefficient", "cvar_coefficient"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"dynamic_objective.{name} must be finite and non-negative")
        if self.mean_coefficient <= 0:
            raise ValueError("dynamic_objective.mean_coefficient must remain positive")
        if not np.isfinite(self.cvar_fraction) or not 0 < self.cvar_fraction <= 1:
            raise ValueError("dynamic_objective.cvar_fraction must be in (0,1]")
        if not np.isfinite(self.reference_scale) or self.reference_scale <= 0:
            raise ValueError("dynamic_objective.reference_scale must be finite and positive")
        if self.gradient_strategy not in {"weighted_sum", "dynamic_priority_pcgrad"}:
            raise ValueError("dynamic_objective.gradient_strategy is invalid")
        if (not np.isfinite(self.pcgrad_other_norm_ratio)
                or self.pcgrad_other_norm_ratio <= 0):
            raise ValueError("dynamic_objective.pcgrad_other_norm_ratio must be positive")
        if not isinstance(self.pcgrad_risk_gate, bool):
            raise ValueError("dynamic_objective.pcgrad_risk_gate must be boolean")
        if (not np.isfinite(self.pcgrad_activation_clearance)
                or self.pcgrad_activation_clearance < 0):
            raise ValueError(
                "dynamic_objective.pcgrad_activation_clearance must be finite and non-negative"
            )
        if (not np.isfinite(self.pcgrad_dynamic_grad_epsilon)
                or self.pcgrad_dynamic_grad_epsilon < 0):
            raise ValueError(
                "dynamic_objective.pcgrad_dynamic_grad_epsilon must be finite and non-negative"
            )
        if (not isinstance(self.pcgrad_ratio_switch_step, int)
                or self.pcgrad_ratio_switch_step < -1):
            raise ValueError(
                "dynamic_objective.pcgrad_ratio_switch_step must be -1 or a non-negative integer"
            )
        if (not np.isfinite(self.pcgrad_late_other_norm_ratio)
                or self.pcgrad_late_other_norm_ratio <= 0):
            raise ValueError(
                "dynamic_objective.pcgrad_late_other_norm_ratio must be positive"
            )


@dataclass(frozen=True)
class DynamicObstacleBatch:
    positions_world: torch.Tensor
    velocities_world: torch.Tensor
    position_covariances: torch.Tensor
    radii: torch.Tensor
    track_timestamps: torch.Tensor
    sample_timestamps: torch.Tensor
    confidence: torch.Tensor
    valid_mask: torch.Tensor
    dynamic_mask: torch.Tensor
    observable_mask: torch.Tensor | None = None
    ever_observed_in_history: torch.Tensor | None = None
    future_positions_world: torch.Tensor | None = None
    future_valid_mask: torch.Tensor | None = None
    future_visibility_mask: torch.Tensor | None = None
    future_timestamps: torch.Tensor | None = None

    def __post_init__(self):
        positions = self.positions_world
        if not torch.is_tensor(positions) or positions.ndim != 3 or positions.shape[-1] != 3:
            raise ValueError("positions_world must have shape [B,M,3]")
        batch, obstacles, _ = positions.shape
        expected = {
            "velocities_world": (batch, obstacles, 3),
            "position_covariances": (batch, obstacles, 3, 3),
            "radii": (batch, obstacles),
            "track_timestamps": (batch, obstacles),
            "sample_timestamps": (batch,),
            "confidence": (batch, obstacles),
            "valid_mask": (batch, obstacles),
            "dynamic_mask": (batch, obstacles),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if not torch.is_tensor(value) or tuple(value.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}")
        for name in ("positions_world", "velocities_world", "position_covariances",
                     "radii", "track_timestamps", "sample_timestamps", "confidence"):
            if not bool(torch.isfinite(getattr(self, name)).all()):
                raise ValueError(f"{name} contains NaN/Inf")
        if self.valid_mask.dtype != torch.bool or self.dynamic_mask.dtype != torch.bool:
            raise TypeError("valid_mask and dynamic_mask must be boolean")
        if bool((self.radii < 0).any()):
            raise ValueError("obstacle radii must be non-negative")
        if bool((self.confidence < 0).any()) or bool((self.confidence > 1).any()):
            raise ValueError("confidence must be in [0,1]")
        covariance = self.position_covariances
        if not bool(torch.allclose(covariance, covariance.transpose(-1, -2), atol=1e-6)):
            raise ValueError("position_covariances must be symmetric")
        if covariance.numel() and bool((torch.linalg.eigvalsh(covariance) < -1e-6).any()):
            raise ValueError("position_covariances must be positive semidefinite")
        for name in ("observable_mask", "ever_observed_in_history"):
            value = getattr(self, name)
            if value is not None and (not torch.is_tensor(value) or tuple(value.shape) != (batch, obstacles)
                                      or value.dtype != torch.bool):
                raise ValueError(f"{name} must be boolean [B,M]")
        future = (self.future_positions_world, self.future_valid_mask,
                  self.future_visibility_mask, self.future_timestamps)
        if any(value is not None for value in future):
            if not all(value is not None for value in future):
                raise ValueError("future supervision fields must be supplied together")
            if self.future_positions_world.ndim != 4:
                raise ValueError("future_positions_world must have shape [B,M,N,3]")
            points = self.future_positions_world.shape[2]
            expected_future = {
                "future_positions_world": (batch, obstacles, points, 3),
                "future_valid_mask": (batch, obstacles, points),
                "future_visibility_mask": (batch, obstacles, points),
                "future_timestamps": (batch, points),
            }
            for name, shape in expected_future.items():
                value = getattr(self, name)
                if tuple(value.shape) != shape:
                    raise ValueError(f"{name} must have shape {shape}")
            if self.future_valid_mask.dtype != torch.bool or self.future_visibility_mask.dtype != torch.bool:
                raise TypeError("future masks must be boolean")
            if not bool(torch.isfinite(self.future_positions_world).all()) or not bool(torch.isfinite(self.future_timestamps).all()):
                raise ValueError("future supervision contains NaN/Inf")
            if points > 1 and not bool((torch.diff(self.future_timestamps, dim=1) > 0).all()):
                raise ValueError("future_timestamps must be strictly increasing")

    @property
    def batch_size(self):
        return self.positions_world.shape[0]

    @property
    def max_obstacles(self):
        return self.positions_world.shape[1]

    def to(self, device):
        return DynamicObstacleBatch(**{
            name: (getattr(self, name).to(device) if getattr(self, name) is not None else None)
            for name in self.__dataclass_fields__
        })

    @classmethod
    def empty(cls, batch_size, timestamp=0.0, device="cpu"):
        return cls(
            positions_world=torch.zeros(batch_size, 0, 3, device=device),
            velocities_world=torch.zeros(batch_size, 0, 3, device=device),
            position_covariances=torch.zeros(batch_size, 0, 3, 3, device=device),
            radii=torch.zeros(batch_size, 0, device=device),
            track_timestamps=torch.zeros(batch_size, 0, device=device),
            sample_timestamps=torch.full((batch_size,), float(timestamp), device=device),
            confidence=torch.zeros(batch_size, 0, device=device),
            valid_mask=torch.zeros(batch_size, 0, dtype=torch.bool, device=device),
            dynamic_mask=torch.zeros(batch_size, 0, dtype=torch.bool, device=device),
            observable_mask=torch.zeros(batch_size, 0, dtype=torch.bool, device=device),
            ever_observed_in_history=torch.zeros(batch_size, 0, dtype=torch.bool, device=device),
        )
