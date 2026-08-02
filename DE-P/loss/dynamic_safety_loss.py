"""Time-aligned future collision cost against moving obstacles."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .dynamic_types import DynamicLossConfig, DynamicObstacleBatch, RiskMetricsConfig


@dataclass(frozen=True)
class DynamicSafetyDiagnostics:
    min_dynamic_distance: torch.Tensor
    risky_trajectory_ratio: torch.Tensor
    dynamic_obstacle_count: torch.Tensor
    constant_velocity_cost_delta: torch.Tensor
    candidate_cost: torch.Tensor
    candidate_min_distance: torch.Tensor
    candidate_min_clearance: torch.Tensor
    candidate_time_of_min_clearance: torch.Tensor
    candidate_violation_fraction: torch.Tensor
    candidate_safe_mask: torch.Tensor
    candidate_high_risk_mask: torch.Tensor
    risk_by_time: torch.Tensor


class DynamicCollisionLoss(nn.Module):
    def __init__(self, sampler, config=None, risk_metrics_config=None):
        super().__init__()
        self.sampler = sampler
        self.config = config or DynamicLossConfig.from_global_config()
        self.config.validate()
        self.risk_metrics = risk_metrics_config or RiskMetricsConfig.from_global_config()
        self.risk_metrics.validate()

    def _empty_diagnostics(self, trajectories, obstacle_count, zero):
        batch, trajectory_count, times = trajectories.shape[:3]
        maximum = torch.finfo(trajectories.dtype).max
        return DynamicSafetyDiagnostics(
            min_dynamic_distance=torch.full((batch,), maximum, device=trajectories.device,
                                            dtype=trajectories.dtype),
            risky_trajectory_ratio=torch.zeros(batch, device=trajectories.device,
                                                dtype=trajectories.dtype),
            dynamic_obstacle_count=obstacle_count,
            constant_velocity_cost_delta=torch.zeros_like(zero.reshape(batch, trajectory_count)),
            candidate_cost=torch.zeros_like(zero.reshape(batch, trajectory_count)),
            candidate_min_distance=torch.full((batch, trajectory_count), maximum,
                                              device=trajectories.device, dtype=trajectories.dtype),
            candidate_min_clearance=torch.full((batch, trajectory_count), maximum,
                                               device=trajectories.device, dtype=trajectories.dtype),
            candidate_time_of_min_clearance=torch.full((batch, trajectory_count), -1.0,
                                                       device=trajectories.device,
                                                       dtype=trajectories.dtype),
            candidate_violation_fraction=torch.zeros_like(zero.reshape(batch, trajectory_count)),
            candidate_safe_mask=torch.ones(batch, trajectory_count, dtype=torch.bool,
                                           device=trajectories.device),
            candidate_high_risk_mask=torch.zeros(batch, trajectory_count, dtype=torch.bool,
                                                device=trajectories.device),
            risk_by_time=torch.zeros(batch, trajectory_count, times,
                                     device=trajectories.device, dtype=trajectories.dtype),
        )

    def forward(self, fixed_derivatives, predicted_derivatives,
                obstacles: DynamicObstacleBatch):
        batch = obstacles.batch_size
        trajectories, sample_times = self.sampler.grouped(
            fixed_derivatives, predicted_derivatives, batch
        )
        trajectory_count = trajectories.shape[1]
        zero = trajectories.sum(dim=(-1, -2)) * 0.0
        observable = (obstacles.observable_mask if obstacles.observable_mask is not None
                      else obstacles.valid_mask)
        active = obstacles.valid_mask & obstacles.dynamic_mask & observable
        obstacle_count = active.sum(dim=1)
        if obstacles.max_obstacles == 0 or not bool(active.any()):
            diagnostics = self._empty_diagnostics(trajectories, obstacle_count, zero)
            return zero.reshape(batch, trajectory_count), diagnostics

        current_age_absolute = (
            obstacles.sample_timestamps[:, None] - obstacles.track_timestamps
        )
        if bool(((current_age_absolute < -1e-6) & active).any()):
            raise ValueError("valid dynamic obstacle has a future track_timestamp")
        active = active & (current_age_absolute >= 0) & (
            current_age_absolute <= self.config.max_track_age
        )
        if not bool(active.any()):
            diagnostics = self._empty_diagnostics(trajectories, active.sum(dim=1), zero)
            return zero.reshape(batch, trajectory_count), diagnostics

        # Subtract absolute stamps in float64, then use the small relative time
        # in the trajectory dtype so the geometry path remains float32.
        current_age = current_age_absolute.to(dtype=trajectories.dtype)
        delta_time = current_age[:, :, None] + sample_times[None, None, :]
        constant_velocity_positions = (
            obstacles.positions_world[:, :, None, :]
            + obstacles.velocities_world[:, :, None, :] * delta_time[:, :, :, None]
        )
        future_valid = active[:, :, None].expand(-1, -1, sample_times.numel())
        if self.config.target_source == "recorded_future_gt":
            if obstacles.future_positions_world is None:
                raise ValueError("recorded_future_gt requires future supervision")
            if obstacles.future_positions_world.shape[2] != sample_times.numel():
                raise ValueError("future GT sample count does not match trajectory sampler")
            expected_times = (
                obstacles.sample_timestamps[:, None]
                + sample_times.to(dtype=obstacles.sample_timestamps.dtype)[None, :]
            )
            if not bool(torch.allclose(obstacles.future_timestamps, expected_times, atol=1e-4, rtol=0)):
                raise ValueError("future GT timestamps do not align with trajectory sampler")
            obstacle_positions = obstacles.future_positions_world
            future_valid = future_valid & obstacles.future_valid_mask
        else:
            obstacle_positions = constant_velocity_positions

        def distances(positions):
            return torch.linalg.vector_norm(
                trajectories[:, :, None, :, :] - positions[:, None, :, :, :], dim=-1
            )
        distance = distances(obstacle_positions)

        # Padded/inactive actors must not enter covariance algebra.  In
        # particular, sqrt(clamp_min(0)) has an infinite derivative at zero
        # and previously poisoned estimated-context backward passes even
        # though the actor was masked later.
        active_covariance = torch.where(
            active[:, :, None, None],
            obstacles.position_covariances,
            torch.zeros_like(obstacles.position_covariances),
        )
        covariance_eigenvalue = torch.linalg.eigvalsh(
            active_covariance
        ).amax(dim=-1).clamp_min(0)
        propagated_variance = (
            covariance_eigenvalue[:, :, None]
            + self.config.covariance_growth_rate * delta_time.square()
        )
        variance_floor = torch.finfo(trajectories.dtype).eps ** 2
        propagated_standard_deviation = (
            (propagated_variance + variance_floor).sqrt()
            - propagated_variance.new_tensor(variance_floor).sqrt()
        )
        obstacle_radius = torch.where(
            obstacles.radii > 0,
            obstacles.radii,
            torch.full_like(obstacles.radii, self.config.default_obstacle_radius),
        )
        safe_radius = (
            self.config.uav_radius + obstacle_radius[:, :, None]
            + self.config.covariance_sigma * propagated_standard_deviation
        )
        violation = safe_radius[:, None, :, :] - distance
        risk = F.softplus(violation / self.config.temperature).square()
        risk = risk * obstacles.confidence[:, None, :, None]
        risk = risk.masked_fill(~future_valid[:, None, :, :], 0.0)
        # Max is invariant to padded/duplicated obstacles and cannot grow with M.
        risk_by_time = risk.amax(dim=2)
        discount = torch.exp(-self.config.time_discount * sample_times)
        trajectory_cost = (
            risk_by_time * discount[None, None, :]
        ).sum(dim=-1) / discount.sum().clamp_min(torch.finfo(discount.dtype).eps)

        valid_distance = distance.masked_fill(~future_valid[:, None, :, :], float("inf"))
        valid_clearance = (distance - safe_radius[:, None, :, :]).masked_fill(
            ~future_valid[:, None, :, :], float("inf")
        )
        flat_clearance = valid_clearance.flatten(2)
        candidate_min_clearance, flat_min_index = flat_clearance.min(dim=2)
        candidate_min_distance = valid_distance.flatten(2).min(dim=2).values
        time_index = flat_min_index.remainder(sample_times.numel())
        candidate_min_time = sample_times[time_index]
        has_future = future_valid.any(dim=(1, 2))[:, None]
        maximum = torch.finfo(trajectories.dtype).max
        candidate_min_clearance = torch.where(
            has_future, candidate_min_clearance, torch.full_like(candidate_min_clearance, maximum)
        )
        candidate_min_distance = torch.where(
            has_future, candidate_min_distance, torch.full_like(candidate_min_distance, maximum)
        )
        candidate_min_time = torch.where(
            has_future, candidate_min_time, torch.full_like(candidate_min_time, -1.0)
        )
        valid_count = future_valid.sum(dim=(1, 2)).clamp_min(1).to(trajectories.dtype)[:, None]
        violation_count = ((violation > 0) & future_valid[:, None, :, :]).sum(dim=(2, 3))
        candidate_violation_fraction = violation_count.to(trajectories.dtype) / valid_count
        candidate_safe = candidate_min_clearance >= self.risk_metrics.clearance_threshold
        candidate_high_risk = (
            (candidate_min_clearance < self.risk_metrics.hard_window_threshold)
            | (trajectory_cost > self.risk_metrics.cost_epsilon)
        ) & has_future
        min_distance = valid_distance.amin(dim=(1, 2, 3))
        min_distance = torch.where(
            future_valid.any(dim=(1, 2)), min_distance,
            torch.full_like(min_distance, torch.finfo(trajectories.dtype).max),
        )
        risky_ratio = (violation.masked_fill(~future_valid[:, None, :, :], float("-inf"))
                       .amax(dim=(2, 3)) > 0).float().mean(dim=1)
        label_delta = torch.zeros_like(trajectory_cost)
        if self.config.target_source == "recorded_future_gt":
            cv_distance = distances(constant_velocity_positions)
            cv_violation = safe_radius[:, None, :, :] - cv_distance
            cv_risk = F.softplus(cv_violation / self.config.temperature).square()
            cv_risk = cv_risk * obstacles.confidence[:, None, :, None]
            cv_risk = cv_risk.masked_fill(~future_valid[:, None, :, :], 0.0)
            cv_by_time = cv_risk.amax(dim=2)
            cv_cost = (cv_by_time * discount[None, None, :]).sum(-1) / discount.sum().clamp_min(
                torch.finfo(discount.dtype).eps
            )
            label_delta = (trajectory_cost - cv_cost).abs()
        diagnostics = DynamicSafetyDiagnostics(
            min_dynamic_distance=min_distance,
            risky_trajectory_ratio=risky_ratio,
            dynamic_obstacle_count=active.sum(dim=1),
            constant_velocity_cost_delta=label_delta,
            candidate_cost=trajectory_cost,
            candidate_min_distance=candidate_min_distance,
            candidate_min_clearance=candidate_min_clearance,
            candidate_time_of_min_clearance=candidate_min_time,
            candidate_violation_fraction=candidate_violation_fraction,
            candidate_safe_mask=candidate_safe,
            candidate_high_risk_mask=candidate_high_risk,
            risk_by_time=risk_by_time,
        )
        if not bool(torch.isfinite(trajectory_cost).all()):
            raise FloatingPointError("dynamic collision cost contains NaN/Inf")
        return trajectory_cost, diagnostics
