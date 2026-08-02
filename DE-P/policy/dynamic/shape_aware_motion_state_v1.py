"""Reference-stable bounded sliding-window motion estimation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Tuple

import numpy as np


MOTION_STATE_VERSION = "shape_aware_motion_state_v1"


class MotionObservability(str, Enum):
    MOTION_INITIALIZING = "MOTION_INITIALIZING"
    MOTION_OBSERVABLE = "MOTION_OBSERVABLE"
    MOTION_WEAKLY_OBSERVABLE = "MOTION_WEAKLY_OBSERVABLE"
    MOTION_AMBIGUOUS = "MOTION_AMBIGUOUS"
    SUPPORT_PROPAGATION_ONLY = "SUPPORT_PROPAGATION_ONLY"
    MOTION_EXPIRED = "MOTION_EXPIRED"
    REFERENCE_TRANSITION = "REFERENCE_TRANSITION"


@dataclass(frozen=True)
class MotionEstimateV1:
    reference_mode: str
    timestamp: float
    position_world: np.ndarray
    velocity_world: np.ndarray
    velocity_covariance_world: np.ndarray
    velocity_interval_mps: Tuple[np.ndarray, np.ndarray]
    regression_residual_m: float
    effective_history: int
    outlier_count: int
    observability: MotionObservability
    last_direct_timestamp: float
    runtime_gt_used: bool = False

    def __post_init__(self):
        for name, shape in (
            ("position_world", (3,)), ("velocity_world", (3,)),
            ("velocity_covariance_world", (3, 3)),
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"invalid {name}")
            object.__setattr__(self, name, value.copy())
        lower, upper = (
            np.asarray(self.velocity_interval_mps[0], dtype=np.float64),
            np.asarray(self.velocity_interval_mps[1], dtype=np.float64),
        )
        if lower.shape != (3,) or upper.shape != (3,) or np.any(lower > upper):
            raise ValueError("invalid velocity interval")
        object.__setattr__(
            self, "velocity_interval_mps", (lower.copy(), upper.copy())
        )
        if self.runtime_gt_used:
            raise ValueError("runtime GT is forbidden")


def _weighted_line(times, positions, quality, delta, iterations):
    relative = times-times[-1]
    design = np.column_stack((np.ones(len(times)), relative))
    weights = np.asarray(quality, dtype=np.float64).copy()
    solution = np.zeros((2, 3))
    residual = np.zeros_like(positions)
    for _ in range(iterations):
        root = np.sqrt(np.maximum(weights, 1e-6))
        solution, *_ = np.linalg.lstsq(
            design*root[:, None], positions*root[:, None], rcond=None
        )
        residual = positions-design@solution
        norm = np.linalg.norm(residual, axis=1)
        huber = np.minimum(1., float(delta)/np.maximum(norm, 1e-12))
        weights = np.asarray(quality)*huber
    velocity = solution[1]
    residual_norm = np.linalg.norm(residual, axis=1)
    dof = max(len(times)-2, 1)
    variance = max(float(np.sum(residual_norm**2)/dof), 1e-6)
    information = design.T@(weights[:, None]*design)
    inverse = np.linalg.pinv(information)
    covariance = np.eye(3)*max(variance*inverse[1, 1], 1e-5)
    return (
        solution[0], velocity, covariance,
        float(np.sqrt(np.mean(residual_norm**2))),
        int(np.sum(residual_norm > delta)),
    )


class StableReferenceMotionEstimatorV1:
    """One estimator is bound permanently to one track/generation/shape."""

    version = MOTION_STATE_VERSION

    def __init__(
        self, reference_mode, minimum_history=2, maximum_history=5,
        maximum_history_age_s=.6, huber_delta_m=.08,
        robust_iterations=2, maximum_speed_mps=4.,
        minimum_dt_s=.04,
    ):
        if not 2 <= minimum_history <= maximum_history <= 5:
            raise ValueError("history must satisfy 2 <= min <= max <= 5")
        if robust_iterations not in (1, 2):
            raise ValueError("robust iterations must be bounded to 1 or 2")
        self.reference_mode = str(reference_mode)
        self.minimum_history = int(minimum_history)
        self.maximum_history_age_s = float(maximum_history_age_s)
        self.huber_delta_m = float(huber_delta_m)
        self.robust_iterations = int(robust_iterations)
        self.maximum_speed_mps = float(maximum_speed_mps)
        self.minimum_dt_s = float(minimum_dt_s)
        self.history: Deque = deque(maxlen=int(maximum_history))
        self.last_estimate = None

    def reset(self):
        self.history.clear()
        self.last_estimate = None

    def update(
        self, timestamp, position_world, *,
        geometry_uncertainty_m, quality=1., weak_vertical=False,
    ):
        timestamp = float(timestamp)
        position = np.asarray(position_world, dtype=np.float64)
        if self.history and timestamp-self.history[-1][0] < self.minimum_dt_s:
            raise ValueError("motion update interval is too small")
        self.history.append((
            timestamp, position.copy(), float(np.clip(quality, .05, 1.)),
            max(float(geometry_uncertainty_m), 1e-4),
        ))
        while (
            len(self.history) > 1
            and timestamp-self.history[0][0] > self.maximum_history_age_s
        ):
            self.history.popleft()
        if len(self.history) < self.minimum_history:
            covariance = np.eye(3)*self.maximum_speed_mps**2
            estimate = MotionEstimateV1(
                self.reference_mode, timestamp, position, np.zeros(3),
                covariance,
                (
                    np.full(3, -self.maximum_speed_mps),
                    np.full(3, self.maximum_speed_mps),
                ),
                0., len(self.history), 0,
                MotionObservability.MOTION_INITIALIZING,
                timestamp,
            )
            self.last_estimate = estimate
            return estimate
        times = np.asarray([row[0] for row in self.history])
        positions = np.asarray([row[1] for row in self.history])
        quality_values = np.asarray([
            row[2]/max(row[3]**2, 1e-6) for row in self.history
        ])
        position_now, velocity, covariance, residual, outliers = _weighted_line(
            times, positions, quality_values, self.huber_delta_m,
            self.robust_iterations,
        )
        speed = np.linalg.norm(velocity)
        if speed > self.maximum_speed_mps:
            velocity *= self.maximum_speed_mps/speed
        sigma = 2*np.sqrt(np.maximum(np.diag(covariance), 1e-8))
        observability = (
            MotionObservability.MOTION_WEAKLY_OBSERVABLE
            if weak_vertical else MotionObservability.MOTION_OBSERVABLE
        )
        if weak_vertical:
            sigma[2] = self.maximum_speed_mps
            covariance[2, 2] = self.maximum_speed_mps**2
        estimate = MotionEstimateV1(
            self.reference_mode, timestamp, position_now, velocity,
            covariance, (velocity-sigma, velocity+sigma),
            residual, len(self.history), outliers, observability, timestamp,
        )
        self.last_estimate = estimate
        return estimate


__all__ = [
    "MOTION_STATE_VERSION", "MotionObservability", "MotionEstimateV1",
    "StableReferenceMotionEstimatorV1",
]

