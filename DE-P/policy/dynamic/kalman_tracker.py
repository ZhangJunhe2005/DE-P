"""Standard linear constant-velocity Kalman filter."""

from __future__ import annotations

import numpy as np
from filterpy.kalman import KalmanFilter

from .types import DynamicPerceptionConfig


class LinearKalmanTracker:
    """State [x,y,z,vx,vy,vz], measurement [x,y,z]."""

    def __init__(self, initial_position_world, timestamp, config: DynamicPerceptionConfig,
                 initial_measurement_covariance=None):
        config.validate()
        position = np.asarray(initial_position_world, dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("initial_position_world must be finite shape [3]")
        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        self.config = config
        self.filter = KalmanFilter(dim_x=6, dim_z=3)
        self.filter.x = np.concatenate(
            (position, np.zeros(3, dtype=np.float64))
        ).reshape(6, 1)
        self.filter.H = np.zeros((3, 6), dtype=np.float64)
        self.filter.H[:, :3] = np.eye(3)
        self.filter.P = np.diag(
            [config.initial_position_variance] * 3 + [config.initial_velocity_variance] * 3
        ).astype(np.float64)
        self.filter.R = self._measurement_covariance(initial_measurement_covariance)
        self.filter.F = np.eye(6, dtype=np.float64)
        self.filter.Q = np.zeros((6, 6), dtype=np.float64)
        self.last_timestamp = float(timestamp)
        self.last_raw_dt = None
        self.last_effective_dt = None

    def _measurement_covariance(self, covariance):
        base = np.eye(3, dtype=np.float64) * self.config.measurement_noise ** 2
        if covariance is None:
            return base
        value = np.asarray(covariance, dtype=np.float64)
        if value.shape != (3, 3) or not np.isfinite(value).all():
            raise ValueError("measurement covariance must be finite shape [3,3]")
        value = 0.5 * (value + value.T)
        if np.min(np.linalg.eigvalsh(value)) < -1e-10:
            raise ValueError("measurement covariance must be positive semidefinite")
        return value + base

    def _set_dynamics(self, dt):
        self.filter.F = np.eye(6, dtype=np.float64)
        self.filter.F[:3, 3:] = np.eye(3) * dt
        q = self.config.process_noise_acceleration ** 2
        self.filter.Q = np.block([
            [np.eye(3) * (dt ** 4 / 4) * q, np.eye(3) * (dt ** 3 / 2) * q],
            [np.eye(3) * (dt ** 3 / 2) * q, np.eye(3) * (dt ** 2) * q],
        ])

    def predict_to(self, timestamp):
        timestamp = float(timestamp)
        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        raw_dt = timestamp - self.last_timestamp
        if raw_dt == 0:
            raise ValueError("duplicate timestamp is not allowed")
        if raw_dt < 0:
            raise ValueError("out-of-order timestamp is not allowed")
        effective_dt = float(np.clip(raw_dt, self.config.min_dt, self.config.max_dt))
        self._set_dynamics(effective_dt)
        self.filter.predict()
        self.filter.P = 0.5 * (self.filter.P + self.filter.P.T)
        self.last_timestamp = timestamp
        self.last_raw_dt = raw_dt
        self.last_effective_dt = effective_dt
        self._validate_finite()
        return self.state

    def update(self, position_world, position_covariance=None):
        measurement = np.asarray(position_world, dtype=np.float64)
        if measurement.shape != (3,) or not np.isfinite(measurement).all():
            raise ValueError("position_world must be finite shape [3]")
        self.filter.R = self._measurement_covariance(position_covariance)
        self.filter.update(measurement)
        self.filter.P = 0.5 * (self.filter.P + self.filter.P.T)
        diagonal = np.diag(self.filter.P).copy()
        diagonal[3:] = np.maximum(
            diagonal[3:], self.config.velocity_covariance_floor ** 2
        )
        np.fill_diagonal(self.filter.P, diagonal)
        self._validate_finite()
        return self.state

    def set_velocity(self, velocity_world):
        """Inject a robust velocity estimate while preserving covariance safety."""
        velocity = np.asarray(velocity_world, dtype=np.float64)
        if velocity.shape != (3,) or not np.isfinite(velocity).all():
            raise ValueError("velocity_world must be finite shape [3]")
        speed = float(np.linalg.norm(velocity))
        limit = self.config.physically_plausible_speed_max
        if speed > limit:
            velocity = velocity * (limit / speed)
        self.filter.x[3:, 0] = velocity
        diagonal = np.diag(self.filter.P).copy()
        diagonal[3:] = np.maximum(
            diagonal[3:], self.config.velocity_covariance_floor ** 2
        )
        np.fill_diagonal(self.filter.P, diagonal)
        self._validate_finite()
        return self.state

    @property
    def state(self):
        return np.asarray(self.filter.x, dtype=np.float64).reshape(6).copy()

    @property
    def covariance(self):
        return np.asarray(self.filter.P, dtype=np.float64).copy()

    def innovation_covariance(self, position_covariance=None):
        measurement_covariance = self._measurement_covariance(position_covariance)
        return self.filter.H @ self.filter.P @ self.filter.H.T + measurement_covariance

    def _validate_finite(self):
        if not np.isfinite(self.filter.x).all() or not np.isfinite(self.filter.P).all():
            raise FloatingPointError("Kalman state or covariance contains NaN/Inf")
        if not np.allclose(self.filter.P, self.filter.P.T, atol=1e-8):
            raise FloatingPointError("Kalman covariance lost symmetry")
