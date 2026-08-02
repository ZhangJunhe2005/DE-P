"""Deprecated compatibility names for the replaced EKF/attention experiment."""

from __future__ import annotations

import warnings
import numpy as np
import torch.nn as nn

from policy.dynamic.kalman_tracker import LinearKalmanTracker
from policy.dynamic.types import DynamicPerceptionConfig


class Obstacle:
    def __init__(self, cluster_id, position, timestamp):
        warnings.warn("Obstacle is deprecated; use DynamicTrack", DeprecationWarning, stacklevel=2)
        self.cluster_id = int(cluster_id)
        self.position = np.asarray(position, dtype=np.float64)
        self.velocity = np.zeros(3, dtype=np.float64)
        self.timestamp = float(timestamp)
        self.is_dynamic = False


class EKFTracker:
    """Compatibility wrapper; implementation is now a standard linear KF."""

    def __init__(self, initial_position, dt=0.1):
        warnings.warn(
            "EKFTracker is deprecated and now wraps LinearKalmanTracker; use "
            "policy.dynamic.kalman_tracker.LinearKalmanTracker with timestamps",
            DeprecationWarning,
            stacklevel=2,
        )
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.dt = float(dt)
        self.timestamp = 0.0
        self.tracker = LinearKalmanTracker(
            initial_position, self.timestamp, DynamicPerceptionConfig.from_global_config()
        )

    def predict(self):
        self.timestamp += self.dt
        return self.tracker.predict_to(self.timestamp)[:3]

    def update(self, position, velocity=None):
        if velocity is not None:
            warnings.warn("legacy velocity update is ignored by the position-only linear KF",
                          DeprecationWarning, stacklevel=2)
        return self.tracker.update(position)[:3]

    def get_state(self):
        state = self.tracker.state
        return state[:3], state[3:]


class DynamicObstacleAttention(nn.Module):
    """Import-compatible guard; the formal stage-4 API is ``DynamicPerception``."""

    def __init__(self, input_feature_size=(24, 40), cluster_eps=None,
                 cluster_min_samples=None, cluster_iterations=None, cluster_size=None,
                 **legacy_options):
        super().__init__()
        warnings.warn(
            "DynamicObstacleAttention is deprecated and is not a production integration; "
            "use policy.dynamic.DynamicPerception.update and build_dynamic_attention",
            DeprecationWarning,
            stacklevel=2,
        )
        config = DynamicPerceptionConfig.from_global_config()
        if cluster_size is not None:
            warnings.warn(
                "cluster_size is deprecated; use dynamic_perception.cluster_min_samples",
                DeprecationWarning,
                stacklevel=2,
            )
            if cluster_min_samples is None:
                cluster_min_samples = cluster_size
        self.input_feature_size = tuple(input_feature_size)
        self.cluster_eps = config.cluster_eps if cluster_eps is None else float(cluster_eps)
        self.cluster_min_samples = (config.cluster_min_samples if cluster_min_samples is None
                                    else int(cluster_min_samples))
        self.cluster_iterations = (config.cluster_max_iterations if cluster_iterations is None
                                   else int(cluster_iterations))
        if self.cluster_eps <= 0 or self.cluster_min_samples <= 0 or self.cluster_iterations <= 0:
            raise ValueError("legacy clustering parameters must be positive")

    def forward(self, current_pcl, feature_map):
        raise RuntimeError(
            "Deprecated DynamicObstacleAttention lacks camera pose, timestamps, and intrinsics. "
            "Run policy.dynamic.DynamicPerception independently; it is intentionally not "
            "connected to CNN features in stage 4."
        )
