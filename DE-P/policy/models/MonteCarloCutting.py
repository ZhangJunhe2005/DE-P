"""Deprecated compatibility wrapper for the former misnamed clustering module."""

from __future__ import annotations

import warnings
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


class PointCloudProcessor:
    """Legacy +X-forward point-cloud helper; new code uses ``policy.dynamic``."""

    def __init__(self, fov_angle=90, min_distance=0.5, max_distance=20.0):
        warnings.warn(
            "PointCloudProcessor is deprecated; use policy.dynamic point-cloud and "
            "StandardDBSCANClustering APIs",
            DeprecationWarning,
            stacklevel=2,
        )
        if not (0 < fov_angle <= 360):
            raise ValueError("fov_angle must be in (0,360]")
        if min_distance < 0 or max_distance <= min_distance:
            raise ValueError("invalid distance range")
        self.half_fov = np.radians(fov_angle) / 2
        self.min_dist = float(min_distance)
        self.max_dist = float(max_distance)

    def clip_region(self, point_cloud):
        points = np.asarray(point_cloud)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("point_cloud must have shape [N,3]")
        if len(points) == 0:
            return points
        distances = np.linalg.norm(points, axis=1)
        angles = np.arctan2(points[:, 1], points[:, 0])
        mask = ((points[:, 0] > 0) & (distances >= self.min_dist)
                & (distances <= self.max_dist) & (np.abs(angles) <= self.half_fov)
                & np.isfinite(points).all(axis=1))
        return points[mask]

    def remove_noise(self, point_cloud, k=5, dist_threshold=0.3):
        points = np.asarray(point_cloud)
        if len(points) < k:
            return points
        distances, _ = NearestNeighbors(n_neighbors=k).fit(points).kneighbors(points)
        return points[np.mean(distances, axis=1) < dist_threshold]

    def monte_carlo_clustering(self, point_cloud, eps=0.6, min_samples=6, iterations=300):
        """Compatibility name backed by exhaustive deterministic standard DBSCAN."""
        warnings.warn(
            "monte_carlo_clustering was never random and is deprecated; the compatibility "
            "path now uses sklearn.cluster.DBSCAN. 'iterations' is ignored.",
            DeprecationWarning,
            stacklevel=2,
        )
        points = np.asarray(point_cloud)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("point_cloud must have shape [N,3]")
        if eps <= 0 or min_samples <= 0 or iterations <= 0:
            raise ValueError("eps, min_samples and iterations must be positive")
        if len(points) == 0:
            return np.empty((0,), dtype=np.int64)
        return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points).astype(np.int64)
