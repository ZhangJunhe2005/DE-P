"""Deterministic standard DBSCAN clustering for world-frame point clouds."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN

from .types import ClusterObservation, DynamicPerceptionConfig


class StandardDBSCANClustering:
    algorithm_name = "sklearn.cluster.DBSCAN"

    def __init__(self, config: DynamicPerceptionConfig):
        config.validate()
        self.config = config
        self.eps = float(config.cluster_eps)
        self.min_samples = int(config.cluster_min_samples)
        self.cluster_max_iterations = int(config.cluster_max_iterations)

    def cluster(self, points_world, points_camera, timestamp):
        world = np.asarray(points_world, dtype=np.float64)
        camera = np.asarray(points_camera, dtype=np.float64)
        if world.shape != camera.shape or world.ndim != 2 or world.shape[1] != 3:
            raise ValueError("world and camera point clouds must both have shape [N,3]")
        if len(world) == 0:
            return (), np.empty((0,), dtype=np.int64)
        labels = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit_predict(world)
        observations = []
        for temporary_id in sorted(int(label) for label in np.unique(labels) if label != -1):
            mask = labels == temporary_id
            world_cluster = world[mask]
            camera_cluster = camera[mask]
            point_count = len(world_cluster)
            median = np.median(world_cluster, axis=0)
            distance = np.linalg.norm(world_cluster - median, axis=1)
            retain_count = max(1, int(round(
                point_count * (1.0 - self.config.centroid_trim_fraction)
            )))
            retained_indices = np.argsort(distance, kind="stable")[:retain_count]
            retained_world = world_cluster[retained_indices]
            retained_camera = camera_cluster[retained_indices]
            robust_centroid_world = retained_world.mean(axis=0)
            robust_centroid_camera = retained_camera.mean(axis=0)
            sample_covariance = (np.cov(retained_world, rowvar=False, ddof=1)
                                 if retain_count > 1 else np.zeros((3, 3)))
            bbox = np.stack((world_cluster.min(axis=0), world_cluster.max(axis=0)))
            extent = bbox[1] - bbox[0]
            # Large planar surfaces remain uncertain under view cropping even
            # with many points; point_count alone must not collapse covariance.
            extent_floor_std = (
                self.config.measurement_noise
                + self.config.centroid_extent_covariance_scale * extent
            )
            centroid_covariance = (
                sample_covariance / max(retain_count, 1)
                + np.diag(np.square(extent_floor_std))
            )
            eigenvalues = np.linalg.eigvalsh(sample_covariance).clip(min=0)
            largest = max(float(eigenvalues[-1]), 1e-12)
            shape_ratios = np.asarray([
                (eigenvalues[-1] - eigenvalues[-2]) / largest,
                (eigenvalues[-2] - eigenvalues[0]) / largest,
            ])
            voxel_keys = np.unique(np.floor(
                world_cluster / self.config.background_voxel_size
            ).astype(np.int64), axis=0)
            if len(voxel_keys) > 256:
                positions = np.linspace(0, len(voxel_keys) - 1, 256, dtype=int)
                voxel_keys = voxel_keys[positions]
            observations.append(ClusterObservation(
                temporary_cluster_id=temporary_id,
                centroid_world=robust_centroid_world,
                centroid_camera=robust_centroid_camera,
                point_count=point_count,
                bounding_box_world=bbox,
                position_covariance=centroid_covariance,
                timestamp=float(timestamp),
                extent=extent,
                covariance_eigenvalues=eigenvalues,
                shape_ratios=shape_ratios,
                voxel_signature=tuple(tuple(int(value) for value in row) for row in voxel_keys),
                foreground_support=1.0,
                # A DBSCAN component is still a direct current-frame sensor
                # observation in legacy point-cloud mode.  It has no image
                # pixels, so instance evaluators cannot claim actor identity.
                direct_image_evidence=True,
            ))
        return tuple(observations), labels.astype(np.int64, copy=False)
