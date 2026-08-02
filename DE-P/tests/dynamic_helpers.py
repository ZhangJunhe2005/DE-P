from dataclasses import replace

import numpy as np

from policy.dynamic.pointcloud import world_points_to_camera
from policy.dynamic.types import (
    CameraModel,
    ClusterObservation,
    DynamicPerceptionConfig,
    DynamicTrack,
    Pose,
)


def test_config(**overrides):
    config = DynamicPerceptionConfig.from_global_config()
    config = replace(config, **overrides)
    config.validate()
    return config


def camera_model():
    return CameraModel(
        width=160, height=96, fx=100.0, fy=100.0, cx=79.5, cy=47.5,
        depth_scale=1.0, min_depth=0.1, max_depth=20.0,
    )


def pose(position=(0, 0, 0), timestamp=0.0, rotation=None):
    return Pose(
        position_world=np.asarray(position, dtype=np.float64),
        rotation_world_from_camera=np.eye(3) if rotation is None else rotation,
        timestamp=float(timestamp),
    )


def observation(position, timestamp, temporary_id=0, covariance=1e-3):
    position = np.asarray(position, dtype=np.float64)
    return ClusterObservation(
        temporary_cluster_id=temporary_id,
        centroid_world=position,
        centroid_camera=position,
        point_count=20,
        bounding_box_world=np.stack((position - 0.05, position + 0.05)),
        position_covariance=np.eye(3) * covariance,
        timestamp=float(timestamp),
        extent=np.full(3, 0.2, dtype=np.float64),
    )


def world_cluster(center, seed=0, count=30, scale=0.025):
    rng = np.random.default_rng(seed)
    return rng.normal(np.asarray(center, dtype=np.float64), scale, size=(count, 3))


def camera_cloud_from_world(points_world, camera_pose):
    return world_points_to_camera(points_world, camera_pose).astype(np.float32)


def dynamic_track(track_id=1, position=(0, 0, 5), velocity=(0, 0, -1),
                  confirmed=True, dynamic=True, covariance=0.01, hits=6):
    return DynamicTrack(
        track_id=track_id,
        position_world=np.asarray(position, dtype=np.float64),
        velocity_world=np.asarray(velocity, dtype=np.float64),
        state_covariance=np.eye(6) * covariance,
        age=hits,
        hit_count=hits,
        missed_count=0,
        is_confirmed=confirmed,
        is_dynamic=dynamic,
        timestamp=1.0,
        dynamic_reason="test",
    )
