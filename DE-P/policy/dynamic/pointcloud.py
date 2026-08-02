"""Explicit CPU point-cloud validation and camera/world transforms."""

from __future__ import annotations

import numpy as np
import torch

from .types import CameraModel, Pose


def validate_point_cloud_camera(point_cloud_camera, camera_model: CameraModel) -> np.ndarray:
    if torch.is_tensor(point_cloud_camera):
        raise TypeError("CPU clustering requires a NumPy point_cloud_camera; CUDA tensors are not converted implicitly")
    points = np.asarray(point_cloud_camera)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"point_cloud_camera must have shape [N,3], got {points.shape}")
    points = points.astype(np.float64, copy=False)
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float64)
    valid = np.isfinite(points).all(axis=1)
    depth = points[:, 2]
    valid &= (depth >= camera_model.min_depth) & (depth <= camera_model.max_depth)
    return np.ascontiguousarray(points[valid])


def camera_points_to_world(points_camera: np.ndarray, pose: Pose) -> np.ndarray:
    points = np.asarray(points_camera, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_camera must have shape [N,3]")
    return points @ pose.rotation_world_from_camera.T + pose.position_world


def world_points_to_camera(points_world: np.ndarray, pose: Pose) -> np.ndarray:
    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_world must have shape [N,3]")
    return (points - pose.position_world) @ pose.rotation_world_from_camera
