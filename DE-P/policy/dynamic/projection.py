"""World-to-image perspective projection and image/feature coordinate mapping."""

from __future__ import annotations

import numpy as np

from .pointcloud import world_points_to_camera
from .types import CameraModel, Pose


def project_world_points_to_image(points_world, camera_pose_world: Pose,
                                  camera_model: CameraModel, return_mask=False):
    """Project valid world points to pixel-center coordinates.

    Points behind the optical camera, outside the configured depth range, or
    outside image pixel-center bounds are filtered. If ``return_mask`` is true,
    returns ``(pixels, valid_mask, depths_camera)``.
    """
    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_world must have shape [N,3]")
    if len(points) == 0:
        empty_pixels = np.empty((0, 2), dtype=np.float64)
        if return_mask:
            return empty_pixels, np.zeros((0,), dtype=bool), np.empty((0,), dtype=np.float64)
        return empty_pixels
    camera = world_points_to_camera(points, camera_pose_world)
    z = camera[:, 2]
    valid = (np.isfinite(camera).all(axis=1) & (z >= camera_model.min_depth)
             & (z <= camera_model.max_depth))
    safe_z = np.where(valid, z, 1.0)
    u = camera_model.fx * camera[:, 0] / safe_z + camera_model.cx
    v = camera_model.fy * camera[:, 1] / safe_z + camera_model.cy
    valid &= (u >= -0.5) & (u < camera_model.width - 0.5)
    valid &= (v >= -0.5) & (v < camera_model.height - 0.5)
    pixels = np.stack((u[valid], v[valid]), axis=1)
    if not np.isfinite(pixels).all():
        raise FloatingPointError("projection produced NaN/Inf")
    if return_mask:
        return pixels, valid, z[valid].copy()
    return pixels


def image_to_feature_coordinates(pixels, image_shape, feature_shape):
    """Map pixel centers with the align_corners=False convention.

    Coordinate formula: ``f = (pixel + 0.5) * feature_size/image_size - 0.5``.
    """
    pixels = np.asarray(pixels, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError("pixels must have shape [N,2]")
    image_h, image_w = (int(image_shape[0]), int(image_shape[1]))
    feature_h, feature_w = (int(feature_shape[0]), int(feature_shape[1]))
    if min(image_h, image_w, feature_h, feature_w) <= 0:
        raise ValueError("image and feature shapes must be positive")
    output = np.empty_like(pixels, dtype=np.float64)
    output[:, 0] = (pixels[:, 0] + 0.5) * feature_w / image_w - 0.5
    output[:, 1] = (pixels[:, 1] + 0.5) * feature_h / image_h - 0.5
    if not np.isfinite(output).all():
        raise FloatingPointError("feature coordinate mapping produced NaN/Inf")
    return output
