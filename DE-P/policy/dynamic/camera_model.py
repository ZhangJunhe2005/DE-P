"""Depth-image back-projection with explicit optical-camera intrinsics."""

from __future__ import annotations

import numpy as np
import torch

from .types import CameraModel, DepthFrame, Pose


def make_depth_frame(depth, camera_model: CameraModel, camera_pose_world: Pose,
                     timestamp: float, stride: int = 1) -> DepthFrame:
    """Validate a raw depth image and preserve sampled pixel/point identity."""
    if torch.is_tensor(depth):
        if depth.device.type != "cpu":
            raise ValueError("dynamic CPU perception requires CPU depth input")
        depth = depth.detach().numpy()
    if not isinstance(stride, int) or stride <= 0:
        raise ValueError("stride must be a positive integer")
    array = np.asarray(depth)
    expected = (camera_model.height, camera_model.width)
    if array.shape != expected:
        raise ValueError(f"raw depth shape must be {expected}, got {array.shape}")
    depth_m = np.asarray(array, dtype=np.float32) * np.float32(camera_model.depth_scale)
    valid_mask = (np.isfinite(depth_m) & (depth_m > 0)
                  & (depth_m >= camera_model.min_depth)
                  & (depth_m < camera_model.max_depth))
    vv, uu = np.meshgrid(
        np.arange(0, camera_model.height, stride, dtype=np.int32),
        np.arange(0, camera_model.width, stride, dtype=np.int32), indexing="ij"
    )
    sampled_valid = valid_mask[::stride, ::stride]
    pixels = np.stack((uu[sampled_valid], vv[sampled_valid]), axis=1).astype(np.int32)
    z = depth_m[pixels[:, 1], pixels[:, 0]] if len(pixels) else np.empty((0,), np.float32)
    points = np.stack((
        (pixels[:, 0].astype(np.float32) - camera_model.cx) * z / camera_model.fx,
        (pixels[:, 1].astype(np.float32) - camera_model.cy) * z / camera_model.fy,
        z,
    ), axis=1).astype(np.float32) if len(pixels) else np.empty((0, 3), np.float32)
    return DepthFrame(depth_m, valid_mask.astype(np.bool_), points, pixels,
                      camera_model, camera_pose_world, float(timestamp), stride)


def depth_to_pointcloud(depth, camera_model: CameraModel, stride: int = 1):
    """Back-project depth to [N,3] optical-camera points (+Z forward).

    NumPy input stays on CPU and returns float32 NumPy. Torch input stays on its
    current device and returns float32 Torch; no implicit CUDA-to-NumPy copy occurs.
    ``depth_scale=1`` is suitable for 32FC1 metres and ``0.001`` for 16UC1 mm.
    """
    if not isinstance(stride, int) or stride <= 0:
        raise ValueError("stride must be a positive integer")
    if tuple(depth.shape) != (camera_model.height, camera_model.width):
        raise ValueError(
            f"depth shape must be {(camera_model.height, camera_model.width)}, got {tuple(depth.shape)}"
        )
    rows = slice(0, camera_model.height, stride)
    cols = slice(0, camera_model.width, stride)
    if torch.is_tensor(depth):
        sampled = depth[rows, cols].to(dtype=torch.float32) * camera_model.depth_scale
        v, u = torch.meshgrid(
            torch.arange(0, camera_model.height, stride, device=depth.device, dtype=torch.float32),
            torch.arange(0, camera_model.width, stride, device=depth.device, dtype=torch.float32),
            indexing="ij",
        )
        # The Simulator uses exactly max_depth as its no-hit saturation
        # value; it is not an observed surface.
        valid = (torch.isfinite(sampled) & (sampled > 0)
                 & (sampled >= camera_model.min_depth) & (sampled < camera_model.max_depth))
        z = sampled[valid]
        points = torch.stack(((u[valid] - camera_model.cx) * z / camera_model.fx,
                              (v[valid] - camera_model.cy) * z / camera_model.fy, z), dim=1)
        if points.ndim != 2 or points.shape[1] != 3 or points.dtype != torch.float32:
            raise RuntimeError("invalid Torch point-cloud output")
        return points
    array = np.asarray(depth)
    sampled = array[rows, cols].astype(np.float32, copy=False) * np.float32(camera_model.depth_scale)
    v, u = np.meshgrid(
        np.arange(0, camera_model.height, stride, dtype=np.float32),
        np.arange(0, camera_model.width, stride, dtype=np.float32),
        indexing="ij",
    )
    valid = (np.isfinite(sampled) & (sampled > 0)
             & (sampled >= camera_model.min_depth) & (sampled < camera_model.max_depth))
    z = sampled[valid]
    points = np.stack(((u[valid] - camera_model.cx) * z / camera_model.fx,
                       (v[valid] - camera_model.cy) * z / camera_model.fy, z), axis=1).astype(np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or points.dtype != np.float32:
        raise RuntimeError("invalid NumPy point-cloud output")
    return points
