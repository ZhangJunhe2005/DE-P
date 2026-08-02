"""Finite current-frame occupancy support without an actor-center claim."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .types import DepthFrame


CONTRACT_VERSION = "near_field_safety_support_v1"


def _finite_tuple(name, value, size):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite shape ({size},)")
    return tuple(float(item) for item in array)


@dataclass(frozen=True)
class NearFieldSafetySupportV1:
    support_id: int
    frame_index: int
    timestamp: float
    source_component_ids: Tuple[int, ...]
    camera_frame: str
    world_frame: str
    near_field_clipped: bool
    occupied_depth_interval_m: Tuple[float, float]
    angular_support_rad: Tuple[float, float, float, float]
    position_set_min_world: Tuple[float, float, float]
    position_set_max_world: Tuple[float, float, float]
    extent_bound_m: float
    valid_point_count: int
    expiry_timestamp: float
    runtime_gt_used: bool = False
    formal_eligible: bool = False

    def __post_init__(self):
        if self.support_id < 0 or self.frame_index < 0:
            raise ValueError("support/frame IDs must be non-negative")
        if not self.source_component_ids:
            raise ValueError("component provenance is required")
        if self.runtime_gt_used or self.formal_eligible:
            raise ValueError("support cannot use GT or enter formal tracking")
        depth = _finite_tuple(
            "occupied_depth_interval_m",
            self.occupied_depth_interval_m, 2,
        )
        angular = _finite_tuple(
            "angular_support_rad", self.angular_support_rad, 4
        )
        lower = _finite_tuple(
            "position_set_min_world", self.position_set_min_world, 3
        )
        upper = _finite_tuple(
            "position_set_max_world", self.position_set_max_world, 3
        )
        if depth[0] <= 0 or depth[1] < depth[0]:
            raise ValueError("occupied depth interval is invalid")
        if angular[1] < angular[0] or angular[3] < angular[2]:
            raise ValueError("angular support is invalid")
        if any(high < low for low, high in zip(lower, upper)):
            raise ValueError("world position set is invalid")
        if (
            self.extent_bound_m <= 0 or self.valid_point_count <= 0
            or not np.isfinite((self.timestamp, self.expiry_timestamp,
                                self.extent_bound_m)).all()
            or self.expiry_timestamp <= self.timestamp
        ):
            raise ValueError("support lifetime/extent is invalid")
        object.__setattr__(
            self, "source_component_ids",
            tuple(int(item) for item in self.source_component_ids),
        )

    @property
    def support_bounds(self):
        return {
            "min_world": self.position_set_min_world,
            "max_world": self.position_set_max_world,
        }


def build_near_field_support(
    frame: DepthFrame,
    pixels_vu,
    source_component_ids,
    frame_index: int,
    support_id: int,
    maximum_age_s: float,
    uav_safety_radius_m: float,
    frozen_extent_prior_m: float,
    minimum_finite_points: int,
    maximum_depth_span_m: float,
    maximum_angular_span_rad: float,
):
    """Build a bounded set from the current valid subset, or return ``None``.

    No centroid, velocity, shape class, identity, GT, or future frame is used.
    """
    pixels = np.asarray(pixels_vu, dtype=np.int64)
    if pixels.ndim != 2 or pixels.shape[1:] != (2,):
        raise ValueError("pixels_vu must have shape [N,2]")
    if not len(pixels):
        return None
    height, width = frame.depth_m.shape
    v, u = pixels.T
    inside = (v >= 0) & (v < height) & (u >= 0) & (u < width)
    if not np.all(inside):
        raise ValueError("component pixels lie outside the depth image")
    valid = frame.valid_mask[v, u]
    valid_pixels = pixels[valid]
    if len(valid_pixels) < int(minimum_finite_points):
        return None
    vv, uu = valid_pixels.T
    depth = frame.depth_m[vv, uu].astype(np.float64)
    if (
        not np.isfinite(depth).all()
        or float(np.ptp(depth)) > float(maximum_depth_span_m)
    ):
        return None
    model = frame.camera_model
    x_angle = np.arctan2(
        uu.astype(np.float64)-model.cx, model.fx
    )
    y_angle = np.arctan2(
        vv.astype(np.float64)-model.cy, model.fy
    )
    angular = (
        float(x_angle.min()), float(x_angle.max()),
        float(y_angle.min()), float(y_angle.max()),
    )
    if (
        angular[1]-angular[0] > float(maximum_angular_span_rad)
        or angular[3]-angular[2] > float(maximum_angular_span_rad)
    ):
        return None
    points_camera = np.stack((
        (uu.astype(np.float64)-model.cx)*depth/model.fx,
        (vv.astype(np.float64)-model.cy)*depth/model.fy,
        depth,
    ), axis=1)
    pose = frame.camera_pose_world
    points_world = (
        points_camera @ pose.rotation_world_from_camera.T
        + pose.position_world
    )
    expansion = float(uav_safety_radius_m)+float(
        frozen_extent_prior_m
    )
    lower = points_world.min(axis=0)-expansion
    upper = points_world.max(axis=0)+expansion
    raw = frame.depth_m[v, u]
    near_clipped = bool(np.any(
        np.isfinite(raw) & (raw > 0) & (raw < model.min_depth)
    ))
    return NearFieldSafetySupportV1(
        support_id=int(support_id),
        frame_index=int(frame_index),
        timestamp=float(frame.timestamp),
        source_component_ids=tuple(source_component_ids),
        camera_frame="camera_optical",
        world_frame="world",
        near_field_clipped=near_clipped,
        occupied_depth_interval_m=(
            float(depth.min()), float(depth.max())
        ),
        angular_support_rad=angular,
        position_set_min_world=tuple(lower),
        position_set_max_world=tuple(upper),
        extent_bound_m=expansion,
        valid_point_count=len(depth),
        expiry_timestamp=float(frame.timestamp)+float(maximum_age_s),
    )


__all__ = [
    "CONTRACT_VERSION", "NearFieldSafetySupportV1",
    "build_near_field_support",
]
