"""Causal runtime visibility provenance from depth and camera poses only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import Pose


RUNTIME_PROVENANCE_VERSION = "runtime_visibility_provenance_v1"


@dataclass(frozen=True)
class RuntimeVisibilityProvenance:
    stable_overlap: np.ndarray
    projected_outside_previous_image: np.ndarray
    newly_visible_candidate: np.ndarray
    newly_invalid_candidate: np.ndarray
    previous_depth_invalid: np.ndarray
    current_depth_invalid: np.ndarray
    max_depth_transition: np.ndarray
    occlusion_candidate: np.ndarray
    disocclusion_candidate: np.ndarray
    depth_consistency_error: np.ndarray
    warp_confidence: np.ndarray
    current_to_previous_uv: np.ndarray
    previous_to_current_uv: np.ndarray
    geometric_overlap: np.ndarray
    sampled_previous_depth: np.ndarray
    expected_previous_depth: np.ndarray
    future_frames_used: int = 0
    actor_gt_used: bool = False
    reference_mask_used: bool = False


def _unit_rays(height, width, intrinsics):
    fx, fy, cx, cy = [float(value) for value in intrinsics]
    v, u = np.meshgrid(
        np.arange(height, dtype=np.float64),
        np.arange(width, dtype=np.float64), indexing="ij",
    )
    rays = np.stack(((u-cx)/fx, (v-cy)/fy, np.ones_like(u)), axis=-1)
    return rays / np.linalg.norm(rays, axis=-1, keepdims=True)


def _valid(depth, minimum, maximum):
    return (
        np.isfinite(depth) & (depth >= minimum)
        & (depth < maximum-1e-6)
    )


def _world_points(depth, pose, rays):
    camera = rays * depth[..., None]
    return (
        camera @ pose.rotation_world_from_camera.T
        + pose.position_world
    )


def _project(points_world, pose, intrinsics):
    camera = (
        points_world-pose.position_world
    ) @ pose.rotation_world_from_camera
    z = camera[..., 2]
    radial = np.linalg.norm(camera, axis=-1)
    fx, fy, cx, cy = [float(value) for value in intrinsics]
    safe = np.where(z > 1e-12, z, 1.0)
    u = fx*camera[..., 0]/safe+cx
    v = fy*camera[..., 1]/safe+cy
    return u, v, radial, z


def compute_visibility_provenance(
    previous_depth,
    current_depth,
    previous_pose: Pose,
    current_pose: Pose,
    intrinsics,
    max_depth,
    tolerance_m,
    min_depth=0.1,
):
    """Compute causal masks; no authority, GT, future frame, or reference input."""
    if not isinstance(previous_pose, Pose) or not isinstance(current_pose, Pose):
        raise TypeError("previous_pose/current_pose must be Pose")
    previous = np.asarray(previous_depth, dtype=np.float64)
    current = np.asarray(current_depth, dtype=np.float64)
    if previous.shape != current.shape or previous.ndim != 2:
        raise ValueError("previous/current depth must share a 2-D shape")
    height, width = current.shape
    rays = _unit_rays(height, width, intrinsics)
    previous_valid = _valid(previous, float(min_depth), float(max_depth))
    current_valid = _valid(current, float(min_depth), float(max_depth))

    current_world = _world_points(current, current_pose, rays)
    up, vp, expected_previous, zp = _project(
        current_world, previous_pose, intrinsics
    )
    in_previous = (
        (zp > 0) & (up >= 0) & (up <= width-1)
        & (vp >= 0) & (vp <= height-1)
    )
    ui = np.clip(np.rint(up), 0, width-1).astype(np.int64)
    vi = np.clip(np.rint(vp), 0, height-1).astype(np.int64)
    sampled_previous = previous[vi, ui]
    sampled_previous_valid = previous_valid[vi, ui]
    error = sampled_previous-expected_previous
    geometric_overlap = current_valid & in_previous & sampled_previous_valid
    stable = geometric_overlap & (np.abs(error) <= float(tolerance_m))
    outside = current_valid & ~in_previous
    previous_invalid = current_valid & in_previous & ~sampled_previous_valid
    closer = geometric_overlap & (error > float(tolerance_m))
    farther = geometric_overlap & (error < -float(tolerance_m))
    max_transition = in_previous & (
        (current >= float(max_depth)-1e-6)
        != (sampled_previous >= float(max_depth)-1e-6)
    )

    previous_world = _world_points(previous, previous_pose, rays)
    uc, vc, _, zc = _project(previous_world, current_pose, intrinsics)
    in_current = (
        (zc > 0) & (uc >= 0) & (uc <= width-1)
        & (vc >= 0) & (vc <= height-1)
    )
    newly_invalid = previous_valid & ~in_current
    confidence = np.zeros_like(current, dtype=np.float32)
    confidence[stable] = np.clip(
        1.0-np.abs(error[stable])/max(float(tolerance_m), 1e-9),
        0.0, 1.0,
    )
    return RuntimeVisibilityProvenance(
        stable_overlap=stable,
        projected_outside_previous_image=outside,
        newly_visible_candidate=outside,
        newly_invalid_candidate=newly_invalid,
        previous_depth_invalid=previous_invalid,
        current_depth_invalid=~current_valid,
        max_depth_transition=max_transition,
        occlusion_candidate=closer,
        disocclusion_candidate=farther,
        depth_consistency_error=error.astype(np.float32),
        warp_confidence=confidence,
        current_to_previous_uv=np.stack((up, vp), axis=-1),
        previous_to_current_uv=np.stack((uc, vc), axis=-1),
        geometric_overlap=geometric_overlap,
        sampled_previous_depth=sampled_previous.astype(np.float32),
        expected_previous_depth=expected_previous.astype(np.float32),
    )


__all__ = [
    "RUNTIME_PROVENANCE_VERSION", "RuntimeVisibilityProvenance",
    "compute_visibility_provenance",
]
