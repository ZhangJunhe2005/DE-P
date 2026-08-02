"""Deterministic CPU reference for static depth correspondence provenance."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


REFERENCE_VERSION = "warp_visibility_reference_v1"


def _rays(sensor):
    height, width = int(sensor["height"]), int(sensor["width"])
    fx, fy, cx, cy = [float(x) for x in sensor["intrinsics"]]
    v, u = np.meshgrid(
        np.arange(height, dtype=np.float64),
        np.arange(width, dtype=np.float64), indexing="ij",
    )
    rays = np.stack((
        np.ones_like(u), (u-cx)/fx, (v-cy)/fy,
    ), axis=-1)
    return rays/np.linalg.norm(rays, axis=-1, keepdims=True)


def _world_rays(sensor, yaw):
    rays = _rays(sensor)
    c, s = np.cos(float(yaw)), np.sin(float(yaw))
    rotation = np.asarray(
        [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]
    )
    return rays @ rotation


def _project_world(points_world, position, yaw, sensor):
    relative_world = points_world-np.asarray(position, dtype=np.float64)
    c, s = np.cos(float(yaw)), np.sin(float(yaw))
    world_from_body = np.asarray(
        [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]
    )
    body = relative_world @ world_from_body.T
    radial = np.linalg.norm(body, axis=-1)
    forward = body[..., 0]
    fx, fy, cx, cy = [float(x) for x in sensor["intrinsics"]]
    safe = np.where(forward > 1e-12, forward, 1.0)
    u = fx*body[..., 1]/safe+cx
    v = fy*body[..., 2]/safe+cy
    return u, v, radial, forward


def _valid(depth, sensor):
    maximum = float(sensor["max_depth_m"])
    minimum = float(sensor.get("min_depth_m", 0.1))
    return np.isfinite(depth) & (depth >= minimum) & (depth < maximum-1e-6)


def classify_pair(
    previous_depth, current_depth,
    previous_position, previous_yaw,
    current_position, current_yaw,
    sensor, tolerance_m,
):
    """Classify current and previous pixels without actor or detector inputs."""
    previous = np.asarray(previous_depth, dtype=np.float64)
    current = np.asarray(current_depth, dtype=np.float64)
    shape = (int(sensor["height"]), int(sensor["width"]))
    if previous.shape != shape or current.shape != shape:
        raise ValueError("depth shape does not match sensor")
    height, width = shape
    current_valid = _valid(current, sensor)
    previous_valid = _valid(previous, sensor)
    current_points = (
        np.asarray(current_position, dtype=np.float64)
        + _world_rays(sensor, current_yaw)*current[..., None]
    )
    u_prev, v_prev, expected_prev, forward_prev = _project_world(
        current_points, previous_position, previous_yaw, sensor
    )
    in_previous = (
        (forward_prev > 0)
        & (u_prev >= 0) & (u_prev <= width-1)
        & (v_prev >= 0) & (v_prev <= height-1)
    )
    up = np.clip(np.rint(u_prev), 0, width-1).astype(np.int64)
    vp = np.clip(np.rint(v_prev), 0, height-1).astype(np.int64)
    sampled_previous = previous[vp, up]
    sampled_previous_valid = previous_valid[vp, up]
    depth_error = sampled_previous-expected_prev
    consistent = (
        current_valid & in_previous & sampled_previous_valid
        & (np.abs(depth_error) <= float(tolerance_m))
    )
    outside_previous = current_valid & ~in_previous
    previous_depth_invalid = (
        current_valid & in_previous & ~sampled_previous_valid
    )
    disocclusion = (
        current_valid & in_previous & sampled_previous_valid
        & (depth_error < -float(tolerance_m))
    )
    occlusion = (
        current_valid & in_previous & sampled_previous_valid
        & (depth_error > float(tolerance_m))
    )
    max_transition = (
        (current >= float(sensor["max_depth_m"])-1e-6)
        != (sampled_previous >= float(sensor["max_depth_m"])-1e-6)
    ) & in_previous

    previous_points = (
        np.asarray(previous_position, dtype=np.float64)
        + _world_rays(sensor, previous_yaw)*previous[..., None]
    )
    u_current, v_current, _, forward_current = _project_world(
        previous_points, current_position, current_yaw, sensor
    )
    in_current = (
        (forward_current > 0)
        & (u_current >= 0) & (u_current <= width-1)
        & (v_current >= 0) & (v_current <= height-1)
    )
    newly_invalid = previous_valid & ~in_current
    current_depth_invalid = ~current_valid
    border_distance = np.minimum.reduce([
        np.broadcast_to(np.arange(width), shape),
        np.broadcast_to(np.arange(width-1, -1, -1), shape),
        np.broadcast_to(np.arange(height)[:, None], shape),
        np.broadcast_to(np.arange(height-1, -1, -1)[:, None], shape),
    ]).astype(np.float32)
    confidence = np.zeros(shape, dtype=np.float32)
    confidence[consistent] = np.clip(
        1.0-np.abs(depth_error[consistent])/max(float(tolerance_m), 1e-9),
        0.0, 1.0,
    )
    return {
        "version": REFERENCE_VERSION,
        "current_to_previous_uv": np.stack((u_prev, v_prev), axis=-1),
        "previous_to_current_uv": np.stack(
            (u_current, v_current), axis=-1
        ),
        "warp_valid_mask": consistent,
        "stable_overlap": consistent,
        "newly_visible_from_image_fov": outside_previous,
        "newly_visible_from_static_disocclusion": disocclusion,
        "previously_occluded": disocclusion,
        "occlusion_boundary": occlusion,
        "disocclusion_boundary": disocclusion,
        "projected_outside_previous_image": outside_previous,
        "newly_invalid_to_image_fov": newly_invalid,
        "current_depth_invalid": current_depth_invalid,
        "previous_depth_invalid": previous_depth_invalid,
        "max_depth_transition": max_transition,
        "depth_consistency_error_m": depth_error.astype(np.float32),
        "border_distance_pixels": border_distance,
        "warp_confidence": confidence,
        "future_frames_used": 0,
        "actor_gt_used": False,
        "detector_dependency": False,
        "implementation_hash": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }


def summarize(value):
    keys = (
        "stable_overlap", "newly_visible_from_image_fov",
        "newly_visible_from_static_disocclusion",
        "projected_outside_previous_image",
        "newly_invalid_to_image_fov", "current_depth_invalid",
        "previous_depth_invalid", "max_depth_transition",
        "occlusion_boundary", "disocclusion_boundary",
    )
    return {f"{key}_pixels": int(np.asarray(value[key]).sum()) for key in keys}


__all__ = ["REFERENCE_VERSION", "classify_pair", "summarize"]
