"""Standalone bounded dynamic-track attention generation."""

from __future__ import annotations

import numpy as np
import torch

from .projection import image_to_feature_coordinates, project_world_points_to_image
from .types import (
    CameraModel,
    DynamicPerceptionConfig,
    Pose,
    ProjectedDynamicTrack,
)


def _track_weight(track, camera_pose_world: Pose, config: DynamicPerceptionConfig):
    relative = track.position_world - camera_pose_world.position_world
    distance = float(np.linalg.norm(relative))
    if distance <= 1e-9:
        radial_speed = 0.0
    else:
        radial_speed = float(np.dot(track.velocity_world, relative / distance))
    closing_speed = max(0.0, -radial_speed)
    distance_weight = float(np.exp(-distance / config.attention_distance_scale))
    confidence = float(track.confidence)
    position_std = float(np.sqrt(max(np.max(np.diag(track.state_covariance)[:3]), 0.0)))
    uncertainty_weight = 1.0 / (1.0 + position_std / config.attention_uncertainty_scale)
    approach_weight = min(2.0, 1.0 + closing_speed / config.dynamic_enter_speed)
    departing_weight = 0.25 if radial_speed > 0 else 1.0
    weight = distance_weight * confidence * uncertainty_weight * approach_weight * departing_weight
    return float(np.clip(weight, 0.0, config.attention_max))


def build_dynamic_attention_with_projection(tracks, camera_pose_world: Pose,
                                            camera_model: CameraModel, feature_shape,
                                            config: DynamicPerceptionConfig, device="cpu"):
    feature_h, feature_w = int(feature_shape[0]), int(feature_shape[1])
    if feature_h <= 0 or feature_w <= 0:
        raise ValueError("feature_shape must be positive")
    attention = torch.zeros((1, 1, feature_h, feature_w), dtype=torch.float32, device=device)
    candidates = [
        track for track in tracks
        if track.is_confirmed
        and track.is_dynamic
        and track.confidence >= config.track_confidence_threshold
    ]
    if not candidates:
        return attention, ()
    positions = np.stack([track.position_world for track in candidates])
    pixels, valid_mask, depths = project_world_points_to_image(
        positions, camera_pose_world, camera_model, return_mask=True
    )
    valid_indices = np.flatnonzero(valid_mask)
    if not len(valid_indices):
        return attention, ()
    feature_coordinates = image_to_feature_coordinates(
        pixels, (camera_model.height, camera_model.width), (feature_h, feature_w)
    )
    grid_y, grid_x = torch.meshgrid(
        torch.arange(feature_h, dtype=torch.float32, device=device),
        torch.arange(feature_w, dtype=torch.float32, device=device),
        indexing="ij",
    )
    projected = []
    for output_index, candidate_index in enumerate(valid_indices.tolist()):
        track = candidates[candidate_index]
        weight = _track_weight(track, camera_pose_world, config)
        coordinate = feature_coordinates[output_index]
        dx = grid_x - float(coordinate[0])
        dy = grid_y - float(coordinate[1])
        gaussian = torch.exp(-(dx.square() + dy.square()) / (2 * config.attention_sigma ** 2))
        attention = torch.maximum(attention, (gaussian * weight).view(1, 1, feature_h, feature_w))
        projected.append(ProjectedDynamicTrack(
            track_id=track.track_id,
            pixel=pixels[output_index].copy(),
            feature_coordinate=coordinate.copy(),
            depth_camera=float(depths[output_index]),
            attention_weight=weight,
        ))
    attention.clamp_(0.0, config.attention_max)
    if not torch.isfinite(attention).all():
        raise FloatingPointError("dynamic attention contains NaN/Inf")
    return attention, tuple(projected)


def build_dynamic_attention(tracks, camera_pose_world: Pose, camera_model: CameraModel,
                            feature_shape, config=None, device="cpu"):
    """Return bounded [1,1,H,W] attention from confirmed dynamic tracks only."""
    config = config or DynamicPerceptionConfig.from_global_config()
    attention, _ = build_dynamic_attention_with_projection(
        tracks, camera_pose_world, camera_model, feature_shape, config, device
    )
    return attention
