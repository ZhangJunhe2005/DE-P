"""Batch independent sequence contexts and padded dynamic obstacle labels."""

import torch

from loss.dynamic_types import DynamicObstacleBatch
from policy.dynamic.context import ATTENTION_BACKBONE_OUTPUT, DynamicContext


def dynamic_sequence_collate(samples, max_obstacles=None):
    if not samples:
        raise ValueError("cannot collate an empty dynamic batch")
    batch = len(samples)
    observed_max = max(len(sample["objects"]) for sample in samples)
    max_obstacles = observed_max if max_obstacles is None else min(observed_max, max_obstacles)
    positions = torch.zeros(batch, max_obstacles, 3)
    velocities = torch.zeros_like(positions)
    covariances = torch.zeros(batch, max_obstacles, 3, 3)
    radii = torch.zeros(batch, max_obstacles)
    track_timestamps = torch.zeros(batch, max_obstacles, dtype=torch.float64)
    confidence = torch.zeros(batch, max_obstacles)
    valid = torch.zeros(batch, max_obstacles, dtype=torch.bool)
    dynamic = torch.zeros_like(valid)
    observable = torch.zeros_like(valid)
    ever_observed = torch.zeros_like(valid)
    future_points = samples[0]["future_timestamps"].numel()
    future_positions = torch.zeros(batch, max_obstacles, future_points, 3)
    future_valid = torch.zeros(batch, max_obstacles, future_points, dtype=torch.bool)
    future_visibility = torch.zeros_like(future_valid)
    for batch_index, sample in enumerate(samples):
        for obstacle_index, obj in enumerate(sample["objects"][:max_obstacles]):
            positions[batch_index, obstacle_index] = torch.tensor(obj["context_position_world"])
            velocities[batch_index, obstacle_index] = torch.tensor(obj["context_velocity_world"])
            covariances[batch_index, obstacle_index] = torch.tensor(obj["position_covariance"])
            radii[batch_index, obstacle_index] = float(obj["radius"])
            track_timestamps[batch_index, obstacle_index] = float(obj["track_timestamp"])
            confidence[batch_index, obstacle_index] = float(obj["supervision_confidence"])
            valid[batch_index, obstacle_index] = True
            dynamic[batch_index, obstacle_index] = bool(obj["dynamic"])
            observable[batch_index, obstacle_index] = bool(obj["observable"])
            ever_observed[batch_index, obstacle_index] = bool(obj["ever_observed_in_history"])
            future_positions[batch_index, obstacle_index] = sample["future_positions_world"][obstacle_index]
            future_valid[batch_index, obstacle_index] = sample["future_valid_mask"][obstacle_index]
            future_visibility[batch_index, obstacle_index] = sample["future_visibility_mask"][obstacle_index]
    obstacles = DynamicObstacleBatch(
        positions_world=positions, velocities_world=velocities,
        position_covariances=covariances, radii=radii,
        track_timestamps=track_timestamps,
        sample_timestamps=torch.tensor([sample["sample_timestamp"] for sample in samples],
                                       dtype=torch.float64),
        confidence=confidence, valid_mask=valid, dynamic_mask=dynamic,
        observable_mask=observable, ever_observed_in_history=ever_observed,
        future_positions_world=future_positions, future_valid_mask=future_valid,
        future_visibility_mask=future_visibility,
        future_timestamps=torch.stack([sample["future_timestamps"] for sample in samples]),
    )
    context = DynamicContext(
        attention_maps_by_level={ATTENTION_BACKBONE_OUTPUT: torch.stack([
            sample["attention"] for sample in samples
        ])},
        timestamp=None,
        source="depth",
        valid=True,
        diagnostics={"batch_semantics": "independent ground-truth sequence samples"},
    )
    tensor_keys = (
        "depth_history", "current_depth", "timestamps", "camera_positions_world",
        "observation_9d", "position_world", "rotation_world_from_body", "goal_world",
        "sequence_mask",
    )
    result = {key: torch.stack([sample[key] for sample in samples]) for key in tensor_keys}
    result.update({
        "map_id": torch.tensor([sample["map_id"] for sample in samples], dtype=torch.long),
        "dynamic_context": context,
        "dynamic_obstacles": obstacles,
        "sequence_id": [sample["sequence_id"] for sample in samples],
        "sample_category": [sample["sample_category"] for sample in samples],
        "frame_index": torch.tensor([sample["frame_index"] for sample in samples]),
    })
    return result
