"""Frozen causal perception probe used by V3 generation and validation."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from policy.dynamic.dynamic_perception import DynamicPerception
from policy.dynamic.types import (
    CameraModel, DynamicPerceptionConfig, Pose,
)


CAMERA_BODY_FROM_OPTICAL = np.asarray(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)


def frozen_perception_config(sensor):
    fx, fy, cx, cy = sensor["intrinsics"]
    return replace(
        DynamicPerceptionConfig.from_global_config(),
        enabled=True,
        foreground_mode="range_image_hybrid",
        camera_width=int(sensor["width"]),
        camera_height=int(sensor["height"]),
        camera_fx=float(fx), camera_fy=float(fy),
        camera_cx=float(cx), camera_cy=float(cy),
    )


def camera_model(sensor):
    fx, fy, cx, cy = sensor["intrinsics"]
    return CameraModel(
        width=int(sensor["width"]), height=int(sensor["height"]),
        fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
        depth_scale=1.0, min_depth=0.1,
        max_depth=float(sensor["max_depth_m"]),
    )


def _pose(position, yaw, timestamp):
    c, s = np.cos(yaw), np.sin(yaw)
    body = np.asarray([
        [c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0],
    ])
    return Pose(
        np.asarray(position, dtype=np.float64),
        body @ CAMERA_BODY_FROM_OPTICAL,
        float(timestamp),
    )


def run_frozen_perception_probe(
    depths, uav_positions, yaws, actor_positions, frame_times,
    sensor, minimum_sustained_frames=3, association_distance_m=1.0,
    visibility_mask=None, require_occlusion_identity=False,
    prediction_position_error_max_m=1.0,
):
    """Return actual track/attention evidence without actor metadata inputs."""
    config = frozen_perception_config(sensor)
    perception = DynamicPerception(
        config, (3, 5), attention_device="cpu"
    )
    model = camera_model(sensor)
    actor_positions = np.asarray(actor_positions, dtype=np.float64)
    actor_count = (
        int(actor_positions.shape[1]) if actor_positions.ndim == 3 else 0
    )
    consecutive = np.zeros(actor_count, dtype=np.int64)
    maximum_consecutive = np.zeros(actor_count, dtype=np.int64)
    entered = np.zeros(actor_count, dtype=bool)
    estimated_speeds = [[] for _ in range(actor_count)]
    track_ids = [set() for _ in range(actor_count)]
    all_track_ids_by_frame = []
    position_errors_by_frame = []
    actor_matches_by_frame = []
    attention_frames = 0
    dynamic_track_total = 0
    all_track_total = 0
    for index, timestamp in enumerate(frame_times):
        result = perception.update_depth(
            np.asarray(depths[index], dtype=np.float32),
            _pose(uav_positions[index], yaws[index], timestamp),
            float(timestamp), model,
        )
        dynamic_track_total += len(result.dynamic_tracks)
        all_track_total += len(result.all_tracks)
        current_all_ids = set()
        current_errors = []
        frame_matches = [None for _ in range(actor_count)]
        if actor_count and result.all_tracks:
            gt_all = actor_positions[index]
            tracks_all = np.stack([
                row.position_world for row in result.all_tracks
            ])
            distances_all = np.linalg.norm(
                gt_all[:, None, :]-tracks_all[None, :, :], axis=2
            )
            actor_indices, track_indices = linear_sum_assignment(distances_all)
            for actor_index, track_index in zip(
                actor_indices.tolist(), track_indices.tolist()
            ):
                if distances_all[actor_index, track_index] <= association_distance_m:
                    matched_track = result.all_tracks[track_index]
                    current_all_ids.add(
                        int(matched_track.track_id)
                    )
                    error = float(distances_all[actor_index, track_index])
                    current_errors.append(error)
                    frame_matches[actor_index] = {
                        "actor_id": int(actor_index),
                        "matched_track_id": int(matched_track.track_id),
                        "is_confirmed": bool(matched_track.is_confirmed),
                        "is_dynamic": bool(matched_track.is_dynamic),
                        "prediction_only_age":
                            int(matched_track.prediction_only_age),
                        "missed_count": int(matched_track.missed_count),
                        "visibility_state":
                            str(matched_track.visibility_state),
                        "attention_authorized":
                            bool(matched_track.attention_authorized),
                        "position_error_m": error,
                        "direct_observation": bool(
                            matched_track.prediction_only_age == 0
                            and matched_track.visibility_state
                            in {"direct_observation",
                                "non_image_observation"}
                        ),
                    }
        all_track_ids_by_frame.append(current_all_ids)
        position_errors_by_frame.append(current_errors)
        actor_matches_by_frame.append(frame_matches)
        attention_frames += int(torch.count_nonzero(
            result.attention_map
        ).item() > 0)
        matched = set()
        if actor_count and result.dynamic_tracks:
            gt = actor_positions[index]
            track = np.stack([
                row.position_world for row in result.dynamic_tracks
            ])
            distances = np.linalg.norm(
                gt[:, None, :]-track[None, :, :], axis=2
            )
            actor_indices, track_indices = linear_sum_assignment(distances)
            for actor_index, track_index in zip(
                actor_indices.tolist(), track_indices.tolist()
            ):
                if distances[actor_index, track_index] > association_distance_m:
                    continue
                matched.add(actor_index)
                value = result.dynamic_tracks[track_index]
                estimated_speeds[actor_index].append(
                    float(np.linalg.norm(value.velocity_world))
                )
                track_ids[actor_index].add(int(value.track_id))
        for actor_index in range(actor_count):
            if actor_index in matched:
                consecutive[actor_index] += 1
                maximum_consecutive[actor_index] = max(
                    maximum_consecutive[actor_index],
                    consecutive[actor_index],
                )
            else:
                consecutive[actor_index] = 0
            entered[actor_index] |= (
                maximum_consecutive[actor_index]
                >= int(minimum_sustained_frames)
            )
    occlusion = {
        "required": bool(require_occlusion_identity),
        "gap_start": None, "gap_end": None,
        "identity_continuous": not require_occlusion_identity,
        "prediction_position_error_max_m": None,
    }
    if require_occlusion_identity:
        visible = np.asarray(visibility_mask, dtype=bool)
        for start in range(1, len(visible)-1):
            if visible[start]:
                continue
            end = start
            while end+1 < len(visible) and not visible[end+1]:
                end += 1
            if visible[:start].any() and visible[end+1:].any():
                pre = actor_matches_by_frame[start-1][0]
                gap = [
                    actor_matches_by_frame[index][0]
                    for index in range(start, end+1)
                ]
                post = [
                    actor_matches_by_frame[index][0]
                    for index in range(
                        end+1, min(end+3, len(visible)))
                ]
                track_id = (
                    pre["matched_track_id"] if pre is not None else None)
                gap_same = bool(
                    track_id is not None
                    and all(
                        row is not None
                        and row["matched_track_id"] == track_id
                        and row["is_confirmed"]
                        and row["is_dynamic"]
                        and row["visibility_state"] == "occluded"
                        and 1 <= row["prediction_only_age"]
                        <= config.max_missed_frames
                        and row["position_error_m"]
                        <= prediction_position_error_max_m
                        for row in gap
                    )
                )
                post_same = bool(
                    len(post) >= 2
                    and track_id is not None
                    and all(
                        row is not None
                        and row["matched_track_id"] == track_id
                        and row["direct_observation"]
                        and row["prediction_only_age"] == 0
                        for row in post
                    )
                )
                pre_ready = bool(
                    pre is not None
                    and pre["is_confirmed"] and pre["is_dynamic"]
                    and pre["direct_observation"])
                identity = pre_ready and gap_same and post_same
                seen_ids = {
                    row["matched_track_id"]
                    for frame in actor_matches_by_frame
                    for row in frame if row is not None
                }
                occlusion = {
                    "required": True,
                    "gap_start": start, "gap_end": end,
                    "pre_gap_track_id": track_id,
                    "gap_track_ids": [
                        None if row is None
                        else row["matched_track_id"] for row in gap],
                    "post_gap_track_id": (
                        None if not post or post[0] is None
                        else post[0]["matched_track_id"]),
                    "identity_continuous": bool(identity),
                    "prediction_error_per_gap_frame": [
                        None if row is None
                        else row["position_error_m"] for row in gap],
                    "visibility_state_per_gap_frame": [
                        None if row is None
                        else row["visibility_state"] for row in gap],
                    "attention_state_per_gap_frame": [
                        False if row is None
                        else row["attention_authorized"] for row in gap],
                    "track_deleted": any(row is None for row in gap),
                    "replacement_track_created": bool(
                        track_id is not None
                        and any(value != track_id for value in seen_ids)),
                    "pre_gap_confirmed_dynamic": pre_ready,
                    "post_gap_direct_frames": len(post),
                    "prediction_position_error_limit_m":
                        float(prediction_position_error_max_m),
                }
                break
    base_pass = (
            bool(np.all(entered)) if actor_count
            else attention_frames == 0 and dynamic_track_total == 0
    )
    status = "PASS" if (
        base_pass and occlusion["identity_continuous"]
    ) else "FAIL"
    return {
        "status": status,
        "actor_count": actor_count,
        "actors_entered_dynamic": int(entered.sum()),
        "entered_dynamic": entered.tolist(),
        "maximum_consecutive_dynamic_frames":
            maximum_consecutive.tolist(),
        "estimated_speeds_mps": estimated_speeds,
        "track_ids": [sorted(value) for value in track_ids],
        "dynamic_track_total": dynamic_track_total,
        "all_track_total": all_track_total,
        "attention_frames": attention_frames,
        "no_target_false_attention_frames":
            attention_frames if actor_count == 0 else 0,
        "minimum_sustained_frames": int(minimum_sustained_frames),
        "occlusion": occlusion,
        "actor_frame_matches": actor_matches_by_frame,
        "future_actor_metadata_used": False,
        "coordinate_frame": "world",
        "speed_source": "frozen_linear_kalman_track_state",
    }
