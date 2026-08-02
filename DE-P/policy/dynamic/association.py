"""Deterministic one-to-one gated Hungarian data association."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .types import DynamicPerceptionConfig


def _expanded_bbox_overlap(left, right, margin):
    left = np.asarray(left, dtype=float).copy()
    right = np.asarray(right, dtype=float).copy()
    left[0] -= margin; left[1] += margin
    right[0] -= margin; right[1] += margin
    overlap = np.maximum(0.0, np.minimum(left[1], right[1]) - np.maximum(left[0], right[0]))
    intersection = float(np.prod(overlap))
    union = (float(np.prod(left[1] - left[0]))
             + float(np.prod(right[1] - right[0])) - intersection)
    return intersection / union if union > 1e-12 else 0.0


def voxel_jaccard(left, right):
    left_set, right_set = set(left), set(right)
    if not left_set and not right_set:
        return 0.0
    return len(left_set & right_set) / max(len(left_set | right_set), 1)


def associate_observations(trackers, observations, config: DynamicPerceptionConfig,
                           previous_observations=None):
    """Return matches (track_id, observation_index) plus unmatched IDs/indices."""
    track_ids = sorted(trackers)
    observation_order = sorted(
        range(len(observations)), key=lambda index: observations[index].temporary_cluster_id
    )
    if not track_ids or not observation_order:
        return (), tuple(track_ids), tuple(observation_order), np.empty(
            (len(track_ids), len(observation_order)), dtype=np.float64
        )
    invalid_cost = 1e12
    previous_observations = previous_observations or {}
    cost = np.full((len(track_ids), len(observation_order)), invalid_cost, dtype=np.float64)
    for row, track_id in enumerate(track_ids):
        tracker = trackers[track_id]
        predicted = tracker.state[:3]
        for column, observation_index in enumerate(observation_order):
            observation = observations[observation_index]
            residual = observation.centroid_world - predicted
            euclidean = float(np.linalg.norm(residual))
            covariance = tracker.innovation_covariance(observation.position_covariance)
            try:
                mahalanobis_sq = float(residual @ np.linalg.solve(covariance, residual))
            except np.linalg.LinAlgError:
                mahalanobis_sq = float("inf")
            previous = previous_observations.get(track_id)
            descriptor_cost = 0.0
            plausible = True
            if previous is not None:
                dt = max(float(tracker.last_effective_dt or config.min_dt), config.min_dt)
                raw_speed = float(np.linalg.norm(
                    observation.centroid_world - previous.centroid_world
                ) / dt)
                point_ratio = max(
                    previous.point_count / observation.point_count,
                    observation.point_count / previous.point_count,
                )
                extent_ratio = float(np.max(np.maximum(
                    previous.extent / np.maximum(observation.extent, 1e-3),
                    observation.extent / np.maximum(previous.extent, 1e-3),
                )))
                predicted_bbox = previous.bounding_box_world + (
                    tracker.state[3:] * dt
                )[None, :]
                overlap = _expanded_bbox_overlap(
                    predicted_bbox, observation.bounding_box_world,
                    config.association_bbox_margin,
                )
                voxel_overlap = voxel_jaccard(
                    previous.voxel_signature, observation.voxel_signature
                )
                shape_difference = float(np.linalg.norm(
                    previous.shape_ratios - observation.shape_ratios
                ))
                descriptor_cost = (
                    config.association_size_weight * abs(np.log(point_ratio))
                    + config.association_shape_weight * shape_difference
                    + config.association_overlap_weight * (1.0 - overlap)
                    + config.association_overlap_weight * 0.5 * (1.0 - voxel_overlap)
                )
                plausible = (
                    raw_speed <= config.physically_plausible_speed_max
                    and not (
                        point_ratio > config.association_point_count_ratio_max
                        and overlap <= 0.0
                    )
                    and not (
                        extent_ratio > config.association_extent_ratio_max
                        and overlap <= 0.0
                    )
                )
            if (plausible and euclidean <= config.association_distance_threshold
                    and mahalanobis_sq <= config.association_mahalanobis_threshold):
                cost[row, column] = mahalanobis_sq + descriptor_cost
    row_indices, column_indices = linear_sum_assignment(cost)
    matches = []
    used_tracks, used_observations = set(), set()
    for row, column in zip(row_indices.tolist(), column_indices.tolist()):
        if cost[row, column] >= invalid_cost:
            continue
        track_id = track_ids[row]
        observation_index = observation_order[column]
        matches.append((track_id, observation_index))
        used_tracks.add(track_id)
        used_observations.add(observation_index)
    matches.sort(key=lambda item: item[0])
    unmatched_tracks = tuple(track_id for track_id in track_ids if track_id not in used_tracks)
    unmatched_observations = tuple(
        index for index in observation_order if index not in used_observations
    )
    return tuple(matches), unmatched_tracks, unmatched_observations, cost
