"""Stable track IDs, lifecycle, association, and dynamic-state hysteresis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import hashlib
import numpy as np

from .association import associate_observations, voxel_jaccard
from .kalman_tracker import LinearKalmanTracker
from .types import ClusterObservation, DynamicPerceptionConfig, DynamicTrack


@dataclass
class _ManagedTrack:
    tracker: LinearKalmanTracker
    age: int = 1
    hit_count: int = 1
    missed_count: int = 0
    is_confirmed: bool = False
    is_dynamic: bool = False
    dynamic_reason: str = "new_track"
    last_observation: Optional[ClusterObservation] = None
    raw_velocity_history: list = field(default_factory=list)
    motion_consistency_count: int = 0
    association_quality: float = 0.0
    split_merge_suspected: bool = False
    confidence: float = 0.0
    confidence_components: dict = field(default_factory=dict)
    birth_observation_id: int = -1
    birth_frame: int = -1
    last_observation_id: int = -1
    last_direct_observation_frame: int = -1
    direct_observation_count: int = 0
    consecutive_direct_hits: int = 0
    prediction_only_age: int = 0
    ever_directly_observed: bool = False
    last_pixel_bbox: tuple = (-1, -1, -1, -1)
    last_pixel_mask_signature: str = ""
    attention_authorized: bool = False
    visibility_state: str = "unknown"
    ever_confirmed_dynamic: bool = False
    last_direct_observation_timestamp: float | None = None


def _bbox_iou(left, right):
    overlap = np.maximum(0.0, np.minimum(left[1], right[1]) - np.maximum(left[0], right[0]))
    intersection = float(np.prod(overlap))
    left_volume = float(np.prod(np.maximum(left[1] - left[0], 0.0)))
    right_volume = float(np.prod(np.maximum(right[1] - right[0], 0.0)))
    union = left_volume + right_volume - intersection
    return intersection / union if union > 1e-12 else 0.0


class TrackManager:
    def __init__(self, config: DynamicPerceptionConfig):
        config.validate()
        self.config = config
        self._tracks = {}
        self._next_track_id = 0
        self._last_timestamp = None
        self.last_diagnostics = {}

    def reset(self):
        self._tracks.clear()
        self._next_track_id = 0
        self._last_timestamp = None
        self.last_diagnostics = {}

    def update(self, observations, timestamp, depth_frame=None):
        timestamp = float(timestamp)
        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        if self._last_timestamp is not None:
            if timestamp == self._last_timestamp:
                raise ValueError("duplicate frame timestamp is not allowed")
            if timestamp < self._last_timestamp:
                raise ValueError("out-of-order frame timestamp is not allowed")
        for observation in observations:
            if abs(observation.timestamp - timestamp) > 1e-9:
                raise ValueError("observation timestamp does not match frame timestamp")

        for track_id in sorted(self._tracks):
            managed = self._tracks[track_id]
            managed.tracker.predict_to(timestamp)
            managed.age += 1

        tracker_map = {track_id: managed.tracker for track_id, managed in self._tracks.items()}
        matches, unmatched_tracks, unmatched_observations, cost = associate_observations(
            tracker_map, observations, self.config,
            {track_id: managed.last_observation
             for track_id, managed in self._tracks.items()
             if managed.last_observation is not None},
        )
        match_details = []
        for track_id, observation_index in matches:
            observation = observations[observation_index]
            managed = self._tracks[track_id]
            predicted = managed.tracker.state[:3]
            residual = observation.centroid_world - predicted
            innovation_covariance = managed.tracker.innovation_covariance(
                observation.position_covariance
            )
            try:
                mahalanobis_sq = float(
                    residual @ np.linalg.solve(innovation_covariance, residual)
                )
            except np.linalg.LinAlgError:
                mahalanobis_sq = float("inf")
            previous = managed.last_observation
            dt = max(timestamp - previous.timestamp, self.config.min_dt) if previous else None
            raw_velocity = (
                (observation.centroid_world - previous.centroid_world) / dt
                if previous else np.zeros(3, dtype=np.float64)
            )
            point_ratio = (
                max(previous.point_count, observation.point_count)
                / max(min(previous.point_count, observation.point_count), 1)
                if previous else 1.0
            )
            bbox_overlap = (
                _bbox_iou(previous.bounding_box_world, observation.bounding_box_world)
                if previous else 1.0
            )
            voxel_overlap = (
                voxel_jaccard(previous.voxel_signature, observation.voxel_signature)
                if previous else 0.0
            )
            split_merge = bool(
                previous is not None
                and point_ratio > self.config.association_point_count_ratio_max
                and bbox_overlap < self.config.association_bbox_margin
            )
            match_details.append({
                "track_id": int(track_id),
                "observation_index": int(observation_index),
                "temporary_cluster_id": int(observation.temporary_cluster_id),
                "association_distance": float(np.linalg.norm(residual)),
                "association_mahalanobis_sq": mahalanobis_sq,
                "innovation": residual.tolist(),
                "innovation_covariance":
                    innovation_covariance.tolist(),
                "previous_current_cluster_size_ratio": (
                    float(previous.point_count / observation.point_count) if previous else None
                ),
                "bbox_iou": (
                    bbox_overlap
                    if previous else None
                ),
                "raw_centroid_velocity": (
                    raw_velocity.tolist()
                    if previous else None
                ),
                "split_merge_suspected": split_merge,
                "voxel_jaccard": voxel_overlap,
                "point_count": int(observation.point_count),
                "extent": observation.extent.tolist(),
                "shape_ratios": observation.shape_ratios.tolist(),
            })
            previous_velocity = managed.tracker.state[3:].copy()
            managed.tracker.update(
                observation.centroid_world, observation.position_covariance
            )
            raw_speed = float(np.linalg.norm(raw_velocity))
            acceleration = (
                float(np.linalg.norm(raw_velocity - previous_velocity) / dt)
                if managed.raw_velocity_history else 0.0
            )
            hard_outlier = bool(
                split_merge or raw_speed > self.config.physically_plausible_speed_max
            )
            acceleration_outlier = bool(
                acceleration > self.config.physically_plausible_acceleration_max
            )
            velocity_outlier = hard_outlier or acceleration_outlier
            if velocity_outlier:
                managed.tracker.set_velocity(previous_velocity)
                managed.motion_consistency_count = max(
                    0, managed.motion_consistency_count - 1
                )
            else:
                managed.raw_velocity_history.append(raw_velocity.copy())
                managed.raw_velocity_history = managed.raw_velocity_history[-5:]
                robust_velocity = np.median(
                    np.stack(managed.raw_velocity_history, axis=0), axis=0
                )
                filtered_velocity = managed.tracker.state[3:]
                managed.tracker.set_velocity(0.5 * filtered_velocity + 0.5 * robust_velocity)
                history = np.stack(managed.raw_velocity_history, axis=0)
                median_speed = float(np.linalg.norm(robust_velocity))
                if len(history) < 3 or median_speed <= 1e-9:
                    coherent = True
                else:
                    forward_fraction = float(np.mean(
                        history @ robust_velocity > 0.0
                    ))
                    median_residual = float(np.median(np.linalg.norm(
                        history - robust_velocity, axis=1
                    )))
                    coherent = (
                        forward_fraction >= 2.0 / 3.0
                        and median_residual <= max(1.0, 1.5 * median_speed)
                    )
                if coherent and raw_speed >= self.config.dynamic_exit_speed:
                    managed.motion_consistency_count += 1
                else:
                    managed.motion_consistency_count = 0
            managed.split_merge_suspected = hard_outlier
            managed.association_quality = float(np.exp(
                -0.5 * mahalanobis_sq
                / max(self.config.association_mahalanobis_threshold, 1e-9)
            )) if np.isfinite(mahalanobis_sq) else 0.0
            managed.missed_count = 0
            managed.last_observation = observation
            self._record_direct_observation(managed, observation)

        for track_id in unmatched_tracks:
            managed = self._tracks[track_id]
            managed.missed_count += 1
            managed.prediction_only_age += 1
            managed.consecutive_direct_hits = 0
            managed.visibility_state = self._visibility_state(managed, depth_frame)
            if managed.visibility_state == "clear_missing":
                managed.missed_count += 1

        current_explanations = [
            observations[observation_index]
            for _track_id, observation_index in matches
        ]
        for observation_index in unmatched_observations:
            observation = observations[observation_index]
            if (observation.direct_image_evidence
                    and observation.component_pixel_count > 0
                    and observation.point_count
                    < 4 * self.config.dynamic_min_cluster_points):
                continue
            # Seeded range components can split one physical surface into
            # adjacent fragments.  Hungarian association is one-to-one, so
            # without this causal current-frame deduplication every leftover
            # fragment becomes a new tentative identity.  A genuinely
            # separate object outside one DBSCAN radius remains eligible.
            if any(
                np.linalg.norm(
                    observation.centroid_world - explained.centroid_world
                ) <= self.config.cluster_eps
                for explained in current_explanations
            ):
                continue
            track_id = self._next_track_id
            self._next_track_id += 1
            managed = _ManagedTrack(
                tracker=LinearKalmanTracker(
                    observation.centroid_world,
                    timestamp,
                    self.config,
                    observation.position_covariance,
                ),
                is_confirmed=(
                    observation.direct_image_evidence
                    and self.config.min_confirmed_hits <= 1
                ),
                hit_count=0,
                last_observation=observation,
                birth_observation_id=observation.observation_id,
                birth_frame=observation.frame_index,
            )
            self._record_direct_observation(managed, observation)
            self._tracks[track_id] = managed
            current_explanations.append(observation)

        deleted = []
        for track_id in sorted(tuple(self._tracks)):
            managed = self._tracks[track_id]
            if managed.missed_count > self.config.max_missed_frames:
                deleted.append(track_id)
                del self._tracks[track_id]
                continue
            managed.is_confirmed = managed.hit_count >= self.config.min_confirmed_hits
            self._update_confidence(managed)
            self._update_dynamic_state(managed)
            managed.ever_confirmed_dynamic = bool(
                managed.ever_confirmed_dynamic
                or (managed.is_confirmed and managed.is_dynamic)
            )
            managed.attention_authorized = bool(
                managed.is_confirmed
                and managed.is_dynamic
                and managed.ever_directly_observed
                and managed.prediction_only_age <= self.config.max_missed_frames
                and managed.visibility_state != "clear_missing"
                and managed.confidence >= self.config.track_confidence_threshold
            )

        self._last_timestamp = timestamp
        self.last_diagnostics = {
            "matches": tuple(matches),
            "unmatched_track_ids": tuple(unmatched_tracks),
            "unmatched_observation_indices": tuple(unmatched_observations),
            "deleted_track_ids": tuple(deleted),
            "association_cost_shape": tuple(cost.shape),
            "association_cost_matrix": cost.copy(),
            "association_track_ids": tuple(sorted(tracker_map)),
            "association_observation_indices": tuple(
                sorted(
                    range(len(observations)),
                    key=lambda index:
                        observations[index].temporary_cluster_id,
                )
            ),
            "association_distance_threshold":
                float(self.config.association_distance_threshold),
            "association_mahalanobis_threshold":
                float(self.config.association_mahalanobis_threshold),
            "lifecycle_update_order": (
                "predict",
                "associate",
                "matched_update",
                "unmatched_miss_increment",
                "unmatched_observation_birth",
                "delete_if_missed_count_gt_max",
                "confirmation",
                "confidence",
                "dynamic_state",
                "attention_authorization",
            ),
            "match_details": tuple(match_details),
            "next_track_id": self._next_track_id,
        }
        return self.snapshot()

    @staticmethod
    def _record_direct_observation(managed, observation):
        managed.last_observation_id = observation.observation_id
        if not observation.direct_image_evidence:
            managed.consecutive_direct_hits = 0
            managed.visibility_state = "non_image_observation"
            return
        managed.hit_count += 1
        managed.last_direct_observation_frame = observation.frame_index
        managed.last_direct_observation_timestamp = float(
            observation.timestamp
        )
        managed.direct_observation_count += 1
        managed.consecutive_direct_hits += 1
        managed.prediction_only_age = 0
        managed.ever_directly_observed = True
        managed.last_pixel_bbox = observation.pixel_bbox
        packed = np.asarray(observation.pixel_indices, dtype=np.int32).tobytes()
        managed.last_pixel_mask_signature = hashlib.sha256(packed).hexdigest()
        managed.visibility_state = "direct_observation"

    def _visibility_state(self, managed, depth_frame):
        if depth_frame is None:
            return "unknown"
        pose = depth_frame.camera_pose_world
        point_camera = pose.rotation_world_from_camera.T @ (
            managed.tracker.state[:3] - pose.position_world
        )
        predicted_depth = float(point_camera[2])
        camera = depth_frame.camera_model
        if predicted_depth <= camera.min_depth:
            return "behind_camera"
        u = int(round(camera.fx * point_camera[0] / predicted_depth + camera.cx))
        v = int(round(camera.fy * point_camera[1] / predicted_depth + camera.cy))
        if u < 0 or u >= camera.width or v < 0 or v >= camera.height:
            return "outside_fov"
        if not depth_frame.valid_mask[v, u]:
            return "clear_missing"
        measured = float(depth_frame.depth_m[v, u])
        threshold = (
            self.config.range_abs_residual_threshold
            + self.config.range_rel_residual_threshold * predicted_depth
        )
        if measured < predicted_depth - threshold:
            return "occluded"
        if measured > predicted_depth + threshold:
            return "clear_missing"
        return "surface_consistent_without_component"

    def _update_dynamic_state(self, managed):
        state = managed.tracker.state
        covariance = managed.tracker.covariance
        speed = float(np.linalg.norm(state[3:]))
        velocity_std = float(np.sqrt(max(np.max(np.diag(covariance)[3:]), 0.0)))
        enough_hits = (managed.is_confirmed
                       and managed.hit_count >= self.config.dynamic_min_confirmed_hits)
        certain = velocity_std <= self.config.dynamic_max_velocity_std
        observation = managed.last_observation
        geometric = bool(
            observation is not None
            and observation.point_count >= self.config.dynamic_min_cluster_points
            and float(np.min(observation.extent))
                >= self.config.dynamic_min_cluster_extent
            and float(np.min(observation.extent))
                <= self.config.dynamic_max_cluster_min_extent
            and float(np.max(observation.extent))
                <= self.config.dynamic_max_cluster_extent
        )
        consistent = (
            managed.motion_consistency_count >= self.config.motion_consistency_frames
        )
        if not enough_hits:
            managed.is_dynamic = False
            managed.dynamic_reason = "insufficient_confirmed_hits"
        elif not certain:
            managed.is_dynamic = False
            managed.dynamic_reason = "velocity_uncertainty_too_high"
        elif managed.is_dynamic:
            if speed <= self.config.dynamic_exit_speed:
                managed.is_dynamic = False
                managed.dynamic_reason = "speed_below_exit_threshold"
            elif managed.confidence < self.config.track_confidence_threshold:
                managed.is_dynamic = False
                managed.dynamic_reason = "track_confidence_too_low"
            elif managed.split_merge_suspected or not geometric:
                managed.dynamic_reason = "degraded_observation_hold_dynamic"
            else:
                managed.dynamic_reason = "hysteresis_hold_dynamic"
        elif managed.split_merge_suspected:
            managed.dynamic_reason = "split_merge_or_velocity_outlier"
        elif not geometric:
            managed.dynamic_reason = "cluster_geometry_not_dynamic_candidate"
        elif not consistent:
            managed.dynamic_reason = "motion_not_yet_consistent"
        elif managed.confidence < self.config.track_confidence_threshold:
            managed.dynamic_reason = "track_confidence_too_low"
        elif speed >= self.config.dynamic_enter_speed:
            managed.is_dynamic = True
            managed.dynamic_reason = "speed_above_enter_threshold"
        else:
            managed.dynamic_reason = "speed_below_enter_threshold"

    def _update_confidence(self, managed):
        covariance = managed.tracker.covariance
        velocity_std = float(np.sqrt(max(np.max(np.diag(covariance)[3:]), 0.0)))
        confirmation = min(
            1.0, managed.hit_count / max(self.config.dynamic_min_confirmed_hits, 1)
        )
        association = float(np.clip(managed.association_quality, 0.0, 1.0))
        uncertainty = float(np.exp(
            -velocity_std / max(self.config.dynamic_max_velocity_std, 1e-9)
        ))
        consistency = min(
            1.0,
            managed.motion_consistency_count / max(self.config.motion_consistency_frames, 1),
        )
        foreground = float(
            managed.last_observation.foreground_support
            if managed.last_observation is not None else 0.0
        )
        observation = managed.last_observation
        geometry = float(bool(
            observation is not None
            and observation.point_count >= self.config.dynamic_min_cluster_points
            and float(np.min(observation.extent))
                >= self.config.dynamic_min_cluster_extent
            and float(np.min(observation.extent))
                <= self.config.dynamic_max_cluster_min_extent
            and float(np.max(observation.extent))
                <= self.config.dynamic_max_cluster_extent
        ))
        confidence = (
            0.15 * confirmation
            + 0.25 * association
            + 0.15 * uncertainty
            + 0.25 * consistency
            + 0.10 * foreground
            + 0.10 * geometry
        )
        if managed.split_merge_suspected:
            confidence *= 0.75 if managed.is_dynamic else 0.25
        if managed.missed_count:
            confidence *= self.config.confidence_missed_decay ** managed.missed_count
        managed.confidence = float(np.clip(confidence, 0.0, 1.0))
        managed.confidence_components = {
            "confirmation": confirmation,
            "association": association,
            "uncertainty": uncertainty,
            "motion_consistency": consistency,
            "foreground_support": foreground,
            "cluster_geometry": geometry,
        }

    def snapshot(self):
        snapshots = []
        for track_id in sorted(self._tracks):
            managed = self._tracks[track_id]
            state = managed.tracker.state
            snapshots.append(DynamicTrack(
                track_id=track_id,
                position_world=state[:3],
                velocity_world=state[3:],
                state_covariance=managed.tracker.covariance,
                age=managed.age,
                hit_count=managed.hit_count,
                missed_count=managed.missed_count,
                is_confirmed=managed.is_confirmed,
                is_dynamic=managed.is_dynamic,
                timestamp=managed.tracker.last_timestamp,
                dynamic_reason=managed.dynamic_reason,
                confidence=managed.confidence,
                confidence_components=managed.confidence_components,
                birth_observation_id=managed.birth_observation_id,
                birth_frame=managed.birth_frame,
                last_observation_id=managed.last_observation_id,
                last_direct_observation_frame=managed.last_direct_observation_frame,
                direct_observation_count=managed.direct_observation_count,
                consecutive_direct_hits=managed.consecutive_direct_hits,
                prediction_only_age=managed.prediction_only_age,
                ever_directly_observed=managed.ever_directly_observed,
                last_pixel_bbox=managed.last_pixel_bbox,
                last_pixel_mask_signature=managed.last_pixel_mask_signature,
                attention_authorized=managed.attention_authorized,
                visibility_state=managed.visibility_state,
                ever_confirmed_dynamic=managed.ever_confirmed_dynamic,
                last_direct_observation_timestamp=(
                    managed.last_direct_observation_timestamp
                ),
                last_observed_extent=(
                    (0.0, 0.0, 0.0)
                    if managed.last_observation is None else
                    tuple(float(value) for value in managed.last_observation.extent)
                ),
            ))
        return tuple(snapshots)
