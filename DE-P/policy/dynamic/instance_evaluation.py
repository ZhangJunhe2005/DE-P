"""Instance-only evaluation helpers.

This module is deliberately outside the runtime perception path.  It consumes
simulator instance IDs only after :class:`DynamicPerception` has returned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from scipy.optimize import linear_sum_assignment

from .types import DynamicPerceptionResult


def observation_actor_overlap(observation, instance_ids, actor_id):
    flat = np.asarray(instance_ids, dtype=np.int32).reshape(-1)
    indices = np.asarray(observation.pixel_indices, dtype=np.int64)
    if not len(indices):
        return 0, 0.0
    if np.any(indices < 0) or np.any(indices >= len(flat)):
        raise ValueError("observation pixel provenance lies outside instance image")
    count = int(np.count_nonzero(flat[indices] == int(actor_id)))
    return count, count / len(indices)


@dataclass
class InstanceEvaluationState:
    """Causal identity state for one sequence."""

    occlusion_window: int = 5
    observed_actor_last_frame: dict = field(default_factory=dict)
    track_actor_identity: dict = field(default_factory=dict)
    seen_track_ids: set = field(default_factory=set)
    events: list = field(default_factory=list)
    counters: dict = field(default_factory=lambda: {
        "visible_actor_frames": 0,
        "visible_actor_matches": 0,
        "track_births": 0,
        "grounded_track_births": 0,
        "confirmed_tracks": 0,
        "grounded_confirmed_tracks": 0,
        "attention_tracks": 0,
        "grounded_attention_tracks": 0,
        "never_observed_proximity_event_count": 0,
        "never_observed_illegal_observation_count": 0,
        "never_observed_ungrounded_track_count": 0,
        "never_observed_attention_event_count": 0,
        "prediction_only_confirmed_track_count": 0,
        "prediction_only_attention_count": 0,
        "frame_record_count": 0,
    })

    def _event(self, sequence_id, frame, track, actor, reason, overlap=0.0,
               distance=float("nan"), attention=0.0):
        self.events.append({
            "sequence_id": str(sequence_id),
            "frame": int(frame),
            "track_id": int(track.track_id) if track is not None else -1,
            "actor_id": int(actor.get("object_id", -1)) if actor is not None else -1,
            "birth_frame": int(track.birth_frame) if track is not None else -1,
            "first_direct_observation_frame": (
                int(track.birth_frame) if track is not None
                and track.ever_directly_observed else -1
            ),
            "last_direct_observation_frame": (
                int(track.last_direct_observation_frame) if track is not None else -1
            ),
            "component_pixel_overlap": float(overlap),
            "world_distance": float(distance),
            "attention_value": float(attention),
            "classification_reason": str(reason),
        })

    def update(self, sequence_id, frame, actors, instance_ids,
               result: DynamicPerceptionResult, never_observed_actor_ids=None):
        instance = np.asarray(instance_ids)
        if instance.ndim != 2 or instance.dtype != np.int32:
            raise ValueError("instance_ids must be an int32 HxW array")
        actor_by_id = {int(actor["object_id"]): actor for actor in actors}
        mask_ids = set(int(x) for x in np.unique(instance) if int(x) != 0)
        unknown = mask_ids - set(actor_by_id)
        if unknown:
            raise ValueError(f"instance image contains IDs absent from GT: {sorted(unknown)}")

        for actor_id, actor in actor_by_id.items():
            pixels = int(np.count_nonzero(instance == actor_id))
            declared = int(actor.get("rendered_pixel_count", pixels))
            if pixels != declared:
                raise ValueError(
                    f"instance/depth GT count mismatch actor={actor_id}: {pixels}!={declared}"
                )
            if pixels:
                self.observed_actor_last_frame[actor_id] = int(frame)
                self.counters["visible_actor_frames"] += 1
            elif actor.get("visible", False):
                self.counters["never_observed_illegal_observation_count"] += 1

        observation_by_id = {
            int(observation.observation_id): observation
            for observation in result.observations
        }
        current_identity = {}
        for track in result.all_tracks:
            observation = observation_by_id.get(int(track.last_observation_id))
            if observation is not None and observation.frame_index == frame:
                overlaps = []
                for actor_id in mask_ids:
                    count, fraction = observation_actor_overlap(
                        observation, instance, actor_id
                    )
                    if count:
                        overlaps.append((count, fraction, actor_id))
                if overlaps:
                    _, overlap, actor_id = max(overlaps)
                    current_identity[track.track_id] = actor_id
                    self.track_actor_identity[track.track_id] = actor_id
                    self._event(sequence_id, frame, track, actor_by_id[actor_id],
                                "direct_instance_overlap", overlap=overlap,
                                distance=np.linalg.norm(
                                    track.position_world
                                    - np.asarray(actor_by_id[actor_id]["position_world"])
                                ))

            if track.track_id not in self.seen_track_ids:
                self.seen_track_ids.add(track.track_id)
                self.counters["track_births"] += 1
                if track.track_id in current_identity:
                    self.counters["grounded_track_births"] += 1
            if track.is_confirmed:
                self.counters["confirmed_tracks"] += 1
                if track.ever_directly_observed:
                    self.counters["grounded_confirmed_tracks"] += 1
                else:
                    self.counters["prediction_only_confirmed_track_count"] += 1
            if track.attention_authorized:
                self.counters["attention_tracks"] += 1
                if track.ever_directly_observed:
                    self.counters["grounded_attention_tracks"] += 1
                else:
                    self.counters["prediction_only_attention_count"] += 1
            if (track.is_confirmed or track.attention_authorized) and not (
                    track.ever_directly_observed):
                self.counters["never_observed_ungrounded_track_count"] += 1
                self._event(sequence_id, frame, track, None, "ungrounded_track")

        eligible_actor_ids = {
            actor_id for actor_id, last in self.observed_actor_last_frame.items()
            if frame - last <= self.occlusion_window
        }
        track_ids = [track.track_id for track in result.all_tracks]
        tracks = {track.track_id: track for track in result.all_tracks}
        if track_ids and eligible_actor_ids:
            actor_ids = sorted(eligible_actor_ids)
            cost = np.full((len(track_ids), len(actor_ids)), 1e9, dtype=float)
            for row, track_id in enumerate(track_ids):
                track = tracks[track_id]
                for column, actor_id in enumerate(actor_ids):
                    actor = actor_by_id.get(actor_id)
                    if actor is None:
                        continue
                    identity = current_identity.get(
                        track_id, self.track_actor_identity.get(track_id)
                    )
                    if identity != actor_id:
                        continue
                    radius = max(float(actor.get("radius", 0.3)), 0.1)
                    distance = np.linalg.norm(
                        track.position_world - np.asarray(actor["position_world"])
                    )
                    if distance <= max(1.0, 3.0 * radius):
                        cost[row, column] = distance / radius
            rows, columns = linear_sum_assignment(cost)
            matched_actors = {
                actor_ids[column] for row, column in zip(rows, columns)
                if cost[row, column] < 1e9
            }
            self.counters["visible_actor_matches"] += len(
                matched_actors & mask_ids
            )

        never_observed = (
            set(int(x) for x in never_observed_actor_ids)
            if never_observed_actor_ids is not None
            else set(actor_by_id) - set(self.observed_actor_last_frame)
        )
        never_observed &= set(actor_by_id)
        for actor_id in never_observed:
            actor = actor_by_id[actor_id]
            actor_position = np.asarray(actor["position_world"], dtype=float)
            radius = max(float(actor.get("radius", 0.3)), 0.1)
            for track in result.all_tracks:
                distance = float(np.linalg.norm(track.position_world - actor_position))
                if distance > max(1.0, 3.0 * radius):
                    continue
                self.counters["never_observed_proximity_event_count"] += 1
                self._event(sequence_id, frame, track, actor, "proximity_only",
                            distance=distance)
                # An attention-carrying track already grounded to a different
                # visible instance is not an observation of this never-seen
                # actor.  It remains a proximity-only diagnostic.
                grounded_elsewhere = (
                    self.track_actor_identity.get(track.track_id) is not None
                )
                if track.attention_authorized and not grounded_elsewhere:
                    self.counters["never_observed_attention_event_count"] += 1
                    self._event(
                        sequence_id, frame, track, actor,
                        "false_attention_near_never_observed",
                        distance=distance, attention=float(track.confidence),
                    )
        self.counters["frame_record_count"] += 1

    def summary(self):
        def ratio(numerator, denominator):
            return float(numerator / denominator) if denominator else 1.0

        counters = dict(self.counters)
        unique = {
            (event["track_id"], event["actor_id"], event["classification_reason"])
            for event in self.events
        }
        return {
            **counters,
            "visible_instance_recall": ratio(
                counters["visible_actor_matches"], counters["visible_actor_frames"]
            ),
            "track_birth_instance_precision": ratio(
                counters["grounded_track_births"], counters["track_births"]
            ),
            "confirmed_track_instance_precision": ratio(
                counters["grounded_confirmed_tracks"], counters["confirmed_tracks"]
            ),
            "attention_instance_precision": ratio(
                counters["grounded_attention_tracks"], counters["attention_tracks"]
            ),
            "unique_track_actor_event_count": len(unique),
            "events": list(self.events),
        }
