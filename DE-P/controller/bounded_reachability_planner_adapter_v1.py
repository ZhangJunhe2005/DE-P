"""Atomic, feature-gated adapter from frozen YOPO candidates to BDRR1."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time
from typing import Callable

import numpy as np

from controller.dynamic_safety_decision_router_v1 import (
    FeatureMode, route_dynamic_safety_decision,
)
from policy.dynamic.asynchronous_multi_target_risk_v1 import (
    RISK_VERSION, evaluate_asynchronous_reachability_risk,
)
from policy.dynamic.bounded_dynamic_reachability_v1 import (
    DynamicReachabilityStateV1, REACHABILITY_VERSION,
)
from policy.dynamic.shape_reachable_occupancy_v1 import OCCUPANCY_VERSION
from policy.dynamic.stale_geometry_time_contract_v1 import TIME_CONTRACT_VERSION


ADAPTER_VERSION = "bounded_reachability_planner_adapter_v1"
SNAPSHOT_VERSION = "planner_safety_snapshot_v1"


def _readonly(value, dtype):
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _array_hash(hasher, value):
    array = np.ascontiguousarray(value)
    hasher.update(str(array.dtype).encode())
    hasher.update(str(array.shape).encode())
    hasher.update(array.tobytes())


@dataclass(frozen=True)
class TrackSnapshotIdentityV1:
    track_id: int
    generation: str
    source_timestamp: float


@dataclass(frozen=True)
class PlannerSafetySnapshotV1:
    frame_index: int
    candidate_frame_index: int
    track_frame_index: int
    reachability_frame_index: int
    query_timestamp: float
    camera_pose_timestamp: float
    track_snapshot_timestamp: float
    reachability_snapshot_timestamp: float
    camera_pose_world_from_camera: np.ndarray
    candidate_set_id: str
    candidate_ids: np.ndarray
    candidate_positions: np.ndarray
    candidate_times: np.ndarray
    candidate_scores: np.ndarray
    candidate_time_origin: float
    track_snapshot_id: str
    reachability_snapshot_id: str
    track_identities: tuple[TrackSnapshotIdentityV1, ...]
    reachability_states: tuple[DynamicReachabilityStateV1, ...]
    unresolved_dynamic_risk: bool
    contract_versions: tuple[str, ...]
    validity: str
    construction_digest: str

    @classmethod
    def create(
        cls, *, frame_index, query_timestamp, camera_pose_timestamp,
        camera_pose_world_from_camera, candidate_set_id, candidate_ids,
        candidate_positions, candidate_times, candidate_scores,
        candidate_time_origin, track_snapshot_id,
        reachability_snapshot_id, track_identities,
        reachability_states, validity="VALID",
        candidate_frame_index=None, track_frame_index=None,
        reachability_frame_index=None, track_snapshot_timestamp=None,
        reachability_snapshot_timestamp=None,
        unresolved_dynamic_risk=False,
    ):
        pose = _readonly(camera_pose_world_from_camera, np.float64)
        ids = _readonly(candidate_ids, np.int64)
        positions = _readonly(candidate_positions, np.float64)
        times = _readonly(candidate_times, np.float64)
        scores = _readonly(candidate_scores, np.float64)
        identities = tuple(
            item if isinstance(item, TrackSnapshotIdentityV1)
            else TrackSnapshotIdentityV1(**item)
            for item in track_identities
        )
        states = tuple(reachability_states)
        base = cls(
            int(frame_index),
            int(frame_index if candidate_frame_index is None
                else candidate_frame_index),
            int(frame_index if track_frame_index is None
                else track_frame_index),
            int(frame_index if reachability_frame_index is None
                else reachability_frame_index),
            float(query_timestamp), float(camera_pose_timestamp),
            float(query_timestamp if track_snapshot_timestamp is None
                  else track_snapshot_timestamp),
            float(query_timestamp if reachability_snapshot_timestamp is None
                  else reachability_snapshot_timestamp),
            pose, str(candidate_set_id),
            ids, positions, times, scores, float(candidate_time_origin),
            str(track_snapshot_id), str(reachability_snapshot_id),
            identities, states, bool(unresolved_dynamic_risk),
            (
                SNAPSHOT_VERSION, TIME_CONTRACT_VERSION,
                REACHABILITY_VERSION, OCCUPANCY_VERSION, RISK_VERSION,
            ),
            str(validity), "",
        )
        object.__setattr__(base, "construction_digest", snapshot_digest(base))
        return base


def snapshot_digest(snapshot):
    hasher = hashlib.sha256()
    for value in (
        snapshot.frame_index, snapshot.query_timestamp,
        snapshot.candidate_frame_index, snapshot.track_frame_index,
        snapshot.reachability_frame_index,
        snapshot.camera_pose_timestamp, snapshot.candidate_set_id,
        snapshot.track_snapshot_timestamp,
        snapshot.reachability_snapshot_timestamp,
        snapshot.candidate_time_origin, snapshot.track_snapshot_id,
        snapshot.reachability_snapshot_id,
        snapshot.unresolved_dynamic_risk, snapshot.validity,
    ):
        hasher.update(repr(value).encode())
    for value in (
        snapshot.camera_pose_world_from_camera, snapshot.candidate_ids,
        snapshot.candidate_positions, snapshot.candidate_times,
        snapshot.candidate_scores,
    ):
        _array_hash(hasher, value)
    for item in snapshot.track_identities:
        hasher.update(repr((
            item.track_id, item.generation, item.source_timestamp
        )).encode())
    for state in snapshot.reachability_states:
        hasher.update(repr((
            state.track_id, state.generation, state.hypothesis_id,
            state.source_geometry_timestamp, state.state_timestamp,
            state.query_timestamp, state.geometry_age_s, state.status.value,
        )).encode())
        for interval in (
            state.position_set_world, state.velocity_set_world_mps,
            state.acceleration_set_world_mps2,
        ):
            _array_hash(hasher, interval.lower)
            _array_hash(hasher, interval.upper)
    return hasher.hexdigest()


def validate_snapshot(snapshot, config):
    errors = []
    tolerance = float(config["snapshot"]["timestamp_tolerance_s"])
    if snapshot.validity != "VALID":
        errors.append("snapshot_marked_invalid")
    if len({
        snapshot.frame_index, snapshot.candidate_frame_index,
        snapshot.track_frame_index, snapshot.reachability_frame_index,
    }) != 1:
        errors.append("mixed_frame_snapshot")
    arrays = (
        snapshot.camera_pose_world_from_camera, snapshot.candidate_positions,
        snapshot.candidate_times, snapshot.candidate_scores,
    )
    if not all(np.isfinite(value).all() for value in arrays):
        errors.append("non_finite_snapshot_state")
    if snapshot.camera_pose_world_from_camera.shape != (4, 4):
        errors.append("invalid_camera_pose_shape")
    if snapshot.candidate_positions.ndim != 3 or (
        snapshot.candidate_positions.shape[-1:] != (3,)
    ):
        errors.append("invalid_candidate_shape")
    count = len(snapshot.candidate_ids)
    if count == 0 or snapshot.candidate_positions.shape[0] != count:
        errors.append("missing_or_misaligned_candidates")
    if snapshot.candidate_scores.shape != (count,):
        errors.append("misaligned_candidate_scores")
    if snapshot.candidate_positions.ndim == 3 and (
        snapshot.candidate_positions.shape[1] != len(snapshot.candidate_times)
    ):
        errors.append("misaligned_candidate_times")
    if len(set(snapshot.candidate_ids.tolist())) != count:
        errors.append("duplicate_candidate_id")
    if np.any(snapshot.candidate_times < 0):
        errors.append("negative_candidate_time")
    if abs(snapshot.candidate_time_origin-snapshot.query_timestamp) > tolerance:
        errors.append("candidate_time_origin_mismatch")
    if abs(snapshot.camera_pose_timestamp-snapshot.query_timestamp) > tolerance:
        errors.append("camera_pose_timestamp_mismatch")
    if abs(snapshot.track_snapshot_timestamp-snapshot.query_timestamp) > tolerance:
        errors.append("track_snapshot_timestamp_mismatch")
    if (
        snapshot.query_timestamp-snapshot.reachability_snapshot_timestamp
        > float(config["snapshot"]["maximum_reachability_snapshot_age_s"])
        or snapshot.reachability_snapshot_timestamp
        > snapshot.query_timestamp+tolerance
    ):
        errors.append("stale_reachability_snapshot")
    identity_keys = [
        (item.track_id, item.generation) for item in snapshot.track_identities
    ]
    if len(identity_keys) != len(set(identity_keys)):
        errors.append("duplicate_track_identity")
    identity_lookup = set(identity_keys)
    state_keys = []
    for state in snapshot.reachability_states:
        key = (state.track_id, state.generation, state.hypothesis_id)
        state_keys.append(key)
        if (state.track_id, state.generation) not in identity_lookup:
            errors.append("track_generation_mismatch")
        if abs(state.query_timestamp-snapshot.query_timestamp) > tolerance:
            errors.append("mixed_frame_reachability_timestamp")
        if state.source_geometry_timestamp > snapshot.query_timestamp+tolerance:
            errors.append("future_geometry_timestamp")
        if (
            snapshot.query_timestamp-state.query_timestamp
            > float(config["snapshot"][
                "maximum_reachability_snapshot_age_s"
            ])
        ):
            errors.append("stale_reachability_snapshot")
        if not all(np.isfinite(interval).all() for interval in (
            state.position_set_world.lower, state.position_set_world.upper,
            state.velocity_set_world_mps.lower,
            state.velocity_set_world_mps.upper,
            state.acceleration_set_world_mps2.lower,
            state.acceleration_set_world_mps2.upper,
        )):
            errors.append("non_finite_reachability_state")
    if len(state_keys) != len(set(state_keys)):
        errors.append("duplicate_reachability_identity")
    if snapshot_digest(snapshot) != snapshot.construction_digest:
        errors.append("snapshot_mutated_after_construction")
    return tuple(dict.fromkeys(errors))


class BoundedReachabilityPlannerAdapterV1:
    version = ADAPTER_VERSION

    def __init__(
        self, *, config, reachability_builder, mode=None,
        risk_evaluator: Callable = evaluate_asynchronous_reachability_risk,
        development_launcher=False,
    ):
        self.config = config
        integration = config["dynamic_reachability_integration"]
        self.enabled = bool(integration["enabled"])
        self.mode = FeatureMode(mode or integration["mode"])
        if self.mode != FeatureMode.LEGACY_OFF:
            if not development_launcher:
                raise ValueError("active integration requires development launcher")
            self.enabled = True
        if self.mode == FeatureMode.LEGACY_OFF and self.enabled:
            raise ValueError("enabled integration cannot use LEGACY_OFF")
        self.builder = reachability_builder
        self.risk_evaluator = risk_evaluator
        self.risk_call_count = 0
        self._last_frame_index = None
        self._last_query_timestamp = None

    def reset(self):
        self._last_frame_index = None
        self._last_query_timestamp = None

    def evaluate(self, snapshot, original_candidate_id):
        started = time.perf_counter()
        original = int(original_candidate_id)
        if self.mode == FeatureMode.LEGACY_OFF:
            routed = route_dynamic_safety_decision(
                mode=self.mode, decision_status="LEGACY_OFF_BYPASS",
                original_candidate_id=original,
                recommended_candidate_id=None, safe_candidate_ids=(),
            )
            return self._result(snapshot, routed, None, (), started, ())
        validation_started = time.perf_counter()
        errors = list(validate_snapshot(snapshot, self.config))
        if (
            self._last_frame_index is not None
            and (
                snapshot.frame_index <= self._last_frame_index
                or snapshot.query_timestamp <= self._last_query_timestamp
            )
        ):
            errors.append("timestamp_or_frame_rollback")
        errors = tuple(dict.fromkeys(errors))
        validation_ms = (time.perf_counter()-validation_started)*1000
        if errors:
            routed = route_dynamic_safety_decision(
                mode=self.mode, decision_status="INVALID_EVALUATION",
                original_candidate_id=original,
                recommended_candidate_id=None, safe_candidate_ids=(),
            )
            return self._result(
                snapshot, routed, None, errors, started, (),
                validation_ms=validation_ms,
            )
        before = snapshot_digest(snapshot)
        try:
            self.risk_call_count += 1
            risk = self.risk_evaluator(
                snapshot.candidate_positions, snapshot.candidate_times,
                snapshot.reachability_states, self.builder,
                candidate_scores=snapshot.candidate_scores,
                unresolved_dynamic_risk=snapshot.unresolved_dynamic_risk,
            )
            self._last_frame_index = snapshot.frame_index
            self._last_query_timestamp = snapshot.query_timestamp
            if snapshot_digest(snapshot) != before:
                raise RuntimeError("snapshot changed during risk query")
            safe = tuple(
                int(row["candidate_trajectory_id"])
                for row in risk["candidate_rows"] if not row["would_veto"]
            )
            routed = route_dynamic_safety_decision(
                mode=self.mode, decision_status=risk["decision_status"],
                original_candidate_id=original,
                recommended_candidate_id=risk["recommended_candidate_id"],
                safe_candidate_ids=safe,
            )
            return self._result(
                snapshot, routed, risk, (), started, safe,
                validation_ms=validation_ms,
            )
        except Exception as error:
            routed = route_dynamic_safety_decision(
                mode=self.mode, decision_status="INVALID_EVALUATION",
                original_candidate_id=original,
                recommended_candidate_id=None, safe_candidate_ids=(),
            )
            return self._result(
                snapshot, routed, None,
                (f"adapter_exception:{type(error).__name__}:{error}",),
                started, (), validation_ms=validation_ms,
            )

    def _result(
        self, snapshot, routed, risk, errors, started, safe,
        validation_ms=0.,
    ):
        rows = () if risk is None else tuple(risk["candidate_rows"])
        limiting = min(
            (
                row for row in rows
                if np.isfinite(row["robust_minimum_clearance_m"])
            ),
            key=lambda row: row["robust_minimum_clearance_m"],
            default=None,
        )
        witness = None if limiting is None else limiting["witness"]
        original_score = (
            float(snapshot.candidate_scores[
                np.where(snapshot.candidate_ids == routed.original_candidate_id)[0][0]
            ])
            if routed.original_candidate_id in snapshot.candidate_ids else None
        )
        selected_score = (
            None if routed.selected_candidate_id is None
            else float(snapshot.candidate_scores[
                np.where(snapshot.candidate_ids == routed.selected_candidate_id)[0][0]
            ])
        )
        total_ms = (time.perf_counter()-started)*1000
        return {
            "integration_version": self.version,
            "contract_version": self.config["contract_version"],
            "feature_mode": self.mode.value,
            "shadow_only": self.mode == FeatureMode.SHADOW,
            "frame_index": snapshot.frame_index,
            "query_timestamp": snapshot.query_timestamp,
            "decision_status": routed.decision_status,
            "original_candidate_id": routed.original_candidate_id,
            "recommended_candidate_id": routed.selected_candidate_id,
            "shadow_recommended_candidate_id":
                routed.shadow_recommended_candidate_id,
            "original_candidate_score": original_score,
            "recommended_candidate_score": selected_score,
            "active_track_count": len(snapshot.track_identities),
            "active_hypothesis_count": len(snapshot.reachability_states),
            "limiting_track":
                None if witness is None else witness["limiting_track"],
            "limiting_hypothesis":
                None if witness is None else witness["limiting_hypothesis"],
            "minimum_robust_clearance_m":
                None if limiting is None
                else limiting["robust_minimum_clearance_m"],
            "safe_candidate_ids": list(safe),
            "safe_abort": routed.safe_abort,
            "control_disposition": routed.disposition.value,
            "reason": routed.reason,
            "snapshot_validity": "VALID" if not errors else "INVALID",
            "snapshot_errors": list(errors),
            "input_validation_ms": validation_ms,
            "reachability_risk_ms":
                0. if risk is None else risk["total_runtime_ms"],
            "total_integration_overhead_ms": total_ms,
            "formal_command_modified": routed.formal_command_modified,
            "runtime_gt_used": False,
        }


__all__ = [
    "ADAPTER_VERSION", "SNAPSHOT_VERSION", "TrackSnapshotIdentityV1",
    "PlannerSafetySnapshotV1", "snapshot_digest", "validate_snapshot",
    "BoundedReachabilityPlannerAdapterV1",
]
