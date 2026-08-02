"""Development-only bounded coasting safety adapter.

This module is planner-facing and read-only with respect to TrackManager.  It
keeps a bounded memory derived only from runtime track snapshots.  Ground truth
is intentionally absent from every runtime interface.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
import time

import numpy as np


ADAPTER_VERSION = "dynamic_safety_shadow_adapter_v2"
CONTRACT_VERSION = "occlusion_coasting_safety_contract_v1_candidate"


class SafetyState(str, Enum):
    OBSERVED_DYNAMIC = "OBSERVED_DYNAMIC"
    COASTING_DYNAMIC = "COASTING_DYNAMIC"
    REACQUIRED_UNCERTAIN = "REACQUIRED_UNCERTAIN"
    OBSERVED_NON_DYNAMIC = "OBSERVED_NON_DYNAMIC"
    EXPIRED = "EXPIRED"
    INVALID = "INVALID"


class DecisionStatus(str, Enum):
    KEEP_ORIGINAL = "KEEP_ORIGINAL"
    SWITCH_TO_SAFE_CANDIDATE = "SWITCH_TO_SAFE_CANDIDATE"
    NO_SAFE_CANDIDATE = "NO_SAFE_CANDIDATE"
    NO_ACTIVE_DYNAMIC_RISK = "NO_ACTIVE_DYNAMIC_RISK"
    INVALID_EVALUATION = "INVALID_EVALUATION"


@dataclass(frozen=True)
class CoastingSafetyConfig:
    maximum_coasting_misses: int = 2
    maximum_coasting_time_s: float = .25
    maximum_position_std_m: float = .60
    maximum_velocity_std_mps: float = 1.0
    covariance_sigma: float = 2.5
    process_noise_acceleration: float = 1.0
    uav_radius_m: float = .30
    default_actor_radius_m: float = .30
    clearance_margin_m: float = .10
    maximum_reacquired_frames: int = 2
    static_reclassification_direct_hits: int = 3
    dynamic_exit_speed_mps: float = .15
    maximum_state_jump_m: float = 2.0
    maximum_velocity_jump_mps: float = 4.0

    def validate(self):
        numeric = asdict(self)
        if not all(np.isfinite(value) for value in numeric.values()):
            raise ValueError("coasting config must be finite")
        if self.maximum_coasting_misses < 1:
            raise ValueError("maximum_coasting_misses must be positive")
        if self.maximum_coasting_time_s <= 0:
            raise ValueError("maximum_coasting_time_s must be positive")
        if self.maximum_position_std_m <= 0 or self.maximum_velocity_std_mps <= 0:
            raise ValueError("uncertainty bounds must be positive")
        if self.covariance_sigma <= 0:
            raise ValueError("covariance_sigma must be positive")


@dataclass
class RecentDynamicEvidence:
    track_id: int
    state_generation: str
    last_dynamic_frame: int
    last_dynamic_timestamp: float
    last_direct_measurement_frame: int
    last_direct_measurement_timestamp: float
    dynamic_streak_before_loss: int
    confirmed_age_before_loss: int
    last_position_world: np.ndarray
    last_velocity_world: np.ndarray
    reacquired_frames: int = 0
    expiry_reason: str | None = None


def _value(track, name, default=None):
    if isinstance(track, dict):
        return track.get(name, default)
    return getattr(track, name, default)


def _generation(track):
    explicit = _value(track, "state_generation")
    if explicit is not None:
        return str(explicit)
    return (
        f"{int(_value(track, 'birth_frame', -1))}:"
        f"{int(_value(track, 'birth_observation_id', -1))}"
    )


def _covariance(track):
    value = np.asarray(_value(track, "state_covariance"), dtype=np.float64)
    if value.shape != (6, 6) or not np.isfinite(value).all():
        raise ValueError("state covariance must be finite [6,6]")
    if not np.allclose(value, value.T, atol=1e-8):
        raise ValueError("state covariance must be symmetric")
    if np.min(np.linalg.eigvalsh(value)) < -1e-8:
        raise ValueError("state covariance must be positive semidefinite")
    return value


def _uncertainty(covariance):
    position_std = math.sqrt(max(
        float(np.linalg.eigvalsh(covariance[:3, :3]).max()), 0.
    ))
    velocity_std = math.sqrt(max(
        float(np.linalg.eigvalsh(covariance[3:, 3:]).max()), 0.
    ))
    return position_std, velocity_std


def predict_state(position, velocity, covariance, times, acceleration_noise):
    """Frozen constant-velocity prediction without mutating a Kalman object."""
    position = np.asarray(position, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    predicted_positions, predicted_covariances = [], []
    q = float(acceleration_noise) ** 2
    for elapsed in times:
        transition = np.eye(6)
        transition[:3, 3:] = np.eye(3) * elapsed
        process = np.block([
            [np.eye(3) * elapsed**4 / 4 * q,
             np.eye(3) * elapsed**3 / 2 * q],
            [np.eye(3) * elapsed**3 / 2 * q,
             np.eye(3) * elapsed**2 * q],
        ])
        predicted_positions.append(position + elapsed * velocity)
        predicted_covariances.append(
            transition @ covariance @ transition.T + process
        )
    return np.asarray(predicted_positions), np.asarray(predicted_covariances)


class BoundedCoastingSafetyAdapter:
    """Stateful shadow adapter; one instance must serve one temporal stream."""

    def __init__(self, config=None):
        self.config = config or CoastingSafetyConfig()
        self.config.validate()
        self._memory: dict[int, RecentDynamicEvidence] = {}
        self._last_timestamp = None
        self._last_frame = None

    @property
    def memory(self):
        return {
            track_id: {
                **asdict(evidence),
                "last_position_world":
                    evidence.last_position_world.tolist(),
                "last_velocity_world":
                    evidence.last_velocity_world.tolist(),
            }
            for track_id, evidence in self._memory.items()
        }

    def reset(self):
        self._memory.clear()
        self._last_timestamp = None
        self._last_frame = None

    def _expire(self, track_id, reason, expiry_rows):
        evidence = self._memory.pop(track_id, None)
        if evidence is not None:
            evidence.expiry_reason = reason
            expiry_rows.append({
                "track_id": int(track_id),
                "state_generation": evidence.state_generation,
                "expiry_reason": reason,
            })

    def update_safety_states(self, tracks, timestamp, frame_index):
        timestamp = float(timestamp)
        frame_index = int(frame_index)
        if not np.isfinite(timestamp):
            return [], [], [], "non_finite_timestamp"
        if (
            self._last_timestamp is not None
            and (timestamp <= self._last_timestamp or frame_index <= self._last_frame)
        ):
            return [], [], [], "timestamp_or_frame_not_monotonic"
        track_ids = [int(_value(track, "track_id", -1)) for track in tracks]
        if len(track_ids) != len(set(track_ids)) or any(value < 0 for value in track_ids):
            return [], [], [], "duplicate_or_invalid_track_id"
        expiry_rows, rejected, state_rows = [], [], []
        present = set(track_ids)
        for track_id in tuple(self._memory):
            if track_id not in present:
                self._expire(track_id, "track_deleted", expiry_rows)

        for track, track_id in zip(tracks, track_ids):
            generation = _generation(track)
            memory = self._memory.get(track_id)
            if memory is not None and memory.state_generation != generation:
                self._expire(track_id, "id_generation_changed", expiry_rows)
                memory = None
            try:
                covariance = _covariance(track)
                position_std, velocity_std = _uncertainty(covariance)
                position = np.asarray(
                    _value(track, "position_world"), dtype=np.float64
                )
                velocity = np.asarray(
                    _value(track, "velocity_world"), dtype=np.float64
                )
                if (
                    position.shape != (3,) or velocity.shape != (3,)
                    or not np.isfinite(position).all()
                    or not np.isfinite(velocity).all()
                ):
                    raise ValueError("state vectors must be finite [3]")
            except (TypeError, ValueError) as error:
                self._expire(track_id, "invalid_state", expiry_rows)
                rejected.append({
                    "track_id": track_id, "safety_state": SafetyState.INVALID.value,
                    "reason": str(error),
                })
                continue

            direct = bool(_value(
                track, "measurement_present",
                int(_value(track, "last_direct_observation_frame", -1))
                == frame_index,
            ))
            confirmed = bool(_value(track, "is_confirmed", False))
            dynamic = bool(_value(track, "is_dynamic", False))
            attention = bool(_value(track, "attention_authorized", False))
            missed = int(_value(track, "missed_count", 0))
            split_merge = bool(_value(track, "split_merge_suspected", False))
            observed_dynamic = confirmed and direct and dynamic and attention
            if memory is not None:
                position_jump = float(np.linalg.norm(
                    position - memory.last_position_world
                ))
                velocity_jump = float(np.linalg.norm(
                    velocity - memory.last_velocity_world
                ))
                if split_merge:
                    self._expire(track_id, "split_merge_incompatible", expiry_rows)
                    memory = None
                elif (
                    position_jump > self.config.maximum_state_jump_m
                    or velocity_jump > self.config.maximum_velocity_jump_mps
                ):
                    self._expire(track_id, "incompatible_state_jump", expiry_rows)
                    memory = None

            if observed_dynamic:
                streak = (
                    memory.dynamic_streak_before_loss + 1
                    if memory is not None
                    and frame_index == memory.last_dynamic_frame + 1 else 1
                )
                self._memory[track_id] = RecentDynamicEvidence(
                    track_id=track_id,
                    state_generation=generation,
                    last_dynamic_frame=frame_index,
                    last_dynamic_timestamp=timestamp,
                    last_direct_measurement_frame=frame_index,
                    last_direct_measurement_timestamp=timestamp,
                    dynamic_streak_before_loss=streak,
                    confirmed_age_before_loss=int(_value(track, "age", 0)),
                    last_position_world=position.copy(),
                    last_velocity_world=velocity.copy(),
                )
                state = SafetyState.OBSERVED_DYNAMIC
                reason = "direct_confirmed_dynamic_attention"
            else:
                memory = self._memory.get(track_id)
                if memory is None:
                    state = (
                        SafetyState.OBSERVED_NON_DYNAMIC
                        if direct else SafetyState.EXPIRED
                    )
                    reason = (
                        "direct_without_recent_dynamic"
                        if direct else "no_recent_dynamic_evidence"
                    )
                else:
                    elapsed = timestamp - memory.last_dynamic_timestamp
                    expiry = None
                    if missed > self.config.maximum_coasting_misses:
                        expiry = "missed_count_exceeded"
                    elif elapsed > self.config.maximum_coasting_time_s + 1e-9:
                        expiry = "coasting_time_exceeded"
                    elif position_std > self.config.maximum_position_std_m:
                        expiry = "position_uncertainty_exceeded"
                    elif velocity_std > self.config.maximum_velocity_std_mps:
                        expiry = "velocity_uncertainty_exceeded"
                    if expiry is not None:
                        self._expire(track_id, expiry, expiry_rows)
                        state, reason = SafetyState.EXPIRED, expiry
                    elif direct:
                        stable_stop = (
                            float(np.linalg.norm(velocity))
                            <= self.config.dynamic_exit_speed_mps
                            and int(_value(track, "consecutive_direct_hits", 0))
                            >= self.config.static_reclassification_direct_hits
                        )
                        if stable_stop:
                            self._expire(
                                track_id, "stable_non_dynamic_reclassification",
                                expiry_rows,
                            )
                            state = SafetyState.OBSERVED_NON_DYNAMIC
                            reason = "stable_non_dynamic_reclassification"
                        else:
                            memory.reacquired_frames += 1
                            memory.last_direct_measurement_frame = frame_index
                            memory.last_direct_measurement_timestamp = timestamp
                            memory.last_position_world = position.copy()
                            memory.last_velocity_world = velocity.copy()
                            if (
                                memory.reacquired_frames
                                > self.config.maximum_reacquired_frames
                            ):
                                self._expire(
                                    track_id, "reacquired_window_exceeded",
                                    expiry_rows,
                                )
                                state = SafetyState.EXPIRED
                                reason = "reacquired_window_exceeded"
                            else:
                                state = SafetyState.REACQUIRED_UNCERTAIN
                                reason = "direct_measurement_dynamic_not_restored"
                    else:
                        memory.last_position_world = position.copy()
                        memory.last_velocity_world = velocity.copy()
                        state = SafetyState.COASTING_DYNAMIC
                        reason = "bounded_recent_dynamic_prediction"

            state_rows.append({
                "track_id": track_id,
                "state_generation": generation,
                "safety_state": state.value,
                "reason": reason,
                "measurement_present": direct,
                "missed_count": missed,
                "position_std_m": position_std,
                "velocity_std_mps": velocity_std,
                "position_world": position,
                "velocity_world": velocity,
                "state_covariance": covariance,
                "radius_m": float(_value(
                    track, "radius_m", self.config.default_actor_radius_m
                )),
            })
        self._last_timestamp = timestamp
        self._last_frame = frame_index
        return state_rows, rejected, expiry_rows, None

    def evaluate(
        self, candidate_positions_world, sample_times_s, tracks, *,
        timestamp, frame_index, original_candidate_id,
        candidate_scores=None,
    ):
        started = time.perf_counter()
        try:
            candidates = np.asarray(candidate_positions_world, dtype=np.float64)
            times = np.asarray(sample_times_s, dtype=np.float64)
            if (
                candidates.ndim != 3 or candidates.shape[2] != 3
                or not np.isfinite(candidates).all()
                or times.shape != (candidates.shape[1],)
                or not np.isfinite(times).all()
                or np.any(times < 0)
                or not 0 <= int(original_candidate_id) < len(candidates)
            ):
                raise ValueError("invalid candidate/time/original-ID input")
            if candidate_scores is None:
                scores = np.arange(len(candidates), dtype=np.float64)
            else:
                scores = np.asarray(candidate_scores, dtype=np.float64)
                if scores.shape != (len(candidates),) or not np.isfinite(scores).all():
                    raise ValueError("candidate_scores must be finite [N]")
        except (TypeError, ValueError) as error:
            return self._invalid_result(str(error), original_candidate_id, started)

        states, rejected, expiry, invalid = self.update_safety_states(
            tracks, timestamp, frame_index
        )
        if invalid is not None:
            return self._invalid_result(invalid, original_candidate_id, started)
        active_states = {
            SafetyState.OBSERVED_DYNAMIC.value,
            SafetyState.COASTING_DYNAMIC.value,
            SafetyState.REACQUIRED_UNCERTAIN.value,
        }
        active = [row for row in states if row["safety_state"] in active_states]
        candidate_rows = []
        for candidate_id, candidate in enumerate(candidates):
            best = {
                "clearance": float("inf"), "distance": float("inf"),
                "time": None, "track_id": None, "safety_state": None,
                "inflated_radius": 0.,
            }
            for track in active:
                predicted, covariances = predict_state(
                    track["position_world"], track["velocity_world"],
                    track["state_covariance"], times,
                    self.config.process_noise_acceleration,
                )
                position_std = np.sqrt(np.maximum(
                    np.linalg.eigvalsh(covariances[:, :3, :3])[:, -1], 0.
                ))
                inflated = (
                    self.config.uav_radius_m + track["radius_m"]
                    + self.config.clearance_margin_m
                    + self.config.covariance_sigma * position_std
                )
                distance = np.linalg.norm(candidate - predicted, axis=1)
                clearance = distance - inflated
                index = int(np.argmin(clearance))
                if clearance[index] < best["clearance"]:
                    best = {
                        "clearance": float(clearance[index]),
                        "distance": float(distance[index]),
                        "time": float(times[index]),
                        "track_id": track["track_id"],
                        "safety_state": track["safety_state"],
                        "inflated_radius": float(inflated[index]),
                    }
            veto = best["clearance"] < 0.
            candidate_rows.append({
                "candidate_trajectory_id": candidate_id,
                "predicted_minimum_actor_distance_m": best["distance"],
                "predicted_minimum_clearance_m": best["clearance"],
                "minimum_clearance_time_s": best["time"],
                "uncertainty_inflated_safety_radius_m":
                    best["inflated_radius"],
                "would_veto": bool(veto),
                "veto_reason": (
                    "predicted_uncertainty_inflated_collision"
                    if veto else "none"
                ),
                "limiting_track_id": best["track_id"],
                "limiting_safety_state": best["safety_state"],
            })
        original = int(original_candidate_id)
        if not active:
            decision = DecisionStatus.NO_ACTIVE_DYNAMIC_RISK
            recommended = original
            no_safe_reason = None
        else:
            safe = [
                row["candidate_trajectory_id"] for row in candidate_rows
                if not row["would_veto"]
            ]
            if not safe:
                decision = DecisionStatus.NO_SAFE_CANDIDATE
                recommended = None
                no_safe_reason = "all_candidates_vetoed_by_active_dynamic_risk"
            elif original in safe:
                decision = DecisionStatus.KEEP_ORIGINAL
                recommended = original
                no_safe_reason = None
            else:
                decision = DecisionStatus.SWITCH_TO_SAFE_CANDIDATE
                recommended = min(safe, key=lambda index: (scores[index], index))
                no_safe_reason = None
        return {
            "adapter_version": ADAPTER_VERSION,
            "contract_version": CONTRACT_VERSION,
            "shadow_only": True,
            "formal_control_modified": False,
            "runtime_gt_used": False,
            "decision_status": decision.value,
            "original_candidate_id": original,
            "recommended_candidate_id": recommended,
            "active_safety_track_count": len(active),
            "observed_dynamic_count": sum(
                row["safety_state"] == SafetyState.OBSERVED_DYNAMIC.value
                for row in active
            ),
            "coasting_dynamic_count": sum(
                row["safety_state"] == SafetyState.COASTING_DYNAMIC.value
                for row in active
            ),
            "reacquired_uncertain_count": sum(
                row["safety_state"] == SafetyState.REACQUIRED_UNCERTAIN.value
                for row in active
            ),
            "candidate_rows": candidate_rows,
            "track_state_rows": [
                self._serializable_state(row) for row in states
            ],
            "rejected_track_rows": rejected,
            "expiry_rows": expiry,
            "no_safe_candidate_reason": no_safe_reason,
            "runtime_ms": (time.perf_counter() - started) * 1000.,
        }

    @staticmethod
    def _serializable_state(row):
        result = dict(row)
        for key in ("position_world", "velocity_world", "state_covariance"):
            result[key] = np.asarray(result[key]).tolist()
        return result

    @staticmethod
    def _invalid_result(reason, original_candidate_id, started):
        return {
            "adapter_version": ADAPTER_VERSION,
            "contract_version": CONTRACT_VERSION,
            "shadow_only": True,
            "formal_control_modified": False,
            "runtime_gt_used": False,
            "decision_status": DecisionStatus.INVALID_EVALUATION.value,
            "original_candidate_id": (
                int(original_candidate_id)
                if isinstance(original_candidate_id, (int, np.integer)) else None
            ),
            "recommended_candidate_id": None,
            "active_safety_track_count": 0,
            "observed_dynamic_count": 0,
            "coasting_dynamic_count": 0,
            "reacquired_uncertain_count": 0,
            "candidate_rows": [],
            "rejected_track_rows": [],
            "expiry_rows": [],
            "no_safe_candidate_reason": reason,
            "runtime_ms": (time.perf_counter() - started) * 1000.,
        }


__all__ = [
    "ADAPTER_VERSION", "CONTRACT_VERSION", "SafetyState",
    "DecisionStatus", "CoastingSafetyConfig", "RecentDynamicEvidence",
    "BoundedCoastingSafetyAdapter", "predict_state",
]
