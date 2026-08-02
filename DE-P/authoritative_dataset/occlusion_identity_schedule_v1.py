"""Causal time-shift-only schedule for retained v2.2 occlusion lines."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np


SCHEDULE_VERSION = "occlusion_identity_schedule_v1"


@dataclass(frozen=True)
class OcclusionIdentitySchedule:
    actor_initial_position: np.ndarray
    actor_velocity: np.ndarray
    requested_gap_frames: int
    expected_gap_start: int
    visible_pre_roll_frames: int
    required_post_gap_visible_frames: int
    shift_frames: int
    frame_period_s: float
    certificate: dict

    def actor_positions(self, timestamps):
        timestamps = np.asarray(timestamps, dtype=np.float64)
        return (
            np.asarray(self.actor_initial_position, dtype=np.float64)[None, :]
            + timestamps[:, None]
            * np.asarray(self.actor_velocity, dtype=np.float64)[None, :]
        )


def derive_time_shift_schedule(
    original_initial_position,
    actor_velocity,
    original_gap_start,
    requested_gap_frames,
    shift_frames,
    frame_period_s,
    *,
    required_post_gap_visible_frames=2,
):
    """Move only the time origin on an already frozen constant-velocity line."""
    original = np.asarray(original_initial_position, dtype=np.float64)
    velocity = np.asarray(actor_velocity, dtype=np.float64)
    shift_frames = int(shift_frames)
    frame_period_s = float(frame_period_s)
    if original.shape != (3,) or velocity.shape != (3,):
        raise ValueError("initial position and velocity must be xyz")
    if not np.isfinite(original).all() or not np.isfinite(velocity).all():
        raise ValueError("schedule inputs must be finite")
    if shift_frames < 0:
        raise ValueError("schedule may not move the gap earlier")
    if int(requested_gap_frames) not in (1, 2, 3):
        raise ValueError("requested gap must be 1, 2, or 3")
    shifted = original - velocity * shift_frames * frame_period_s
    expected_gap_start = int(original_gap_start) + shift_frames
    payload = {
        "schedule_version": SCHEDULE_VERSION,
        "original_initial_position": original.tolist(),
        "shifted_initial_position": shifted.tolist(),
        "actor_velocity": velocity.tolist(),
        "shift_frames": shift_frames,
        "frame_period_s": frame_period_s,
        "expected_gap_start": expected_gap_start,
        "requested_gap_frames": int(requested_gap_frames),
        "speed_unchanged": True,
        "direction_unchanged": True,
        "acceleration_mps2": 0.0,
        "time_shift_only": True,
    }
    payload["certificate_hash"] = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    return OcclusionIdentitySchedule(
        actor_initial_position=shifted,
        actor_velocity=velocity.copy(),
        requested_gap_frames=int(requested_gap_frames),
        expected_gap_start=expected_gap_start,
        visible_pre_roll_frames=expected_gap_start,
        required_post_gap_visible_frames=int(
            required_post_gap_visible_frames
        ),
        shift_frames=shift_frames,
        frame_period_s=frame_period_s,
        certificate=payload,
    )


def verify_time_shift_only(
    schedule,
    original_initial_position,
    original_velocity,
    timestamps,
):
    original_initial_position = np.asarray(
        original_initial_position, dtype=np.float64
    )
    original_velocity = np.asarray(
        original_velocity, dtype=np.float64
    )
    timestamps = np.asarray(timestamps, dtype=np.float64)
    shifted_times = (
        timestamps - schedule.shift_frames * schedule.frame_period_s
    )
    expected = (
        original_initial_position[None, :]
        + shifted_times[:, None] * original_velocity[None, :]
    )
    actual = schedule.actor_positions(timestamps)
    return {
        "time_shift_only": bool(np.allclose(
            actual, expected, atol=1e-12
        )),
        "speed_unchanged": bool(np.isclose(
            np.linalg.norm(schedule.actor_velocity),
            np.linalg.norm(original_velocity),
            atol=1e-12,
        )),
        "direction_unchanged": bool(np.allclose(
            schedule.actor_velocity, original_velocity, atol=1e-12
        )),
        "acceleration_zero": True,
        "maximum_position_error_m": float(np.max(np.linalg.norm(
            actual - expected, axis=1
        ))),
    }


__all__ = [
    "OcclusionIdentitySchedule",
    "SCHEDULE_VERSION",
    "derive_time_shift_schedule",
    "verify_time_shift_only",
]
