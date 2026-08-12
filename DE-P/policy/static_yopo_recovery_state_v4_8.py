"""Deterministic original-YOPO/recovery observation mixture for V4.8.

The static depth, pose, map identity, and split remain unchanged.  Eighty
percent of samples reuse the V4.4 original-YOPO observation sampler exactly;
the remaining twenty percent represent the slow, off-axis body-frame state
seen after a bounded camera-yaw recovery scan.

This contract is intentionally separate from ``static_yopo_recovery_state_v2``.
V2 belongs to the historical V4.2.7 experiments and must remain reproducible.
"""

from __future__ import annotations

import hashlib

import numpy as np
from torch.utils.data import Dataset

from policy.static_yopo_original_state_v4_4 import (
    sample_original_yopo_observation,
)


CONTRACT_VERSION = "route_a_recovery_state_v4_8"
RECOVERY_SAMPLE_PROBABILITY = 0.20
RECOVERY_SPEED_MAX_MPS = 1.0
RECOVERY_ACCELERATION_MAX_MPS2 = 1.0
RECOVERY_GOAL_YAW_MIN_DEG = 45.0
RECOVERY_GOAL_YAW_MAX_DEG = 120.0
RECOVERY_GOAL_PITCH_MAX_DEG = 30.0
RECOVERY_GOAL_DISTANCE_M = 10.0


def _rng(sample_id: str, split: str, global_seed: int) -> np.random.Generator:
    payload = (
        f"{CONTRACT_VERSION}:{int(global_seed)}:{split}:{sample_id}"
    ).encode()
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(seed)


def _sample_recovery_body_observation(rng: np.random.Generator) -> np.ndarray:
    """Sample one low-motion recovery observation directly in body axes."""
    speed = float(rng.uniform(0.0, RECOVERY_SPEED_MAX_MPS))
    velocity_yaw = float(rng.uniform(-np.pi, np.pi))
    velocity_pitch = np.deg2rad(float(np.clip(
        rng.normal(0.0, 10.0),
        -RECOVERY_GOAL_PITCH_MAX_DEG,
        RECOVERY_GOAL_PITCH_MAX_DEG,
    )))
    velocity = speed * np.asarray([
        np.cos(velocity_yaw) * np.cos(velocity_pitch),
        np.sin(velocity_yaw) * np.cos(velocity_pitch),
        np.sin(velocity_pitch),
    ], dtype=np.float64)

    acceleration_magnitude = float(rng.uniform(
        0.0, RECOVERY_ACCELERATION_MAX_MPS2
    ))
    acceleration_direction = rng.normal(size=3)
    direction_norm = float(np.linalg.norm(acceleration_direction))
    if direction_norm <= 1.0e-12:
        # The probability of this branch is negligible, but keeping an exact
        # deterministic fallback avoids a divide-by-zero contract hole.
        acceleration_direction = np.asarray([1.0, 0.0, 0.0])
    else:
        acceleration_direction /= direction_norm
    acceleration = acceleration_magnitude * acceleration_direction

    goal_sign = -1.0 if float(rng.random()) < 0.5 else 1.0
    goal_yaw = np.deg2rad(goal_sign * float(rng.uniform(
        RECOVERY_GOAL_YAW_MIN_DEG, RECOVERY_GOAL_YAW_MAX_DEG
    )))
    goal_pitch = np.deg2rad(float(rng.uniform(
        -RECOVERY_GOAL_PITCH_MAX_DEG, RECOVERY_GOAL_PITCH_MAX_DEG
    )))
    goal = RECOVERY_GOAL_DISTANCE_M * np.asarray([
        np.cos(goal_yaw) * np.cos(goal_pitch),
        np.sin(goal_yaw) * np.cos(goal_pitch),
        np.sin(goal_pitch),
    ], dtype=np.float64)

    return np.concatenate((velocity, acceleration, goal)).astype(np.float32)


def sample_recovery_observation_v4_8(
    sample_id: str,
    split: str,
    global_seed: int,
    rotation_world_from_body,
):
    """Return ``(observation_9d, is_recovery)`` for one frozen sample.

    Non-recovery observations are delegated directly to the original-YOPO
    sampler.  Its output is returned without arithmetic or conversion so the
    eighty-percent branch stays byte-for-byte compatible with V4.4.
    """
    rng = _rng(sample_id, split, global_seed)
    is_recovery = float(rng.random()) < RECOVERY_SAMPLE_PROBABILITY
    if not is_recovery:
        observation = sample_original_yopo_observation(
            sample_id, split, global_seed, rotation_world_from_body,
        )
        return observation, False
    return _sample_recovery_body_observation(rng), True


class StaticYOPORecoveryStateDatasetV48(Dataset):
    """Replace only the observation in a copied sample dictionary."""

    def __init__(self, base: Dataset, seed: int):
        self.base = base
        self.seed = int(seed)
        self.split = str(base.split)
        self.arrays = base.arrays

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        # Some lightweight dataset fixtures return the same dictionary object
        # repeatedly.  Copy it before replacing observation so this view can
        # never mutate the frozen base dataset.
        batch = dict(self.base[index])
        observation, is_recovery = sample_recovery_observation_v4_8(
            batch["sample_id"], self.split, self.seed,
            batch["rotation_world_from_body"],
        )
        batch["observation"] = batch["observation"].new_tensor(observation)
        batch["recovery_state"] = bool(is_recovery)
        return batch


__all__ = [
    "CONTRACT_VERSION",
    "RECOVERY_SAMPLE_PROBABILITY",
    "RECOVERY_SPEED_MAX_MPS",
    "RECOVERY_ACCELERATION_MAX_MPS2",
    "RECOVERY_GOAL_YAW_MIN_DEG",
    "RECOVERY_GOAL_YAW_MAX_DEG",
    "RECOVERY_GOAL_PITCH_MAX_DEG",
    "RECOVERY_GOAL_DISTANCE_M",
    "StaticYOPORecoveryStateDatasetV48",
    "sample_recovery_observation_v4_8",
]
