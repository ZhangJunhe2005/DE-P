"""Deterministic recovery-state observation view for Route-A V4.2.7.

Depth, pose, map identity and split remain frozen.  Only the synthetic YOPO
state vector is resampled so training covers the near-stationary, off-axis
states produced by a real brake/yaw recovery cycle.
"""

from __future__ import annotations

import hashlib

import numpy as np
from torch.utils.data import Dataset

from policy.static_yopo_wide_state_v1 import sample_wide_observation


CONTRACT_VERSION = "route_a_recovery_state_v2"
RECOVERY_SAMPLE_PROBABILITY = 0.35


def _rng(sample_id: str, split: str, global_seed: int):
    payload = f"{CONTRACT_VERSION}:{global_seed}:{split}:{sample_id}".encode()
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(seed)


def sample_recovery_observation(sample_id: str, split: str, global_seed: int):
    rng = _rng(sample_id, split, global_seed)
    if float(rng.random()) >= RECOVERY_SAMPLE_PROBABILITY:
        return sample_wide_observation(sample_id, split, global_seed)

    speed = float(rng.uniform(0.0, 0.75))
    velocity_yaw = np.deg2rad(rng.uniform(-60.0, 60.0))
    velocity_pitch = np.deg2rad(rng.normal(0.0, 10.0))
    velocity = speed * np.asarray([
        np.cos(velocity_yaw) * np.cos(velocity_pitch),
        np.sin(velocity_yaw) * np.cos(velocity_pitch),
        np.sin(velocity_pitch),
    ])

    acceleration_magnitude = float(rng.uniform(0.0, 1.0))
    acceleration_direction = rng.normal(size=3)
    acceleration_direction[2] *= 0.6
    acceleration_direction /= max(
        float(np.linalg.norm(acceleration_direction)), 1.0e-9
    )
    acceleration = acceleration_magnitude * acceleration_direction

    goal_distance = float(rng.uniform(10.0, 40.0))
    goal_yaw = np.deg2rad(rng.uniform(-75.0, 75.0))
    goal_pitch = np.deg2rad(np.clip(rng.normal(0.0, 10.0), -30.0, 30.0))
    goal = goal_distance * np.asarray([
        np.cos(goal_yaw) * np.cos(goal_pitch),
        np.sin(goal_yaw) * np.cos(goal_pitch),
        np.sin(goal_pitch),
    ])
    return np.concatenate((velocity, acceleration, goal)).astype(np.float32)


class StaticYOPORecoveryStateDatasetV2(Dataset):
    def __init__(self, base: Dataset, seed: int):
        self.base = base
        self.seed = int(seed)
        self.split = str(base.split)
        self.arrays = base.arrays

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        batch = self.base[index]
        observation = sample_recovery_observation(
            batch["sample_id"], self.split, self.seed
        )
        batch["observation"] = batch["observation"].new_tensor(observation)
        return batch


__all__ = [
    "CONTRACT_VERSION", "RECOVERY_SAMPLE_PROBABILITY",
    "StaticYOPORecoveryStateDatasetV2", "sample_recovery_observation",
]
