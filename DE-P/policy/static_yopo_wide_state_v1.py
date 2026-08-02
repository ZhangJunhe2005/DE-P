"""Deterministic wide-state observation view for Route-A static YOPO V4."""

from __future__ import annotations

import hashlib

import numpy as np
from torch.utils.data import Dataset


CONTRACT_VERSION = "route_a_wide_state_v1"


def _seed(global_seed: int, sample_id: str, split: str) -> int:
    payload = f"{CONTRACT_VERSION}:{global_seed}:{split}:{sample_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _tiered_magnitude(rng, tiers):
    value = float(rng.random())
    cumulative = 0.0
    for probability, lower, upper in tiers:
        cumulative += probability
        if value <= cumulative:
            return float(rng.uniform(lower, upper))
    return float(rng.uniform(tiers[-1][1], tiers[-1][2]))


def sample_wide_observation(sample_id: str, split: str, global_seed: int):
    """Return body-frame velocity, acceleration and a forward-biased goal."""
    rng = np.random.default_rng(_seed(global_seed, sample_id, split))
    speed = _tiered_magnitude(rng, (
        (0.05, 0.0, 0.5),
        (0.20, 0.5, 2.0),
        (0.40, 2.0, 4.5),
        (0.35, 4.5, 6.0),
    ))
    velocity_yaw = np.deg2rad(rng.normal(0.0, 20.0))
    velocity_pitch = np.deg2rad(rng.normal(0.0, 10.0))
    velocity_direction = np.asarray([
        np.cos(velocity_yaw) * np.cos(velocity_pitch),
        np.sin(velocity_yaw) * np.cos(velocity_pitch),
        np.sin(velocity_pitch),
    ])
    velocity = speed * velocity_direction

    acceleration_magnitude = _tiered_magnitude(rng, (
        (0.10, 0.0, 0.5),
        (0.25, 0.5, 2.0),
        (0.40, 2.0, 4.0),
        (0.25, 4.0, 6.0),
    ))
    acceleration_direction = rng.normal(size=3)
    acceleration_direction[2] *= 0.6
    acceleration_direction /= max(np.linalg.norm(acceleration_direction), 1e-9)
    acceleration = acceleration_magnitude * acceleration_direction

    goal_distance = float(rng.uniform(10.0, 40.0))
    goal_yaw = np.deg2rad(rng.normal(0.0, 20.0))
    goal_pitch = np.deg2rad(rng.normal(0.0, 10.0))
    goal = goal_distance * np.asarray([
        np.cos(goal_yaw) * np.cos(goal_pitch),
        np.sin(goal_yaw) * np.cos(goal_pitch),
        np.sin(goal_pitch),
    ])
    return np.concatenate((velocity, acceleration, goal)).astype(np.float32)


class StaticYOPOWideStateDatasetV1(Dataset):
    """Replace only observation; depth, pose, split and authority stay frozen."""

    def __init__(self, base: Dataset, seed: int):
        self.base = base
        self.seed = int(seed)
        self.split = str(base.split)
        self.arrays = base.arrays

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        batch = self.base[index]
        observation = sample_wide_observation(
            batch["sample_id"], self.split, self.seed
        )
        batch["observation"] = batch["observation"].new_tensor(observation)
        return batch
