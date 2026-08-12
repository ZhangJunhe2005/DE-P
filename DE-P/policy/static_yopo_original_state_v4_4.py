"""Deterministic original-YOPO observation distribution for Route-A V4.4.

Depth, pose, map identity, and train/validation membership remain frozen.  This
view replaces only the synthetic nine-dimensional observation with the
forward-biased distribution used by the original YOPO implementation.
"""

from __future__ import annotations

import hashlib

import numpy as np
from torch.utils.data import Dataset


CONTRACT_VERSION = "route_a_original_yopo_state_v4_4"
VELOCITY_LIMIT_MPS = 6.0
ACCELERATION_LIMIT_MPS2 = 6.0
GOAL_LENGTH_M = 10.0


def _seed(global_seed: int, sample_id: str, split: str) -> int:
    payload = (
        f"{CONTRACT_VERSION}:{int(global_seed)}:{split}:{sample_id}"
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _sample_level_state(rng: np.random.Generator):
    velocity_mean = np.asarray([0.4, 0.0, 0.0], dtype=np.float64)
    velocity_std = np.asarray([2.0, 0.45, 0.3], dtype=np.float64)
    acceleration_mean = np.zeros(3, dtype=np.float64)
    acceleration_std = np.asarray([0.5, 0.5, 0.3], dtype=np.float64)

    # This is the original right-skewed forward-x construction.  The bounded
    # loops turn a historically unbounded rejection sampler into a fail-fast
    # deterministic data contract.
    for _ in range(10_000):
        velocity = VELOCITY_LIMIT_MPS * (
            velocity_mean + velocity_std * rng.standard_normal(3)
        )
        forward = -1.0
        for _ in range(10_000):
            forward = (
                -VELOCITY_LIMIT_MPS
                * rng.lognormal(mean=np.log(0.6), sigma=np.log(2.0))
                + 1.2 * VELOCITY_LIMIT_MPS
            )
            if forward >= 0.0:
                break
        if forward < 0.0:
            raise RuntimeError("original YOPO forward velocity sampling exhausted")
        velocity[0] = forward
        if np.linalg.norm(velocity) < 1.2 * VELOCITY_LIMIT_MPS:
            break
    else:
        raise RuntimeError("original YOPO velocity rejection sampling exhausted")

    for _ in range(10_000):
        acceleration = ACCELERATION_LIMIT_MPS2 * (
            acceleration_mean + acceleration_std * rng.standard_normal(3)
        )
        if np.linalg.norm(acceleration) < 1.2 * ACCELERATION_LIMIT_MPS2:
            break
    else:
        raise RuntimeError("original YOPO acceleration rejection sampling exhausted")
    return velocity, acceleration


def _sample_level_goal(rng: np.random.Generator):
    pitch = np.deg2rad(rng.normal(0.0, 10.0))
    yaw = np.deg2rad(rng.normal(0.0, 20.0))
    direction = np.asarray([
        np.cos(yaw) * np.cos(pitch),
        np.sin(yaw) * np.cos(pitch),
        np.sin(pitch),
    ], dtype=np.float64)
    near = float(rng.random())
    if near < 0.1:
        direction *= near * 10.0
    return GOAL_LENGTH_M * direction


def _level_to_body(rotation_world_from_body):
    rotation = np.asarray(rotation_world_from_body, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(rotation).all():
        raise ValueError("rotation_world_from_body contains NaN/Inf")
    yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    cosine, sine = np.cos(yaw), np.sin(yaw)
    rotation_world_from_level = np.asarray([
        [cosine, -sine, 0.0],
        [sine, cosine, 0.0],
        [0.0, 0.0, 1.0],
    ])
    return rotation.T @ rotation_world_from_level


def sample_original_yopo_observation(
    sample_id: str, split: str, global_seed: int, rotation_world_from_body,
):
    rng = np.random.default_rng(_seed(global_seed, sample_id, split))
    velocity_level, acceleration_level = _sample_level_state(rng)
    goal_level = _sample_level_goal(rng)
    rotation_body_from_level = _level_to_body(rotation_world_from_body)
    observation = np.concatenate((
        rotation_body_from_level @ velocity_level,
        rotation_body_from_level @ acceleration_level,
        rotation_body_from_level @ goal_level,
    ))
    return observation.astype(np.float32)


class StaticYOPOOriginalStateDatasetV44(Dataset):
    """Replace only observation while retaining actor-free static authority."""

    def __init__(self, base: Dataset, seed: int):
        self.base = base
        self.seed = int(seed)
        self.split = str(base.split)
        self.arrays = base.arrays

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        batch = dict(self.base[index])
        observation = sample_original_yopo_observation(
            batch["sample_id"], self.split, self.seed,
            batch["rotation_world_from_body"],
        )
        batch["observation"] = batch["observation"].new_tensor(observation)
        return batch
