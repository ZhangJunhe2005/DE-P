from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from policy.static_yopo_original_state_v4_4 import (
    sample_original_yopo_observation,
)
from policy.static_yopo_recovery_state_v4_8 import (
    RECOVERY_ACCELERATION_MAX_MPS2,
    RECOVERY_GOAL_DISTANCE_M,
    RECOVERY_GOAL_PITCH_MAX_DEG,
    RECOVERY_GOAL_YAW_MAX_DEG,
    RECOVERY_GOAL_YAW_MIN_DEG,
    RECOVERY_SPEED_MAX_MPS,
    StaticYOPORecoveryStateDatasetV48,
    sample_recovery_observation_v4_8,
)


IDENTITY = np.eye(3, dtype=np.float32)
SEED = 84801


def sample(index, rotation=IDENTITY):
    return sample_recovery_observation_v4_8(
        f"sample-{index}", "train", SEED, rotation,
    )


def test_v48_observation_is_deterministic_finite_and_typed():
    left, left_recovery = sample(17)
    right, right_recovery = sample(17)
    np.testing.assert_array_equal(left, right)
    assert left_recovery is right_recovery
    assert left.shape == (9,)
    assert left.dtype == np.float32
    assert np.isfinite(left).all()


def test_v48_mixture_is_eighty_percent_original_twenty_percent_recovery():
    flags = np.asarray([sample(index)[1] for index in range(10_000)])
    fraction = float(flags.mean())
    # This checks the deterministic hash distribution without pretending a
    # Bernoulli mixture must contain exactly 2,000 recovery samples.
    assert 0.18 <= fraction <= 0.22


def test_non_recovery_branch_exactly_reuses_original_yopo_sampler():
    checked = 0
    rotation = np.asarray([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    for index in range(500):
        observation, is_recovery = sample(index, rotation)
        if is_recovery:
            continue
        expected = sample_original_yopo_observation(
            f"sample-{index}", "train", SEED, rotation,
        )
        np.testing.assert_array_equal(observation, expected)
        checked += 1
    assert checked >= 350


def test_recovery_branch_respects_motion_and_goal_contract():
    recovery = []
    for index in range(5_000):
        observation, is_recovery = sample(index)
        if is_recovery:
            recovery.append(observation)
    values = np.stack(recovery)

    speed = np.linalg.norm(values[:, 0:3], axis=1)
    acceleration = np.linalg.norm(values[:, 3:6], axis=1)
    goal = values[:, 6:9]
    goal_distance = np.linalg.norm(goal, axis=1)
    goal_yaw = np.abs(np.degrees(np.arctan2(goal[:, 1], goal[:, 0])))
    goal_pitch = np.abs(np.degrees(np.arctan2(
        goal[:, 2], np.linalg.norm(goal[:, :2], axis=1)
    )))

    assert float(speed.min()) >= 0.0
    assert float(speed.max()) <= RECOVERY_SPEED_MAX_MPS + 1.0e-6
    assert float(acceleration.min()) >= 0.0
    assert float(acceleration.max()) <= (
        RECOVERY_ACCELERATION_MAX_MPS2 + 1.0e-6
    )
    assert float(goal_yaw.min()) >= RECOVERY_GOAL_YAW_MIN_DEG - 1.0e-5
    assert float(goal_yaw.max()) <= RECOVERY_GOAL_YAW_MAX_DEG + 1.0e-5
    assert float(goal_pitch.max()) <= RECOVERY_GOAL_PITCH_MAX_DEG + 1.0e-5
    np.testing.assert_allclose(
        goal_distance, RECOVERY_GOAL_DISTANCE_M, rtol=1.0e-6, atol=1.0e-6,
    )


class ReusedDictionaryDataset(Dataset):
    split = "train"
    arrays = {"map_id": np.asarray([7])}

    def __init__(self, sample_id):
        self.sample = {
            "sample_id": sample_id,
            "rotation_world_from_body": torch.eye(3),
            "observation": torch.full((9,), -123.0),
            "sentinel": ["unchanged"],
        }

    def __len__(self):
        return 1

    def __getitem__(self, index):
        assert index == 0
        return self.sample


def test_dataset_wrapper_copies_sample_and_exposes_recovery_flag():
    # Find a deterministic recovery identity so this also checks the bool is
    # derived from the same sampler call as the replacement observation.
    sample_id = next(
        f"fixture-{index}" for index in range(100)
        if sample_recovery_observation_v4_8(
            f"fixture-{index}", "train", SEED, IDENTITY,
        )[1]
    )
    base = ReusedDictionaryDataset(sample_id)
    original_observation = base.sample["observation"].clone()
    view = StaticYOPORecoveryStateDatasetV48(base, seed=SEED)
    result = view[0]

    assert result is not base.sample
    assert result["recovery_state"] is True
    assert result["observation"].shape == (9,)
    assert result["observation"].dtype == original_observation.dtype
    assert result["sentinel"] is base.sample["sentinel"]
    torch.testing.assert_close(base.sample["observation"], original_observation)
    assert "recovery_state" not in base.sample
