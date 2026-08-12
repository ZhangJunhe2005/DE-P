import numpy as np

from policy.static_yopo_recovery_state_v2 import sample_recovery_observation


def test_recovery_observation_is_deterministic_and_finite():
    left = sample_recovery_observation("fixture", "train", 82701)
    right = sample_recovery_observation("fixture", "train", 82701)
    np.testing.assert_array_equal(left, right)
    assert left.shape == (9,)
    assert left.dtype == np.float32
    assert np.isfinite(left).all()


def test_distribution_contains_recovery_and_cruise_states():
    values = np.stack([
        sample_recovery_observation(f"sample-{index}", "train", 82701)
        for index in range(2000)
    ])
    speed = np.linalg.norm(values[:, :3], axis=1)
    goal = values[:, 6:9]
    goal_yaw = np.degrees(np.arctan2(goal[:, 1], goal[:, 0]))
    near_stationary = speed <= 0.75 + 1.0e-6
    assert 0.35 <= float(near_stationary.mean()) <= 0.50
    assert np.count_nonzero(np.abs(goal_yaw) >= 45.0) >= 100
    assert float(speed.max()) > 5.0
