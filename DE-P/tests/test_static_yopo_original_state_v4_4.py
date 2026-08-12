import numpy as np

from policy.static_yopo_original_state_v4_4 import (
    sample_original_yopo_observation,
)


def test_original_yopo_state_is_deterministic_and_bounded():
    rotation = np.eye(3, dtype=np.float32)
    first = sample_original_yopo_observation(
        "sample-1", "train", 44, rotation
    )
    second = sample_original_yopo_observation(
        "sample-1", "train", 44, rotation
    )
    np.testing.assert_array_equal(first, second)
    assert first.shape == (9,)
    assert first.dtype == np.float32
    assert np.isfinite(first).all()
    assert first[0] >= 0.0
    assert np.linalg.norm(first[:3]) < 7.2
    assert np.linalg.norm(first[3:6]) < 7.2
    assert 0.0 <= np.linalg.norm(first[6:9]) <= 10.0 + 1.0e-5


def test_original_yopo_state_preserves_vector_norm_under_body_tilt():
    pitch = np.deg2rad(25.0)
    rotation = np.asarray([
        [np.cos(pitch), 0.0, np.sin(pitch)],
        [0.0, 1.0, 0.0],
        [-np.sin(pitch), 0.0, np.cos(pitch)],
    ], dtype=np.float32)
    level = sample_original_yopo_observation(
        "sample-2", "valid", 44, np.eye(3, dtype=np.float32)
    )
    tilted = sample_original_yopo_observation(
        "sample-2", "valid", 44, rotation
    )
    for start in (0, 3, 6):
        np.testing.assert_allclose(
            np.linalg.norm(level[start:start + 3]),
            np.linalg.norm(tilted[start:start + 3]),
            rtol=1.0e-6, atol=1.0e-6,
        )


def test_original_yopo_state_is_forward_biased_over_many_samples():
    rotation = np.eye(3, dtype=np.float32)
    values = np.stack([
        sample_original_yopo_observation(
            f"sample-{index}", "train", 45, rotation
        )
        for index in range(2_000)
    ])
    assert (values[:, 0] >= 0.0).all()
    assert values[:, 0].mean() > 2.0
    assert abs(values[:, 1].mean()) < 0.25
    assert abs(values[:, 2].mean()) < 0.20
    assert np.quantile(np.abs(values[:, 4]), 0.95) \
        < np.quantile(np.abs(values[:, 3]), 0.95)
