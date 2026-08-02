from __future__ import annotations

import torch

from policy.state_transform import StateTransform


def test_normalize_observation_does_not_mutate_physical_input():
    transform = StateTransform()
    physical = torch.tensor(
        [[3.0, -1.5, 0.75, 2.0, -1.0, 0.5, 30.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    before = physical.clone()

    normalized = transform.normalize_obs(physical)

    assert torch.equal(physical, before)
    assert normalized.data_ptr() != physical.data_ptr()
    assert torch.allclose(
        normalized[:, :3],
        before[:, :3] / transform.lattice_primitive.vel_max,
    )
    assert torch.allclose(
        normalized[:, 3:6],
        before[:, 3:6] / transform.lattice_primitive.acc_max,
    )
    # A far goal is represented to the network by direction at the configured
    # local goal scale, while the physical 30 m goal remains available to loss.
    expected = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    assert torch.allclose(normalized[:, 6:9], expected)


def test_normalize_observation_preserves_autograd_to_input():
    physical = torch.tensor(
        [[1.0, 2.0, 3.0, 0.5, 0.25, 0.125, 10.0, 5.0, 0.0]],
        requires_grad=True,
    )
    normalized = StateTransform().normalize_obs(physical)
    normalized.sum().backward()
    assert physical.grad is not None
    assert torch.isfinite(physical.grad).all()
