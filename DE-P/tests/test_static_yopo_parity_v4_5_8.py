import math

import pytest
import torch

from policy.static_yopo_parity_v4_5_8 import (
    StaticYOPOParityConfigV458,
    localized_static_clearance_cost_v4_5_8,
)


def evaluate(clearance):
    return localized_static_clearance_cost_v4_5_8(
        clearance,
        vehicle_radius=0.30,
        clear_distance=0.50,
        far_cost=0.001,
        far_decay=0.10,
        radius_boundary_cost=1.0,
        collision_surrogate_cost=3.0,
        penetration_scale=0.05,
        maximum_training_cost=6.0,
    )


def test_localized_clearance_curve_has_requested_three_regions():
    distance = torch.tensor(
        [1.0, 0.60, 0.50, 0.40, 0.31, 0.30, 0.20],
        dtype=torch.float64,
    )
    cost, collision = evaluate(distance)
    assert torch.isfinite(cost).all()
    assert torch.all(cost[:-1] < cost[1:])
    assert cost[2].item() == pytest.approx(0.001)
    assert cost[3].item() == pytest.approx(math.sqrt(0.001 * 1.0))
    assert cost[0].item() < 1.0e-5
    assert collision.tolist() == [False, False, False, False, False, True, True]
    assert cost[5].item() == pytest.approx(3.0)
    assert cost[6].item() == pytest.approx(3.0 + 3.0 * (1.0 - math.exp(-2.0)))


def test_localized_clearance_curve_is_continuous_at_half_meter():
    epsilon = 1.0e-7
    distance = torch.tensor(
        [0.50 + epsilon, 0.50, 0.50 - epsilon], dtype=torch.float64
    )
    cost, _ = evaluate(distance)
    assert (cost.max() - cost.min()).item() < 1.0e-6


def test_localized_clearance_surrogate_has_finite_outward_gradient():
    distance = torch.tensor([0.60, 0.40, 0.25], requires_grad=True)
    cost, _ = evaluate(distance)
    cost.sum().backward()
    assert torch.isfinite(distance.grad).all()
    assert torch.all(distance.grad < 0.0)


def test_v458_disables_overlapping_clearance_supervision():
    config = StaticYOPOParityConfigV458(enabled=True)
    config.validate()
    parent = config.as_v456()
    assert parent.clearance_score_weight == 0.0
    assert parent.clearance_pairwise_weight == 0.0
    assert parent.dangerous_segment_weight == 0.0
    contract = config.contract()
    assert contract["qualification_gate"] == "none"
    assert contract["static_clearance_semantics"]["collision_runtime"] \
        == "positive_infinity_reject"
