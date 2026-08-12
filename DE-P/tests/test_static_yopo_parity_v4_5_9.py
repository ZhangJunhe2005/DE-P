import math

import pytest
import torch

from policy.static_yopo_parity_v4_5_8 import (
    localized_static_clearance_cost_v4_5_8,
)
from policy.static_yopo_parity_v4_5_9 import (
    StaticYOPOParityConfigV459,
    time_mean_localized_static_cost_v4_5_9,
)


def test_v459_safe_side_warning_is_weak_and_collision_remains_authoritative():
    config = StaticYOPOParityConfigV459(enabled=True)
    distance = torch.tensor(
        [1.0, 0.50, 0.40, 0.35, 0.31, 0.30, 0.20],
        dtype=torch.float64,
    )
    cost, collision = localized_static_clearance_cost_v4_5_8(
        distance,
        vehicle_radius=config.vehicle_radius_m,
        clear_distance=config.clear_distance_m,
        far_cost=config.far_clearance_cost,
        far_decay=config.far_decay_m,
        radius_boundary_cost=config.radius_boundary_cost,
        collision_surrogate_cost=config.collision_surrogate_cost,
        penetration_scale=config.penetration_scale_m,
        maximum_training_cost=config.maximum_training_cost,
    )
    assert cost[0].item() < 1.0e-5
    assert cost[1].item() == pytest.approx(0.001)
    assert cost[2].item() == pytest.approx(math.sqrt(0.001 * 0.1))
    assert cost[3].item() < 0.04
    assert cost[4].item() < 0.10
    assert collision.tolist() == [False, False, False, False, False, True, True]
    assert cost[5].item() == pytest.approx(3.0)
    assert torch.isfinite(cost).all()


def test_v459_uses_time_mean_instead_of_minimum_clearance_cost():
    config = StaticYOPOParityConfigV459(
        enabled=True, static_safety_samples=4, dangerous_segment_window=4,
    )
    distances = torch.tensor([
        [
            [0.60, 0.60, 0.60, 0.60],
            [0.60, 0.40, 0.60, 0.60],
            [0.60, 0.29, 0.60, 0.60],
        ]
    ], dtype=torch.float64, requires_grad=True)
    cost, collision, minimum, point_cost = (
        time_mean_localized_static_cost_v4_5_9(
            distances, config=config,
        )
    )
    assert cost.shape == (1, 3)
    assert point_cost.shape == distances.shape
    assert torch.allclose(
        minimum, torch.tensor([[0.60, 0.40, 0.29]], dtype=torch.float64)
    )
    assert collision.tolist() == [[False, False, True]]
    # One close point contributes one quarter of the trajectory cost; it does
    # not price all four samples as if all of them were at 0.40 m.
    single_close_cost, _ = localized_static_clearance_cost_v4_5_8(
        torch.tensor([0.40], dtype=torch.float64),
        vehicle_radius=config.vehicle_radius_m,
        clear_distance=config.clear_distance_m,
        far_cost=config.far_clearance_cost,
        far_decay=config.far_decay_m,
        radius_boundary_cost=config.radius_boundary_cost,
        collision_surrogate_cost=config.collision_surrogate_cost,
        penetration_scale=config.penetration_scale_m,
        maximum_training_cost=config.maximum_training_cost,
    )
    assert cost[0, 1] < single_close_cost[0]
    cost.sum().backward()
    assert torch.isfinite(distances.grad).all()
    assert torch.all(distances.grad < 0.0)


def test_v459_contract_keeps_one_objective_and_no_new_gate():
    config = StaticYOPOParityConfigV459(enabled=True)
    config.validate()
    parent = config.as_v456()
    assert parent.clearance_score_weight == 0.0
    assert parent.clearance_pairwise_weight == 0.0
    assert parent.dangerous_segment_weight == 0.0
    contract = config.contract()
    assert contract["qualification_gate"] == "none"
    assert contract["static_clearance_semantics"]["trajectory_aggregation"] \
        == "time_mean_over_81_samples"
    assert contract["static_clearance_semantics"]["collision_runtime"] \
        == "positive_infinity_reject"
