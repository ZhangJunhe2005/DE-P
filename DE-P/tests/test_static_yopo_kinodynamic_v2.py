import torch

from loss.trajectory_sampler import QuinticTrajectorySampler
from policy.static_yopo_kinodynamic_v2 import (
    KinodynamicFeasibilityConfigV2,
    dense_quintic_kinodynamic_objective_v2,
)
from policy.state_transform import StateTransform


def coefficient_map(duration=1.7):
    matrix = torch.zeros(6, 6, dtype=torch.float64)
    # Boundary order in the sampler is p0,v0,a0,pT,vT,aT.
    matrix[0] = torch.tensor([1, 0, 0, 0, 0, 0], dtype=torch.float64)
    matrix[1] = torch.tensor([0, 1, 0, 0, 0, 0], dtype=torch.float64)
    matrix[2] = torch.tensor([0, 0, 2, 0, 0, 0], dtype=torch.float64)
    t = float(duration)
    matrix[3] = torch.tensor([1, t, t**2, t**3, t**4, t**5])
    matrix[4] = torch.tensor([0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4])
    matrix[5] = torch.tensor([0, 0, 2, 6*t, 12*t**2, 20*t**3])
    return torch.linalg.inv(matrix)


def objective(end, config=None):
    sampler = QuinticTrajectorySampler(
        coefficient_map().float(), duration=1.7, eval_points=60
    )
    start = torch.zeros_like(end)
    return dense_quintic_kinodynamic_objective_v2(
        sampler, start, end, batch_size=1,
        candidate_count=end.shape[0],
        config=config or KinodynamicFeasibilityConfigV2(enabled=True),
    )


def test_dense_violation_reaches_terminal_state_gradient():
    end = torch.tensor([
        [[10.0, 1.0, 0.5], [2.0, -1.0, 0.4], [1.0, 0.5, -0.2]],
        [[8.0, -2.0, 1.0], [-3.0, 1.0, -0.5], [2.0, -1.0, 0.8]],
        [[7.0, 2.0, -1.0], [2.0, 2.0, 1.0], [-2.0, 1.0, 0.5]],
    ], requires_grad=True)
    result = objective(end)
    assert result["per_sample_loss"].item() > 0
    result["per_sample_loss"].mean().backward()
    assert end.grad is not None
    assert torch.isfinite(end.grad).all()
    assert end.grad.abs().sum().item() > 0


def test_short_straight_candidate_is_hard_feasible():
    end = torch.zeros(3, 3, 3)
    end[:, 0, 0] = torch.tensor([1.0, 1.5, 2.0])
    result = objective(end)
    assert result["feasible_mask"].all()
    assert result["feasible_candidate_count"].item() == 3
    assert torch.allclose(
        result["time_dilation_ratio"], torch.ones_like(
            result["time_dilation_ratio"]
        )
    )


def test_ego_time_dilation_ratio_matches_physical_formula():
    end = torch.zeros(3, 3, 3)
    end[:, 0, 0] = torch.tensor([8.0, 9.0, 10.0])
    result = objective(end)
    expected = torch.maximum(
        result["maximum_speed"] / 6.0,
        torch.sqrt(result["maximum_acceleration"] / 6.0),
    ).clamp_min(1.0)
    assert torch.allclose(result["time_dilation_ratio"], expected)
    assert (result["time_dilation_ratio"] > 1.0).all()


def test_normal_acceleration_penalty_is_trajectory_relative():
    straight = torch.zeros(1, 3, 3)
    straight[0, 0, 0] = 6.0
    curved = straight.clone()
    curved[0, 1, 1] = 6.0
    straight_result = objective(straight)
    curved_result = objective(curved)
    assert (
        curved_result["maximum_normal_acceleration"].item()
        > straight_result["maximum_normal_acceleration"].item()
    )


def test_continuous_candidate_cost_orders_all_unsafe_candidates():
    end = torch.zeros(3, 3, 3)
    end[:, 0, 0] = torch.tensor([7.0, 9.0, 11.0])
    result = objective(end)
    costs = result["candidate_label_cost"][0]
    assert costs[0] < costs[1] < costs[2]


def test_gradient_crosses_yopo_state_transform_for_every_candidate():
    transform = StateTransform()
    raw = torch.zeros(1, 9, 3, 5, requires_grad=True)
    # A long radial prediction recreates the initial all-infeasible condition.
    raw.data[:, 2] = 0.8
    endstate = transform.pred_to_endstate(raw)
    flat = endstate.permute(0, 2, 3, 1).reshape(15, 9)
    predicted = torch.stack(
        (flat[:, 0:3], flat[:, 3:6], flat[:, 6:9]), dim=2
    )
    start = torch.zeros_like(predicted)
    sampler = QuinticTrajectorySampler(
        coefficient_map().float(), duration=1.7, eval_points=60
    )
    result = dense_quintic_kinodynamic_objective_v2(
        sampler, start, predicted, 1, 15,
        KinodynamicFeasibilityConfigV2(enabled=True),
    )
    result["per_sample_loss"].mean().backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all()
    per_candidate_gradient = raw.grad.permute(0, 2, 3, 1).reshape(15, 9)
    assert (per_candidate_gradient.abs().sum(dim=1) > 0).all()
