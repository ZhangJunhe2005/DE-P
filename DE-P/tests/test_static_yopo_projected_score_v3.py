import numpy as np
import torch

from loss.trajectory_sampler import QuinticTrajectorySampler
from policy.poly_solver import Poly5Solver
from policy.runtime_safety_v1 import RuntimeSafetyConfigV1, RuntimeTrajectorySafetyV1
from policy.static_yopo_projected_score_v3 import (
    ProjectedScoreConfigV3,
    differentiable_stopping_distance_loss_v3,
    project_terminal_states_v3,
    projected_score_objective_v3,
)
from tests.test_trajectory_sampler_regression import coefficient_map
from tools.train_mixed_static_yopo_v1 import projected_validation_contract_v3


def test_torch_projection_matches_runtime_discrete_projection():
    duration = 1.7
    config = ProjectedScoreConfigV3(enabled=True)
    sampler = QuinticTrajectorySampler(
        coefficient_map(duration), duration, eval_points=30
    )
    fixed = torch.zeros(3, 3, 3)
    terminal = torch.zeros_like(fixed)
    terminal[0, 0] = torch.tensor([10.0, 9.0, 12.0])
    terminal[1, 0] = torch.tensor([3.0, 2.0, 0.0])
    terminal[1, 1] = torch.tensor([1.5, 1.0, 0.0])
    terminal[2, 0] = torch.tensor([2.0, 0.0, 0.0])
    actual = project_terminal_states_v3(
        sampler, fixed, terminal, 1, 3, config
    )

    shield = RuntimeTrajectorySafetyV1(RuntimeSafetyConfigV1())
    runtime, scales, succeeded = shield.project_endstate_candidates(
        np.zeros(3), np.zeros(3), np.zeros(3), terminal.numpy(), duration
    )
    np.testing.assert_allclose(
        actual["projection_scale"].squeeze(0).numpy(), scales, atol=1e-6
    )
    assert actual["projection_succeeded"].squeeze(0).tolist() == list(succeeded)
    runtime_endpoint = np.asarray([
        [[axis.get_position(duration), axis.get_velocity(duration),
          axis.get_acceleration(duration)] for axis in candidate]
        for candidate in runtime
    ])
    np.testing.assert_allclose(
        actual["projected_derivatives"].numpy(), runtime_endpoint, atol=2e-4
    )


def synthetic_projection():
    position = torch.tensor([[[
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0],
    ], [
        [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0],
    ], [
        [0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0],
    ]]], dtype=torch.float32)
    return {
        "position_world": position,
        "velocity_world": torch.zeros_like(position),
        "acceleration_world": torch.zeros_like(position),
        "sample_times": torch.tensor([0.0, 0.85, 1.7]),
        "projection_succeeded": torch.ones(1, 3, dtype=torch.bool),
        "projection_scale": torch.ones(1, 3),
        "maximum_speed": torch.zeros(1, 3),
        "maximum_acceleration": torch.zeros(1, 3),
    }


def score_objective(scores):
    distances = torch.tensor([[[5.0, 5.0, 5.0], [5.0, 5.0, 5.0],
                               [1.5, 1.0, 0.4]]])
    return projected_score_objective_v3(
        scores, synthetic_projection(), distances, torch.eye(3).unsqueeze(0),
        torch.tensor([[10.0, 0.0, 0.0]]), torch.zeros(1, 3),
        ProjectedScoreConfigV3(enabled=True),
    )


def test_projected_score_uses_safe_then_progress_lexicographic_order():
    result = score_objective(torch.zeros(1, 3, requires_grad=True))
    assert result["safe_mask"].tolist() == [[True, False, False]]
    assert result["labels"][0, 0] < result["labels"][0, 1:].min()
    assert result["candidate_available"].item()


def test_projected_score_penalizes_unsafe_argmin_and_backpropagates():
    unsafe_scores = torch.tensor([[1.0, -2.0, 0.0]], requires_grad=True)
    safe_scores = torch.tensor([[-2.0, 1.0, 0.0]], requires_grad=True)
    unsafe = score_objective(unsafe_scores)
    safe = score_objective(safe_scores)
    assert unsafe["conditional_selection_error"].item()
    assert not safe["conditional_selection_error"].item()
    assert unsafe["per_sample_loss"].item() > safe["per_sample_loss"].item()
    unsafe["per_sample_loss"].mean().backward()
    assert torch.isfinite(unsafe_scores.grad).all()
    assert unsafe_scores.grad.abs().sum() > 0


def test_v429_offline_contract_has_exactly_two_layers():
    means = {
        "projected_candidate_availability": 0.80,
        "projected_conditional_selection_error_numerator": 0.04,
    }
    config = {"offline_gate": {
        "candidate_availability_rate_min": 0.75,
        "conditional_selection_error_rate_max": 0.10,
    }}
    layers, conditional_error, passed = projected_validation_contract_v3(
        means, config
    )
    assert tuple(layers) == ("candidate_capability", "score_selection")
    assert abs(conditional_error - 0.05) < 1.0e-12
    assert passed


def test_candidate_stopping_distance_loss_has_finite_proposal_gradient():
    class PlaneDistance:
        @staticmethod
        def get_distance_cost(position, map_id):
            distance = 2.5 - position[..., 0]
            return torch.zeros_like(distance), distance

    duration = 1.7
    sampler = QuinticTrajectorySampler(
        coefficient_map(duration), duration, eval_points=30
    )
    fixed = torch.zeros(2, 3, 3)
    terminal = torch.zeros_like(fixed, requires_grad=True)
    terminal.data[0, 0, 0] = 2.2
    terminal.data[1, 0, 0] = 0.8
    result = differentiable_stopping_distance_loss_v3(
        sampler, fixed, terminal, 1, 2, PlaneDistance(),
        torch.tensor([0]), ProjectedScoreConfigV3(enabled=True),
    )
    loss = result["per_sample_loss"].mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(terminal.grad).all()
    assert terminal.grad.abs().sum() > 0
