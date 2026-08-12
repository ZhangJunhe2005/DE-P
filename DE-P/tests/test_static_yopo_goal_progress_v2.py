import torch

from policy.static_yopo_goal_progress_v2 import (
    GoalProgressConfigV2,
    goal_directed_progress_objective_v2,
)


def test_signed_goal_progress_distinguishes_forward_lateral_and_reverse():
    scores = torch.tensor([[0.0, 1.0, 2.0]], requires_grad=True)
    displacement = torch.tensor([[[3.0, 0.0, 0.0],
                                  [0.0, 3.0, 0.0],
                                  [-3.0, 0.0, 0.0]]], requires_grad=True)
    goal = torch.tensor([[10.0, 0.0, 0.0]])
    result = goal_directed_progress_objective_v2(
        scores, displacement, goal,
        torch.ones((1, 3), dtype=torch.bool),
        GoalProgressConfigV2(enabled=True),
    )
    torch.testing.assert_close(
        result["signed_goal_progress"], torch.tensor([[3.0, 0.0, -3.0]])
    )
    assert result["selected_goal_progress"].item() == 3.0
    assert not result["selected_reverse"].item()
    result["per_sample_loss"].mean().backward()
    assert torch.isfinite(displacement.grad).all()


def test_goal_progress_candidate_generation_does_not_require_hardware_mask():
    scores = torch.zeros((1, 2))
    displacement = torch.tensor([[[4.0, 0.0, 0.0], [0.2, 0.0, 0.0]]],
                                requires_grad=True)
    result = goal_directed_progress_objective_v2(
        scores, displacement, torch.tensor([[1.0, 0.0, 0.0]]),
        torch.ones((1, 2), dtype=torch.bool),
        GoalProgressConfigV2(
            enabled=True, minimum_goal_progress_candidates=2,
        ),
    )
    result["per_sample_loss"].mean().backward()
    assert displacement.grad is not None
    assert torch.isfinite(displacement.grad).all()
    assert result["goal_progress_candidate_count"].item() == 1
