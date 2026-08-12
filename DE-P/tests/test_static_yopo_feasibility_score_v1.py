import torch

from policy.static_yopo_feasibility_score_v1 import (
    FeasibilityScoreConfigV1,
    feasibility_score_objective_v1,
)


def _objective(scores):
    config = FeasibilityScoreConfigV1(enabled=True)
    quality = torch.tensor([[0.8, 0.2, 0.1, 0.5]])
    clearance = torch.tensor([[0.9, 0.8, 0.2, 0.9]])
    speed = torch.tensor([[4.0, 5.0, 4.0, 7.0]])
    acceleration = torch.tensor([[4.0, 5.0, 4.0, 4.0]])
    return feasibility_score_objective_v1(
        scores, quality, clearance, speed, acceleration, config
    )


def test_feasible_labels_are_lexicographically_better():
    result = _objective(torch.zeros(1, 4, requires_grad=True))
    assert result["hard_feasible_mask"].tolist() == [[True, True, False, False]]
    assert result["labels"][0, :2].max() <= 1.0
    assert result["labels"][0, 2:].min() >= 2.0


def test_argmin_aligned_loss_penalizes_unsafe_selection_and_backpropagates():
    unsafe_selected = torch.tensor([[1.0, 0.5, -2.0, 2.0]], requires_grad=True)
    safe_selected = torch.tensor([[0.5, -2.0, 1.0, 2.0]], requires_grad=True)
    unsafe_loss = _objective(unsafe_selected)["per_sample_loss"].mean()
    safe_loss = _objective(safe_selected)["per_sample_loss"].mean()
    assert unsafe_loss > safe_loss
    unsafe_loss.backward()
    assert torch.isfinite(unsafe_selected.grad).all()
    assert unsafe_selected.grad.abs().sum() > 0


def test_all_infeasible_candidates_receive_least_violation_order():
    config = FeasibilityScoreConfigV1(enabled=True)
    scores = torch.zeros(1, 3, requires_grad=True)
    result = feasibility_score_objective_v1(
        scores,
        torch.tensor([[0.3, 0.2, 0.1]]),
        torch.tensor([[0.60, 0.20, 0.10]]),
        torch.tensor([[6.1, 8.0, 9.0]]),
        torch.tensor([[6.1, 8.0, 9.0]]),
        config,
    )
    assert not result["has_hard_feasible_candidate"].item()
    assert result["labels"].argmin(dim=1).item() == 0
    assert torch.isfinite(result["per_sample_loss"]).all()
