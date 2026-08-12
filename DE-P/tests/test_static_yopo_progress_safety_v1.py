import torch

from policy.static_yopo_progress_safety_v1 import (
    ProgressSafetyConfigV1,
    progress_safety_objective_v1,
)


def test_safe_progress_ranks_a_moving_candidate_ahead_of_safe_hover():
    config = ProgressSafetyConfigV1(
        enabled=True, minimum_progress_m=1.0, preferred_progress_m=3.5
    )
    predicted = torch.tensor([[0.0, 1.0, 2.0]], requires_grad=True)
    distance = torch.tensor([[0.2, 2.0, 4.0]], requires_grad=True)
    safe = torch.tensor([[True, True, True]])
    result = progress_safety_objective_v1(
        predicted, distance, safe, config
    )
    assert result["ranking_per_sample"].item() > 0
    assert result["candidate_label_cost"][0, 0] > result[
        "candidate_label_cost"
    ][0, 2]
    assert result["safe_progress_candidate_count"].item() == 2
    loss = (
        result["per_sample_loss"]
        + config.ranking_weight * result["ranking_per_sample"]
    ).mean()
    loss.backward()
    assert torch.isfinite(predicted.grad).all()
    assert torch.isfinite(distance.grad).all()


def test_unsafe_long_candidate_gets_no_progress_preference():
    config = ProgressSafetyConfigV1(enabled=True)
    predicted = torch.tensor([[0.0, 1.0, 2.0]])
    distance = torch.tensor([[5.0, 0.2, 2.0]])
    safe = torch.tensor([[False, True, True]])
    result = progress_safety_objective_v1(
        predicted, distance, safe, config
    )
    assert result["candidate_label_cost"][0, 0].item() == 0.0
    assert not result["safe_mask"][0, 0]


def test_no_safe_candidate_has_finite_zero_generation_loss():
    config = ProgressSafetyConfigV1(enabled=True)
    predicted = torch.zeros(2, 3)
    distance = torch.ones(2, 3, requires_grad=True)
    safe = torch.zeros(2, 3, dtype=torch.bool)
    result = progress_safety_objective_v1(
        predicted, distance, safe, config
    )
    assert torch.equal(result["per_sample_loss"], torch.zeros(2))
    assert torch.isfinite(result["per_sample_loss"]).all()
