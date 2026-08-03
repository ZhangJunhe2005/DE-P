import torch

from policy.static_yopo_preventive_safety_v1 import (
    PreventiveSafetyConfigV1,
    preventive_safety_objective_v1,
)


def test_speed_increases_anticipatory_clearance_without_exceeding_cap():
    config = PreventiveSafetyConfigV1(enabled=True)
    scores = torch.zeros(2, 3)
    clearance = torch.full((2, 3), 3.0)
    depth = torch.ones(2, 1, 8, 8)
    result = preventive_safety_objective_v1(
        scores, clearance, torch.tensor([0.0, 6.0]), depth, config
    )
    required = result["required_clearance"]
    assert required[1] > required[0]
    assert required[1] <= config.maximum_clearance_m


def test_near_field_samples_receive_more_weight_and_finite_gradient():
    config = PreventiveSafetyConfigV1(enabled=True)
    scores = torch.tensor([[0.0, 2.0, 3.0]], requires_grad=True)
    clearance = torch.tensor(
        [[0.8, 1.8, 2.2]], requires_grad=True
    )
    depth = torch.full((1, 1, 8, 8), 0.10)
    result = preventive_safety_objective_v1(
        scores, clearance, torch.tensor([4.0]), depth, config
    )
    assert result["sample_weight"].item() > 1.0
    assert result["ranking_per_sample"].item() > 0.0
    assert result["clear_candidate_count"].item() >= 1
    loss = result["per_sample_loss"] + result["ranking_per_sample"]
    loss.mean().backward()
    assert torch.isfinite(scores.grad).all()
    assert torch.isfinite(clearance.grad).all()


def test_clear_candidate_is_given_better_detached_score_label():
    config = PreventiveSafetyConfigV1(enabled=True)
    result = preventive_safety_objective_v1(
        torch.zeros(1, 2), torch.tensor([[0.6, 2.5]]),
        torch.tensor([2.0]), torch.ones(1, 1, 4, 4), config,
    )
    assert result["candidate_label_cost"][0, 0] > (
        result["candidate_label_cost"][0, 1]
    )
