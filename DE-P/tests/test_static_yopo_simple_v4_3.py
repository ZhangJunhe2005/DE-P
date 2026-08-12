import torch

from loss.trajectory_sampler import QuinticTrajectorySampler
from policy.static_yopo_simple_v4_3 import (
    SimpleYOPOConfigV43, simple_yopo_objective_v4_3,
)
from tests.test_trajectory_sampler_regression import coefficient_map


class SyntheticPlaneSafety:
    @staticmethod
    def get_distance_cost(position, map_id):
        del map_id
        distance = 4.0 - position[..., 0]
        cost = torch.exp(((1.2 - distance) / 0.6).clamp(-60.0, 60.0))
        return cost, distance


def synthetic_objective():
    duration = 1.7
    sampler = QuinticTrajectorySampler(
        coefficient_map(duration), duration, eval_points=30
    )
    fixed = torch.zeros(3, 3, 3)
    predicted = torch.zeros(3, 3, 3, requires_grad=True)
    with torch.no_grad():
        predicted[0, 0, 0] = 1.5
        predicted[1, 0, :2] = torch.tensor([3.0, 0.8])
        predicted[2, 0, 0] = 8.0
        predicted[2, 0, 1] = 8.0
    scores = torch.zeros(1, 3, requires_grad=True)
    result = simple_yopo_objective_v4_3(
        sampler=sampler,
        safety_loss=SyntheticPlaneSafety(),
        fixed=fixed,
        predicted=predicted,
        predicted_scores=scores,
        goal_world=torch.tensor([[10.0, 0.0, 0.0]]).repeat_interleave(3, 0),
        map_id=torch.tensor([0]),
        batch_size=1,
        candidate_count=3,
        route_goal_distance=torch.tensor([20.0]),
        config=SimpleYOPOConfigV43(enabled=True),
    )
    return predicted, scores, result


def test_v43_has_one_detached_total_cost_score_label():
    predicted, scores, result = synthetic_objective()
    assert result["score_label"].shape == (3,)
    assert not result["score_label"].requires_grad
    expected = (
        result["candidate_smooth_cost"]
        + result["candidate_static_cost"]
        + result["candidate_guidance_cost"]
        + result["candidate_kinodynamic_cost"]
    )
    torch.testing.assert_close(result["score_label"], expected.detach())
    assert torch.isfinite(result["total_loss"])
    result["total_loss"].backward()
    assert torch.isfinite(predicted.grad).all()
    assert torch.isfinite(scores.grad).all()
    # All proposals, not only the selected proposal, train the candidate head.
    assert (predicted.grad.reshape(3, -1).abs().sum(dim=1) > 0).all()
    assert (scores.grad.abs() > 0).all()


def test_v43_hardware_limit_is_a_gradient_not_a_score_qualification_class():
    _, _, result = synthetic_objective()
    hardware = result["candidate_kinodynamic_cost"].reshape(1, 3)
    assert hardware[0, 2] > hardware[0, :2].max()
    assert result["per_sample_feasible_candidate_count"].item() < 3
    contract = SimpleYOPOConfigV43(enabled=True).contract()
    assert contract["qualification_gate"] == "none"
    assert contract["candidate_gradient"] == "all_15_candidates"
