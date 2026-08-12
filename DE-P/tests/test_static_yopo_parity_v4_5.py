import torch

from loss.trajectory_sampler import QuinticTrajectorySampler
from policy.static_yopo_parity_v4_4 import (
    StaticYOPOParityConfigV44,
    static_yopo_parity_objective_v4_4,
)
from policy.static_yopo_parity_v4_5 import (
    StaticYOPOParityConfigV45,
    bounded_dangerous_segment_cost_v4_5,
    static_yopo_parity_objective_v4_5,
)
from tests.test_trajectory_sampler_regression import coefficient_map


class SyntheticPlaneSafety:
    @staticmethod
    def get_distance_cost(position, map_id):
        del map_id
        distance = 4.0 - position[..., 0]
        cost = torch.exp(((1.2 - distance) / 0.6).clamp(-60.0, 60.0))
        return cost, distance


def inputs():
    sampler = QuinticTrajectorySampler(
        coefficient_map(1.7), 1.7, eval_points=30
    )
    fixed = torch.zeros(15, 3, 3)
    predicted = torch.zeros(15, 3, 3, requires_grad=True)
    with torch.no_grad():
        predicted[:, 0, 0] = torch.linspace(1.0, 7.0, 15)
        predicted[:, 0, 1] = torch.linspace(-1.0, 1.0, 15)
        predicted[:, 0, 2] = torch.linspace(-2.0, 2.0, 15)
    scores = torch.linspace(1.0, 0.0, 15).reshape(1, 15).requires_grad_()
    common = dict(
        sampler=sampler, safety_loss=SyntheticPlaneSafety(), fixed=fixed,
        predicted=predicted, predicted_scores=scores,
        goal_world=torch.tensor([[10.0, 0.0, 0.0]]).repeat_interleave(15, 0),
        map_id=torch.tensor([0]), batch_size=1, candidate_count=15,
        route_goal_distance=torch.tensor([10.0]),
    )
    return predicted, scores, common


def test_bounded_dangerous_segment_is_smooth_finite_and_strictly_bounded():
    samples = torch.tensor([[[0.0, 0.2, 1.0, 20.0, 1.0, 0.2, 0.0]]],
                           requires_grad=True)
    cost = bounded_dangerous_segment_cost_v4_5(
        samples, weight=0.15, window=3, focus=8.0
    )
    assert 0.0 <= float(cost) <= 0.15
    cost.sum().backward()
    assert torch.isfinite(samples.grad).all()
    assert samples.grad.abs().sum() > 0.0


def test_zero_weight_v45_is_v44_objective_equivalent():
    _, _, common = inputs()
    v44 = static_yopo_parity_objective_v4_4(
        **common, config=StaticYOPOParityConfigV44(enabled=True)
    )
    v45 = static_yopo_parity_objective_v4_5(
        **common, config=StaticYOPOParityConfigV45(
            enabled=True, dangerous_segment_weight=0.0
        )
    )
    for name in (
        "total_loss", "trajectory_loss", "score_loss", "smoothness_loss",
        "static_safety_loss", "guidance_loss", "score_label",
    ):
        torch.testing.assert_close(v45[name], v44[name], rtol=0.0, atol=0.0)


def test_v45_addition_is_bounded_has_finite_gradients_and_vertical_diagnostics():
    predicted, scores, common = inputs()
    result = static_yopo_parity_objective_v4_5(
        **common, config=StaticYOPOParityConfigV45(enabled=True)
    )
    candidate = result["candidate_dangerous_segment_cost"]
    assert float(candidate.min()) >= 0.0
    assert float(candidate.max()) <= 0.15
    assert result["score_label"].shape == (15,)
    assert result["per_sample_selected_primitive_row"].shape == (1,)
    assert result["per_sample_oracle_primitive_row"].shape == (1,)
    assert result["per_sample_selected_vertical_displacement"].shape == (1,)
    result["total_loss"].backward()
    assert torch.isfinite(predicted.grad).all()
    assert torch.isfinite(scores.grad).all()
    assert predicted.grad.abs().sum() > 0.0
    assert scores.grad.abs().sum() > 0.0
    assert StaticYOPOParityConfigV45(enabled=True).contract()[
        "qualification_gate"
    ] == "none"
