import torch

from loss.trajectory_sampler import QuinticTrajectorySampler
from policy.static_yopo_parity_v4_4 import (
    StaticYOPOParityConfigV44,
    exact_derivative_integral,
    static_yopo_parity_objective_v4_4,
)
from tests.test_trajectory_sampler_regression import coefficient_map


class SyntheticPlaneSafety:
    @staticmethod
    def get_distance_cost(position, map_id):
        del map_id
        distance = 4.0 - position[..., 0]
        cost = torch.exp(((1.2 - distance) / 0.6).clamp(-60.0, 60.0))
        return cost, distance


def test_exact_analytic_derivative_integral_matches_dense_numerical_result():
    duration = 1.7
    generator = torch.Generator().manual_seed(44)
    coefficient = torch.randn(4, 3, 6, generator=generator, dtype=torch.float64)
    times = torch.linspace(0.0, duration, 20_001, dtype=torch.float64)
    for order in (2, 3):
        values = torch.zeros(4, len(times), 3, dtype=torch.float64)
        for power in range(order, 6):
            scale = 1
            for factor in range(power - order + 1, power + 1):
                scale *= factor
            values += (
                coefficient[:, :, power, None].permute(0, 2, 1)
                * scale * times[None, :, None] ** (power - order)
            )
        numerical = torch.trapezoid(values.square().sum(dim=2), times, dim=1)
        analytic = exact_derivative_integral(
            coefficient, duration, derivative_order=order
        )
        torch.testing.assert_close(analytic, numerical, rtol=2.0e-7, atol=2.0e-7)


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
        predicted[2, 0, 0] = 7.0
        predicted[2, 0, 1] = 3.0
    scores = torch.tensor([[0.8, 0.5, 0.2]], requires_grad=True)
    result = static_yopo_parity_objective_v4_4(
        sampler=sampler,
        safety_loss=SyntheticPlaneSafety(),
        fixed=fixed,
        predicted=predicted,
        predicted_scores=scores,
        goal_world=torch.tensor([[10.0, 0.0, 0.0]]).repeat_interleave(3, 0),
        map_id=torch.tensor([0]),
        batch_size=1,
        candidate_count=3,
        route_goal_distance=torch.tensor([10.0]),
        config=StaticYOPOParityConfigV44(enabled=True),
    )
    return predicted, scores, result


def test_v44_label_is_only_original_smooth_static_guidance_cost():
    predicted, scores, result = synthetic_objective()
    expected = (
        result["candidate_smooth_cost"]
        + result["candidate_static_cost"]
        + result["candidate_guidance_cost"]
    )
    torch.testing.assert_close(result["score_label"], expected.detach())
    assert not result["score_label"].requires_grad
    assert result["candidate_kinodynamic_cost"].count_nonzero() == 0
    assert result["ranking_loss"] > 0.0
    result["total_loss"].backward()
    assert torch.isfinite(predicted.grad).all()
    assert torch.isfinite(scores.grad).all()
    assert (predicted.grad.reshape(3, -1).abs().sum(dim=1) > 0).all()
    assert scores.grad.abs().sum() > 0


def test_v44_has_one_diagnostic_regret_and_no_qualification_gate():
    _, _, result = synthetic_objective()
    torch.testing.assert_close(
        result["per_sample_score_oracle_regret"],
        result["per_sample_projected_selection_regret"],
    )
    contract = StaticYOPOParityConfigV44(enabled=True).contract()
    assert contract["qualification_gate"] == "none"
    assert contract["score_calibration"] == "single_within_sample_listwise_kl"
