import torch

from loss.trajectory_sampler import QuinticTrajectorySampler
from policy.static_yopo_parity_v4_5_1 import (
    StaticYOPOParityConfigV451,
    smooth_kinematic_excess_cost_v4_5_1,
    standardize_relative_cost_v4_5_1,
    static_yopo_parity_objective_v4_5_1,
)
from tests.test_trajectory_sampler_regression import coefficient_map
from tools.train_mixed_static_yopo_v1 import training_contract_classification


class SyntheticPlaneSafety:
    @staticmethod
    def get_distance_cost(position, map_id):
        del map_id
        distance = 4.0 - position[..., 0]
        cost = torch.exp(((1.2 - distance) / 0.6).clamp(-60.0, 60.0))
        return cost, distance


def objective_inputs():
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
        sampler=sampler,
        safety_loss=SyntheticPlaneSafety(),
        fixed=fixed,
        predicted=predicted,
        predicted_scores=scores,
        goal_world=torch.tensor([[10.0, 0.0, 0.0]]).repeat_interleave(15, 0),
        map_id=torch.tensor([0]),
        batch_size=1,
        candidate_count=15,
        route_goal_distance=torch.tensor([10.0]),
    )
    return predicted, scores, common


def test_relative_cost_is_invariant_to_positive_affine_scale():
    cost = torch.tensor([[1.0, 3.0, 8.0], [9.0, 9.5, 12.0]])
    reference, scale = standardize_relative_cost_v4_5_1(cost)
    transformed, transformed_scale = standardize_relative_cost_v4_5_1(
        7.0 * cost + 113.0
    )
    torch.testing.assert_close(reference, transformed)
    torch.testing.assert_close(transformed_scale, 7.0 * scale)
    torch.testing.assert_close(reference.mean(dim=1), torch.zeros(2), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        reference.std(dim=1, unbiased=False), torch.ones(2), atol=1e-6, rtol=0
    )


def test_smooth_kinematic_excess_is_finite_monotonic_and_differentiable():
    speed = torch.tensor([[3.0, 6.0, 9.0]], requires_grad=True)
    acceleration = torch.tensor([[3.0, 6.0, 12.0]], requires_grad=True)
    cost = smooth_kinematic_excess_cost_v4_5_1(
        speed,
        acceleration,
        max_speed=6.0,
        max_acceleration=6.0,
        speed_weight=1.0,
        acceleration_weight=1.0,
        softness=0.05,
    )
    assert torch.isfinite(cost).all()
    assert bool(cost[0, 0] < cost[0, 1] < cost[0, 2])
    assert float(cost[0, 0]) < 1.0e-8
    cost.sum().backward()
    assert torch.isfinite(speed.grad).all()
    assert torch.isfinite(acceleration.grad).all()
    assert float(speed.grad[0, 2]) > 0.0
    assert float(acceleration.grad[0, 2]) > 0.0


def test_v451_objective_has_relative_labels_kinematic_gradient_and_vertical_metrics():
    predicted, scores, common = objective_inputs()
    result = static_yopo_parity_objective_v4_5_1(
        **common, config=StaticYOPOParityConfigV451(enabled=True)
    )
    label = result["score_label"].reshape(1, 15)
    torch.testing.assert_close(label.mean(dim=1), torch.zeros(1), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        label.std(dim=1, unbiased=False), torch.ones(1), atol=1e-5, rtol=0
    )
    assert float(result["kinematic_loss"]) > 0.0
    assert result["candidate_kinodynamic_cost"].shape == (15,)
    assert result["per_sample_selected_absolute_vertical_displacement"].shape == (1,)
    assert result["per_sample_oracle_large_vertical_maneuver"].shape == (1,)
    result["total_loss"].backward()
    assert torch.isfinite(predicted.grad).all()
    assert torch.isfinite(scores.grad).all()
    assert predicted.grad.abs().sum() > 0.0
    assert scores.grad.abs().sum() > 0.0
    assert StaticYOPOParityConfigV451(enabled=True).contract()[
        "qualification_gate"
    ] == "none"


def test_v45_family_and_shakedown_cannot_claim_production_qualification():
    v45 = training_contract_classification({
        "validation": {"contract_version": "route_a_v4_5_bounded_danger_v1"}
    })
    assert v45["ungated_static"] is True
    shakedown = training_contract_classification({
        "validation": {
            "contract_version": "route_a_v4_5_1_relative_kinematic_v1"
        },
        "experiment_role": "mixed_five_epoch_shakedown",
    })
    assert shakedown == {
        "three_layer": False,
        "ungated_static": True,
        "diagnostic": True,
    }
