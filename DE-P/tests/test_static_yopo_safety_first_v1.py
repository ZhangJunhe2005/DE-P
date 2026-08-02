import torch

from policy.static_yopo_safety_first_v1 import (
    StaticSafetyFirstConfigV1,
    safety_first_score_objective_v1,
)


def test_unsafe_low_score_is_ranked_behind_safe_candidate():
    config = StaticSafetyFirstConfigV1(enabled=True)
    predicted = torch.tensor([[0.0, 2.0, 3.0]], requires_grad=True)
    base = torch.tensor([[0.1, 0.2, 0.3]])
    clearance = torch.tensor([[0.2, 1.0, 1.2]])
    static_cost = torch.tensor([[4.0, 0.2, 0.1]])
    result = safety_first_score_objective_v1(
        predicted, base, clearance, static_cost, config
    )
    assert result["labels"][0, 0] > result["labels"][0, 1]
    assert result["ranking_per_sample"].item() > 0
    assert result["selected_unsafe"].item()
    loss = (
        result["regression_per_sample"]
        + config.ranking_weight * result["ranking_per_sample"]
        + config.safety_cvar_weight * result["safety_cvar_per_sample"]
    ).mean()
    loss.backward()
    assert torch.isfinite(predicted.grad).all()


def test_no_safe_unsafe_pair_has_zero_ranking_loss():
    config = StaticSafetyFirstConfigV1(enabled=True)
    predicted = torch.zeros(2, 3)
    base = torch.zeros_like(predicted)
    clearance = torch.tensor([[1.0, 1.1, 1.2], [0.1, 0.2, 0.3]])
    static_cost = torch.ones_like(predicted)
    result = safety_first_score_objective_v1(
        predicted, base, clearance, static_cost, config
    )
    assert torch.equal(result["ranking_per_sample"], torch.zeros(2))


def test_hardware_violation_is_unsafe_even_with_clear_static_geometry():
    config = StaticSafetyFirstConfigV1(enabled=True)
    predicted = torch.tensor([[0.0, 1.0]])
    base = torch.zeros_like(predicted)
    clearance = torch.ones_like(predicted)
    static_cost = torch.zeros_like(predicted)
    result = safety_first_score_objective_v1(
        predicted, base, clearance, static_cost, config,
        trajectory_max_speed=torch.tensor([[7.0, 5.0]]),
        trajectory_max_acceleration=torch.tensor([[4.0, 4.0]]),
    )
    assert result["hardware_unsafe_mask"].tolist() == [[True, False]]
    assert result["kinematic_per_sample"].item() > 0


def test_anti_hover_thresholds_are_not_part_of_safety_label_rewrite():
    fields = StaticSafetyFirstConfigV1.__dataclass_fields__
    assert "hover_selection_rate_max" not in fields
    assert "selected_endpoint_speed_mean_min" not in fields
    assert "selected_endpoint_distance_mean_min" not in fields


def test_continuous_kinematic_cost_orders_candidates_when_all_are_unsafe():
    config = StaticSafetyFirstConfigV1(enabled=True)
    predicted = torch.zeros(1, 3)
    base = torch.zeros_like(predicted)
    clearance = torch.ones_like(predicted)
    static_cost = torch.zeros_like(predicted)
    result = safety_first_score_objective_v1(
        predicted, base, clearance, static_cost, config,
        trajectory_max_speed=torch.tensor([[7.0, 8.0, 9.0]]),
        trajectory_max_acceleration=torch.tensor([[7.0, 8.0, 9.0]]),
        candidate_kinematic_cost=torch.tensor([[0.2, 0.6, 1.4]]),
    )
    assert result["hardware_unsafe_mask"].all()
    assert result["ranking_per_sample"].item() == 0.0
    assert torch.allclose(
        result["labels"], torch.tensor([[5.2, 5.6, 6.4]])
    )
