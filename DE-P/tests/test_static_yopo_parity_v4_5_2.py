import torch

from policy.static_yopo_parity_v4_5_1 import (
    smooth_kinematic_excess_cost_v4_5_1,
)
from policy.static_yopo_parity_v4_5_2 import (
    StaticYOPOParityConfigV452,
    linear_kinematic_excess_cost_v4_5_2,
    static_yopo_parity_objective_v4_5_2,
)
from tests.test_static_yopo_parity_v4_5_1 import objective_inputs
from tools.train_mixed_static_yopo_v1 import training_contract_classification


def test_linear_softplus_has_stronger_near_limit_gradient_than_squared_v451():
    speed_linear = torch.tensor([[5.8, 6.0, 6.2]], requires_grad=True)
    acceleration_linear = torch.full_like(speed_linear, 5.0, requires_grad=True)
    linear = linear_kinematic_excess_cost_v4_5_2(
        speed_linear, acceleration_linear, max_speed=6.0, max_acceleration=6.0,
        speed_weight=1.0, acceleration_weight=1.0, softness=0.05,
    )
    linear.sum().backward()

    speed_squared = torch.tensor([[5.8, 6.0, 6.2]], requires_grad=True)
    acceleration_squared = torch.full_like(
        speed_squared, 5.0, requires_grad=True
    )
    squared = smooth_kinematic_excess_cost_v4_5_1(
        speed_squared, acceleration_squared,
        max_speed=6.0, max_acceleration=6.0,
        speed_weight=1.0, acceleration_weight=1.0, softness=0.05,
    )
    squared.sum().backward()
    assert torch.isfinite(linear).all()
    assert bool(linear[0, 0] < linear[0, 1] < linear[0, 2])
    assert float(speed_linear.grad[0, 1]) > float(speed_squared.grad[0, 1])


def test_v452_decomposes_geometry_hardware_oracle_and_conditional_selection():
    predicted, scores, common = objective_inputs()
    result = static_yopo_parity_objective_v4_5_2(
        **common, config=StaticYOPOParityConfigV452(enabled=True)
    )
    for name in (
        "speed_unsafe_selection", "acceleration_unsafe_selection",
        "collision_free_candidate_count", "hardware_feasible_candidate_count",
        "collision_free_candidate_available",
        "physical_feasible_candidate_available",
        "conditional_collision_selection_error",
        "conditional_physical_selection_error", "oracle_collision_unsafe",
        "oracle_speed_unsafe", "oracle_acceleration_unsafe",
        "oracle_hardware_unsafe", "oracle_physical_unsafe",
    ):
        value = result[f"per_sample_{name}"]
        assert value.shape == (1,)
        assert torch.isfinite(value).all()
    assert result["candidate_minimum_clearance"].shape == (15,)
    assert result["candidate_maximum_speed"].shape == (15,)
    assert result["candidate_maximum_acceleration"].shape == (15,)
    result["total_loss"].backward()
    assert torch.isfinite(predicted.grad).all()
    assert torch.isfinite(scores.grad).all()


def test_v452_is_ungated_and_shakedown_is_diagnostic_only():
    config = StaticYOPOParityConfigV452(enabled=True)
    assert config.relative_order_weight == 1.0
    assert config.contract()["qualification_gate"] == "none"
    semantics = training_contract_classification({
        "validation": {
            "contract_version": (
                "route_a_v4_5_2_calibrated_relative_kinematic_v1"
            )
        },
        "experiment_role": "mixed_five_epoch_shakedown",
    })
    assert semantics == {
        "three_layer": False,
        "ungated_static": True,
        "diagnostic": True,
    }
