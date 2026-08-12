import torch

from policy.static_yopo_parity_v4_5_2 import (
    StaticYOPOParityConfigV452,
    static_yopo_parity_objective_v4_5_2,
)
from policy.static_yopo_parity_v4_5_3 import (
    StaticYOPOParityConfigV453,
    continuous_clearance_barrier_v4_5_3,
    static_yopo_parity_objective_v4_5_3,
)
from tests.test_static_yopo_parity_v4_5_1 import objective_inputs
from tools.train_mixed_static_yopo_v1 import training_contract_classification


def test_continuous_clearance_barrier_is_smooth_monotonic_and_not_a_gate():
    clearance = torch.tensor(
        [[0.60, 0.40, 0.30, 0.20, 0.00]], requires_grad=True
    )
    cost = continuous_clearance_barrier_v4_5_3(
        clearance,
        vehicle_radius=0.30,
        weight=1.0,
        softness=0.05,
    )
    assert torch.isfinite(cost).all()
    assert bool(torch.all(cost[:, 1:] > cost[:, :-1]))
    assert float(cost[0, 0]) < 0.01
    assert float(cost[0, -1]) > 5.0
    cost.sum().backward()
    assert torch.isfinite(clearance.grad).all()
    assert bool(torch.all(clearance.grad < 0.0))


def test_zero_clearance_weight_is_v452_objective_equivalent():
    _, _, common = objective_inputs()
    v452 = static_yopo_parity_objective_v4_5_2(
        **common, config=StaticYOPOParityConfigV452(enabled=True)
    )
    v453 = static_yopo_parity_objective_v4_5_3(
        **common,
        config=StaticYOPOParityConfigV453(
            enabled=True, clearance_barrier_weight=0.0
        ),
    )
    for name in (
        "total_loss", "trajectory_loss", "score_loss", "ranking_loss",
        "score_label", "candidate_raw_total_cost",
        "per_sample_score_oracle_regret", "per_sample_score_top1_match",
    ):
        torch.testing.assert_close(v453[name], v452[name], rtol=0.0, atol=0.0)


def test_v453_barrier_changes_same_total_label_and_has_finite_gradients():
    predicted, scores, common = objective_inputs()
    result = static_yopo_parity_objective_v4_5_3(
        **common, config=StaticYOPOParityConfigV453(enabled=True)
    )
    barrier = result["candidate_clearance_barrier_cost"]
    assert barrier.shape == (15,)
    assert torch.isfinite(barrier).all()
    assert float(result["clearance_barrier_loss"]) > 0.0
    assert result["score_label"].shape == (15,)
    assert result["per_sample_clearance_barrier_loss"].shape == (1,)
    result["total_loss"].backward()
    assert torch.isfinite(predicted.grad).all()
    assert torch.isfinite(scores.grad).all()
    assert predicted.grad.abs().sum() > 0.0
    assert scores.grad.abs().sum() > 0.0


def test_v453_contract_is_ungated_two_stage_diagnostic():
    config = StaticYOPOParityConfigV453(enabled=True)
    assert config.contract()["qualification_gate"] == "none"
    semantics = training_contract_classification({
        "validation": {
            "contract_version": (
                "route_a_v4_5_3_continuous_clearance_two_stage_v1"
            )
        },
        "experiment_role": "mixed_two_stage_eight_epoch_shakedown",
    })
    assert semantics == {
        "three_layer": False,
        "ungated_static": True,
        "diagnostic": True,
    }
