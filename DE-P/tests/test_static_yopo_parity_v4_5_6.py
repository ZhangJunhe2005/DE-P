import torch

from policy.static_yopo_parity_v4_5_5 import (
    StaticYOPOParityConfigV455,
    static_yopo_parity_objective_v4_5_5,
)
from policy.static_yopo_parity_v4_5_6 import (
    StaticYOPOParityConfigV456,
    continuous_clearance_pairwise_ranking_v4_5_6,
    static_yopo_parity_objective_v4_5_6,
)
from tests.test_static_yopo_parity_v4_5_1 import objective_inputs
from tools.train_mixed_static_yopo_v1 import training_contract_classification


def test_continuous_clearance_order_prefers_lower_risk_scores():
    clearance = torch.tensor([[1.20, 0.45, 0.15]])
    correct = torch.tensor([[-1.0, 0.0, 1.0]], requires_grad=True)
    reversed_score = torch.tensor([[1.0, 0.0, -1.0]])
    correct_loss = continuous_clearance_pairwise_ranking_v4_5_6(
        correct, clearance, vehicle_radius=0.30, softness=0.08, margin=0.50
    )
    reversed_loss = continuous_clearance_pairwise_ranking_v4_5_6(
        reversed_score, clearance,
        vehicle_radius=0.30, softness=0.08, margin=0.50,
    )
    assert float(correct_loss) < float(reversed_loss)
    correct_loss.sum().backward()
    assert torch.isfinite(correct.grad).all()


def _objective(config):
    predicted, scores, common = objective_inputs()
    result = static_yopo_parity_objective_v4_5_6(
        **common, config=config
    )
    return predicted, scores, common, result


def test_zero_safety_score_terms_are_v455_equivalent():
    predicted, scores, common = objective_inputs()
    v455 = static_yopo_parity_objective_v4_5_5(
        **common, config=StaticYOPOParityConfigV455(enabled=True)
    )
    v456 = static_yopo_parity_objective_v4_5_6(
        **common,
        config=StaticYOPOParityConfigV456(
            enabled=True,
            clearance_score_weight=0.0,
            clearance_pairwise_weight=0.0,
        ),
    )
    for name in (
        "total_loss", "trajectory_loss", "score_loss", "ranking_loss",
        "score_label", "candidate_raw_total_cost",
        "per_sample_score_oracle_regret", "per_sample_score_top1_match",
    ):
        torch.testing.assert_close(v456[name], v455[name], rtol=0.0, atol=0.0)


def test_safety_score_changes_score_not_proposal_gradient():
    predicted0, scores0, common0 = objective_inputs()
    base = static_yopo_parity_objective_v4_5_5(
        **common0, config=StaticYOPOParityConfigV455(enabled=True)
    )
    base["total_loss"].backward()
    proposal_gradient = predicted0.grad.detach().clone()
    score_gradient = scores0.grad.detach().clone()

    predicted, scores, _, result = _objective(
        StaticYOPOParityConfigV456(enabled=True)
    )
    result["total_loss"].backward()
    torch.testing.assert_close(
        predicted.grad, proposal_gradient, rtol=0.0, atol=0.0
    )
    assert not torch.equal(scores.grad, score_gradient)
    assert torch.isfinite(scores.grad).all()
    assert result["clearance_pairwise_ranking_loss"] > 0
    torch.testing.assert_close(
        result["candidate_raw_total_cost"],
        result["candidate_proposal_total_cost"],
    )


def test_v456_contract_is_one_training_loss_and_no_gate():
    contract = StaticYOPOParityConfigV456(enabled=True).contract()
    assert contract["qualification_gate"] == "none"
    assert contract["candidate_rejection"] == "none"
    semantics = training_contract_classification({
        "validation": {
            "contract_version": "route_a_v4_5_6_continuous_score_safety_v1"
        },
        "experiment_role": "continuous_safety_score_calibration",
    })
    assert semantics == {
        "three_layer": False,
        "ungated_static": True,
        "diagnostic": True,
    }
