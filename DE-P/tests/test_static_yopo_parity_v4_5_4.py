import torch

from policy.static_yopo_parity_v4_5_2 import (
    StaticYOPOParityConfigV452,
    static_yopo_parity_objective_v4_5_2,
)
from policy.static_yopo_parity_v4_5_4 import (
    StaticYOPOParityConfigV454,
    static_yopo_parity_objective_v4_5_4,
)
from tests.test_static_yopo_parity_v4_5_1 import objective_inputs
from tools.train_mixed_static_yopo_v1 import training_contract_classification


def run_objective(weight):
    predicted, scores, common = objective_inputs()
    result = static_yopo_parity_objective_v4_5_4(
        **common,
        config=StaticYOPOParityConfigV454(
            enabled=True, clearance_barrier_weight=weight
        ),
    )
    return predicted, scores, result


def test_zero_barrier_is_exactly_v452_equivalent():
    _, _, common = objective_inputs()
    v452 = static_yopo_parity_objective_v4_5_2(
        **common, config=StaticYOPOParityConfigV452(enabled=True)
    )
    v454 = static_yopo_parity_objective_v4_5_4(
        **common,
        config=StaticYOPOParityConfigV454(
            enabled=True, clearance_barrier_weight=0.0
        ),
    )
    for name in (
        "total_loss", "trajectory_loss", "score_loss", "ranking_loss",
        "score_label", "candidate_raw_total_cost",
        "per_sample_score_oracle_regret", "per_sample_score_top1_match",
    ):
        torch.testing.assert_close(v454[name], v452[name], rtol=0.0, atol=0.0)


def test_barrier_changes_score_gradient_but_not_proposal_gradient():
    predicted_zero, scores_zero, zero = run_objective(0.0)
    zero["total_loss"].backward()
    proposal_gradient_zero = predicted_zero.grad.detach().clone()
    score_gradient_zero = scores_zero.grad.detach().clone()

    predicted_barrier, scores_barrier, barrier = run_objective(1.0)
    barrier["total_loss"].backward()
    torch.testing.assert_close(
        predicted_barrier.grad, proposal_gradient_zero, rtol=0.0, atol=0.0
    )
    assert not torch.equal(scores_barrier.grad, score_gradient_zero)
    torch.testing.assert_close(
        barrier["candidate_raw_total_cost"],
        barrier["candidate_proposal_total_cost"],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        barrier["candidate_score_target_cost"],
        barrier["candidate_proposal_total_cost"]
        + barrier["candidate_clearance_barrier_cost"],
    )


def test_v454_contract_is_independent_score_only_and_ungated():
    contract = StaticYOPOParityConfigV454(enabled=True).contract()
    assert contract["candidate_objective"] == "unchanged_v4_5_2_total_cost"
    assert contract["qualification_gate"] == "none"
    semantics = training_contract_classification({
        "validation": {
            "contract_version": "route_a_v4_5_4_independent_score_only_v1"
        },
        "experiment_role": "independent_score_five_epoch_shakedown",
    })
    assert semantics == {
        "three_layer": False,
        "ungated_static": True,
        "diagnostic": True,
    }
