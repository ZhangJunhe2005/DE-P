import torch

from policy.static_yopo_parity_v4_5_5 import (
    StaticYOPOParityConfigV455,
    sample_quintic_positions_dense_v4_5_5,
    static_yopo_parity_objective_v4_5_5,
)
from tests.test_static_yopo_parity_v4_5_1 import objective_inputs
from tools.train_mixed_static_yopo_v1 import training_contract_classification


class RecordingPlaneSafety:
    def __init__(self):
        self.query_sizes = []

    def get_distance_cost(self, position, map_id):
        del map_id
        self.query_sizes.append(int(position.shape[1]))
        distance = 4.0 - position[..., 0]
        cost = torch.exp(((1.2 - distance) / 0.6).clamp(-60.0, 60.0))
        return cost, distance


def test_dense_sampler_uses_requested_grid_and_preserves_gradients():
    predicted, _, common = objective_inputs()
    dense = sample_quintic_positions_dense_v4_5_5(
        common["sampler"], common["fixed"], predicted, 81
    )
    assert dense.shape == (15, 81, 3)
    dense.sum().backward()
    assert torch.isfinite(predicted.grad).all()
    assert predicted.grad.abs().sum() > 0.0


def test_v455_replaces_coarse_static_supervision_with_81_point_grid():
    predicted, scores, common = objective_inputs()
    safety = RecordingPlaneSafety()
    common["safety_loss"] = safety
    result = static_yopo_parity_objective_v4_5_5(
        **common, config=StaticYOPOParityConfigV455(enabled=True)
    )
    # Training performs exactly one ESDF query, on the effective dense grid.
    assert safety.query_sizes == [15 * 81]
    assert result["candidate_minimum_clearance"].shape == (15,)
    assert result["candidate_coarse_minimum_clearance"].shape == (15,)
    assert result["per_sample_coarse_false_safe_candidate_count"].shape == (1,)
    assert result["per_sample_dense_clearance_drop"].shape == (1,)
    result["total_loss"].backward()
    assert torch.isfinite(predicted.grad).all()
    assert torch.isfinite(scores.grad).all()
    assert predicted.grad.abs().sum() > 0.0
    assert scores.grad.abs().sum() > 0.0


def test_v455_computes_legacy_grid_gap_only_during_validation():
    predicted, scores, common = objective_inputs()
    safety = RecordingPlaneSafety()
    common["safety_loss"] = safety
    with torch.no_grad():
        result = static_yopo_parity_objective_v4_5_5(
            **common, config=StaticYOPOParityConfigV455(enabled=True)
        )
    # Validation first evaluates the effective 81-point grid, then a detached
    # 30-point legacy diagnostic.  Neither query participates in gradients.
    assert safety.query_sizes == [15 * 81, 15 * 30]
    assert result["per_sample_coarse_false_safe_candidate_count"].shape == (1,)
    assert result["per_sample_dense_clearance_drop"].shape == (1,)


def test_v455_is_one_continuous_objective_without_gate():
    contract = StaticYOPOParityConfigV455(enabled=True).contract()
    assert contract["static_safety_samples"] == 81
    assert contract["qualification_gate"] == "none"
    assert "barrier" not in contract["candidate_objective"]
    semantics = training_contract_classification({
        "validation": {
            "contract_version": "route_a_v4_5_5_dense_static_esdf_v1"
        },
        "experiment_role": "dense_static_esdf_five_epoch_shakedown",
    })
    assert semantics == {
        "three_layer": False,
        "ungated_static": True,
        "diagnostic": True,
    }
