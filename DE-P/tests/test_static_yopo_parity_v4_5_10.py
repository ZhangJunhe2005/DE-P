import pytest
import torch

from policy.static_yopo_parity_v4_5_10 import (
    StaticYOPOParityConfigV4510,
    tail_aware_static_cost_v4_5_10,
)


def test_v4510_tail_aggregation_is_bounded_and_mean_dominant():
    config = StaticYOPOParityConfigV4510(
        enabled=True, static_safety_samples=81,
    )
    point_cost = torch.zeros((1, 2, 81), dtype=torch.float64)
    point_cost[0, 0] = 1.0
    point_cost[0, 1, 0] = 10.0
    combined, mean_cost, worst_cost = tail_aware_static_cost_v4_5_10(
        point_cost, config=config,
    )
    assert combined[0, 0].item() == pytest.approx(1.0)
    assert mean_cost[0, 1].item() == pytest.approx(10.0 / 81.0)
    assert worst_cost[0, 1].item() == pytest.approx(2.0)
    assert combined[0, 1].item() == pytest.approx(
        0.9 * 10.0 / 81.0 + 0.1 * 2.0
    )
    # A single dangerous point becomes visible, but cannot price all 81
    # samples as dangerous (the V4.5.3-style collapse mechanism).
    assert mean_cost[0, 1] < combined[0, 1] < point_cost[0, 1, 0]


def test_v4510_every_sample_retains_gradient_from_time_mean():
    config = StaticYOPOParityConfigV4510(
        enabled=True, static_safety_samples=10, worst_sample_count=5,
        dangerous_segment_window=10,
    )
    point_cost = torch.arange(
        1.0, 11.0, dtype=torch.float64,
    ).reshape(1, 1, 10).requires_grad_()
    combined, _, _ = tail_aware_static_cost_v4_5_10(
        point_cost, config=config,
    )
    combined.sum().backward()
    assert torch.isfinite(point_cost.grad).all()
    assert torch.all(point_cost.grad > 0.0)
    assert point_cost.grad[0, 0, -1] > point_cost.grad[0, 0, 0]


def test_v4510_contract_has_one_continuous_safety_term_and_no_gate():
    config = StaticYOPOParityConfigV4510(enabled=True)
    config.validate()
    contract = config.contract()
    assert contract["qualification_gate"] == "none"
    assert contract["time_mean_weight"] == pytest.approx(0.90)
    assert contract["worst_sample_weight"] == pytest.approx(0.10)
    assert contract["worst_sample_count"] == 5
    assert contract["static_clearance_semantics"]["trajectory_aggregation"] \
        == "90_percent_time_mean_plus_10_percent_worst_5_mean"


def test_v4510_rejects_unbounded_tail_weight_or_count():
    with pytest.raises(ValueError, match="sum to one"):
        StaticYOPOParityConfigV4510(
            enabled=True, time_mean_weight=0.8, worst_sample_weight=0.1,
        ).validate()
    with pytest.raises(ValueError, match="fit static safety samples"):
        StaticYOPOParityConfigV4510(
            enabled=True, static_safety_samples=4, worst_sample_count=5,
            dangerous_segment_window=4,
        ).validate()
