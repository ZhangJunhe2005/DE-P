from __future__ import annotations

import torch

import policy.static_yopo_recovery_coverage_v4_8 as v48


class ConstantClearanceSafety:
    def __init__(self, clearance):
        self.clearance = float(clearance)

    def get_distance_cost(self, points, map_id):
        shape = points.shape[:2]
        distance = torch.full(
            shape, self.clearance, device=points.device, dtype=points.dtype,
        )
        return torch.zeros_like(distance), distance


class AuthorityOutOfBoundsSafety(ConstantClearanceSafety):
    out_of_bounds_cost = 10.0

    def get_distance_cost(self, points, map_id):
        shape = points.shape[:2]
        distance = torch.zeros(
            shape, device=points.device, dtype=points.dtype,
        )
        return torch.full_like(distance, self.out_of_bounds_cost), distance


def inputs(endpoint_length, *, recovery=True, requires_grad=True):
    batch_size, candidate_count = 2, 15
    predicted = torch.zeros(
        batch_size * candidate_count, 3, 3,
        dtype=torch.float32, requires_grad=requires_grad,
    )
    values = predicted.detach().clone()
    directions = v48._network_order_lattice_directions(
        device=values.device,
        dtype=values.dtype,
        candidate_count=candidate_count,
    ).repeat(batch_size, 1)
    values[:, :, 0] = directions * float(endpoint_length)
    predicted = values.requires_grad_(requires_grad)
    return {
        "current_position": torch.zeros(batch_size, 3),
        "rotation_world_from_body": torch.eye(3).repeat(batch_size, 1, 1),
        "predicted": predicted,
        "map_id": torch.zeros(batch_size, dtype=torch.long),
        "recovery_state": torch.full(
            (batch_size,), bool(recovery), dtype=torch.bool,
        ),
        "batch_size": batch_size,
        "candidate_count": candidate_count,
    }


def config(**changes):
    values = {
        "enabled": True,
        "safe_sector_weight": 0.05,
        "safe_sector_ray_start_m": 0.5,
        "safe_sector_ray_end_m": 4.0,
        "safe_sector_ray_samples": 12,
        "safe_sector_openness_center_m": 0.45,
        "safe_sector_openness_softness_m": 0.10,
        "safe_sector_target_length_m": 3.0,
        "safe_sector_shortfall_softness_m": 0.35,
    }
    values.update(changes)
    return v48.StaticYOPORecoveryCoverageConfigV48.from_mapping(values)


def test_open_recovery_sector_penalizes_short_more_than_long_candidates():
    cfg = config()
    short = v48.recovery_safe_sector_terms_v4_8(
        safety_loss=ConstantClearanceSafety(2.0),
        config=cfg, **inputs(0.5),
    )
    long = v48.recovery_safe_sector_terms_v4_8(
        safety_loss=ConstantClearanceSafety(2.0),
        config=cfg, **inputs(4.0),
    )
    assert float(short["per_sample_loss"].mean()) > 0.05
    assert float(long["per_sample_loss"].mean()) < 0.001
    assert torch.all(short["openness"] > 0.99)


def test_blocked_sector_and_non_recovery_samples_receive_negligible_pull():
    cfg = config()
    blocked = v48.recovery_safe_sector_terms_v4_8(
        safety_loss=ConstantClearanceSafety(-1.0),
        config=cfg, **inputs(0.5),
    )
    ordinary = v48.recovery_safe_sector_terms_v4_8(
        safety_loss=ConstantClearanceSafety(2.0),
        config=cfg, **inputs(0.5, recovery=False),
    )
    assert float(blocked["per_sample_loss"].amax()) < 1.0e-7
    torch.testing.assert_close(
        ordinary["per_sample_loss"],
        torch.zeros_like(ordinary["per_sample_loss"]),
    )


def test_safe_sector_loss_has_finite_endpoint_gradient_only():
    values = inputs(0.5)
    predicted = values["predicted"]
    terms = v48.recovery_safe_sector_terms_v4_8(
        safety_loss=ConstantClearanceSafety(2.0),
        config=config(), **values,
    )
    terms["per_sample_loss"].mean().backward()
    assert predicted.grad is not None
    assert bool(torch.isfinite(predicted.grad).all())
    assert float(predicted.grad.abs().sum()) > 0.0


def test_collapsed_endpoint_still_has_nonzero_recovery_gradient():
    values = inputs(0.0)
    predicted = values["predicted"]
    terms = v48.recovery_safe_sector_terms_v4_8(
        safety_loss=ConstantClearanceSafety(2.0),
        config=config(), **values,
    )
    terms["per_sample_loss"].mean().backward()
    assert predicted.grad is not None
    assert bool(torch.isfinite(predicted.grad).all())
    assert float(predicted.grad.abs().sum()) > 0.0


def test_authority_out_of_bounds_ray_is_not_rewarded_as_open():
    terms = v48.recovery_safe_sector_terms_v4_8(
        safety_loss=AuthorityOutOfBoundsSafety(0.0),
        config=config(), **inputs(0.0),
    )
    torch.testing.assert_close(
        terms["openness"], torch.zeros_like(terms["openness"]),
    )
    torch.testing.assert_close(
        terms["per_sample_loss"],
        torch.zeros_like(terms["per_sample_loss"]),
    )


def test_wrapper_preserves_v4510_score_supervision(monkeypatch):
    batch_size, candidate_count = 2, 15
    predicted_scores = torch.randn(
        batch_size, candidate_count, requires_grad=True,
    )
    score_label = torch.linspace(-1.0, 1.0, batch_size * candidate_count)
    score_target = torch.linspace(0.0, 2.0, batch_size * candidate_count)
    raw_total = torch.linspace(1.0, 3.0, batch_size * candidate_count)
    base_total = predicted_scores.sum(dim=1) * 0.0 + 2.0
    base_trajectory = predicted_scores.sum(dim=1) * 0.0 + 1.0

    def fake_v4510(**kwargs):
        return {
            "total_loss": base_total.mean(),
            "trajectory_loss": base_trajectory.mean(),
            "score_loss": predicted_scores.sum() * 0.0 + 0.25,
            "ranking_loss": predicted_scores.sum() * 0.0 + 0.5,
            "score_label": score_label.clone(),
            "candidate_score_target_cost": score_target.clone(),
            "candidate_raw_total_cost": raw_total.clone(),
            "candidate_proposal_total_cost": raw_total.clone(),
            "per_sample_total_loss": base_total.clone(),
            "per_sample_trajectory_loss": base_trajectory.clone(),
        }

    monkeypatch.setattr(
        v48, "static_yopo_parity_objective_v4_5_10", fake_v4510,
    )
    fixed = torch.zeros(batch_size * candidate_count, 3, 3)
    predicted = torch.zeros(batch_size * candidate_count, 3, 3)
    predicted[:, 0, 0] = 0.5
    result = v48.static_yopo_recovery_coverage_objective_v4_8(
        sampler=None,
        safety_loss=ConstantClearanceSafety(2.0),
        fixed=fixed,
        predicted=predicted,
        predicted_scores=predicted_scores,
        goal_world=torch.zeros(batch_size * candidate_count, 3),
        map_id=torch.zeros(batch_size, dtype=torch.long),
        batch_size=batch_size,
        candidate_count=candidate_count,
        route_goal_distance=torch.full((batch_size,), 10.0),
        recovery_state=torch.tensor([True, False]),
        rotation_world_from_body=torch.eye(3).repeat(batch_size, 1, 1),
        config=config(),
    )
    torch.testing.assert_close(result["score_label"], score_label)
    torch.testing.assert_close(
        result["score_loss"], predicted_scores.sum() * 0.0 + 0.25,
    )
    torch.testing.assert_close(
        result["ranking_loss"], predicted_scores.sum() * 0.0 + 0.5,
    )
    torch.testing.assert_close(
        result["candidate_score_target_cost"], score_target,
    )
    torch.testing.assert_close(result["candidate_raw_total_cost"], raw_total)
    torch.testing.assert_close(
        result["candidate_proposal_total_cost"],
        raw_total + result["candidate_safe_sector_coverage_cost"],
    )
    assert float(result["safe_sector_coverage_loss"]) > 0.0
    assert float(result["per_sample_safe_sector_coverage_loss"][1]) == 0.0
    assert float(result["recovery_sample_fraction"]) == 0.5
    assert float(result["recovery_open_sector_soft_count"]) > 14.9
    assert float(result["recovery_mean_endpoint_distance"]) == 0.5
    assert float(result["recovery_max_endpoint_distance"]) == 0.5
    result["total_loss"].backward()
    torch.testing.assert_close(
        predicted_scores.grad, torch.zeros_like(predicted_scores),
    )


def test_config_rejects_invalid_ray_or_softness_contract():
    for changes in (
        {"safe_sector_ray_end_m": 0.4},
        {"safe_sector_ray_samples": 1},
        {"safe_sector_openness_softness_m": 0.0},
        {"safe_sector_weight": -0.1},
    ):
        try:
            config(**changes)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid config accepted: {changes}")
