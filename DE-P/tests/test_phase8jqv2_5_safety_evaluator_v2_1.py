import json
from pathlib import Path
import uuid

import numpy as np
import pytest

from authoritative_dataset.exact_backend_v1 import ExactAuthorityBVH
from geometry_authority.static_v1 import build_authority_artifact
from loss.static_continuous_authority_v1 import (
    CertificateState,
    certify_bezier_authority,
    line_bezier_control_points,
    quadratic_bezier_control_points,
    quintic_bezier_control_points,
)
from policy.safety_evaluator_v2 import SafetyEvaluatorV2
from policy.safety_evaluator_v2_1 import (
    SafetyEvaluatorV2_1,
    SafetyEvaluatorV2_1Config,
)


@pytest.fixture()
def authority(tmp_path):
    root = tmp_path / "authority"
    build_authority_artifact(
        root,
        np.asarray([[0.01, 0.01, 0.01]], dtype=np.float32),
        map_uuid=str(uuid.UUID(int=1)),
        generator_seed=1,
        map_id="synthetic_contact",
        generator_source_hash="fixture-source",
        generator_config_hash="fixture-config",
        parent_git_commit="fixture",
        bounds_min=[-2.0, -2.0, -2.0],
        bounds_max=[2.0, 2.0, 2.0],
    )
    return ExactAuthorityBVH(root)


def test_exact_contact_safe_collision_unknown_and_oob(authority):
    tangent = certify_bezier_authority(
        authority,
        line_bezier_control_points([-0.5, -0.3, 0.05], [0.5, -0.3, 0.05]),
    )
    collision = certify_bezier_authority(
        authority,
        line_bezier_control_points([-0.5, 0.05, 0.05], [0.5, 0.05, 0.05]),
    )
    safe = certify_bezier_authority(
        authority,
        line_bezier_control_points([-0.5, 1.0, 0.05], [0.5, 1.0, 0.05]),
    )
    unknown = certify_bezier_authority(
        authority,
        line_bezier_control_points([-1.0, 0.41, 0.05], [1.0, 0.41, 0.05]),
        max_depth=0,
    )
    oob = certify_bezier_authority(
        authority,
        line_bezier_control_points([-1.9, 1.0, 0.0], [-1.8, 1.0, 0.0]),
    )
    assert tangent.state is CertificateState.CONFIRMED_COLLISION
    assert collision.state is CertificateState.CONFIRMED_COLLISION
    assert safe.state is CertificateState.CERTIFIED_SAFE
    assert unknown.state is CertificateState.UNKNOWN
    assert unknown.unknown_reason == "recursion_budget_exhausted"
    assert oob.state is CertificateState.CONFIRMED_COLLISION
    assert oob.out_of_bounds


def test_quintic_and_latency_prefix_complete_curve(authority):
    start = np.zeros((3, 3), dtype=np.float64)
    end = np.zeros((3, 3), dtype=np.float64)
    start[:, 0] = [-1.0, 1.0, 0.05]
    end[:, 0] = [1.0, 1.0, 0.05]
    quintic = quintic_bezier_control_points(start, end, 1.7)
    certificate = certify_bezier_authority(authority, quintic)
    assert certificate.state is CertificateState.CERTIFIED_SAFE

    latency_state = start.copy()
    latency_state[:, 1] = [1.0, 0.0, 0.0]
    latency_state[:, 2] = [2.0, 0.0, 0.0]
    control = quadratic_bezier_control_points(latency_state, 0.2)
    expected = (
        latency_state[:, 0] + 0.2 * latency_state[:, 1]
        + 0.5 * 0.2**2 * latency_state[:, 2]
    )
    np.testing.assert_allclose(control[-1], expected, atol=1e-12)
    assert certify_bezier_authority(
        authority, control
    ).state is CertificateState.CERTIFIED_SAFE


def test_frozen_config_and_dynamic_method_inheritance(tmp_path):
    controller = tmp_path / "controller.json"
    controller.write_text(json.dumps({"first_controllable_time_s": 0.1}))
    config = SafetyEvaluatorV2_1Config.load(
        "configs/safety_evaluator_v2_1.yaml", controller
    )
    evaluator = SafetyEvaluatorV2_1(config)
    assert evaluator.evaluate_dynamic.__func__ is SafetyEvaluatorV2.evaluate_dynamic
    assert evaluator.uncertainty_margin.__func__ is SafetyEvaluatorV2.uncertainty_margin
    assert evaluator.canonical_spec()["oob_policy"] == "occupied_fail_closed"


def test_dynamic_no_target_and_three_state_ordering(tmp_path):
    controller = tmp_path / "controller.json"
    controller.write_text(json.dumps({"first_controllable_time_s": 0.1}))
    evaluator = SafetyEvaluatorV2_1(SafetyEvaluatorV2_1Config.load(
        "configs/safety_evaluator_v2_1.yaml", controller
    ))
    timeline = {"positions": np.zeros((3, 3), dtype=np.float64)}
    result = evaluator.evaluate_dynamic(timeline, [[], [], []], "valid_gt")
    assert np.isposinf(result["continuous_physical_min"])
    assert np.isposinf(result["planning_min"])

    labels = evaluator.three_state_safety_first_label(
        [
            CertificateState.CERTIFIED_SAFE,
            CertificateState.UNKNOWN,
            CertificateState.CONFIRMED_COLLISION,
        ],
        [1.0, 1.0, -0.1],
        [100.0, 0.0, -100.0],
    )
    assert labels[0] < labels[1] < labels[2]


def test_gt_uncertainty_zero_estimated_once(tmp_path):
    controller = tmp_path / "controller.json"
    controller.write_text(json.dumps({"first_controllable_time_s": 0.1}))
    evaluator = SafetyEvaluatorV2_1(SafetyEvaluatorV2_1Config.load(
        "configs/safety_evaluator_v2_1.yaml", controller
    ))
    covariance = np.eye(3) * 0.25
    assert evaluator.uncertainty_margin(
        [1, 0, 0], [0, 0, 0], covariance, "valid_gt"
    ) == 0.0
    assert evaluator.uncertainty_margin(
        [1, 0, 0], [0, 0, 0], covariance, "valid_estimated"
    ) == pytest.approx(1.0)

