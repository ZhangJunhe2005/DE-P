#!/usr/bin/env python3
"""Self-contained Safety Evaluator V2.1 validation without pytest."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import tempfile
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _assert(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        default="reports/phase8jqv2_5_static_checker_validation.json",
    )
    args = parser.parse_args()
    v2_hashes_before = {
        path: _sha256(path)
        for path in ("policy/safety_evaluator_v2.py", "loss/safety_geometry_v2.py")
    }
    with tempfile.TemporaryDirectory(prefix="phase8jqv2_5_") as directory:
        temporary = Path(directory)
        root = temporary / "authority"
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
        backend = ExactAuthorityBVH(root)
        cases = {
            "tangent": certify_bezier_authority(
                backend, line_bezier_control_points(
                    [-0.5, -0.3, 0.05], [0.5, -0.3, 0.05]
                )
            ),
            "penetrating": certify_bezier_authority(
                backend, line_bezier_control_points(
                    [-0.5, 0.05, 0.05], [0.5, 0.05, 0.05]
                )
            ),
            "safe": certify_bezier_authority(
                backend, line_bezier_control_points(
                    [-0.5, 1.0, 0.05], [0.5, 1.0, 0.05]
                )
            ),
            "unknown": certify_bezier_authority(
                backend, line_bezier_control_points(
                    [-1.0, 0.41, 0.05], [1.0, 0.41, 0.05]
                ), max_depth=0,
            ),
            "oob": certify_bezier_authority(
                backend, line_bezier_control_points(
                    [-1.9, 1.0, 0.0], [-1.8, 1.0, 0.0]
                )
            ),
        }
        expected = {
            "tangent": CertificateState.CONFIRMED_COLLISION,
            "penetrating": CertificateState.CONFIRMED_COLLISION,
            "safe": CertificateState.CERTIFIED_SAFE,
            "unknown": CertificateState.UNKNOWN,
            "oob": CertificateState.CONFIRMED_COLLISION,
        }
        for name, state in expected.items():
            _assert(cases[name].state is state, f"{name}: {cases[name].state}")
        _assert(cases["oob"].out_of_bounds, "OOB must fail closed")

        start = np.zeros((3, 3), dtype=np.float64)
        end = np.zeros((3, 3), dtype=np.float64)
        start[:, 0] = [-1.0, 1.0, 0.05]
        end[:, 0] = [1.0, 1.0, 0.05]
        quintic = certify_bezier_authority(
            backend, quintic_bezier_control_points(start, end, 1.7)
        )
        _assert(
            quintic.state is CertificateState.CERTIFIED_SAFE,
            "safe quintic was not certified",
        )
        latency_state = start.copy()
        latency_state[:, 1] = [1.0, 0.0, 0.0]
        latency_state[:, 2] = [2.0, 0.0, 0.0]
        prefix = quadratic_bezier_control_points(latency_state, 0.2)
        expected_end = (
            latency_state[:, 0] + 0.2 * latency_state[:, 1]
            + 0.5 * 0.2**2 * latency_state[:, 2]
        )
        _assert(
            np.allclose(prefix[-1], expected_end, atol=1e-12),
            "latency-prefix endpoint mismatch",
        )

        controller = temporary / "controller.json"
        controller.write_text(json.dumps({"first_controllable_time_s": 0.1}))
        evaluator = SafetyEvaluatorV2_1(SafetyEvaluatorV2_1Config.load(
            "configs/safety_evaluator_v2_1.yaml", controller
        ))
        _assert(
            evaluator.evaluate_dynamic.__func__ is SafetyEvaluatorV2.evaluate_dynamic,
            "dynamic implementation was not inherited unchanged",
        )
        no_target = evaluator.evaluate_dynamic(
            {"positions": np.zeros((3, 3))}, [[], [], []], "valid_gt"
        )
        _assert(
            np.isposinf(no_target["continuous_physical_min"]),
            "no-target dynamic risk must be zero/infinite clearance",
        )
        labels = evaluator.three_state_safety_first_label(
            [
                CertificateState.CERTIFIED_SAFE,
                CertificateState.UNKNOWN,
                CertificateState.CONFIRMED_COLLISION,
            ],
            [1.0, 1.0, -0.1],
            [100.0, 0.0, -100.0],
        )
        _assert(labels[0] < labels[1] < labels[2], "three-state order violated")
        covariance = np.eye(3) * 0.25
        _assert(
            evaluator.uncertainty_margin(
                [1, 0, 0], [0, 0, 0], covariance, "valid_gt"
            ) == 0.0,
            "GT uncertainty must be zero",
        )
        _assert(
            abs(evaluator.uncertainty_margin(
                [1, 0, 0], [0, 0, 0], covariance, "valid_estimated"
            ) - 1.0) < 1e-12,
            "estimated directional covariance must be applied once",
        )

    v2_hashes_after = {
        path: _sha256(path) for path in v2_hashes_before
    }
    _assert(v2_hashes_before == v2_hashes_after, "V2 source was modified")
    result = {
        "status": "PASS",
        "evaluator_version": "safety_evaluator_v2_1",
        "static_authority": "static_geometry_authority_v1",
        "synthetic_cases": {
            name: {
                "state": value.state.value,
                "minimum_sampled_gap_m": value.minimum_sampled_gap_m,
                "maximum_depth": value.maximum_depth,
                "queried_sample_count": value.queried_sample_count,
            }
            for name, value in cases.items()
        },
        "continuous_line": "PASS",
        "continuous_quintic": "PASS",
        "latency_prefix": "PASS",
        "tangent_is_collision": True,
        "oob_fail_closed": True,
        "three_state_order": "PASS",
        "no_target_dynamic_risk_zero": True,
        "gt_uncertainty_zero": True,
        "estimated_covariance_applied_once": True,
        "dynamic_method_owner":
            evaluator.evaluate_dynamic.__func__.__qualname__,
        "dynamic_method_sha256": hashlib.sha256(
            inspect.getsource(SafetyEvaluatorV2.evaluate_dynamic).encode()
        ).hexdigest(),
        "v2_source_hashes_unchanged": v2_hashes_after,
        "production_test_used": False,
        "blind_used": False,
        "optimizer_step_executed": False,
    }
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2) + "\n")
    dynamic_report = {
        "status": "PASS",
        "v2_1_dynamic_implementation_owner":
            evaluator.evaluate_dynamic.__func__.__qualname__,
        "v2_1_uncertainty_implementation_owner":
            evaluator.uncertainty_margin.__func__.__qualname__,
        "v2_source_hashes_unchanged": v2_hashes_after,
        "no_target_dynamic_risk_zero": True,
        "gt_uncertainty_zero": True,
        "estimated_covariance_applied_once": True,
        "timeline_implementation_owner":
            evaluator.timeline.__func__.__qualname__,
        "production_test_used": False,
        "blind_used": False,
    }
    (report.parent / "phase8jqv2_5_dynamic_regression.json").write_text(
        json.dumps(dynamic_report, indent=2) + "\n"
    )
    label_report = {
        "status": "PASS",
        "order": [
            "certified_safe", "unknown", "confirmed_collision"
        ],
        "observed_fixture_labels": labels.tolist(),
        "ordering_violation_count": 0,
        "unknown_is_not_collision": True,
        "finite_sentinel_used_for_clearance": False,
    }
    (report.parent / "phase8jqv2_5_label_semantics.json").write_text(
        json.dumps(label_report, indent=2) + "\n"
    )
    specification = """# Safety Evaluator V2.1

- Final static truth: `static_geometry_authority_v1` canonical occupancy.
- Exact backend: sphere-to-voxel AABB BVH, UAV radius 0.3 m.
- Contact tolerance: 1e-6 m; tangent is collision.
- Out of bounds: occupied/fail-closed.
- Continuous trajectory: latency-prefix quadratic plus complete controlled
  quintic, certified with Bézier convex-hull/Lipschitz recursion.
- Physical states remain distinct: certified-safe, unknown, collision.
- Dynamic geometry, timeline, and uncertainty methods are inherited directly
  from Safety Evaluator V2.
- Derived geometry/local AABB distance is surrogate-only; every final outcome
  requires the exact continuous verifier.
"""
    (report.parent / "phase8jqv2_5_evaluator_spec.md").write_text(
        specification
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
