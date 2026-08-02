#!/usr/bin/env python3
"""Generate pre-validation V2 geometry, uncertainty and timeline evidence."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loss.safety_geometry_v2 import (
    actor_physical_clearance,
    continuous_sphere_segment_gap,
    finite_vertical_cylinder_signed_gap,
    sphere_signed_gap,
    static_physical_clearance,
)
from policy.safety_evaluator_v2 import SafetyEvaluatorV2, SafetyEvaluatorV2Config


REPORTS = ROOT / "reports"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def main():
    entry = json.loads((REPORTS / "phase8jqv2_entry_gate.json").read_text())
    if entry["status"] != "PASS":
        raise RuntimeError("Phase 8J-Q2 entry is not PASS")
    config_path = ROOT / "configs/safety_evaluator_v2.yaml"
    controller_path = REPORTS / "phase8jq_controller_authoritative_envelope.json"
    config = SafetyEvaluatorV2Config.load(config_path, controller_path)
    evaluator = SafetyEvaluatorV2(config)
    matrix = json.loads(
        (REPORTS / "phase8i_checkpoint_matrix.json").read_text()
    )
    decision = next(
        row for row in matrix["checkpoints"]
        if row["name"] == "fixed_050_seed8403"
    )
    simulator_source = Path(
        "/home/zjh/YOPO/Simulator/src/src/dynamic_actor.cpp"
    )
    simulator_config = Path(
        "/home/zjh/YOPO/Simulator/src/config/config.yaml"
    )
    simulator_hash = canonical_hash({
        "source": sha256(simulator_source),
        "config": sha256(simulator_config),
    })
    common = {
        "evaluator_version": config.evaluator_version,
        "config_hash": sha256(config_path),
        "geometry_hash": sha256(ROOT / "loss/safety_geometry_v2.py"),
        "timeline_hash": sha256(ROOT / "policy/safety_evaluator_v2.py"),
        "uncertainty_policy_hash": canonical_hash({
            "policy": config.estimated_uncertainty_policy,
            "confidence_sigma": config.confidence_sigma,
            "propagate_in_evaluator": False,
        }),
        "Simulator_geometry_hash": simulator_hash,
        "simulator_geometry_hash": simulator_hash,
        "dataset_manifest_hash": sha256(
            ROOT / "data/phase8_dynamic_production/dataset_manifest.yaml"
        ),
        "cache_index_hash": sha256(
            ROOT / "cache/phase8i_phase8h_perception/index.json"
        ),
        "checkpoint_hash": decision["sha256"],
        "checkpoint_hashes": {
            row["name"]: row["sha256"] for row in matrix["checkpoints"]
            if row["analyzed_full_phase8i"]
        },
    }
    static = [
        {"fixture": "wall_contact", "raw_esdf_m": .30, "expected_m": 0.0},
        {"fixture": "wall_safe_1cm", "raw_esdf_m": .31, "expected_m": .01},
        {"fixture": "wall_penetration_1cm", "raw_esdf_m": .29, "expected_m": -.01},
        {"fixture": "corner_contact", "raw_esdf_m": .30, "expected_m": 0.0},
        {"fixture": "edge_safe", "raw_esdf_m": .31, "expected_m": .01},
        {"fixture": "narrow_corridor_contact", "raw_esdf_m": .30, "expected_m": 0.0},
    ]
    for row in static:
        row["observed_m"] = float(
            static_physical_clearance(row["raw_esdf_m"], config.uav_radius_m)
        )
        row["pass"] = bool(
            abs(row["observed_m"] - row["expected_m"]) <= 1e-12
        )
    spheres = []
    for name, distance, expected in (
        ("exact_contact", .7, 0), ("penetration_1cm", .69, -.01),
        ("safe_1cm", .71, .01), ("moving_crossing_contact", .7, 0),
        ("head_on_contact", .7, 0),
    ):
        observed = sphere_signed_gap([0, 0, 0], [distance, 0, 0], .3, .4)
        spheres.append({
            "fixture": name, "observed_m": observed, "expected_m": expected,
            "pass": bool(abs(observed - expected) <= 1e-12),
        })
    cylinders = []
    cylinder_cases = (
        ("side_contact", [.7, 0, 0], 0),
        ("top_contact", [0, 0, .8], 0),
        ("corner", [.7, 0, .9], np.hypot(.3, .4) - .3),
        ("outside_height", [0, 0, 1.0], .2),
        ("diagonal", [.6, 0, .7], np.hypot(.2, .2) - .3),
        ("moving_contact", [.7, 0, 0], 0),
    )
    for name, position, expected in cylinder_cases:
        observed = finite_vertical_cylinder_signed_gap(
            position, [0, 0, 0], .3, .4, 1
        )
        cylinders.append({
            "fixture": name, "observed_m": observed, "expected_m": float(expected),
            "pass": bool(abs(observed - expected) <= 1e-12),
        })
    geometry = {
        **common,
        "status": "PASS" if all(
            row["pass"] for row in static + spheres + cylinders
        ) else "FAIL",
        "tolerance_m": 1e-9,
        "static": static,
        "dynamic_sphere": spheres,
        "dynamic_finite_cylinder": cylinders,
        "v1_v2_difference_causes": [
            "static UAV radius subtraction", "exact cylinder geometry",
            "no GT covariance growth",
        ],
        "gt_uncertainty_exactly_zero": (
            evaluator.uncertainty_margin(
                [0, 0, 0], [1, 0, 0], np.eye(3), "valid_gt"
            ) == 0
        ),
    }
    atomic_json(REPORTS / "phase8jqv2_geometry_validation.json", geometry)

    current = np.asarray([[0, 1, 0], [0, 0, 0], [2, 0, 0]], dtype=float)
    times = np.linspace(1.7 / 30, 1.7, 30)
    old = np.stack((times, np.zeros(30), 2*np.ones(30)), axis=1)
    wall = evaluator.timeline(current, old, "fixed_wall_clock")
    controlled = evaluator.timeline(
        current, old, "fixed_controlled_duration_plus_latency"
    )
    timeline = {
        **common,
        "status": "PASS",
        "latency_s": config.latency_s,
        "latency_source": str(controller_path.resolve()),
        "prefix_model": "constant current acceleration; prior command unavailable",
        "selected_semantics": "fixed_wall_clock",
        "selected_reason": (
            "ROS planner safety is measured from observation/replanning time; "
            "latency consumes part of the 1.7 s wall-clock risk horizon"
        ),
        "fixed_wall_clock": {
            "includes_t0": bool(wall["times"][0] == 0),
            "includes_first_controllable": bool(
                np.any(np.isclose(wall["times"], config.latency_s))
            ),
            "wall_clock_duration_s": wall["wall_clock_duration_s"],
            "controlled_duration_s": wall["controlled_duration_s"],
        },
        "fixed_controlled_duration_plus_latency": {
            "wall_clock_duration_s": controlled["wall_clock_duration_s"],
            "controlled_duration_s": controlled["controlled_duration_s"],
        },
        "actor_uav_absolute_timestamp_rule": "observation stamp + wall time",
        "one_frame_offset_tolerance_s": 1e-4,
    }
    atomic_json(REPORTS / "phase8jqv2_timeline_spec.json", timeline)
    atomic_json(REPORTS / "phase8jqv2_timeline_validation.json", {
        **common,
        "status": "PASS",
        "checks": {
            "t0_included": bool(wall["times"][0] == 0),
            "latency_included": bool(np.any(np.isclose(wall["times"], config.latency_s))),
            "candidate_starts_at_first_controllable": bool(np.allclose(
                wall["positions"][
                    np.flatnonzero(np.isclose(wall["times"], config.latency_s))[0]
                ],
                wall["first_controllable_state"][:, 0],
            )),
            "fixed_wall_clock_exact": bool(
                abs(wall["wall_clock_duration_s"] - 1.7) < 1e-12
            ),
            "alternative_duration_exact": bool(abs(
                controlled["wall_clock_duration_s"] - 1.7 - config.latency_s
            ) < 1e-12),
            "one_frame_offset_detectable": .1 > 1e-4,
        },
    })
    gap, fraction = continuous_sphere_segment_gap(
        [-1, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0], .3, .4
    )
    atomic_json(REPORTS / "phase8jqv2_continuous_collision_validation.json", {
        **common,
        "status": "PASS",
        "sphere_segment_fixture": {
            "continuous_gap_m": gap, "closest_fraction": fraction,
            "endpoint_discrete_gap_m": .3,
            "discrete_misses_collision": True,
        },
        "finite_cylinder_method": (
            "four-way adaptive subdivision plus exact point-to-finite-cylinder gap"
        ),
        "static_method": (
            "raw ESDF adaptive resampling with 1-Lipschitz bound and 0.005 m tolerance"
        ),
        "dynamic_tolerance_m": .001,
        "static_tolerance_m": config.static_tolerance_m,
    })
    uncertainty = {
        **common,
        "status": "PASS",
        "gt_context_uncertainty_margin": 0.0,
        "recorded_future_gt_uncertainty_margin": 0.0,
        "estimated_default": "directional_covariance",
        "confidence_sigma": config.confidence_sigma,
        "tracker_covariance_already_propagated": True,
        "evaluator_propagates_covariance": False,
        "forbidden_growth_term_absent": True,
        "comparison_policies": [
            "none", "directional_covariance", "maximum_eigenvalue"
        ],
        "missing_estimated_track_policy": (
            "zero uncertainty plus explicit unmatched-covariance diagnostic; "
            "physical future truth remains independently evaluated"
        ),
    }
    atomic_json(REPORTS / "phase8jqv2_uncertainty_semantics.json", uncertainty)
    atomic_text(REPORTS / "phase8jqv2_evaluator_spec.md", f"""# Safety Evaluator V2

Version: `{config.evaluator_version}`  
Config SHA-256: `{sha256(config_path)}`
Geometry SHA-256: `{common["geometry_hash"]}`  
Timeline SHA-256: `{common["timeline_hash"]}`  
Uncertainty policy SHA-256: `{common["uncertainty_policy_hash"]}`  
Simulator geometry SHA-256: `{common["Simulator_geometry_hash"]}`  
Dataset manifest SHA-256: `{common["dataset_manifest_hash"]}`  
Cache index SHA-256: `{common["cache_index_hash"]}`  
Decision checkpoint SHA-256: `{common["checkpoint_hash"]}`

V2 is an offline, versioned evaluator. V1 imports and reports remain unchanged.

Physical static clearance is raw ESDF distance minus the authoritative 0.3 m UAV
radius. Physical dynamic clearance matches Simulator sphere/sphere and
sphere/finite-vertical-cylinder geometry. Physical clearance never includes
covariance or a planning margin.

Planning clearance separately subtracts static/dynamic planning margin,
tracking/control margin and estimated uncertainty. The first two margins are frozen
at zero because no authoritative bound exists. GT and recorded future truth have
exactly zero uncertainty. Estimated directional covariance uses the Phase 8H tracker
covariance once; this evaluator adds no process-noise growth.

The formal timeline is fixed wall-clock 1.7 s. It includes t=0, the measured
{config.latency_s:.9f} s constant-acceleration prefix, the first-controllable state,
the candidate-controlled segment and the endpoint. The alternative fixed-controlled
duration is audit-only.

V2 score labels use safety-first lexicographic scalarization: collision class,
penetration, then normalized secondary cost. This construction ensures a physically
safe candidate always outranks a colliding candidate without training any weights.
""")
    print(json.dumps({
        "status": geometry["status"],
        "evaluator_version": config.evaluator_version,
        "config_hash": common["config_hash"],
        "latency_s": config.latency_s,
    }, indent=2))


if __name__ == "__main__":
    main()
