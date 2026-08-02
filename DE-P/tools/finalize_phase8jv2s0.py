#!/usr/bin/env python3
"""Finalize S0 at the mandatory static numerical-conformance stop."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
CONFORMANCE_ARTIFACT = (
    ROOT / "artifacts/phase8jv2s0/static_continuous_conformance.npz"
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def not_executed(reason, stage):
    return {
        "status": "NOT_EXECUTED_STATIC_NUMERICAL_GATE_FAIL",
        "stage": stage,
        "reason": reason,
        "network_backward_executed": False,
        "optimizer_step_executed": False,
        "production_test_used": False,
        "blind_used": False,
    }


def main():
    entry = json.loads(
        (REPORTS / "phase8jv2s0_entry_gate.json").read_text()
    )
    fixture = json.loads(
        (REPORTS / "phase8jv2s0_static_fixture_determinism.json").read_text()
    )
    conformance = json.loads(
        (REPORTS / "phase8jv2s0_static_continuous_conformance.json").read_text()
    )
    if entry["status"] != "PASS" or fixture["status"] != "PASS":
        raise RuntimeError("S0 entry/fixture Gate is not PASS")
    if conformance["status"] != "FAIL_NUMERICAL_REBASELINE_REQUIRED":
        raise RuntimeError("S0 is not on numerical-rebaseline route A")

    # Correct the conditional denominators in the already generated initial
    # state report without rerunning inference.
    envelope_path = REPORTS / "phase8jv2s0_initial_state_envelope.json"
    envelope = json.loads(envelope_path.read_text())
    paired = np.load(
        ROOT / "artifacts/phase8jv2s0/fixed_050_seed8403-paired-static.npz",
        allow_pickle=False,
    )
    speed = np.linalg.norm(paired["start"][:, :, 1], axis=1)
    acceleration = np.linalg.norm(paired["start"][:, :, 2], axis=1)
    outside = (speed > 6.0) | (acceleration > 6.0)
    t0 = paired["t0_raw"] - 0.3
    first = paired["first_raw"] - 0.3
    non_deepening = first[outside] >= t0[outside] - 1e-6
    envelope["classification"]["non_deepening_static_clearance"] = {
        "numerator": int(non_deepening.sum()),
        "denominator": int(outside.sum()),
        "fraction": float(non_deepening.mean()),
        "inclusion_rule": "initially outside nominal envelope",
        "exclusion_rule": "initial state inside nominal envelope",
    }
    envelope["classification"]["control_infeasible"] = {
        "status": "NOT_CLASSIFIED",
        "reason": (
            "requires a trustworthy action oracle after the numerical "
            "continuous checker is rebaselined"
        ),
    }
    atomic_json(envelope_path, envelope)

    artifact = np.load(CONFORMANCE_ARTIFACT, allow_pickle=False)
    windows = np.asarray(artifact["trajectory_window_index"], int)
    certified = np.asarray(artifact["recursive_certified_safe"], bool)
    certified_windows = np.unique(windows[certified])
    conformance["comparison"][
        "current_unsafe_recursive_certified_safe_windows"
    ] = int(len(certified_windows))
    conformance["comparison"][
        "current_unsafe_recursive_certified_safe_window_fraction"
    ] = float(
        len(certified_windows) / conformance["audited_failure_window_count"]
    )
    independent = conformance["independent_pointcloud_fixtures"]
    pointcloud_collisions = sum(
        bool(row["pointcloud_collision"]) for row in independent
    )
    conformance["independent_static_geometry_summary"] = {
        "representative_trajectory_count": len(independent),
        "dense_esdf_safe_but_pointcloud_collision_count":
            pointcloud_collisions,
        "disagreement_fraction": (
            pointcloud_collisions / len(independent)
            if independent else None
        ),
        "interpretation": (
            "The next numerical rebaseline must not replace the conservative "
            "bound with dense ESDF sampling alone; it must resolve ESDF voxel/"
            "interpolation versus source-point geometry disagreement."
        ),
    }
    conformance["versioned_repair"] = {
        "implementation_version": "phase8jqv2_static_continuous_v2_1",
        "implementation_path": str(
            (ROOT / "loss/static_continuous_v2_1.py").resolve()
        ),
        "implementation_hash":
            sha256(ROOT / "loss/static_continuous_v2_1.py"),
        "integrated_into_v2_evaluator": False,
        "requires_separate_rebaseline": True,
        "three_state_semantics": [
            "confirmed_collision", "certified_safe", "unknown"
        ],
    }
    atomic_json(
        REPORTS / "phase8jv2s0_static_continuous_conformance.json",
        conformance,
    )

    reason = (
        "The frozen V2 fixed-subdivision endpoint-minus-full-segment bound "
        "exceeds the predeclared false-unsafe tolerance; S0 section 8 "
        "requires a versioned static numerical rebaseline before capacity."
    )
    blocked = {
        "phase8jv2s0_estimated_gradient_fix.json":
            "estimated gradient repair",
        "phase8jv2s0_surrogate_alignment.json":
            "surrogate/exact alignment",
        "phase8jv2s0_candidate_bank_effective_capacity.json":
            "fallback/reachable candidate bank",
        "phase8jv2s0_static_capacity_reaudit.json":
            "paired static capacity reaudit",
        "phase8jv2s0_dynamic_capacity_reaudit.json":
            "dynamic capacity reaudit",
    }
    for name, stage in blocked.items():
        atomic_json(REPORTS / name, not_executed(reason, stage))

    final = {
        "status": "FAIL",
        "audit_complete": False,
        "primary_cause": "static_continuous_certificate_implementation",
        "static_fixture_deterministic": True,
        "static_pairing_repaired": True,
        "static_numerical_checker_trustworthy": False,
        "estimated_gradient_fix_executed": False,
        "surrogate_alignment_executed": False,
        "static_capacity_reaudit_executed": False,
        "dynamic_capacity_reaudit_executed": False,
        "network_weights_modified": False,
        "training_executed": False,
        "score_training_executed": False,
        "candidate_generator_frozen": False,
        "production_test_used": False,
        "blind_used": False,
        "depth_dataset_rebuilt": False,
        "PLY_dataset_rebuilt": False,
        "next_allowed_phase": "phase8jqv2_1_static_numerical_rebaseline",
    }
    atomic_json(REPORTS / "phase8jv2s0_final_result.json", final)
    recommendation = f"""# Phase 8J-V2-S0 final recommendation

Status: **STOP — route A selected.**

The immutable static fixture Gate passed for all 20 worker/batch combinations.
Paired C0 reproduces a 36.83% legacy-certificate failure rate, so random
pairing was real but is not the only cause.

The full continuous-conformance audit found:

- 3,683 paired C0 failure windows;
- {conformance['comparison']['current_unsafe_dense_128_safe_windows']} windows
  ({conformance['comparison']['current_unsafe_dense_128_safe_window_fraction']:.2%})
  with at least one dense-128 safe candidate;
- {len(certified_windows)} windows
  ({len(certified_windows)/conformance['audited_failure_window_count']:.2%})
  with at least one recursively certified-safe candidate;
- {conformance['comparison']['recursive_unknown_trajectories']} trajectories
  remain unknown and are not silently classified as collision.
- {pointcloud_collisions}/{len(independent)} representative dense-ESDF-safe
  trajectories collide under the independent source-point nearest-distance
  fixture.

This exceeds the predeclared 0.5% window and 0.1% trajectory tolerances.
`current lower-bound < 0` was therefore conflating inability to certify with
confirmed collision. Dense ESDF sampling alone is also not an acceptable
replacement because the independent geometry fixture exposes a separate
voxel/interpolation disagreement.

The audit-only `phase8jqv2_static_continuous_v2_1` implementation preserves
raw ESDF minus the 0.3 m UAV radius and introduces explicit
confirmed-collision / certified-safe / unknown states. It is not integrated
into the frozen V2 evaluator. The next authorized work is a separate
`phase8jqv2_1_static_numerical_rebaseline`.

Estimated-gradient repair, surrogate alignment, candidate-bank redesign and
static/dynamic capacity audits were not executed because they are downstream
of the failed numerical Gate.
"""
    (REPORTS / "phase8jv2s0_final_recommendation.md").write_text(
        recommendation, encoding="utf-8"
    )
    readiness = """# Phase 8J-V2-S0 final readiness

**FAIL — static numerical rebaseline required.**

Fixture pairing and determinism are ready. The frozen V2 static continuous
numerical implementation is not trustworthy as a binary collision Gate.
Coverage training, candidate architecture changes, dynamic re-audit and score
training remain blocked.

Next allowed phase: `phase8jqv2_1_static_numerical_rebaseline`.
"""
    (REPORTS / "phase8jv2s0_final_readiness.md").write_text(
        readiness, encoding="utf-8"
    )
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
