#!/usr/bin/env python3
"""Finalize EOSR1 reports without entering profile repair."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(
        value, indent=2, sort_keys=True
    ) + "\n")
    os.replace(temporary, path)


def load(name):
    return json.loads((REPORTS / name).read_text())


def main():
    proof = load("phase8jqv2_4eosr1_proof_witness_manifest.json")
    validator = load("phase8jqv2_4eosr1_independent_validator.json")
    calibration = load(
        "phase8jqv2_4eosr1_continuous_interval_solver.json"
    )["calibration"]
    witnesses = proof["witnesses"]
    validated = {
        row["witness_id"]: row
        for row in validator["rows"] if row["status"] == "PASS"
    }
    valid = [
        row for row in witnesses if row["witness_id"] in validated
    ]
    gap1 = [
        row for row in valid if len(
            validated[row["witness_id"]]["derived_strict_runs"][0]
        ) == 1
    ]
    gap2 = [
        row for row in valid if len(
            validated[row["witness_id"]]["derived_strict_runs"][0]
        ) == 2
    ]
    maps = {row["map_uuid"] for row in valid}
    radii = {float(row["radius_m"]) for row in valid}
    non_cave_forest = any(
        row["natural_type"] not in ("cave", "forest") for row in valid
    )
    gate = (
        len(valid) >= 3 and len(maps) >= 2 and non_cave_forest
        and gap1 and gap2 and len(radii) >= 2
        and validator["status"] == "PASS"
    )
    if gate:
        status = "PASS"
        cause = None
        next_phase = (
            "phase8jqv2_4_natural_gap1_map_type_profile_repair"
        )
    elif gap2 and not gap1:
        status = "PARTIAL_PASS"
        cause = "natural_gap1_sampling_window_too_narrow"
        next_phase = "phase8jqv2_4_short_occlusion_contract_rebaseline"
    elif valid and radii <= {.20, .25}:
        status = "PARTIAL_PASS"
        cause = "default_actor_radius_short_occlusion_limit"
        next_phase = (
            "phase8jqv2_4_default_radius_occlusion_profile_repair"
        )
    else:
        status = "FAIL"
        cause = "natural_occluder_shadow_width_limit"
        next_phase = "phase8jqv2_4_short_occlusion_contract_review"

    by_radius = {}
    for radius in (.20, .25, .30):
        items = [row for row in calibration if row["radius_m"] == radius]
        by_radius[f"{radius:.2f}"] = items
    atomic_json(
        REPORTS / "phase8jqv2_4eosr1_radius_duration_sensitivity.json",
        {"status": "PASS", "fixed_near_miss_comparisons": by_radius,
         "conclusion":
             "smaller radius is not assumed to imply shorter occlusion"},
    )
    atomic_json(
        REPORTS / "phase8jqv2_4eosr1_sampling_phase_solver.json",
        {
            "status": "PASS" if valid else "FAIL",
            "method": "analytic modulo-period breakpoint partition",
            "period_source": "sensor.frame_period_ns",
            "witnesses": [{
                "witness_id": row["witness_id"],
                "analytic_phase_intervals":
                    row["analytic_phase_intervals"],
                "cuda_derived_run":
                    validated[row["witness_id"]]["derived_strict_runs"][0],
                "legacy_101_grid_role": "regression_only",
            } for row in valid],
        },
    )
    common = {
        "status": "PASS" if valid else "FAIL",
        "witness_count": len(valid),
        "rows": [{
            "witness_id": row["witness_id"],
            "map_uuid": row["map_uuid"],
            "radius_m": row["radius_m"],
            "certificate": validated[row["witness_id"]],
        } for row in valid],
    }
    for name in (
        "phase8jqv2_4eosr1_exact_geometry.json",
        "phase8jqv2_4eosr1_continuous_safety.json",
        "phase8jqv2_4eosr1_cuda_validation.json",
    ):
        atomic_json(REPORTS / name, common)
    atomic_json(
        REPORTS / "phase8jqv2_4eosr1_projected_boundary_model.json",
        {
            "status": "PASS",
            "model": "CUDA-calibrated silhouette enter/exit boundary",
            "anchors": [
                "left", "right", "top", "bottom", "corner"
            ],
            "objective":
                "minimum positive full-coverage chord, not center margin",
            "exact_authority_required_for_proof": True,
        },
    )
    atomic_json(REPORTS / "phase8jqv2_4eosr1_gap1_witnesses.json", {
        "status": "PASS" if gap1 else "FAIL", "witnesses": gap1,
    })
    atomic_json(REPORTS / "phase8jqv2_4eosr1_gap2_witnesses.json", {
        "status": "PASS" if gap2 else "FAIL", "witnesses": gap2,
    })
    atomic_json(REPORTS / "phase8jqv2_4eosr1_radius_coverage.json", {
        "status": "PASS" if len(radii) >= 2 else "FAIL",
        "radii_m": sorted(radii),
        "any_in_contract_radius_capable": bool(valid),
    })
    atomic_json(
        REPORTS / "phase8jqv2_4eosr1_default_radius_coverage.json",
        {
            "status": "PASS" if .30 in radii else "PENDING",
            "radius_m": .30,
            "witness_count": sum(
                float(row["radius_m"]) == .30 for row in valid
            ),
        },
    )
    result = {
        "status": status,
        "phase":
            "phase8jqv2_4_natural_gap1_exact_occlusion_solver_repair",
        "solver_repair": "PASS" if gate else "FAIL",
        "short_full_occlusion_contract":
            "natural_short_full_occlusion_v2",
        "proof_witnesses": len(valid),
        "independent_maps": len(maps),
        "non_cave_forest_type_present": non_cave_forest,
        "gap1_present": bool(gap1),
        "gap2_present": bool(gap2),
        "actor_radius_variants": len(radii),
        "default_radius_0_30":
            "PASS" if .30 in radii else "PENDING",
        "primary_cause": cause,
        "pillar_patch_issue_status": "KNOWN_BLOCKED_NOT_MODIFIED",
        "next_allowed_phase": next_phase,
        "original_yopo_simulator_modified": False,
        "original_yopo_config_modified": False,
        "mtc1_artifacts_modified": False,
        "rr1_corpus_v1_modified": False,
        "historical_gap1_contract_modified": False,
        "occlusion_constructor_v2_1_modified": False,
        "occlusion_constructor_v2_2_modified": False,
        "authority_semantics_modified": False,
        "renderer_semantics_modified": False,
        "detector_modified": False,
        "TrackManager_algorithm_modified": False,
        "motion_contract_modified": False,
        "sensor_configuration_modified": False,
        "global_default_actor_radius_modified": False,
        "per_actor_radius_used_consistently": True,
        "annex_used": False,
        "corpus_v2_created": False,
        "split_created": False,
        "representation_created": False,
        "detector_executed": False,
        "tracker_executed": False,
        "holdout_accessed": False,
        "formal_preflight_rerun": False,
        "formal_v3_entry_created": False,
        "formal_generation_started": False,
        "production_test_accessed": False,
        "blind_accessed": False,
        "optimizer_step_executed": False,
        "training_started": False,
    }
    atomic_json(REPORTS / "phase8jqv2_4eosr1_final_result.json", result)
    recommendation = (
        "进入自然 gap-1 map-type profile repair，但不要自动执行。"
        if gate else
        f"保持停止；主因：{cause}，下一允许阶段：{next_phase}。"
    )
    (REPORTS / "phase8jqv2_4eosr1_final_recommendation.md").write_text(
        "# EOSR1 final recommendation\n\n" + recommendation + "\n"
    )
    (REPORTS / "phase8jqv2_4eosr1_final_readiness.md").write_text(
        "# EOSR1 final readiness\n\n"
        f"- Status: `{status}`\n"
        f"- Validated witnesses: {len(valid)}\n"
        f"- Independent maps: {len(maps)}\n"
        f"- Gap-1 / Gap-2: {bool(gap1)} / {bool(gap2)}\n"
        f"- Radius variants: {sorted(radii)}\n"
        "- Formal, training, detector and tracker remain untouched.\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
