#!/usr/bin/env python3
"""Separate natural/annex capability and emit the fail-closed preflight."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIRST = ROOT / (
    "reports/occlusion_constructor_v2_2_"
    "natural_sweep_first_seed_all_types_gaps123.json"
)
REMAINING = ROOT / (
    "reports/occlusion_constructor_v2_2_"
    "natural_sweep_remaining_seeds_all_types_gaps123.json"
)
ANNEX_IDENTITY = ROOT / (
    "reports/phase8jqv2_4m1_annex_fixture_v1_identity_failure.json"
)
CAPABILITY = ROOT / (
    "reports/occlusion_constructor_v2_2_capability_separation.json"
)
PREFLIGHT = ROOT / (
    "reports/occlusion_constructor_v2_2_formal_preflight.json"
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_new(path, value):
    if path.exists():
        raise FileExistsError(f"refusing to overwrite report: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main():
    natural_reports = [
        json.loads(FIRST.read_text()),
        json.loads(REMAINING.read_text()),
    ]
    rows = [
        row for report in natural_reports for row in report["results"]
    ]
    if len(rows) != 45:
        raise RuntimeError("natural sweep must contain 15 maps x 3 gaps")
    annex = json.loads(ANNEX_IDENTITY.read_text())
    natural_types = sorted({row["natural_type"] for row in rows})
    type_rows = {}
    for natural_type in natural_types:
        selected = [
            row for row in rows if row["natural_type"] == natural_type
        ]
        type_rows[natural_type] = {
            "independent_maps": len({row["map_uuid"] for row in selected}),
            "independent_seeds": len({row["seed"] for row in selected}),
            "cases": len(selected),
            "geometry_pass_count": sum(
                row["geometry_status"] == "PASS" for row in selected
            ),
            "strict_identity_pass_count": sum(
                row["identity_status"] == "PASS" for row in selected
            ),
            "natural_occlusion_capable": any(
                row["identity_status"] == "PASS" for row in selected
            ),
        }
    geometry_pass = [
        row for row in rows if row["geometry_status"] == "PASS"
    ]
    identity_pass = [
        row for row in rows if row["identity_status"] == "PASS"
    ]
    annex_geometry = any(
        "constructed_gap" in row for row in annex["map_types"]
    )
    annex_strict = any(
        row["status"] == "PASS" for row in annex["map_types"]
    )
    capability = {
        "status": "PASS",
        "meaning": "classification completed; capability Gates remain closed",
        "constructor_implementation_hash": sha256(
            ROOT / "authoritative_dataset/occlusion_constructor_v2_2.py"
        ),
        "natural": {
            "source": "original development maps without annex",
            "maps": 15,
            "seeds": 15,
            "cases": 45,
            "gaps_swept": [1, 2, 3],
            "geometry_pass_count": len(geometry_pass),
            "strict_identity_pass_count": len(identity_pass),
            "natural_geometry_occlusion_observed": bool(geometry_pass),
            "natural_occlusion_capable": bool(identity_pass),
            "by_type": type_rows,
        },
        "annex_fixture": {
            "fixture_version": "occlusion_fixture_annex_v1",
            "legacy_annex_evidence_only": True,
            "geometry_gap_observed": annex_geometry,
            "strict_identity_pass_count": sum(
                row["status"] == "PASS" for row in annex["map_types"]
            ),
            "annex_fixture_occlusion_capable": annex_strict,
            "base_maze_type_natural_credit_awarded": False,
            "future_policy": {
                "limited_specialized_subset_only": True,
                "attach_to_all_maps": False,
                "dynamic_dataset_majority": False,
                "multiple_positions_orientations_sizes_required": True,
            },
        },
        "capabilities_merged": False,
        "formal_generation_started": False,
        "training_executed": False,
        "test_accessed": False,
        "blind_accessed": False,
        "source_report_hashes": {
            str(FIRST.relative_to(ROOT)): sha256(FIRST),
            str(REMAINING.relative_to(ROOT)): sha256(REMAINING),
            str(ANNEX_IDENTITY.relative_to(ROOT)): sha256(ANNEX_IDENTITY),
            "reports/phase8jqv2_4m1_annex_fixture_v1_geometry_snapshot.json":
                sha256(ROOT / (
                    "reports/phase8jqv2_4m1_"
                    "annex_fixture_v1_geometry_snapshot.json"
                )),
            "reports/occlusion_constructor_v2_2_cuda_controls_initial_failure.json":
                sha256(ROOT / (
                    "reports/occlusion_constructor_v2_2_"
                    "cuda_controls_initial_failure.json"
                )),
            "reports/occlusion_constructor_v2_2_cuda_controls_gap2_trajectory_failure.json":
                sha256(ROOT / (
                    "reports/occlusion_constructor_v2_2_"
                    "cuda_controls_gap2_trajectory_failure.json"
                )),
            "reports/occlusion_constructor_v2_2_natural_sweep_smoke_wall_gap1.json":
                sha256(ROOT / (
                    "reports/occlusion_constructor_v2_2_"
                    "natural_sweep_smoke_wall_gap1.json"
                )),
            "reports/occlusion_constructor_v2_2_natural_sweep_smoke_wall_gap1_v2.json":
                sha256(ROOT / (
                    "reports/occlusion_constructor_v2_2_"
                    "natural_sweep_smoke_wall_gap1_v2.json"
                )),
            "reports/occlusion_constructor_v2_2_natural_sweep_smoke_wall_gap1_v3.json":
                sha256(ROOT / (
                    "reports/occlusion_constructor_v2_2_"
                    "natural_sweep_smoke_wall_gap1_v3.json"
                )),
        },
    }
    strict_maps = {
        (row["map_uuid"], row["seed"]) for row in identity_pass
    }
    strict_gaps = sorted({
        row["requested_gap_frames"] for row in identity_pass
    })
    preflight = {
        "status": "FAIL",
        "formal_preflight": "FAIL",
        "blocking_reasons": [
            "strict natural identity pass count is zero",
            "no three independent maps satisfy the identity contract",
            "no 1/2/3-gap set satisfies the identity contract",
            "annex legacy fixture identity validation failed",
        ],
        "observed": {
            "strict_independent_maps": len(strict_maps),
            "strict_gaps": strict_gaps,
            "natural_geometry_pass_count": len(geometry_pass),
            "natural_strict_identity_pass_count": len(identity_pass),
            "annex_fixture_strict_identity_pass": annex_strict,
        },
        "required_before_formal_review": {
            "independent_maps_minimum": 3,
            "multiple_independent_seeds": True,
            "gaps": [1, 2, 3],
            "pre_gap_confirmed_dynamic": True,
            "gap_prediction_only_same_track": True,
            "post_gap_same_id": True,
            "independent_validator_pass": True,
            "no_track_birth_deletion_replacement": True,
        },
        "formal_v3_generation_entry_created": False,
        "formal_generation_started": False,
        "training_executed": False,
        "test_accessed": False,
        "blind_accessed": False,
        "next_allowed_work": (
            "diagnose frozen perception pre-gap confirmation and identity "
            "using the six retained natural geometry-positive cases"
        ),
    }
    atomic_new(CAPABILITY, capability)
    atomic_new(PREFLIGHT, preflight)
    print(json.dumps({
        "status": "PASS",
        "capability_report": str(CAPABILITY),
        "preflight_report": str(PREFLIGHT),
        "formal_preflight": "FAIL",
        "natural_geometry_pass_count": len(geometry_pass),
        "natural_strict_identity_pass_count": len(identity_pass),
        "annex_fixture_strict_identity_pass": annex_strict,
    }, indent=2))


if __name__ == "__main__":
    main()
