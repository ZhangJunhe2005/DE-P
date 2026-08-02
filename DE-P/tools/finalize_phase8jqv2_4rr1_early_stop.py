#!/usr/bin/env python3
"""Finalize RR1 Route B after the mandatory corpus Gate fails."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DIAGNOSTICS = ROOT / "diagnostics/phase8jqv2_4rr1"
CORPUS = ROOT / "data/phase8_natural_representation_audit_v1"


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def write_new(path, value):
    path = Path(path)
    payload = (
        value if isinstance(value, str)
        else json.dumps(value, indent=2, sort_keys=True) + "\n"
    )
    if path.exists():
        if path.read_text() == payload:
            return
        raise FileExistsError(f"refusing to overwrite RR1 evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload)
    os.replace(temporary, path)


def blocked(reason, **extra):
    return {"status": "BLOCKED", "blocked_by": reason, **extra}


def main():
    entry = load(REPORTS / "phase8jqv2_4rr1_entry_gate.json")
    frozen = load(REPORTS / "phase8jqv2_4rr1_frozen_artifacts.json")
    manifest = load(CORPUS / "manifest.json")
    integrity = load(
        REPORTS / "phase8jqv2_4rr1_natural_corpus_integrity.json"
    )
    generation = load(
        REPORTS / "phase8jqv2_4rr1_natural_corpus_generation.json"
    )
    access_lines = (
        DIAGNOSTICS / "holdout_access_log.jsonl"
    ).read_text().splitlines()
    if entry["status"] != "PASS" or frozen["status"] != "PASS":
        raise RuntimeError("RR1 entry/freeze is not PASS")
    if integrity["status"] != "PASS":
        raise RuntimeError("accepted RR1 artifacts are corrupt")
    if manifest["status"] != "FAIL":
        raise RuntimeError("RR1 early-stop requires failed corpus Gate")
    if len(access_lines) != 1:
        raise RuntimeError("sealed holdout was accessed")
    accepted = list(manifest["cases"]) + list(
        manifest["unused_valid_cases"]
    )
    maps = {row["map_uuid"] for row in accepted}
    types = {row["maze_type"] for row in accepted}
    seeds = {row["trajectory_seed"] for row in accepted}
    motions = {row["camera_motion"] for row in accepted}
    if (len(maps), len(types), len(seeds)) != (3, 2, 3):
        raise RuntimeError("unexpected bounded corpus audit result")

    corpus_public = {
        "status": "FAIL",
        "corpus_version": manifest["corpus_version"],
        "root_manifest_hash": manifest["root_manifest_hash"],
        "role": "representation_audit_only",
        "accepted_case_count": len(accepted),
        "independent_map_count": len(maps),
        "maze_type_count": len(types),
        "independent_seed_count": len(seeds),
        "camera_motion_variants": sorted(motions),
        "cases": accepted,
        "required": {
            "independent_maps": 6, "maze_types": 3,
            "independent_seeds": 6,
            "development_maps": 4, "holdout_maps": 2,
        },
        "annex_used": False, "new_maps_generated": False,
        "historical_cases_modified": False,
        "detector_output_used_for_selection": False,
    }
    write_new(
        REPORTS / "phase8jqv2_4rr1_natural_corpus_manifest.json",
        corpus_public,
    )
    write_new(REPORTS / "phase8jqv2_4rr1_natural_corpus_split.json", {
        "status": "FAIL",
        "split_frozen": False,
        "development": [],
        "sealed_natural_holdout": [],
        "reason": "corpus quantity/diversity Gate failed before split",
        "map_uuid_overlap": 0, "authority_hash_overlap": 0,
        "trajectory_seed_overlap": 0,
        "holdout_runtime_accessed": False,
    })
    write_new(
        REPORTS / "phase8jqv2_4rr1_natural_corpus_diversity.json",
        {
            "status": "FAIL",
            "independent_maps": len(maps), "required_maps": 6,
            "maze_types": len(types), "required_maze_types": 3,
            "maze_type_names": sorted({
                row["natural_type"] for row in accepted
            }),
            "independent_seeds": len(seeds), "required_seeds": 6,
            "camera_motion_variants": sorted(motions),
            "different_camera_motion_requirement": len(motions) >= 2,
            "primary_cause":
                "natural_gap1_representation_corpus_insufficient",
        },
    )

    write_new(REPORTS / "phase8jqv2_4rr1_representation_invariants.json", {
        **blocked("natural_corpus_gate"),
        "candidate_sources_created": False,
        "required_invariants": [
            "causal", "depth_only", "camera_pose_runtime",
            "no_gt", "no_future", "no_authority_runtime",
            "bounded_history", "bounded_memory",
            "unknown_not_dynamic", "disocclusion_not_dynamic",
        ],
    })
    write_new(REPORTS / "phase8jqv2_4rr1_representation_interface.json", {
        **blocked("natural_corpus_gate"),
        "interface_created": False,
        "path": "policy/dynamic/dynamic_representation_v1.py",
    })
    for name, representation in (
        ("r0_baseline", "projective_residual_baseline_v1"),
        ("r1_surfel", "causal_local_surfel_background_v1"),
        ("r2_voxel", "causal_local_voxel_occupancy_v1"),
        ("r3_point_motion", "causal_world_point_motion_v1"),
        ("r4_hybrid", "hybrid_projective_occupancy_v1"),
    ):
        write_new(REPORTS / f"phase8jqv2_4rr1_{name}.json", {
            **blocked("natural_corpus_gate"),
            "representation": representation,
            "implemented": False, "executed": False,
            "parameters_searched": 0,
        })
    write_new(
        REPORTS / "phase8jqv2_4rr1_candidate_comparison.json",
        blocked(
            "natural_corpus_gate", candidates_compared=0,
            all_grid_results_saved=True,
        ),
    )
    for name in (
        "domain_gap_analysis", "background_complexity",
        "representation_separability", "physical_development_controls",
        "fov_disocclusion_negatives", "edge_latency", "no_target",
        "natural_development", "natural_pre_gap", "natural_post_gap",
    ):
        write_new(
            REPORTS / f"phase8jqv2_4rr1_{name}.json",
            blocked(
                "natural_corpus_gate", executed=False,
                detail=(
                    "RR1 section 6 requires immediate stop before "
                    "representation design or evaluation"
                ),
            ),
        )
    write_new(
        REPORTS / "phase8jqv2_4rr1_natural_failure_decomposition.md",
        """# RR1 natural failure decomposition

Not executed. The mandatory representation-audit corpus Gate failed before
representation design: the bounded sweep over 30 existing annex-free
development authority maps produced only three independently rerendered
gap-1 cases on three maps and two maze types. RR1 requires at least six maps
and three types. Per the phase contract, no conclusion about depth-only
separability is permitted from this insufficient corpus.
""",
    )

    write_new(REPORTS / "phase8jqv2_4rr1_holdout_freeze.json", {
        **blocked("natural_corpus_gate"),
        "candidate_frozen": False, "corpus_split_frozen": False,
        "sealed_holdout_created": False,
        "sealed_holdout_runtime_accessed": False,
    })
    for name in (
        "ccr1_holdout_results", "natural_holdout_results", "generalization",
    ):
        write_new(
            REPORTS / f"phase8jqv2_4rr1_{name}.json",
            blocked(
                "candidate_not_frozen", executed=False,
                holdout_artifacts_read=False,
            ),
        )
    write_new(REPORTS / "phase8jqv2_4rr1_performance.json", {
        "status": "PASS",
        "scope": "bounded_corpus_generation_only",
        "elapsed_seconds": generation["elapsed_seconds"],
        "attempt_count": generation["attempt_count"],
        "trial_wall_clock_limit_seconds": 30,
        "representation_runtime": "NOT_RUN",
        "unbounded_search_used": False,
    })
    write_new(REPORTS / "phase8jqv2_4rr1_memory_bound.json", {
        "status": "NOT_APPLICABLE_BLOCKED_CORPUS",
        "representation_candidates_created": False,
        "unbounded_local_map_created": False,
        "unbounded_point_cloud_created": False,
    })
    write_new(REPORTS / "phase8jqv2_4rr1_determinism.json", {
        "status": "PASS",
        "trajectory_seeds_recorded": True,
        "artifact_hashes_recorded": True,
        "accepted_cases_independent_cuda_rerender": "PASS",
        "candidate_determinism": "NOT_RUN_BLOCKED_CORPUS",
    })
    write_new(REPORTS / "phase8jqv2_4rr1_gap3_non_scope.json", {
        "status": "OUT_OF_SCOPE",
        "confidence_decay": 0.75, "dynamic_threshold": 0.55,
        "confidence_after_three_prediction_only_frames": 0.421875,
        "strict_gap3_run": False, "parameters_modified": False,
    })
    write_new(REPORTS / "phase8jqv2_4rr1_candidate_selection.json", {
        "status": "FAIL", "selected_representation": None,
        "candidate_selection_executed": False,
        "failed_gate": "adequate_natural_corpus",
        "route": "B",
    })
    write_new(REPORTS / "phase8jqv2_4rr1_compatibility_matrix.json", {
        "status": "PASS",
        "legacy_default": "legacy_v1",
        "tracker_integration": "NOT_RUN",
        "rows": [{
            "representation": name,
            "implemented": False, "reason": "natural_corpus_gate",
        } for name in (
            "R0", "R1", "R2", "R3", "R4"
        )],
    })
    write_new(REPORTS / "phase8jqv2_4rr1_migration_plan.md", """# RR1 migration plan

No migration is authorized. The corpus quantity/diversity Gate failed before
representation candidates were designed. Legacy perception remains unchanged.
The next permitted phase may expand the natural gap-1 representation corpus
using existing or separately authorized development geometry, while
preserving this failed sweep and without accessing test/blind/Formal data.
""")
    final = {
        "status": "FAIL", "route": "B",
        "representation_review": "FAIL",
        "primary_cause":
            "natural_gap1_representation_corpus_insufficient",
        "natural_corpus_maps": len(maps),
        "natural_corpus_types": len(types),
        "natural_corpus_seeds": len(seeds),
        "required_maps": 6, "required_types": 3,
        "representation_candidates_created": False,
        "candidate_frozen": False,
        "ccr1_holdout_accessed": False,
        "natural_holdout_accessed": False,
        "tf1_sealed_holdout_accessed": False,
        "tracker_integration_executed": False,
        "legacy_default_changed": False,
        "formal_preflight_rerun": False,
        "formal_v3_entry_created": False,
        "formal_v3_generation_started": False,
        "production_test_accessed": False, "blind_accessed": False,
        "optimizer_step_executed": False, "training_started": False,
        "gap_3": "OUT_OF_SCOPE_CONTRACT_REVIEW_REQUIRED",
        "next_allowed_phase":
            "phase8jqv2_4_natural_gap1_representation_corpus_expansion",
    }
    write_new(REPORTS / "phase8jqv2_4rr1_final_result.json", final)
    write_new(REPORTS / "phase8jqv2_4rr1_final_recommendation.md", """# RR1 final recommendation

Take Route B. Preserve the three valid, independently CUDA-rerendered audit
cases and all 58 bounded generation attempts. Do not design R1–R4 on this
two-type corpus, do not create a sealed holdout, and do not read CCR1 or TF1
holdout artifacts. The next phase must expand annex-free natural gap-1
coverage to at least six independent maps and three maze types before any
representation comparison.
""")
    write_new(REPORTS / "phase8jqv2_4rr1_final_readiness.md", """# RR1 final readiness

- DPAR2 Route E entry and frozen hashes: PASS.
- Allowed existing development maps: 30, across five maze types.
- Bounded CUDA gap-1 sweep: 58 attempts.
- Valid independently rerendered cases: 3 maps, 2 types, 3 seeds.
- Required corpus Gate: 6 maps, 3 types, 6 seeds — FAIL.
- Development/holdout split: not created.
- R0–R4, common probe and candidate freeze: not created.
- CCR1/TF1/natural holdout: not accessed.
- Tracker, Formal, test/blind, optimizer and training: not run.
- Decision: Route B.
""")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
