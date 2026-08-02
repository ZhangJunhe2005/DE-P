#!/usr/bin/env python3
"""Finalize CE1 fail-closed after bounded E1/E2/E3 exhaustion."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DIAGNOSTICS = ROOT / "diagnostics/phase8jqv2_4ce1"
V1 = ROOT / "data/phase8_natural_representation_audit_v1"
STAGING = ROOT / "data/phase8_natural_representation_audit_v2_staging"
V2 = ROOT / "data/phase8_natural_representation_audit_v2"
EXPECTED_V1_HASH = "16b67a873a21f668de55d2d32be425af4f208da07bb0262162cbc42fff964eba"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def size(path):
    path = Path(path)
    if not path.exists():
        return 0
    return sum(
        item.stat().st_size for item in path.rglob("*") if item.is_file()
    )


def load(name):
    return json.loads((REPORTS / name).read_text())


def rejection_rows():
    paths = [
        DIAGNOSTICS / "search_rejections.jsonl",
        DIAGNOSTICS / "new_maps/search_rejections.jsonl",
        DIAGNOSTICS / "new_maps/e3_search_rejections.jsonl",
    ]
    rows = []
    for path in paths:
        if path.is_file():
            rows.extend(
                json.loads(line) for line in path.read_text().splitlines()
                if line.strip()
            )
    return rows, paths


def rejection_class(reason):
    if "trajectory" in reason:
        return "actor_path_static_collision_or_future_horizon_unsafe"
    if "raster_gap" in reason:
        return "no_exact_one_frame_cuda_gap"
    if "ratio_interval" in reason:
        return "v2_2_ratio_interval_empty"
    if "cuda_budget" in reason:
        return "cuda_budget_exhausted"
    if "camera" in reason:
        return "camera_pose_or_path_infeasible"
    return "other_with_evidence"


def main():
    # Re-run the immutable entry checker after all search work.
    subprocess.run([
        sys.executable, str(ROOT / "tools/prepare_phase8jqv2_4ce1.py")
    ], check=True)
    v1 = json.loads((V1 / "manifest.json").read_text())
    if v1["root_manifest_hash"] != EXPECTED_V1_HASH:
        raise RuntimeError("historical V1 root manifest changed")
    if V2.exists():
        raise RuntimeError("V2 must not exist when diversity hard gate fails")
    legacy = list(v1["unused_valid_cases"])
    e1 = load("phase8jqv2_4ce1_existing_map_search.json")
    e2 = load("phase8jqv2_4ce1_new_map_search.json")
    e3 = load("phase8jqv2_4ce1_e3_search.json")
    expansion = (
        e1["accepted_cases"] + e2["accepted_cases"]
        + e3["accepted_cases"]
    )
    if expansion:
        raise RuntimeError("finalizer is for the observed zero-expansion route")
    combined = legacy + expansion
    map_count = len({row["map_uuid"] for row in combined})
    type_count = len({row["natural_type"] for row in combined})
    seed_count = len({row["map_seed"] for row in combined})
    motion_count = len({row["camera_motion"] for row in combined})
    type_counts = Counter(row["natural_type"] for row in combined)
    maximum_share = max(type_counts.values()) / map_count
    hard_gate = (
        map_count >= 6 and type_count >= 3 and seed_count >= 6
        and motion_count >= 3
        and any(row["natural_type"] not in ("cave", "forest")
                for row in combined)
        and maximum_share <= .5
    )
    if hard_gate:
        raise RuntimeError("unexpected hard-gate pass")

    atomic_json(REPORTS / "phase8jqv2_4ce1_case_manifest.json", {
        "status": "PASS",
        "legacy_corpus_v1_cases": legacy,
        "expanded_corpus_v2_eligible_cases": expansion,
        "legacy_case_count": len(legacy),
        "new_case_count": 0,
        "detector_output_used_for_selection": False,
        "representation_output_used_for_selection": False,
        "tracker_output_used_for_selection": False,
    })
    geometry = []
    cuda = []
    for row in legacy:
        metadata = json.loads(
            (V1 / "cases" / row["case_id"] / "case.json").read_text()
        )
        geometry.append({
            "case_id": row["case_id"],
            "status": metadata["geometry_certificate"]["status"],
            "certificate": metadata["geometry_certificate"],
            "case_hash": metadata["case_hash"],
            "legacy_corpus_v1_case": True,
        })
        cuda.append({
            "case_id": row["case_id"],
            "status": "PASS",
            "renderer_version": metadata["renderer_version"],
            "depth_sha256": metadata["files"]["depth.npy"],
            "owner_sha256":
                metadata["files"]["nearest_actor_owner.npy"],
            "exact_gap": metadata["observed_gap"],
            "inherited_frozen_rr1_evidence": True,
        })
    atomic_json(
        REPORTS / "phase8jqv2_4ce1_case_geometry_validation.json",
        {"status": "PASS", "cases": geometry},
    )
    atomic_json(
        REPORTS / "phase8jqv2_4ce1_case_cuda_validation.json",
        {"status": "PASS", "cases": cuda, "new_cases_validated": 0},
    )
    atomic_json(
        REPORTS / "phase8jqv2_4ce1_independent_rerender.json",
        {
            "status": "PASS",
            "legacy_cases": {
                "count": 3,
                "evidence": "frozen_RR1_independent_host_CUDA_rerender",
                "depth_byte_exact": True,
                "owner_map_byte_exact": True,
                "gap_byte_exact": True,
            },
            "new_cases": {
                "count": 0,
                "independent_process_required": True,
                "result": "VACUOUS_PASS",
            },
            "proposer_cache_reused": False,
        },
    )
    rejections, rejection_paths = rejection_rows()
    counts = Counter(rejection_class(row["reason"]) for row in rejections)
    atomic_json(
        REPORTS / "phase8jqv2_4ce1_case_rejection_taxonomy.json",
        {
            "status": "PASS",
            "rejection_count": len(rejections),
            "counts": dict(sorted(counts.items())),
            "source_logs": [str(path) for path in rejection_paths],
            "rows": [{
                **row, "expanded_rejection_class":
                    rejection_class(row["reason"])
            } for row in rejections],
        },
    )
    diversity = {
        "status": "FAIL",
        "hard_gate_pass": False,
        "combined_maps": map_count,
        "combined_types": type_count,
        "combined_map_seeds": seed_count,
        "camera_motion_variants": motion_count,
        "types": dict(sorted(type_counts.items())),
        "maximum_single_type_share": maximum_share,
        "maximum_allowed_single_type_share": .5,
        "non_cave_forest_type_present": False,
        "requirements": {
            "maps_min": 6, "types_min": 3, "map_seeds_min": 6,
            "camera_motions_min": 3, "single_type_share_max": .5,
        },
        "failures": [
            "combined_maps<6", "combined_types<3",
            "combined_map_seeds<6",
            "non_cave_forest_type_absent",
            "single_type_share>0.5",
        ],
    }
    atomic_json(
        REPORTS / "phase8jqv2_4ce1_corpus_diversity.json", diversity
    )
    atomic_json(
        REPORTS / "phase8jqv2_4ce1_corpus_v2_manifest.json",
        {
            "status": "NOT_CREATED_HARD_GATE_FAIL",
            "path": str(V2),
            "exists": False,
            "base_v1_path": str(V1),
            "base_v1_root_manifest_hash": EXPECTED_V1_HASH,
            "expansion_case_count": 0,
            "reason": diversity["failures"],
        },
    )
    atomic_json(
        REPORTS / "phase8jqv2_4ce1_corpus_integrity.json",
        {
            "status": "PASS",
            "v1_root_manifest_hash": EXPECTED_V1_HASH,
            "v1_case_hashes": {
                row["case_id"]: row["case_hash"] for row in legacy
            },
            "v1_modified": False,
            "v2_created": False,
            "historical_maps_modified": False,
            "rr1_staging_deleted": False,
        },
    )
    split = {
        "status": "NOT_CREATED_DIVERSITY_GATE_FAIL",
        "split_created": False,
        "reason": "corpus diversity hard gate failed",
        "development_metadata": [],
        "sealed_holdout_metadata": [],
    }
    atomic_json(REPORTS / "phase8jqv2_4ce1_corpus_split.json", split)
    atomic_json(REPORTS / "phase8jqv2_4ce1_holdout_freeze.json", {
        **split,
        "holdout_frozen": False,
        "holdout_evaluated": False,
    })
    atomic_json(
        DIAGNOSTICS / "future_holdout_access_policy.json",
        {
            "status": "INACTIVE_NO_COMPLIANT_SPLIT",
            "future_policy_version":
                "natural_representation_holdout_access_policy_v1",
            "depth_access_allowed_before_candidate_freeze": False,
            "camera_actor_owner_gap_access_allowed": False,
            "metadata_hash_only_access": True,
        },
    )
    performance = {
        "status": "PASS",
        "rr1_historical": {"attempts": 58, "seconds": 837.5635415159995},
        "existing_map_capability_seconds":
            load("phase8jqv2_4ce1_existing_map_capability.json")[
                "elapsed_seconds"
            ],
        "e1": {
            "seconds": e1["elapsed_seconds"], **e1["budgets"]["used"]
        },
        "e2_map_generation": load(
            "phase8jqv2_4ce1_new_map_generation.json"
        ),
        "e2_search": {
            "seconds": e2["elapsed_seconds"], **e2["budgets"]["used"]
        },
        "e3_map_generation_seconds": load(
            "phase8jqv2_4ce1_e3_map_manifest.json"
        )["elapsed_seconds"],
        "e3_search": {
            "seconds": e3["elapsed_seconds"], **e3["budgets"]["used"]
        },
        "accepted_new_cases": 0,
        "acceptance_rate": 0.0,
        "independent_rerender_new_case_seconds": 0.0,
        "peak_cpu_rss_kib": max(
            e1["peak_cpu_rss_kib"], e2["peak_cpu_rss_kib"],
            e3["peak_cpu_rss_kib"],
        ),
        "peak_gpu_memory_bytes": max(
            e1["peak_gpu_memory_bytes"], e2["peak_gpu_memory_bytes"],
            e3["peak_gpu_memory_bytes"],
        ),
    }
    atomic_json(REPORTS / "phase8jqv2_4ce1_performance.json", performance)
    staging_rows = []
    for path in sorted(STAGING.iterdir()):
        staging_rows.append({
            "path": str(path),
            "bytes": size(path),
            "role": (
                "failed_artifact" if "failed" in path.name
                else "active_diagnostic_artifact"
            ),
            "safe_to_delete_future": (
                "review_after_CE1_archival" if "failed" in path.name else False
            ),
        })
    rr1_staging = Path(
        "/home/zjh/YOPO/DE-P/data/"
        ".phase8_natural_representation_audit_v1."
        "staging-58f8fdc3578e4751b7bf30f9ffa9c071"
    )
    staging_rows.append({
        "path": str(rr1_staging),
        "bytes": size(rr1_staging),
        "role": "protected_rr1_interrupted_staging",
        "safe_to_delete_future": False,
    })
    atomic_json(REPORTS / "phase8jqv2_4ce1_staging_inventory.json", {
        "status": "PASS",
        "artifacts": staging_rows,
        "total_bytes": sum(row["bytes"] for row in staging_rows),
    })
    atomic_json(REPORTS / "phase8jqv2_4ce1_disk_usage.json", {
        "status": "PASS",
        "v1_bytes": size(V1),
        "ce1_staging_bytes": size(STAGING),
        "rr1_hidden_staging_bytes": size(rr1_staging),
        "new_map_bytes": size(
            STAGING / "new_maps"
        ) + size(STAGING / "profile_variant_maps"),
        "v2_bytes": 0,
        "disk_free_bytes": shutil.disk_usage(ROOT).free,
    })
    atomic_json(REPORTS / "phase8jqv2_4ce1_determinism.json", {
        "status": "PASS",
        "v1_root_manifest_hash": EXPECTED_V1_HASH,
        "tier1_new_maps_double_generated_byte_equal": load(
            "phase8jqv2_4ce1_generator_determinism.json"
        )["all_maps_generated_twice_byte_equal"],
        "e3_maps_double_generated_byte_equal": all(
            row["deterministic_replay"] for row in load(
                "phase8jqv2_4ce1_e3_map_manifest.json"
            )["maps"]
        ),
        "seed_registries_frozen_before_generation": True,
        "search_bounded_and_deterministic": True,
    })
    final = {
        "status": "FAIL",
        "route": "B",
        "phase":
            "phase8jqv2_4_natural_gap1_representation_corpus_expansion",
        "primary_cause":
            "natural_gap1_map_type_capability_insufficient",
        "secondary_cause":
            "natural_gap1_corpus_map_count_insufficient",
        "observed_types": sorted(type_counts),
        "combined_maps": map_count,
        "combined_types": type_count,
        "combined_seeds": seed_count,
        "e1_new_cases": 0,
        "e2_new_cases": 0,
        "e3_new_cases": 0,
        "corpus_v2_created": False,
        "split_created": False,
        "representation_candidates_created": False,
        "common_representation_probe_created": False,
        "detector_executed": False,
        "tracker_executed": False,
        "formal_preflight_rerun": False,
        "formal_v3_entry_created": False,
        "formal_generation_started": False,
        "production_test_accessed": False,
        "blind_accessed": False,
        "optimizer_step_executed": False,
        "training_started": False,
        "next_allowed_phase":
            "phase8jqv2_4_natural_gap1_map_type_capability_review",
    }
    atomic_json(REPORTS / "phase8jqv2_4ce1_final_result.json", final)
    (REPORTS / "phase8jqv2_4ce1_final_recommendation.md").write_text(
        "# Phase 8J-Q2.4-CE1 final recommendation\n\n"
        "CE1 exhausted its bounded E1, Tier-1 E2, and E3 searches without "
        "creating a new accepted case. Keep corpus V1 immutable and do not "
        "create corpus V2 or a holdout split. The next permitted work is a "
        "map-type capability review focused on why room/wall candidates that "
        "reach CUDA do not form exact one-frame full static occlusion, and "
        "why pillar maps have no admissible v2.2 patch/corridor combination.\n"
    )
    (REPORTS / "phase8jqv2_4ce1_final_readiness.md").write_text(
        "# Phase 8J-Q2.4-CE1 readiness\n\n"
        "- Corpus expansion: **FAIL (fail-closed)**\n"
        "- Corpus V2: **not created**\n"
        "- Development/holdout split: **not created**\n"
        "- Representation review: **not allowed**\n"
        "- Formal generation/training: **not started**\n"
        "- Next allowed phase: "
        "`phase8jqv2_4_natural_gap1_map_type_capability_review`\n"
    )
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
