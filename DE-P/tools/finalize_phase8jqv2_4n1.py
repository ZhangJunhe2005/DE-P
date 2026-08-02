#!/usr/bin/env python3
"""Finalize N1 route C after bounded gap-1 profile screening."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DIAG = ROOT / "diagnostics/phase8jqv2_4n1"


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def blocked(blocker):
    return {
        "status": "NOT_RUN_BLOCKED_GAP1",
        "blocker": blocker,
        "formal_preflight_rerun": False,
        "formal_generation_started": False,
        "training_started": False,
    }


def main():
    entry = load(REPORTS / "phase8jqv2_4n1_entry_gate.json")
    maps = load(REPORTS / "phase8jqv2_4n1_profile_candidates.json")
    initial = load(DIAG / "profile_debug_gap1_sweep.json")
    final = load(DIAG / "profile_debug_gap1_final_bounded_screening.json")
    pixel = load(REPORTS / "phase8jqv2_4n1_stage1_pixel_pipeline.json")
    pixel_counts = load(
        DIAG / "stage1_pixel_counts_per_step.json"
    )
    profile_counts = Counter(row["profile_name"] for row in maps["maps"])
    type_counts = Counter(str(row["maze_type"]) for row in maps["maps"])
    ordinary = [
        row for row in maps["maps"] if row["ordinary_navigation_capable"]
    ]
    p0_pass = []
    p1_pass = []
    all_proposals = []
    for source_name, sweep in (("initial_v2_2", initial), ("joint_n1", final)):
        for map_row in sweep["maps"]:
            for proposal in map_row["proposals"]:
                item = {
                    "source": source_name,
                    "map_uuid": map_row["map_uuid"],
                    "profile_name": map_row["profile_name"],
                    "maze_type": map_row["maze_type"],
                    **proposal,
                }
                all_proposals.append(item)
                if proposal.get("p0") == "PASS":
                    p0_pass.append(item)
                if proposal.get("p1") == "PASS":
                    p1_pass.append(item)

    write_new(REPORTS / "phase8jqv2_4n1_observability_proxy_validation.json", {
        "status": "PASS_AS_BROAD_PHASE_NEGATIVE",
        "fixed_i1_case_reproduced": pixel["status"] == "REPRODUCED",
        "pre_gap_visible_pixels": pixel["pre_gap_visible_pixels"],
        "first_zero_stage": pixel["first_zero_stage"],
        "frozen_measurement_valid": False,
        "proxy_accepts_as_observable": False,
        "proxy_is_final_detector": False,
        "validator_reads_proxy": False,
    })
    write_new(REPORTS / "phase8jqv2_4n1_observability_false_positive_analysis.json", {
        "status": "NO_ACCEPTED_PROXY_CANDIDATES",
        "p0_pass_count": len(p0_pass),
        "p1_pass_count": len(p1_pass),
        "p1_false_positive_count": 0,
        "p1_false_negative_audit": "NOT_RUN_WITHOUT_SAFE_P0_CONTROL",
        "finding": "all safe CUDA-gap candidates were rejected by observability proxy",
    })
    write_new(REPORTS / "phase8jqv2_4n1_profile_sweep.json", {
        "status": "FAIL",
        "stage_pa": {
            "maps": final["map_count"],
            "profiles": len(profile_counts),
            "maze_types": len(type_counts),
            "proposals_per_map": final["maximum_proposals_per_map"],
            "passing_maps": final["passing_map_count"],
            "witnesses": final["witness_count"],
            "rejection_reasons": final["rejection_reasons"],
            "elapsed_seconds": final["elapsed_seconds"],
        },
        "historical_first_screen_preserved":
            "diagnostics/phase8jqv2_4n1/profile_debug_gap1_sweep.json",
        "joint_proposer_screen_preserved":
            "diagnostics/phase8jqv2_4n1/profile_debug_gap1_final_bounded_screening.json",
        "stage_pb": "NO_P1_PASS",
        "unbounded_search_used": False,
    })
    write_new(REPORTS / "phase8jqv2_4n1_profile_holdout.json", blocked(
        "debug gap-1 has zero strict witnesses"
    ))
    write_new(REPORTS / "phase8jqv2_4n1_selected_profiles.json", {
        "status": "FAIL", "selected_profiles": [],
        "reason": "no debug profile produced a gap-1 witness",
        "case_specific_selection": False,
        "annex_used": False,
    })
    write_new(REPORTS / "phase8jqv2_4n1_map_type_diversity.json", {
        "status": "FAIL",
        "ordinary_navigation_map_types": sorted(type_counts),
        "gap1_identity_map_types": [],
        "gap2_identity_map_types": [],
        "gap1_required_types": 3,
        "gap2_required_types": 2,
    })
    write_new(REPORTS / "phase8jqv2_4n1_gap1_validation.json", {
        "status": "FAIL",
        "independent_maps": 0, "map_types": 0, "witnesses": 0,
        "required_independent_maps": 6,
        "required_map_types": 3,
        "required_witnesses_per_map": 2,
        "primary_failure": "no P1-observable safe natural gap candidate",
    })
    write_new(REPORTS / "phase8jqv2_4n1_gap2_validation.json", blocked(
        "gap-1 identity gate failed"
    ))
    write_new(REPORTS / "phase8jqv2_4n1_gap3_diagnostic.json", {
        "status": "CONTRACT_DIAGNOSTIC_ONLY",
        "natural_sequence_run": False,
        "track_identity_result": "NOT_EVALUATED_BLOCKED_GAP1",
        "confidence_sequence_upper_bound": [1.0, .75, .5625, .421875],
        "dynamic_exit_by_third_miss": True,
        "strict_pass_claimed": False,
        "next_allowed_phase":
            "phase8jqv2_4_gap3_occlusion_contract_review",
    })
    write_new(REPORTS / "phase8jqv2_4n1_identity_lifecycle.json", {
        "status": "NO_WITNESS",
        "gap1_same_track": "NOT_DEMONSTRATED_ON_N1_MAPS",
        "gap2_same_track": "NOT_RUN_BLOCKED_GAP1",
        "gap3": "CONTRACT_ISOLATED",
        "tracker_modified": False,
        "i1_synthetic_lifecycle_regression": "PASS",
    })
    write_new(REPORTS / "phase8jqv2_4n1_independent_validator.json", {
        **blocked("no accepted debug gap-1 candidate"),
        "certificate_input_allowed": False,
        "proposer_pass_trusted": False,
        "positive_validation_claimed": False,
    })
    write_new(REPORTS / "phase8jqv2_4n1_negative_controls.json", {
        "status": "INCOMPLETE_BLOCKED_WITHOUT_POSITIVE_PROFILE",
        "validated_rejections": {
            "cuda_visible_temporal_foreground_zero": "PASS",
            "insufficient_background_depth_contrast": "PASS",
            "insufficient_component_support": "PASS",
            "pre_gap_measurement_interruption": "PASS",
            "first_post_gap_measurement_invalid": "PASS",
            "long_gap_deletion": "PASS_I1_PROBE",
        },
        "not_run_without_selected_profile": [
            "pure_tangential_repeated_ray", "projected_support",
            "gap_before_confirmation", "gap_before_dynamic", "FOV_exit",
            "behind_camera", "max_depth_exit", "partial_occlusion",
            "metadata_only", "artificial_hidden_frame", "new_ID_replacement",
        ],
        "all_controls_rejected": False,
        "natural_capability_credit": False,
    })
    write_new(REPORTS / "phase8jqv2_4n1_determinism.json", {
        "status": "PASS",
        "map_replay": "raw cloud A/B byte-identical for all 30 maps",
        "map_count": maps["map_count"],
        "profile_hash": maps["profile_hash"],
        "proposal_order":
            "maze_type/profile/map then bounded patch/standoff/radial/tangent",
        "case_id_branch": False, "map_uuid_branch": False,
        "seed_branch": False,
    })
    write_new(REPORTS / "phase8jqv2_4n1_static_feasibility.json", {
        "status": "PASS" if len(ordinary) == maps["map_count"] else "FAIL",
        "maps": maps["map_count"], "ordinary_navigation_capable": len(ordinary),
        "maze_types": sorted(type_counts), "old_artifacts_modified": False,
    })
    write_new(REPORTS / "phase8jqv2_4n1_ordinary_dynamic_feasibility.json", blocked(
        "gap-1 profile screening failed before ordinary dynamic regression"
    ))
    write_new(REPORTS / "phase8jqv2_4n1_distribution_regression.json", {
        "status": "PARTIAL",
        "static_distribution": "PASS",
        "ordinary_dynamic_distribution": "NOT_RUN_BLOCKED_GAP1",
        "profile_repair_breaks_general_navigation_distribution": False,
        "formal_readiness": False,
    })
    write_new(REPORTS / "phase8jqv2_4n1_performance.json", {
        "status": "PASS_BOUNDED",
        "map_generation_seconds": maps["elapsed_seconds"],
        "initial_gap1_sweep_seconds": initial["elapsed_seconds"],
        "final_gap1_screen_seconds": final["elapsed_seconds"],
        "maximum_maps": 30,
        "maximum_debug_maps": 15,
        "maximum_final_proposals_per_map": 4,
        "unbounded_search_used": False,
    })
    rejected = [
        {
            "map_uuid": item["map_uuid"],
            "profile_name": item["profile_name"],
            "maze_type": item["maze_type"],
            "proposal_index": item.get("proposal_index"),
            "stage": (
                "P0" if item.get("p0") != "PASS"
                else "P1" if item.get("p1") != "PASS" else "P2"
            ),
            "reason": item.get("reason") or item.get(
                "observability", {}
            ).get("rejection_reasons"),
            "error": item.get("error"),
        }
        for item in all_proposals
        if item.get("p2") != "PASS"
    ]
    write_new(DIAG / "profile_rejections.json", {
        "count": len(rejected), "events": rejected
    })
    pre = pixel_counts["pre_gap_frame"]
    post = pixel_counts["first_post_gap_frame"]
    write_new(DIAG / "foreground_failures.json", {
        "events": [
            {"location": "fixed_i1_pre_gap", **pre},
            {"location": "fixed_i1_first_post_gap", **post},
        ]
    })
    write_new(DIAG / "history_failures.json", {
        "pre_gap_history_supported": pre["history_supported_visible_pixels"],
        "pre_gap_closer_supported": pre["closer_supported_visible_pixels"],
        "post_gap_history_supported": post["history_supported_visible_pixels"],
    })
    write_new(DIAG / "component_failures.json", {
        "pre_gap_union_seeds": pre["seed_union_visible_pixels"],
        "minimum_seed_pixels": 8,
        "pre_gap_component_pixels": pre["component_visible_pixels"],
        "post_gap_union_seeds": post["seed_union_visible_pixels"],
        "post_gap_component_pixels": post["component_visible_pixels"],
    })
    write_new(DIAG / "post_gap_failures.json", {"events": [post]})
    write_new(DIAG / "identity_failures.json", {
        "status": "NO_P2_IDENTITY_CANDIDATE", "events": []
    })
    write_new(DIAG / "holdout_failures.json", {
        "status": "NOT_RUN_BLOCKED_GAP1", "events": []
    })
    invariants = {
        "occlusion_constructor_v2_1_modified": False,
        "occlusion_constructor_v2_2_modified": False,
        "occlusion_identity_schedule_v1_modified": False,
        "frozen_temporal_foreground_modified": False,
        "TrackManager_algorithm_modified": False,
        "frozen_perception_parameters_modified": False,
        "motion_contract_modified": False,
        "authority_semantics_modified": False,
        "annex_used": False,
        "formal_preflight_rerun": False,
        "formal_v3_entry_created": False,
        "formal_v3_generation_started": False,
        "optimizer_step_executed": False,
        "training_started": False,
        "production_test_accessed": False,
        "blind_accessed": False,
    }
    frozen_now = {
        key: sha(ROOT / {
            "occlusion_constructor_v2_1.py":
                "authoritative_dataset/occlusion_constructor_v2_1.py",
            "occlusion_constructor_v2_2.py":
                "authoritative_dataset/occlusion_constructor_v2_2.py",
            "occlusion_identity_schedule_v1.py":
                "authoritative_dataset/occlusion_identity_schedule_v1.py",
            "temporal_foreground.py":
                "policy/dynamic/temporal_foreground.py",
            "track_manager.py": "policy/dynamic/track_manager.py",
            "motion_contract_v2_1.yaml":
                "configs/authoritative_dynamic_motion_contract_v2_1.yaml",
            "mixed_scene_map_profiles_v1.yaml":
                "configs/mixed_scene_map_profiles_v1.yaml",
        }[key])
        for key in (
            "occlusion_constructor_v2_1.py",
            "occlusion_constructor_v2_2.py",
            "occlusion_identity_schedule_v1.py",
            "temporal_foreground.py", "track_manager.py",
            "motion_contract_v2_1.yaml",
            "mixed_scene_map_profiles_v1.yaml",
        )
    }
    if any(
        frozen_now[key] != entry["frozen_hashes"][key]
        for key in frozen_now
    ):
        raise RuntimeError("frozen source changed during N1")
    write_new(REPORTS / "phase8jqv2_4n1_final_result.json", {
        "status": "FAIL",
        "primary_cause":
            "frozen_temporal_foreground_natural_observability",
        "natural_profile_repair": "FAIL",
        "gap_1_identity": "FAIL",
        "gap_2_identity": "NOT_RUN_BLOCKED_GAP1",
        "gap_3": "CONTRACT_REVIEW_REQUIRED",
        "gap_1_independent_maps": 0,
        "gap_2_independent_maps": 0,
        "annex_used": False,
        "frozen_hashes_verified": True,
        "invariants": invariants,
        "next_allowed_phase":
            "phase8jqv2_4_temporal_foreground_observability_contract_review",
    })
    print(json.dumps({
        "status": "FAIL",
        "primary_cause":
            "frozen_temporal_foreground_natural_observability",
        "gap1_witnesses": 0,
        "gap2_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
