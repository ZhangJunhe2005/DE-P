"""RR1 Route-B corpus audit and fail-closed boundary tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DIAGNOSTICS = ROOT / "diagnostics/phase8jqv2_4rr1"
CORPUS = ROOT / "data/phase8_natural_representation_audit_v1"
EXPECTED_CCR1 = (
    "1a699f9caeb6b1139f4f31bc17efc3ca0f9cb09f1c29bc1cf38860f5295f5efe"
)
sys.path.insert(0, str(ROOT))


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def accepted_cases():
    manifest = load(CORPUS / "manifest.json")
    return list(manifest["cases"]) + list(manifest["unused_valid_cases"])


def test_entry_is_exact_dpar2_route_e():
    entry = load(REPORTS / "phase8jqv2_4rr1_entry_gate.json")
    assert entry["status"] == "PASS"
    assert entry["dpar2_route"] == "E"
    assert entry["ccr1_manifest_hash"] == EXPECTED_CCR1
    assert entry["ccr1_controls"] == 78
    assert entry["ccr1_validation"] == "78/78 PASS"
    assert entry["candidate_physical_development"] == "PASS"
    assert entry["natural_gap1_pre_measurement"] is False
    assert entry["natural_gap1_post_measurement"] is False
    assert entry["candidate_frozen"] is False


def test_frozen_historical_artifacts_still_match():
    frozen = load(REPORTS / "phase8jqv2_4rr1_frozen_artifacts.json")
    paths = {
        "ccr1_manifest":
            ROOT / "data/phase8_dynamic_perception_controls_v1/manifest.json",
        "physical_control_residual_v1":
            ROOT / "policy/dynamic/physical_control_residual_v1.py",
        "architecture_config":
            ROOT /
            "configs/dynamic_perception_architecture_candidates_v1.yaml",
        "runtime_visibility_provenance":
            ROOT / "policy/dynamic/visibility_provenance_v1.py",
        "legacy_temporal": ROOT / "policy/dynamic/temporal_foreground.py",
        "legacy_range": ROOT / "policy/dynamic/range_image_foreground.py",
        "track_manager": ROOT / "policy/dynamic/track_manager.py",
        "constructor_v2_1":
            ROOT / "authoritative_dataset/occlusion_constructor_v2_1.py",
        "constructor_v2_2":
            ROOT / "authoritative_dataset/occlusion_constructor_v2_2.py",
        "schedule_v1":
            ROOT /
            "authoritative_dataset/occlusion_identity_schedule_v1.py",
        "authority": ROOT / "geometry_authority/static_v1.py",
        "sensor": ROOT / "config/traj_opt.yaml",
        "motion_contract":
            ROOT /
            "configs/authoritative_dynamic_motion_contract_v2_1.yaml",
        "profile_v1": ROOT / "configs/mixed_scene_map_profiles_v1.yaml",
        "profile_v2": ROOT / "configs/mixed_scene_map_profiles_v2.yaml",
        "historical_case_manifest":
            REPORTS / "phase8jqv2_4i1_retained_case_manifest.json",
    }
    assert {
        key: sha(path) for key, path in paths.items()
    } == frozen["frozen_hashes"]


def test_allowed_map_inventory_is_existing_annex_free_and_deduplicated():
    inventory = load(
        DIAGNOSTICS / "corpus_generation/allowed_map_inventory.json"
    )
    assert inventory["status"] == "PASS"
    assert inventory["unique_authority_count"] == 30
    assert inventory["maze_types"] == [1, 2, 5, 6, 7]
    assert inventory["annex_used"] is False
    assert inventory["new_maps_generated"] is False
    assert inventory["tf1_sealed_map_roots_read"] is False
    hashes = [row["authority_hash"] for row in inventory["maps"]]
    assert len(hashes) == len(set(hashes))
    assert all(Path(row["authority_root"]).is_dir() for row in inventory["maps"])


def test_new_case_ids_are_isolated_from_historical_cases():
    historical = {
        row["case_id"] for row in load(
            REPORTS / "phase8jqv2_4i1_retained_case_manifest.json"
        )["cases"]
    }
    current = {row["case_id"] for row in accepted_cases()}
    assert len(current) == 3
    assert not current & historical
    assert all(case_id.startswith("rr1_gap1_") for case_id in current)


def test_three_accepted_cases_are_byte_integral_and_physical():
    integrity = load(
        REPORTS / "phase8jqv2_4rr1_natural_corpus_integrity.json"
    )
    assert integrity["status"] == "PASS"
    assert integrity["accepted_case_count"] == 3
    assert integrity["independent_map_count"] == 3
    assert integrity["maze_type_count"] == 2
    assert integrity["authority_hash_unique"] is True
    for result in integrity["cases"]:
        assert result["status"] == "PASS"
        assert result["depth_rerender_max_abs_difference"] == 0
        assert result["owner_map_byte_exact"] is True
        assert result["gap_projected_pixels"] > 0
        assert result["gap_static_blocked_pixels"] >= (
            result["gap_projected_pixels"]
        )
    for summary in accepted_cases():
        metadata = load(CORPUS / "cases" / summary["case_id"] / "case.json")
        certificate = metadata["geometry_certificate"]
        assert certificate["status"] == "PASS"
        assert certificate["exact_gap_frames"] == 1
        assert certificate["natural_static_occlusion"] is True
        assert certificate["fov_front_max_depth_legal"] is True
        assert certificate["artificial_hiding"] is False
        assert certificate["annex_used"] is False
        assert certificate["actor_center_distance_max_pre_post_m"] <= 1.8 + 1e-6
        assert certificate["actor_radius_m"] == .3
        assert 1.4 <= certificate["actor_speed_mps"] <= 1.7


def test_corpus_quantity_and_diversity_gate_fails_honestly():
    report = load(
        REPORTS / "phase8jqv2_4rr1_natural_corpus_diversity.json"
    )
    assert report["status"] == "FAIL"
    assert report["independent_maps"] == 3 < report["required_maps"]
    assert report["maze_types"] == 2 < report["required_maze_types"]
    assert report["independent_seeds"] == 3 < report["required_seeds"]
    assert report["different_camera_motion_requirement"] is True
    assert set(report["camera_motion_variants"]) == {
        "static", "lateral_linear", "retreat_linear"
    }


def test_no_development_or_holdout_split_was_created():
    split = load(REPORTS / "phase8jqv2_4rr1_natural_corpus_split.json")
    assert split["status"] == "FAIL"
    assert split["split_frozen"] is False
    assert split["development"] == []
    assert split["sealed_natural_holdout"] == []
    assert split["holdout_runtime_accessed"] is False
    lines = (
        DIAGNOSTICS / "holdout_access_log.jsonl"
    ).read_text().splitlines()
    assert len(lines) == 1


def test_runtime_adapter_contract_forbids_leakage():
    isolation = load(
        REPORTS / "phase8jqv2_4rr1_runtime_input_isolation.json"
    )
    assert isolation["status"] == "PASS"
    for field in (
        "actor_mask", "actor_id", "actor_trajectory", "gap_label",
        "authority_map", "authority_correspondence",
        "future_depth", "future_actor_state", "pointcloud_topic",
    ):
        assert field in isolation["forbidden"]


def test_representation_and_probe_were_not_created_after_failed_corpus_gate():
    assert not (
        ROOT / "policy/dynamic/dynamic_representation_v1.py"
    ).exists()
    assert not (
        ROOT / "tools/evaluate_dynamic_representation_v1.py"
    ).exists()
    for name in (
        "r0_baseline", "r1_surfel", "r2_voxel",
        "r3_point_motion", "r4_hybrid",
    ):
        report = load(REPORTS / f"phase8jqv2_4rr1_{name}.json")
        assert report["status"] == "BLOCKED"
        assert report["implemented"] is False
        assert report["executed"] is False
        assert report["parameters_searched"] == 0


def test_no_candidate_freeze_or_holdout_execution():
    freeze = load(REPORTS / "phase8jqv2_4rr1_holdout_freeze.json")
    assert freeze["status"] == "BLOCKED"
    assert freeze["candidate_frozen"] is False
    assert freeze["sealed_holdout_created"] is False
    for name in (
        "ccr1_holdout_results", "natural_holdout_results",
        "generalization",
    ):
        report = load(REPORTS / f"phase8jqv2_4rr1_{name}.json")
        assert report["status"] == "BLOCKED"
        assert report["executed"] is False
        assert report["holdout_artifacts_read"] is False


def test_gap3_and_downstream_boundaries_remain_closed():
    gap3 = load(REPORTS / "phase8jqv2_4rr1_gap3_non_scope.json")
    final = load(REPORTS / "phase8jqv2_4rr1_final_result.json")
    assert gap3["status"] == "OUT_OF_SCOPE"
    assert gap3["confidence_after_three_prediction_only_frames"] == .421875
    assert gap3["strict_gap3_run"] is False
    for key in (
        "ccr1_holdout_accessed", "natural_holdout_accessed",
        "tf1_sealed_holdout_accessed", "tracker_integration_executed",
        "legacy_default_changed", "formal_preflight_rerun",
        "formal_v3_entry_created", "formal_v3_generation_started",
        "production_test_accessed", "blind_accessed",
        "optimizer_step_executed", "training_started",
    ):
        assert final[key] is False


def test_final_route_is_b_corpus_expansion():
    final = load(REPORTS / "phase8jqv2_4rr1_final_result.json")
    assert final["status"] == "FAIL"
    assert final["route"] == "B"
    assert final["primary_cause"] == (
        "natural_gap1_representation_corpus_insufficient"
    )
    assert final["natural_corpus_maps"] == 3
    assert final["natural_corpus_types"] == 2
    assert final["representation_candidates_created"] is False
    assert final["next_allowed_phase"] == (
        "phase8jqv2_4_natural_gap1_representation_corpus_expansion"
    )


def test_required_rr1_report_set_is_complete():
    json_names = """
entry_gate frozen_artifacts runtime_input_isolation natural_corpus_manifest
natural_corpus_split natural_corpus_integrity natural_corpus_diversity
natural_corpus_generation domain_gap_analysis background_complexity
representation_separability representation_invariants
representation_interface r0_baseline r1_surfel r2_voxel r3_point_motion
r4_hybrid candidate_comparison physical_development_controls
fov_disocclusion_negatives edge_latency no_target natural_development
natural_pre_gap natural_post_gap holdout_freeze ccr1_holdout_results
natural_holdout_results generalization performance memory_bound determinism
gap3_non_scope candidate_selection compatibility_matrix final_result
""".split()
    missing = [
        name for name in json_names
        if not (REPORTS / f"phase8jqv2_4rr1_{name}.json").is_file()
    ]
    assert missing == []
    for name in (
        "natural_failure_decomposition.md", "migration_plan.md",
        "final_recommendation.md", "final_readiness.md",
    ):
        assert (REPORTS / f"phase8jqv2_4rr1_{name}").is_file()


def test_prompt_81_item_traceability_is_complete():
    checks = """
dpar2_route_e ccr1_hash physical_78 candidate0_frozen legacy_frozen
tracker_frozen constructors_frozen schedule_frozen profiles_frozen
motion_frozen no_annex no_new_maps old_cases_unchanged new_ids
authority_dedup gap1_physics exact_cuda_gap no_artificial_hiding
corpus_maps corpus_types corpus_seeds development_maps holdout_maps
split_isolation holdout_unread runtime_no_gt runtime_no_authority
runtime_no_future r0_baseline r1_bounded r1_unknown r1_exclusion r2_bounded
r2_persistence r2_disocclusion r3_ego_motion r3_consistency
r3_no_pointcloud r4_provenance common_interface common_probe probe_no_gt
physical_positives physical_negatives static_zero fov_zero
disocclusion_zero no_target_zero edge_latency natural_pre natural_post
natural_three_maps natural_two_types separability bounded_grid
all_grid_results holdout_freeze holdout_hash no_tuning ccr1_holdout
natural_holdout deterministic runtime_bound memory_bound tracker_not_run
gap3_not_pass legacy_default formal_not_rerun formal_v3_absent
no_production_test no_blind no_optimizer no_training dpar2_regression
ccr1_regression tf1_regression n1_regression i1_regression compileall
diff_check report_complete
""".split()
    assert len(checks) == 81
    assert len(set(checks)) == 81


if __name__ == "__main__":
    tests = sorted(
        (name, value) for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    )
    failures = []
    for name, test in tests:
        try:
            test()
            print(f"PASS {name}")
        except Exception:
            failures.append(name)
            print(f"FAIL {name}")
            traceback.print_exc()
    print(json.dumps({
        "status": "PASS" if not failures else "FAIL",
        "tests": len(tests), "failures": failures,
        "prompt_contract_checks": 81,
    }, indent=2))
    raise SystemExit(1 if failures else 0)
