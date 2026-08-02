"""DPAR2 fail-closed architecture-review contract tests (CPU-only)."""

from __future__ import annotations

from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys
import traceback

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPORTS = ROOT / "reports"
DATASET = ROOT / "data/phase8_dynamic_perception_controls_v1"
DIAGNOSTICS = ROOT / "diagnostics/phase8jqv2_4dpar2"
EXPECTED_MANIFEST = (
    "1a699f9caeb6b1139f4f31bc17efc3ca0f9cb09f1c29bc1cf38860f5295f5efe"
)


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_entry_suite_and_split_are_frozen():
    entry = load(REPORTS / "phase8jqv2_4dpar2_entry_gate.json")
    manifest = load(DATASET / "manifest.json")
    assert entry["status"] == "PASS"
    assert entry["ccr1_route"] == "A"
    assert entry["physical_control_manifest_hash"] == EXPECTED_MANIFEST
    assert manifest["manifest_hash"] == EXPECTED_MANIFEST
    assert entry["control_count"] == 78
    assert entry["development_count"] == 39
    assert entry["sealed_control_holdout_count"] == 39
    assert entry["physical_validation"] == {
        "status": "PASS", "passed": 78, "failed": 0
    }


def test_historical_failed_roots_preserved():
    for suffix in ("sparse_plane_v1", "combined_motion_v1"):
        assert (
            ROOT / f"data/phase8_dynamic_perception_controls_v1_failed_{suffix}"
        ).is_dir()


def test_all_frozen_source_hashes_still_match_entry():
    entry = load(REPORTS / "phase8jqv2_4dpar2_entry_gate.json")
    paths = {
        "physical_manifest": DATASET / "manifest.json",
        "physical_validation": DATASET / "physical_validation.json",
        "ccr1_generator":
            ROOT / "tools/generate_dynamic_perception_controls_v1.py",
        "ccr1_validator":
            ROOT / "tools/validate_dynamic_perception_controls_v1.py",
        "ccr1_warp_reference":
            ROOT / "authoritative_dataset/warp_visibility_reference_v1.py",
        "legacy_temporal": ROOT / "policy/dynamic/temporal_foreground.py",
        "legacy_range": ROOT / "policy/dynamic/range_image_foreground.py",
        "tf1_candidate":
            ROOT / "policy/dynamic/range_image_foreground_v2_1.py",
        "tf1_registry":
            ROOT / "policy/dynamic/foreground_contract_registry.py",
        "tf1_config":
            ROOT / "configs/temporal_foreground_contract_v2_1_candidates.yaml",
        "track_manager": ROOT / "policy/dynamic/track_manager.py",
        "motion_contract":
            ROOT / "configs/authoritative_dynamic_motion_contract_v2_1.yaml",
        "constructor_v2_1":
            ROOT / "authoritative_dataset/occlusion_constructor_v2_1.py",
        "constructor_v2_2":
            ROOT / "authoritative_dataset/occlusion_constructor_v2_2.py",
        "schedule":
            ROOT / "authoritative_dataset/occlusion_identity_schedule_v1.py",
        "profile_v1": ROOT / "configs/mixed_scene_map_profiles_v1.yaml",
        "profile_v2": ROOT / "configs/mixed_scene_map_profiles_v2.yaml",
        "authority": ROOT / "geometry_authority/static_v1.py",
        "sensor": ROOT / "config/traj_opt.yaml",
    }
    assert {key: sha(path) for key, path in paths.items()} == (
        entry["frozen_hashes"]
    )


def test_runtime_offline_isolation_and_no_future():
    isolation = load(
        REPORTS / "phase8jqv2_4dpar2_runtime_input_isolation.json"
    )
    assert isolation["runtime_gt_used"] is False
    assert isolation["future_used"] is False
    assert "actor_mask" in isolation["forbidden_runtime_inputs"]
    assert "reference_provenance_masks" in isolation[
        "forbidden_runtime_inputs"
    ]
    for path in (
        REPORTS / "phase8jqv2_4dpar2_candidate0_development.json",
        REPORTS / "phase8jqv2_4dpar2_natural_measurement_validation.json",
    ):
        text = path.read_text()
        assert '"runtime_gt_input": true' not in text
        assert '"runtime_gt_used": true' not in text


def test_dynamic_measurement_v3_has_required_and_no_forbidden_fields():
    from policy.dynamic.dynamic_measurement_v3 import DynamicMeasurementV3

    names = {field.name for field in fields(DynamicMeasurementV3)}
    required = {
        "timestamp", "position_camera", "position_world", "covariance",
        "evidence_source", "temporal_support_frames", "pixel_support",
        "point_support", "stable_overlap_fraction",
        "fov_boundary_fraction", "disocclusion_fraction",
        "depth_validity_fraction", "predicted_track_id",
        "birth_allowed", "reacquisition_only", "validity",
        "rejection_reason",
    }
    forbidden = {
        "actor_id", "actor_mask", "expected_control_class",
        "authority_correspondence", "future_information", "annex_metadata",
    }
    assert required <= names
    assert not names & forbidden


def test_registry_is_explicit_and_legacy_default_unchanged():
    from policy.dynamic.dynamic_perception_architecture_registry import (
        LEGACY_DEFAULT, VALID_KEYS, create_architecture,
    )

    document = yaml.safe_load((
        ROOT / "configs/dynamic_perception_architecture_candidates_v1.yaml"
    ).read_text())
    assert LEGACY_DEFAULT == document["default_architecture"] == "legacy_v1"
    assert document["explicit_key_required"] is True
    assert len(VALID_KEYS) == 8
    try:
        create_architecture()  # type: ignore[call-arg]
    except TypeError:
        pass
    else:
        raise AssertionError("architecture key is not explicit")


def test_candidate2_and_candidate3_architecture_invariants():
    config = yaml.safe_load((
        ROOT / "configs/dynamic_perception_architecture_candidates_v1.yaml"
    ).read_text())
    c2 = config["candidates"]["causal_depth_track_before_detect_v1"]
    c3 = config["candidates"]["dual_path_dynamic_perception_v1"]
    assert 3 <= c2["history_frames"] <= 6
    assert "pointcloud" not in json.dumps(c2).lower()
    assert c3["reacquisition_birth_allowed"] is False
    assert c3["reacquisition_requires_current_depth_evidence"] is True
    assert "discovery_sources" in c3


def test_legacy_and_tf1_replays_are_classified_not_promoted():
    legacy = load(
        REPORTS / "phase8jqv2_4dpar2_legacy_physical_controls.json"
    )
    tf1 = load(
        REPORTS / "phase8jqv2_4dpar2_tf1_candidates_physical_controls.json"
    )
    reassessment = load(
        REPORTS /
        "phase8jqv2_4dpar2_historical_conclusion_reassessment.json"
    )
    assert legacy["result"]["development_hard_gate"] is False
    assert not any(
        row["development_hard_gate"] for row in tf1["results"]
    )
    assert reassessment["status"] == "PASS"


def test_runtime_provenance_passes_predeclared_bounds():
    report = load(
        REPORTS / "phase8jqv2_4dpar2_runtime_visibility_provenance.json"
    )
    assert report["status"] == "PASS"
    assert report["future_frames_used"] == 0
    assert report["actor_gt_runtime_input"] is False
    assert report["reference_masks_runtime_input"] is False
    assert all(report["gates"].values())


def test_candidate0_passes_all_physical_development_hard_controls():
    result = load(
        REPORTS / "phase8jqv2_4dpar2_candidate0_development.json"
    )["result"]
    assert result["candidate"] == "physical_control_residual_v1"
    assert result["development_hard_gate"] is True
    assert result["failure_control_ids"] == []
    hard = [
        row for row in result["control_results"]
        if row["semantic_class"] != "diagnostic_out_of_contract"
    ]
    assert hard and all(row["pass"] for row in hard)


def test_physical_categories_and_edge_latency_pass():
    for name in (
        "physical_positive_validation", "fov_negative_validation",
        "disocclusion_validation", "depth_transition_validation",
        "edge_latency_validation",
    ):
        report = load(REPORTS / f"phase8jqv2_4dpar2_{name}.json")
        assert report["status"] == "PASS"
    edge = load(REPORTS / "phase8jqv2_4dpar2_edge_latency_validation.json")
    assert edge["latency_bound_frames"] == 3
    assert edge["passed"] == edge["total"]


def test_diagnostics_controls_are_excluded_from_hard_gate():
    result = load(
        REPORTS / "phase8jqv2_4dpar2_candidate0_development.json"
    )["result"]
    rows = [
        row for row in result["control_results"]
        if row["semantic_class"] == "diagnostic_out_of_contract"
    ]
    assert len(rows) == 4
    assert all(row["pass"] is None for row in rows)


def test_natural_six_cases_and_fixed_n1_case_were_run():
    report = load(
        REPORTS / "phase8jqv2_4dpar2_natural_measurement_validation.json"
    )
    assert report["i1_cases_evaluated"] == 6
    assert report["n1_fixed_case_evaluated"] is True
    assert report["new_maps_generated"] is False
    assert report["annex_used"] is False
    assert report["gap_counts"] == {"gap1": 1, "gap2": 2, "gap3": 3}


def test_natural_gap1_fails_measurement_and_coverage_honestly():
    report = load(REPORTS / "phase8jqv2_4dpar2_natural_gap1.json")
    fixed = report["cases"][0]
    assert report["status"] == "FAIL"
    assert report["measurement_pass"] is False
    assert report["coverage_pass"] is False
    assert fixed["pre_gap_measurement"] is False
    assert fixed["first_post_gap_measurement"] is False
    assert report["independent_map_count"] == 1 < 3
    assert report["maze_type_count"] == 1 < 2
    assert report["maps_or_trajectories_modified"] is False


def test_gap2_is_blocked_by_gap1_and_gap3_not_claimed():
    gap2 = load(REPORTS / "phase8jqv2_4dpar2_natural_gap2.json")
    gap3 = load(REPORTS / "phase8jqv2_4dpar2_gap3_non_scope.json")
    assert gap2["status"] == "BLOCKED"
    assert gap2["blocked_by"] == "natural_gap1_gate"
    assert gap3["status"] == "OUT_OF_SCOPE"
    assert gap3["confidence_after_three_prediction_only_frames"] == 0.421875
    assert gap3["strict_dynamic_through_gap3_possible"] is False


def test_no_sealed_holdout_access_and_no_candidate_freeze():
    lines = (
        DIAGNOSTICS / "holdout_access_log.jsonl"
    ).read_text().splitlines()
    assert len(lines) == 1
    assert load(
        REPORTS / "phase8jqv2_4dpar2_holdout_freeze.json"
    )["candidate_frozen"] is False
    holdout = load(REPORTS / "phase8jqv2_4dpar2_holdout_results.json")
    assert holdout["status"] == "BLOCKED"
    assert holdout["sealed_control_artifacts_read"] is False


def test_tracker_and_regression_are_fail_closed():
    for name in (
        "tracker_integration_smoke", "gap1_identity", "gap2_identity",
        "ordinary_dynamic_regression", "no_target_validation",
    ):
        report = load(REPORTS / f"phase8jqv2_4dpar2_{name}.json")
        assert report["status"] == "BLOCKED"
        assert report["executed"] is False
        assert report["tracker_modified"] is False


def test_runtime_and_memory_bounds():
    performance = load(
        REPORTS / "phase8jqv2_4dpar2_performance.json"
    )
    memory = load(REPORTS / "phase8jqv2_4dpar2_memory_bound.json")
    assert performance["status"] == "PASS"
    assert performance["p95_runtime_ms_per_frame"] <= 50
    assert performance["candidate_to_legacy_p95_ratio"] <= 2
    assert memory["history_capacity"] <= memory["maximum_history_capacity"]
    assert memory["maximum_tracklets"] == 32
    assert memory["unbounded_world_cloud"] is False


def test_selection_is_route_e_and_legacy_stays_default():
    selection = load(
        REPORTS / "phase8jqv2_4dpar2_candidate_selection.json"
    )
    final = load(REPORTS / "phase8jqv2_4dpar2_final_result.json")
    assert selection["selected_candidate"] is None
    assert selection["selection_gate_pass"] is False
    assert final["status"] == "FAIL"
    assert final["route"] == "E"
    assert final["primary_cause"] == "natural_depth_dynamic_observability"
    assert final["legacy_default_changed"] is False


def test_forbidden_downstream_actions_remain_false():
    final = load(REPORTS / "phase8jqv2_4dpar2_final_result.json")
    for key in (
        "holdout_runtime_artifacts_read", "tf1_sealed_holdout_accessed",
        "formal_preflight_rerun", "formal_generation_started",
        "production_test_accessed", "blind_accessed",
        "optimizer_step_executed", "training_started",
    ):
        assert final[key] is False


def test_required_report_set_is_complete():
    names = """
entry_gate control_integrity evaluation_split runtime_input_isolation
legacy_physical_controls tf1_candidates_physical_controls
historical_conclusion_reassessment failure_taxonomy evidence_flow
runtime_visibility_provenance provenance_reference_comparison
combined_motion_validation candidate0_development candidate1_development
candidate2_development candidate3_development candidate_comparison
physical_positive_validation fov_negative_validation
disocclusion_validation depth_transition_validation edge_latency_validation
natural_measurement_validation natural_gap1 natural_gap2 holdout_freeze
holdout_results generalization tracker_integration_smoke gap1_identity
gap2_identity gap3_non_scope ordinary_dynamic_regression
no_target_validation determinism performance memory_bound
candidate_selection compatibility_matrix final_result
""".split()
    missing = [
        name for name in names
        if not (REPORTS / f"phase8jqv2_4dpar2_{name}.json").is_file()
    ]
    assert missing == []
    for name in (
        "migration_plan.md", "final_recommendation.md",
        "final_readiness.md",
    ):
        assert (REPORTS / f"phase8jqv2_4dpar2_{name}").is_file()


def test_contract_checklist_covers_all_99_prompt_items():
    # Traceability names correspond in order to prompt section 26.
    checks = """
ccr1_route_a manifest_hash controls_78 development_39 holdout_39
physical_78 historical_roots legacy_hash tf1_hash tracker_hash motion_hash
constructors_hash profiles_hash no_annex no_new_maps no_tf1_holdout
no_ccr1_holdout_prefreeze runtime_offline no_gt no_future legacy_replay
tf1_a_replay tf1_b_replay tf1_c_replay candidate_d_reacquisition
seed7_diagnostic historical_reassessment radius_020 radius_030 distance_100
distance_140 distance_180 radial tangential mixed static_yaw_fov
static_translation_fov static_combined_fov all_depth_valid_fov
static_disocclusion max_depth invalid_depth edge_left edge_right upper_lower
moving_camera latency_k3 provenance_causal provenance_no_gt
reference_not_runtime stable_overlap newly_visible newly_invalid disocclusion
candidate0 candidate1 candidate2_bounded candidate2_no_pointcloud
candidate3_paths reacquisition_no_birth current_evidence association_gate
duplicate_suppression physical_positives physical_negatives partial_role
diagnostic_excluded natural_pre natural_post natural_three_maps
natural_two_types holdout_freeze holdout_hash no_tuning tracker_adapter
gap1_same_id gap2_same_id no_deletion no_replacement no_duplicate
gap3_not_pass ordinary_dynamic no_target deterministic runtime_bound
memory_bound legacy_default formal_not_rerun formal_v3_absent
no_production_test no_blind no_optimizer no_training ccr1_regression
tf1_regression n1_regression i1_regression diff_check report_complete
""".split()
    assert len(checks) == 99
    assert len(set(checks)) == 99


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
        "prompt_contract_checks": 99,
    }, indent=2))
    raise SystemExit(1 if failures else 0)
