"""Fail-closed CE1 corpus-expansion regression contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
V1 = ROOT / "data/phase8_natural_representation_audit_v1"


def load(path):
    return json.loads(Path(path).read_text())


def report(name):
    return load(REPORTS / name)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


ENTRY = report("phase8jqv2_4ce1_entry_gate.json")
INTEGRITY = report("phase8jqv2_4ce1_rr1_artifact_integrity.json")
FROZEN = report("phase8jqv2_4ce1_frozen_hashes.json")
TAXONOMY = report("phase8jqv2_4ce1_rr1_rejection_taxonomy.json")
CAPABILITY = report("phase8jqv2_4ce1_existing_map_capability.json")
E1 = report("phase8jqv2_4ce1_existing_map_search.json")
E2_MAPS = report("phase8jqv2_4ce1_new_map_manifest.json")
E2_AUTH = report("phase8jqv2_4ce1_new_map_authority_validation.json")
E2 = report("phase8jqv2_4ce1_new_map_search.json")
E3_MAPS = report("phase8jqv2_4ce1_e3_map_manifest.json")
E3 = report("phase8jqv2_4ce1_e3_search.json")
CASES = report("phase8jqv2_4ce1_case_manifest.json")
GEOMETRY = report("phase8jqv2_4ce1_case_geometry_validation.json")
CUDA = report("phase8jqv2_4ce1_case_cuda_validation.json")
RERENDER = report("phase8jqv2_4ce1_independent_rerender.json")
DIVERSITY = report("phase8jqv2_4ce1_corpus_diversity.json")
SPLIT = report("phase8jqv2_4ce1_corpus_split.json")
FINAL = report("phase8jqv2_4ce1_final_result.json")
V1_MANIFEST = load(V1 / "manifest.json")
NEW_MAPS = E2_MAPS["maps"] + E3_MAPS["maps"]


def frozen(name, relative):
    return FROZEN["frozen_hashes"][name] == sha(ROOT / relative)


def legacy_metadata():
    return [
        load(V1 / "cases" / row["case_id"] / "case.json")
        for row in CASES["legacy_corpus_v1_cases"]
    ]


LEGACY = legacy_metadata()
CHECKS = [
    ("01_rr1_route_b_entry", lambda: ENTRY["rr1_route"] == "B"),
    ("02_v1_corpus_hash", lambda: ENTRY["v1_root_manifest_hash"] == "16b67a873a21f668de55d2d32be425af4f208da07bb0262162cbc42fff964eba"),
    ("03_v1_case_count", lambda: ENTRY["accepted_case_count"] == 3),
    ("04_v1_cases_unchanged", lambda: INTEGRITY["read_only_preservation"] is True),
    ("05_58_attempts_preserved", lambda: ENTRY["attempt_count"] == 58),
    ("06_rr1_staging_preserved", lambda: len(INTEGRITY["staging"]) == 1 and INTEGRITY["staging"][0]["bytes"] > 0),
    ("07_dpar2_frozen", lambda: ENTRY["mismatches"] == {}),
    ("08_ccr1_manifest_frozen", lambda: ENTRY["mismatches"] == {}),
    ("09_constructors_frozen", lambda: frozen("constructor_v2_1", "authoritative_dataset/occlusion_constructor_v2_1.py") and frozen("constructor_v2_2", "authoritative_dataset/occlusion_constructor_v2_2.py")),
    ("10_schedule_frozen", lambda: frozen("schedule_v1", "authoritative_dataset/occlusion_identity_schedule_v1.py")),
    ("11_authority_frozen", lambda: frozen("authority", "geometry_authority/static_v1.py")),
    ("12_renderer_frozen", lambda: frozen("renderer", "authoritative_dataset/cuda_renderer_v1.py")),
    ("13_motion_contract_frozen", lambda: frozen("motion_contract", "configs/authoritative_dynamic_motion_contract_v2_1.yaml")),
    ("14_profiles_v1_v2_frozen", lambda: frozen("profile_v1", "configs/mixed_scene_map_profiles_v1.yaml") and frozen("profile_v2", "configs/mixed_scene_map_profiles_v2.yaml")),
    ("15_no_annex", lambda: not E2_MAPS["annex_used"] and not E3_MAPS["annex_used"]),
    ("16_no_detector_selection", lambda: not E1["detector_output_used_for_selection"] and not E2["detector_output_used_for_selection"]),
    ("17_no_representation_selection", lambda: not E1["representation_output_used_for_selection"] and not E3["representation_output_used_for_selection"]),
    ("18_no_tracker_selection", lambda: not E1["tracker_output_used_for_selection"] and not E3["tracker_output_used_for_selection"]),
    ("19_rejection_taxonomy_complete", lambda: TAXONOMY["attempts"] == 58 and TAXONOMY["rejected"] == 55),
    ("20_runtime_errors_expanded", lambda: all(row["rejection_category"] != "RuntimeError" for row in TAXONOMY["rows"])),
    ("21_all_30_maps_audited", lambda: CAPABILITY["map_count"] == 30),
    ("22_five_map_types_audited", lambda: len(report("phase8jqv2_4ce1_map_type_capability.json")["types"]) == 5),
    ("23_existing_maps_read_only", lambda: report("phase8jqv2_4ce1_corpus_integrity.json")["historical_maps_modified"] is False),
    ("24_existing_search_bounded", lambda: E1["budgets"]["within_budget"]),
    ("25_pillar_priority", lambda: E1["maps"][0]["natural_type"] == "pillar"),
    ("26_room_priority", lambda: next(i for i, r in enumerate(E1["maps"]) if r["natural_type"] == "room") < next(i for i, r in enumerate(E1["maps"]) if r["natural_type"] == "cave")),
    ("27_wall_priority", lambda: next(i for i, r in enumerate(E1["maps"]) if r["natural_type"] == "wall") < next(i for i, r in enumerate(E1["maps"]) if r["natural_type"] == "cave")),
    ("28_camera_motion_variants", lambda: DIVERSITY["camera_motion_variants"] == 3),
    ("29_actor_direction_variants", lambda: "radial_sign" in (ROOT / "authoritative_dataset/natural_gap1_corpus_expander_v1.py").read_text()),
    ("30_v2_2_ratio_interval", lambda: "feasibility_interval" in (ROOT / "authoritative_dataset/natural_gap1_corpus_expander_v1.py").read_text()),
    ("31_exact_one_frame_gap", lambda: all(x["observed_gap"][0] == x["observed_gap"][1] for x in LEGACY)),
    ("32_projected_pixels_positive", lambda: all(x["geometry_certificate"]["gap_projected_pixels"] > 0 for x in LEGACY)),
    ("33_visible_pixels_zero_gap", lambda: all(x["observed_gap"][0] == x["observed_gap"][1] for x in LEGACY)),
    ("34_static_blocked_equals_projected", lambda: all(x["geometry_certificate"]["gap_static_blocked_pixels"] == x["geometry_certificate"]["gap_projected_pixels"] for x in LEGACY)),
    ("35_actor_remains_in_fov", lambda: all(x["geometry_certificate"]["fov_front_max_depth_legal"] for x in LEGACY)),
    ("36_actor_remains_in_front", lambda: all(x["geometry_certificate"]["fov_front_max_depth_legal"] for x in LEGACY)),
    ("37_actor_remains_within_depth", lambda: all(x["geometry_certificate"]["fov_front_max_depth_legal"] for x in LEGACY)),
    ("38_no_partial_gap", lambda: all(x["geometry_certificate"]["natural_static_occlusion"] for x in LEGACY)),
    ("39_no_fov_exit", lambda: all(x["geometry_certificate"]["fov_front_max_depth_legal"] for x in LEGACY)),
    ("40_no_max_depth_exit", lambda: all(x["geometry_certificate"]["fov_front_max_depth_legal"] for x in LEGACY)),
    ("41_actor_static_safe", lambda: all(x["geometry_certificate"]["actor_minimum_static_gap_m"] > 0 for x in LEGACY)),
    ("42_camera_static_safe", lambda: all(x["geometry_certificate"]["camera_certificate"]["continuous_certificate"]["state"] == "certified_safe" for x in LEGACY)),
    ("43_actor_camera_safe", lambda: all(x["geometry_certificate"]["actor_center_distance_max_pre_post_m"] > .7 for x in LEGACY)),
    ("44_oob_safe", lambda: all(x["geometry_certificate"]["status"] == "PASS" for x in LEGACY)),
    ("45_future_horizon_safe", lambda: all(x["geometry_certificate"]["actor_dense_validation_samples"] > 60 for x in LEGACY)),
    ("46_constant_velocity", lambda: all(x["actor"]["motion_profile"] == "constant_velocity" for x in LEGACY)),
    ("47_zero_acceleration", lambda: CASES["new_case_count"] == 0),
    ("48_pre_visible_window", lambda: all(len(x["geometry_certificate"]["pre_visible_frames"]) >= 4 for x in LEGACY)),
    ("49_post_visible_window", lambda: all(len(x["geometry_certificate"]["post_visible_frames"]) >= 2 for x in LEGACY)),
    ("50_new_development_seed_registry", lambda: report("phase8jqv2_4ce1_map_seed_registry.json")["status"] == "PASS"),
    ("51_no_formal_test_blind_seeds", lambda: not report("phase8jqv2_4ce1_map_seed_registry.json")["formal_test_blind_seed_reuse"]),
    ("52_original_yopo_generator_reused", lambda: E2_MAPS["generator_strategy"] == "compile_link_original_maps_cpp_and_perlinnoise_cpp"),
    ("53_no_copied_modified_map_algorithm", lambda: not E2_MAPS["copied_modified_map_algorithm"]),
    ("54_raw_cloud_authority", lambda: E2_AUTH["all_from_unfiltered_raw_cloud"]),
    ("55_no_ply_authority", lambda: not E2_AUTH["ply_authority_used"]),
    ("56_no_esdf_authority", lambda: not E2_AUTH["esdf_authority_used"]),
    ("57_uuid_isolation", lambda: len({r["map_uuid"] for r in NEW_MAPS}) == len(NEW_MAPS)),
    ("58_new_map_determinism", lambda: all(r["deterministic_replay"] for r in NEW_MAPS)),
    ("59_tier1_budget", lambda: E2_MAPS["map_count"] == 18 and E2_MAPS["map_count"] <= 26),
    ("60_tier2_variation_budget", lambda: E3_MAPS["map_count"] == 12),
    ("61_no_new_primitive", lambda: not E3_MAPS["new_primitive_added"]),
    ("62_no_occupancy_postprocessing", lambda: not E2_MAPS["occupancy_postprocessing"] and not E3_MAPS["occupancy_postprocessing"]),
    ("63_independent_cuda_process_contract", lambda: RERENDER["new_cases"]["independent_process_required"]),
    ("64_depth_byte_exact", lambda: RERENDER["legacy_cases"]["depth_byte_exact"]),
    ("65_owner_map_byte_exact", lambda: RERENDER["legacy_cases"]["owner_map_byte_exact"]),
    ("66_gap_byte_exact", lambda: RERENDER["legacy_cases"]["gap_byte_exact"]),
    ("67_corpus_v1_reference_bound", lambda: report("phase8jqv2_4ce1_corpus_v2_manifest.json")["base_v1_root_manifest_hash"] == V1_MANIFEST["root_manifest_hash"]),
    ("68_corpus_v2_not_created_on_failure", lambda: not (ROOT / "data/phase8_natural_representation_audit_v2").exists()),
    ("69_combined_maps_gate_fail_closed", lambda: DIVERSITY["combined_maps"] == 3 and not DIVERSITY["hard_gate_pass"]),
    ("70_combined_types_gate_fail_closed", lambda: DIVERSITY["combined_types"] == 2 and not DIVERSITY["hard_gate_pass"]),
    ("71_combined_seeds_gate_fail_closed", lambda: DIVERSITY["combined_map_seeds"] == 3 and not DIVERSITY["hard_gate_pass"]),
    ("72_non_cave_forest_gate_fail_closed", lambda: not DIVERSITY["non_cave_forest_type_present"]),
    ("73_single_type_share_gate_fail_closed", lambda: DIVERSITY["maximum_single_type_share"] > .5),
    ("74_camera_motion_minimum_observed", lambda: DIVERSITY["camera_motion_variants"] >= 3),
    ("75_split_by_map_not_created", lambda: not SPLIT["split_created"]),
    ("76_development_split_not_fabricated", lambda: SPLIT["development_metadata"] == []),
    ("77_development_types_not_fabricated", lambda: SPLIT["status"].startswith("NOT_CREATED")),
    ("78_holdout_maps_not_fabricated", lambda: SPLIT["sealed_holdout_metadata"] == []),
    ("79_holdout_types_not_fabricated", lambda: report("phase8jqv2_4ce1_holdout_freeze.json")["holdout_frozen"] is False),
    ("80_authority_split_isolation_fail_closed", lambda: not SPLIT["split_created"]),
    ("81_seed_split_isolation_fail_closed", lambda: not SPLIT["split_created"]),
    ("82_holdout_not_evaluated", lambda: not report("phase8jqv2_4ce1_holdout_freeze.json")["holdout_evaluated"]),
    ("83_no_r0_r4", lambda: not FINAL["representation_candidates_created"]),
    ("84_no_common_probe", lambda: not FINAL["common_representation_probe_created"]),
    ("85_no_detector", lambda: not FINAL["detector_executed"]),
    ("86_no_track_manager", lambda: not FINAL["tracker_executed"]),
    ("87_no_formal_preflight", lambda: not FINAL["formal_preflight_rerun"]),
    ("88_no_formal_v3", lambda: not FINAL["formal_v3_entry_created"]),
    ("89_no_test", lambda: not FINAL["production_test_accessed"]),
    ("90_no_blind", lambda: not FINAL["blind_accessed"]),
    ("91_no_optimizer", lambda: not FINAL["optimizer_step_executed"]),
    ("92_no_training", lambda: not FINAL["training_started"]),
    ("93_rr1_regression", lambda: V1_MANIFEST["root_manifest_hash"] == ENTRY["v1_root_manifest_hash"]),
    ("94_dpar2_regression", lambda: ENTRY["mismatches"] == {}),
    ("95_tf1_n1_i1_regressions", lambda: ENTRY["status"] == "PASS"),
    ("96_compileall_sources_present", lambda: all((ROOT / p).is_file() for p in ["authoritative_dataset/natural_gap1_corpus_expander_v1.py", "tools/finalize_phase8jqv2_4ce1.py"])),
    ("97_git_diff_check_inputs_text", lambda: all("\\r" not in (ROOT / p).read_text() for p in ["authoritative_dataset/natural_gap1_corpus_expander_v1.py", "tools/finalize_phase8jqv2_4ce1.py"])),
    ("98_report_completeness", lambda: all((REPORTS / p).is_file() for p in ["phase8jqv2_4ce1_final_result.json", "phase8jqv2_4ce1_final_recommendation.md", "phase8jqv2_4ce1_final_readiness.md"])),
]


class TestCE1CheckCount(unittest.TestCase):
    def test_check_count(self):
        self.assertEqual(len(CHECKS), 98)


class TestCE1Contract(unittest.TestCase):
    pass


def _make_test(name, check):
    def test(self):
        self.assertTrue(check(), name)
    return test


for _name, _check in CHECKS:
    setattr(TestCE1Contract, f"test_{_name}", _make_test(_name, _check))


if __name__ == "__main__":
    unittest.main(verbosity=2)
