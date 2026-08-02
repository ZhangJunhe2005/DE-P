"""DPAR1 Stage-A control audit and fail-closed regression tests."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def load(name):
    return json.loads((REPORTS / name).read_text())


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


class Phase8JQ24DPAR1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.entry = load("phase8jqv2_4dpar1_entry_gate.json")
        cls.control = load("phase8jqv2_4dpar1_control_contract_audit.json")
        cls.fov = load(
            "phase8jqv2_4dpar1_fov_boundary_fixture_validation.json"
        )
        cls.final = load("phase8jqv2_4dpar1_final_result.json")
        cls.constraints = load(
            "phase8jqv2_4dpar1_final_constraints.json"
        )
        cls.tf1_entry = load("phase8jqv2_4tf1_entry_gate.json")

    def test_01_tf1_route_d_entry(self):
        self.assertEqual(self.entry["status"], "PASS")
        self.assertEqual(self.entry["tf1_route"], "D")
        self.assertEqual(
            self.entry["tf1_primary_cause"],
            "temporal_foreground_contract_architecture_limit",
        )

    def test_02_to_08_frozen_sources(self):
        paths = {
            "temporal_foreground.py":
                "policy/dynamic/temporal_foreground.py",
            "range_image_foreground.py":
                "policy/dynamic/range_image_foreground.py",
            "track_manager.py": "policy/dynamic/track_manager.py",
            "occlusion_constructor_v2_1.py":
                "authoritative_dataset/occlusion_constructor_v2_1.py",
            "occlusion_constructor_v2_2.py":
                "authoritative_dataset/occlusion_constructor_v2_2.py",
            "occlusion_identity_schedule_v1.py":
                "authoritative_dataset/occlusion_identity_schedule_v1.py",
            "mixed_scene_map_profiles_v1.yaml":
                "configs/mixed_scene_map_profiles_v1.yaml",
            "mixed_scene_map_profiles_v2.yaml":
                "configs/mixed_scene_map_profiles_v2.yaml",
            "motion_contract_v2_1.yaml":
                "configs/authoritative_dynamic_motion_contract_v2_1.yaml",
        }
        for key, path in paths.items():
            with self.subTest(key=key):
                self.assertEqual(
                    digest(path), self.tf1_entry["frozen_hashes"][key]
                )

    def test_09_to_10_no_annex_or_maps(self):
        self.assertFalse(self.entry["annex_used"])
        self.assertFalse(self.entry["new_maps_generated"])

    def test_11_small_projection_physical_audit(self):
        value = self.control["small_projection"]
        self.assertEqual(value["classification"], "fixture_semantics_invalid")
        self.assertIsNone(value["actor_physical_radius_m"])
        self.assertEqual(value["single_frame_projected_area_pixels"], 16)
        self.assertEqual(value["patch_depth_m"], 4.0)
        self.assertFalse(value["inside_detection_envelope"])
        self.assertFalse(value["inside_formal_radius_range"])
        minimum = value["formal_projection_at_detection_boundary"]["0.2"]
        expected = 2*80*.2/math.sqrt(1.8**2-.2**2)
        self.assertAlmostEqual(minimum["diameter_pixels"], expected, places=12)

    def test_12_fov_negative_semantic_audit(self):
        self.assertEqual(self.fov["classification"], "fixture_semantics_invalid")
        self.assertFalse(self.fov["contains_dynamic_actor"])
        self.assertEqual(self.fov["camera_translation_max_m"], 0)
        self.assertEqual(self.fov["camera_yaw_delta_max_rad"], 0)
        self.assertEqual(self.fov["depth_changed_pixels_at_transition"], 40)
        self.assertEqual(self.fov["newly_valid_pixels_from_depth_validity"], 0)
        self.assertEqual(self.fov["newly_invalid_pixels_from_depth_validity"], 0)
        self.assertFalse(self.fov["hard_negative_semantics_verified"])

    def test_13_control_contract_version(self):
        self.assertEqual(
            self.control["control_contract_version"],
            "dynamic_perception_control_contract_audit_v1",
        )
        self.assertEqual(self.control["status"], "FAIL")
        self.assertFalse(self.control["architecture_design_allowed"])

    def test_14_to_23_visibility_stage_correctly_blocked(self):
        for suffix in (
            "visibility_provenance_validation",
            "warp_validity_validation",
            "fov_boundary_stress",
        ):
            with self.subTest(suffix=suffix):
                report = load(f"phase8jqv2_4dpar1_{suffix}.json")
                self.assertEqual(
                    report["status"], "NOT_RUN_CONTROL_GATE_FAILED"
                )
                self.assertFalse(report["result_claimed"])
        for path in (
            "policy/dynamic/visibility_provenance_v1.py",
            "policy/dynamic/dynamic_measurement_v3.py",
        ):
            self.assertFalse((ROOT / path).exists())

    def test_24_to_31_candidate_a_b_correctly_blocked(self):
        for suffix in (
            "candidate_a_development", "candidate_b_development",
            "tracklet_validation", "small_projection_validation",
        ):
            report = load(f"phase8jqv2_4dpar1_{suffix}.json")
            self.assertEqual(
                report["status"], "NOT_RUN_CONTROL_GATE_FAILED"
            )
        for path in (
            "policy/dynamic/visibility_aware_residual_v1.py",
            "policy/dynamic/causal_depth_track_before_detect_v1.py",
        ):
            self.assertFalse((ROOT / path).exists())
        self.assertEqual(self.entry["official_dynamic_sensor_source"], "depth")
        self.assertFalse(self.entry["pointcloud_sensor_enabled"])

    def test_32_to_38_candidate_c_correctly_blocked(self):
        for suffix in (
            "candidate_c_development", "reacquisition_validation",
            "duplicate_suppression",
        ):
            report = load(f"phase8jqv2_4dpar1_{suffix}.json")
            self.assertEqual(
                report["status"], "NOT_RUN_CONTROL_GATE_FAILED"
            )
        self.assertFalse((
            ROOT / "policy/dynamic/dual_path_dynamic_perception_v1.py"
        ).exists())
        interface = load("phase8jqv2_4dpar1_interface_spec.json")
        self.assertFalse(interface["interface_created"])
        self.assertFalse(interface["track_manager_interface_changed"])

    def test_39_to_49_regressions_not_overclaimed(self):
        for suffix in (
            "static_validation", "no_target_validation",
            "ordinary_dynamic_regression", "natural_measurement_validation",
        ):
            report = load(f"phase8jqv2_4dpar1_{suffix}.json")
            self.assertEqual(
                report["status"], "NOT_RUN_CONTROL_GATE_FAILED"
            )
            self.assertFalse(report["result_claimed"])

    def test_50_to_54_natural_gate_not_overclaimed(self):
        report = load(
            "phase8jqv2_4dpar1_natural_measurement_validation.json"
        )
        self.assertEqual(report["status"], "NOT_RUN_CONTROL_GATE_FAILED")
        selection = load("phase8jqv2_4dpar1_candidate_selection.json")
        self.assertEqual(selection["candidates_evaluated"], 0)
        self.assertIsNone(selection["selected_candidate"])
        self.assertFalse(selection["candidate_selected"])

    def test_55_to_56_holdout_sealed(self):
        report = load("phase8jqv2_4dpar1_holdout_freeze.json")
        self.assertEqual(
            report["status"], "NOT_RUN_CONTROL_GATE_FAILED"
        )
        self.assertEqual(report["selected_for_holdout"], [])
        self.assertFalse(report["sealed_holdout_opened"])
        self.assertTrue(report["no_tuning_after_holdout"])
        self.assertFalse(self.entry["holdout_accessed"])

    def test_57_to_62_tracker_integration_blocked(self):
        integration = load(
            "phase8jqv2_4dpar1_tracker_integration_smoke.json"
        )
        self.assertEqual(
            integration["status"], "NOT_RUN_CONTROL_GATE_FAILED"
        )
        self.assertFalse(integration["adapter_created"])
        self.assertFalse(integration["track_manager_modified"])
        for suffix in ("gap1_identity", "gap2_identity"):
            report = load(f"phase8jqv2_4dpar1_{suffix}.json")
            self.assertFalse(report["result_claimed"])

    def test_63_gap3_not_claimed_pass(self):
        report = load("phase8jqv2_4dpar1_gap3_non_scope.json")
        self.assertEqual(report["confidence_after_three_misses"], .421875)
        self.assertFalse(report["strict_dynamic_through_gap3_possible"])
        self.assertFalse(report["gap3_pass_claimed"])

    def test_64_to_66_candidate_quality_gates_blocked(self):
        for suffix in ("determinism", "performance", "memory_bound"):
            report = load(f"phase8jqv2_4dpar1_{suffix}.json")
            self.assertEqual(
                report["status"], "NOT_RUN_CONTROL_GATE_FAILED"
            )
            self.assertFalse(report["result_claimed"])

    def test_67_legacy_default_unchanged(self):
        self.assertFalse(self.final["legacy_default_changed"])
        self.assertFalse(self.final["architecture_prototype_created"])

    def test_68_to_74_prohibited_actions_absent(self):
        self.assertFalse(self.final["formal_preflight_rerun"])
        self.assertFalse(self.entry["formal_v3_entry_created"])
        self.assertFalse(self.final["formal_generation_started"])
        self.assertFalse(self.final["test_accessed"])
        self.assertFalse(self.final["blind_accessed"])
        self.assertFalse(self.final["optimizer_step_executed"])
        self.assertFalse(self.final["training_started"])

    def test_75_to_77_historical_regression_and_hashes(self):
        self.assertFalse(self.final["tf1_candidates_modified"])
        self.assertFalse(self.final["tracker_modified"])
        self.assertTrue(self.constraints["status"] == "PASS")
        self.assertEqual(
            self.final["next_allowed_phase"],
            "phase8jqv2_4_dynamic_perception_control_contract_repair",
        )

    def test_78_fail_closed_route_b(self):
        self.assertEqual(self.final["status"], "FAIL")
        self.assertEqual(
            self.final["primary_cause"],
            "dynamic_perception_control_contract_invalid",
        )
        selection = load("phase8jqv2_4dpar1_candidate_selection.json")
        self.assertEqual(selection["decision_route"], "B")

    def test_79_report_completeness(self):
        suffixes = [
            "entry_gate.json", "evaluation_split.json",
            "control_contract_audit.json",
            "small_projection_contract.md",
            "fov_boundary_fixture_validation.json",
            "architecture_invariants.json", "architecture_review.md",
            "candidate_specs.json", "interface_spec.json",
            "visibility_provenance_validation.json",
            "warp_validity_validation.json", "fov_boundary_stress.json",
            "candidate_a_development.json",
            "candidate_b_development.json", "tracklet_validation.json",
            "small_projection_validation.json",
            "candidate_c_development.json",
            "reacquisition_validation.json",
            "duplicate_suppression.json", "static_validation.json",
            "no_target_validation.json",
            "ordinary_dynamic_regression.json",
            "natural_measurement_validation.json",
            "holdout_freeze.json", "holdout_results.json",
            "generalization.json", "tracker_integration_smoke.json",
            "gap1_identity.json", "gap2_identity.json",
            "gap3_non_scope.json", "performance.json",
            "memory_bound.json", "determinism.json",
            "candidate_selection.json", "compatibility_matrix.json",
            "migration_plan.md", "final_result.json",
            "final_recommendation.md", "final_readiness.md",
        ]
        missing = [
            suffix for suffix in suffixes
            if not (REPORTS / f"phase8jqv2_4dpar1_{suffix}").is_file()
        ]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
