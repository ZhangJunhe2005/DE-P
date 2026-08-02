"""TF1 contract-review regression and fail-closed gate tests (stdlib only)."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def load(name):
    return json.loads((REPORTS / name).read_text())


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


class Phase8JQ24TF1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.entry = load("phase8jqv2_4tf1_entry_gate.json")
        cls.baseline = load("phase8jqv2_4tf1_baseline_replay.json")
        cls.development = load(
            "phase8jqv2_4tf1_candidate_development_results.json"
        )
        cls.selection = load("phase8jqv2_4tf1_candidate_selection.json")
        cls.split = load("phase8jqv2_4tf1_evaluation_split.json")
        cls.final = load("phase8jqv2_4tf1_final_result.json")
        cls.config = yaml.safe_load((
            ROOT
            / "configs/temporal_foreground_contract_v2_1_candidates.yaml"
        ).read_text())

    def test_01_n1_entry_state(self):
        self.assertEqual(self.entry["status"], "PASS")
        self.assertEqual(self.entry["n1_status"], "FAIL")

    def test_02_to_10_frozen_hashes(self):
        checks = [
            ("temporal_foreground.py", "policy/dynamic/temporal_foreground.py"),
            (
                "range_image_foreground.py",
                "policy/dynamic/range_image_foreground.py",
            ),
            ("track_manager.py", "policy/dynamic/track_manager.py"),
            (
                "occlusion_constructor_v2_1.py",
                "authoritative_dataset/occlusion_constructor_v2_1.py",
            ),
            (
                "occlusion_constructor_v2_2.py",
                "authoritative_dataset/occlusion_constructor_v2_2.py",
            ),
            (
                "occlusion_identity_schedule_v1.py",
                "authoritative_dataset/occlusion_identity_schedule_v1.py",
            ),
            (
                "motion_contract_v2_1.yaml",
                "configs/authoritative_dynamic_motion_contract_v2_1.yaml",
            ),
            ("sensor_traj_opt.yaml", "config/traj_opt.yaml"),
            (
                "mixed_scene_map_profiles_v1.yaml",
                "configs/mixed_scene_map_profiles_v1.yaml",
            ),
            (
                "mixed_scene_map_profiles_v2.yaml",
                "configs/mixed_scene_map_profiles_v2.yaml",
            ),
        ]
        for key, path in checks:
            with self.subTest(key=key):
                self.assertEqual(
                    digest(path), self.entry["frozen_hashes"][key]
                )

    def test_11_to_16_baseline_exact_pixel_evidence(self):
        self.assertEqual(self.baseline["status"], "PASS")
        fixed = self.baseline["fixed_case"]
        self.assertEqual(fixed["mismatches"], {})
        self.assertEqual(fixed["numerical_tolerance"], 0)
        for field, expected in {
            "visible": 383, "positive_residual": 174,
            "two_closer_histories": 8,
            "union_seed_after_static_exclusion": 7,
            "component_count": 0,
        }.items():
            with self.subTest(field=field):
                self.assertEqual(fixed["actual"][field], expected)

    def test_17_registry_requires_explicit_key(self):
        from policy.dynamic.foreground_contract_registry import (
            create_foreground_contract,
        )
        parameter = inspect.signature(
            create_foreground_contract
        ).parameters["key"]
        self.assertIs(parameter.default, inspect.Parameter.empty)

    def test_18_legacy_default_unchanged(self):
        self.assertEqual(self.config["legacy_default"], "legacy_v1")
        self.assertFalse(self.config["candidate_selected"])
        self.assertIsNone(self.config["selected_candidate"])

    def test_19_to_22_candidate_families(self):
        from policy.dynamic.foreground_contract_registry import (
            create_foreground_contract,
        )
        expected = {
            "candidate_a": "causal_multiframe_evidence",
            "candidate_b": "component_first_residual",
            "candidate_c": "occupied_freed_dual_channel",
            "candidate_d": "existing_track_reacquisition_only",
        }
        for candidate, family in expected.items():
            with self.subTest(candidate=candidate):
                self.assertEqual(
                    self.config["candidates"][candidate]["family"], family
                )
        candidate_d = create_foreground_contract(
            "candidate_d", None,
            self.config["candidates"]["candidate_d"]["grid"][0],
        )
        result = candidate_d.evaluate(
            [[True, True], [True, True]],
            [[True, True], [True, True]],
            track_alive=True,
        )
        self.assertTrue(result["accepted"])
        self.assertFalse(result["new_track_birth_allowed"])

    def test_23_to_26_no_forbidden_runtime_input(self):
        specs = load("phase8jqv2_4tf1_candidate_specs.json")
        invariant = specs["runtime_invariants"]
        self.assertTrue(invariant["causal"])
        self.assertEqual(invariant["future_frames"], 0)
        self.assertFalse(invariant["runtime_gt"])
        self.assertFalse(invariant["annex_provenance"])

    def test_27_to_32_negative_controls_present(self):
        controls = {
            "static_depth", "camera_translation", "camera_rotation",
            "depth_quantization_noise", "disocclusion_without_actor",
            "seven_scattered_seeds",
        }
        for row in self.development["results"]:
            actual = {item["name"] for item in row["negative_controls"]}
            self.assertTrue(controls <= actual)

    def test_33_no_target_zero(self):
        report = load("phase8jqv2_4tf1_no_target_validation.json")
        self.assertTrue(report["all_zero_false_measurements"])
        self.assertTrue(report["all_zero_false_attention"])

    def test_34_to_38_bounded_dynamic_controls_present(self):
        controls = {
            "pure_tangential_crossing", "near_radial_approach",
            "gap1_reappearance", "gap2_reappearance",
            "large_projection",
        }
        for row in self.development["results"]:
            actual = {item["name"] for item in row["positive_controls"]}
            self.assertTrue(controls <= actual)

    def test_39_to_41_measurement_gate_not_overclaimed(self):
        report = load("phase8jqv2_4tf1_measurement_validation.json")
        self.assertEqual(report["status"], "FAIL")
        self.assertFalse(report["eligible_for_tracker_integration"])
        self.assertEqual(
            report["best_development_candidate"]["natural_passing_maps"], 0
        )

    def test_42_to_48_tracker_claims_gate_blocked(self):
        report = load("phase8jqv2_4tf1_tracker_integration_smoke.json")
        self.assertEqual(
            report["status"], "NOT_RUN_MEASUREMENT_GATE_FAILED"
        )
        self.assertFalse(report["identity_pass_claimed"])
        self.assertEqual(report["gap1"], "BLOCKED")
        self.assertEqual(report["gap2"], "BLOCKED")
        self.assertEqual(report["gap3"], "OUT_OF_SCOPE")

    def test_49_to_50_holdout_remains_sealed(self):
        report = load("phase8jqv2_4tf1_holdout_freeze.json")
        self.assertEqual(
            report["status"], "BLOCKED_DEVELOPMENT_GATE_FAILED"
        )
        self.assertEqual(report["selected_for_holdout"], [])
        self.assertTrue(report["no_tuning_after_holdout"])
        self.assertFalse(report["sealed_holdout_accessed"])

    def test_51_deterministic_protocol(self):
        report = load("phase8jqv2_4tf1_determinism.json")
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["parameter_grid_deterministic"])

    def test_52_to_53_runtime_and_history_bounded(self):
        report = load("phase8jqv2_4tf1_performance.json")
        self.assertTrue(report["history_bounded"])
        self.assertLessEqual(report["history_capacity_frames"], 4)
        self.assertLessEqual(report["maximum_components"], 64)

    def test_54_to_55_seed7_diagnostic_only(self):
        report = load("phase8jqv2_4tf1_candidate_ablation.json")
        self.assertTrue(report["diagnostic_only"])
        self.assertFalse(report["automatically_selectable"])
        self.assertFalse(self.selection["seed7_overfit_route"])

    def test_56_gap3_not_claimed_pass(self):
        report = load("phase8jqv2_4tf1_gap3_non_scope.json")
        self.assertFalse(report["strict_dynamic_through_gap3_possible"])
        self.assertFalse(report["gap3_pass_claimed"])

    def test_57_legacy_source_unchanged(self):
        self.assertEqual(
            digest("policy/dynamic/range_image_foreground.py"),
            self.entry["frozen_hashes"]["range_image_foreground.py"],
        )

    def test_58_to_62_prohibited_actions_absent(self):
        self.assertFalse(self.final["formal_preflight_rerun"])
        self.assertFalse(self.entry["formal_v3_entry_created"])
        self.assertFalse(self.final["formal_generation_started"])
        self.assertFalse(self.final["training_started"])
        self.assertFalse(self.final["test_accessed"])
        self.assertFalse(self.final["blind_accessed"])

    def test_63_to_64_historical_regressions(self):
        self.assertTrue(self.baseline["i1_probe_lifecycle_positive"])
        self.assertTrue(self.baseline["n1_pixel_report_match"])

    def test_65_development_fail_closed(self):
        self.assertFalse(self.development["sealed_holdout_accessed"])
        self.assertEqual(sum(
            row["development_hard_gate"]
            for row in self.development["results"]
        ), 0)
        self.assertEqual(
            self.final["primary_cause"],
            "temporal_foreground_contract_architecture_limit",
        )

    def test_66_report_completeness(self):
        names = [
            "entry_gate.json", "baseline_replay.json",
            "evaluation_split.json", "contract_invariants.json",
            "contract_review.json", "evidence_loss_analysis.json",
            "candidate_specs.json", "candidate_parameter_grid.json",
            "candidate_development_results.json",
            "candidate_ablation.json", "candidate_selection.json",
            "synthetic_positive.json", "synthetic_negative.json",
            "no_target_validation.json", "ordinary_dynamic_regression.json",
            "holdout_freeze.json", "holdout_results.json",
            "generalization.json", "measurement_validation.json",
            "tracker_integration_smoke.json", "gap3_non_scope.json",
            "compatibility_matrix.json", "performance.json",
            "determinism.json", "final_result.json",
        ]
        missing = [
            name for name in names
            if not (
                REPORTS / f"phase8jqv2_4tf1_{name}"
            ).is_file()
        ]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
