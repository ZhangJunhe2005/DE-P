import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import yaml

from authoritative_dataset.foreground_observability_v1 import (
    certify_observability,
)


ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads((ROOT / path).read_text())


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


class Phase8JQ24N1Tests(unittest.TestCase):
    def test_entry_and_frozen_sources(self):
        entry = load("reports/phase8jqv2_4n1_entry_gate.json")
        self.assertEqual(entry["status"], "PASS")
        mapping = {
            "occlusion_constructor_v2_1.py":
                "authoritative_dataset/occlusion_constructor_v2_1.py",
            "occlusion_constructor_v2_2.py":
                "authoritative_dataset/occlusion_constructor_v2_2.py",
            "occlusion_identity_schedule_v1.py":
                "authoritative_dataset/occlusion_identity_schedule_v1.py",
            "temporal_foreground.py": "policy/dynamic/temporal_foreground.py",
            "range_image_foreground.py":
                "policy/dynamic/range_image_foreground.py",
            "image_foreground_components.py":
                "policy/dynamic/image_foreground_components.py",
            "track_manager.py": "policy/dynamic/track_manager.py",
            "dynamic_perception.py": "policy/dynamic/dynamic_perception.py",
            "dynamic_motion_v2.py":
                "authoritative_dataset/dynamic_motion_v2.py",
            "motion_contract_v2_1.yaml":
                "configs/authoritative_dynamic_motion_contract_v2_1.yaml",
            "authority_static_v1.py": "geometry_authority/static_v1.py",
            "cuda_renderer_v1.py":
                "authoritative_dataset/cuda_renderer_v1.py",
            "traj_opt.yaml": "config/traj_opt.yaml",
            "mixed_scene_map_profiles_v1.yaml":
                "configs/mixed_scene_map_profiles_v1.yaml",
        }
        for key, path in mapping.items():
            with self.subTest(key=key):
                self.assertEqual(sha(path), entry["frozen_hashes"][key])

    def test_visible_not_foreground_pixel_reproduction(self):
        report = load("reports/phase8jqv2_4n1_stage1_pixel_pipeline.json")
        counts = load(
            "diagnostics/phase8jqv2_4n1/stage1_pixel_counts_per_step.json"
        )
        pre = counts["pre_gap_frame"]
        post = counts["first_post_gap_frame"]
        self.assertEqual(report["pre_gap_visible_pixels"], 383)
        self.assertEqual(pre["history_supported_visible_pixels"], 383)
        self.assertEqual(pre["positive_residual_visible_pixels"], 174)
        self.assertEqual(pre["closer_supported_visible_pixels"], 8)
        self.assertEqual(pre["range_seed_visible_pixels"], 1)
        self.assertEqual(pre["seed_union_visible_pixels"], 7)
        self.assertEqual(pre["component_visible_pixels"], 0)
        self.assertEqual(report["first_zero_stage"], "component_visible_pixels")
        self.assertEqual(post["visible_pixels"], 26)
        self.assertFalse(post["measurement_valid"])

    def test_observability_certificate_positive_and_repeated_ray_negative(self):
        frames, height, width = 8, 96, 160
        static = np.full((frames, height, width), 10.0, np.float32)
        near = np.full_like(static, np.inf)
        visible = np.zeros_like(static, dtype=bool)
        for frame in (0, 1, 2, 3, 5, 6, 7):
            u = 10 + 12 * frame
            visible[frame, 20:30, u:u+10] = True
            near[frame, 20:30, u:u+10] = 2.0
        composed = np.minimum(static, near)
        actor = np.stack((
            np.arange(frames) * .1,
            np.ones(frames) * 2,
            np.ones(frames),
        ), axis=1)
        camera = np.zeros((frames, 3))
        yaw = np.zeros(frames)
        times = np.arange(frames) * .1
        positive = certify_observability(
            composed, static, near, visible, actor, camera, yaw, times, 4, 4
        )
        self.assertTrue(positive["observable"])
        repeated = np.zeros_like(visible)
        repeated[:, 20:30, 20:30] = True
        repeated[4] = False
        repeated_near = np.where(repeated, 2.0, np.inf).astype(np.float32)
        negative = certify_observability(
            np.minimum(static, repeated_near), static, repeated_near,
            repeated, actor, camera, yaw, times, 4, 4
        )
        self.assertFalse(negative["observable"])
        self.assertFalse(positive["runtime_measurement_emitted"])
        self.assertFalse(positive["runtime_gt_input_authorized"])

    def test_profiles_and_development_split(self):
        profiles = yaml.safe_load(
            (ROOT / "configs/mixed_scene_map_profiles_v2.yaml").read_text()
        )
        self.assertEqual(
            profiles["profiles_version"], "mixed_scene_map_profiles_v2"
        )
        self.assertFalse(profiles["annex_allowed"])
        self.assertEqual(len(profiles["profiles"]), 15)
        types = {}
        for name, row in profiles["profiles"].items():
            types.setdefault(int(row["maze_type"]), []).append(name)
        self.assertEqual(set(types), {1, 2, 5, 6, 7})
        self.assertTrue(all(len(names) == 3 for names in types.values()))
        manifest = load("reports/phase8jqv2_4n1_profile_candidates.json")
        self.assertEqual(manifest["map_count"], 30)
        self.assertEqual(
            set(manifest["seed_namespaces"]),
            {"profile_debug", "profile_holdout"},
        )
        self.assertFalse(manifest["formal_eligible"])
        self.assertFalse(manifest["annex_used"])
        self.assertTrue(all(
            row["ordinary_navigation_capable"] for row in manifest["maps"]
        ))

    def test_general_proposer_has_no_case_or_identity_branch(self):
        source = (
            ROOT
            / "authoritative_dataset/natural_observable_occlusion_proposer_v1.py"
        ).read_text()
        self.assertNotIn("if case_id", source)
        self.assertNotIn("if map_uuid", source)
        self.assertNotIn("if seed", source)
        self.assertNotIn("instance_id", source)
        self.assertIn("constant_velocity", source)
        self.assertIn("velocity", source)

    def test_gate_and_gap3_isolation(self):
        result = load("reports/phase8jqv2_4n1_final_result.json")
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(
            result["primary_cause"],
            "frozen_temporal_foreground_natural_observability",
        )
        self.assertEqual(result["gap_1_identity"], "FAIL")
        self.assertEqual(
            result["gap_2_identity"], "NOT_RUN_BLOCKED_GAP1"
        )
        self.assertEqual(result["gap_3"], "CONTRACT_REVIEW_REQUIRED")
        self.assertFalse(any(result["invariants"].values()))
        bound = load("reports/phase8jqv2_4n1_gap3_contract_bound.json")
        self.assertEqual(
            bound["confidence_after_miss_3"], 0.421875
        )
        self.assertFalse(bound["strict_dynamic_through_gap3_possible"])
        self.assertFalse(bound["gap3_strict_pass_claimed"])

    def test_no_runtime_or_formal_integration(self):
        runtime_sources = [
            "policy/dynamic/dynamic_perception.py",
            "policy/dynamic/temporal_foreground.py",
            "policy/dynamic/track_manager.py",
            "authoritative_dataset/generate_v1.py",
        ]
        for path in runtime_sources:
            source = (ROOT / path).read_text()
            self.assertNotIn("foreground_observability_v1", source)
            self.assertNotIn("mixed_scene_map_profiles_v2", source)
        result = load("reports/phase8jqv2_4n1_final_result.json")
        for key in (
            "annex_used", "formal_preflight_rerun",
            "formal_v3_entry_created", "formal_v3_generation_started",
            "optimizer_step_executed", "training_started",
            "production_test_accessed", "blind_accessed",
        ):
            self.assertFalse(result["invariants"][key])

    def test_report_completeness(self):
        names = [
            "entry_gate", "problem_statement",
            "temporal_foreground_contract", "stage1_pixel_pipeline",
            "observability_contract", "observability_proxy_validation",
            "observability_false_positive_analysis", "profile_candidates",
            "profile_sweep", "profile_holdout", "selected_profiles",
            "map_type_diversity", "gap1_validation", "gap2_validation",
            "gap3_diagnostic", "gap3_contract_bound", "identity_lifecycle",
            "independent_validator", "negative_controls", "determinism",
            "static_feasibility", "ordinary_dynamic_feasibility",
            "distribution_regression", "performance", "final_result",
        ]
        for name in names:
            with self.subTest(name=name):
                path = ROOT / "reports" / f"phase8jqv2_4n1_{name}.json"
                self.assertTrue(path.is_file(), path)
                json.loads(path.read_text())
        for name in ("final_recommendation.md", "final_readiness.md",
                     "temporal_foreground_dataflow.md"):
            self.assertTrue(
                (ROOT / "reports" / f"phase8jqv2_4n1_{name}").is_file()
            )


if __name__ == "__main__":
    unittest.main()
