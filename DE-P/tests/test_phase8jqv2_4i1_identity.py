import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return json.loads((ROOT / name).read_text())


def sha(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


class Phase8JQ2I1FailClosedTest(unittest.TestCase):
    def test_frozen_sources(self):
        entry = load("reports/phase8jqv2_4i1_entry_gate.json")
        self.assertEqual(
            sha("authoritative_dataset/occlusion_constructor_v2_1.py"),
            entry["facts"]["v2_1_hash"],
        )
        self.assertEqual(
            sha("authoritative_dataset/occlusion_constructor_v2_2.py"),
            entry["facts"]["v2_2_hash"],
        )
        self.assertEqual(
            sha("configs/authoritative_dynamic_motion_contract_v2_1.yaml"),
            entry["facts"]["motion_contract_v2_1_hash"],
        )
        self.assertEqual(
            sha("configs/mixed_scene_map_profiles_v1.yaml"),
            entry["frozen_baseline_hashes"]["mixed_scene_map_profiles_v1.yaml"],
        )
        self.assertEqual(
            sha("authoritative_dataset/perception_probe_v2.py"),
            entry["frozen_baseline_hashes"]["perception_probe_v2.py"],
        )
        self.assertEqual(
            sha("policy/dynamic/dynamic_perception.py"),
            entry["frozen_baseline_hashes"]["dynamic_perception.py"],
        )

    def test_retained_manifest_scope(self):
        manifest = load("reports/phase8jqv2_4i1_retained_case_manifest.json")
        self.assertEqual(manifest["case_count"], 6)
        self.assertGreaterEqual(manifest["independent_map_count"], 3)
        self.assertGreaterEqual(manifest["independent_seed_count"], 2)
        self.assertEqual(manifest["gap_counts"], {"1": 1, "2": 2, "3": 3})
        for key in (
            "annex_used", "new_map_sweep_executed",
            "formal_generation_started", "test_accessed",
            "blind_accessed", "training_executed",
        ):
            self.assertFalse(manifest[key])

    def test_probe_and_trace_contract(self):
        probe = load("reports/phase8jqv2_4i1_probe_validation.json")
        self.assertEqual(probe["status"], "PASS")
        self.assertTrue(probe["persistent_track_manager"])
        self.assertFalse(probe["frozen_parameters_modified"])
        summary = load("reports/phase8jqv2_4i1_per_frame_trace_summary.json")
        self.assertEqual(summary["case_count"], 6)
        self.assertTrue(summary["persistent_perception_instance_per_case"])
        self.assertFalse(summary["runtime_gt_actor_input_used"])

    def test_schedule_is_time_shift_only(self):
        stage = load("reports/phase8jqv2_4i1_stage1_gap1.json")
        self.assertFalse(stage["actor_speed_modified"])
        self.assertFalse(stage["actor_direction_modified"])
        self.assertFalse(stage["actor_acceleration_modified"])
        self.assertFalse(stage["camera_geometry_modified"])
        for rejection in stage["rejections"]:
            self.assertTrue(rejection["invariants"]["time_shift_only"])
            self.assertTrue(rejection["invariants"]["speed_unchanged"])
            self.assertTrue(rejection["invariants"]["direction_unchanged"])
            self.assertTrue(rejection["invariants"]["acceleration_zero"])

    def test_stage_gate_is_fail_closed(self):
        final = load("reports/phase8jqv2_4i1_final_result.json")
        self.assertEqual(final["status"], "FAIL")
        self.assertEqual(
            final["primary_cause"], "natural_occlusion_detection_support"
        )
        self.assertEqual(final["gap_1"], "FAIL")
        self.assertEqual(final["gap_2"], "NOT_RUN_BLOCKED_STAGE1")
        self.assertEqual(final["gap_3"], "NOT_RUN_BLOCKED_STAGE1")
        self.assertFalse(final["frozen_identity_timeline_ready"])
        self.assertFalse(any(final["invariants"].values()))

    def test_report_completeness(self):
        required = [
            "phase8jqv2_4i1_pre_gap_root_cause.json",
            "phase8jqv2_4i1_identity_lifecycle_root_cause.json",
            "phase8jqv2_4i1_schedule_derivation.md",
            "phase8jqv2_4i1_schedule_validation.json",
            "phase8jqv2_4i1_schedule_determinism.json",
            "phase8jqv2_4i1_stage2_gap2.json",
            "phase8jqv2_4i1_stage3_gap3.json",
            "phase8jqv2_4i1_stage4_all_cases.json",
            "phase8jqv2_4i1_independent_validator.json",
            "phase8jqv2_4i1_negative_controls.json",
            "phase8jqv2_4i1_final_result.json",
            "phase8jqv2_4i1_final_recommendation.md",
            "phase8jqv2_4i1_final_readiness.md",
        ]
        for name in required:
            self.assertTrue((ROOT / "reports" / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
