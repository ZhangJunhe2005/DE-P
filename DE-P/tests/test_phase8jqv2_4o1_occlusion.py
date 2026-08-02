from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import yaml

from authoritative_dataset.cuda_renderer_v1 import CudaAuthorityRenderer
from authoritative_dataset.dynamic_motion_v2 import (
    CONTRACT_VERSION, CONTRACT_VERSION_V2_1, load_motion_contract,
)
from authoritative_dataset.generate_v1 import load_config, tasks_for
from authoritative_dataset.occlusion_constructor_v2_1 import (
    natural_occlusion_pattern,
)
from authoritative_dataset.occlusion_geometry_v1 import (
    ray_aabb_interval, segment_aabb_interval,
)


ROOT = Path(__file__).resolve().parents[1]
V1_HASH = "a81319d734871c48d49058e22a36a733baead3d85db2ec39206221d3b154f723"
V2_HASH = "56d7118862afc22d8df8672d747c930bc4fdcfcb1fe0795c19e5ceef47989461"
V2_CONTRACT_HASH = "da7cec3ba03887049bf746d2d25c7e444c4cd4f604ce11b5d25d2bd5404c54af"
V21_CONTRACT_HASH = "5a6bc337d73617aa42b8b30d1a33386b5cfe876acb43c73667bdf32c45b8e7a4"


def fake_backend():
    dimensions = np.asarray([80, 80, 40])
    flat = np.zeros(int(np.prod(dimensions)), dtype=bool)
    for y in (39, 40):
        for z in (19, 20):
            flat[20*dimensions[1]*dimensions[2]+y*dimensions[2]+z] = True
    map_value = SimpleNamespace(
        metadata={"map_uuid": "synthetic_2x2",
                  "occupancy_hash": "synthetic_2x2"},
        flat=flat, origin=np.zeros(3), dimensions=dimensions,
        resolution=.1,
    )
    return SimpleNamespace(map=map_value)


class Phase8JQ24O1ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v2_path = ROOT/"configs/authoritative_dynamic_motion_contract_v2.yaml"
        cls.v21_path = ROOT/"configs/authoritative_dynamic_motion_contract_v2_1.yaml"
        cls.v2 = load_motion_contract(cls.v2_path)
        cls.v21 = load_motion_contract(cls.v21_path)
        cls.capability = json.loads((
            ROOT/"reports/phase8jqv2_4o1_map_occlusion_capability.json"
        ).read_text())

    def test_01_v2_contract_preserved(self):
        self.assertEqual(self.v2["contract_version"], CONTRACT_VERSION)
        self.assertEqual(hashlib.sha256(
            self.v2_path.read_bytes()).hexdigest(), V2_CONTRACT_HASH)

    def test_02_v21_contract_hash(self):
        self.assertEqual(self.v21["contract_version"], CONTRACT_VERSION_V2_1)
        self.assertEqual(hashlib.sha256(
            self.v21_path.read_bytes()).hexdigest(), V21_CONTRACT_HASH)

    def test_03_frozen_thresholds_unchanged(self):
        for key in (
            "dynamic_enter_speed_mps", "dynamic_exit_speed_mps",
            "confirmation_hits", "motion_consistency_frames",
            "maximum_missed_frames",
        ):
            self.assertEqual(
                self.v2["frozen_perception"][key],
                self.v21["frozen_perception"][key])

    def test_04_pre_gap_is_derived(self):
        frozen = self.v21["frozen_perception"]
        expected = max(
            frozen["confirmation_hits"],
            frozen["dynamic_min_confirmed_hits"],
            frozen["motion_consistency_frames"]+1,
        ) + self.v21["occlusion"]["pre_gap_guard_frames"]
        self.assertEqual(
            self.v21["occlusion"]["pre_gap_visible_min"], expected)
        self.assertGreaterEqual(expected, 4)

    def test_05_natural_only(self):
        value = self.v21["occlusion"]
        self.assertEqual(
            value["source"], "canonical_occupancy_sensor_visibility")
        self.assertFalse(value["artificial_frame_hiding"])
        self.assertFalse(value["metadata_only_occlusion"])

    def test_06_exit_modes_forbidden(self):
        value = self.v21["occlusion"]
        self.assertFalse(value["fov_exit_allowed"])
        self.assertFalse(value["behind_camera_allowed"])
        self.assertFalse(value["max_depth_exit_allowed"])

    def test_07_post_gap_minimum(self):
        self.assertGreaterEqual(
            self.v21["occlusion"]["post_gap_visible_min"], 2)
        self.assertFalse(
            self.v21["occlusion"]["last_frame_may_be_reappearance"])

    def test_08_occluded_speed_preserved(self):
        self.assertEqual(
            self.v21["scenarios"]["occluded_but_tracked"][
                "speed_range_mps"], [1.4, 1.7])

    def test_09_actor_radius_preserved(self):
        self.assertEqual(self.v21["sampling"]["actor_radius_m"], .3)

    def test_10_multi_radius_preserved(self):
        self.assertEqual(
            self.v21["scenarios"]["multi_target"]["actor_radius_m"], .2)

    def test_11_v1_manifest_unchanged(self):
        path = ROOT/"data/phase8_authoritative_v1/manifests/dataset_manifest.json"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), V1_HASH)

    def test_12_v2_manifest_unchanged(self):
        path = ROOT/"data/phase8_authoritative_v2/manifests/dataset_manifest.json"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), V2_HASH)

    def test_13_all_formal_maps_audited(self):
        self.assertEqual(self.capability["maps_scanned"], 60)
        self.assertEqual(self.capability["train_maps_scanned"], 48)
        self.assertEqual(self.capability["valid_maps_scanned"], 12)

    def test_14_diversity_declared(self):
        value = self.capability["minimum_diversity_declared_before_scan"]
        self.assertEqual(value["train_eligible_maps_min"], 8)
        self.assertEqual(value["valid_eligible_maps_min"], 3)

    def test_15_no_actionable_map(self):
        self.assertEqual(
            self.capability["eligible_counts"], {"train": 0, "valid": 0})
        self.assertEqual(self.capability["status"], "FAIL")

    def test_16_no_coplanar_patch(self):
        self.assertTrue(all(
            row["actionable_coplanar_2x2_patch_count"] == 0
            for row in self.capability["maps"]))

    def test_17_split_isolation(self):
        manifest = yaml.safe_load((
            ROOT/"configs/phase8_authoritative_v3_occlusion_map_eligibility.yaml"
        ).read_text())
        self.assertTrue(manifest["split_isolation"])

    def test_18_no_model_selection(self):
        manifest = yaml.safe_load((
            ROOT/"configs/phase8_authoritative_v3_occlusion_map_eligibility.yaml"
        ).read_text())
        self.assertFalse(manifest["model_output_used"])

    def test_19_no_test_or_blind_selection(self):
        manifest = yaml.safe_load((
            ROOT/"configs/phase8_authoritative_v3_occlusion_map_eligibility.yaml"
        ).read_text())
        self.assertFalse(manifest["test_or_blind_used"])

    def test_20_v3_config_is_fail_closed(self):
        config = load_config(
            ROOT/"configs/phase8_authoritative_v3_generation.yaml")
        self.assertFalse(config["_occlusion_eligibility_gate_passed"])

    def test_21_scheduler_has_no_occlusion_task_map(self):
        config = load_config(
            ROOT/"configs/phase8_authoritative_v3_generation.yaml")
        value = config["_formal_scenario_map_eligibility"][
            "occluded_but_tracked"]
        self.assertEqual(value["train"], [])
        self.assertEqual(value["valid"], [])

    def test_22_formal_v3_not_generated(self):
        self.assertFalse((
            ROOT/"data/phase8_authoritative_v3/generation_state/"
            "completion/FULL_GENERATION_COMPLETE").exists())

    def test_23_no_handoff_scripts_on_failed_gate(self):
        self.assertFalse((
            ROOT/"scripts/phase8jqv2_4_v3_dynamic_generate_host.sh"
        ).exists())

    def test_24_real_actor_radius_used_in_source(self):
        source = (
            ROOT/"authoritative_dataset/generate_v1.py").read_text()
        self.assertIn(
            'current, actor["radius_m"]', source)
        self.assertIn(
            'point, actor["radius_m"]', source)

    def test_25_scenario_wide_occluded_removed(self):
        source = (
            ROOT/"authoritative_dataset/generate_v1.py").read_text()
        self.assertNotIn(
            '"occluded":\\n                        '
            'task["scenario"] == "occluded_but_tracked"', source)


class Phase8JQ24O1GeometryTests(unittest.TestCase):
    def test_26_ray_aabb_first_interval(self):
        value = ray_aabb_interval(
            [0, 0, 0], [1, 0, 0], [2, -1, -1], [3, 1, 1])
        self.assertEqual(value, (2.0, 3.0))

    def test_27_parallel_ray_miss(self):
        self.assertIsNone(ray_aabb_interval(
            [0, 2, 0], [1, 0, 0], [2, -1, -1], [3, 1, 1]))

    def test_28_segment_clips_exit(self):
        value = segment_aabb_interval(
            [0, 0, 0], [2.5, 0, 0], [2, -1, -1], [3, 1, 1])
        self.assertEqual(value, (2.0, 2.5))

    def test_29_pattern_accepts_full_static_gap(self):
        diagnostics = {
            "per_actor_projected_pixel_count":
                np.asarray([[10]]*8),
            "per_actor_visible_pixel_count":
                np.asarray([[10], [10], [10], [10], [0], [10], [10], [10]]),
            "per_actor_static_blocked_pixel_count":
                np.asarray([[0], [0], [0], [0], [10], [0], [0], [0]]),
            "per_actor_outside_fov": np.zeros((8, 1), bool),
            "per_actor_behind_camera": np.zeros((8, 1), bool),
            "per_actor_beyond_max_depth": np.zeros((8, 1), bool),
        }
        contract = load_motion_contract(
            ROOT/"configs/authoritative_dynamic_motion_contract_v2_1.yaml")
        self.assertEqual(
            natural_occlusion_pattern(
                diagnostics, 0, contract)["accepted_gaps"], [(4, 4)])

    def test_30_partial_occlusion_rejected(self):
        diagnostics = {
            "per_actor_projected_pixel_count": np.asarray([[10]]*8),
            "per_actor_visible_pixel_count":
                np.asarray([[10], [10], [10], [10], [1], [10], [10], [10]]),
            "per_actor_static_blocked_pixel_count":
                np.asarray([[0], [0], [0], [0], [9], [0], [0], [0]]),
            "per_actor_outside_fov": np.zeros((8, 1), bool),
            "per_actor_behind_camera": np.zeros((8, 1), bool),
            "per_actor_beyond_max_depth": np.zeros((8, 1), bool),
        }
        contract = load_motion_contract(
            ROOT/"configs/authoritative_dynamic_motion_contract_v2_1.yaml")
        self.assertEqual(natural_occlusion_pattern(
            diagnostics, 0, contract)["accepted_gaps"], [])

    def test_31_fov_exit_rejected(self):
        diagnostics = {
            "per_actor_projected_pixel_count": np.asarray([[10]]*8),
            "per_actor_visible_pixel_count":
                np.asarray([[10], [10], [10], [10], [0], [10], [10], [10]]),
            "per_actor_static_blocked_pixel_count":
                np.asarray([[0], [0], [0], [0], [10], [0], [0], [0]]),
            "per_actor_outside_fov":
                np.asarray([[False]]*4+[[True]]+[[False]]*3),
            "per_actor_behind_camera": np.zeros((8, 1), bool),
            "per_actor_beyond_max_depth": np.zeros((8, 1), bool),
        }
        contract = load_motion_contract(
            ROOT/"configs/authoritative_dynamic_motion_contract_v2_1.yaml")
        self.assertEqual(natural_occlusion_pattern(
            diagnostics, 0, contract)["accepted_gaps"], [])


@unittest.skipUnless(torch.cuda.is_available(), "host CUDA required")
class Phase8JQ24O1CudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sensor = yaml.safe_load((
            ROOT/"configs/phase8_authoritative_v3_generation.yaml"
        ).read_text())["sensor_settings"]
        cls.backend = fake_backend()
        cls.renderer = CudaAuthorityRenderer(cls.sensor, "cuda:0")
        cls.camera = np.asarray([[1.695, 4.0, 2.0]])
        cls.yaw = np.asarray([0.0])

    def test_32_no_actor_old_output_unchanged(self):
        old = self.renderer.render(
            self.backend, self.camera, self.yaw,
            np.empty((1, 0, 3)), [])
        new = self.renderer.render_with_actor_diagnostics(
            self.backend, self.camera, self.yaw,
            np.empty((1, 0, 3)), [])
        self.assertTrue(np.array_equal(old[0], new["composed_depth"]))

    def test_33_projected_pixels(self):
        result = self.renderer.render_with_actor_diagnostics(
            self.backend, self.camera, self.yaw,
            np.asarray([[[3.045, 4.0, 2.0]]]), [.3])
        self.assertGreater(
            result["per_actor_projected_pixel_count"][0, 0], 0)

    def test_34_visible_pixels_full_occlusion(self):
        result = self.renderer.render_with_actor_diagnostics(
            self.backend, self.camera, self.yaw,
            np.asarray([[[3.045, 4.0, 2.0]]]), [.3])
        self.assertEqual(
            result["per_actor_visible_pixel_count"][0, 0], 0)

    def test_35_static_blocked_pixels(self):
        result = self.renderer.render_with_actor_diagnostics(
            self.backend, self.camera, self.yaw,
            np.asarray([[[3.045, 4.0, 2.0]]]), [.3])
        self.assertEqual(
            result["per_actor_static_blocked_pixel_count"][0, 0],
            result["per_actor_projected_pixel_count"][0, 0])

    def test_36_behind_camera_classified(self):
        result = self.renderer.render_with_actor_diagnostics(
            self.backend, self.camera, self.yaw,
            np.asarray([[[1.0, 4.0, 2.0]]]), [.3])
        self.assertTrue(result["per_actor_behind_camera"][0, 0])

    def test_37_beyond_depth_classified(self):
        result = self.renderer.render_with_actor_diagnostics(
            self.backend, self.camera, self.yaw,
            np.asarray([[[30.0, 4.0, 2.0]]]), [.3])
        self.assertTrue(result["per_actor_beyond_max_depth"][0, 0])

    def test_38_outside_fov_classified(self):
        result = self.renderer.render_with_actor_diagnostics(
            self.backend, self.camera, self.yaw,
            np.asarray([[[3.0, 20.0, 2.0]]]), [.3])
        self.assertTrue(result["per_actor_outside_fov"][0, 0])

    def test_39_nearest_actor_depth_order_independent(self):
        actors = np.asarray([[[3.0, 3.0, 2.0], [3.0, 5.0, 2.0]]])
        first = self.renderer.render_with_actor_diagnostics(
            self.backend, self.camera, self.yaw, actors, [.3, .3])
        second = self.renderer.render_with_actor_diagnostics(
            self.backend, self.camera, self.yaw,
            actors[:, ::-1], [.3, .3])
        self.assertTrue(np.array_equal(
            first["composed_depth"], second["composed_depth"]))


if __name__ == "__main__":
    unittest.main()
