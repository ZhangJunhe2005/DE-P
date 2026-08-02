#!/usr/bin/env python3
"""69-item CCR1 control-contract gate; no detector or tracker is executed."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/phase8_dynamic_perception_controls_v1"
REPORTS = ROOT / "reports"
PREFIX = "phase8jqv2_4ccr1_"
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Phase8JQ24CCR1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = read(DATASET / "manifest.json")
        cls.validation = read(DATASET / "physical_validation.json")
        cls.results = {
            row["control_id"]: row for row in cls.validation["controls"]
        }
        cls.controls = {}
        for path in (DATASET / "controls").glob("*/control.json"):
            value = read(path)
            cls.controls[value["control_id"]] = value
        cls.entry = read(REPORTS / f"{PREFIX}entry_gate.json")
        cls.mapping = read(
            REPORTS / f"{PREFIX}historical_control_mapping.json"
        )
        cls.regression = read(REPORTS / f"{PREFIX}regression.json")
        cls.final = read(REPORTS / f"{PREFIX}final_result.json")
        cls.config = yaml.safe_load(
            (ROOT / "configs/dynamic_perception_physical_control_contract_v1.yaml")
            .read_text()
        )

    def test_01_to_69_contract_checklist(self):
        controls = self.controls
        results = self.results
        static = [
            value for value in controls.values()
            if value["role"] == "hard_negative"
        ]
        actor = [
            row for row in self.validation["controls"]
            if row["actor_validation"]
        ]
        hard_actor = [
            row for row in actor
            if controls[row["control_id"]]["role"] == "hard_positive"
        ]
        runtime_keys = {
            key for value in controls.values()
            for key in value["runtime_inputs"]
        }
        architecture_paths = [
            ROOT / "policy/dynamic/visibility_provenance_v1.py",
            ROOT / "policy/dynamic/dynamic_measurement_v3.py",
            ROOT / "policy/dynamic/visibility_aware_residual_v1.py",
            ROOT / "policy/dynamic/causal_depth_track_before_detect_v1.py",
            ROOT / "policy/dynamic/dual_path_dynamic_perception_v1.py",
            ROOT / "policy/dynamic/dynamic_perception_architecture_registry.py",
        ]
        expected_frozen_paths = {
            "legacy_temporal_foreground":
                ROOT / "policy/dynamic/temporal_foreground.py",
            "legacy_range_image_foreground":
                ROOT / "policy/dynamic/range_image_foreground.py",
            "tf1_candidate":
                ROOT / "policy/dynamic/range_image_foreground_v2_1.py",
            "track_manager": ROOT / "policy/dynamic/track_manager.py",
            "motion_contract":
                ROOT / "configs/authoritative_dynamic_motion_contract_v2_1.yaml",
            "constructor_v2_1":
                ROOT / "authoritative_dataset/occlusion_constructor_v2_1.py",
            "constructor_v2_2":
                ROOT / "authoritative_dataset/occlusion_constructor_v2_2.py",
            "profile_v1": ROOT / "configs/mixed_scene_map_profiles_v1.yaml",
            "profile_v2": ROOT / "configs/mixed_scene_map_profiles_v2.yaml",
        }
        required_reports = [
            "entry_gate.json", "historical_control_mapping.json",
            "historical_artifact_integrity.json", "control_contract.json",
            "control_taxonomy.json", "control_schema.json",
            "actor_renderer_validation.json", "small_projection_suite.json",
            "small_projection_physical_validation.json",
            "projection_discretization.json",
            "actor_metadata_consistency.json", "fov_control_suite.json",
            "fov_camera_motion_validation.json",
            "warp_visibility_reference.json",
            "fov_provenance_validation.json",
            "static_disocclusion_validation.json",
            "depth_validity_transition.json",
            "dynamic_edge_positive_suite.json",
            "paired_control_matrix.json",
            "detection_latency_contract.json",
            "runtime_offline_isolation.json", "control_split.json",
            "holdout_freeze.json", "determinism.json",
            "performance.json", "regression.json", "final_result.json",
            "final_recommendation.md", "final_readiness.md",
        ]
        git_diff_ok = subprocess.run(
            ["git", "diff", "--check"], cwd=ROOT, check=False,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        ).returncode == 0
        checks = [
            ("DPAR1 Route B entry",
             self.entry["dpar1_route"] == "B"
             and self.entry["dpar1_primary_cause"]
             == "dynamic_perception_control_contract_invalid"),
            ("historical small fixture preserved",
             self.mapping["legacy_small_projection"]
             ["historical_evidence_preserved"]),
            ("historical FOV fixture preserved",
             self.mapping["legacy_fov_boundary_change"]
             ["historical_evidence_preserved"]),
            ("historical controls excluded from new Gate",
             not self.mapping["legacy_small_projection"]
             ["eligible_for_future_gate"]
             and not self.mapping["legacy_fov_boundary_change"]
             ["eligible_for_future_gate"]),
            ("legacy hashes frozen",
             all(sha(expected_frozen_paths[key])
                 == self.entry["frozen_hashes"][key]
                 for key in ("legacy_temporal_foreground",
                             "legacy_range_image_foreground",
                             "tf1_candidate"))),
            ("tracker hash frozen",
             sha(expected_frozen_paths["track_manager"])
             == self.entry["frozen_hashes"]["track_manager"]),
            ("v2.1/v2.2 frozen",
             all(sha(expected_frozen_paths[key])
                 == self.entry["frozen_hashes"][key]
                 for key in ("constructor_v2_1", "constructor_v2_2"))),
            ("Profile V1/V2 frozen",
             all(sha(expected_frozen_paths[key])
                 == self.entry["frozen_hashes"][key]
                 for key in ("profile_v1", "profile_v2"))),
            ("motion contract frozen",
             sha(expected_frozen_paths["motion_contract"])
             == self.entry["frozen_hashes"]["motion_contract"]),
            ("canonical actor sphere",
             self.manifest["renderer_version"]
             == "canonical_occupancy_cuda_raycast_v1"
             and all(row["rerender_owner_equal"] for row in actor)),
            ("radius 0.20",
             any(item["radius_m"] == 0.2 for row in hard_actor
                 for item in row["actor_validation"])),
            ("radius 0.30",
             any(item["radius_m"] == 0.3 for row in hard_actor
                 for item in row["actor_validation"])),
            ("distance 0.60 classified physically infeasible",
             {item["radius_m"] for item in
              self.manifest["physically_infeasible_matrix_entries"]}
             == {0.2, 0.3}),
            ("distance 1.00",
             any(item["distance_reference_m"] == 1.0 for row in hard_actor
                 for item in row["actor_validation"])),
            ("distance 1.40",
             any(item["distance_reference_m"] == 1.4 for row in hard_actor
                 for item in row["actor_validation"])),
            ("distance 1.80",
             any(item["distance_reference_m"] == 1.8 for row in hard_actor
                 for item in row["actor_validation"])),
            ("analytic diameter",
             abs(results["actor_min_full_r020_d180_center"]
                 ["actor_validation"][0]["analytic_diameter_pixels"]
                 - 17.88854381999832) < 1e-12),
            ("CUDA projected area",
             all(not item["center_axis_continuous_formula_applicable"]
                 or item["area_relative_error"] <=
                 self.config["validation"]["projection_area_relative_tolerance"]
                 for row in actor for item in row["actor_validation"])),
            ("metadata/render position",
             all(row["rerender_depth_max_abs_error_m"] == 0 for row in actor)),
            ("metadata/render velocity",
             all(item["velocity_max_abs_error_mps"] <= 1e-5
                 for row in actor for item in row["actor_validation"])),
            ("speed contract",
             all(any(abs(item["speed_mps"] - allowed) < 1e-9
                     for allowed in (0.9, 1.05, 1.2, 1.45, 1.6, 1.75))
                 for row in hard_actor for item in row["actor_validation"])),
            ("no manual depth overwrite",
             not self.manifest["manual_depth_overwrite"]
             and all(not value["manual_depth_overwrite"]
                     for value in controls.values())),
            ("full in-contract projection",
             results["actor_min_full_r020_d180_center"]
             ["actor_validation"][0]["cuda_visible_pixels"] == 249),
            ("partial edge projection classification",
             controls["edge_actor_enter_left_r020"]["semantic_class"]
             == "bounded_latency_positive"),
            ("partial occlusion classification",
             controls["actor_partial_static_occlusion"]["role"]
             == "partial_occlusion_positive"),
            ("out-of-contract projection",
             controls["diagnostic_legacy_equivalent_r010_d400"]
             ["semantic_class"] == "diagnostic_out_of_contract"),
            ("camera yaw FOV",
             any(results[c["control_id"]]["camera_yaw_delta_max_rad"] > 0
                 for c in static)),
            ("camera translation FOV",
             any(results[c["control_id"]]["camera_translation_max_m"] > 0
                 for c in static)),
            ("camera combined motion",
             any(results[c["control_id"]]["camera_yaw_delta_max_rad"] > 0
                 and results[c["control_id"]]["camera_translation_max_m"] > 0
                 for c in static)),
            ("real static geometry",
             all(c["authority"]["scope"] == "development_control"
                 for c in static)),
            ("warp current-to-previous correspondence",
             read(REPORTS / f"{PREFIX}warp_visibility_reference.json")
             ["current_to_previous_correspondence"]),
            ("warp previous-to-current correspondence",
             read(REPORTS / f"{PREFIX}warp_visibility_reference.json")
             ["previous_to_current_correspondence"]),
            ("stable overlap",
             all(results[c["control_id"]]["provenance_totals"]
                 ["stable_overlap_pixels"] > 0 for c in static)),
            ("newly visible from image FOV",
             any(results[c["control_id"]]["provenance_totals"]
                 ["newly_visible_from_image_fov_pixels"] > 0 for c in static)),
            ("newly invalid to image FOV",
             any(results[c["control_id"]]["provenance_totals"]
                 ["newly_invalid_to_image_fov_pixels"] > 0 for c in static)),
            ("static disocclusion",
             results["static_disocclusion"]["provenance_totals"]
             ["newly_visible_from_static_disocclusion_pixels"] > 0),
            ("max-depth transition",
             results["max_depth_background_entry"]["provenance_totals"]
             ["max_depth_transition_pixels"] > 0),
            ("invalid-depth transition",
             any(results[c["control_id"]]["provenance_totals"]
                 ["current_depth_invalid_pixels"] > 0 for c in static)),
            ("all-depth-valid FOV event",
             results["static_fov_yaw_left_small"]["all_depth_valid"]
             and results["static_fov_yaw_left_small"]["provenance_totals"]
             ["newly_visible_from_image_fov_pixels"] > 0),
            ("no actor static negative",
             all(c["actor_trajectory"] is None for c in static)),
            ("actor entering left edge",
             "edge_actor_enter_left_r020" in controls),
            ("actor entering right edge",
             "edge_actor_enter_right_r020" in controls),
            ("actor upper edge", "edge_actor_upper_r030" in controls),
            ("actor lower edge", "edge_actor_lower_r030" in controls),
            ("moving camera + actor",
             results["edge_actor_moving_camera"]["camera_translation_max_m"] > 0
             or results["edge_actor_moving_camera"]
             ["camera_yaw_delta_max_rad"] > 0),
            ("no blanket edge crop semantics",
             all(c["provenance_expectation"]
                 ["blanket_border_crop_forbidden"]
                 for c in controls.values()
                 if c["role"] == "paired_edge_positive")),
            ("bounded detection latency contract",
             self.config["latency"]["causal_support_frames_k"] == 3),
            ("offline GT isolated",
             all(set(c["runtime_inputs"]).isdisjoint(
                 c["offline_ground_truth"]) for c in controls.values())),
            ("runtime fields contain no GT",
             runtime_keys.isdisjoint({
                 "actor_id", "actor_mask", "instance_id",
                 "expected_classification", "authority_correspondence",
                 "future_depth", "future_actor_state",
             })),
            ("generator/validator separation",
             self.manifest["generator_hash"] != self.manifest["validator_hash"]),
            ("development split",
             sum(c["split"] == "development"
                 for c in controls.values()) == 39),
            ("sealed holdout split",
             sum(c["split"] == "sealed_control_holdout"
                 for c in controls.values()) == 39),
            ("holdout architecture not evaluated",
             not self.validation["architecture_executed"]),
            ("deterministic controls",
             read(REPORTS / f"{PREFIX}determinism.json")["status"] == "PASS"),
            ("authority hashes",
             all(c["authority"]["authority_hash"]
                 == self.manifest["authority"]
                 [c["physical"].get("background",
                  c["physical"].get("static_geometry",
                  c["authority"]["map_uuid"]))]
                 ["authority_hash"]
                 if c["physical"].get("background",
                    c["physical"].get("static_geometry")) in
                    self.manifest["authority"]
                 else bool(c["authority"]["authority_hash"])
                 for c in controls.values())),
            ("sensor hash",
             all(c["sensor_hash"] == self.manifest["sensor_hash"]
                 for c in controls.values())),
            ("CUDA renderer hash",
             all(c["renderer_hash"] == self.manifest["renderer_hash"]
                 for c in controls.values())),
            ("no architecture prototype",
             not any(path.exists() for path in architecture_paths)),
            ("no holdout detector run",
             not self.manifest["detector_executed"]
             and not self.validation["detector_executed"]),
            ("no Formal preflight", not self.final["formal_preflight_rerun"]),
            ("no Formal V3", not self.final["formal_v3_entry_created"]
             and not self.final["formal_generation_started"]),
            ("no test", not self.final["production_test_accessed"]),
            ("no blind", not self.final["blind_accessed"]),
            ("no optimizer", not self.final["optimizer_step_executed"]),
            ("no training", not self.final["training_started"]),
            ("TF1 regression",
             self.regression["tf1_regression"] == "PASS"),
            ("DPAR1 regression",
             self.regression["dpar1_regression"] == "PASS"),
            ("git diff --check", git_diff_ok),
            ("report completeness",
             all((REPORTS / f"{PREFIX}{name}").is_file()
                 for name in required_reports)),
        ]
        self.assertEqual(len(checks), 69)
        for index, (name, passed) in enumerate(checks, 1):
            with self.subTest(item=index, name=name):
                self.assertTrue(passed, f"{index}. {name}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
