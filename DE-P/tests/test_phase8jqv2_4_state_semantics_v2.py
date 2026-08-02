import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import yaml

from authoritative_dataset import FORMAL_DATASET_VERSION_V2
from authoritative_dataset.state_semantics_v2 import (
    goal_body_from_world, quaternion_wxyz_from_yaw,
    yaw_rotation_world_from_body,
)


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/"data/phase8_authoritative_v1"
SMOKE = ROOT/"data/phase8_authoritative_v2_smoke"
OLD_HASH = "a81319d734871c48d49058e22a36a733baead3d85db2ec39206221d3b154f723"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class StateSemanticsV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.semantic = json.loads((
            ROOT/"reports/phase8jqv2_4_smoke_semantic_validation.json"
        ).read_text())
        cls.integrity = json.loads((
            ROOT/"reports/phase8jqv2_4_smoke_generation_summary.json"
        ).read_text())
        cls.manifest = json.loads((
            SMOKE/"manifests/dataset_manifest.json"
        ).read_text())
        cls.config = yaml.safe_load((
            ROOT/"configs/phase8_authoritative_v2_generation.yaml"
        ).read_text())

    def test_old_root_manifest_unchanged(self):
        self.assertEqual(
            sha256(OLD/"manifests/dataset_manifest.json"), OLD_HASH
        )

    def test_new_dataset_version(self):
        self.assertEqual(
            self.manifest["dataset_version"], FORMAL_DATASET_VERSION_V2
        )

    def test_smoke_integrity_pass(self):
        self.assertEqual(self.integrity["status"], "PASS")

    def test_smoke_semantics_pass(self):
        self.assertEqual(self.semantic["status"], "PASS")

    def test_train_and_valid_count(self):
        self.assertEqual(self.integrity["counts"]["train_frames"], 300)
        self.assertEqual(self.integrity["counts"]["valid_frames"], 300)

    def test_three_reused_maps_per_split(self):
        for split in ("train", "valid"):
            self.assertEqual(
                self.semantic["splits"][split][
                    "authority_manifest_hash_count"
                ], 3
            )

    def test_motion_is_material(self):
        for split in ("train", "valid"):
            counts = self.semantic["splits"][split]["counts"]
            self.assertGreaterEqual(counts["moving_frames"], 225)
            self.assertGreaterEqual(counts["nonzero_velocity_frames"], 225)
            self.assertGreaterEqual(
                counts["nonzero_acceleration_frames"], 150
            )

    def test_hold_is_only_stationary_scenario(self):
        for split in ("train", "valid"):
            value = self.semantic["splits"][split]
            self.assertEqual(value["counts"]["stationary_sequences"], 1)
            self.assertEqual(
                value["scenario_counts"]["hold"]["moving_frames"], 0
            )

    def test_temporal_difference_consistency(self):
        for split in ("train", "valid"):
            checks = self.semantic["splits"][split]["checks"]
            self.assertTrue(checks["velocity_difference_consistent"])
            self.assertTrue(checks["acceleration_difference_consistent"])
            self.assertTrue(checks["timestamp_monotonic"])
            self.assertTrue(checks["dt_consistent"])

    def test_goal_transform_round_trip_nonzero_yaw(self):
        position = np.asarray([1.0, 2.0, 3.0])
        goal = np.asarray([4.0, -1.0, 5.0])
        yaw = .73
        body = goal_body_from_world(position, goal, yaw)
        reconstructed = yaw_rotation_world_from_body(yaw)@body
        np.testing.assert_allclose(
            reconstructed, goal-position, atol=1e-12
        )
        self.assertFalse(np.allclose(body, goal-position))

    def test_quaternion_order_is_wxyz(self):
        yaw = .8
        quaternion = quaternion_wxyz_from_yaw(yaw)
        np.testing.assert_allclose(
            quaternion,
            [np.cos(yaw/2), 0, 0, np.sin(yaw/2)],
            atol=1e-12,
        )

    def test_goal_transform_error_zero(self):
        for split in ("train", "valid"):
            maximum = self.semantic["splits"][split]["distributions"][
                "goal_body_transform_error_m"
            ]["maximum"]
            self.assertLessEqual(maximum, 1e-9)

    def test_goal_distribution_not_collapsed(self):
        for split in ("train", "valid"):
            value = self.semantic["splits"][split]["distributions"][
                "goal_distance_m"
            ]
            self.assertGreaterEqual(value["minimum"], 2.0)
            self.assertGreater(value["p95"], 6.0)
            self.assertLessEqual(value["maximum"], 8.0)

    def test_label_coverage(self):
        for split in ("train", "valid"):
            counts = self.semantic["splits"][split]["counts"]
            self.assertGreater(counts["nominal_frames"], 0)
            self.assertGreater(counts["stress_frames"], 0)
            self.assertGreater(counts["recoverable_frames"], 0)
            self.assertEqual(counts.get("unknown_frames", 0), 0)

    def test_label_partition(self):
        for split in ("train", "valid"):
            self.assertTrue(
                self.semantic["splits"][split]["checks"]["label_partition"]
            )

    def test_camera_pose_bound_to_state(self):
        for split in ("train", "valid"):
            self.assertTrue(
                self.semantic["splits"][split]["checks"][
                    "camera_pose_matches_persisted_state"
                ]
            )

    def test_no_actor_static_collision(self):
        self.assertEqual(self.integrity["counts"]["actor_collisions"], 0)

    def test_no_unknown_certificate(self):
        self.assertEqual(self.integrity["counts"]["unknown"], 0)

    def test_no_test_or_blind(self):
        self.assertEqual(self.integrity["test_access_count"], 0)
        self.assertEqual(self.integrity["blind_access_count"], 0)
        self.assertFalse(self.config["test_disabled"] is False)
        self.assertFalse(self.config["blind_disabled"] is False)

    def test_no_runtime_random_sampling(self):
        self.assertFalse(self.manifest["runtime_random_sampling"])

    def test_authority_source_frozen(self):
        self.assertEqual(
            self.manifest["authority_source_root_manifest_hash"], OLD_HASH
        )
        self.assertEqual(
            self.manifest["authority_source_dataset"],
            "phase8_authoritative_v1",
        )

    def test_split_completion_markers_unambiguous(self):
        completion = SMOKE/"generation_state/completion"
        self.assertTrue(
            (completion/"TRAIN_SPLIT_GENERATION_COMPLETE").is_file()
        )
        self.assertTrue(
            (completion/"VALID_SPLIT_GENERATION_COMPLETE").is_file()
        )

    def test_no_training_artifacts_in_smoke(self):
        self.assertFalse((SMOKE/"checkpoints").exists())
        self.assertFalse((SMOKE/"optimizer.pt").exists())


if __name__ == "__main__":
    unittest.main()
