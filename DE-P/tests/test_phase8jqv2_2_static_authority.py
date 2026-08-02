import json
from pathlib import Path
import tempfile
import unittest
import uuid

import numpy as np

from geometry_authority.static_v1 import (
    AUTHORITY_VERSION, AuthorityArtifactError, StaticAuthorityMap,
    build_authority_artifact, sha256,
)


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def load(name):
    return json.loads((REPORTS / name).read_text())


class StaticAuthorityV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name) / "authority"
        cls.map_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, "q22-tests"))
        cls.cloud = np.asarray([
            [0, 0, 0], [2, 2, 2], [1, 1, 1], [1.1, 1, 1],
        ], dtype=np.float32)
        cls.metadata = build_authority_artifact(
            cls.root, cls.cloud, map_uuid=cls.map_uuid,
            generator_seed=1, map_id="test",
            generator_source_hash="a" * 64,
            generator_config_hash="b" * 64,
            parent_git_commit="c" * 40,
        )
        cls.authority = StaticAuthorityMap(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def query(self, center, radius=.3):
        return self.authority.query(np.asarray(center), radius)

    def test_01_authority_schema(self):
        self.assertEqual(self.metadata["authority_version"],
                         AUTHORITY_VERSION)

    def test_02_canonical_serialization(self):
        self.assertEqual(self.metadata["storage_order"],
                         "x_major_y_middle_z_fast")
        self.assertEqual(self.metadata["bit_packing"], "lsb_first")

    def test_03_deterministic_builder(self):
        before = {p.name: sha256(p) for p in self.root.iterdir()}
        build_authority_artifact(
            self.root, self.cloud, map_uuid=self.map_uuid,
            generator_seed=1, map_id="test",
            generator_source_hash="a" * 64,
            generator_config_hash="b" * 64,
            parent_git_commit="c" * 40,
        )
        after = {p.name: sha256(p) for p in self.root.iterdir()}
        self.assertEqual(before, after)

    def test_04_raw_cloud_hash(self):
        self.assertEqual(
            sha256(self.root / "raw_cloud.bin"),
            self.metadata["source_cloud_hash"])

    def test_05_occupancy_hash(self):
        self.assertEqual(
            sha256(self.root / "occupancy.bin"),
            self.metadata["occupancy_hash"])

    def test_06_point_to_voxel_indexing(self):
        self.assertTrue(self.authority.flat.reshape(
            tuple(self.authority.dimensions))[10, 10, 10])

    def test_07_voxel_aabb_bounds(self):
        result = self.query([1.05, 1.05, 1.05])
        np.testing.assert_allclose(
            result.contacted_voxel_bounds,
            [[1, 1, 1], [1.1, 1.1, 1.1]], atol=2e-7)

    def test_08_sphere_one_cm_safe(self):
        result = self.query([.69, 1.05, 1.05])
        self.assertFalse(result.collision)
        self.assertAlmostEqual(result.minimum_gap_m, .01, places=6)

    def test_09_exact_tangent(self):
        self.assertTrue(self.query([.7, 1.05, 1.05]).collision)

    def test_10_one_cm_penetration(self):
        result = self.query([.71, 1.05, 1.05])
        self.assertTrue(result.collision)
        self.assertLess(result.minimum_gap_m, 0)

    def test_11_center_inside_voxel(self):
        self.assertTrue(self.query([1.05, 1.05, 1.05]).collision)

    def test_12_corner_contact(self):
        offset = .3 / np.sqrt(3)
        self.assertTrue(self.query([1.1 + offset] * 3).collision)

    def test_13_edge_contact(self):
        offset = .3 / np.sqrt(2)
        self.assertTrue(
            self.query([1.1 + offset, 1.1 + offset, 1.05]).collision)

    def test_14_multi_voxel_contact(self):
        result = self.query([1.1, 1.05, 1.05], 0)
        self.assertTrue(result.collision)
        np.testing.assert_array_equal(
            result.contacted_voxel_index, [10, 10, 10])

    def test_15_oob_fail_closed(self):
        result = self.query([.1, .1, .1])
        self.assertTrue(result.collision)
        self.assertTrue(result.out_of_bounds)

    def test_16_cpu_gpu_bool_consistency(self):
        self.assertEqual(
            load("phase8jqv2_2_backend_equivalence.json")
            ["mismatch_count"], 0)

    def test_17_cpu_offline_bool_consistency(self):
        report = load("phase8jqv2_2_backend_equivalence.json")
        self.assertTrue(all(
            row["cpu"]["collision_mismatch"] == 0
            for row in report["maps"]))

    def test_18_gpu_offline_bool_consistency(self):
        report = load("phase8jqv2_2_backend_equivalence.json")
        self.assertTrue(all(
            row["gpu"]["collision_mismatch"] == 0
            for row in report["maps"]))

    def test_19_minimum_gap_tolerance(self):
        report = load("phase8jqv2_2_backend_equivalence.json")
        self.assertLessEqual(
            report["minimum_gap_max_abs_error_m"],
            report["minimum_gap_tolerance_m"])

    def test_20_contacted_voxel_consistency(self):
        report = load("phase8jqv2_2_backend_equivalence.json")
        self.assertTrue(all(
            row[backend]["contacted_voxel_mismatch"] == 0
            for row in report["maps"] for backend in ("cpu", "gpu")))

    def test_21_raycast_occupancy_hash(self):
        report = load("phase8jqv2_2_sensor_collision_coherence.json")
        self.assertTrue(report["same_gridmap_loader"])

    def test_22_collision_occupancy_hash(self):
        report = load("phase8jqv2_2_pilot_cross_backend_validation.json")
        self.assertTrue(all(
            row["occupancy_hash"] == row["collision_occupancy_hash"]
            for row in report["maps"]))

    def test_23_dataset_metadata_authority_hash(self):
        integration = load("phase8jqv2_2_simulator_integration.json")
        self.assertTrue(integration["dataset_metadata_authority_hash"])

    def test_24_map_uuid_mismatch_fails(self):
        with self.assertRaises(AuthorityArtifactError):
            StaticAuthorityMap(self.root, expected_map_uuid=str(uuid.uuid4()))

    def test_25_config_hash_mismatch_fails(self):
        with self.assertRaises(AuthorityArtifactError):
            StaticAuthorityMap(
                self.root, expected_generator_config_hash="d" * 64)

    def test_26_source_hash_mismatch_fails(self):
        with self.assertRaises(AuthorityArtifactError):
            StaticAuthorityMap(
                self.root, expected_generator_source_hash="d" * 64)

    def test_27_artifact_interruption_recovery(self):
        stale = self.root.parent / f".{self.root.name}.staging-999"
        stale.mkdir()
        (stale / "partial").write_text("partial")
        self.assertEqual(StaticAuthorityMap(self.root).metadata["map_uuid"],
                         self.map_uuid)

    def test_28_atomic_write(self):
        self.assertFalse(any(
            path.name.endswith(".tmp") for path in self.root.iterdir()))
        self.assertTrue((self.root / "authority_manifest.json").is_file())

    def test_29_legacy_ply_not_authority(self):
        self.assertFalse(
            load("phase8jqv2_2_legacy_geometry_status.json")
            ["legacy_ply_is_authority"])

    def test_30_legacy_esdf_not_authority(self):
        self.assertFalse(
            load("phase8jqv2_2_legacy_geometry_status.json")
            ["legacy_esdf_is_authority"])

    def test_31_no_production_test(self):
        self.assertFalse(
            load("phase8jqv2_2_final_result.json")
            ["production_test_used"])

    def test_32_no_blind(self):
        self.assertFalse(
            load("phase8jqv2_2_final_result.json")["blind_used"])

    def test_33_v1_v2_preserved(self):
        final = load("phase8jqv2_2_final_result.json")
        self.assertTrue(final["gates"]["v1_v2_preserved"])
        self.assertTrue((ROOT / "policy/safety_evaluator_v2.py").is_file())

    def test_34_no_v2_1_created(self):
        self.assertFalse(
            load("phase8jqv2_2_final_result.json")["v2_1_created"])
        self.assertFalse(
            (ROOT / "policy/safety_evaluator_v2_1.py").exists())

    def test_35_no_training(self):
        final = load("phase8jqv2_2_final_result.json")
        self.assertFalse(final["training_executed"])
        self.assertFalse(final["score_training_executed"])
        self.assertFalse(final["network_weights_modified"])

    def test_36_report_completeness(self):
        names = (
            "phase8jqv2_2_entry_gate.json",
            "phase8jqv2_2_legacy_geometry_status.json",
            "phase8jqv2_2_authority_spec.md",
            "phase8jqv2_2_oob_contract.json",
            "phase8jqv2_2_artifact_schema.json",
            "phase8jqv2_2_builder_determinism.json",
            "phase8jqv2_2_simulator_integration.json",
            "phase8jqv2_2_backend_equivalence.json",
            "phase8jqv2_2_performance.json",
            "phase8jqv2_2_synthetic_validation.json",
            "phase8jqv2_2_pilot_map_manifest.json",
            "phase8jqv2_2_pilot_cross_backend_validation.json",
            "phase8jqv2_2_sensor_collision_coherence.json",
            "phase8jqv2_2_final_result.json",
            "phase8jqv2_2_final_recommendation.md",
            "phase8jqv2_2_final_readiness.md",
        )
        self.assertTrue(all((REPORTS / name).is_file() for name in names))
        final = load("phase8jqv2_2_final_result.json")
        self.assertEqual(final["status"], "PASS")
        self.assertEqual(
            final["next_allowed_phase"],
            "phase8jqv2_3_authoritative_dataset_protocol")


if __name__ == "__main__":
    unittest.main()
