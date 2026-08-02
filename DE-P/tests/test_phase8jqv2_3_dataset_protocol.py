import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from authoritative_dataset import DATASET_VERSION, PROTOCOL_VERSION
from authoritative_dataset.continuous_v1 import (
    ContinuousState, certify_curve, quintic_position,
)
from authoritative_dataset.exact_backend_v1 import ExactAuthorityBVH
from authoritative_dataset.loader_v1 import (
    AuthoritativePilotDataset, sha256,
)


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DATASET = ROOT / f"data/{DATASET_VERSION}"


def load(name):
    return json.loads((REPORTS/name).read_text())


class _ConstantBackend:
    def __init__(self, gap=.1, collision=False):
        self.gap = gap
        self.collision = collision

    def query_one(self, _position, _radius):
        return {
            "collision": self.collision,
            "minimum_gap_m": self.gap,
            "contacted_voxel_index": [0, 0, 0]
            if self.collision else [-1, -1, -1],
        }


class AuthoritativeDatasetProtocolV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader = AuthoritativePilotDataset(DATASET)
        cls.first = cls.loader[0]["metadata"]

    def test_01_authority_hash_binding(self):
        self.assertRegex(self.first["authority_manifest_hash"],
                         r"^[0-9a-f]{64}$")
        self.assertRegex(self.first["occupancy_hash"], r"^[0-9a-f]{64}$")

    def test_02_empty_map_sentinel(self):
        report = load("phase8jqv2_3_empty_space_contract.json")
        self.assertTrue(report["minimum_gap_mask_required"])
        self.assertFalse(report["included_in_clearance_mean"])

    def test_03_empty_cpu_gpu_offline_equal(self):
        self.assertTrue(load(
            "phase8jqv2_3_empty_space_contract.json"
        )["cpu_gpu_offline_exact_equal"])

    def test_04_fullsize_authority_load(self):
        report = load("phase8jqv2_3_fullsize_scalability.json")
        self.assertEqual(len(report["maps"]), 2)
        self.assertTrue(all(row["bvh_load_seconds"] > 0
                            for row in report["maps"]))

    def test_05_fullsize_exact_query(self):
        report = load("phase8jqv2_3_fullsize_scalability.json")
        self.assertTrue(all(row["bvh_collision_mismatch"] == 0
                            for row in report["maps"]))
        self.assertTrue(all(row["bvh_gap_max_error_m"] <= 1e-6
                            for row in report["maps"]))

    def test_06_schema_version(self):
        self.assertEqual(
            load("phase8jqv2_3_dataset_schema.json")["schema_version"],
            "authoritative_dataset_schema_v1")

    def test_07_split_maps_do_not_overlap(self):
        split = load("phase8jqv2_3_split_protocol.json")
        seeds = [
            seed for values in split["seed_registry"].values()
            for seed in values
        ]
        self.assertEqual(len(seeds), len(set(seeds)))

    def test_08_occupancy_hashes_do_not_overlap(self):
        values = load(
            "phase8jqv2_3_pilot_manifest.json")["authority_hashes"]
        self.assertEqual(len(values), len(set(values)))

    def test_09_state_fields_persisted(self):
        for name in (
            "position_world", "velocity_world", "velocity_body",
            "acceleration_world", "acceleration_body",
            "angular_velocity_body", "previous_command",
        ):
            self.assertIn(name, self.first)

    def test_10_goal_fields_persisted(self):
        for name in (
            "goal_world", "goal_body", "goal_type",
            "minimum_progress_requirement_m",
        ):
            self.assertIn(name, self.first)

    def test_11_no_runtime_random_state(self):
        self.assertFalse(
            json.loads((DATASET/"manifests/dataset_manifest.json"
                        ).read_text())["runtime_random_sampling"])
        self.assertIn("rng_seed", self.first)

    def test_12_latency_state(self):
        self.assertGreater(self.first["measured_command_latency_s"], 0)
        self.assertIn("first_controllable_state", self.first)

    def test_13_sensor_state_timestamps(self):
        self.assertEqual(self.first["sensor_timestamp_ns"],
                         self.first["state_timestamp_ns"])
        self.assertEqual(self.first["odometry_timestamp_ns"],
                         self.first["state_timestamp_ns"])

    def test_14_actor_timestamps(self):
        dynamic = next(
            self.loader[i]["metadata"] for i in range(len(self.loader))
            if self.loader[i]["metadata"]["actor_metadata"])
        actor = dynamic["actor_metadata"][0]
        self.assertEqual(len(actor["future_timestamps_ns"]),
                         len(actor["future_world"]))

    def test_15_static_continuous_checker(self):
        result = certify_curve(
            lambda _time: np.zeros(3), _ConstantBackend(gap=1),
            1.0)
        self.assertEqual(result.state, ContinuousState.CERTIFIED_SAFE)

    def test_16_polynomial_interval_checker(self):
        curve = quintic_position([0, 0, 0], [.01, 0, 0], 1)
        result = certify_curve(curve, _ConstantBackend(gap=1), 1)
        self.assertEqual(result.state, ContinuousState.CERTIFIED_SAFE)

    def test_17_unknown_classification(self):
        result = certify_curve(
            lambda t: np.asarray([t, 0, 0]),
            _ConstantBackend(gap=.1), 1, max_depth=0)
        self.assertEqual(result.state, ContinuousState.UNKNOWN)
        self.assertEqual(result.unknown_reason,
                         "maximum_depth_without_proof")

    def test_18_static_safety_certificate(self):
        report = load("phase8jqv2_3_pilot_feasibility.json")
        self.assertGreaterEqual(
            report["static_safety_only_success_fraction"], .99)

    def test_19_static_progress_certificate(self):
        certificate = json.loads(
            (DATASET/self.first["certificate_path"]).read_text())
        self.assertTrue(certificate["static_progress_success"])

    def test_20_recovery_certificate(self):
        certificates = [
            json.loads(path.read_text())
            for path in (DATASET/"certificates").glob("*.json")]
        self.assertTrue(any(row["recovery_certificate"]
                            for row in certificates))

    def test_21_actor_static_collision(self):
        report = load("phase8jqv2_3_pilot_dynamic_validation.json")
        self.assertEqual(report["actor_static_collisions"], 0)

    def test_22_actor_actor_collision(self):
        report = load("phase8jqv2_3_pilot_dynamic_validation.json")
        self.assertEqual(report["actor_actor_collisions"], 0)

    def test_23_dynamic_preventability_certificate(self):
        report = load("phase8jqv2_3_pilot_feasibility.json")
        self.assertGreaterEqual(report["dynamic_joint_success_fraction"], .99)

    def test_24_certificate_deterministic(self):
        report = load("phase8jqv2_3_pilot_determinism.json")
        self.assertTrue(report["semantic_and_file_hash_equal"])

    def test_25_derived_esdf_non_authoritative(self):
        report = load("phase8jqv2_3_derived_esdf_manifest.json")
        self.assertFalse(report["authoritative"])
        self.assertTrue(report["exact_verifier_required"])

    def test_26_dataset_manifest_hash(self):
        report = load("phase8jqv2_3_pilot_manifest.json")
        self.assertEqual(
            sha256(DATASET/"manifests/dataset_manifest.json"),
            report["dataset_manifest_hash"])

    def test_27_sequence_manifest_hash(self):
        manifest = json.loads(
            (DATASET/"manifests/dataset_manifest.json").read_text())
        row = manifest["sequences"][0]
        self.assertEqual(sha256(DATASET/row["manifest"]), row["sha256"])

    def test_28_frame_manifest_hash(self):
        manifest = json.loads(
            (DATASET/"manifests/dataset_manifest.json").read_text())
        sequence = json.loads(
            (DATASET/manifest["sequences"][0]["manifest"]).read_text())
        row = sequence["frames"][0]
        self.assertEqual(sha256(DATASET/row["path"]), row["sha256"])

    def _mismatch_copy(self, field, value, root_field=False):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)/"dataset"
        shutil.copytree(DATASET, root)
        dataset_path = root/"manifests/dataset_manifest.json"
        dataset = json.loads(dataset_path.read_text())
        if root_field:
            dataset[field] = value
            dataset_path.write_text(json.dumps(
                dataset, indent=2, sort_keys=True)+"\n")
            return temporary, root
        sequence_path = root/dataset["sequences"][0]["manifest"]
        sequence = json.loads(sequence_path.read_text())
        frame_path = root/sequence["frames"][0]["path"]
        frame = json.loads(frame_path.read_text())
        frame[field] = value
        frame_path.write_text(json.dumps(
            frame, indent=2, sort_keys=True)+"\n")
        sequence["frames"][0]["sha256"] = sha256(frame_path)
        sequence_path.write_text(json.dumps(
            sequence, indent=2, sort_keys=True)+"\n")
        dataset["sequences"][0]["sha256"] = sha256(sequence_path)
        dataset_path.write_text(json.dumps(
            dataset, indent=2, sort_keys=True)+"\n")
        return temporary, root

    def test_29_map_uuid_mismatch_fails(self):
        temporary, root = self._mismatch_copy("map_uuid", "wrong")
        try:
            loader = AuthoritativePilotDataset(root)
            with self.assertRaises(RuntimeError):
                loader[0]
        finally:
            temporary.cleanup()

    def test_30_authority_mismatch_fails(self):
        temporary, root = self._mismatch_copy(
            "authority_manifest_hash", "0"*64)
        try:
            loader = AuthoritativePilotDataset(root)
            with self.assertRaises(RuntimeError):
                loader[0]
        finally:
            temporary.cleanup()

    def test_31_protocol_mismatch_fails(self):
        temporary, root = self._mismatch_copy(
            "protocol_version", "wrong", root_field=True)
        try:
            with self.assertRaises(RuntimeError):
                AuthoritativePilotDataset(root)
        finally:
            temporary.cleanup()

    def test_32_worker_batch_determinism(self):
        report = load("phase8jqv2_3_pilot_loader_validation.json")
        self.assertEqual(report["worker_counts"], [0, 1, 4, 8])
        self.assertEqual(report["batch_sizes"], [1, 16, 32, 64])
        self.assertTrue(report["schedule_semantic_hash_equal"])

    def test_33_no_split_leakage(self):
        self.assertEqual(json.loads(
            (ROOT/"diagnostics/phase8jqv2_3/split_leakage.json"
             ).read_text())["count"], 0)

    def test_34_legacy_fallback_forbidden(self):
        self.assertFalse(load(
            "phase8jqv2_3_pilot_loader_validation.json"
        )["legacy_fallback"])
        with self.assertRaises(FileNotFoundError):
            AuthoritativePilotDataset("/home/zjh/YOPO/dataset")

    def test_35_no_production_test(self):
        self.assertFalse(load(
            "phase8jqv2_3_final_result.json")["production_test_used"])

    def test_36_no_blind(self):
        self.assertFalse(
            load("phase8jqv2_3_final_result.json")["blind_used"])

    def test_37_no_v2_1_evaluator(self):
        self.assertFalse(load(
            "phase8jqv2_3_final_result.json"
        )["safety_evaluator_v2_1_created"])

    def test_38_no_training(self):
        final = load("phase8jqv2_3_final_result.json")
        self.assertFalse(final["training_executed"])
        self.assertFalse(final["network_weights_modified"])

    def test_39_pilot_reproducibility(self):
        self.assertEqual(
            load("phase8jqv2_3_pilot_determinism.json")["status"],
            "PASS")

    def test_40_report_completeness(self):
        required = (
            "phase8jqv2_3_entry_gate.json",
            "phase8jqv2_3_dataset_protocol.md",
            "phase8jqv2_3_dataset_schema.json",
            "phase8jqv2_3_split_protocol.json",
            "phase8jqv2_3_state_sampling_contract.json",
            "phase8jqv2_3_static_feasibility_contract.md",
            "phase8jqv2_3_dynamic_feasibility_contract.md",
            "phase8jqv2_3_empty_space_contract.json",
            "phase8jqv2_3_fullsize_scalability.json",
            "phase8jqv2_3_continuous_checker_validation.json",
            "phase8jqv2_3_derived_esdf_manifest.json",
            "phase8jqv2_3_pilot_manifest.json",
            "phase8jqv2_3_pilot_static_validation.json",
            "phase8jqv2_3_pilot_dynamic_validation.json",
            "phase8jqv2_3_pilot_feasibility.json",
            "phase8jqv2_3_pilot_determinism.json",
            "phase8jqv2_3_pilot_loader_validation.json",
            "phase8jqv2_3_generation_resource_plan.json",
            "phase8jqv2_3_final_result.json",
            "phase8jqv2_3_final_recommendation.md",
            "phase8jqv2_3_final_readiness.md",
        )
        self.assertTrue(all((REPORTS/name).is_file() for name in required))
        final = load("phase8jqv2_3_final_result.json")
        self.assertEqual(final["status"], "PASS")
        self.assertEqual(
            final["next_allowed_phase"],
            "phase8jqv2_4_authoritative_dataset_generation")


if __name__ == "__main__":
    unittest.main()
