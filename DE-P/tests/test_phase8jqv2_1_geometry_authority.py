import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DIAGNOSTICS = ROOT / "diagnostics/phase8jqv2_1"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def report(name):
    return load(REPORTS / name)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Phase8JQ2Point1GeometryAuthorityTests(unittest.TestCase):
    """Validate the mandatory stop route when static authority is absent."""

    @classmethod
    def setUpClass(cls):
        cls.entry = report("phase8jqv2_1_entry_gate.json")
        cls.fixture = report("phase8jqv2_1_fixture_lock.json")
        cls.provenance = report(
            "phase8jqv2_1_static_geometry_provenance.json"
        )
        cls.decision = report(
            "phase8jqv2_1_geometry_backend_decision.json"
        )
        cls.final = report("phase8jqv2_1_final_result.json")

    def assert_blocked(self, name):
        value = report(name)
        self.assertEqual(
            value["status"],
            "NOT_EXECUTED_GEOMETRY_AUTHORITY_GATE_FAIL",
        )
        self.assertFalse(value["training_executed"])
        self.assertFalse(value["production_test_used"])
        self.assertFalse(value["blind_used"])

    def test_01_geometry_provenance_hashes(self):
        q22 = REPORTS / "phase8jqv2_2_final_result.json"
        current = (
            report("phase8jqv2_2_simulator_integration.json")
            ["source_hashes"] if q22.is_file() else {}
        )
        q23_entry_path = REPORTS / "phase8jqv2_3_entry_gate.json"
        q23_entry = load(q23_entry_path) if q23_entry_path.is_file() else {}
        for name, expected in self.provenance["source_hashes"].items():
            if (
                name.endswith("/Simulator/src/src/sensor_simulator.cu")
                and "cpu_gpu_backend_hash" in q23_entry
            ):
                # Q2.3 changed only the empty-map sentinel to the exact
                # float32 maximum and records the new authorized backend hash.
                self.assertEqual(
                    sha256(name),
                    q23_entry["cpu_gpu_backend_hash"],
                    name,
                )
            elif name in current:
                # Q2.1 remains a historical snapshot. Q2.2 is an explicitly
                # authorized Simulator authority-contract modification.
                self.assertEqual(sha256(name), current[name], name)
            else:
                self.assertEqual(sha256(name), expected, name)

    def test_02_ply_occupancy_esdf_simulator_pairing_is_not_claimed(self):
        pairing = self.provenance["same_hash_provenance"]
        self.assertTrue(pairing["saved_ply_catalog_locked"])
        self.assertFalse(pairing["unfiltered_grid_hash_available"])
        self.assertFalse(
            pairing["simulator_static_collision_geometry_hash_available"]
        )
        self.assertFalse(pairing["offline_esdf_derived_artifact_persisted"])

    def test_03_map_origin_semantics_are_recorded(self):
        facts = self.provenance["facts"]
        self.assertTrue(facts["grid_origin_is_cloud_min_corner"])
        graph = self.provenance["geometry_graph"]
        esdf = next(row for row in graph if row.get("distance"))
        self.assertEqual(esdf["origin"], "filtered PLY min - map_expand_min")

    def test_04_voxel_center_convention_is_recorded(self):
        self.assertTrue(
            self.provenance["facts"]["grid_voxel_center_is_half_cell"]
        )

    def test_05_authoritative_static_contact_is_absent(self):
        self.assertTrue(
            self.provenance["facts"][
                "simulator_has_no_static_uav_collision_query"
            ]
        )
        self.assertFalse(self.provenance["static_geometry_authority_ready"])

    def test_06_exact_1cm_safe_is_not_claimed_without_authority(self):
        self.assert_blocked("phase8jqv2_1_synthetic_validation.json")

    def test_07_exact_1cm_penetration_is_not_claimed_without_authority(self):
        self.assert_blocked("phase8jqv2_1_synthetic_validation.json")

    def test_08_esdf_false_safe_is_not_labelled_without_authority(self):
        value = load(DIAGNOSTICS / "esdf_false_safe.json")
        self.assertEqual(
            value["status"],
            "NOT_EXECUTED_GEOMETRY_AUTHORITY_GATE_FAIL",
        )

    def test_09_one_sided_error_bound_is_not_fabricated(self):
        self.assert_blocked("phase8jqv2_1_esdf_error_envelope.json")

    def test_10_tolerance_termination_is_not_claimed(self):
        self.assert_blocked("phase8jqv2_1_synthetic_validation.json")

    def test_11_lazy_query_count_is_not_claimed(self):
        self.assert_blocked("phase8jqv2_1_synthetic_validation.json")

    def test_12_recursion_cache_is_not_claimed(self):
        self.assert_blocked("phase8jqv2_1_synthetic_validation.json")

    def test_13_maximum_depth_unknown_is_not_claimed(self):
        self.assert_blocked("phase8jqv2_1_unknown_analysis.json")

    def test_14_out_of_bounds_unknown_is_not_claimed(self):
        self.assert_blocked("phase8jqv2_1_unknown_analysis.json")
        value = load(DIAGNOSTICS / "out_of_bounds.json")
        self.assertEqual(
            value["status"],
            "NOT_EXECUTED_GEOMETRY_AUTHORITY_GATE_FAIL",
        )

    def test_15_non_finite_fail_closed_is_not_claimed(self):
        self.assert_blocked("phase8jqv2_1_synthetic_validation.json")

    def test_16_line_segment_certificate_is_not_claimed(self):
        self.assert_blocked("phase8jqv2_1_synthetic_validation.json")

    def test_17_polynomial_certificate_is_not_claimed(self):
        self.assert_blocked("phase8jqv2_1_synthetic_validation.json")

    def test_18_arc_length_bound_is_not_claimed(self):
        text = (
            REPORTS / "phase8jqv2_1_certificate_math.md"
        ).read_text(encoding="utf-8")
        self.assertIn("NOT EXECUTED", text)

    def test_19_synthetic_sphere_is_not_claimed(self):
        self.assert_blocked("phase8jqv2_1_synthetic_validation.json")

    def test_20_synthetic_box_is_not_claimed(self):
        self.assert_blocked("phase8jqv2_1_synthetic_validation.json")

    def test_21_synthetic_corner_is_not_claimed(self):
        self.assert_blocked("phase8jqv2_1_synthetic_validation.json")

    def test_22_synthetic_corridor_is_not_claimed(self):
        self.assert_blocked("phase8jqv2_1_synthetic_validation.json")

    def test_23_synthetic_tangent_is_not_claimed(self):
        self.assert_blocked("phase8jqv2_1_synthetic_validation.json")

    def test_24_certified_safe_false_safe_zero_is_not_fabricated(self):
        self.assert_blocked("phase8jqv2_1_real_map_validation.json")

    def test_25_confirmed_collision_false_collision_zero_not_fabricated(self):
        self.assert_blocked("phase8jqv2_1_real_map_validation.json")

    def test_26_unknown_budget_is_not_claimed(self):
        self.assert_blocked("phase8jqv2_1_unknown_analysis.json")

    def test_27_depth_monotonicity_is_not_claimed(self):
        self.assert_blocked("phase8jqv2_1_determinism_validation.json")

    def test_28_tolerance_monotonicity_is_not_claimed(self):
        self.assert_blocked("phase8jqv2_1_determinism_validation.json")

    def test_29_immutable_fixture_hash(self):
        self.assertEqual(self.fixture["status"], "PASS")
        self.assertTrue(all(self.fixture["checks"].values()))
        self.assertEqual(
            sha256(self.fixture["fixture"]),
            self.fixture["fixture_sha256"],
        )
        self.assertFalse(self.fixture["fixture_resampled"])

    def test_30_strict_checkpoint_load_remains_preserved(self):
        matrix = report("phase8jqv2_checkpoint_matrix.json")
        results = matrix["strict_load_results"]
        self.assertGreater(len(results), 0)
        self.assertTrue(all(row["strict_load"] for row in results))
        self.assert_blocked("phase8jqv2_1_checkpoint_matrix.json")

    def test_31_no_production_test(self):
        self.assertFalse(self.entry["production_test_used"])
        self.assertFalse(self.final["production_test_used"])

    def test_32_no_blind(self):
        self.assertFalse(self.entry["blind_used"])
        self.assertFalse(self.final["blind_used"])

    def test_33_v1_v2_and_s0_outputs_coexist_unchanged(self):
        for name, expected in self.entry["frozen_s0_hashes"].items():
            self.assertEqual(sha256(REPORTS / name), expected, name)
        self.assertTrue((ROOT / "policy/safety_evaluator_v2.py").is_file())
        self.assertTrue((ROOT / "loss/safety_geometry_v2.py").is_file())
        self.assertTrue((ROOT / "loss/static_continuous_v2_1.py").is_file())

    def test_34_deterministic_resume_not_rerun_after_failed_gate(self):
        old = report("phase8jqv2_determinism_validation.json")
        self.assertTrue(old["same_config_resume_supported"])
        self.assert_blocked("phase8jqv2_1_determinism_validation.json")

    def test_35_report_completeness_and_final_stop(self):
        report_names = (
            "phase8jqv2_1_entry_gate.json",
            "phase8jqv2_1_fixture_lock.json",
            "phase8jqv2_1_static_geometry_provenance.json",
            "phase8jqv2_1_simulator_collision_contract.md",
            "phase8jqv2_1_geometry_registration.json",
            "phase8jqv2_1_esdf_error_envelope.json",
            "phase8jqv2_1_geometry_backend_decision.json",
            "phase8jqv2_1_certificate_math.md",
            "phase8jqv2_1_synthetic_validation.json",
            "phase8jqv2_1_real_map_validation.json",
            "phase8jqv2_1_unknown_analysis.json",
            "phase8jqv2_1_determinism_validation.json",
            "phase8jqv2_1_checkpoint_matrix.json",
            "phase8jqv2_1_static_validation.json",
            "phase8jqv2_1_dynamic_validation.json",
            "phase8jqv2_1_label_audit.json",
            "phase8jqv2_1_v2_delta.json",
            "phase8jqv2_1_final_result.json",
            "phase8jqv2_1_final_recommendation.md",
            "phase8jqv2_1_final_readiness.md",
        )
        diagnostic_names = (
            "esdf_false_safe.json",
            "esdf_false_unsafe.json",
            "geometry_disagreements.json",
            "certificate_unknown.json",
            "out_of_bounds.json",
            "static_window_flips.json",
        )
        self.assertTrue(all((REPORTS / name).is_file()
                            for name in report_names))
        self.assertTrue(all((DIAGNOSTICS / name).is_file()
                            for name in diagnostic_names))
        self.assertEqual(self.decision["selected_scheme"], "D")
        self.assertFalse(self.decision["ply_nearest_promoted_to_authority"])
        self.assertFalse(self.decision["dense_esdf_promoted_to_authority"])
        self.assertFalse(self.decision["occupancy_promoted_to_authority"])
        self.assertEqual(self.final["status"], "FAIL")
        self.assertEqual(
            self.final["primary_cause"], "static_geometry_authority"
        )
        self.assertFalse(self.final["static_geometry_authority_ready"])
        self.assertFalse(self.final["static_numerical_rebaseline_ready"])
        self.assertFalse(self.final["training_executed"])
        self.assertIsNone(self.final["next_allowed_phase"])


if __name__ == "__main__":
    unittest.main()
