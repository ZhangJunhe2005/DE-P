import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def load(name):
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Phase8JV2CapacityGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.entry = load("phase8jv2_entry_gate.json")
        cls.baseline = load("phase8jv2_baseline_reproduction.json")
        cls.oracle = load("phase8jv2_capacity_oracle.json")
        cls.gate = load("phase8jv2_capacity_gate.json")
        cls.final = load("phase8jv2_final_result.json")

    def test_01_entry_and_baseline_pass(self):
        self.assertEqual(self.entry["status"], "PASS")
        self.assertTrue(all(self.entry["checks"].values()))
        self.assertEqual(self.baseline["status"], "PASS")
        self.assertTrue(all(self.baseline["checks"].values()))

    def test_02_frozen_inputs_and_checkpoints_still_match(self):
        for relative, expected in self.entry["required_input_hashes"].items():
            self.assertEqual(sha256(ROOT / relative), expected, relative)
        self.assertEqual(len(self.entry["checkpoint_audit"]), 10)
        for row in self.entry["checkpoint_audit"]:
            self.assertTrue(row["available"])
            self.assertTrue(row["hash_match"])
            self.assertEqual(sha256(row["path"]), row["expected_sha256"])
        self.assertFalse(self.entry["phase8b"]["available"])

    def test_03_v2_baseline_is_exact(self):
        metrics = self.baseline["metrics"]
        expected = {
            "valid_estimated_planning_joint": (516, 2052),
            "valid_gt_physical_joint": (501, 2052),
            "preventable_estimated_planning": (367, 1903),
            "combined_selection": (746, 2052),
            "physical_label_inversion": (0, 2052),
            "planning_label_inversion": (0, 2052),
            "valid_static_decision": (3680, 10000),
            "already_unsafe": (142, 2052),
            "valid_static_corrected": (2697, 10000),
        }
        for name, (numerator, denominator) in expected.items():
            self.assertEqual(metrics[name]["numerator"], numerator)
            self.assertEqual(metrics[name]["denominator"], denominator)
        self.assertEqual(
            self.baseline["semantic_determinism_hash"],
            "41ff8b9203ca7fe92325addae7be2fef1b51569281763fa34a1a78915871bd45",
        )

    def test_04_capacity_audit_is_full_and_denominators_are_fixed(self):
        self.assertEqual(self.oracle["audit_scope"], "FULL_VALIDATION")
        self.assertTrue(self.oracle["capacity_audit_complete"])
        for metrics in self.gate["scheme_metrics"].values():
            self.assertEqual(metrics["valid_static"]["denominator"], 10000)
            self.assertEqual(
                metrics["valid_estimated_preventable"]["denominator"], 1903
            )
            self.assertEqual(
                metrics["valid_gt_preventable"]["denominator"], 1903
            )

    def test_05_c2_fixed_sobol_results_are_regression_locked(self):
        expected = {
            "c2_sobol_64": (0.2894, 0.10299527062532843, 0.09721492380451918),
            "c2_sobol_128": (0.2811, 0.10299527062532843, 0.09721492380451918),
            "c2_sobol_256": (0.2730, 0.10299527062532843, 0.09721492380451918),
            "c2_sobol_512": (0.2659, 0.10299527062532843, 0.09721492380451918),
        }
        for name, values in expected.items():
            metrics = self.gate["scheme_metrics"][name]
            actual = (
                metrics["valid_static"]["fraction"],
                metrics["valid_estimated_preventable"]["fraction"],
                metrics["valid_gt_preventable"]["fraction"],
            )
            for observed, wanted in zip(actual, values):
                self.assertAlmostEqual(observed, wanted)

    def test_06_no_bounded_scheme_passes_all_three_thresholds(self):
        self.assertFalse(self.gate["v2_capacity_sufficient"])
        self.assertEqual(self.gate["status"], "FAIL")
        self.assertTrue(self.gate["scheme_checks"])
        self.assertTrue(all(
            not checks["all_three"]
            for checks in self.gate["scheme_checks"].values()
        ))

    def test_07_latency_and_semantics_are_retained(self):
        checks = self.gate["supporting_checks"]
        for name in (
            "continuous_collision_enabled",
            "latency_prefix_44ms_enabled",
            "uav_radius_0_3m",
            "exact_actor_geometry",
            "estimated_covariance_applied_once",
            "gt_uncertainty_zero",
            "no_target_dynamic_cost_exactly_zero",
            "c2_512_generation_p95_below_control_period",
        ):
            self.assertTrue(checks[name], name)
        self.assertFalse(checks["maps_0_through_14_all_present"])
        self.assertEqual(
            self.oracle["maps_present"]["absent_from_validation"], [10, 11]
        )

    def test_08_breakdowns_and_required_diagnostics_exist(self):
        c2 = self.oracle["c2_dense_existing_parameterization"]
        self.assertEqual(
            c2["valid_static"]["c0_breakdown"]["record_count"], 3680
        )
        self.assertEqual(
            c2["valid_estimated"]["c0_breakdown"]["record_count"], 367
        )
        self.assertEqual(c2["valid_gt"]["c0_breakdown"]["record_count"], 352)
        for name in (
            "static_coverage_failures.json",
            "preventable_coverage_failures.json",
            "recovery_failures.json",
            "candidate_collapse.json",
            "remaining_model_ranking_failures.json",
            "static_score_failures.json",
            "gt_estimated_disagreements.json",
        ):
            self.assertTrue((ROOT / "diagnostics/phase8jv2" / name).is_file())

    def test_09_failed_a0_stops_every_downstream_stage(self):
        self.assertEqual(self.final["status"], "FAIL")
        self.assertFalse(self.final["v2_capacity_sufficient"])
        self.assertFalse(self.final["coverage_training_executed"])
        self.assertFalse(self.final["score_training_executed"])
        self.assertFalse(self.final["candidate_generator_frozen"])
        self.assertIsNone(self.final["next_allowed_phase"])
        for name in (
            "phase8jv2_train_actionability_manifest.json",
            "phase8jv2_surrogate_alignment.json",
            "phase8jv2_gradient_conflict_audit.json",
            "phase8jv2_coverage_ablation.json",
            "phase8jv2_coverage_three_seed_summary.json",
            "phase8jv2_coverage_gate.json",
            "phase8jv2_candidate_generator_freeze.json",
            "phase8jv2_score_ablation.json",
            "phase8jv2_score_three_seed_summary.json",
            "phase8jv2_score_gate.json",
        ):
            self.assertEqual(
                load(name)["status"],
                "NOT_EXECUTED_A0_CAPACITY_GATE_FAIL",
                name,
            )

    def test_10_no_forbidden_path_was_used(self):
        self.assertFalse(self.oracle["network_weights_modified"])
        self.assertFalse(self.oracle["coverage_training_executed"])
        self.assertFalse(self.oracle["score_training_executed"])
        self.assertFalse(self.oracle["production_test_used"])
        self.assertFalse(self.oracle["blind_used"])
        self.assertFalse(self.final["dataset_rebuilt"])
        self.assertFalse(self.final["long_training_started"])


if __name__ == "__main__":
    unittest.main()
