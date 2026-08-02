import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from loss.static_continuous_v2_1 import (
    CertificateState, IMPLEMENTATION_VERSION,
    certify_dyadic_samples, combine_segment_certificates,
)
from policy.static_v2_fixture import (
    FIXTURE_VERSION, StaticV2FixtureDataset, fixture_semantic_hash,
    per_index_seed,
)


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIXTURE = ROOT / "artifacts/phase8jv2s0/static_valid_fixture_v1.npz"


def load(name):
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Phase8JV2S0StaticIntegrityTests(unittest.TestCase):
    def test_01_fixture_is_complete_and_self_authenticating(self):
        dataset = StaticV2FixtureDataset(FIXTURE, cache_size=0)
        self.assertEqual(len(dataset), 10000)
        self.assertEqual(
            str(dataset.arrays["fixture_version"]), FIXTURE_VERSION
        )
        self.assertEqual(
            fixture_semantic_hash(dataset.arrays), dataset.semantic_hash
        )
        self.assertEqual(len(set(dataset.arrays["fixture_id"].tolist())), 10000)

    def test_02_per_index_seed_is_deterministic_and_index_sensitive(self):
        first = per_index_seed(1, 2, 3, "a" * 64)
        self.assertEqual(first, per_index_seed(1, 2, 3, "a" * 64))
        self.assertNotEqual(first, per_index_seed(1, 2, 4, "a" * 64))

    def test_03_fixture_loader_schedule_gate_passed(self):
        report = load("phase8jv2s0_static_fixture_determinism.json")
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(len(report["combinations"]), 20)
        self.assertTrue(all(report["checks"].values()))

    def test_04_versioned_checker_certifies_safe(self):
        positions = np.asarray([[0, 0, 0], [.01, 0, 0]])
        result = certify_dyadic_samples(positions, [.5, .5])
        self.assertEqual(result.state, CertificateState.CERTIFIED_SAFE)
        self.assertEqual(IMPLEMENTATION_VERSION,
                         "phase8jqv2_static_continuous_v2_1")

    def test_05_versioned_checker_confirms_only_queried_collision(self):
        positions = np.asarray([[0, 0, 0], [.01, 0, 0]])
        result = certify_dyadic_samples(positions, [.5, .2])
        self.assertEqual(result.state, CertificateState.CONFIRMED_COLLISION)

    def test_06_versioned_checker_preserves_unknown(self):
        positions = np.asarray([[0, 0, 0], [.02, 0, 0]])
        result = certify_dyadic_samples(positions, [.31, .31])
        self.assertEqual(result.state, CertificateState.UNKNOWN)
        combined = combine_segment_certificates([result])
        self.assertEqual(combined, CertificateState.UNKNOWN)

    def test_07_full_conformance_selects_route_a(self):
        report = load("phase8jv2s0_static_continuous_conformance.json")
        self.assertEqual(report["scope"], "FULL_C0_FAILURE_SET")
        self.assertEqual(
            report["status"], "FAIL_NUMERICAL_REBASELINE_REQUIRED"
        )
        self.assertEqual(report["audited_failure_window_count"], 3683)
        self.assertEqual(
            report["comparison"][
                "current_unsafe_dense_128_safe_trajectories"
            ],
            2645,
        )
        self.assertEqual(
            report["comparison"]["current_unsafe_dense_128_safe_windows"],
            739,
        )
        self.assertEqual(
            report["next_allowed_phase"],
            "phase8jqv2_1_static_numerical_rebaseline",
        )
        self.assertFalse(
            report["versioned_repair"]["integrated_into_v2_evaluator"]
        )

    def test_08_old_reports_and_frozen_sources_are_unchanged(self):
        entry = load("phase8jv2s0_entry_gate.json")
        for name, expected in entry["input_report_hashes"].items():
            self.assertEqual(sha256(REPORTS / name), expected, name)
        # S0 adds a versioned checker but must not change the frozen evaluator.
        for name in (
            "policy/safety_evaluator_v2.py",
            "loss/safety_geometry_v2.py",
            "policy/dep_dataset.py",
        ):
            self.assertEqual(sha256(ROOT / name), entry["source_hashes"][name])

    def test_09_downstream_work_is_explicitly_blocked(self):
        for name in (
            "phase8jv2s0_estimated_gradient_fix.json",
            "phase8jv2s0_surrogate_alignment.json",
            "phase8jv2s0_candidate_bank_effective_capacity.json",
            "phase8jv2s0_static_capacity_reaudit.json",
            "phase8jv2s0_dynamic_capacity_reaudit.json",
        ):
            self.assertEqual(
                load(name)["status"],
                "NOT_EXECUTED_STATIC_NUMERICAL_GATE_FAIL",
            )

    def test_10_final_state_stops_before_training(self):
        final = load("phase8jv2s0_final_result.json")
        self.assertEqual(final["status"], "FAIL")
        self.assertFalse(final["audit_complete"])
        self.assertEqual(
            final["primary_cause"],
            "static_continuous_certificate_implementation",
        )
        self.assertFalse(final["network_weights_modified"])
        self.assertFalse(final["training_executed"])
        self.assertFalse(final["score_training_executed"])
        self.assertFalse(final["production_test_used"])
        self.assertFalse(final["blind_used"])
        self.assertEqual(
            final["next_allowed_phase"],
            "phase8jqv2_1_static_numerical_rebaseline",
        )


if __name__ == "__main__":
    unittest.main()
