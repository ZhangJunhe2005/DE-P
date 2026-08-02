from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def load(name):
    return json.loads((REPORTS / name).read_text())


class SOCR1ContractTests(unittest.TestCase):
    def test_entry_historical_fail(self):
        self.assertEqual(
            load("phase8jqv2_4socr1_entry_gate.json")["historical_status"],
            "FAIL",
        )

    def test_six_witnesses(self):
        self.assertEqual(
            load("phase8jqv2_4socr1_entry_gate.json")["proof_witnesses"], 6
        )

    def test_one_independent_map(self):
        self.assertEqual(
            load("phase8jqv2_4socr1_entry_gate.json")["independent_maps"], 1
        )

    def test_gap1_present_gap2_absent(self):
        value = load("phase8jqv2_4socr1_entry_gate.json")
        self.assertTrue(value["gap1_present"])
        self.assertFalse(value["gap2_present"])

    def test_radius_grid(self):
        self.assertEqual(
            load("phase8jqv2_4socr1_entry_gate.json")[
                "actor_radius_variants"
            ], [.2, .25, .3]
        )

    def test_default_radius_pass(self):
        self.assertEqual(
            load("phase8jqv2_4socr1_entry_gate.json")[
                "default_radius_0_30"
            ], "PASS"
        )

    def test_independent_rerender_pass(self):
        self.assertEqual(
            load("phase8jqv2_4socr1_entry_gate.json")[
                "independent_rerender_status"
            ], "PASS"
        )

    def test_solver_mechanics_separate_and_pass(self):
        value = load("phase8jqv2_4socr1_gate_decomposition.json")
        self.assertEqual(value["solver_mechanics"]["status"], "PASS")
        self.assertFalse(
            value["solver_mechanics"]["second_map_required_for_mechanics"]
        )

    def test_map_diversity_separate_and_fail(self):
        value = load("phase8jqv2_4socr1_gate_decomposition.json")
        self.assertEqual(value["map_corpus_diversity"]["status"], "FAIL")

    def test_tracker_survival_separate(self):
        value = load("phase8jqv2_4socr1_gate_decomposition.json")
        self.assertEqual(value["tracker_survival"]["status"], "CONDITIONAL")

    def test_confidence_order(self):
        order = load(
            "phase8jqv2_4socr1_frozen_tracker_state_contract.json"
        )["lifecycle_order"]
        self.assertLess(order.index("confidence"), order.index("dynamic_state"))
        self.assertLess(
            order.index("dynamic_state"),
            order.index("attention_authorization"),
        )

    def test_confidence_not_recursive(self):
        value = load(
            "phase8jqv2_4socr1_frozen_tracker_state_contract.json"
        )
        self.assertTrue(value["confidence_is_not_recursive_previous_confidence"])

    def test_frozen_decay_threshold(self):
        value = load(
            "phase8jqv2_4socr1_frozen_tracker_state_contract.json"
        )
        self.assertEqual(value["confidence_missed_decay"], .75)
        self.assertEqual(value["track_confidence_threshold"], .55)

    def test_gap_formulae(self):
        value = load(
            "phase8jqv2_4socr1_gap_confidence_derivation.json"
        )["gaps"]
        self.assertAlmostEqual(
            value["1"]["constant_base_minimum_pre_gap_confidence"],
            .55/.75,
        )
        self.assertAlmostEqual(
            value["2"]["constant_base_minimum_pre_gap_confidence"],
            .55/.75**2,
        )
        self.assertGreater(
            value["3"]["constant_base_minimum_pre_gap_confidence"], 1.
        )

    def test_gap2_replay_boundary(self):
        rows = {
            (round(row["requested_pre_gap_confidence"], 2),
             row["gap_frames"]): row
            for row in load(
                "phase8jqv2_4socr1_tracker_contract_replay.json"
            )["rows"]
        }
        self.assertFalse(rows[(.95, 2)]["dynamic_survives_gap"])
        self.assertFalse(rows[(.98, 2)]["dynamic_survives_gap"])
        self.assertTrue(rows[(1., 2)]["dynamic_survives_gap"])

    def test_gap3_dynamic_rejected(self):
        rows = load(
            "phase8jqv2_4socr1_tracker_contract_replay.json"
        )["rows"]
        self.assertFalse(any(
            row["dynamic_survives_gap"]
            for row in rows if row["gap_frames"] == 3
        ))

    def test_track_distinct_from_dynamic_attention(self):
        rows = load(
            "phase8jqv2_4socr1_tracker_contract_replay.json"
        )["rows"]
        row = next(
            row for row in rows
            if row["requested_pre_gap_confidence"] == .70
            and row["gap_frames"] == 3
        )
        self.assertTrue(row["track_survives_gap"])
        self.assertFalse(row["dynamic_survives_gap"])
        self.assertFalse(row["attention_survives_gap"])

    def test_pre_gap_distribution_not_assumed(self):
        value = load(
            "phase8jqv2_4socr1_pre_gap_confidence_distribution.json"
        )
        self.assertEqual(value["status"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(value["exact_pre_gap_track_confidence_samples"], 0)
        self.assertFalse(value["confidence_one_assumed"])

    def test_gap2_is_conditional(self):
        value = load("phase8jqv2_4socr1_gap2_classification.json")
        self.assertEqual(
            value["classification"], "CONDITIONAL_TRACKER_SCENARIO"
        )
        self.assertFalse(value["hard_required"])

    def test_v3_candidate_not_activated(self):
        value = yaml.safe_load((
            ROOT / "configs/natural_short_occlusion_contract_v3_candidate.yaml"
        ).read_text())
        self.assertEqual(
            value["version"],
            "natural_short_occlusion_contract_v3_candidate",
        )
        self.assertFalse(value["activation"]["enabled"])
        self.assertFalse(value["activation"]["formal_contract_activated"])

    def test_loader_never_receives_gt_confidence(self):
        value = yaml.safe_load((
            ROOT / "configs/natural_short_occlusion_contract_v3_candidate.yaml"
        ).read_text())
        self.assertFalse(
            value["loader_constraints"][
                "pre_gap_ground_truth_confidence_as_model_input"
            ]
        )


class SOCR1MatrixTests(unittest.TestCase):
    pass


def add_case(name, function):
    setattr(SOCR1MatrixTests, name, function)


frozen_false = (
    "eosr1_artifacts_modified", "eosr1_witnesses_modified",
    "eosr1_solver_modified", "historical_gap1_contract_modified",
    "natural_short_full_occlusion_v2_modified",
    "TrackManager_algorithm_modified", "confidence_decay_modified",
    "dynamic_threshold_modified", "detector_modified",
    "authority_semantics_modified", "renderer_semantics_modified",
    "motion_contract_modified", "sensor_configuration_modified",
    "map_profiles_modified", "original_yopo_simulator_modified",
    "new_maps_generated", "new_trajectories_generated",
    "new_witnesses_generated", "corpus_v2_created", "split_created",
    "representation_created", "detector_executed",
    "tracker_integration_executed", "holdout_accessed",
    "formal_preflight_rerun", "formal_v3_entry_created",
    "formal_generation_started", "production_test_accessed",
    "blind_accessed", "optimizer_step_executed", "training_started",
)
for index, field in enumerate(frozen_false):
    add_case(
        f"test_forbidden_state_{index}",
        lambda self, field=field: self.assertIs(
            load("phase8jqv2_4socr1_final_result.json")[field], False
        ),
    )

required = (
    "phase8jqv2_4socr1_entry_gate.json",
    "phase8jqv2_4socr1_frozen_artifacts.json",
    "phase8jqv2_4socr1_historical_result_decomposition.json",
    "phase8jqv2_4socr1_gate_decomposition.json",
    "phase8jqv2_4socr1_solver_mechanics_status.json",
    "phase8jqv2_4socr1_map_diversity_status.json",
    "phase8jqv2_4socr1_frozen_tracker_state_contract.json",
    "phase8jqv2_4socr1_gap_confidence_derivation.json",
    "phase8jqv2_4socr1_gap_state_transition_table.json",
    "phase8jqv2_4socr1_tracker_contract_replay.json",
    "phase8jqv2_4socr1_pre_gap_confidence_distribution.json",
    "phase8jqv2_4socr1_contract_candidates.json",
    "phase8jqv2_4socr1_gap2_classification.json",
    "phase8jqv2_4socr1_occlusion_taxonomy.json",
    "phase8jqv2_4socr1_contract_decision.json",
    "phase8jqv2_4socr1_continuous_sampled_semantics.json",
    "phase8jqv2_4socr1_phase_interval_robustness.json",
    "phase8jqv2_4socr1_compatibility_matrix.json",
    "phase8jqv2_4socr1_determinism.json",
    "phase8jqv2_4socr1_regression.json",
    "phase8jqv2_4socr1_final_result.json",
)
for index, name in enumerate(required):
    add_case(
        f"test_required_report_{index}",
        lambda self, name=name: self.assertTrue((REPORTS / name).is_file()),
    )

for index, (confidence, gap) in enumerate(
    (confidence, gap)
    for confidence in (.70, .75, .80, .90, .95, .98, 1.)
    for gap in (1, 2, 3)
):
    def check_replay(self, confidence=confidence, gap=gap):
        rows = load(
            "phase8jqv2_4socr1_tracker_contract_replay.json"
        )["rows"]
        matches = [
            row for row in rows
            if row["requested_pre_gap_confidence"] == confidence
            and row["gap_frames"] == gap
        ]
        self.assertEqual(len(matches), 1)
        self.assertTrue(matches[0]["track_survives_gap"])
    add_case(f"test_replay_cell_{index}", check_replay)


if __name__ == "__main__":
    unittest.main()
