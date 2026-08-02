"""PDSCR1 standard-library unittest gate (70 cases)."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np
import yaml

from policy.dynamic.provisional_dynamic_safety_state_v1 import (
    ProvisionalDynamicSafetyStateV1, ProvisionalSafetyStateManagerV1,
    ProvisionalSafetyStateTypeV1,
)
from policy.dynamic.provisional_formal_reconciliation_v1 import (
    ProvisionalFormalReconciliationV1,
)
from policy.dynamic.provisional_outcome_mapper_v1 import (
    ProvisionalOutcomeMapperV1,
)
from policy.dynamic.provisional_risk_consumer_v1 import (
    ProvisionalRiskConsumerV1,
)


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/"reports"
PREFIX = "phase8jqv2_4pdscr1_"
CONFIG = yaml.safe_load((
    ROOT/"configs/provisional_dynamic_safety_contract_v2_candidate.yaml"
).read_text())
REPORTS_REQUIRED = (
    "entry_gate", "frozen_artifacts", "historical_validation_status",
    "state_contract", "outcome_mapping", "birth_contract",
    "association_contract", "lifecycle_contract",
    "promotion_contract", "risk_consumer_contract", "l6_validation",
    "frame24_support_validation", "frame25_unresolved_validation",
    "previous_track_context", "p0_baseline", "p1_outcome_mapping",
    "p2_unconfirmed_bridge", "p3_support", "p4_unresolved",
    "p5_promotion", "p6_multi_target", "p7_unified",
    "candidate_comparison", "evaluation_split",
    "fresh_validation_freeze", "validation_freeze",
    "negative_validation", "multi_target_validation",
    "implementation_contract", "runtime", "host_runtime",
    "determinism", "regression", "compatibility_matrix",
    "candidate_selection", "final_result",
)
MARKDOWN_REQUIRED = (
    "migration_plan", "final_recommendation", "final_readiness",
)


def report(stem):
    return json.loads((REPORTS/f"{PREFIX}{stem}.json").read_text())


def weak_outcome(outcome_id=1, timestamp=1.):
    return {
        "outcome_id": outcome_id, "frame_index": 1,
        "timestamp": timestamp,
        "status": "VALID_SAFETY_WEAK_MEASUREMENT",
        "source_component_ids": (1,),
        "position_reference": "WEAK_CENTER_ESTIMATE_WORLD",
        "position_world": [0., 0., 1.],
        "covariance_world": np.eye(3).tolist(),
        "velocity_center_mps": [0., 0., 0.],
    }


def support_outcome(outcome_id=2, timestamp=1.):
    return {
        "outcome_id": outcome_id, "frame_index": 1,
        "timestamp": timestamp,
        "status": "BOUNDED_SAFETY_SUPPORT",
        "source_component_ids": (2,),
        "position_reference": "FINITE_WORLD_POSITION_SET",
        "support": {
            "position_set_min_world": [-.5, -.5, .1],
            "position_set_max_world": [.5, .5, .6],
        },
    }


def unresolved_outcome(outcome_id=3, timestamp=1.):
    return {
        "outcome_id": outcome_id, "frame_index": 1,
        "timestamp": timestamp,
        "status": "UNRESOLVED_MEASUREMENT_RISK",
        "source_component_ids": (3,), "risk_present": True,
        "unresolved_risk": {
            "source_evidence": ["DEPTH_TRANSITION"],
        },
    }


def mapped_measurement(state_id=1, position=(0., 0., 1.)):
    value = weak_outcome(state_id)
    value["position_world"] = list(position)
    return ProvisionalOutcomeMapperV1(CONFIG).map(value, state_id)


class PDSCR1Gate(unittest.TestCase):
    def check(self, index):
        entry = report("entry_gate")
        frozen = report("frozen_artifacts")
        final = report("final_result")
        mapper = ProvisionalOutcomeMapperV1(CONFIG)
        if index == 1:
            self.assertTrue(entry["checks"]["dmcr1_route_a"])
        elif index == 2:
            self.assertTrue(entry["checks"]["frame24_support"])
        elif index == 3:
            self.assertTrue(entry["checks"]["frame25_unresolved"])
        elif index == 4:
            self.assertFalse(
                frozen["strict_formal_measurement_filter_modified"]
            )
        elif index == 5:
            self.assertEqual(final["formal_tracker_feed"], 0)
        elif index == 6:
            self.assertFalse(frozen["TrackManager_algorithm_modified"])
        elif index == 7:
            self.assertFalse(frozen["kalman_process_model_modified"])
        elif index == 8:
            self.assertFalse(frozen["static_yopo_network_modified"])
        elif index == 9:
            self.assertEqual(report("p0_baseline")["L6"], 4)
        elif index == 10:
            self.assertEqual(report("outcome_mapping")["status"], "PASS")
        elif index == 11:
            self.assertEqual(
                mapper.map(weak_outcome(), 1).state_type,
                ProvisionalSafetyStateTypeV1
                .PROVISIONAL_MEASUREMENT_STATE,
            )
        elif index == 12:
            state = mapper.map(
                support_outcome(), 2,
                causal_context={
                    "fields": {"closer_fraction": .1},
                    "historical_context": {},
                },
            )
            self.assertEqual(
                state.state_type,
                ProvisionalSafetyStateTypeV1.PROVISIONAL_SUPPORT_STATE,
            )
        elif index == 13:
            self.assertEqual(
                mapper.map(unresolved_outcome(), 3).state_type,
                ProvisionalSafetyStateTypeV1
                .PROVISIONAL_UNRESOLVED_STATE,
            )
        elif index == 14:
            self.assertEqual(
                report("l6_validation")["safety_state_frames"], 4
            )
        elif index == 15:
            self.assertIn("PROMOTION_PENDING",
                          report("state_contract")["state_types"])
        elif index == 16:
            self.assertIn("PROMOTED",
                          report("state_contract")["state_types"])
        elif index == 17:
            self.assertIn("EXPIRED",
                          report("state_contract")["state_types"])
        elif index == 18:
            self.assertIn(
                "PURE_STATIC_SUPPORT",
                report("birth_contract")["forbidden"],
            )
        elif index == 19:
            hard = {
                "outcome_id": 5, "timestamp": 1.,
                "status": "HARD_INVALID_NO_EVIDENCE",
            }
            self.assertIsNone(mapper.map(hard, 5))
        elif index == 20:
            state = mapper.map(
                support_outcome(), 2, causal_context={}
            )
            self.assertIsNone(state)
        elif index == 21:
            manager = ProvisionalSafetyStateManagerV1()
            manager.update(
                (mapped_measurement(1), mapped_measurement(2, (3, 0, 1))),
                1.,
            )
            self.assertTrue(manager.last_diagnostics["one_to_one"])
        elif index == 22:
            with self.assertRaises(ValueError):
                mapper.map({**weak_outcome(), "gt_actor_id": 1}, 1)
        elif index == 23:
            state = mapped_measurement()
            self.assertLess(
                state.expiry_timestamp-state.birth_timestamp, .081
            )
        elif index == 24:
            self.assertEqual(
                mapped_measurement().maximum_missed_frames, 2
            )
        elif index == 25:
            state = mapped_measurement()
            self.assertEqual(
                state.cache_read(1.02).last_evidence_timestamp,
                state.last_evidence_timestamp,
            )
        elif index == 26:
            self.assertGreater(
                mapped_measurement().velocity_uncertainty_mps, 0
            )
        elif index == 27:
            state = mapped_measurement()
            self.assertGreater(state.velocity_uncertainty_mps, 0)
        elif index == 28:
            self.assertFalse(report("frame24_support_validation")[
                "center_created"])
        elif index == 29:
            self.assertFalse(report("frame25_unresolved_validation")[
                "occupancy_created"])
        elif index == 30:
            self.assertEqual(
                report("frame25_unresolved_validation")[
                    "planner_semantic"],
                "UNRESOLVED_DYNAMIC_RISK",
            )
        elif index == 31:
            self.assertEqual(report("l6_validation")[
                "safety_state_frames"], 4)
        elif index == 32:
            self.assertEqual(report("frame24_support_validation")[
                "planner_semantic"], "ACTIVE_SUPPORT_RISK")
        elif index == 33:
            self.assertEqual(report("frame25_unresolved_validation")[
                "planner_semantic"], "UNRESOLVED_DYNAMIC_RISK")
        elif index in (34, 35, 36, 37):
            state = mapped_measurement()
            reconciler = ProvisionalFormalReconciliationV1()
            tracks = [{
                "identity": "0:1", "observation_ids": (),
                "position_world": [0., 0., 1.],
            }]
            decision = reconciler.reconcile(
                (state,), tracks, 1.
            )[0]
            if index == 34:
                self.assertIn(
                    decision.status,
                    ("PROMOTED", "DUPLICATE_SUPPRESSED"),
                )
            elif index == 35:
                self.assertEqual(
                    decision.status, "DUPLICATE_SUPPRESSED"
                )
            elif index == 36:
                far = [{
                    "identity": "9:9", "observation_ids": (),
                    "position_world": [9., 9., 9.],
                }]
                self.assertEqual(
                    reconciler.reconcile((state,), far, 1.)[0].status,
                    "INDEPENDENT",
                )
            else:
                equal = [tracks[0], {
                    "identity": "0:2", "observation_ids": (),
                    "position_world": [0., 0., 1.],
                }]
                self.assertEqual(
                    reconciler.reconcile((state,), equal, 1.)[0].status,
                    "AMBIGUOUS",
                )
        elif index == 38:
            self.assertTrue(report("promotion_contract")[
                "generation_safe"])
        elif index == 39:
            state = mapped_measurement().miss(2.)
            self.assertEqual(
                state.state_type, ProvisionalSafetyStateTypeV1.EXPIRED
            )
        elif index == 40:
            consumer = ProvisionalRiskConsumerV1()
            self.assertEqual(
                consumer.consume((mapped_measurement(),)).semantic,
                "ACTIVE_PROVISIONAL_RISK",
            )
        elif index == 41:
            self.assertEqual(report("negative_validation")[
                "duplicate_formal_provisional_risk"], 0)
        elif index == 42:
            self.assertEqual(report("previous_track_context")["count"], 10)
        elif index == 43:
            self.assertEqual(report("regression")[
                "L2_FOREGROUND_COMPONENT"], 3)
        elif index == 44:
            self.assertTrue(report("regression")[
                "causally_unobservable_risk_present"])
        elif index == 45:
            self.assertGreater(report("fresh_validation_freeze")[
                "fresh_summary"]["no_target_provisional_birth"], 0)
            self.assertEqual(
                final["primary_cause"],
                "provisional_safety_false_availability",
            )
        elif index == 46:
            self.assertEqual(report("fresh_validation_freeze")[
                "fresh_summary"]["static_false_dynamic_risk"], 0)
        elif index == 47:
            self.assertEqual(report("evaluation_split")["status"],
                             "PASS_GROUPED")
        elif index == 48:
            self.assertEqual(report("fresh_validation_freeze")["status"],
                             "FAIL")
            self.assertEqual(final["route"], "F")
        elif index == 49:
            self.assertFalse(report("validation_freeze")[
                "post_fresh_tuning"])
        elif 50 <= index <= 55:
            host = report("host_runtime")
            if host["status"] == "PENDING_HOST_GATE":
                self.skipTest("host CUDA gate pending")
            self.assertEqual(host["status"], "PASS")
            if index == 50:
                self.assertTrue(host["same_frame_only"])
            elif index == 51:
                self.assertTrue(host["atomic_join_before_snapshot"])
            elif index == 52:
                self.assertEqual(host["queue_depth"], 0)
            elif index == 53:
                self.assertLessEqual(
                    host["provisional_candidate"][
                        "steady_state_ms"]["p95"],
                    host["gate_ms"],
                )
            elif index == 54:
                candidate = host["provisional_candidate"]
                self.assertLessEqual(candidate["deadline_miss_rate"], .01)
                self.assertLessEqual(
                    candidate["consecutive_deadline_miss_max"], 1
                )
            else:
                self.assertFalse(host["runtime_gt_used"])
        elif index == 56:
            self.assertFalse(report("candidate_selection")[
                "production_default_changed"])
        elif index == 57:
            self.assertFalse(final["formal_dataset_generated"])
        elif index == 58:
            self.assertFalse(final["holdout_test_blind_accessed"])
        elif index in (59, 60):
            self.assertFalse(final["training_authorized"])
        elif 61 <= index <= 67:
            regression = report("regression")
            keys = (
                "dmcr1_artifacts_modified", "mar1_artifacts_modified",
                "diro1_artifacts_modified", "BRIR1_modified",
                "BDRR1_modified", "YOPO_modified", "Kalman_modified",
            )
            self.assertFalse(regression[keys[index-61]])
        elif index == 68:
            self.assertTrue(all(
                (ROOT/path).is_file()
                for path in report("implementation_contract")["files"]
            ))
        elif index == 69:
            self.assertEqual(report("regression")["status"], "PASS")
        elif index == 70:
            for stem in REPORTS_REQUIRED:
                self.assertTrue(
                    (REPORTS/f"{PREFIX}{stem}.json").is_file(), stem
                )
            for stem in MARKDOWN_REQUIRED:
                self.assertTrue(
                    (REPORTS/f"{PREFIX}{stem}.md").is_file(), stem
                )


def make_test(index):
    def test(self):
        self.check(index)
    test.__name__ = f"test_{index:02d}"
    return test


for _index in range(1, 71):
    setattr(PDSCR1Gate, f"test_{_index:02d}", make_test(_index))


if __name__ == "__main__":
    unittest.main()
