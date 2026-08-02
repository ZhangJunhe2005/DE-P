import unittest

import numpy as np

from policy.dynamic.planner_counterfactual_actionability_v2 import (
    LearnedDynamicConflictAdapterV1,
    planner_counterfactual_actionability,
)


class CounterfactualAuthorityTest(unittest.TestCase):
    def test_safe_candidate_loss_is_actionable(self):
        row = planner_counterfactual_actionability(
            [True, True, False], [False, True, False], [0., 1., 2.],
            actor_future_complete=True,
        )
        self.assertTrue(row.valid)
        self.assertEqual(row.target, 1)
        self.assertIn("safe_candidate_became_unsafe", row.reasons)
        self.assertEqual(row.static_recommendation, 0)
        self.assertEqual(row.composite_recommendation, 1)

    def test_present_but_irrelevant_actor_is_negative(self):
        row = planner_counterfactual_actionability(
            [True, True], [True, True], [1., 0.],
            actor_future_complete=True,
        )
        self.assertTrue(row.valid)
        self.assertEqual(row.target, 0)
        self.assertEqual(row.reasons, ())

    def test_missing_future_is_invalid_not_negative(self):
        row = planner_counterfactual_actionability(
            [True], [True], [0.], actor_future_complete=False,
        )
        self.assertFalse(row.valid)
        self.assertEqual(row.target, 0)
        self.assertIn("actor_future_authority_missing", row.reasons)

    def test_composite_cannot_create_safety(self):
        row = planner_counterfactual_actionability(
            [False, True], [True, True], [0., 1.],
            actor_future_complete=True,
        )
        self.assertFalse(row.valid)

    def test_adapter_is_fail_closed(self):
        adapter = LearnedDynamicConflictAdapterV1()
        active = adapter.map(.9, "DYNAMIC_SUPPORT", .9)
        unknown = adapter.map(.9, "UNKNOWN_AMBIGUOUS", .9)
        low = adapter.map(.1, "DYNAMIC_SUPPORT", .9)
        self.assertEqual(active["status"], "ACTIVE_LEARNED_DYNAMIC_RISK")
        self.assertEqual(unknown["status"], "UNRESOLVED_TRANSIENT_RISK")
        self.assertEqual(low["status"], "PENDING_OR_UNKNOWN_SUPPORT")
        self.assertFalse(active["formal_tracker_write"])
        self.assertIsNone(active["planner_command"])


if __name__ == "__main__":
    unittest.main()
