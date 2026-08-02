import unittest

import numpy as np
import torch

from policy.dynamic.temporal_evidence_input_v1 import (
    FORBIDDEN_INPUT_FIELDS, InputLeakageGuardV1, build_candidate,
    deterministic_runtime_proposals, preprocess_causal_window,
)


class DETIAR1ContractTest(unittest.TestCase):
    def test_forbidden_fields_fail_closed(self):
        for field in FORBIDDEN_INPUT_FIELDS:
            with self.assertRaises(RuntimeError):
                InputLeakageGuardV1.validate_keys({field})

    def test_preprocessing_is_causal_and_deterministic(self):
        rng = np.random.default_rng(7)
        depth = rng.uniform(.5, 10, size=(4, 96, 160)).astype(np.float32)
        valid = np.ones_like(depth, dtype=np.bool_)
        timestamp = np.asarray((0., .1, .2, .3))
        pose = np.zeros((4, 6), np.float32)
        proposal = deterministic_runtime_proposals(160, 96)[0]
        first = preprocess_causal_window(
            depth, valid, timestamp, pose, proposal, 20.,
        )
        second = preprocess_causal_window(
            depth, valid, timestamp, pose, proposal, 20.,
        )
        for key in ("roi", "time_deltas", "relative_pose", "time_mask"):
            np.testing.assert_array_equal(first[key], second[key])

    def test_three_candidates_have_frozen_shapes(self):
        roi = torch.rand(2, 4, 3, 32, 32)
        delta = torch.tensor(((-.3, -.2, -.1, 0.),) * 2)
        pose = torch.zeros(2, 4, 6)
        mask = torch.ones(2, 4, dtype=torch.bool)
        for name in ("s0", "s1", "s2"):
            model = build_candidate(name).eval()
            with torch.no_grad():
                result = model(roi, delta, pose, mask)
            self.assertEqual(tuple(result["semantic_logits"].shape), (2, 4))
            self.assertEqual(tuple(result["actionability_logit"].shape), (2,))
            self.assertEqual(tuple(result["relative_motion"].shape), (2, 3))


if __name__ == "__main__":
    unittest.main()
