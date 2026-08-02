from types import SimpleNamespace
import unittest

import torch

from tools.phase8c_validation_metrics import ValidationMetrics


def details(raw=None, score=None, clearance=None, counts=None, static=2.0):
    zero = torch.tensor(0.0)
    result = {
        "trajectory_loss": torch.tensor(2.0),
        "score_loss": torch.tensor(1.0),
        "guidance_loss": torch.tensor(0.5),
        "smooth_loss": torch.tensor(0.1),
        "static_safety_loss": torch.tensor(static),
    }
    if raw is not None:
        result.update({
            "candidate_dynamic_cost_raw": torch.tensor(raw, dtype=torch.float32),
            "predicted_score": torch.tensor(score, dtype=torch.float32),
            "dynamic_diagnostics": SimpleNamespace(
                candidate_min_clearance=torch.tensor(clearance, dtype=torch.float32),
                dynamic_obstacle_count=torch.tensor(counts),
            ),
        })
    return result


class ValidationMetricTests(unittest.TestCase):
    def test_no_target_is_not_allowed_to_dilute_dynamic_metrics(self):
        accumulator = ValidationMetrics(4 / 15)
        accumulator.add("static", details(static=2.5))
        raw = [[0.0] * 15, [1.0] * 15]
        score = [[0.0] * 15, list(range(15))]
        clearance = [[100.0] * 15, [-0.1] + [0.2] * 14]
        accumulator.add("dynamic", details(raw, score, clearance, [0, 1]))
        result = accumulator.finalize()
        self.assertEqual(result["raw_dynamic_mean"], 1.0)
        self.assertEqual(result["no_target_dynamic_mean"], 0.0)
        self.assertEqual(result["top1_collision_fraction"], 1.0)
        self.assertEqual(result["static_cost"], 2.5)


if __name__ == "__main__":
    unittest.main()
