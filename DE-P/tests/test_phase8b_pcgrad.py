import unittest

import torch

from policy.dep_trainer import DepTrainer


class DynamicPriorityPCGradTests(unittest.TestCase):
    def test_conflicting_non_dynamic_component_is_projected(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
        dynamic = parameter[0]
        other = -parameter[0] + parameter[1]
        conflict = DepTrainer._dynamic_priority_pcgrad(dynamic, other, [parameter])
        self.assertEqual(float(conflict), 1.0)
        self.assertTrue(torch.allclose(
            parameter.grad, torch.tensor([1.0, 2 ** -0.5]), atol=1e-6
        ))
        self.assertGreaterEqual(float(torch.dot(parameter.grad, torch.tensor([1.0, 0.0]))), 0)

    def test_aligned_gradients_are_ordinary_sum(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
        dynamic = parameter[0]
        other = 2 * parameter[0] + parameter[1]
        conflict = DepTrainer._dynamic_priority_pcgrad(dynamic, other, [parameter])
        self.assertEqual(float(conflict), 0.0)
        expected = torch.tensor([1.0 + 2 / (5 ** 0.5), 1 / (5 ** 0.5)])
        self.assertTrue(torch.allclose(parameter.grad, expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
