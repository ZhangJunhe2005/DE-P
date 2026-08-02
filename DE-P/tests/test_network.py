import unittest
from pathlib import Path

import torch

from tests.baseline_helpers import finite_tensor, make_cpu_network, reset_lattice_singleton


ROOT = Path(__file__).resolve().parents[1]


class NetworkForwardTests(unittest.TestCase):
    def test_backbone_shape_and_finite_on_cpu(self):
        from policy.models.backbone import DepBackbone

        model = DepBackbone(64).cpu().eval()
        depth = torch.zeros(1, 1, 96, 160)
        with torch.inference_mode():
            feature = model(depth)
        self.assertEqual(tuple(feature.shape), (1, 64, 3, 5))
        self.assertTrue(finite_tensor(feature))

    def test_full_inference_shape_and_finite_on_cpu(self):
        model = make_cpu_network().eval()
        depth = torch.zeros(1, 1, 96, 160)
        observation = torch.zeros(1, 9)
        with torch.inference_mode():
            endstate, score = model.inference(depth, observation)
        self.assertEqual(tuple(endstate.shape), (1, 9, 3, 5))
        self.assertEqual(tuple(score.shape), (1, 3, 5))
        self.assertTrue(finite_tensor(endstate))
        self.assertTrue(finite_tensor(score))

    def test_checkpoint_strictly_matches(self):
        model = make_cpu_network()
        checkpoint = ROOT / "saved" / "DEP_0" / "epoch10.pth"
        self.assertTrue(checkpoint.is_file())
        state_dict = torch.load(checkpoint, weights_only=True, map_location="cpu")
        incompatible = model.load_state_dict(state_dict, strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])

    @unittest.skipUnless(torch.cuda.is_available(), "NOT MEASURED IN CODEX SANDBOX: CUDA device is not visible")
    def test_full_inference_shape_and_finite_on_cuda(self):
        from policy.dep_network import DepNetwork

        reset_lattice_singleton()
        model = DepNetwork().cuda().eval()
        depth = torch.zeros(1, 1, 96, 160, device="cuda")
        observation = torch.zeros(1, 9, device="cuda")
        with torch.inference_mode():
            endstate, score = model.inference(depth, observation)
        self.assertEqual(tuple(endstate.shape), (1, 9, 3, 5))
        self.assertEqual(tuple(score.shape), (1, 3, 5))
        self.assertTrue(finite_tensor(endstate))
        self.assertTrue(finite_tensor(score))


if __name__ == "__main__":
    unittest.main()
