import hashlib
import unittest
from pathlib import Path

import torch
from torch import nn

from policy.dep_network import DepNetwork
from policy.models.MobileNetV3 import ConvBNActivation
from policy.models.backbone import DepBackbone


ROOT = Path(__file__).resolve().parents[1]


class LegacyBackboneRegressionTests(unittest.TestCase):
    def test_default_is_legacy_and_checkpoint_is_unchanged(self):
        model = DepNetwork().cpu().eval()
        self.assertEqual(model.backbone_variant, "legacy")
        fixture = torch.load(
            ROOT / "tests/fixtures/legacy_forward_reference.pt",
            map_location="cpu",
            weights_only=True,
        )
        checkpoint = ROOT / "saved/DEP_0/epoch10.pth"
        self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), fixture["checkpoint_sha256"])
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
        incompatible = model.load_state_dict(state_dict, strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertEqual(sum(p.numel() for p in model.parameters()), fixture["parameter_count"])
        self.assertEqual(list(model.state_dict()), fixture["state_dict_keys"])
        with torch.inference_mode():
            endstate, score = model.inference(fixture["depth"], fixture["obs"].clone())
        self.assertLessEqual(torch.max(torch.abs(endstate - fixture["endstate"])).item(), 1e-7)
        self.assertLessEqual(torch.max(torch.abs(score - fixture["score"])).item(), 1e-7)

    def test_legacy_shapes_are_unchanged(self):
        model = DepNetwork(backbone_variant="legacy").cpu().eval()
        depth = torch.zeros(1, 1, 96, 160)
        obs = torch.zeros(1, 9, 3, 5)
        with torch.inference_mode():
            feature = model.image_backbone(depth)
            endstate, score = model(depth, obs)
        self.assertEqual(tuple(feature.shape), (1, 64, 3, 5))
        self.assertEqual(tuple(endstate.shape), (1, 9, 3, 5))
        self.assertEqual(tuple(score.shape), (1, 3, 5))

    def test_host_validator_cpu_golden_path(self):
        from tests.run_host_backbone_variant_validation import cpu_golden_result

        result = cpu_golden_result(ROOT / "saved/DEP_0/epoch10.pth")
        self.assertEqual(result["status"], "PASS")
        self.assertLessEqual(result["endstate_max_abs_diff"], 1e-7)
        self.assertLessEqual(result["score_max_abs_diff"], 1e-7)


class CorrectedBackboneTests(unittest.TestCase):
    def test_corrected_stem_is_standard(self):
        backbone = DepBackbone(64, backbone_variant="corrected").cpu()
        stem = backbone.backbone[0][0]
        descendants = list(stem.modules())
        self.assertIsInstance(stem, ConvBNActivation)
        self.assertEqual(sum(isinstance(m, nn.Conv2d) for m in descendants), 1)
        self.assertEqual(sum(isinstance(m, nn.BatchNorm2d) for m in descendants), 1)
        self.assertEqual(sum(isinstance(m, nn.Hardswish) for m in descendants), 1)
        self.assertFalse(any(
            name and isinstance(module, ConvBNActivation)
            for name, module in stem.named_modules()
        ))
        conv, bn, activation = stem
        self.assertEqual(conv.in_channels, 1)
        self.assertEqual(conv.out_channels, 16)
        self.assertEqual(conv.kernel_size, (3, 3))
        self.assertEqual(conv.stride, (2, 2))
        self.assertEqual(conv.padding, (1, 1))
        self.assertEqual(bn.eps, 0.001)
        self.assertEqual(bn.momentum, 0.01)
        self.assertIsInstance(activation, nn.Hardswish)

    def test_corrected_forward_backward_and_optimizer_step_are_finite(self):
        torch.manual_seed(7)
        model = DepNetwork(backbone_variant="corrected").cpu().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        depth = torch.randn(2, 1, 96, 160)
        obs = torch.randn(2, 9, 3, 5)
        optimizer.zero_grad(set_to_none=True)
        endstate, score = model(depth, obs)
        loss = endstate.square().mean() + score.mean()
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        optimizer.step()
        print("CORRECTED_CPU_GRADIENT_TENSOR_COUNT", len(gradients))
        self.assertEqual(tuple(endstate.shape), (2, 9, 3, 5))
        self.assertEqual(tuple(score.shape), (2, 3, 5))
        self.assertTrue(torch.isfinite(endstate).all())
        self.assertTrue(torch.isfinite(score).all())
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_invalid_variant_fails_immediately(self):
        with self.assertRaisesRegex(ValueError, "Unsupported backbone_variant"):
            DepNetwork(backbone_variant="unknown")


if __name__ == "__main__":
    unittest.main()
