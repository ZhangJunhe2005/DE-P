from dataclasses import replace
import unittest

import torch

from policy.dep_network import DepNetwork
from policy.dynamic.context import ATTENTION_BACKBONE_OUTPUT, DynamicContext
from policy.dynamic.types import DynamicPerceptionConfig


def dynamic_config(**overrides):
    values = {"enabled": True, "use_attention": True, "fallback_to_static": True}
    values.update(overrides)
    config = replace(DynamicPerceptionConfig.from_global_config(), **values)
    config.validate()
    return config


def context(attention, valid=True, tracks=()):
    return DynamicContext(
        attention_maps_by_level={ATTENTION_BACKBONE_OUTPUT: attention} if attention is not None else {},
        dynamic_tracks=tracks,
        timestamp=1.0,
        source="test",
        valid=valid,
        diagnostics={},
    )


class DynamicNetworkIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.depth = torch.rand(1, 1, 96, 160)
        self.obs = torch.rand(1, 9, 3, 5)

    def test_disabled_is_bit_exact_and_has_no_new_state_keys(self):
        for variant in ("legacy", "corrected"):
            with self.subTest(variant=variant):
                model = DepNetwork(backbone_variant=variant).cpu().eval()
                keys = tuple(model.state_dict())
                with torch.inference_mode():
                    static = model(self.depth, self.obs)
                    supplied = model(
                        self.depth, self.obs,
                        context(torch.ones(1, 1, 3, 5)),
                    )
                self.assertTrue(torch.equal(static[0], supplied[0]))
                self.assertTrue(torch.equal(static[1], supplied[1]))
                self.assertFalse(any("dynamic" in key or "attention" in key for key in keys))

    def test_empty_invalid_and_zero_attention_fall_back_or_match(self):
        model = DepNetwork(dynamic_config=dynamic_config()).cpu().eval()
        with torch.inference_mode():
            static = model._forward_impl(self.depth, self.obs)
            missing = model(self.depth, self.obs, None)
            invalid = model(self.depth, self.obs, context(None, valid=False))
            zero = model(self.depth, self.obs, context(torch.zeros(1, 1, 3, 5)))
        for actual in (missing, invalid, zero):
            self.assertTrue(torch.equal(static[0], actual[0]))
            self.assertTrue(torch.equal(static[1], actual[1]))

    def test_single_multiple_and_boundary_attention_are_finite_and_explainable(self):
        model = DepNetwork(dynamic_config=dynamic_config()).cpu().eval()
        maps = (
            torch.nn.functional.one_hot(torch.tensor(7), 15).float().reshape(1, 1, 3, 5),
            torch.tensor([[[[1, 0, 0, 0, 1], [0, 0, 0, 0, 0], [1, 0, 0, 0, 1]]]], dtype=torch.float32),
            torch.ones(1, 1, 96, 160),
        )
        with torch.inference_mode():
            base_feature = model.image_backbone(self.depth)
            static = model._forward_impl(self.depth, self.obs)
            for attention in maps:
                with self.subTest(shape=tuple(attention.shape)):
                    dynamic_feature = model.image_backbone(self.depth, attention, 1.0)
                    dynamic = model(self.depth, self.obs, context(attention))
                    self.assertEqual(tuple(dynamic[0].shape), (1, 9, 3, 5))
                    self.assertEqual(tuple(dynamic[1].shape), (1, 3, 5))
                    self.assertTrue(all(torch.isfinite(item).all() for item in dynamic))
                    self.assertTrue(torch.all(dynamic_feature.abs() >= base_feature.abs()))
                    self.assertFalse(torch.equal(static[0], dynamic[0]))

    def test_forward_backward_and_batch_semantics(self):
        model = DepNetwork(
            backbone_variant="corrected", dynamic_config=dynamic_config()
        ).cpu().train()
        depth = self.depth.repeat(2, 1, 1, 1).requires_grad_(True)
        obs = self.obs.repeat(2, 1, 1, 1)
        attention = torch.stack((torch.zeros(1, 3, 5), torch.ones(1, 3, 5)))
        outputs = model(depth, obs, context(attention))
        loss = outputs[0].square().mean() + outputs[1].mean()
        loss.backward()
        self.assertTrue(torch.isfinite(depth.grad).all())
        self.assertGreater(sum(p.grad is not None for p in model.parameters()), 0)
        with self.assertRaisesRegex(ValueError, "independent context"):
            model(depth, obs, context(torch.zeros(1, 1, 3, 5)))
        with self.assertRaisesRegex(ValueError, "batch size"):
            model(depth, obs, context(torch.zeros(3, 1, 3, 5)))

    def test_fallback_disabled_raises_and_tracker_is_not_in_network(self):
        model = DepNetwork(
            dynamic_config=dynamic_config(fallback_to_static=False)
        ).cpu().eval()
        with self.assertRaisesRegex(RuntimeError, "fallback is disabled"):
            model(self.depth, self.obs, None)
        names = {name.lower() for name, _ in model.named_modules()}
        self.assertFalse(any("track" in name or "cluster" in name for name in names))
        self.assertFalse(hasattr(model, "tracker"))

    def test_invalid_attention_is_rejected_at_context_boundary(self):
        for bad in (
            torch.full((1, 1, 3, 5), float("nan")),
            torch.full((1, 1, 3, 5), -0.1),
            torch.full((1, 1, 3, 5), 1.1),
        ):
            with self.assertRaises((ValueError, FloatingPointError)):
                context(bad)


if __name__ == "__main__":
    unittest.main()
