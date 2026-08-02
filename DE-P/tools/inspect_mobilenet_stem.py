#!/usr/bin/env python3
"""Inspect the DEP MobileNetV3 stem without changing model state."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policy.dep_network import DepNetwork
from policy.models.MobileNetV3 import ConvBNActivation
from policy.models.backbone import DepBackbone


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("legacy", "corrected"), default="legacy")
    parser.add_argument("--checkpoint", type=Path)
    return parser.parse_args()


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def main() -> int:
    args = parse_args()
    try:
        backbone = DepBackbone(64, backbone_variant=args.variant)
        network = DepNetwork(backbone_variant=args.variant)
    except TypeError:
        if args.variant != "legacy":
            raise
        backbone = DepBackbone(64)
        network = DepNetwork()

    stem = backbone.backbone[0][0]
    descendants = list(stem.modules())
    nested = sum(
        isinstance(module, ConvBNActivation)
        for name, module in stem.named_modules()
        if name and isinstance(module, ConvBNActivation)
    )

    print(f"Backbone variant: {args.variant}")
    print("DepBackbone module tree:")
    print(backbone)
    print("\ncnn.features[0] repr:")
    print(repr(stem))
    print("\nfirst layer named_modules:")
    for name, module in stem.named_modules():
        print(f"  {name or '<root>'}: {module.__class__.__name__}")
    print("\nfirst layer state_dict:")
    for key, value in stem.state_dict().items():
        print(f"  {key}: shape={tuple(value.shape)}, dtype={value.dtype}")
    print(f"\nbackbone_parameters: {count_parameters(backbone)}")
    print(f"network_parameters: {count_parameters(network)}")
    print(f"nested_ConvBNActivation: {bool(nested)}")
    print(f"Conv2d_count: {sum(isinstance(m, nn.Conv2d) for m in descendants)}")
    print(f"BatchNorm2d_count: {sum(isinstance(m, nn.BatchNorm2d) for m in descendants)}")
    print(f"Hardswish_count: {sum(isinstance(m, nn.Hardswish) for m in descendants)}")
    print("execution_order:", " -> ".join(
        type(module).__name__ for module in stem.modules()
        if module is not stem and not isinstance(module, nn.Sequential)
    ))
    other_variant = "corrected" if args.variant == "legacy" else "legacy"
    try:
        other = DepNetwork(backbone_variant=other_variant)
        current_keys = set(network.state_dict())
        other_keys = set(other.state_dict())
        print(f"keys_only_in_{args.variant}:")
        for key in sorted(current_keys - other_keys):
            print(f"  {key}")
    except TypeError:
        # Keeps the diagnostic usable against a pre-dual-architecture checkout.
        if args.variant == "legacy":
            print("keys_only_in_legacy: corrected architecture not available yet")
        else:
            raise

    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        prefix = "image_backbone.backbone.0.0."
        print("\ncheckpoint first-layer keys:")
        for key, value in state_dict.items():
            if key.startswith(prefix):
                print(f"  {key}: shape={tuple(value.shape)}, dtype={value.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
