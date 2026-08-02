#!/usr/bin/env python3
"""Convert a legacy DEP checkpoint into a corrected-stem initialization.

This is not a mathematically equivalent conversion. The legacy stem contains
two nonlinear activation steps and two BatchNorm layers; the corrected stem
must be retrained or fine-tuned after conversion.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policy.checkpoint_utils import detect_checkpoint_variant, unpack_checkpoint
from policy.dep_network import DepNetwork

WARNING = (
    "This conversion only initializes the corrected model; it is not functionally "
    "or mathematically equivalent to legacy and requires retraining or fine-tuning."
)
STRATEGY = "inner_conv_and_pre_activation_bn"
LEGACY_PREFIX = "image_backbone.backbone.0.0."
TARGET_PREFIX = "image_backbone.backbone.0.0."


def convert_checkpoint(input_path: Path, output_path: Path):
    source = input_path.expanduser().resolve()
    target = output_path.expanduser().resolve()
    if source == target:
        raise ValueError("Refusing to overwrite the source checkpoint")
    if not source.is_file():
        raise FileNotFoundError(f"Source checkpoint does not exist: {source}")
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {target}")

    payload = torch.load(source, map_location="cpu", weights_only=True)
    detected = detect_checkpoint_variant(payload)
    if detected != "legacy":
        raise ValueError(f"Conversion requires a legacy checkpoint, detected: {detected}")
    source_state, _ = unpack_checkpoint(payload)
    model = DepNetwork(backbone_variant="corrected").cpu()
    target_state = model.state_dict()

    special_map = {
        f"{TARGET_PREFIX}0.weight": f"{LEGACY_PREFIX}0.0.weight",
        f"{TARGET_PREFIX}1.weight": f"{LEGACY_PREFIX}0.1.weight",
        f"{TARGET_PREFIX}1.bias": f"{LEGACY_PREFIX}0.1.bias",
        f"{TARGET_PREFIX}1.running_mean": f"{LEGACY_PREFIX}0.1.running_mean",
        f"{TARGET_PREFIX}1.running_var": f"{LEGACY_PREFIX}0.1.running_var",
        f"{TARGET_PREFIX}1.num_batches_tracked": f"{LEGACY_PREFIX}0.1.num_batches_tracked",
    }
    legacy_stem_keys = {key for key in source_state if key.startswith(LEGACY_PREFIX)}
    corrected_stem_keys = {key for key in target_state if key.startswith(TARGET_PREFIX)}
    copied, mapped, initialized = [], [], []

    converted = {}
    for target_key, initial_value in target_state.items():
        if target_key in special_map:
            source_key = special_map[target_key]
            source_value = source_state[source_key]
            if source_value.shape != initial_value.shape:
                raise ValueError(f"Stem mapping shape mismatch: {source_key} -> {target_key}")
            converted[target_key] = source_value.clone()
            mapped.append({"source": source_key, "target": target_key})
        elif target_key in source_state and source_state[target_key].shape == initial_value.shape:
            converted[target_key] = source_state[target_key].clone()
            copied.append(target_key)
        else:
            converted[target_key] = initial_value.clone()
            initialized.append(target_key)

    used_source = set(copied) | set(special_map.values())
    discarded = sorted(key for key in source_state if key not in used_source)
    # Corrected stem keys are all specially mapped; this guards accidental drift.
    if set(special_map) != corrected_stem_keys:
        raise RuntimeError(
            f"Corrected stem mapping is incomplete: expected={sorted(corrected_stem_keys)}, "
            f"mapped={sorted(special_map)}"
        )
    if not set(discarded).issuperset(legacy_stem_keys - set(special_map.values())):
        raise RuntimeError("Legacy post-activation BN keys were not fully discarded")

    incompatible = model.load_state_dict(converted, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Converted strict load failed: {incompatible}")

    metadata = {
        "backbone_variant": "corrected",
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "detected_source_variant": detected,
        "target_variant": "corrected",
        "conversion_strategy": STRATEGY,
        "copied_keys": sorted(copied),
        "specially_mapped_keys": mapped,
        "discarded_keys": discarded,
        "initialized_keys": sorted(initialized),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "warning": WARNING,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": converted, "metadata": metadata}, target)
    print(f"Source checkpoint: {source}")
    print(f"Detected source variant: {detected}")
    print("Target variant: corrected")
    print(f"Conversion strategy: {STRATEGY}")
    print(f"Copied keys ({len(copied)}): {sorted(copied)}")
    print(f"Specially mapped keys ({len(mapped)}): {mapped}")
    print(f"Discarded keys ({len(discarded)}): {discarded}")
    print(f"Initialized keys ({len(initialized)}): {sorted(initialized)}")
    print(f"WARNING: {WARNING}")
    print(f"Converted checkpoint: {target}")
    return metadata


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_checkpoint(args.input, args.output)
