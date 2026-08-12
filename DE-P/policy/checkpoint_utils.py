"""Architecture-aware DEP checkpoint loading and saving."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from policy.backbone_variant import resolve_backbone_variant

LEGACY_STEM_KEY = "image_backbone.backbone.0.0.0.0.weight"
CORRECTED_STEM_KEY = "image_backbone.backbone.0.0.0.weight"
UNIFIED_HEAD_WEIGHT_KEY = "dep_head.model.4.weight"
UNIFIED_HEAD_BIAS_KEY = "dep_head.model.4.bias"
SPLIT_TRAJECTORY_WEIGHT_KEY = "dep_head.trajectory_head.weight"
SPLIT_TRAJECTORY_BIAS_KEY = "dep_head.trajectory_head.bias"
SPLIT_SCORE_WEIGHT_KEY = "dep_head.score_head.weight"
SPLIT_SCORE_BIAS_KEY = "dep_head.score_head.bias"
SPLIT_FEATURE_WEIGHT_KEY = "dep_head.model.0.weight"
INDEPENDENT_TRAJECTORY_FEATURE_WEIGHT_KEY = (
    "dep_head.trajectory_model.0.weight"
)
INDEPENDENT_SCORE_FEATURE_WEIGHT_KEY = "dep_head.score_model.0.weight"
FORMAL_STATIC_CHECKPOINT_VERSION = "mixed_scene_static_yopo_checkpoint_v1"


def _strip_training_network_prefix(state_dict):
    """Extract ``StaticYOPOTrainingModel.network`` without weakening strict load."""
    keys = list(state_dict)
    if not keys or not all(isinstance(key, str) and key.startswith("network.") for key in keys):
        raise ValueError(
            "Formal static YOPO checkpoint model keys must all start with 'network.'"
        )
    return {key[len("network."):]: value for key, value in state_dict.items()}


def unpack_checkpoint(payload: Mapping[str, Any]):
    """Return ``(state_dict, metadata)`` for inference and formal training files."""
    if payload.get("checkpoint_version") == FORMAL_STATIC_CHECKPOINT_VERSION:
        state_dict = payload.get("model")
        if not isinstance(state_dict, Mapping):
            raise ValueError("Formal static YOPO checkpoint is missing model state")
        metadata = {
            "checkpoint_version": FORMAL_STATIC_CHECKPOINT_VERSION,
            "formal_training_checkpoint": True,
            "epoch": payload.get("epoch"),
            "global_step": payload.get("global_step"),
            "validation_metric": payload.get(
                "validation_metric", payload.get("best_validation_metric")
            ),
            "identities": payload.get("identities"),
        }
        return _strip_training_network_prefix(state_dict), metadata
    if "state_dict" in payload:
        state_dict = payload["state_dict"]
        metadata = payload.get("metadata", {})
        if not isinstance(state_dict, Mapping) or not isinstance(metadata, Mapping):
            raise ValueError("Malformed DEP checkpoint payload")
        return state_dict, dict(metadata)
    return payload, {}


def load_checkpoint_payload(checkpoint_path):
    """Load a local DEP checkpoint, including the formal trainer's resume payload.

    Formal trainer checkpoints also contain optimizer and RNG state, which are not
    accepted by PyTorch's tensor-only loader.  The result is still subjected to the
    exact version/schema checks in :func:`unpack_checkpoint` and strict model load.
    """
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"DEP checkpoint does not exist: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as tensor_only_error:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            not isinstance(payload, Mapping)
            or payload.get("checkpoint_version") != FORMAL_STATIC_CHECKPOINT_VERSION
        ):
            raise ValueError(
                "Checkpoint requires unrestricted pickle loading but is not a "
                f"recognized {FORMAL_STATIC_CHECKPOINT_VERSION} payload"
            ) from tensor_only_error
        return payload


def detect_checkpoint_variant(checkpoint: Mapping[str, Any]) -> str:
    """Identify legacy/corrected using unique stem keys and validate metadata."""
    state_dict, metadata = unpack_checkpoint(checkpoint)
    has_legacy = LEGACY_STEM_KEY in state_dict
    has_corrected = CORRECTED_STEM_KEY in state_dict
    if has_legacy == has_corrected:
        sample = list(state_dict.keys())[:12]
        raise ValueError(
            "Unable to identify DEP checkpoint architecture from stem keys; "
            f"key sample: {sample}"
        )
    detected = "legacy" if has_legacy else "corrected"
    declared = metadata.get("backbone_variant")
    if declared is not None and declared != detected:
        raise ValueError(
            f"Checkpoint metadata declares {declared!r}, but stem keys identify {detected!r}"
        )
    return detected


def detect_head_variant(checkpoint: Mapping[str, Any]) -> str:
    state_dict, metadata = unpack_checkpoint(checkpoint)
    unified = UNIFIED_HEAD_WEIGHT_KEY in state_dict and UNIFIED_HEAD_BIAS_KEY in state_dict
    split = SPLIT_FEATURE_WEIGHT_KEY in state_dict and all(
        key in state_dict for key in (
        SPLIT_TRAJECTORY_WEIGHT_KEY, SPLIT_TRAJECTORY_BIAS_KEY,
        SPLIT_SCORE_WEIGHT_KEY, SPLIT_SCORE_BIAS_KEY,
    ))
    independent = all(key in state_dict for key in (
        INDEPENDENT_TRAJECTORY_FEATURE_WEIGHT_KEY,
        INDEPENDENT_SCORE_FEATURE_WEIGHT_KEY,
        SPLIT_TRAJECTORY_WEIGHT_KEY, SPLIT_TRAJECTORY_BIAS_KEY,
        SPLIT_SCORE_WEIGHT_KEY, SPLIT_SCORE_BIAS_KEY,
    ))
    detected_variants = [
        name for name, present in (
            ("unified", unified), ("split", split),
            ("independent", independent),
        ) if present
    ]
    if len(detected_variants) != 1:
        raise ValueError("Unable to identify DEP head variant")
    detected = detected_variants[0]
    declared = metadata.get("head_variant")
    if declared is not None and declared != detected:
        raise ValueError(
            f"Checkpoint metadata declares head_variant={declared!r}, "
            f"but state keys identify {detected!r}"
        )
    return detected


def convert_unified_head_state_dict(state_dict, target_variant="split"):
    """Exactly split the legacy 10-channel output convolution.

    This is an explicit, shape-checked migration. It does not use ``strict=False``.
    """
    if UNIFIED_HEAD_WEIGHT_KEY not in state_dict or UNIFIED_HEAD_BIAS_KEY not in state_dict:
        raise ValueError("unified head tensors are missing")
    weight = state_dict[UNIFIED_HEAD_WEIGHT_KEY]
    bias = state_dict[UNIFIED_HEAD_BIAS_KEY]
    if tuple(weight.shape[:1]) != (10,) or tuple(bias.shape) != (10,):
        raise ValueError("unified head must contain exactly 10 output channels")
    if target_variant not in {"split", "independent"}:
        raise ValueError("unified head migration target must be split or independent")
    converted = dict(state_dict)
    del converted[UNIFIED_HEAD_WEIGHT_KEY]
    del converted[UNIFIED_HEAD_BIAS_KEY]
    if target_variant == "independent":
        for index in (0, 2):
            for suffix in ("weight", "bias"):
                source = f"dep_head.model.{index}.{suffix}"
                if source not in converted:
                    raise ValueError(
                        f"unified feature tower tensor is missing: {source}"
                    )
                value = converted.pop(source)
                converted[
                    f"dep_head.trajectory_model.{index}.{suffix}"
                ] = value.clone()
                converted[
                    f"dep_head.score_model.{index}.{suffix}"
                ] = value.clone()
    converted[SPLIT_TRAJECTORY_WEIGHT_KEY] = weight[:9].clone()
    converted[SPLIT_TRAJECTORY_BIAS_KEY] = bias[:9].clone()
    converted[SPLIT_SCORE_WEIGHT_KEY] = weight[9:].clone()
    converted[SPLIT_SCORE_BIAS_KEY] = bias[9:].clone()
    return converted


def load_dep_checkpoint(model, checkpoint_path, expected_variant=None,
                        allow_unified_to_split=False):
    """Load only an architecture-compatible DEP checkpoint with ``strict=True``."""
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"DEP checkpoint does not exist: {path}")
    requested = resolve_backbone_variant(
        expected_variant if expected_variant is not None else getattr(model, "backbone_variant", None)
    )
    model_variant = resolve_backbone_variant(getattr(model, "backbone_variant", requested))
    if model_variant != requested:
        raise ValueError(
            f"Model reports backbone_variant={model_variant!r}, but loader expected {requested!r}"
        )
    payload = load_checkpoint_payload(path)
    detected = detect_checkpoint_variant(payload)
    print(f"Checkpoint detected variant: {detected}")
    print(f"Requested model variant: {requested}")
    if detected != requested:
        migration = (
            " Run tools/convert_legacy_to_corrected.py first."
            if detected == "legacy" and requested == "corrected"
            else " Use a checkpoint matching the requested model architecture."
        )
        raise ValueError(
            f"Checkpoint variant mismatch: checkpoint={detected}, model={requested}.{migration}"
        )
    state_dict, metadata = unpack_checkpoint(payload)
    detected_head = detect_head_variant(payload)
    requested_head = getattr(model, "head_variant", "unified")
    migrated = False
    if detected_head != requested_head:
        if detected_head == "unified" \
                and requested_head in {"split", "independent"} \
                and allow_unified_to_split:
            state_dict = convert_unified_head_state_dict(
                state_dict, target_variant=requested_head
            )
            migrated = True
        else:
            raise ValueError(
                f"Checkpoint head variant mismatch: checkpoint={detected_head}, "
                f"model={requested_head}. Use an explicit unified-head migration."
            )
    incompatible = model.load_state_dict(state_dict, strict=True)
    print(f"missing keys: {incompatible.missing_keys}")
    print(f"unexpected keys: {incompatible.unexpected_keys}")
    print("Strict checkpoint load: PASS")
    return {
        "path": str(path),
        "variant": detected,
        "checkpoint_head_variant": detected_head,
        "model_head_variant": requested_head,
        "head_migrated": migrated,
        "metadata": metadata,
        "missing_keys": incompatible.missing_keys,
        "unexpected_keys": incompatible.unexpected_keys,
        "strict": True,
    }


def validate_dep_checkpoint_variant(checkpoint_path, expected_variant) -> str:
    """Preflight a checkpoint architecture before initializing external runtimes."""
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"DEP checkpoint does not exist: {path}")
    requested = resolve_backbone_variant(expected_variant)
    payload = load_checkpoint_payload(path)
    detected = detect_checkpoint_variant(payload)
    print(f"Checkpoint detected variant: {detected}")
    print(f"Requested model variant: {requested}")
    if detected != requested:
        migration = (
            " Run tools/convert_legacy_to_corrected.py first."
            if detected == "legacy" and requested == "corrected"
            else " Use a checkpoint matching the requested model architecture."
        )
        raise ValueError(
            f"Checkpoint variant mismatch: checkpoint={detected}, model={requested}.{migration}"
        )
    return detected


def checkpoint_payload(model, backbone_variant, **metadata):
    """Create the wrapped format used for newly saved corrected checkpoints."""
    variant = resolve_backbone_variant(backbone_variant)
    return {
        "state_dict": model.state_dict(),
        "metadata": {"backbone_variant": variant, **metadata},
    }
