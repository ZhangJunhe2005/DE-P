"""Explicit DPAR2 architecture registry; production default remains legacy."""

from __future__ import annotations

from pathlib import Path

import yaml

from .causal_depth_track_before_detect_v1 import (
    CausalDepthTrackBeforeDetectV1,
)
from .foreground_contract_registry import create_foreground_contract
from .physical_control_residual_v1 import PhysicalControlResidualV1
from .dual_path_dynamic_perception_v1 import DualPathDynamicPerceptionV1
from .visibility_aware_causal_residual_v1 import (
    VisibilityAwareCausalResidualV1,
)


REGISTRY_VERSION = "dynamic_perception_architecture_registry_v1"
LEGACY_DEFAULT = "legacy_v1"
VALID_KEYS = (
    "legacy_v1", "tf1_candidate_a", "tf1_candidate_b",
    "tf1_candidate_c", "physical_control_residual_v1",
    "visibility_aware_causal_residual_v1",
    "causal_depth_track_before_detect_v1",
    "dual_path_dynamic_perception_v1",
)


def load_architecture_config(path=None):
    path = (
        Path(path) if path is not None else
        Path(__file__).resolve().parents[2]
        / "configs/dynamic_perception_architecture_candidates_v1.yaml"
    )
    value = yaml.safe_load(path.read_text())
    if value["default_architecture"] != LEGACY_DEFAULT:
        raise RuntimeError("legacy default changed")
    return value


def create_architecture(key, config, parameters=None, registry_config=None):
    if key not in VALID_KEYS:
        raise KeyError(f"unknown explicit architecture key: {key}")
    document = registry_config or load_architecture_config()
    parameters = dict(
        parameters if parameters is not None
        else document["candidates"].get(key, {})
    )
    if key == "legacy_v1":
        return create_foreground_contract("legacy_v1", config, {})
    if key.startswith("tf1_candidate_"):
        tf1_key = key.removeprefix("tf1_")
        return create_foreground_contract(tf1_key, config, parameters)
    if key == "physical_control_residual_v1":
        return PhysicalControlResidualV1(config, parameters)
    if key == "visibility_aware_causal_residual_v1":
        return VisibilityAwareCausalResidualV1(config, parameters)
    if key == "causal_depth_track_before_detect_v1":
        return CausalDepthTrackBeforeDetectV1(config, parameters)
    if key == "dual_path_dynamic_perception_v1":
        return DualPathDynamicPerceptionV1(config, parameters)
    raise AssertionError(key)


__all__ = [
    "REGISTRY_VERSION", "LEGACY_DEFAULT", "VALID_KEYS",
    "load_architecture_config", "create_architecture",
]
