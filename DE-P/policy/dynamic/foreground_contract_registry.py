"""Explicit foreground contract registry; legacy remains the only default."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import yaml

from .range_image_foreground import CausalRangeImageForeground
from .range_image_foreground_v2_1 import (
    CandidateRangeImageForeground, RestrictedTrackReacquisition,
)


REGISTRY_VERSION = "foreground_contract_registry_v1"
LEGACY_DEFAULT = "legacy_v1"
VALID_KEYS = (
    "legacy_v1", "candidate_a", "candidate_b", "candidate_c",
    "candidate_d", "ablation_seed7", "selected_v2_1",
)


def load_candidate_config(path=None):
    path = (
        Path(path) if path is not None else
        Path(__file__).resolve().parents[2]
        / "configs/temporal_foreground_contract_v2_1_candidates.yaml"
    )
    value = yaml.safe_load(path.read_text())
    if value["legacy_default"] != LEGACY_DEFAULT:
        raise RuntimeError("legacy default changed")
    return value


def create_foreground_contract(key, config, parameters=None, registry_config=None):
    """No default key is accepted: every nonlegacy experiment is explicit."""
    if key not in VALID_KEYS:
        raise KeyError(f"unknown explicit foreground contract: {key}")
    document = registry_config or load_candidate_config()
    if key == "legacy_v1":
        return CausalRangeImageForeground(config)
    if key == "selected_v2_1":
        if not document["candidate_selected"]:
            raise RuntimeError("selected_v2_1 is not enabled")
        key = document["selected_candidate"]
    if key == "candidate_d":
        return RestrictedTrackReacquisition(**(parameters or {}))
    if key == "ablation_seed7":
        if document["candidates"]["ablation_seed7"]["selectable"]:
            raise RuntimeError("seed7 ablation must remain diagnostic-only")
        return CausalRangeImageForeground(replace(
            config, range_min_seed_pixels=7
        ))
    return CandidateRangeImageForeground(config, key, parameters or {})
