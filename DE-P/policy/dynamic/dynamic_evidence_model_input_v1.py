"""Immutable causal-only input contract for learned dynamic evidence."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np


CONTRACT_VERSION = "dynamic_evidence_model_input_v1"
FORBIDDEN_RUNTIME_KEYS = frozenset({
    "gt_actor_id", "gt_actor_mask", "gt_geometry", "owner_map",
    "future_frame", "future_trajectory", "static_occupancy_authority",
})


@dataclass(frozen=True, slots=True)
class DynamicEvidenceModelInputV1:
    frame_id: int
    timestamp: float
    generation: int
    chain_id: str
    source_component_ids: tuple[int, ...]
    causal_frame_indices: tuple[int, ...]
    features: np.ndarray
    validity_mask: np.ndarray
    time_mask: np.ndarray
    no_history: bool = True
    runtime_gt_used: bool = False
    owner_map_runtime_used: bool = False
    future_frame_runtime_used: bool = False

    def __post_init__(self):
        if self.runtime_gt_used or self.owner_map_runtime_used:
            raise ValueError("runtime GT/owner map is forbidden")
        if self.future_frame_runtime_used:
            raise ValueError("future runtime input is forbidden")
        if not self.no_history:
            raise ValueError("learned evidence scope is no/weak-history only")
        if self.frame_id < 0 or self.generation < 0 or not self.chain_id:
            raise ValueError("invalid input identity")
        if not math.isfinite(self.timestamp):
            raise ValueError("timestamp must be finite")
        if not self.causal_frame_indices or max(
            self.causal_frame_indices
        ) != self.frame_id or any(
            value > self.frame_id for value in self.causal_frame_indices
        ):
            raise ValueError("causal window must end at the anchor frame")
        features = np.asarray(self.features, np.float32)
        validity = np.asarray(self.validity_mask, np.bool_)
        time_mask = np.asarray(self.time_mask, np.bool_)
        if features.ndim != 2 or features.shape != validity.shape:
            raise ValueError("features and validity mask must be KxF")
        if time_mask.shape != (features.shape[0],):
            raise ValueError("time mask must have length K")
        if not np.isfinite(features).all():
            raise ValueError("masked feature storage must remain finite")
        for name, value in (
            ("features", features), ("validity_mask", validity),
            ("time_mask", time_mask),
        ):
            value = value.copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @classmethod
    def from_mapping(cls, value: Mapping):
        forbidden = FORBIDDEN_RUNTIME_KEYS.intersection(value)
        if forbidden:
            raise ValueError(f"forbidden runtime keys: {sorted(forbidden)}")
        return cls(**value)


__all__ = [
    "CONTRACT_VERSION", "FORBIDDEN_RUNTIME_KEYS",
    "DynamicEvidenceModelInputV1",
]
