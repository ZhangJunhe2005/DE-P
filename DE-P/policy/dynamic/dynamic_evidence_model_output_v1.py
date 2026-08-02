"""Four-class learned evidence output with mandatory abstention support."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


CONTRACT_VERSION = "dynamic_evidence_model_output_v1"


class DynamicEvidenceClassV1(IntEnum):
    DYNAMIC_SUPPORT = 0
    STATIC_SUPPORT = 1
    SENSOR_OR_SEGMENTATION_ARTIFACT = 2
    UNKNOWN_AMBIGUOUS = 3


@dataclass(frozen=True, slots=True)
class DynamicEvidenceModelOutputV1:
    frame_id: int
    timestamp: float
    generation: int
    logits: np.ndarray
    calibrated_probabilities: np.ndarray | None = None
    predicted_class: DynamicEvidenceClassV1 | None = None
    abstained: bool = True
    runtime_gt_used: bool = False

    def __post_init__(self):
        if self.runtime_gt_used:
            raise ValueError("runtime GT is forbidden")
        logits = np.asarray(self.logits, np.float32)
        if logits.ndim != 2 or logits.shape[-1] != 4:
            raise ValueError("logits must be Bx4")
        if not np.isfinite(logits).all():
            raise ValueError("logits must be finite")
        logits = logits.copy(); logits.setflags(write=False)
        object.__setattr__(self, "logits", logits)
        if self.calibrated_probabilities is not None:
            probs = np.asarray(self.calibrated_probabilities, np.float32)
            if probs.shape != logits.shape or not np.isfinite(probs).all():
                raise ValueError("invalid calibrated probabilities")
            probs = probs.copy(); probs.setflags(write=False)
            object.__setattr__(self, "calibrated_probabilities", probs)


__all__ = [
    "CONTRACT_VERSION", "DynamicEvidenceClassV1",
    "DynamicEvidenceModelOutputV1",
]
