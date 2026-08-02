"""Development-only calibrated semantic adapter; no planner commands."""

from __future__ import annotations

import numpy as np

from .dynamic_evidence_model_output_v1 import DynamicEvidenceClassV1


CONTRACT_VERSION = "learned_dynamic_evidence_adapter_v1"


class LearnedDynamicEvidenceAdapterV1:
    def __init__(self, confidence_threshold):
        self.confidence_threshold = float(confidence_threshold)

    def map(self, output):
        probabilities = output.calibrated_probabilities
        if probabilities is None:
            return "INVALID_EVALUATION"
        row = np.asarray(probabilities)[0]
        index = int(np.argmax(row))
        if float(row[index]) < self.confidence_threshold:
            return "PENDING_OR_UNKNOWN_SUPPORT"
        label = DynamicEvidenceClassV1(index)
        return {
            DynamicEvidenceClassV1.DYNAMIC_SUPPORT:
                "ACTIVE_LEARNED_PROVISIONAL_SUPPORT",
            DynamicEvidenceClassV1.STATIC_SUPPORT:
                "STATIC_SUPPORT_DIAGNOSTIC",
            DynamicEvidenceClassV1.SENSOR_OR_SEGMENTATION_ARTIFACT:
                "ARTIFACT_DIAGNOSTIC",
            DynamicEvidenceClassV1.UNKNOWN_AMBIGUOUS:
                "PENDING_OR_UNKNOWN_SUPPORT",
        }[label]


__all__ = ["CONTRACT_VERSION", "LearnedDynamicEvidenceAdapterV1"]
