"""Frozen DEM-DCR1 model-contract registry."""

from __future__ import annotations

from dataclasses import dataclass


CONTRACT_VERSION = "dynamic_evidence_model_contract_v1"


@dataclass(frozen=True, slots=True)
class DynamicEvidenceModelContractV1:
    contract_id: str
    feature_dim: int
    temporal_lengths: tuple[int, ...]
    output_classes: tuple[str, ...]
    auxiliary_outputs: tuple[str, ...]
    causal_only: bool = True
    supports_multi_target_batch: bool = True
    production_enabled: bool = False


SELECTED_MODEL_CONTRACT = DynamicEvidenceModelContractV1(
    contract_id="causal_feature_temporal_dynamic_evidence_v1",
    feature_dim=32, temporal_lengths=(2, 4),
    output_classes=(
        "DYNAMIC_SUPPORT", "STATIC_SUPPORT",
        "SENSOR_OR_SEGMENTATION_ARTIFACT", "UNKNOWN_AMBIGUOUS",
    ),
    auxiliary_outputs=("dynamic_actionability",),
)


__all__ = [
    "CONTRACT_VERSION", "DynamicEvidenceModelContractV1",
    "SELECTED_MODEL_CONTRACT",
]
