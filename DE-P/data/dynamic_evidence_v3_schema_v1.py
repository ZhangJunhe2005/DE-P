"""NON_FORMAL and Formal V3 chain-sample schema."""

from __future__ import annotations

from dataclasses import dataclass


CONTRACT_VERSION = "dynamic_evidence_v3_schema_v1"


@dataclass(frozen=True, slots=True)
class DynamicEvidenceChainSampleV1:
    chain_id: str
    sequence_id: str
    anchor_frame: int
    causal_frame_indices: tuple[int, ...]
    source_component_ids: tuple[tuple[int, ...], ...]
    association_provenance: tuple[str, ...]
    map_uuid: str
    map_type: str
    scenario_family: str
    no_history: bool
    feature_path: str
    validity_mask_path: str
    main_label: str
    auxiliary_labels: tuple[tuple[str, object], ...]
    authority_metadata: tuple[tuple[str, object], ...]
    content_sha256: str
    split: str = "pilot"
    non_formal: bool = True

    def __post_init__(self):
        if not self.non_formal and self.split == "pilot":
            raise ValueError("pilot must remain NON_FORMAL")
        if max(self.causal_frame_indices) != self.anchor_frame:
            raise ValueError("chain input must be causal and anchor-bound")


__all__ = ["CONTRACT_VERSION", "DynamicEvidenceChainSampleV1"]
