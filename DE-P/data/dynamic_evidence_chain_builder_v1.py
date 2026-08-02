"""Deterministic causal chain builder shared by pilot and future loader."""

from __future__ import annotations

import hashlib
import json


CONTRACT_VERSION = "dynamic_evidence_chain_builder_v1"


def build_causal_indices(anchor_frame, temporal_length):
    anchor = int(anchor_frame)
    length = int(temporal_length)
    if anchor < 0 or length < 1:
        raise ValueError("invalid causal window")
    start = max(0, anchor-length+1)
    return tuple(range(start, anchor+1))


def stable_chain_id(sequence_id, source_components, anchor_frame):
    payload = json.dumps({
        "sequence_id": sequence_id,
        "source_components": source_components,
        "anchor_frame": int(anchor_frame),
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


__all__ = ["CONTRACT_VERSION", "build_causal_indices", "stable_chain_id"]
