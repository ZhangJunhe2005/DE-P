"""Fail-closed causal audit for frames without a foreground component."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


CONTRACT_VERSION = "cold_start_foreground_availability_v1"


@dataclass(frozen=True)
class ColdStartForegroundAuditV1:
    frame_index: int
    has_previous_frame: bool
    history_size: int
    finite_depth_count: int
    foreground_component_count: int
    causal_support_available: bool
    classification: str
    support_created: bool = False
    runtime_gt_used: bool = False

    def __post_init__(self):
        if self.runtime_gt_used:
            raise ValueError("runtime GT is forbidden")
        if self.support_created and not self.causal_support_available:
            raise ValueError("cannot invent cold-start support")


def audit_cold_start(
    frame_index, depth, history_size, foreground_component_count,
):
    finite = int(np.isfinite(np.asarray(depth)).sum())
    available = bool(
        history_size > 0 and foreground_component_count > 0
    )
    return ColdStartForegroundAuditV1(
        frame_index=int(frame_index),
        has_previous_frame=history_size > 0,
        history_size=int(history_size),
        finite_depth_count=finite,
        foreground_component_count=int(
            foreground_component_count
        ),
        causal_support_available=available,
        classification=(
            "CAUSAL_FOREGROUND_SUPPORT"
            if available else "FOREGROUND_INITIALIZATION_LIMIT"
        ),
        support_created=available,
        runtime_gt_used=False,
    )


__all__ = [
    "CONTRACT_VERSION", "ColdStartForegroundAuditV1",
    "audit_cold_start",
]
