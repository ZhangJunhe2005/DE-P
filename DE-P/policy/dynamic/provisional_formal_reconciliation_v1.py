"""Deterministic one-to-one promotion and duplicate suppression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .provisional_dynamic_safety_state_v1 import (
    ProvisionalDynamicSafetyStateV1,
)


CONTRACT_VERSION = "provisional_formal_reconciliation_v1"


@dataclass(frozen=True)
class ReconciliationDecisionV1:
    provisional_state_id: int
    status: str
    formal_identity: str | None
    reason: str
    runtime_gt_used: bool = False

    def __post_init__(self):
        if self.runtime_gt_used:
            raise ValueError("runtime GT is forbidden")
        if self.status not in {
            "PROMOTED", "DUPLICATE_SUPPRESSED",
            "INDEPENDENT", "AMBIGUOUS",
        }:
            raise ValueError("invalid reconciliation status")


def _center_from_support(bounds):
    if bounds is None:
        return None
    return .5*(
        np.asarray(bounds["min_world"], np.float64)
        + np.asarray(bounds["max_world"], np.float64)
    )


class ProvisionalFormalReconciliationV1:
    def __init__(
        self, maximum_time_delta_s=.12,
        maximum_position_distance_m=1.,
    ):
        self.maximum_time_delta_s = float(maximum_time_delta_s)
        self.maximum_position_distance_m = float(
            maximum_position_distance_m
        )

    def reconcile(
        self, states: Sequence[ProvisionalDynamicSafetyStateV1],
        formal_tracks: Sequence[Mapping], timestamp: float,
    ):
        candidates = []
        for state in states:
            state_center = (
                state.position_world
                if state.position_world is not None
                else _center_from_support(state.support_bounds)
            )
            compatible = []
            for track in formal_tracks:
                identity = str(track["identity"])
                provenance = set(track.get("observation_ids", ()))
                overlap = bool(
                    provenance.intersection(
                        state.source_observation_ids
                    )
                )
                distance = (
                    np.inf if state_center is None else
                    float(np.linalg.norm(
                        np.asarray(
                            track["position_world"], np.float64
                        )-state_center
                    ))
                )
                time_ok = abs(
                    float(timestamp)-state.last_evidence_timestamp
                ) <= self.maximum_time_delta_s
                generation_ok = (
                    state.formal_generation is None
                    or state.formal_generation == identity
                )
                if (
                    time_ok and generation_ok
                    and (overlap or distance
                         <= self.maximum_position_distance_m)
                ):
                    compatible.append((
                        0 if overlap else 1, distance, identity, overlap
                    ))
            compatible.sort()
            candidates.append((state, compatible))
        assigned = set()
        decisions = []
        for state, compatible in sorted(
            candidates, key=lambda item: item[0].state_id
        ):
            available = [
                row for row in compatible if row[2] not in assigned
            ]
            if not available:
                decisions.append(ReconciliationDecisionV1(
                    state.state_id, "INDEPENDENT", None,
                    "no_generation_safe_compatible_formal",
                ))
                continue
            if len(available) > 1 and available[0][:2] == available[1][:2]:
                decisions.append(ReconciliationDecisionV1(
                    state.state_id, "AMBIGUOUS", None,
                    "equal_compatibility_candidates",
                ))
                continue
            _, _, identity, provenance = available[0]
            assigned.add(identity)
            decisions.append(ReconciliationDecisionV1(
                state.state_id,
                "PROMOTED" if provenance else "DUPLICATE_SUPPRESSED",
                identity,
                "provenance_match" if provenance else "bounded_overlap",
            ))
        return tuple(decisions)


__all__ = [
    "CONTRACT_VERSION", "ReconciliationDecisionV1",
    "ProvisionalFormalReconciliationV1",
]
