"""Bounded non-dynamic state used by the PEPCR1 policy wrappers."""

from __future__ import annotations

from dataclasses import dataclass
import math


CONTRACT_VERSION = "pending_support_state_v1"


@dataclass(frozen=True, slots=True)
class PendingSupportStateV1:
    state_id: int
    generation: int
    source_outcome_ids: tuple[int, ...]
    source_component_ids: tuple[int, ...]
    birth_frame: int
    birth_timestamp: float
    last_evidence_timestamp: float
    expiry_timestamp: float
    support_bounds: tuple[float, float, float, float, float, float]
    provenance_signature: tuple[str, ...]
    planner_semantic: str = "PENDING_EVIDENCE_STATE"
    dynamic_claim: bool = False
    dynamic_consumer_allowed: bool = False
    formal_tracker_feed: bool = False
    runtime_gt_used: bool = False

    def __post_init__(self):
        if self.state_id < 0 or self.generation < 0 or self.birth_frame < 0:
            raise ValueError("pending identity must be non-negative")
        if self.runtime_gt_used or self.formal_tracker_feed:
            raise ValueError("runtime GT/formal tracker feed is forbidden")
        if self.dynamic_claim or self.dynamic_consumer_allowed:
            raise ValueError("pending support cannot claim dynamic risk")
        if self.planner_semantic not in {
            "PENDING_EVIDENCE_STATE", "PENDING_OR_UNKNOWN_SUPPORT",
        }:
            raise ValueError("invalid pending semantic")
        if (
            len(self.support_bounds) != 6
            or not all(math.isfinite(value) for value in self.support_bounds)
            or any(
                self.support_bounds[index + 3] < self.support_bounds[index]
                for index in range(3)
            )
        ):
            raise ValueError("pending support bounds are invalid")
        if (
            not all(math.isfinite(value) for value in (
                self.birth_timestamp, self.last_evidence_timestamp,
                self.expiry_timestamp,
            ))
            or self.last_evidence_timestamp != self.birth_timestamp
            or self.expiry_timestamp <= self.birth_timestamp
        ):
            raise ValueError("pending lifetime is invalid")

    def cache_read(self, query_timestamp: float):
        """A read is observational and cannot refresh evidence or expiry."""
        if float(query_timestamp) < self.last_evidence_timestamp:
            raise ValueError("cache query precedes evidence")
        return self

    def is_live(self, frame_index: int, timestamp: float) -> bool:
        return (
            int(frame_index) > self.birth_frame
            and float(timestamp) <= self.expiry_timestamp
        )


def bounds_from_outcome(outcome) -> tuple[float, ...]:
    support = outcome["support"]
    return tuple(float(value) for value in (
        *support["position_set_min_world"],
        *support["position_set_max_world"],
    ))


__all__ = [
    "CONTRACT_VERSION", "PendingSupportStateV1",
    "bounds_from_outcome",
]
