"""Bounded development-only states consumed before formal confirmation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Optional, Tuple

import numpy as np


CONTRACT_VERSION = "provisional_dynamic_safety_state_v1"


class ProvisionalSafetyStateTypeV1(str, Enum):
    PROVISIONAL_MEASUREMENT_STATE = "PROVISIONAL_MEASUREMENT_STATE"
    PROVISIONAL_SUPPORT_STATE = "PROVISIONAL_SUPPORT_STATE"
    PROVISIONAL_UNRESOLVED_STATE = "PROVISIONAL_UNRESOLVED_STATE"
    FORMAL_UNCONFIRMED_SAFETY_STATE = (
        "FORMAL_UNCONFIRMED_SAFETY_STATE"
    )
    PROMOTION_PENDING = "PROMOTION_PENDING"
    PROMOTED = "PROMOTED"
    EXPIRED = "EXPIRED"
    INVALID = "INVALID"


@dataclass(frozen=True)
class ProvisionalDynamicSafetyStateV1:
    state_id: int
    generation: int
    state_type: ProvisionalSafetyStateTypeV1
    source_outcome_ids: Tuple[int, ...]
    source_component_ids: Tuple[int, ...]
    source_observation_ids: Tuple[int, ...]
    birth_timestamp: float
    last_evidence_timestamp: float
    expiry_timestamp: float
    missed_frames: int
    maximum_missed_frames: int
    evidence_frames: int
    position_reference: str
    position_world: Optional[np.ndarray] = None
    covariance_world: Optional[np.ndarray] = None
    velocity_center_mps: Optional[np.ndarray] = None
    velocity_uncertainty_mps: Optional[float] = None
    acceleration_bound_mps2: Optional[float] = None
    support_bounds: Any = None
    unresolved_evidence: Tuple[str, ...] = ()
    formal_track_id: Optional[int] = None
    formal_generation: Optional[str] = None
    confirmed: bool = False
    promotion_status: str = "UNBOUND"
    planner_semantic: str = "ACTIVE_PROVISIONAL_RISK"
    runtime_gt_used: bool = False
    formal_tracker_feed: bool = False

    def __post_init__(self):
        if self.state_id < 0 or self.generation < 0:
            raise ValueError("state ID/generation must be non-negative")
        if self.runtime_gt_used or self.formal_tracker_feed:
            raise ValueError("GT/formal tracker feed is forbidden")
        if (
            not np.isfinite((
                self.birth_timestamp, self.last_evidence_timestamp,
                self.expiry_timestamp,
            )).all()
            or self.last_evidence_timestamp < self.birth_timestamp
            or self.expiry_timestamp <= self.last_evidence_timestamp
        ):
            raise ValueError("state lifetime is invalid")
        if (
            self.missed_frames < 0 or self.maximum_missed_frames < 0
            or self.evidence_frames < 1
        ):
            raise ValueError("state evidence counters are invalid")
        no_center = {
            ProvisionalSafetyStateTypeV1.PROVISIONAL_SUPPORT_STATE,
            ProvisionalSafetyStateTypeV1.PROVISIONAL_UNRESOLVED_STATE,
        }
        if self.state_type in no_center and any((
            self.position_world is not None,
            self.covariance_world is not None,
            self.velocity_center_mps is not None,
        )):
            raise ValueError("support/unresolved state cannot fake center")
        if self.state_type == (
            ProvisionalSafetyStateTypeV1.PROVISIONAL_SUPPORT_STATE
        ) and self.support_bounds is None:
            raise ValueError("support state requires finite bounds")
        if self.state_type == (
            ProvisionalSafetyStateTypeV1.PROVISIONAL_UNRESOLVED_STATE
        ) and not self.unresolved_evidence:
            raise ValueError("unresolved state requires evidence")
        if self.state_type == (
            ProvisionalSafetyStateTypeV1.FORMAL_UNCONFIRMED_SAFETY_STATE
        ):
            if (
                self.confirmed or self.formal_track_id is None
                or self.formal_generation is None
                or self.position_world is None
            ):
                raise ValueError("unconfirmed bridge state is invalid")
        for name, shape in (
            ("position_world", (3,)),
            ("covariance_world", (3, 3)),
            ("velocity_center_mps", (3,)),
        ):
            value = getattr(self, name)
            if value is None:
                continue
            array = np.asarray(value, dtype=np.float64)
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"{name} is invalid")
            array = array.copy()
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    def cache_read(self, query_timestamp):
        """Read without refreshing evidence time or lifetime."""
        query = float(query_timestamp)
        if query < self.last_evidence_timestamp:
            raise ValueError("cache query precedes evidence")
        return self

    def miss(self, query_timestamp):
        misses = self.missed_frames+1
        expired = (
            float(query_timestamp) > self.expiry_timestamp
            or misses > self.maximum_missed_frames
        )
        return replace(
            self, missed_frames=misses,
            state_type=(
                ProvisionalSafetyStateTypeV1.EXPIRED
                if expired else self.state_type
            ),
            planner_semantic=(
                "NO_CONSUMABLE_STATE"
                if expired else self.planner_semantic
            ),
        )


class ProvisionalSafetyStateManagerV1:
    """Deterministic bounded one-to-one state association."""

    def __init__(self, maximum_position_distance_m=1.0):
        self.maximum_position_distance_m = float(
            maximum_position_distance_m
        )
        self._states = {}
        self.last_diagnostics = {}

    @property
    def states(self):
        return tuple(self._states[key] for key in sorted(self._states))

    @staticmethod
    def _center(state):
        if state.position_world is not None:
            return state.position_world
        if state.support_bounds is not None:
            return .5*(
                np.asarray(
                    state.support_bounds["min_world"], np.float64
                )
                + np.asarray(
                    state.support_bounds["max_world"], np.float64
                )
            )
        return None

    def update(self, candidates, timestamp):
        candidates = tuple(candidates)
        source_keys = [
            (
                row.source_outcome_ids,
                row.source_component_ids,
                row.source_observation_ids,
            ) for row in candidates
        ]
        if len(set(source_keys)) != len(source_keys):
            raise ValueError("duplicate source cannot birth twice")
        pairs = []
        for state in self.states:
            left = self._center(state)
            for candidate in candidates:
                right = self._center(candidate)
                same_source = bool(
                    set(state.source_observation_ids).intersection(
                        candidate.source_observation_ids
                    )
                )
                distance = (
                    np.inf if left is None or right is None else
                    float(np.linalg.norm(left-right))
                )
                if (
                    same_source
                    or distance <= self.maximum_position_distance_m
                ):
                    pairs.append((
                        0 if same_source else 1, distance,
                        state.state_id, candidate.state_id,
                    ))
        matched_old, matched_new, matches = set(), set(), []
        lookup = {row.state_id: row for row in candidates}
        for _, _, old_id, new_id in sorted(pairs):
            if old_id in matched_old or new_id in matched_new:
                continue
            old, new = self._states[old_id], lookup[new_id]
            self._states[old_id] = replace(
                new, state_id=old_id, generation=old.generation,
                birth_timestamp=old.birth_timestamp,
                evidence_frames=old.evidence_frames+1,
                missed_frames=0,
            )
            matched_old.add(old_id)
            matched_new.add(new_id)
            matches.append((old_id, new_id))
        expired = []
        for old_id in set(self._states)-matched_old:
            missed = self._states[old_id].miss(timestamp)
            if missed.state_type == ProvisionalSafetyStateTypeV1.EXPIRED:
                expired.append(old_id)
                del self._states[old_id]
            else:
                self._states[old_id] = missed
        for candidate in candidates:
            if candidate.state_id not in matched_new:
                if candidate.state_id in self._states:
                    raise ValueError("new state ID collides with live state")
                self._states[candidate.state_id] = candidate
        self.last_diagnostics = {
            "one_to_one": (
                len({row[0] for row in matches}) == len(matches)
                and len({row[1] for row in matches}) == len(matches)
            ),
            "matches": tuple(matches),
            "expired_ids": tuple(sorted(expired)),
            "runtime_gt_used": False,
        }
        return self.states


__all__ = [
    "CONTRACT_VERSION", "ProvisionalSafetyStateTypeV1",
    "ProvisionalDynamicSafetyStateV1",
    "ProvisionalSafetyStateManagerV1",
]
