"""Three-state continuous checker using exact authority point queries."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class ContinuousState(str, Enum):
    CERTIFIED_SAFE = "certified_safe"
    CONFIRMED_COLLISION = "confirmed_collision"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ContinuousResult:
    state: ContinuousState
    unknown_reason: str | None
    recursion_depth: int
    query_count: int
    contacted_voxel_index: list
    minimum_exact_sampled_gap_m: float
    minimum_interval_lower_bound_m: float


def certify_curve(position_at, backend, duration, *, max_depth=14):
    cache = {}
    minimum_gap = np.inf
    minimum_lower = np.inf
    contact = [-1, -1, -1]
    deepest = 0

    def query(time):
        nonlocal minimum_gap, contact
        if time not in cache:
            cache[time] = backend.query_one(position_at(time), .3)
        value = cache[time]
        minimum_gap = min(minimum_gap, value["minimum_gap_m"])
        if value["collision"]:
            contact = value["contacted_voxel_index"]
        return value

    def recurse(left_time, right_time, depth):
        nonlocal minimum_lower, deepest
        deepest = max(deepest, depth)
        left, right = query(left_time), query(right_time)
        if left["collision"] or right["collision"]:
            return ContinuousState.CONFIRMED_COLLISION
        left_position = np.asarray(position_at(left_time))
        right_position = np.asarray(position_at(right_time))
        chord = float(np.linalg.norm(right_position-left_position))
        lower = min(
            left["minimum_gap_m"], right["minimum_gap_m"]
        ) - chord
        minimum_lower = min(minimum_lower, lower)
        if lower > 1e-6:
            return ContinuousState.CERTIFIED_SAFE
        if depth >= max_depth:
            return ContinuousState.UNKNOWN
        middle = (left_time+right_time)*.5
        middle_value = query(middle)
        if middle_value["collision"]:
            return ContinuousState.CONFIRMED_COLLISION
        first = recurse(left_time, middle, depth+1)
        second = recurse(middle, right_time, depth+1)
        if ContinuousState.CONFIRMED_COLLISION in (first, second):
            return ContinuousState.CONFIRMED_COLLISION
        if ContinuousState.UNKNOWN in (first, second):
            return ContinuousState.UNKNOWN
        return ContinuousState.CERTIFIED_SAFE

    state = recurse(0.0, float(duration), 0)
    return ContinuousResult(
        state,
        "maximum_depth_without_proof"
        if state is ContinuousState.UNKNOWN else None,
        deepest, len(cache), contact, float(minimum_gap),
        float(minimum_lower),
    )


def quintic_position(start, end, duration):
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    def evaluate(time):
        u = np.clip(time/duration, 0, 1)
        blend = 10*u**3 - 15*u**4 + 6*u**5
        return start + blend*(end-start)
    return evaluate
