"""Authoritative continuous static collision certificates.

The final physical query is always ``static_geometry_authority_v1`` through
``ExactAuthorityBVH``.  Bézier convex-hull bounds certify complete line,
quadratic and quintic trajectory segments; exhausting the recursion budget
without a proof returns UNKNOWN.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np

from geometry_authority.static_v1 import (
    CONTACT_TOLERANCE_M,
    DEFAULT_UAV_RADIUS_M,
)


IMPLEMENTATION_VERSION = "static_continuous_authority_v1"


class CertificateState(str, Enum):
    CERTIFIED_SAFE = "certified_safe"
    CONFIRMED_COLLISION = "confirmed_collision"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class StaticAuthorityCertificate:
    state: CertificateState
    minimum_sampled_gap_m: float
    minimum_proven_lower_bound_m: float
    maximum_depth: int
    queried_sample_count: int
    queried_voxel_count: int
    unresolved_interval_count: int
    remaining_bound_gap_m: float
    unknown_reason: str | None
    contacted_voxel_index: tuple[int, int, int]
    out_of_bounds: bool
    uav_radius_m: float
    contact_tolerance_m: float
    map_authority_hash: str
    implementation_version: str = IMPLEMENTATION_VERSION


def _validate_state(state):
    value = np.asarray(state, dtype=np.float64)
    if value.shape != (3, 3) or not np.isfinite(value).all():
        raise ValueError("state must be finite [xyz,pva]")
    return value


def quintic_bezier_control_points(start_state, end_state, duration):
    """Return six xyz Bézier points for a quintic P/V/A boundary segment."""
    start = _validate_state(start_state)
    end = _validate_state(end_state)
    duration = float(duration)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and positive")
    points = np.empty((6, 3), dtype=np.float64)
    points[0] = start[:, 0]
    points[1] = points[0] + start[:, 1] * duration / 5.0
    points[2] = (
        start[:, 2] * duration**2 / 20.0 + 2.0 * points[1] - points[0]
    )
    points[5] = end[:, 0]
    points[4] = points[5] - end[:, 1] * duration / 5.0
    points[3] = (
        end[:, 2] * duration**2 / 20.0 + 2.0 * points[4] - points[5]
    )
    return points


def quadratic_bezier_control_points(initial_state, duration):
    """Return the exact constant-acceleration latency-prefix curve."""
    state = _validate_state(initial_state)
    duration = float(duration)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and positive")
    start = state[:, 0]
    end = start + state[:, 1] * duration + 0.5 * state[:, 2] * duration**2
    return np.stack(
        (start, start + 0.5 * state[:, 1] * duration, end), axis=0
    )


def line_bezier_control_points(start, end):
    points = np.asarray((start, end), dtype=np.float64)
    if points.shape != (2, 3) or not np.isfinite(points).all():
        raise ValueError("line endpoints must be finite xyz")
    return points


def _split_bezier(control):
    levels = [np.asarray(control, dtype=np.float64)]
    while len(levels[-1]) > 1:
        current = levels[-1]
        levels.append(0.5 * (current[:-1] + current[1:]))
    left = np.stack([level[0] for level in levels], axis=0)
    right = np.stack([level[-1] for level in levels[::-1]], axis=0)
    return left, right


def certify_bezier_authority(
    backend,
    control_points,
    *,
    uav_radius_m=DEFAULT_UAV_RADIUS_M,
    contact_tolerance_m=CONTACT_TOLERANCE_M,
    max_depth=18,
):
    """Certify one complete Bézier curve against the exact authority.

    The signed sphere-to-occupancy gap is 1-Lipschitz.  For each queried
    anchor, ``gap(anchor) - max(distance(anchor, control hull))`` is therefore
    a conservative lower bound over the complete curve.
    """
    control = np.asarray(control_points, dtype=np.float64)
    radius = float(uav_radius_m)
    tolerance = float(contact_tolerance_m)
    max_depth = int(max_depth)
    if (
        control.ndim != 2 or control.shape[1] != 3 or len(control) < 2
        or not np.isfinite(control).all()
    ):
        raise ValueError("control_points must be finite [degree+1,3]")
    if abs(radius - DEFAULT_UAV_RADIUS_M) > 1e-12:
        raise ValueError("authoritative UAV radius is frozen at 0.3 m")
    if abs(tolerance - CONTACT_TOLERANCE_M) > 1e-15:
        raise ValueError("authoritative contact tolerance is frozen at 1e-6 m")
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")

    queries = {}
    minimum_gap = math.inf
    minimum_lower = math.inf
    queried_voxels = 0
    deepest = 0
    unresolved = 0
    remaining_gap = 0.0
    collision_query = None

    def query(point):
        nonlocal minimum_gap, queried_voxels, collision_query
        key = tuple(np.asarray(point, dtype=np.float64).tobytes())
        if key not in queries:
            result = backend.query_one(point, radius)
            queries[key] = result
            gap = float(result["minimum_gap_m"])
            if not math.isfinite(gap):
                raise RuntimeError("authority backend returned non-finite gap")
            minimum_gap = min(minimum_gap, gap)
            queried_voxels += int(result["queried_voxel_count"])
            if bool(result["collision"]) and collision_query is None:
                collision_query = result
        return queries[key]

    def recurse(points, depth):
        nonlocal deepest, minimum_lower, unresolved, remaining_gap
        deepest = max(deepest, depth)
        left_result = query(points[0])
        if collision_query is not None:
            return CertificateState.CONFIRMED_COLLISION
        right_result = query(points[-1])
        if collision_query is not None:
            return CertificateState.CONFIRMED_COLLISION
        midpoint = _split_bezier(points)[0][-1]
        middle_result = query(midpoint)
        if collision_query is not None:
            return CertificateState.CONFIRMED_COLLISION

        lower_bounds = []
        for anchor, result in (
            (points[0], left_result),
            (points[-1], right_result),
            (midpoint, middle_result),
        ):
            hull_radius = float(np.linalg.norm(points - anchor, axis=1).max())
            lower_bounds.append(float(result["minimum_gap_m"]) - hull_radius)
        lower = max(lower_bounds)
        minimum_lower = min(minimum_lower, lower)
        if lower > tolerance:
            return CertificateState.CERTIFIED_SAFE
        if depth >= max_depth:
            unresolved += 1
            remaining_gap = max(remaining_gap, tolerance - lower)
            return CertificateState.UNKNOWN
        left, right = _split_bezier(points)
        left_state = recurse(left, depth + 1)
        if left_state is CertificateState.CONFIRMED_COLLISION:
            return left_state
        right_state = recurse(right, depth + 1)
        if right_state is CertificateState.CONFIRMED_COLLISION:
            return right_state
        if (
            left_state is CertificateState.CERTIFIED_SAFE
            and right_state is CertificateState.CERTIFIED_SAFE
        ):
            return CertificateState.CERTIFIED_SAFE
        return CertificateState.UNKNOWN

    state = recurse(control, 0)
    metadata = backend.map.metadata
    collision = collision_query or {}
    voxel = tuple(
        int(value) for value in collision.get(
            "contacted_voxel_index", [-1, -1, -1]
        )
    )
    return StaticAuthorityCertificate(
        state=state,
        minimum_sampled_gap_m=float(minimum_gap),
        minimum_proven_lower_bound_m=float(minimum_lower),
        maximum_depth=deepest,
        queried_sample_count=len(queries),
        queried_voxel_count=queried_voxels,
        unresolved_interval_count=unresolved,
        remaining_bound_gap_m=float(remaining_gap),
        unknown_reason=(
            "recursion_budget_exhausted"
            if state is CertificateState.UNKNOWN else None
        ),
        contacted_voxel_index=voxel,
        out_of_bounds=bool(collision.get("out_of_bounds", False)),
        uav_radius_m=radius,
        contact_tolerance_m=tolerance,
        map_authority_hash=str(metadata["artifact_manifest_hash"]),
    )


def combine_authority_certificates(certificates):
    values = tuple(certificates)
    if not values:
        raise ValueError("at least one certificate is required")
    states = {value.state for value in values}
    if CertificateState.CONFIRMED_COLLISION in states:
        return CertificateState.CONFIRMED_COLLISION
    if states == {CertificateState.CERTIFIED_SAFE}:
        return CertificateState.CERTIFIED_SAFE
    return CertificateState.UNKNOWN

