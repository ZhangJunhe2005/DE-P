"""Versioned three-state static continuous certificate (audit-only).

This module repairs the numerical interpretation of the frozen V2 geometry:
raw ESDF distance minus a 0.3 m UAV radius.  It is intentionally not wired
into SafetyEvaluatorV2 in Phase 8J-V2-S0.  A separate numerical rebaseline is
required before it can become authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


IMPLEMENTATION_VERSION = "phase8jqv2_static_continuous_v2_1"


class CertificateState(str, Enum):
    CONFIRMED_COLLISION = "confirmed_collision"
    CERTIFIED_SAFE = "certified_safe"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class StaticContinuousCertificate:
    state: CertificateState
    minimum_sampled_clearance_m: float
    maximum_depth: int
    queried_sample_count: int
    uav_radius_m: float
    static_tolerance_m: float
    contact_tolerance_m: float


def certify_dyadic_samples(
    positions,
    raw_esdf_distance,
    *,
    uav_radius_m=0.3,
    tolerance_m=0.005,
    contact_tolerance_m=1e-6,
):
    """Classify one segment from a complete dyadic sample grid.

    ``len(samples)-1`` must be a power of two.  A collision is confirmed only
    by a queried sample inside the physical radius.  Safety is certified by
    the 1-Lipschitz ESDF endpoint lower bound, recursively.  Exhausting the
    supplied grid without either proof returns UNKNOWN rather than collision.
    """
    positions = np.asarray(positions, dtype=float)
    raw = np.asarray(raw_esdf_distance, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must be [sample,3]")
    if raw.shape != (len(positions),):
        raise ValueError("raw_esdf_distance must match positions")
    intervals = len(positions) - 1
    if intervals <= 0 or intervals & (intervals - 1):
        raise ValueError("sample interval count must be a positive power of two")
    if not np.isfinite(positions).all() or not np.isfinite(raw).all():
        raise ValueError("certificate inputs must be finite")
    radius = float(uav_radius_m)
    tolerance = float(tolerance_m)
    contact_tolerance = float(contact_tolerance_m)
    if radius <= 0 or tolerance <= 0 or contact_tolerance < 0:
        raise ValueError("radius/tolerance must be positive and contact tolerance non-negative")
    minimum = float(raw.min() - radius)
    if minimum < -contact_tolerance:
        return StaticContinuousCertificate(
            CertificateState.CONFIRMED_COLLISION,
            minimum, int(np.log2(intervals)), len(raw), radius, tolerance,
            contact_tolerance,
        )

    def recurse(left, right):
        chord = float(np.linalg.norm(positions[right] - positions[left]))
        lower = min(float(raw[left]), float(raw[right])) - chord - radius
        if lower >= -contact_tolerance:
            return True
        if right - left == 1:
            return False
        middle = (left + right) // 2
        return recurse(left, middle) and recurse(middle, right)

    safe = recurse(0, intervals)
    return StaticContinuousCertificate(
        CertificateState.CERTIFIED_SAFE if safe else CertificateState.UNKNOWN,
        minimum, int(np.log2(intervals)), len(raw), radius, tolerance,
        contact_tolerance,
    )


def combine_segment_certificates(certificates):
    certificates = tuple(certificates)
    if not certificates:
        raise ValueError("at least one segment certificate is required")
    states = {value.state for value in certificates}
    if CertificateState.CONFIRMED_COLLISION in states:
        return CertificateState.CONFIRMED_COLLISION
    if states == {CertificateState.CERTIFIED_SAFE}:
        return CertificateState.CERTIFIED_SAFE
    return CertificateState.UNKNOWN
