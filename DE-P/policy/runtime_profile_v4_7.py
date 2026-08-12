"""V4.7 balanced runtime profile for causal moving-obstacle trials.

The profile keeps the V4.5.10 hard physical floor and candidate retiming.  It
adds no progress/FOV/qualification Gate.  Its only static-path change is a
small bounded clearance preference among already feasible candidates.  The
dynamic radius matches the largest interactive actor rather than the older
0.35 m generic assumption.
"""

from __future__ import annotations

from policy.runtime_profile_v4_5_10 import runtime_safety_mapping_v4_5_10


PROFILE_NAME = "v4_7_balanced_dynamic"
RUNTIME_BEHAVIOR_VERSION = (
    "v4_7_range_image_dynamic_filter_bounded_clearance_preference_v2"
)


def runtime_safety_mapping_v4_7(base):
    value = runtime_safety_mapping_v4_5_10(base)
    value.update({
        # Interactive actors use radii up to 0.46 m.  Use 0.50 m rather than
        # under-approximating them with the historical 0.35 m default.
        "dynamic_track_radius_m": 0.50,
        "dynamic_track_max_age_s": 0.40,
        # Only the near-term controllable horizon is a hard dynamic veto.  A
        # 3.33 s constant-velocity extrapolation is not valid for ping-pong
        # actors and previously rejected every route long before it became
        # actionable.  The planner replans at sensor rate.
        "dynamic_track_prediction_horizon_s": 1.50,
        "dynamic_track_covariance_sigma": 2.0,
        # A maximum 0.15 score benefit remains only a tie-breaker between
        # candidates that already passed every physical check.  Concentrating
        # the benefit in the first 0.45 m of spare clearance discourages
        # grazing without rewarding progressively wider detours.
        "clearance_preference_weight": 0.15,
        "clearance_preference_saturation_m": 0.45,
    })
    return value


__all__ = [
    "PROFILE_NAME", "RUNTIME_BEHAVIOR_VERSION",
    "runtime_safety_mapping_v4_7",
]
