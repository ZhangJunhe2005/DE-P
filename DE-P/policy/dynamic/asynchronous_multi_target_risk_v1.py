"""Asynchronous per-track robust clearance and shadow decision guard."""

from __future__ import annotations

import time

import numpy as np

from .bounded_dynamic_reachability_v1 import ReachabilityStatus
from .shape_reachable_occupancy_v1 import (
    predict_reachable_occupancy, reachable_signed_distance,
)


RISK_VERSION = "asynchronous_multi_target_risk_v1"


def evaluate_asynchronous_reachability_risk(
    candidates, times, states, builder, *,
    uav_radius_m=.30, required_margin_m=.10, candidate_scores=None,
    unresolved_dynamic_risk=False,
):
    started = time.perf_counter()
    candidates = np.asarray(candidates, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    occupancies = []
    time_rows = []
    for state in states:
        if state.status == ReachabilityStatus.EXPIRED:
            continue
        horizon = builder.horizons(state, times)
        occupancy = predict_reachable_occupancy(state, horizon)
        occupancies.append(occupancy)
        time_rows.append({
            "track_id": state.track_id, "generation": state.generation,
            "hypothesis_id": state.hypothesis_id,
            "geometry_age_s": horizon.geometry_age_s,
            "state_age_s": horizon.state_age_s,
            "candidate_relative_times_s": times.tolist(),
            "effective_prediction_horizons_s":
                horizon.effective_prediction_horizons_s.tolist(),
            "time_origin_mode": horizon.time_origin_mode.value,
            "double_age_guard": horizon.double_age_guard,
            "stale_age_included": horizon.stale_age_included,
        })
    generation_ms = (time.perf_counter()-started)*1000
    rows = []
    for candidate_id, candidate in enumerate(candidates):
        best = float("inf")
        witness = None
        for occupancy in occupancies:
            clearance = (
                reachable_signed_distance(candidate, occupancy)
                - uav_radius_m-required_margin_m
            )
            index = int(np.argmin(clearance))
            if clearance[index] < best:
                state = occupancy.state
                best = float(clearance[index])
                witness = {
                    "limiting_track": state.track_id,
                    "limiting_generation": state.generation,
                    "limiting_hypothesis": state.hypothesis_id,
                    "limiting_shape": state.shape_type,
                    "limiting_effective_horizon_s":
                        float(occupancy.effective_horizons_s[index]),
                    "limiting_geometry_age_s": state.geometry_age_s,
                    "minimum_clearance_time_s": float(times[index]),
                    "clearance_witness_type":
                        "bounded_shape_reachable_occupancy",
                }
        rows.append({
            "candidate_trajectory_id": candidate_id,
            "robust_minimum_clearance_m": best,
            "would_veto": bool(best <= 0 or unresolved_dynamic_risk),
            "witness": witness,
        })
    scores = (
        np.arange(len(candidates), dtype=np.float64)
        if candidate_scores is None
        else np.asarray(candidate_scores, dtype=np.float64)
    )
    retained = [
        row["candidate_trajectory_id"] for row in rows
        if not row["would_veto"]
    ]
    original = int(np.argmin(scores))
    recommended = (
        min(retained, key=lambda index: float(scores[index]))
        if retained else None
    )
    if unresolved_dynamic_risk:
        decision = "UNRESOLVED_DYNAMIC_RISK"
    elif not occupancies:
        decision = "NO_ACTIVE_DYNAMIC_RISK"
        recommended = original
    elif recommended is None:
        decision = "NO_SAFE_CANDIDATE"
    elif recommended == original:
        decision = "KEEP_ORIGINAL"
    else:
        decision = "SWITCH_TO_SAFE_CANDIDATE"
    return {
        "runtime_gt_used": False,
        "decision_status": decision,
        "original_candidate_id": original,
        "recommended_candidate_id": recommended,
        "candidate_rows": rows,
        "per_track_time_rows": time_rows,
        "reachability_generation_ms": generation_ms,
        "total_runtime_ms": (time.perf_counter()-started)*1000,
    }


__all__ = ["RISK_VERSION", "evaluate_asynchronous_reachability_risk"]
