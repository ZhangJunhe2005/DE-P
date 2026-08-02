#!/usr/bin/env python3
"""Exact runtime-safe sphere/finite-cylinder multi-hypothesis risk."""

from __future__ import annotations

import numpy as np


EVALUATOR_VERSION = "dynamic_geometry_risk_evaluator_v1"


def sphere_signed_distance(points, centers, radius):
    points = np.asarray(points, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    return np.linalg.norm(points-centers, axis=-1)-float(radius)


def finite_vertical_cylinder_signed_distance(
    points, centers, radius, half_height,
):
    points = np.asarray(points, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    radial = np.linalg.norm(points[..., :2]-centers[..., :2], axis=-1)
    q_radial = radial-float(radius)
    q_vertical = np.abs(points[..., 2]-centers[..., 2])-float(half_height)
    outside = np.sqrt(
        np.maximum(q_radial, 0.)**2
        + np.maximum(q_vertical, 0.)**2
    )
    inside = np.minimum(np.maximum(q_radial, q_vertical), 0.)
    return outside+inside


def evaluate_dynamic_geometry_risk(
    candidate_positions_world, sample_times_s, occupancy_states, *,
    uav_radius_m=.30, required_margin_m=.10,
):
    candidates = np.asarray(
        candidate_positions_world, dtype=np.float64
    )
    times = np.asarray(sample_times_s, dtype=np.float64)
    if candidates.ndim != 3 or candidates.shape[1:] != (len(times), 3):
        raise ValueError("candidates must be [N,T,3]")
    if not np.isfinite(candidates).all() or not np.isfinite(times).all():
        raise ValueError("candidate inputs must be finite")
    rows = []
    for candidate_id, candidate in enumerate(candidates):
        best = {
            "clearance": float("inf"), "hypothesis": None,
            "track_id": None, "time": None,
        }
        for state in occupancy_states:
            elapsed = times+(times*0+max(
                0., float(getattr(state, "prediction_age_s", 0.))
            ))
            for hypothesis in state.shape_hypotheses:
                centers = (
                    hypothesis.center_world[None, :]
                    + elapsed[:, None]*hypothesis.velocity_world[None, :]
                )
                if hypothesis.geometry_type == "sphere":
                    signed = sphere_signed_distance(
                        candidate, centers,
                        hypothesis.radius_interval_m[1],
                    )
                elif hypothesis.geometry_type == "vertical_cylinder":
                    signed = finite_vertical_cylinder_signed_distance(
                        candidate, centers,
                        hypothesis.radius_interval_m[1],
                        hypothesis.half_height_interval_m[1],
                    )
                else:
                    raise ValueError("unknown geometry hypothesis")
                clearance = signed-uav_radius_m-required_margin_m
                index = int(np.argmin(clearance))
                if clearance[index] < best["clearance"]:
                    best = {
                        "clearance": float(clearance[index]),
                        "hypothesis": hypothesis.geometry_type,
                        "track_id": int(state.track_id),
                        "time": float(times[index]),
                    }
        rows.append({
            "candidate_trajectory_id": candidate_id,
            "predicted_minimum_clearance_m": best["clearance"],
            "limiting_geometry_type": best["hypothesis"],
            "limiting_track_id": best["track_id"],
            "minimum_clearance_time_s": best["time"],
            "would_veto": bool(best["clearance"] < 0),
        })
    return {
        "evaluator_version": EVALUATOR_VERSION,
        "runtime_gt_used": False,
        "exact_per_shape_evaluation": True,
        "multi_hypothesis_reduction": "minimum_clearance",
        "candidate_rows": rows,
    }


__all__ = [
    "EVALUATOR_VERSION", "sphere_signed_distance",
    "finite_vertical_cylinder_signed_distance",
    "evaluate_dynamic_geometry_risk",
]
