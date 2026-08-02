#!/usr/bin/env python3
"""Offline-only GT risk evaluation for already generated YOPO candidates."""

from __future__ import annotations

import numpy as np


EVALUATOR_VERSION = "yopo_dynamic_candidate_gt_risk_v1"


def evaluate_gt_candidate_risk(
    candidate_positions_world,
    sample_times_s,
    actor_positions_world,
    actor_radii_m,
    *,
    uav_radius_m=.30,
    required_margin_m=.10,
    actor_valid=None,
):
    """Classify candidate risk without feeding GT into runtime decisions.

    ``actor_positions_world`` is ``[A,T,3]``. ``actor_valid`` is ``[A,T]`` and
    marks where the offline trajectory has authority.
    """
    candidates = np.asarray(candidate_positions_world, dtype=np.float64)
    times = np.asarray(sample_times_s, dtype=np.float64)
    actors = np.asarray(actor_positions_world, dtype=np.float64)
    radii = np.asarray(actor_radii_m, dtype=np.float64)
    if candidates.ndim != 3 or candidates.shape[-1] != 3:
        raise ValueError("candidates must have shape [N,T,3]")
    if times.shape != (candidates.shape[1],):
        raise ValueError("sample_times_s must match candidate time samples")
    if actors.ndim != 3 or actors.shape[1:] != (len(times), 3):
        raise ValueError("actor positions must have shape [A,T,3]")
    if radii.shape != (actors.shape[0],):
        raise ValueError("actor radii must have shape [A]")
    if actor_valid is None:
        valid = np.isfinite(actors).all(axis=-1)
    else:
        valid = np.asarray(actor_valid, dtype=bool)
        if valid.shape != actors.shape[:2]:
            raise ValueError("actor_valid must have shape [A,T]")
        valid &= np.isfinite(actors).all(axis=-1)
    if not np.isfinite(candidates).all() or not np.isfinite(times).all():
        raise ValueError("candidate inputs must be finite")
    if np.any(radii <= 0) or uav_radius_m <= 0 or required_margin_m < 0:
        raise ValueError("radii/margin are invalid")

    rows = []
    for candidate_id, candidate in enumerate(candidates):
        best = {
            "clearance": float("inf"), "physical_clearance": float("inf"),
            "distance": float("inf"), "actor": None, "time": None,
        }
        any_valid = False
        complete = actors.shape[0] == 0 or bool(valid.all())
        for actor_id, (actor, radius, actor_mask) in enumerate(
            zip(actors, radii, valid)
        ):
            if not actor_mask.any():
                continue
            any_valid = True
            distance = np.linalg.norm(candidate[actor_mask] - actor[actor_mask], axis=1)
            physical = distance - uav_radius_m - radius
            clearance = physical - required_margin_m
            local = int(np.argmin(clearance))
            indices = np.flatnonzero(actor_mask)
            index = int(indices[local])
            if clearance[local] < best["clearance"]:
                best = {
                    "clearance": float(clearance[local]),
                    "physical_clearance": float(physical[local]),
                    "distance": float(distance[local]),
                    "actor": actor_id,
                    "time": float(times[index]),
                }
        if actors.shape[0] == 0:
            classification = "GT_SAFE"
            best["clearance"] = float("inf")
        elif not any_valid or not complete:
            classification = "GT_UNKNOWN_OUTSIDE_HORIZON"
        elif best["physical_clearance"] < 0:
            classification = "GT_COLLISION"
        elif best["clearance"] < 0:
            classification = "GT_UNSAFE_CLEARANCE"
        else:
            classification = "GT_SAFE"
        rows.append({
            "candidate_trajectory_id": candidate_id,
            "classification": classification,
            "minimum_distance_m": best["distance"],
            "minimum_physical_clearance_m": best["physical_clearance"],
            "minimum_required_clearance_m": best["clearance"],
            "minimum_clearance_time_s": best["time"],
            "limiting_actor_index": best["actor"],
        })
    return {
        "evaluator_version": EVALUATOR_VERSION,
        "offline_only": True,
        "runtime_input": False,
        "candidate_rows": rows,
        "true_unsafe_candidate_ids": [
            row["candidate_trajectory_id"] for row in rows
            if row["classification"] in {
                "GT_COLLISION", "GT_UNSAFE_CLEARANCE"
            }
        ],
        "safe_candidate_ids": [
            row["candidate_trajectory_id"] for row in rows
            if row["classification"] == "GT_SAFE"
        ],
        "unknown_candidate_ids": [
            row["candidate_trajectory_id"] for row in rows
            if row["classification"] == "GT_UNKNOWN_OUTSIDE_HORIZON"
        ],
    }


def compare_shadow_with_gt(shadow, ground_truth):
    gt = {
        row["candidate_trajectory_id"]: row
        for row in ground_truth["candidate_rows"]
    }
    rows = []
    for predicted in shadow["candidate_rows"]:
        truth = gt[predicted["candidate_trajectory_id"]]
        unsafe = truth["classification"] in {
            "GT_COLLISION", "GT_UNSAFE_CLEARANCE"
        }
        known_safe = truth["classification"] == "GT_SAFE"
        veto = bool(predicted["would_veto"])
        rows.append({
            "candidate_trajectory_id":
                predicted["candidate_trajectory_id"],
            "gt_classification": truth["classification"],
            "would_veto": veto,
            "correctly_vetoed_unsafe": unsafe and veto,
            "missed_unsafe": unsafe and not veto,
            "false_vetoed_safe": known_safe and veto,
            "retained_safe": known_safe and not veto,
        })
    original = int(shadow["original_candidate_id"])
    recommendation = shadow["recommended_candidate_id"]
    safe = set(ground_truth["safe_candidate_ids"])
    unsafe = set(ground_truth["true_unsafe_candidate_ids"])
    all_known_unsafe = (
        len(unsafe) == len(gt) and not ground_truth["unknown_candidate_ids"]
    )
    return {
        "candidate_rows": rows,
        "true_unsafe_candidates": len(unsafe),
        "correctly_vetoed_unsafe": sum(
            row["correctly_vetoed_unsafe"] for row in rows
        ),
        "missed_unsafe": sum(row["missed_unsafe"] for row in rows),
        "false_vetoed_safe": sum(
            row["false_vetoed_safe"] for row in rows
        ),
        "retained_safe": sum(row["retained_safe"] for row in rows),
        "all_candidates_vetoed": bool(
            shadow["candidate_rows"]
            and all(row["would_veto"] for row in shadow["candidate_rows"])
        ),
        "no_safe_candidate_truth": all_known_unsafe,
        "original_candidate_gt_safe": original in safe,
        "safe_alternative_available": bool(safe - {original}),
        "recommended_candidate_gt_safe":
            recommendation in safe if recommendation is not None else False,
        "unsafe_recommendation":
            recommendation in unsafe if recommendation is not None else False,
        "no_safe_candidate_correct": (
            shadow["decision_status"] == "NO_SAFE_CANDIDATE"
            and all_known_unsafe
        ),
        "false_emergency": (
            shadow["decision_status"] == "NO_SAFE_CANDIDATE"
            and bool(safe)
        ),
    }


__all__ = [
    "EVALUATOR_VERSION", "evaluate_gt_candidate_risk",
    "compare_shadow_with_gt",
]
