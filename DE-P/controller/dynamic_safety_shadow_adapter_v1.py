"""Read-only dynamic safety shadow adapter.

The adapter never changes the selected command. It evaluates candidate
positions against track predictions and returns diagnostics only.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


ADAPTER_VERSION = "dynamic_safety_shadow_adapter_v1"


@dataclass(frozen=True)
class ShadowSafetyConfig:
    uav_radius_m: float = .30
    default_actor_radius_m: float = .30
    covariance_sigma: float = 2.0
    covariance_growth_rate: float = .10
    clearance_margin_m: float = .10
    maximum_coasting_misses: int = 3
    maximum_position_std_m: float = 1.0


def _track_value(track, name, default=None):
    if isinstance(track, dict):
        return track.get(name, default)
    return getattr(track, name, default)


def contract_active(track, contract, config: ShadowSafetyConfig):
    exists = bool(_track_value(track, "track_exists", True))
    confirmed = bool(_track_value(track, "is_confirmed", False))
    dynamic = bool(_track_value(track, "is_dynamic", False))
    attention = bool(
        _track_value(track, "attention_authorized", False)
    )
    confidence = float(_track_value(track, "confidence", 0.))
    missed = int(_track_value(track, "missed_count", 0))
    previously_dynamic = bool(
        _track_value(track, "previously_dynamic", dynamic)
    )
    covariance = np.asarray(
        _track_value(track, "state_covariance", np.eye(6)),
        dtype=np.float64,
    )
    position_std = math.sqrt(max(
        float(np.linalg.eigvalsh(covariance[:3, :3]).max()), 0.
    ))
    if contract == "C0_current":
        active = exists and dynamic and attention
    elif contract == "C1_track_survival":
        active = exists and confirmed
    elif contract == "C2_recent_dynamic_coasting":
        active = (
            exists and confirmed and previously_dynamic
            and missed <= config.maximum_coasting_misses
            and position_std <= config.maximum_position_std_m
        )
    elif contract == "C3_confidence_only":
        active = exists and confidence >= .55
    else:
        raise ValueError(f"unknown shadow contract: {contract}")
    return active, {
        "track_exists": exists,
        "confirmed": confirmed,
        "dynamic": dynamic,
        "attention_authorized": attention,
        "previously_dynamic": previously_dynamic,
        "missed_count": missed,
        "confidence": confidence,
        "position_std_m": position_std,
    }


def evaluate_shadow(
    candidate_positions_world,
    sample_times_s,
    tracks,
    *,
    contract="C2_recent_dynamic_coasting",
    original_candidate_id=0,
    config=None,
):
    """Return per-candidate veto diagnostics without selecting a command."""
    config = config or ShadowSafetyConfig()
    candidates = np.asarray(candidate_positions_world, dtype=np.float64)
    times = np.asarray(sample_times_s, dtype=np.float64)
    if candidates.ndim != 3 or candidates.shape[2] != 3:
        raise ValueError("candidate_positions_world must be [N,T,3]")
    if times.shape != (candidates.shape[1],):
        raise ValueError("sample_times_s must match candidate time axis")
    rows = []
    active_tracks = []
    for track in tracks:
        active, state = contract_active(track, contract, config)
        if active:
            active_tracks.append((track, state))
    for candidate_id, candidate in enumerate(candidates):
        best = {
            "distance": float("inf"), "clearance": float("inf"),
            "time": None, "track_id": None, "radius": 0.,
            "source": None,
        }
        for track, state in active_tracks:
            position = np.asarray(
                _track_value(track, "position_world"), dtype=np.float64
            )
            velocity = np.asarray(
                _track_value(track, "velocity_world"), dtype=np.float64
            )
            covariance = np.asarray(
                _track_value(track, "state_covariance"), dtype=np.float64
            )
            radius = float(_track_value(
                track, "radius_m", config.default_actor_radius_m
            ))
            predicted = position[None, :] + times[:, None]*velocity[None, :]
            variance = max(float(np.linalg.eigvalsh(
                covariance[:3, :3]
            ).max()), 0.) + config.covariance_growth_rate*times**2
            inflated = (
                config.uav_radius_m + radius + config.clearance_margin_m
                + config.covariance_sigma*np.sqrt(variance)
            )
            distance = np.linalg.norm(candidate-predicted, axis=1)
            clearance = distance-inflated
            index = int(np.argmin(clearance))
            if clearance[index] < best["clearance"]:
                best = {
                    "distance": float(distance[index]),
                    "clearance": float(clearance[index]),
                    "time": float(times[index]),
                    "track_id": int(_track_value(track, "track_id", -1)),
                    "radius": float(inflated[index]),
                    "source": state,
                }
        veto = best["clearance"] < 0.
        rows.append({
            "candidate_trajectory_id": candidate_id,
            "predicted_minimum_actor_distance_m": best["distance"],
            "predicted_minimum_clearance_m": best["clearance"],
            "collision_time_s": best["time"] if veto else None,
            "uncertainty_inflated_safety_radius_m": best["radius"],
            "would_veto": bool(veto),
            "veto_reason":
                "predicted_uncertainty_inflated_collision"
                if veto else "none",
            "track_id": best["track_id"],
            "track_state_source": best["source"],
        })
    allowed = [
        row["candidate_trajectory_id"]
        for row in rows if not row["would_veto"]
    ]
    if int(original_candidate_id) in allowed:
        recommended = int(original_candidate_id)
    elif allowed:
        recommended = allowed[0]
    else:
        recommended = int(original_candidate_id)
    return {
        "adapter_version": ADAPTER_VERSION,
        "shadow_only": True,
        "formal_control_modified": False,
        "runtime_gt_used": False,
        "contract": contract,
        "original_candidate_id": int(original_candidate_id),
        "shadow_recommended_candidate_id": recommended,
        "candidate_rows": rows,
        "active_track_count": len(active_tracks),
    }


__all__ = [
    "ADAPTER_VERSION", "ShadowSafetyConfig", "contract_active",
    "evaluate_shadow",
]
