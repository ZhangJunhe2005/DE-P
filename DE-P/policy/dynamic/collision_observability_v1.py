"""Conservative observability labels for dynamic collision diagnostics.

This module deliberately does not make planning decisions.  It combines the
Simulator's per-actor render diagnostics with the most recent planner
telemetry to answer a narrower post-hoc question: was the colliding actor
visible, uniquely associated with a track, demonstrably unavailable to the
planner, or not classifiable from the evidence that was recorded?

The current planner telemetry reports track IDs but does not report a binding
between those IDs and Simulator ground-truth object IDs.  For diagnostics
only, this module can therefore align a fresh track prediction with the
current ground-truth actor position and apply a strict bidirectional spatial
gate.  Ambiguous or non-mutual matches remain ``unknown``.  Ground truth is
never returned to the planner and never changes control.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional


CONTRACT_VERSION = "dynamic_collision_observability_v1"
POSTHOC_BINDING_METHOD = "posthoc_spatial_binding"

SIMULATOR_VISIBILITY_FIELDS = (
    "visible",
    "occluded",
    "inside_image",
    "projected_u",
    "projected_v",
    "expected_surface_depth",
    "observed_depth",
    "depth_error",
    "rendered_pixel_count",
)


def _optional_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    return bool(value)


def _optional_finite_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _optional_nonnegative_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _finite_vector3(value: Any) -> Optional[tuple[float, float, float]]:
    if isinstance(value, (str, bytes, Mapping)):
        return None
    try:
        values = tuple(value)
    except TypeError:
        return None
    if len(values) != 3:
        return None
    result = tuple(_optional_finite_float(item) for item in values)
    if any(item is None for item in result):
        return None
    return result  # type: ignore[return-value]


def _state_position(state: Mapping[str, Any]) -> Optional[tuple[float, float, float]]:
    return _finite_vector3(state.get("position_world", state.get("position")))


def _state_velocity(state: Mapping[str, Any]) -> tuple[float, float, float]:
    value = _finite_vector3(state.get("velocity_world", state.get("velocity")))
    return (0.0, 0.0, 0.0) if value is None else value


def _empty_posthoc_binding(
    actor_id: int,
    *,
    status: str,
    telemetry_age_s: Optional[float],
    time_threshold_s: float,
    distance_threshold_m: float,
    ambiguity_margin_m: float,
) -> dict[str, Any]:
    return {
        "binding_method": POSTHOC_BINDING_METHOD,
        "binding_status": status,
        "actor_id": int(actor_id),
        "matched_track_id": None,
        "distance_m": None,
        "second_best_distance_m": None,
        "ambiguity_gap_m": None,
        "distance_threshold_m": float(distance_threshold_m),
        "time_offset_s": telemetry_age_s,
        "time_threshold_s": float(time_threshold_s),
        "time_basis": "planner_telemetry_receipt_age_monotonic",
        "ambiguity_margin_m": float(ambiguity_margin_m),
        "candidate_track_ids": [],
        "candidate_track_count": 0,
        "diagnostic_only": True,
        "planner_control_affected": False,
    }


def associate_gt_actors_to_tracks_posthoc(
    *,
    actor_states: Mapping[int, Mapping[str, Any]],
    planner_telemetry: Optional[Mapping[str, Any]],
    planner_telemetry_age_s: Optional[float],
    time_threshold_s: float = 0.5,
    distance_threshold_m: float = 1.0,
    ambiguity_margin_m: float = 0.25,
) -> dict[int, dict[str, Any]]:
    """Bind GT actors to planner tracks for post-hoc diagnostics only.

    Track positions are advanced by the telemetry receipt age using the
    reported constant velocity.  A match is accepted only if it is inside the
    distance/time gates, is the mutual nearest actor/track pair, and has the
    requested separation from every competing gated match on both sides.
    This prevents one planner track from being credited to two nearby actors.
    """

    if time_threshold_s <= 0:
        raise ValueError("time_threshold_s must be positive")
    if distance_threshold_m <= 0:
        raise ValueError("distance_threshold_m must be positive")
    if ambiguity_margin_m < 0:
        raise ValueError("ambiguity_margin_m must be non-negative")

    actor_ids = sorted(int(actor_id) for actor_id in actor_states)
    telemetry_age = _optional_finite_float(planner_telemetry_age_s)
    results = {
        actor_id: _empty_posthoc_binding(
            actor_id,
            status="binding_unavailable",
            telemetry_age_s=telemetry_age,
            time_threshold_s=time_threshold_s,
            distance_threshold_m=distance_threshold_m,
            ambiguity_margin_m=ambiguity_margin_m,
        )
        for actor_id in actor_ids
    }
    if not isinstance(planner_telemetry, Mapping):
        return results
    if telemetry_age is None or not 0.0 <= telemetry_age <= time_threshold_s:
        for result in results.values():
            result["binding_status"] = "time_gate_failed"
        return results

    actor_positions = {
        actor_id: _state_position(actor_states[actor_id])
        for actor_id in actor_ids
    }
    tracks = []
    seen_track_ids = set()
    duplicate_track_ids = set()
    for state in planner_telemetry.get("dynamic_track_summaries") or ():
        if not isinstance(state, Mapping) or not bool(state.get("is_dynamic", True)):
            continue
        track_id = _optional_nonnegative_int(state.get("track_id"))
        position = _state_position(state)
        if track_id is None or position is None:
            continue
        if track_id in seen_track_ids:
            duplicate_track_ids.add(track_id)
            continue
        seen_track_ids.add(track_id)
        velocity = _state_velocity(state)
        predicted = tuple(
            position[index] + telemetry_age * velocity[index]
            for index in range(3)
        )
        tracks.append((track_id, predicted))
    if duplicate_track_ids:
        tracks = [value for value in tracks if value[0] not in duplicate_track_ids]

    if not tracks:
        for actor_id, result in results.items():
            result["binding_status"] = (
                "actor_position_unavailable"
                if actor_positions[actor_id] is None else "no_valid_tracks"
            )
        return results

    by_actor: dict[int, list[tuple[float, int]]] = {actor_id: [] for actor_id in actor_ids}
    by_track: dict[int, list[tuple[float, int]]] = {
        track_id: [] for track_id, _ in tracks
    }
    all_distances: dict[tuple[int, int], float] = {}
    for actor_id, actor_position in actor_positions.items():
        if actor_position is None:
            results[actor_id]["binding_status"] = "actor_position_unavailable"
            continue
        for track_id, track_position in tracks:
            distance = float(math.dist(actor_position, track_position))
            all_distances[(actor_id, track_id)] = distance
            if distance <= distance_threshold_m:
                by_actor[actor_id].append((distance, track_id))
                by_track[track_id].append((distance, actor_id))
        by_actor[actor_id].sort()
    for values in by_track.values():
        values.sort()

    for actor_id in actor_ids:
        result = results[actor_id]
        gated = by_actor[actor_id]
        result["candidate_track_ids"] = [track_id for _, track_id in gated]
        result["candidate_track_count"] = len(gated)
        if not gated:
            nearest = sorted(
                (distance, track_id)
                for (candidate_actor_id, track_id), distance in all_distances.items()
                if candidate_actor_id == actor_id
            )
            result["binding_status"] = "distance_gate_failed"
            if nearest:
                result["distance_m"] = nearest[0][0]
            continue

        best_distance, best_track_id = gated[0]
        result["matched_track_id"] = int(best_track_id)
        result["distance_m"] = float(best_distance)
        if len(gated) > 1:
            result["second_best_distance_m"] = float(gated[1][0])
            result["ambiguity_gap_m"] = float(gated[1][0] - best_distance)
            if gated[1][0] - best_distance < ambiguity_margin_m:
                result["binding_status"] = "ambiguous_actor_to_tracks"
                continue

        reverse = by_track[best_track_id]
        if not reverse or reverse[0][1] != actor_id:
            result["binding_status"] = "non_mutual_nearest"
            continue
        if len(reverse) > 1 and reverse[1][0] - reverse[0][0] < ambiguity_margin_m:
            result["binding_status"] = "ambiguous_track_to_actors"
            continue
        result["binding_status"] = "unique_match"

    return results


def normalize_simulator_visibility(
    actor: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the complete, JSON-safe Simulator visibility snapshot.

    Missing fields remain ``None``.  That is intentional: treating an absent
    field as ``False`` would incorrectly turn old or incomplete messages into
    evidence that an actor was outside the camera view.
    """

    return {
        "visible": _optional_bool(actor.get("visible")),
        "occluded": _optional_bool(actor.get("occluded")),
        "inside_image": _optional_bool(actor.get("inside_image")),
        "projected_u": _optional_finite_float(actor.get("projected_u")),
        "projected_v": _optional_finite_float(actor.get("projected_v")),
        "expected_surface_depth": _optional_finite_float(
            actor.get("expected_surface_depth")
        ),
        "observed_depth": _optional_finite_float(actor.get("observed_depth")),
        "depth_error": _optional_finite_float(actor.get("depth_error")),
        "rendered_pixel_count": _optional_nonnegative_int(
            actor.get("rendered_pixel_count")
        ),
    }


def _matching_track_summaries(
    actor_id: int,
    telemetry: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    """Return only tracks carrying an explicit Simulator identity binding."""

    matches = []
    for value in telemetry.get("dynamic_track_summaries") or ():
        if not isinstance(value, Mapping):
            continue
        bound_id = value.get("source_object_id", value.get("object_id"))
        try:
            bound_id = int(bound_id)
        except (TypeError, ValueError):
            continue
        if bound_id == int(actor_id) and bool(value.get("is_dynamic", True)):
            matches.append(value)
    return matches


def classify_dynamic_collision_observability(
    *,
    actor_id: int,
    simulator_visibility: Mapping[str, Any],
    planner_telemetry: Optional[Mapping[str, Any]] = None,
    planner_telemetry_age_s: Optional[float] = None,
    planner_telemetry_max_age_s: float = 0.5,
    posthoc_binding: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Classify one colliding actor using only recorded diagnostic evidence.

    The coarse ``classification`` is one of ``visible``, ``tracked``,
    ``unobservable`` or ``unknown``.  ``tracked`` requires either an explicit
    ``object_id``/``source_object_id`` binding or a unique, fresh, post-hoc
    spatial match produced by :func:`associate_gt_actors_to_tracks_posthoc`.
    An ambiguous spatial match is never described as tracked.
    """

    if planner_telemetry_max_age_s <= 0:
        raise ValueError("planner_telemetry_max_age_s must be positive")

    visibility = normalize_simulator_visibility(simulator_visibility)
    visible = visibility["visible"]
    rendered_pixels = visibility["rendered_pixel_count"]
    inside_image = visibility["inside_image"]
    occluded = visibility["occluded"]

    telemetry_present = isinstance(planner_telemetry, Mapping)
    telemetry_age = _optional_finite_float(planner_telemetry_age_s)
    telemetry_fresh = bool(
        telemetry_present
        and telemetry_age is not None
        and 0.0 <= telemetry_age <= planner_telemetry_max_age_s
    )
    telemetry = planner_telemetry if telemetry_present else {}
    matching_tracks = (
        _matching_track_summaries(actor_id, telemetry)
        if telemetry_fresh else []
    )
    posthoc = dict(posthoc_binding) if isinstance(posthoc_binding, Mapping) else None
    posthoc_unique = bool(
        telemetry_fresh
        and posthoc is not None
        and posthoc.get("binding_method") == POSTHOC_BINDING_METHOD
        and posthoc.get("binding_status") == "unique_match"
        and _optional_nonnegative_int(posthoc.get("actor_id")) == int(actor_id)
        and _optional_nonnegative_int(posthoc.get("matched_track_id")) is not None
    )
    identity_unique = len(matching_tracks) == 1
    predicted_track_count = _optional_nonnegative_int(
        telemetry.get("predicted_dynamic_track_count")
    ) if telemetry_fresh else None
    if predicted_track_count is None and telemetry_fresh:
        summaries = telemetry.get("dynamic_track_summaries")
        if isinstance(summaries, (list, tuple)):
            predicted_track_count = len(summaries)

    if visible is True:
        classification = "visible"
        reason = (
            "simulator_rendered_actor_at_collision"
            if rendered_pixels is not None and rendered_pixels > 0
            else "simulator_reported_actor_visible_at_collision"
        )
    elif identity_unique:
        classification = "tracked"
        reason = "fresh_identity_bound_dynamic_track_at_collision"
    elif posthoc_unique:
        classification = "tracked"
        reason = "fresh_unique_posthoc_spatial_track_at_collision"
    elif not telemetry_fresh:
        classification = "unknown"
        reason = (
            "planner_telemetry_missing"
            if not telemetry_present else "planner_telemetry_stale_or_unaged"
        )
    elif len(matching_tracks) > 1:
        classification = "unknown"
        reason = "explicit_identity_binding_ambiguous"
    elif predicted_track_count is not None and predicted_track_count > 0:
        classification = "unknown"
        binding_status = None if posthoc is None else posthoc.get("binding_status")
        if isinstance(binding_status, str) and binding_status.startswith("ambiguous_"):
            reason = "posthoc_spatial_binding_ambiguous"
        elif binding_status == "non_mutual_nearest":
            reason = "posthoc_spatial_binding_non_mutual"
        elif binding_status == "distance_gate_failed":
            reason = "no_posthoc_spatial_track_within_gate"
        elif posthoc is not None:
            reason = "posthoc_spatial_binding_unavailable"
        else:
            reason = "dynamic_tracks_present_without_actor_identity_binding"
    elif predicted_track_count == 0:
        classification = "unobservable"
        if inside_image is False:
            reason = "outside_camera_image_and_no_dynamic_track"
        elif occluded is True:
            reason = "camera_occluded_and_no_dynamic_track"
        elif inside_image is True and visible is False:
            reason = "inside_image_not_rendered_and_no_dynamic_track"
        else:
            reason = "not_visible_and_no_dynamic_track"
    else:
        classification = "unknown"
        reason = "fresh_telemetry_lacks_dynamic_track_count"

    if visible is True:
        camera_state = "visible"
    elif inside_image is False:
        camera_state = "outside_image"
    elif occluded is True:
        camera_state = "occluded"
    elif inside_image is True and visible is False:
        camera_state = "inside_image_not_rendered"
    else:
        camera_state = "unknown"

    return {
        "contract_version": CONTRACT_VERSION,
        "actor_id": int(actor_id),
        "classification": classification,
        "reason": reason,
        "camera_state": camera_state,
        "simulator_visibility": visibility,
        "planner_telemetry_present": telemetry_present,
        "planner_telemetry_fresh": telemetry_fresh,
        "planner_telemetry_age_s": telemetry_age,
        "planner_telemetry_max_age_s": float(planner_telemetry_max_age_s),
        "dynamic_context_valid": (
            bool(telemetry.get("dynamic_context_valid"))
            if telemetry_fresh and "dynamic_context_valid" in telemetry
            else None
        ),
        "predicted_dynamic_track_count": predicted_track_count,
        "identity_bound_track_ids": [
            int(track["track_id"])
            for track in matching_tracks
            if _optional_nonnegative_int(track.get("track_id")) is not None
        ],
        "identity_binding_available": identity_unique,
        "posthoc_binding": posthoc,
        "posthoc_binding_available": posthoc_unique,
        "actor_track_binding_available": bool(identity_unique or posthoc_unique),
        "binding_method": (
            "explicit_identity_binding" if identity_unique else
            POSTHOC_BINDING_METHOD if posthoc_unique else None
        ),
    }


__all__ = [
    "CONTRACT_VERSION",
    "POSTHOC_BINDING_METHOD",
    "SIMULATOR_VISIBILITY_FIELDS",
    "associate_gt_actors_to_tracks_posthoc",
    "classify_dynamic_collision_observability",
    "normalize_simulator_visibility",
]
