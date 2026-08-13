from __future__ import annotations

import math
from pathlib import Path

import pytest

from policy.dynamic.collision_observability_v1 import (
    CONTRACT_VERSION,
    POSTHOC_BINDING_METHOD,
    SIMULATOR_VISIBILITY_FIELDS,
    associate_gt_actors_to_tracks_posthoc,
    classify_dynamic_collision_observability,
    normalize_simulator_visibility,
)


ROOT = Path(__file__).resolve().parents[1]


def visibility(**overrides):
    value = {
        "visible": False,
        "occluded": False,
        "inside_image": False,
        "projected_u": 81.5,
        "projected_v": 42.25,
        "expected_surface_depth": 2.4,
        "observed_depth": 2.4,
        "depth_error": 0.0,
        "rendered_pixel_count": 0,
    }
    value.update(overrides)
    return value


def test_normalization_preserves_all_simulator_visibility_fields_json_safely():
    result = normalize_simulator_visibility(visibility(
        projected_u=math.nan,
        observed_depth=math.inf,
        rendered_pixel_count=19,
    ))
    assert tuple(result) == SIMULATOR_VISIBILITY_FIELDS
    assert result["projected_u"] is None
    assert result["observed_depth"] is None
    assert result["rendered_pixel_count"] == 19
    assert result["inside_image"] is False


def test_visible_actor_is_visible_even_without_planner_telemetry():
    result = classify_dynamic_collision_observability(
        actor_id=7,
        simulator_visibility=visibility(
            visible=True, inside_image=True, rendered_pixel_count=31,
        ),
    )
    assert result["contract_version"] == CONTRACT_VERSION
    assert result["classification"] == "visible"
    assert result["camera_state"] == "visible"
    assert result["reason"] == "simulator_rendered_actor_at_collision"
    assert result["planner_telemetry_fresh"] is False


def test_nonvisible_actor_requires_explicit_identity_binding_to_be_tracked():
    result = classify_dynamic_collision_observability(
        actor_id=7,
        simulator_visibility=visibility(
            inside_image=True, occluded=True,
        ),
        planner_telemetry={
            "dynamic_context_valid": True,
            "predicted_dynamic_track_count": 1,
            "dynamic_track_summaries": [{
                "track_id": 12,
                "source_object_id": 7,
                "is_dynamic": True,
            }],
        },
        planner_telemetry_age_s=0.03,
    )
    assert result["classification"] == "tracked"
    assert result["camera_state"] == "occluded"
    assert result["identity_bound_track_ids"] == [12]


@pytest.mark.parametrize(
    ("actor_visibility", "reason"),
    (
        (visibility(inside_image=False),
         "outside_camera_image_and_no_dynamic_track"),
        (visibility(inside_image=True, occluded=True),
         "camera_occluded_and_no_dynamic_track"),
        (visibility(inside_image=True, visible=False, occluded=False),
         "inside_image_not_rendered_and_no_dynamic_track"),
    ),
)
def test_fresh_zero_track_telemetry_proves_unobservable(actor_visibility, reason):
    result = classify_dynamic_collision_observability(
        actor_id=3,
        simulator_visibility=actor_visibility,
        planner_telemetry={
            "dynamic_context_valid": True,
            "predicted_dynamic_track_count": 0,
            "dynamic_track_summaries": [],
        },
        planner_telemetry_age_s=0.1,
    )
    assert result["classification"] == "unobservable"
    assert result["reason"] == reason


def test_unassociated_tracks_are_unknown_not_misreported_as_actor_track():
    result = classify_dynamic_collision_observability(
        actor_id=7,
        simulator_visibility=visibility(inside_image=False),
        planner_telemetry={
            "dynamic_context_valid": True,
            "predicted_dynamic_track_count": 2,
            # V4.8.x exposes track_id but no Simulator object identity.
            "dynamic_track_summaries": [
                {"track_id": 1, "is_dynamic": True},
                {"track_id": 2, "is_dynamic": True},
            ],
        },
        planner_telemetry_age_s=0.01,
    )
    assert result["classification"] == "unknown"
    assert result["identity_binding_available"] is False
    assert result["reason"] == (
        "dynamic_tracks_present_without_actor_identity_binding"
    )


def test_unique_fresh_posthoc_binding_can_prove_actor_was_tracked():
    telemetry = {
        "dynamic_context_valid": True,
        "predicted_dynamic_track_count": 1,
        "dynamic_track_summaries": [{
            "track_id": 12,
            "is_dynamic": True,
            "position_world": [1.0, 2.0, 3.0],
            "velocity_world": [1.0, 0.0, 0.0],
        }],
    }
    bindings = associate_gt_actors_to_tracks_posthoc(
        actor_states={7: {"position": [1.2, 2.0, 3.0]}},
        planner_telemetry=telemetry,
        planner_telemetry_age_s=0.2,
        time_threshold_s=0.5,
        distance_threshold_m=0.5,
        ambiguity_margin_m=0.1,
    )
    binding = bindings[7]
    assert binding["binding_method"] == POSTHOC_BINDING_METHOD
    assert binding["binding_status"] == "unique_match"
    assert binding["matched_track_id"] == 12
    assert binding["distance_m"] == pytest.approx(0.0)
    assert binding["diagnostic_only"] is True
    assert binding["planner_control_affected"] is False

    result = classify_dynamic_collision_observability(
        actor_id=7,
        simulator_visibility=visibility(inside_image=False),
        planner_telemetry=telemetry,
        planner_telemetry_age_s=0.2,
        posthoc_binding=binding,
    )
    assert result["classification"] == "tracked"
    assert result["binding_method"] == POSTHOC_BINDING_METHOD
    assert result["posthoc_binding_available"] is True
    assert result["identity_binding_available"] is False
    assert result["reason"] == (
        "fresh_unique_posthoc_spatial_track_at_collision"
    )


def test_posthoc_distance_gate_failure_remains_unknown_when_tracks_exist():
    telemetry = {
        "predicted_dynamic_track_count": 1,
        "dynamic_track_summaries": [{
            "track_id": 4,
            "position_world": [9.0, 0.0, 0.0],
            "velocity_world": [0.0, 0.0, 0.0],
        }],
    }
    binding = associate_gt_actors_to_tracks_posthoc(
        actor_states={2: {"position": [0.0, 0.0, 0.0]}},
        planner_telemetry=telemetry,
        planner_telemetry_age_s=0.1,
        distance_threshold_m=1.0,
    )[2]
    assert binding["binding_status"] == "distance_gate_failed"
    assert binding["distance_m"] == pytest.approx(9.0)

    result = classify_dynamic_collision_observability(
        actor_id=2,
        simulator_visibility=visibility(inside_image=False),
        planner_telemetry=telemetry,
        planner_telemetry_age_s=0.1,
        posthoc_binding=binding,
    )
    assert result["classification"] == "unknown"
    assert result["reason"] == "no_posthoc_spatial_track_within_gate"


def test_two_nearby_tracks_are_ambiguous_and_never_called_tracked():
    telemetry = {
        "predicted_dynamic_track_count": 2,
        "dynamic_track_summaries": [
            {"track_id": 1, "position_world": [0.10, 0.0, 0.0]},
            {"track_id": 2, "position_world": [0.18, 0.0, 0.0]},
        ],
    }
    binding = associate_gt_actors_to_tracks_posthoc(
        actor_states={9: {"position": [0.0, 0.0, 0.0]}},
        planner_telemetry=telemetry,
        planner_telemetry_age_s=0.02,
        distance_threshold_m=1.0,
        ambiguity_margin_m=0.25,
    )[9]
    assert binding["binding_status"] == "ambiguous_actor_to_tracks"
    assert binding["candidate_track_ids"] == [1, 2]
    assert binding["ambiguity_gap_m"] == pytest.approx(0.08)

    result = classify_dynamic_collision_observability(
        actor_id=9,
        simulator_visibility=visibility(inside_image=False),
        planner_telemetry=telemetry,
        planner_telemetry_age_s=0.02,
        posthoc_binding=binding,
    )
    assert result["classification"] == "unknown"
    assert result["reason"] == "posthoc_spatial_binding_ambiguous"


def test_bidirectional_ambiguity_prevents_one_track_binding_two_actors():
    bindings = associate_gt_actors_to_tracks_posthoc(
        actor_states={
            3: {"position": [0.0, 0.0, 0.0]},
            4: {"position": [0.1, 0.0, 0.0]},
        },
        planner_telemetry={
            "dynamic_track_summaries": [{
                "track_id": 8,
                "position_world": [0.04, 0.0, 0.0],
            }],
        },
        planner_telemetry_age_s=0.01,
        distance_threshold_m=1.0,
        ambiguity_margin_m=0.25,
    )
    assert bindings[3]["binding_status"] == "ambiguous_track_to_actors"
    assert bindings[4]["binding_status"] == "non_mutual_nearest"
    assert not any(
        binding["binding_status"] == "unique_match"
        for binding in bindings.values()
    )


def test_posthoc_time_gate_is_explicit_and_blocks_stale_association():
    binding = associate_gt_actors_to_tracks_posthoc(
        actor_states={1: {"position": [0.0, 0.0, 0.0]}},
        planner_telemetry={
            "dynamic_track_summaries": [{
                "track_id": 2,
                "position_world": [0.0, 0.0, 0.0],
            }],
        },
        planner_telemetry_age_s=0.51,
        time_threshold_s=0.5,
    )[1]
    assert binding["binding_status"] == "time_gate_failed"
    assert binding["time_offset_s"] == pytest.approx(0.51)
    assert binding["time_threshold_s"] == pytest.approx(0.5)


def test_stale_telemetry_is_unknown_not_unobservable():
    result = classify_dynamic_collision_observability(
        actor_id=7,
        simulator_visibility=visibility(inside_image=False),
        planner_telemetry={"predicted_dynamic_track_count": 0},
        planner_telemetry_age_s=0.51,
        planner_telemetry_max_age_s=0.5,
    )
    assert result["classification"] == "unknown"
    assert result["reason"] == "planner_telemetry_stale_or_unaged"


def test_invalid_maximum_telemetry_age_is_rejected():
    with pytest.raises(ValueError, match="must be positive"):
        classify_dynamic_collision_observability(
            actor_id=1,
            simulator_visibility=visibility(),
            planner_telemetry_max_age_s=0.0,
        )


def test_interactive_monitor_wires_visibility_and_observability_diagnostics():
    source = (
        ROOT / "tools/monitor_dep_interactive_collisions.py"
    ).read_text(encoding="utf-8")
    for field in SIMULATOR_VISIBILITY_FIELDS:
        assert f'"{field}"' in source
    assert '"/dep_net/safety_decision"' in source
    assert "classify_dynamic_collision_observability(" in source
    assert '"dynamic_collision_observability"' in source
    assert '"dynamic_collision_observability_counts"' in source
    assert "associate_gt_actors_to_tracks_posthoc(" in source
    assert '"posthoc_binding_contract"' in source
    assert '"planner_control_affected": False' in source
