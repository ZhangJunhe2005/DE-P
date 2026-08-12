from types import SimpleNamespace

import torch

from policy.deadlock_recovery_v3 import DeadlockRecoveryV3
from policy.dynamic.context import DynamicContext
from policy.runtime_profile_v4_7 import runtime_safety_mapping_v4_7
from policy.runtime_safety_v1 import (
    CandidateSafetyV1,
    RuntimeSafetyConfigV1,
    RuntimeTrajectorySafetyV1,
)


def _evaluation(clearance):
    return CandidateSafetyV1(
        feasible=True,
        reasons=(),
        max_speed_mps=3.0,
        max_acceleration_mps2=2.0,
        min_observed_clearance_m=clearance,
        endpoint_progress_m=5.0,
    )


def test_balanced_profile_prefers_small_safe_detour_but_not_large_detour():
    config = RuntimeSafetyConfigV1.from_mapping(
        runtime_safety_mapping_v4_7({})
    )
    shield = RuntimeTrajectorySafetyV1(config)
    close, clear = _evaluation(0.36), _evaluation(1.25)
    assert shield.select([0.0, 0.04], (close, clear)).action_id == 1
    assert shield.select([0.0, 0.20], (close, clear)).action_id == 0


def test_recovery_can_disable_clearance_preference_without_relaxing_safety():
    config = RuntimeSafetyConfigV1.from_mapping(
        runtime_safety_mapping_v4_7({})
    )
    shield = RuntimeTrajectorySafetyV1(config)
    close, clear = _evaluation(0.36), _evaluation(1.25)

    # Normal flight retains the small bounded clearance tie-breaker.
    normal = shield.select([0.0, 0.04], (close, clear))
    assert normal.action_id == 1
    assert normal.mode == "network_safe_clearance_preference"

    # Bounded scan handoff uses the unmodified learned score, while the same
    # feasibility mask remains authoritative.
    recovery = shield.select(
        [0.0, 0.04], (close, clear), apply_clearance_preference=False,
    )
    assert recovery.action_id == 0
    assert recovery.mode == "network_safe"

    rejected = CandidateSafetyV1(
        feasible=False,
        reasons=("collision_floor",),
        max_speed_mps=3.0,
        max_acceleration_mps2=2.0,
        min_observed_clearance_m=0.05,
        endpoint_progress_m=5.0,
    )
    still_safe = shield.select(
        [-100.0, 0.04], (rejected, clear),
        apply_clearance_preference=False,
    )
    assert still_safe.action_id == 1


def test_balanced_clearance_preference_is_bounded_and_saturates():
    config = RuntimeSafetyConfigV1.from_mapping(
        runtime_safety_mapping_v4_7({})
    )
    assert config.clearance_preference_weight == 0.15
    assert config.clearance_preference_saturation_m == 0.45
    shield = RuntimeTrajectorySafetyV1(config)
    # Above the saturation point, a farther detour receives no additional
    # preference; the learned score remains authoritative.
    saturated = _evaluation(config.collision_floor_m + 0.45)
    much_farther = _evaluation(config.collision_floor_m + 2.0)
    assert shield.select([0.01, 0.0], (saturated, much_farther)).action_id == 1
    # The maximum 0.15 preference cannot overturn a material score gap.
    close = _evaluation(config.collision_floor_m + 0.02)
    assert shield.select([0.0, 0.16], (close, much_farther)).action_id == 0


def test_balanced_profile_keeps_existing_physical_and_no_heuristic_contract():
    config = RuntimeSafetyConfigV1.from_mapping(
        runtime_safety_mapping_v4_7({})
    )
    assert config.collision_floor_m == 0.35
    assert config.dynamic_track_radius_m == 0.50
    assert not config.enforce_camera_visibility
    assert not config.enforce_minimum_progress
    assert not config.enforce_goal_progress


def test_collision_shield_tracks_are_not_silently_limited_by_attention_gate():
    safety_track = SimpleNamespace(
        track_id=3,
        position_world=(1.0, 2.0, 3.0),
        velocity_world=(0.5, 0.0, 0.0),
        is_dynamic=True,
        timestamp=4.0,
        ever_directly_observed=True,
        visibility_state="occluded",
        attention_authorized=False,
    )
    result = SimpleNamespace(
        attention_map=torch.zeros(1, 1, 3, 5),
        confirmed_tracks=(safety_track,),
        dynamic_tracks=(),
        diagnostics={},
    )
    context = DynamicContext.from_perception_result(result, 4.0, "depth")
    assert len(context.dynamic_tracks) == 1
    assert context.dynamic_tracks[0].track_id == 3


def test_clear_missing_track_is_not_forwarded_to_collision_shield():
    missing = SimpleNamespace(
        track_id=7,
        position_world=(1.0, 2.0, 3.0),
        velocity_world=(0.5, 0.0, 0.0),
        is_dynamic=True,
        timestamp=4.0,
        ever_directly_observed=True,
        visibility_state="clear_missing",
        attention_authorized=False,
    )
    result = SimpleNamespace(
        attention_map=torch.zeros(1, 1, 3, 5),
        confirmed_tracks=(missing,),
        dynamic_tracks=(),
        diagnostics={},
    )
    context = DynamicContext.from_perception_result(result, 4.0, "depth")
    assert context.dynamic_tracks == ()


def test_previously_dynamic_occluded_track_coasts_for_safety_not_attention():
    coast = SimpleNamespace(
        track_id=9,
        position_world=(1.0, 2.0, 3.0),
        velocity_world=(0.5, 0.0, 0.0),
        is_dynamic=False,
        ever_confirmed_dynamic=True,
        timestamp=4.1,
        last_direct_observation_timestamp=4.0,
        prediction_only_age=2,
        ever_directly_observed=True,
        visibility_state="occluded",
        confidence=0.40,
        state_covariance=torch.eye(6).numpy() * 0.01,
        last_observed_extent=(0.6, 0.6, 1.4),
    )
    result = SimpleNamespace(
        attention_map=torch.zeros(1, 1, 3, 5),
        confirmed_tracks=(coast,),
        dynamic_tracks=(),
        diagnostics={},
    )
    context = DynamicContext.from_perception_result(result, 4.1, "depth")
    assert len(context.dynamic_tracks) == 1
    summary = context.dynamic_tracks[0]
    assert summary.is_dynamic
    assert summary.last_direct_observation_timestamp == 4.0
    assert summary.prediction_only_age == 2
    assert summary.observed_extent == (0.6, 0.6, 1.4)


def test_prediction_only_track_expires_from_last_measurement_not_state_time():
    config = RuntimeSafetyConfigV1.from_mapping(
        runtime_safety_mapping_v4_7({})
    )
    shield = RuntimeTrajectorySafetyV1(config)
    from tests.test_runtime_safety_v1 import straight
    forward = straight([3.4, 0.0, 0.0], [2.0, 0.0, 0.0])
    stale = SimpleNamespace(
        is_dynamic=True,
        timestamp=10.0,
        last_direct_observation_timestamp=9.0,
        position_world=(1.7, 0.0, 0.0),
        velocity_world=(0.0, 0.0, 0.0),
        state_covariance=(),
        observed_extent=(0.0, 0.0, 0.0),
    )
    evaluation = shield.evaluate(
        [forward], 1.7, torch.empty(0, 3).numpy(),
        torch.zeros(3).numpy(), torch.eye(3).numpy(),
        dynamic_tracks=(stale,), query_timestamp=10.0,
    )[0]
    assert evaluation.feasible
    assert evaluation.min_predicted_dynamic_clearance_m is None


def test_v4_7_dynamic_occupancy_uses_covariance_and_short_horizon():
    from tests.test_runtime_safety_v1 import straight

    config = RuntimeSafetyConfigV1.from_mapping(
        runtime_safety_mapping_v4_7({})
    )
    assert config.dynamic_track_prediction_horizon_s == 1.50
    assert config.dynamic_track_covariance_sigma == 2.0
    shield = RuntimeTrajectorySafetyV1(config)
    forward = straight([3.4, 0.0, 0.0], [2.0, 0.0, 0.0])
    covariance = torch.eye(6).numpy() * 0.04
    track = SimpleNamespace(
        is_dynamic=True,
        timestamp=10.0,
        last_direct_observation_timestamp=10.0,
        position_world=(1.7, 1.55, 0.0),
        velocity_world=(0.0, 0.0, 0.0),
        state_covariance=covariance,
        observed_extent=(0.6, 0.6, 1.4),
    )
    evaluation = shield.evaluate(
        [forward], 1.7, torch.empty(0, 3).numpy(),
        torch.zeros(3).numpy(), torch.eye(3).numpy(),
        dynamic_tracks=(track,), query_timestamp=10.0,
    )[0]
    assert not evaluation.feasible
    assert "predicted_dynamic_clearance" in evaluation.reasons


def test_repeated_failed_scan_expands_and_alternates():
    from tests.test_deadlock_recovery_v3 import (
        enter_scan,
        observe,
        recovery_config,
    )

    recovery = DeadlockRecoveryV3(recovery_config(cooldown_s=0.1))
    enter_scan(recovery)
    first_direction = recovery.scan_direction
    assert recovery.current_scan_limit_deg == 60.0
    while not recovery.scan_leg_complete:
        recovery.yaw_command(0.0, 0.05)
    observe(recovery, 1.60)
    for index in range(15):
        decision = observe(
            recovery, 1.71 + 0.03 * index, speed=0.4
        )
    assert decision.transition == "network_to_braking"
    decision = observe(recovery, 2.18, speed=0.2)
    assert decision.transition == "braking_to_bounded_scan"
    assert recovery.current_scan_limit_deg == 90.0
    assert recovery.scan_direction == -first_direction


def test_v4_7_recovery_uses_short_bounded_confirmation():
    from policy.deadlock_recovery_v3 import (
        DeadlockRecoveryConfigV3,
        deadlock_recovery_mapping_v4_7_pillar,
    )

    config = DeadlockRecoveryConfigV3.from_mapping(
        deadlock_recovery_mapping_v4_7_pillar({"enabled": True})
    )
    assert config.selected_release_replans == 3
    assert config.max_scan_angle_deg == 60.0
    assert config.max_escalated_scan_angle_deg == 120.0


def test_v4_7_wrapper_selects_validated_range_mode_and_balanced_profile():
    from pathlib import Path

    text = Path("scripts/route_a_v4_7_rviz_host.sh").read_text()
    assert "--runtime-profile v4_7_balanced_dynamic" in text
    assert "--dynamic-foreground-mode range_image_hybrid" in text
