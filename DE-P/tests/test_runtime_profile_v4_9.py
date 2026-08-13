from types import SimpleNamespace

import numpy as np

from policy.deadlock_recovery_v3 import DeadlockRecoveryConfigV3
from policy.poly_solver import Poly5Solver
from policy.runtime_profile_v4_8_5 import (
    deadlock_recovery_mapping_v4_8_5,
)
from policy.runtime_profile_v4_9 import (
    RUNTIME_BEHAVIOR_VERSION,
    deadlock_recovery_mapping_v4_9,
    runtime_safety_mapping_v4_9,
)
from policy.runtime_safety_v1 import (
    CandidateSafetyV1,
    RuntimeSafetyConfigV1,
    RuntimeTrajectorySafetyV1,
    classify_dynamic_blocking_v1,
    dynamic_command_is_fresh_v1,
    select_dynamic_braking_option_v1,
)


def minimal_v49(**overrides):
    values = runtime_safety_mapping_v4_9({})
    values.update({
        "feasibility_projection_enabled": False,
        "enforce_camera_visibility": False,
        "enforce_minimum_progress": False,
        "enforce_goal_progress": False,
        "enforce_observed_tracking_clearance": False,
        "enforce_stopping_distance": False,
        "enforce_preferred_flight_volume_clearance": False,
        "dynamic_track_radius_m": 0.20,
        "dynamic_track_prediction_margin_m": 0.0,
        "dynamic_track_uncertainty_growth_mps": 0.0,
        "dynamic_track_covariance_sigma": 0.0,
        "dynamic_track_prediction_horizon_s": 3.0,
        "dynamic_track_max_age_s": 1.0,
    })
    values.update(overrides)
    return RuntimeSafetyConfigV1.from_mapping(values)


def candidate(duration=1.7):
    return tuple(
        Poly5Solver(
            0.0, 0.0, 0.0,
            3.0 if axis == 0 else 0.0,
            0.0, 0.0, duration,
        )
        for axis in range(3)
    )


def track(position, velocity, covariance=None, track_id=7):
    value = dict(
        is_dynamic=True,
        timestamp=10.0,
        last_direct_observation_timestamp=10.0,
        position_world=position,
        velocity_world=velocity,
        observed_extent=(0.0, 0.0, 0.0),
        track_id=track_id,
    )
    if covariance is not None:
        value["state_covariance"] = covariance
    return SimpleNamespace(**value)


def evaluate(shield, trajectory, obstacle=(), actor=None, duration=1.7):
    return shield.evaluate(
        (trajectory,), (duration,), np.asarray(obstacle).reshape(-1, 3),
        np.zeros(3), np.eye(3),
        dynamic_tracks=(() if actor is None else (actor,)),
        query_timestamp=10.0,
    )[0]


def test_v49_preserves_v485_recovery_and_enables_only_dynamic_extensions():
    assert deadlock_recovery_mapping_v4_9({"enabled": True}) == (
        deadlock_recovery_mapping_v4_8_5({"enabled": True})
    )
    DeadlockRecoveryConfigV3.from_mapping(
        deadlock_recovery_mapping_v4_9({"enabled": True})
    )
    config = RuntimeSafetyConfigV1.from_mapping(
        runtime_safety_mapping_v4_9({})
    )
    assert config.dynamic_risk_ranking_enabled
    assert config.dynamic_risk_score_weight == 0.20
    assert config.dynamic_time_retiming_enabled
    assert tuple(config.dynamic_time_retiming_scales) == (1.0, 1.2, 1.4)
    assert config.dynamic_command_freshness_watchdog_enabled
    assert config.dynamic_command_max_age_s == 0.20
    assert RUNTIME_BEHAVIOR_VERSION == (
        "v4_9_bounded_risk_ordered_retiming_fresh_prefix_v1"
    )


def test_approaching_track_has_ttc_and_more_risk_than_receding_track():
    shield = RuntimeTrajectorySafetyV1(minimal_v49())
    trajectory = candidate()
    approaching = evaluate(
        shield, trajectory,
        actor=track((4.0, 0.8, 0.0), (-0.6, 0.0, 0.0)),
    )
    receding = evaluate(
        shield, trajectory,
        actor=track((4.0, 0.8, 0.0), (0.6, 0.0, 0.0)),
    )
    assert approaching.minimum_dynamic_ttc_s is not None
    assert approaching.dynamic_closing_speed_at_closest_mps > 0.0
    assert approaching.dynamic_risk_cost > receding.dynamic_risk_cost
    assert 0.0 <= approaching.dynamic_risk_cost <= 1.0


def test_dynamic_risk_changes_only_feasible_ranking_and_is_bounded():
    shield = RuntimeTrajectorySafetyV1(minimal_v49())
    low_risk = CandidateSafetyV1(
        True, (), 1.0, 1.0, None, 2.0, dynamic_risk_cost=0.0,
    )
    high_risk = CandidateSafetyV1(
        True, (), 1.0, 1.0, None, 2.0, dynamic_risk_cost=1.0,
    )
    rejected = CandidateSafetyV1(
        False, ("predicted_dynamic_clearance",), 1.0, 1.0, None,
        2.0, dynamic_risk_cost=0.0,
    )
    assert shield.select(
        [0.10, 0.0, -100.0], (low_risk, high_risk, rejected)
    ).action_id == 0
    # The full bounded 0.20 risk term cannot overturn a much better network
    # score, which protects the frozen V4.8.3 static navigation behaviour.
    assert shield.select(
        [0.50, 0.0], (low_risk, high_risk)
    ).action_id == 1


def test_more_track_uncertainty_never_reduces_dynamic_risk():
    shield = RuntimeTrajectorySafetyV1(minimal_v49(
        dynamic_track_covariance_sigma=2.0,
    ))
    trajectory = candidate()
    low = evaluate(
        shield, trajectory,
        actor=track(
            (4.5, 1.2, 0.0), (-0.4, 0.0, 0.0),
            covariance=np.eye(6) * 1.0e-4,
        ),
    )
    high = evaluate(
        shield, trajectory,
        actor=track(
            (4.5, 1.2, 0.0), (-0.4, 0.0, 0.0),
            covariance=np.eye(6) * 0.04,
        ),
    )
    assert high.min_predicted_dynamic_clearance_m <= (
        low.min_predicted_dynamic_clearance_m
    )
    assert high.dynamic_risk_cost >= low.dynamic_risk_cost


def test_dynamic_time_retiming_preserves_state_and_cannot_wash_static_reason():
    shield = RuntimeTrajectorySafetyV1(minimal_v49())
    trajectory = candidate()
    actor = track((1.5, 0.5, 0.0), (0.0, -1.2, 0.0))
    base = evaluate(shield, trajectory, actor=actor)
    assert base.reasons == ("predicted_dynamic_clearance",)
    rebuilt, durations, scales, eligible = (
        shield.retime_dynamic_only_candidates(
            (trajectory,), (1.7,), (base,), 1.2
        )
    )
    assert eligible == (True,)
    assert scales == (1.2,)
    np.testing.assert_allclose(durations, (2.04,))
    np.testing.assert_allclose(
        [axis.get_position(0.0) for axis in rebuilt[0]], np.zeros(3)
    )
    np.testing.assert_allclose(
        [axis.get_position(durations[0]) for axis in rebuilt[0]],
        [3.0, 0.0, 0.0], atol=1e-9,
    )
    assert evaluate(
        shield, rebuilt[0], actor=actor, duration=durations[0]
    ).feasible

    static_failure = CandidateSafetyV1(
        False, ("observed_collision_floor",), 1.0, 1.0, 0.0, 3.0,
    )
    unchanged, unchanged_duration, unchanged_scale, allowed = (
        shield.retime_dynamic_only_candidates(
            (trajectory,), (1.7,), (static_failure,), 1.2
        )
    )
    assert unchanged[0] is trajectory
    assert unchanged_duration == (1.7,)
    assert unchanged_scale == (1.0,)
    assert allowed == (False,)


def test_global_search_contract_reaches_14_only_after_12_fails():
    shield = RuntimeTrajectorySafetyV1(minimal_v49())
    trajectory = candidate()
    actor = track((1.5, 0.1, 0.0), (0.0, -0.55, 0.0))
    base = evaluate(shield, trajectory, actor=actor)
    assert classify_dynamic_blocking_v1((base,))["cause"] == "dynamic_only"
    outcomes = []
    for scale in (1.2, 1.4):
        pool, durations, _, _ = shield.retime_dynamic_only_candidates(
            (trajectory,), (1.7,), (base,), scale
        )
        outcomes.append(evaluate(
            shield, pool[0], actor=actor, duration=durations[0]
        ).feasible)
    assert outcomes == [False, True]


def test_dynamic_blocking_classification_does_not_add_a_gate():
    dynamic = CandidateSafetyV1(
        False, ("predicted_dynamic_clearance",), 1.0, 1.0, None, 2.0,
    )
    mixed = CandidateSafetyV1(
        False, ("predicted_dynamic_clearance", "flight_volume"),
        1.0, 1.0, None, 2.0,
    )
    static = CandidateSafetyV1(
        False, ("observed_collision_floor",), 1.0, 1.0, 0.0, 2.0,
    )
    assert classify_dynamic_blocking_v1((dynamic,))["cause"] == "dynamic_only"
    assert classify_dynamic_blocking_v1((mixed,))["cause"] == "mixed"
    assert classify_dynamic_blocking_v1((static,))["cause"] == (
        "non_dynamic_only"
    )


def test_long_candidate_is_only_a_rolling_certified_prefix():
    shield = RuntimeTrajectorySafetyV1(minimal_v49(
        dynamic_track_prediction_horizon_s=1.5,
    ))
    trajectory = candidate(duration=4.67)
    result = evaluate(
        shield, trajectory, duration=4.67,
        actor=track((20.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    )
    assert result.dynamic_certified_horizon_s == 1.5
    assert result.dynamic_full_duration_certified is False
    assert dynamic_command_is_fresh_v1(10.0, 10.19, 0.20)
    assert not dynamic_command_is_fresh_v1(10.0, 10.21, 0.20)


def test_collision_beyond_prefix_is_not_reported_as_full_actor_clearance():
    shield = RuntimeTrajectorySafetyV1(minimal_v49(
        dynamic_track_prediction_horizon_s=1.5,
    ))
    trajectory = candidate(duration=4.67)
    # The vehicle reaches x=3 only at 4.67 s.  This actor lies outside the
    # trustworthy 1.5 s constant-velocity prefix, so the candidate is not
    # falsely described as fully certified even when the prefix is clear.
    result = evaluate(
        shield, trajectory, duration=4.67,
        actor=track((2.8, 0.0, 0.0), (0.0, 0.0, 0.0)),
    )
    assert result.dynamic_full_duration_certified is False
    assert result.dynamic_certified_horizon_s == 1.5


def test_dynamic_braking_selection_prefers_certified_then_minimum_risk():
    collision_soon = CandidateSafetyV1(
        False, ("predicted_dynamic_clearance",), 1.0, 1.0, None, 0.5,
        min_predicted_dynamic_clearance_m=-0.4,
        predicted_dynamic_collision_ttc_s=0.3,
        dynamic_risk_cost=1.0,
    )
    collision_later = CandidateSafetyV1(
        False, ("predicted_dynamic_clearance",), 1.0, 1.0, None, 0.8,
        min_predicted_dynamic_clearance_m=-0.1,
        predicted_dynamic_collision_ttc_s=0.8,
        dynamic_risk_cost=0.6,
    )
    safe = CandidateSafetyV1(True, (), 1.0, 1.0, None, 0.4)
    assert select_dynamic_braking_option_v1(
        (collision_soon, collision_later, safe)
    ) == (2, True)
    assert select_dynamic_braking_option_v1(
        (collision_soon, collision_later)
    ) == (1, False)
    static_failure = CandidateSafetyV1(
        False, ("observed_collision_floor",), 1.0, 1.0, 0.0, 0.1,
    )
    assert select_dynamic_braking_option_v1((static_failure,)) == (
        None, False
    )
