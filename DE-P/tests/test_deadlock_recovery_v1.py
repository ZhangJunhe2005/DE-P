import numpy as np

from policy.deadlock_recovery_v1 import (
    DeadlockRecoveryConfigV1,
    DeadlockRecoveryV1,
)
from policy.runtime_safety_v1 import (
    RuntimeSafetyConfigV1,
    RuntimeTrajectorySafetyV1,
    _sample_polynomials,
)


def config(**updates):
    values = {
        "zero_feasible_trigger_replans": 3,
        "collision_floor_trigger_replans": 2,
        "release_feasible_replans": 2,
        "stationary_speed_mps": 0.35,
        "breadcrumb_spacing_m": 0.1,
        "retreat_distance_m": 0.8,
        "retreat_arrival_radius_m": 0.15,
    }
    values.update(updates)
    return DeadlockRecoveryConfigV1(**values)


def test_clear_deadlock_brakes_scans_and_requires_stable_release():
    recovery = DeadlockRecoveryV1(config())
    depth = np.ones((8, 12), dtype=np.float32)
    position = np.zeros(3)
    for _ in range(2):
        decision = recovery.observe(0, False, 2.0, position, depth)
        assert decision.mode == recovery.NORMAL
    decision = recovery.observe(0, False, 2.0, position, depth)
    assert decision.transition == "network_to_braking"
    decision = recovery.observe(0, False, 0.1, position, depth)
    assert decision.transition == "braking_to_yaw_scan"
    assert recovery.observe(2, False, 0.0, position, depth).mode == recovery.YAW_SCAN
    decision = recovery.observe(2, False, 0.0, position, depth)
    assert decision.transition == "yaw_scan_to_network"


def test_collision_floor_uses_recent_breadcrumb_before_yaw_scan():
    recovery = DeadlockRecoveryV1(config())
    for x in np.linspace(0.0, 2.0, 21):
        recovery.record_position([x, 0.0, 1.0])
    depth = np.ones((8, 12), dtype=np.float32)
    current = np.asarray([2.0, 0.0, 1.0])
    recovery.observe(0, True, 2.0, current, depth)
    recovery.observe(0, True, 2.0, current, depth)
    decision = recovery.observe(0, True, 2.0, current, depth)
    assert decision.transition == "network_to_braking"
    decision = recovery.observe(0, True, 0.1, current, depth)
    assert decision.transition == "braking_to_breadcrumb_retreat"
    target = np.asarray(decision.retreat_target_world)
    assert target[0] <= 1.2 + 1e-9
    decision = recovery.observe(0, False, 0.1, target, depth)
    assert decision.transition == "breadcrumb_retreat_to_yaw_scan"


def test_scan_direction_chooses_freer_image_half():
    left_free = np.concatenate((
        np.full((6, 5), 10.0), np.full((6, 5), 1.0)
    ), axis=1)
    right_free = left_free[:, ::-1]
    assert DeadlockRecoveryV1.scan_direction_from_depth(left_free) == 1.0
    assert DeadlockRecoveryV1.scan_direction_from_depth(right_free) == -1.0


def test_breadcrumb_recovery_trajectory_respects_hardware_limits():
    shield = RuntimeTrajectorySafetyV1(RuntimeSafetyConfigV1())
    trajectory, duration, compliant = shield.recovery_trajectory(
        [2.0, 0.0, 1.0], [0.2, 0.0, 0.0], [0.0, 0.0, 0.0],
        [0.8, 0.0, 1.0],
    )
    assert compliant
    position, velocity, acceleration = _sample_polynomials(
        [trajectory], duration, 101
    )
    assert np.linalg.norm(velocity[0], axis=1).max() <= 6.001
    assert np.linalg.norm(acceleration[0], axis=1).max() <= 6.001
    np.testing.assert_allclose(position[0, -1], [0.8, 0.0, 1.0], atol=1e-8)
