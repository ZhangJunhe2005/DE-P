"""Deterministic continuous UAV state semantics for authoritative dataset V2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from authoritative_dataset.continuous_v1 import ContinuousState, certify_curve


STATE_SEMANTICS_VERSION = "authoritative_state_semantics_v2"
DEMONSTRATED_SPEED_MPS = 2.0
DEMONSTRATED_ACCELERATION_MPS2 = 2.0
NETWORK_SPEED_LIMIT_MPS = 6.0
NETWORK_ACCELERATION_LIMIT_MPS2 = 6.0
GOAL_DISTANCE_RANGE_M = (2.0, 8.0)
GOAL_TRANSFORM_TOLERANCE_M = 1e-9


@dataclass(frozen=True)
class UavSequenceState:
    position_world: np.ndarray
    velocity_world: np.ndarray
    acceleration_world: np.ndarray
    yaw: np.ndarray
    goal_world: np.ndarray
    reference_certificate: object
    reference_goal_world: np.ndarray


def yaw_rotation_world_from_body(yaw):
    yaw = float(yaw)
    cosine, sine = np.cos(yaw), np.sin(yaw)
    return np.asarray([
        [cosine, -sine, 0.0],
        [sine, cosine, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def quaternion_wxyz_from_yaw(yaw):
    return np.asarray([
        np.cos(float(yaw)/2), 0.0, 0.0, np.sin(float(yaw)/2)
    ], dtype=np.float64)


def goal_body_from_world(position_world, goal_world, yaw):
    rotation_world_from_body = yaw_rotation_world_from_body(yaw)
    delta_world = (
        np.asarray(goal_world, dtype=np.float64)
        - np.asarray(position_world, dtype=np.float64)
    )
    goal_body = rotation_world_from_body.T @ delta_world
    reconstructed = rotation_world_from_body @ goal_body
    if np.linalg.norm(reconstructed-delta_world) > GOAL_TRANSFORM_TOLERANCE_M:
        raise RuntimeError("goal world/body round-trip failed")
    return goal_body


def vector_body_from_world(vector_world, yaw):
    return yaw_rotation_world_from_body(yaw).T @ np.asarray(
        vector_world, dtype=np.float64
    )


def _quintic_coefficients(start, end, duration):
    """Scalar [position, velocity, acceleration] quintic coefficients."""
    duration = float(duration)
    matrix = np.asarray([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0],
        [1, duration, duration**2, duration**3, duration**4, duration**5],
        [0, 1, 2*duration, 3*duration**2, 4*duration**3, 5*duration**4],
        [0, 0, 2, 6*duration, 12*duration**2, 20*duration**3],
    ], dtype=np.float64)
    return np.linalg.solve(
        matrix, np.concatenate((
            np.asarray(start, dtype=np.float64),
            np.asarray(end, dtype=np.float64),
        ))
    )


def _evaluate(coefficients, times):
    times = np.asarray(times, dtype=np.float64)
    powers = np.stack([times**index for index in range(6)], axis=1)
    return powers @ np.asarray(coefficients, dtype=np.float64)


def _scenario_profile(scenario, rng, duration):
    """Return scenario-conditioned, physically bounded scalar motion."""
    if scenario == "hold":
        return 0.0, 0.0, 0.0
    if scenario == "near_boundary_recovery_stress":
        # Deliberately outside the demonstrated 2 m/s envelope, while still
        # inside the measured/network 6 m/s control envelope.
        return min(5.5, max(2.25, 0.85*duration)), 2.35, 0.25
    if scenario == "brake":
        return min(5.0, max(1.2, 0.70*duration)), 1.55, 0.05
    displacement = min(
        3.2, max(1.0, float(rng.uniform(.45, .75))*duration)
    )
    return displacement, float(rng.uniform(.25, .85)), float(
        rng.uniform(.15, .55)
    )


def _direction(scenario, rng):
    azimuth = float(rng.uniform(-np.pi, np.pi))
    vertical = (
        float(rng.uniform(.18, .32))
        if scenario == "vertical_reposition"
        else float(rng.uniform(-.14, .14))
    )
    horizontal = np.sqrt(max(1.0-vertical**2, 1e-12))
    return np.asarray([
        horizontal*np.cos(azimuth),
        horizontal*np.sin(azimuth),
        vertical,
    ], dtype=np.float64)


def _finite_difference_states(positions, dt):
    edge_order = 2 if len(positions) >= 3 else 1
    velocity = np.gradient(
        positions, float(dt), axis=0, edge_order=edge_order
    )
    acceleration = np.gradient(
        velocity, float(dt), axis=0, edge_order=edge_order
    )
    return velocity.astype(np.float64), acceleration.astype(np.float64)


def sample_uav_sequence(
    backend, rng, scenario, frame_times, safe_position,
    *, maximum_attempts=1000,
):
    """Sample one safe reference line and a smooth state trajectory on it."""
    frame_times = np.asarray(frame_times, dtype=np.float64)
    if len(frame_times) < 3 or not np.all(np.diff(frame_times) > 0):
        raise ValueError("at least three strictly increasing frame times required")
    duration = float(frame_times[-1])
    dt = float(np.median(np.diff(frame_times)))
    for _ in range(maximum_attempts):
        sampled = safe_position(backend, rng)
        if len(sampled) == 2:
            base, _ = sampled
            spatial_sample = None
        else:
            base, _, spatial_sample = sampled
        direction = _direction(scenario, rng)
        displacement, initial_speed, final_speed = _scenario_profile(
            scenario, rng, duration
        )
        minimum_goal = max(
            GOAL_DISTANCE_RANGE_M[0], displacement+2.0
        )
        if minimum_goal >= GOAL_DISTANCE_RANGE_M[1]:
            continue
        goal_distance = float(rng.uniform(
            minimum_goal, GOAL_DISTANCE_RANGE_M[1]
        ))
        reference_goal = base+direction*goal_distance
        if backend.query_one(reference_goal, .3)["collision"]:
            continue
        reference = lambda time: (
            base + direction*goal_distance*(
                10*(time/1.7)**3
                - 15*(time/1.7)**4
                + 6*(time/1.7)**5
            )
        )
        certificate = certify_curve(
            reference, backend, 1.7, max_depth=14
        )
        if certificate.state is not ContinuousState.CERTIFIED_SAFE:
            continue
        if scenario == "hold":
            progress = np.zeros_like(frame_times)
        else:
            coefficients = _quintic_coefficients(
                [0.0, initial_speed, 0.0],
                [displacement, final_speed, 0.0],
                duration,
            )
            progress = _evaluate(coefficients, frame_times)
            dense_times = np.linspace(0.0, duration, 257)
            dense_progress = _evaluate(coefficients, dense_times)
            if (
                dense_progress.min() < -1e-8
                or dense_progress.max() > goal_distance-2.0
                or np.any(np.diff(dense_progress) < -1e-8)
            ):
                continue
        positions = base+progress[:, None]*direction
        if any(backend.query_one(point, .3)["collision"] for point in positions):
            continue
        velocity, acceleration = _finite_difference_states(positions, dt)
        speed = np.linalg.norm(velocity, axis=1)
        accel = np.linalg.norm(acceleration, axis=1)
        if (
            speed.max() > NETWORK_SPEED_LIMIT_MPS
            or accel.max() > NETWORK_ACCELERATION_LIMIT_MPS2
        ):
            continue
        path_azimuth = float(np.arctan2(direction[1], direction[0]))
        yaw_offset = float(rng.uniform(-.55, .55))
        yaws = np.full(len(frame_times), path_azimuth+yaw_offset)
        goals = np.repeat(reference_goal[None], len(frame_times), axis=0)
        return UavSequenceState(
            position_world=positions,
            velocity_world=velocity,
            acceleration_world=acceleration,
            yaw=yaws,
            goal_world=goals,
            reference_certificate=certificate,
            reference_goal_world=reference_goal,
        )
    raise RuntimeError("maximum deterministic UAV state sampling attempts exceeded")


def actionability_from_state(
    backend, state, index, latency_s, dynamic_gap_m,
):
    position = state.position_world[index]
    velocity = state.velocity_world[index]
    acceleration = state.acceleration_world[index]
    current_query = backend.query_one(position, .3)
    first_position = (
        position + float(latency_s)*velocity
        + .5*float(latency_s)**2*acceleration
    )
    first_query = backend.query_one(first_position, .3)
    speed = float(np.linalg.norm(velocity))
    accel = float(np.linalg.norm(acceleration))
    outside_demonstrated = (
        speed > DEMONSTRATED_SPEED_MPS+1e-9
        or accel > DEMONSTRATED_ACCELERATION_MPS2+1e-9
    )
    future_speed = np.linalg.norm(
        state.velocity_world[index:], axis=1
    )
    future_accel = np.linalg.norm(
        state.acceleration_world[index:], axis=1
    )
    returns_to_envelope = bool(np.any(
        (future_speed <= DEMONSTRATED_SPEED_MPS+1e-9)
        & (future_accel <= DEMONSTRATED_ACCELERATION_MPS2+1e-9)
    ))
    initially_safe = (
        not current_query["collision"] and dynamic_gap_m > 0
    )
    first_safe = not first_query["collision"]
    stress = bool(outside_demonstrated)
    recoverable = bool(
        initially_safe and stress and outside_demonstrated
        and returns_to_envelope
    )
    preventable = bool(
        initially_safe and first_safe
        and state.reference_certificate.state
        is ContinuousState.CERTIFIED_SAFE
    )
    unrecoverable = bool(not preventable and not recoverable)
    actionability_class = (
        "recovery" if recoverable
        else "unrecoverable" if unrecoverable
        else "stress" if stress
        else "nominal"
    )
    return {
        "initially_safe": bool(initially_safe),
        "first_controllable_safe": bool(first_safe),
        "preventable": preventable,
        "recoverable": recoverable,
        "stress": stress,
        "feasibility_unknown": False,
        "nominal": actionability_class == "nominal",
        "unrecoverable": unrecoverable,
        "actionability_class": actionability_class,
        "state_envelope": (
            "outside_demonstrated_recoverable"
            if recoverable else
            "outside_demonstrated" if outside_demonstrated else
            "demonstrated"
        ),
    }, first_position
