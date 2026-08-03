"""Safety-first runtime selection for continuously replanned YOPO trajectories.

The network remains the proposal and score provider.  This module is a
deterministic last line of defence: candidates that violate the vehicle limits
or the current depth observation are not eligible for score-based selection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree

from policy.poly_solver import Poly5Solver


@dataclass(frozen=True)
class RuntimeSafetyConfigV1:
    enabled: bool = True
    max_speed_mps: float = 6.0
    max_acceleration_mps2: float = 6.0
    vehicle_radius_m: float = 0.30
    tracking_margin_m: float = 0.35
    trajectory_samples: int = 81
    depth_stride: int = 2
    limit_tolerance: float = 1.0e-3
    minimum_progress_m: float = 1.0
    braking_duration_min_s: float = 0.8
    braking_duration_max_s: float = 2.5
    braking_duration_step_s: float = 0.05
    feasibility_projection_enabled: bool = True
    feasibility_projection_min_scale: float = 0.15
    feasibility_projection_steps: int = 18
    clearance_sensor_tolerance_m: float = 0.08
    clearance_escape_drop_tolerance_m: float = 0.05
    clearance_escape_gain_m: float = 0.10

    @classmethod
    def from_mapping(cls, value):
        value = {} if value is None else dict(value)
        known = cls.__dataclass_fields__
        unknown = sorted(set(value) - set(known))
        if unknown:
            raise ValueError(f"unknown runtime_safety keys: {unknown}")
        result = cls(**value)
        result.validate()
        return result

    @property
    def required_clearance_m(self):
        return self.vehicle_radius_m + self.tracking_margin_m

    def validate(self):
        if self.max_speed_mps <= 0 or self.max_acceleration_mps2 <= 0:
            raise ValueError("runtime speed/acceleration limits must be positive")
        if self.vehicle_radius_m <= 0 or self.tracking_margin_m < 0:
            raise ValueError("invalid runtime clearance geometry")
        if self.trajectory_samples < 3 or self.depth_stride < 1:
            raise ValueError("runtime sampling contract is too small")
        if self.minimum_progress_m < 0:
            raise ValueError("minimum_progress_m must be non-negative")
        if not (0 < self.braking_duration_min_s <= self.braking_duration_max_s):
            raise ValueError("invalid braking duration interval")
        if self.braking_duration_step_s <= 0:
            raise ValueError("braking_duration_step_s must be positive")
        if not 0 < self.feasibility_projection_min_scale <= 1:
            raise ValueError("invalid feasibility projection minimum scale")
        if self.feasibility_projection_steps < 2:
            raise ValueError("feasibility_projection_steps must be at least two")
        if min(self.clearance_sensor_tolerance_m,
               self.clearance_escape_drop_tolerance_m,
               self.clearance_escape_gain_m) < 0:
            raise ValueError("clearance tolerances must be non-negative")

    def contract(self):
        return {
            "version": "runtime_trajectory_safety_v1",
            "selection": "lowest_network_score_among_hard_feasible_candidates",
            "fallback": "finite_horizon_braking_with_continuous_replanning",
            **asdict(self),
            "required_clearance_m": self.required_clearance_m,
        }


@dataclass(frozen=True)
class CandidateSafetyV1:
    feasible: bool
    reasons: tuple[str, ...]
    max_speed_mps: float
    max_acceleration_mps2: float
    min_observed_clearance_m: float | None
    endpoint_progress_m: float

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class SafetySelectionV1:
    action_id: int | None
    mode: str
    evaluations: tuple[CandidateSafetyV1, ...]


def _sample_polynomials(polynomials, duration_s, sample_count):
    times = np.linspace(0.0, float(duration_s), int(sample_count), dtype=np.float64)
    position = np.stack([
        [axis.get_position(times) for axis in candidate]
        for candidate in polynomials
    ], axis=0)
    velocity = np.stack([
        [axis.get_velocity(times) for axis in candidate]
        for candidate in polynomials
    ], axis=0)
    acceleration = np.stack([
        [axis.get_acceleration(times) for axis in candidate]
        for candidate in polynomials
    ], axis=0)
    return (
        np.transpose(position, (0, 2, 1)),
        np.transpose(velocity, (0, 2, 1)),
        np.transpose(acceleration, (0, 2, 1)),
    )


def depth_to_body_points_v1(depth, camera_model, rotation_body_from_camera,
                            stride=2):
    """Back-project raw optical depth and express points in body/YOPO axes."""
    array = np.asarray(depth, dtype=np.float32)
    expected = (int(camera_model.height), int(camera_model.width))
    if array.shape != expected:
        raise ValueError(f"runtime depth shape must be {expected}, got {array.shape}")
    sampled = array[::stride, ::stride] * np.float32(camera_model.depth_scale)
    vv, uu = np.meshgrid(
        np.arange(0, expected[0], stride, dtype=np.float32),
        np.arange(0, expected[1], stride, dtype=np.float32),
        indexing="ij",
    )
    valid = (
        np.isfinite(sampled)
        & (sampled >= float(camera_model.min_depth))
        & (sampled < float(camera_model.max_depth))
    )
    z = sampled[valid]
    if not len(z):
        return np.empty((0, 3), dtype=np.float64)
    optical = np.stack(
        ((uu[valid] - camera_model.cx) * z / camera_model.fx,
         (vv[valid] - camera_model.cy) * z / camera_model.fy,
         z), axis=1,
    )
    rotation = np.asarray(rotation_body_from_camera, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("rotation_body_from_camera must be 3x3")
    return np.asarray(optical, dtype=np.float64) @ rotation.T


class RuntimeTrajectorySafetyV1:
    def __init__(self, config: RuntimeSafetyConfigV1):
        config.validate()
        self.config = config

    def evaluate(self, polynomials: Sequence[Sequence[Poly5Solver]], duration_s,
                 obstacle_points_body, origin_world, rotation_world_from_body):
        if not polynomials:
            raise ValueError("at least one candidate polynomial is required")
        positions, velocities, accelerations = _sample_polynomials(
            polynomials, duration_s, self.config.trajectory_samples
        )
        origin = np.asarray(origin_world, dtype=np.float64).reshape(3)
        rotation = np.asarray(rotation_world_from_body, dtype=np.float64).reshape(3, 3)
        points = np.asarray(obstacle_points_body, dtype=np.float64).reshape(-1, 3)
        tree = cKDTree(points) if len(points) else None
        results = []
        for candidate_position, candidate_velocity, candidate_acceleration in zip(
            positions, velocities, accelerations
        ):
            speed = np.linalg.norm(candidate_velocity, axis=1)
            acceleration = np.linalg.norm(candidate_acceleration, axis=1)
            relative_body = (candidate_position - origin) @ rotation
            if tree is None:
                clearance = None
                initial_clearance = None
                final_clearance = None
            else:
                clearance_samples = tree.query(relative_body, workers=1)[0]
                clearance = float(np.min(clearance_samples))
                initial_clearance = float(clearance_samples[0])
                final_clearance = float(clearance_samples[-1])
            progress = float(np.linalg.norm(candidate_position[-1] - origin))
            reasons = []
            if not (np.isfinite(candidate_position).all()
                    and np.isfinite(candidate_velocity).all()
                    and np.isfinite(candidate_acceleration).all()):
                reasons.append("non_finite")
            if float(np.max(speed)) > (
                self.config.max_speed_mps + self.config.limit_tolerance
            ):
                reasons.append("speed_limit")
            if float(np.max(acceleration)) > (
                self.config.max_acceleration_mps2 + self.config.limit_tolerance
            ):
                reasons.append("acceleration_limit")
            if clearance is not None:
                hard_floor = max(
                    0.0,
                    self.config.vehicle_radius_m
                    - self.config.clearance_sensor_tolerance_m,
                )
                if clearance < hard_floor:
                    reasons.append("observed_collision_floor")
                elif initial_clearance >= self.config.required_clearance_m:
                    if clearance < self.config.required_clearance_m:
                        reasons.append("observed_clearance")
                else:
                    # A collision-free vehicle may start inside the preferred
                    # tracking margin.  It must not move closer and must make
                    # measurable progress out of that margin; otherwise the
                    # t=0 sample would reject every possible escape forever.
                    if clearance < (
                        initial_clearance
                        - self.config.clearance_escape_drop_tolerance_m
                    ):
                        reasons.append("clearance_escape_moves_closer")
                    required_final = min(
                        self.config.required_clearance_m,
                        initial_clearance + self.config.clearance_escape_gain_m,
                    )
                    if final_clearance < required_final:
                        reasons.append("clearance_escape_no_gain")
            if progress < self.config.minimum_progress_m:
                reasons.append("insufficient_progress")
            results.append(CandidateSafetyV1(
                feasible=not reasons,
                reasons=tuple(reasons),
                max_speed_mps=float(np.max(speed)),
                max_acceleration_mps2=float(np.max(acceleration)),
                min_observed_clearance_m=clearance,
                endpoint_progress_m=progress,
            ))
        return tuple(results)

    def project_endstate_candidates(self, start_position, start_velocity,
                                    start_acceleration, endstate_world,
                                    duration_s):
        """Project proposals into the 6/6 vehicle envelope without stopping.

        The largest feasible spatial scale is retained independently for every
        candidate. Direction and score identity are unchanged. This is not a
        limit relaxation: the returned quintic is explicitly re-sampled and
        must satisfy both hard limits before it can be selected.
        """
        start_position = np.asarray(start_position, dtype=np.float64).reshape(3)
        start_velocity = np.asarray(start_velocity, dtype=np.float64).reshape(3)
        start_acceleration = np.asarray(start_acceleration, dtype=np.float64).reshape(3)
        states = np.asarray(endstate_world, dtype=np.float64).reshape(-1, 3, 3)
        scales = np.linspace(
            1.0, self.config.feasibility_projection_min_scale,
            self.config.feasibility_projection_steps,
        ) if self.config.feasibility_projection_enabled else np.asarray([1.0])
        projected = []
        used_scales = []
        projection_succeeded = []
        for state in states:
            accepted = None
            for scale in scales:
                candidate = tuple(
                    Poly5Solver(
                        start_position[axis], start_velocity[axis],
                        start_acceleration[axis],
                        start_position[axis] + scale * state[axis, 0],
                        scale * state[axis, 1], scale * state[axis, 2],
                        duration_s,
                    )
                    for axis in range(3)
                )
                _, velocity, acceleration = _sample_polynomials(
                    [candidate], duration_s, self.config.trajectory_samples
                )
                if (
                    float(np.max(np.linalg.norm(velocity[0], axis=1)))
                    <= self.config.max_speed_mps + self.config.limit_tolerance
                    and float(np.max(np.linalg.norm(acceleration[0], axis=1)))
                    <= self.config.max_acceleration_mps2 + self.config.limit_tolerance
                ):
                    accepted = candidate
                    used_scales.append(float(scale))
                    projection_succeeded.append(True)
                    break
            if accepted is None:
                scale = float(scales[-1])
                accepted = tuple(
                    Poly5Solver(
                        start_position[axis], start_velocity[axis],
                        start_acceleration[axis],
                        start_position[axis] + scale * state[axis, 0],
                        scale * state[axis, 1], scale * state[axis, 2],
                        duration_s,
                    )
                    for axis in range(3)
                )
                used_scales.append(scale)
                projection_succeeded.append(False)
            projected.append(accepted)
        return tuple(projected), tuple(used_scales), tuple(projection_succeeded)

    def select(self, scores, evaluations):
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if len(scores) != len(evaluations):
            raise ValueError("score/evaluation candidate counts differ")
        eligible = np.asarray([item.feasible for item in evaluations], dtype=bool)
        if not bool(eligible.any()):
            return SafetySelectionV1(None, "finite_horizon_braking", tuple(evaluations))
        safe_scores = np.where(eligible, scores, np.inf)
        return SafetySelectionV1(
            int(np.argmin(safe_scores)), "network_safe", tuple(evaluations)
        )

    def braking_trajectory(self, position, velocity, acceleration):
        """Return the shortest limit-compliant smooth stop within the contract."""
        position = np.asarray(position, dtype=np.float64).reshape(3)
        velocity = np.asarray(velocity, dtype=np.float64).reshape(3)
        acceleration = np.asarray(acceleration, dtype=np.float64).reshape(3)
        acc_norm = float(np.linalg.norm(acceleration))
        if acc_norm > self.config.max_acceleration_mps2:
            acceleration *= self.config.max_acceleration_mps2 / acc_norm
        durations = np.arange(
            self.config.braking_duration_min_s,
            self.config.braking_duration_max_s + 0.5 * self.config.braking_duration_step_s,
            self.config.braking_duration_step_s,
        )
        last = None
        for duration in durations:
            endpoint = position + 0.5 * velocity * duration
            candidate = tuple(
                Poly5Solver(position[i], velocity[i], acceleration[i], endpoint[i], 0.0, 0.0,
                            float(duration))
                for i in range(3)
            )
            _, sampled_velocity, sampled_acceleration = _sample_polynomials(
                [candidate], duration, self.config.trajectory_samples
            )
            last = (candidate, float(duration))
            if (
                float(np.max(np.linalg.norm(sampled_velocity[0], axis=1)))
                <= self.config.max_speed_mps + self.config.limit_tolerance
                and float(np.max(np.linalg.norm(sampled_acceleration[0], axis=1)))
                <= self.config.max_acceleration_mps2 + self.config.limit_tolerance
            ):
                return candidate, float(duration), True
        return (*last, False) if last is not None else (None, None, False)

    def recovery_trajectory(self, position, velocity, acceleration, target):
        """Construct a bounded point-to-point retreat after braking.

        This does not certify unseen space.  Its caller may only supply a
        recent, actually flown breadcrumb and must retain that provenance in
        telemetry.  The method owns only the vehicle-limit calculation.
        """
        position = np.asarray(position, dtype=np.float64).reshape(3)
        velocity = np.asarray(velocity, dtype=np.float64).reshape(3)
        acceleration = np.asarray(acceleration, dtype=np.float64).reshape(3)
        target = np.asarray(target, dtype=np.float64).reshape(3)
        if not all(np.isfinite(value).all() for value in (
            position, velocity, acceleration, target
        )):
            raise FloatingPointError("recovery state contains NaN/Inf")
        durations = np.arange(
            max(1.0, self.config.braking_duration_min_s),
            max(3.0, self.config.braking_duration_max_s) + 0.05,
            self.config.braking_duration_step_s,
        )
        last = None
        for duration in durations:
            candidate = tuple(
                Poly5Solver(
                    position[axis], velocity[axis], acceleration[axis],
                    target[axis], 0.0, 0.0, float(duration),
                )
                for axis in range(3)
            )
            _, sampled_velocity, sampled_acceleration = _sample_polynomials(
                [candidate], duration, self.config.trajectory_samples
            )
            last = (candidate, float(duration))
            if (
                float(np.max(np.linalg.norm(sampled_velocity[0], axis=1)))
                <= self.config.max_speed_mps + self.config.limit_tolerance
                and float(np.max(np.linalg.norm(sampled_acceleration[0], axis=1)))
                <= self.config.max_acceleration_mps2 + self.config.limit_tolerance
            ):
                return candidate, float(duration), True
        return (*last, False) if last is not None else (None, None, False)


def clamp_vector_norm_v1(value, maximum):
    value = np.asarray(value, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm):
        raise FloatingPointError("cannot clamp a non-finite command")
    if norm <= maximum or norm == 0.0:
        return value, False
    return value * (float(maximum) / norm), True
