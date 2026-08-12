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
    collision_representation_margin_m: float = 0.0
    tracking_margin_m: float = 0.35
    reaction_time_s: float = 0.12
    braking_acceleration_mps2: float = 6.0
    horizontal_fov_deg: float = 90.0
    vertical_fov_deg: float = 60.0
    visibility_margin_deg: float = 5.0
    trajectory_samples: int = 81
    depth_stride: int = 2
    limit_tolerance: float = 1.0e-3
    minimum_progress_m: float = 1.0
    maximum_reverse_progress_m: float = 0.05
    enforce_camera_visibility: bool = True
    enforce_minimum_progress: bool = True
    enforce_goal_progress: bool = True
    enforce_observed_tracking_clearance: bool = True
    enforce_stopping_distance: bool = True
    enforce_preferred_flight_volume_clearance: bool = True
    prefer_largest_boundary_escape_clearance: bool = True
    recovery_release_goal_progress_m: float = 0.25
    braking_duration_min_s: float = 0.8
    braking_duration_max_s: float = 2.5
    braking_duration_step_s: float = 0.05
    feasibility_projection_enabled: bool = True
    feasibility_projection_min_scale: float = 0.15
    feasibility_projection_steps: int = 18
    candidate_retiming_enabled: bool = False
    candidate_retiming_max_scale: float = 3.0
    candidate_retiming_steps: int = 17
    candidate_retiming_margin: float = 1.02
    clearance_sensor_tolerance_m: float = 0.08
    clearance_escape_drop_tolerance_m: float = 0.05
    clearance_escape_gain_m: float = 0.10
    dynamic_track_prediction_enabled: bool = True
    dynamic_track_radius_m: float = 0.35
    dynamic_track_prediction_margin_m: float = 0.25
    dynamic_track_uncertainty_growth_mps: float = 0.20
    dynamic_track_max_age_s: float = 0.25
    dynamic_track_prediction_horizon_s: float = 1.50
    dynamic_track_covariance_sigma: float = 2.0
    clearance_preference_weight: float = 0.0
    clearance_preference_saturation_m: float = 0.90

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

    @property
    def collision_floor_m(self):
        """Absolute non-penetration radius used by the active runtime profile.

        The default equals the physical body radius.  A versioned profile may
        add a small non-negative representation margin for depth sparsity and
        trajectory sampling, but sensor tolerance can never shrink the body.
        """
        return self.vehicle_radius_m + self.collision_representation_margin_m

    def validate(self):
        if self.max_speed_mps <= 0 or self.max_acceleration_mps2 <= 0:
            raise ValueError("runtime speed/acceleration limits must be positive")
        if (self.vehicle_radius_m <= 0 or self.tracking_margin_m < 0
                or self.collision_representation_margin_m < 0):
            raise ValueError("invalid runtime clearance geometry")
        if self.reaction_time_s < 0 or self.braking_acceleration_mps2 <= 0:
            raise ValueError("invalid runtime stopping-distance contract")
        if min(self.horizontal_fov_deg, self.vertical_fov_deg) <= 0 \
                or 2 * self.visibility_margin_deg >= min(
                    self.horizontal_fov_deg, self.vertical_fov_deg
                ):
            raise ValueError("invalid runtime visibility contract")
        if self.trajectory_samples < 3 or self.depth_stride < 1:
            raise ValueError("runtime sampling contract is too small")
        if self.minimum_progress_m < 0:
            raise ValueError("minimum_progress_m must be non-negative")
        if self.maximum_reverse_progress_m < 0 \
                or self.recovery_release_goal_progress_m < 0:
            raise ValueError("goal progress limits must be non-negative")
        if not (0 < self.braking_duration_min_s <= self.braking_duration_max_s):
            raise ValueError("invalid braking duration interval")
        if self.braking_duration_step_s <= 0:
            raise ValueError("braking_duration_step_s must be positive")
        if not 0 < self.feasibility_projection_min_scale <= 1:
            raise ValueError("invalid feasibility projection minimum scale")
        if self.feasibility_projection_steps < 2:
            raise ValueError("feasibility_projection_steps must be at least two")
        if self.candidate_retiming_max_scale < 1.0:
            raise ValueError("candidate_retiming_max_scale must be at least one")
        if self.candidate_retiming_steps < 2:
            raise ValueError("candidate_retiming_steps must be at least two")
        if self.candidate_retiming_margin < 1.0:
            raise ValueError("candidate_retiming_margin must be at least one")
        if min(self.clearance_sensor_tolerance_m,
               self.clearance_escape_drop_tolerance_m,
               self.clearance_escape_gain_m) < 0:
            raise ValueError("clearance tolerances must be non-negative")
        if min(
            self.dynamic_track_radius_m,
            self.dynamic_track_prediction_margin_m,
            self.dynamic_track_uncertainty_growth_mps,
            self.dynamic_track_max_age_s,
            self.dynamic_track_prediction_horizon_s,
            self.dynamic_track_covariance_sigma,
        ) < 0:
            raise ValueError("dynamic prediction geometry must be non-negative")
        if self.clearance_preference_weight < 0.0:
            raise ValueError("clearance preference weight must be non-negative")
        if self.clearance_preference_saturation_m <= 0.0:
            raise ValueError("clearance preference saturation must be positive")

    def contract(self):
        selection = (
            "maximum_boundary_recovery_clearance_then_"
            "lowest_network_score_among_hard_feasible_candidates"
            if self.prefer_largest_boundary_escape_clearance else
            "lowest_network_score_among_hard_feasible_candidates"
        )
        flight_volume_policy = (
            "preferred_tracking_envelope_with_physical_floor_"
            "and_one_way_boundary_escape"
            if self.enforce_preferred_flight_volume_clearance else
            "physical_body_floor_with_preferred_envelope_telemetry_only"
        )
        return {
            "version": "runtime_trajectory_safety_v1",
            "selection": selection,
            "fallback": "finite_horizon_braking_with_continuous_replanning",
            "flight_volume_policy": flight_volume_policy,
            "hardware_limit_policy": (
                "rebuild_same_endpoint_with_longer_duration_then_revalidate"
                if self.candidate_retiming_enabled else
                "reject_limit_violating_candidate"
            ),
            **asdict(self),
            "collision_floor_m": self.collision_floor_m,
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
    min_predicted_dynamic_clearance_m: float | None = None
    endpoint_goal_progress_m: float | None = None
    endpoint_goal_alignment: float | None = None
    minimum_stopping_reserve_m: float | None = None
    camera_visible: bool = True
    flight_volume_compliant: bool = True
    flight_volume_escape: bool = False
    initial_boundary_clearance_m: float | None = None
    minimum_boundary_clearance_m: float | None = None
    final_boundary_clearance_m: float | None = None

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class SafetySelectionV1:
    action_id: int | None
    mode: str
    evaluations: tuple[CandidateSafetyV1, ...]


_BOUNDARY_CLEARANCE_NAMES = (
    "x_min", "y_min", "z_min", "x_max", "y_max", "z_max",
)


def _flight_volume_clearances(positions, flight_bounds):
    """Return signed distances from positions to all six raw volume faces."""
    points = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    lower, upper = flight_bounds
    lower = np.asarray(lower, dtype=np.float64).reshape(3)
    upper = np.asarray(upper, dtype=np.float64).reshape(3)
    if not (np.isfinite(points).all() and np.isfinite(lower).all()
            and np.isfinite(upper).all() and np.all(lower < upper)):
        raise ValueError("invalid position or canonical flight bounds")
    return np.concatenate(
        (points - lower[None, :], upper[None, :] - points), axis=1
    )


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


def _candidate_durations(duration_s, candidate_count):
    """Normalize one shared duration or one duration per candidate."""
    values = np.asarray(duration_s, dtype=np.float64)
    if values.ndim == 0:
        values = np.full(int(candidate_count), float(values), dtype=np.float64)
    else:
        values = values.reshape(-1)
    if len(values) != int(candidate_count):
        raise ValueError("candidate duration count mismatch")
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("candidate durations must be finite and positive")
    return values


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

    def flight_volume_state(self, position_world, flight_bounds):
        """Describe the current position relative to physical/preferred bounds."""
        if flight_bounds is None:
            return None
        clearances = _flight_volume_clearances(
            np.asarray(position_world, dtype=np.float64).reshape(1, 3),
            flight_bounds,
        )[0]
        minimum = float(np.min(clearances))
        tolerance = self.config.limit_tolerance
        if minimum < self.config.collision_floor_m - tolerance:
            region = "physical_floor_violation"
        elif minimum < self.config.required_clearance_m - tolerance:
            region = "recovery_band"
        else:
            region = "preferred_envelope"
        return {
            "region": region,
            "minimum_signed_clearance_m": minimum,
            "signed_clearance_m": {
                name: float(value)
                for name, value in zip(_BOUNDARY_CLEARANCE_NAMES, clearances)
            },
            "physical_floor_m": float(self.config.collision_floor_m),
            "preferred_clearance_m": float(self.config.required_clearance_m),
        }

    def evaluate(self, polynomials: Sequence[Sequence[Poly5Solver]], duration_s,
                 obstacle_points_body, origin_world, rotation_world_from_body,
                 dynamic_tracks=(), query_timestamp=None, goal_world=None,
                 flight_bounds=None):
        if not polynomials:
            raise ValueError("at least one candidate polynomial is required")
        durations = _candidate_durations(duration_s, len(polynomials))
        sampled = [
            _sample_polynomials(
                [candidate], candidate_duration,
                self.config.trajectory_samples,
            )
            for candidate, candidate_duration in zip(polynomials, durations)
        ]
        positions = np.stack([value[0][0] for value in sampled], axis=0)
        velocities = np.stack([value[1][0] for value in sampled], axis=0)
        accelerations = np.stack([value[2][0] for value in sampled], axis=0)
        origin = np.asarray(origin_world, dtype=np.float64).reshape(3)
        rotation = np.asarray(rotation_world_from_body, dtype=np.float64).reshape(3, 3)
        points = np.asarray(obstacle_points_body, dtype=np.float64).reshape(-1, 3)
        tree = cKDTree(points) if len(points) else None
        tracks = tuple(dynamic_tracks or ())
        goal_direction = None
        if goal_world is not None:
            goal_delta = np.asarray(goal_world, dtype=np.float64).reshape(3) - origin
            goal_norm = float(np.linalg.norm(goal_delta))
            if np.isfinite(goal_norm) and goal_norm >= 1.0e-6:
                goal_direction = goal_delta / goal_norm
        if tracks and query_timestamp is None:
            raise ValueError("query_timestamp is required with dynamic tracks")
        results = []
        for (candidate_position, candidate_velocity, candidate_acceleration,
             candidate_duration) in zip(
            positions, velocities, accelerations, durations
        ):
            times = np.linspace(
                0.0, float(candidate_duration),
                self.config.trajectory_samples, dtype=np.float64,
            )
            speed = np.linalg.norm(candidate_velocity, axis=1)
            acceleration = np.linalg.norm(candidate_acceleration, axis=1)
            relative_body = (candidate_position - origin) @ rotation
            visible_points = relative_body[1:]
            forward = visible_points[:, 0]
            horizontal = np.arctan2(visible_points[:, 1], forward)
            vertical = np.arctan2(
                visible_points[:, 2],
                np.linalg.norm(visible_points[:, :2], axis=1),
            )
            horizontal_limit = np.deg2rad(
                0.5 * self.config.horizontal_fov_deg
                - self.config.visibility_margin_deg
            )
            vertical_limit = np.deg2rad(
                0.5 * self.config.vertical_fov_deg
                - self.config.visibility_margin_deg
            )
            camera_visible = bool(np.all(
                (forward > 0.0)
                & (np.abs(horizontal) <= horizontal_limit)
                & (np.abs(vertical) <= vertical_limit)
            ))
            flight_volume_compliant = True
            flight_volume_escape = False
            initial_boundary_clearance = None
            minimum_boundary_clearance = None
            final_boundary_clearance = None
            if flight_bounds is not None:
                boundary_clearances = _flight_volume_clearances(
                    candidate_position, flight_bounds
                )
                per_sample_boundary_clearance = np.min(
                    boundary_clearances, axis=1
                )
                initial_boundary_clearance = float(
                    per_sample_boundary_clearance[0]
                )
                minimum_boundary_clearance = float(np.min(
                    per_sample_boundary_clearance
                ))
                final_boundary_clearance = float(
                    per_sample_boundary_clearance[-1]
                )
                tolerance = self.config.limit_tolerance
                physical_floor = self.config.collision_floor_m
                preferred_clearance = self.config.required_clearance_m
                if not self.config.enforce_preferred_flight_volume_clearance:
                    # V4.5 keeps the raw map boundary as a physical body
                    # constraint only.  The wider preferred envelope remains
                    # observable through flight_volume_state but cannot veto
                    # a collision-free network candidate or force a turn.
                    flight_volume_compliant = bool(
                        minimum_boundary_clearance
                        >= physical_floor - tolerance
                    )
                elif initial_boundary_clearance >= preferred_clearance - tolerance:
                    # Normal operation keeps the complete tracking-error
                    # envelope inside the canonical flight volume.
                    flight_volume_compliant = bool(
                        minimum_boundary_clearance
                        >= preferred_clearance - tolerance
                    )
                elif initial_boundary_clearance >= physical_floor - tolerance:
                    # If controller tracking/inertia has already entered the
                    # preferred boundary band, the t=0 sample must not make
                    # every inward trajectory impossible.  Only a certified
                    # one-way escape is allowed: preserve the body floor,
                    # tolerate only a tiny transient drop, and finish farther
                    # from the nearest face.
                    required_final = min(
                        preferred_clearance,
                        initial_boundary_clearance
                        + self.config.clearance_escape_gain_m,
                    )
                    flight_volume_escape = bool(
                        minimum_boundary_clearance
                        >= initial_boundary_clearance
                        - self.config.clearance_escape_drop_tolerance_m
                        - tolerance
                        and minimum_boundary_clearance
                        >= physical_floor - tolerance
                        and final_boundary_clearance
                        >= required_final - tolerance
                    )
                    flight_volume_compliant = flight_volume_escape
                else:
                    # Starting inside the physical body floor is never
                    # authorized by this escape rule.
                    flight_volume_compliant = False
            if tree is None:
                clearance = None
                initial_clearance = None
                final_clearance = None
                minimum_stopping_reserve = None
            else:
                clearance_samples = tree.query(relative_body, workers=1)[0]
                clearance = float(np.min(clearance_samples))
                initial_clearance = float(clearance_samples[0])
                final_clearance = float(clearance_samples[-1])
                dt = float(candidate_duration) / max(
                    1, self.config.trajectory_samples - 1
                )
                closing_speed = np.maximum(
                    0.0, (clearance_samples[:-1] - clearance_samples[1:]) / dt
                )
                closing_speed = np.concatenate(
                    (closing_speed, closing_speed[-1:]), axis=0
                )
                # The absolute floor is the physical body radius.  Sensor
                # tolerance is not permission to penetrate that body; normal
                # operation uses the larger tracking/stopping envelope below.
                hard_floor = self.config.collision_floor_m
                # Do not make an already-close but collision-free vehicle
                # mathematically unable to escape. While its clearance is
                # monotonically recovering from inside the preferred margin,
                # the non-penetration floor is authoritative; normal tracking
                # clearance resumes as soon as that margin is recovered.
                baseline_clearance = np.full_like(
                    clearance_samples, self.config.required_clearance_m
                )
                if initial_clearance < self.config.required_clearance_m:
                    baseline_clearance = np.where(
                        clearance_samples < self.config.required_clearance_m,
                        hard_floor, self.config.required_clearance_m,
                    )
                required_stopping_clearance = (
                    baseline_clearance
                    + closing_speed * self.config.reaction_time_s
                    + closing_speed ** 2
                    / (2.0 * self.config.braking_acceleration_mps2)
                )
                minimum_stopping_reserve = float(np.min(
                    clearance_samples - required_stopping_clearance
                ))
            endpoint_delta = candidate_position[-1] - origin
            progress = float(np.linalg.norm(endpoint_delta))
            goal_progress = None
            goal_alignment = None
            if goal_direction is not None:
                goal_progress = float(np.dot(endpoint_delta, goal_direction))
                goal_alignment = goal_progress / max(progress, 1.0e-6)
            reasons = []
            if self.config.enforce_camera_visibility and not camera_visible:
                reasons.append("outside_camera_visibility")
            if not flight_volume_compliant:
                reasons.append("flight_volume")
            predicted_dynamic_clearance = None
            if self.config.dynamic_track_prediction_enabled and tracks:
                dynamic_clearances = []
                for track in tracks:
                    if not bool(getattr(track, "is_dynamic", False)):
                        continue
                    state_timestamp = float(track.timestamp)
                    direct_timestamp = getattr(
                        track, "last_direct_observation_timestamp", None
                    )
                    if direct_timestamp is None:
                        direct_timestamp = state_timestamp
                    measurement_age = (
                        float(query_timestamp) - float(direct_timestamp)
                    )
                    state_age = float(query_timestamp) - state_timestamp
                    if (measurement_age < -1.0e-6
                            or measurement_age
                            > self.config.dynamic_track_max_age_s
                            or state_age < -1.0e-6):
                        continue
                    active = times <= (
                        self.config.dynamic_track_prediction_horizon_s
                        + self.config.limit_tolerance
                    )
                    if not bool(np.any(active)):
                        continue
                    active_times = times[active]
                    horizon = np.maximum(0.0, state_age + active_times)
                    center = (
                        np.asarray(track.position_world, dtype=np.float64)[None, :]
                        + horizon[:, None]
                        * np.asarray(track.velocity_world, dtype=np.float64)[None, :]
                    )
                    extent = np.asarray(
                        getattr(track, "observed_extent", (0.0, 0.0, 0.0)),
                        dtype=np.float64,
                    )
                    track_radius = max(
                        self.config.dynamic_track_radius_m,
                        0.5 * float(np.max(extent))
                        if extent.shape == (3,) and np.isfinite(extent).all()
                        else 0.0,
                    )
                    covariance_radius = 0.0
                    covariance = np.asarray(
                        getattr(track, "state_covariance", ()),
                        dtype=np.float64,
                    )
                    if covariance.shape == (6, 6) \
                            and np.isfinite(covariance).all():
                        covariance_radius = (
                            self.config.dynamic_track_covariance_sigma
                            * float(np.sqrt(max(
                                np.max(np.linalg.eigvalsh(
                                    covariance[:3, :3]
                                )),
                                0.0,
                            )))
                        )
                    occupied_radius = (
                        self.config.vehicle_radius_m
                        + track_radius
                        + self.config.dynamic_track_prediction_margin_m
                        + covariance_radius
                        + self.config.dynamic_track_uncertainty_growth_mps * horizon
                    )
                    signed = np.linalg.norm(
                        candidate_position[active] - center, axis=1
                    ) - occupied_radius
                    dynamic_clearances.append(float(np.min(signed)))
                if dynamic_clearances:
                    predicted_dynamic_clearance = min(dynamic_clearances)
                    if predicted_dynamic_clearance < 0.0:
                        reasons.append("predicted_dynamic_clearance")
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
                if clearance < hard_floor:
                    reasons.append("observed_collision_floor")
                elif (self.config.enforce_observed_tracking_clearance
                      and initial_clearance >= self.config.required_clearance_m):
                    if clearance < self.config.required_clearance_m:
                        reasons.append("observed_clearance")
                elif self.config.enforce_observed_tracking_clearance:
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
                if (self.config.enforce_stopping_distance
                        and minimum_stopping_reserve < 0.0):
                    reasons.append("stopping_distance")
            if (self.config.enforce_minimum_progress
                    and progress < self.config.minimum_progress_m):
                reasons.append("insufficient_progress")
            if (self.config.enforce_goal_progress
                    and goal_progress is not None
                    and goal_progress < -self.config.maximum_reverse_progress_m):
                reasons.append("reverse_goal_progress")
            results.append(CandidateSafetyV1(
                feasible=not reasons,
                reasons=tuple(reasons),
                max_speed_mps=float(np.max(speed)),
                max_acceleration_mps2=float(np.max(acceleration)),
                min_observed_clearance_m=clearance,
                endpoint_progress_m=progress,
                min_predicted_dynamic_clearance_m=predicted_dynamic_clearance,
                endpoint_goal_progress_m=goal_progress,
                endpoint_goal_alignment=goal_alignment,
                minimum_stopping_reserve_m=minimum_stopping_reserve,
                camera_visible=camera_visible,
                flight_volume_compliant=flight_volume_compliant,
                flight_volume_escape=flight_volume_escape,
                initial_boundary_clearance_m=initial_boundary_clearance,
                minimum_boundary_clearance_m=minimum_boundary_clearance,
                final_boundary_clearance_m=final_boundary_clearance,
            ))
        return tuple(results)

    def retime_candidates(self, polynomials, duration_s):
        """Slow limit-violating candidates without changing their endpoint.

        A larger duration is searched independently for every candidate.  The
        measured start velocity/acceleration are preserved, while terminal
        derivatives are scaled consistently with the longer time base.  The
        resulting quintic is re-sampled before it is authorized; callers must
        still run the complete static/dynamic collision evaluation because a
        new duration can slightly change the geometric curve and actor timing.

        This is deliberately different from command clipping: position,
        velocity and acceleration all come from one internally consistent
        polynomial and one duration.
        """
        candidates = tuple(polynomials)
        if not candidates:
            raise ValueError("at least one candidate polynomial is required")
        base_duration = float(duration_s)
        if not np.isfinite(base_duration) or base_duration <= 0.0:
            raise ValueError("candidate duration must be finite and positive")
        if not self.config.candidate_retiming_enabled:
            return (
                candidates,
                tuple(base_duration for _ in candidates),
                tuple(1.0 for _ in candidates),
                tuple(True for _ in candidates),
            )

        retimed = []
        durations = []
        used_scales = []
        succeeded = []
        tolerance = self.config.limit_tolerance
        for candidate in candidates:
            if len(candidate) != 3:
                raise ValueError("candidate must contain three axis polynomials")
            _, original_velocity, original_acceleration = _sample_polynomials(
                [candidate], base_duration, self.config.trajectory_samples
            )
            maximum_speed = float(np.max(np.linalg.norm(
                original_velocity[0], axis=1
            )))
            maximum_acceleration = float(np.max(np.linalg.norm(
                original_acceleration[0], axis=1
            )))
            if (
                maximum_speed <= self.config.max_speed_mps + tolerance
                and maximum_acceleration
                <= self.config.max_acceleration_mps2 + tolerance
            ):
                retimed.append(candidate)
                durations.append(base_duration)
                used_scales.append(1.0)
                succeeded.append(True)
                continue

            required_scale = max(
                1.0,
                maximum_speed / self.config.max_speed_mps,
                np.sqrt(maximum_acceleration / self.config.max_acceleration_mps2),
            ) * self.config.candidate_retiming_margin
            first_scale = min(
                self.config.candidate_retiming_max_scale,
                max(1.0, required_scale),
            )
            scale_grid = np.linspace(
                first_scale,
                self.config.candidate_retiming_max_scale,
                self.config.candidate_retiming_steps,
                dtype=np.float64,
            )
            start_position = np.asarray([
                axis.get_position(0.0) for axis in candidate
            ], dtype=np.float64)
            start_velocity = np.asarray([
                axis.get_velocity(0.0) for axis in candidate
            ], dtype=np.float64)
            start_acceleration = np.asarray([
                axis.get_acceleration(0.0) for axis in candidate
            ], dtype=np.float64)
            end_position = np.asarray([
                axis.get_position(base_duration) for axis in candidate
            ], dtype=np.float64)
            end_velocity = np.asarray([
                axis.get_velocity(base_duration) for axis in candidate
            ], dtype=np.float64)
            end_acceleration = np.asarray([
                axis.get_acceleration(base_duration) for axis in candidate
            ], dtype=np.float64)

            accepted = None
            accepted_duration = None
            accepted_scale = None
            last = None
            for scale in scale_grid:
                candidate_duration = base_duration * float(scale)
                rebuilt = tuple(
                    Poly5Solver(
                        start_position[axis], start_velocity[axis],
                        start_acceleration[axis], end_position[axis],
                        end_velocity[axis] / float(scale),
                        end_acceleration[axis] / float(scale) ** 2,
                        candidate_duration,
                    )
                    for axis in range(3)
                )
                _, velocity, acceleration = _sample_polynomials(
                    [rebuilt], candidate_duration,
                    self.config.trajectory_samples,
                )
                last = (rebuilt, candidate_duration, float(scale))
                if (
                    float(np.max(np.linalg.norm(velocity[0], axis=1)))
                    <= self.config.max_speed_mps + tolerance
                    and float(np.max(np.linalg.norm(acceleration[0], axis=1)))
                    <= self.config.max_acceleration_mps2 + tolerance
                ):
                    accepted = rebuilt
                    accepted_duration = candidate_duration
                    accepted_scale = float(scale)
                    break
            if accepted is None:
                accepted, accepted_duration, accepted_scale = last
                succeeded.append(False)
            else:
                succeeded.append(True)
            retimed.append(accepted)
            durations.append(float(accepted_duration))
            used_scales.append(float(accepted_scale))
        return (
            tuple(retimed), tuple(durations), tuple(used_scales),
            tuple(succeeded),
        )

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

    def select(
        self, scores, evaluations, *, apply_clearance_preference=True,
    ):
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if len(scores) != len(evaluations):
            raise ValueError("score/evaluation candidate counts differ")
        eligible = np.asarray([item.feasible for item in evaluations], dtype=bool)
        if not bool(eligible.any()):
            return SafetySelectionV1(None, "finite_horizon_braking", tuple(evaluations))
        boundary_escape = np.asarray([
            item.feasible and item.flight_volume_escape
            for item in evaluations
        ], dtype=bool)
        if (self.config.prefer_largest_boundary_escape_clearance
                and bool(boundary_escape.any())):
            # Boundary recovery is position-critical: select the certified
            # candidate that restores the most signed boundary clearance.
            # Network score is only the tie-breaker among equal-clearance
            # escapes.  Every candidate here has already passed all other
            # physical and dynamic safety checks.
            endpoint_clearance = np.asarray([
                -np.inf if item.final_boundary_clearance_m is None
                else item.final_boundary_clearance_m
                for item in evaluations
            ], dtype=np.float64)
            best_clearance = float(np.max(endpoint_clearance[boundary_escape]))
            best_escape = boundary_escape & np.isclose(
                endpoint_clearance, best_clearance, rtol=0.0, atol=1.0e-9
            )
            escape_scores = np.where(best_escape, scores, np.inf)
            return SafetySelectionV1(
                int(np.argmin(escape_scores)), "boundary_escape",
                tuple(evaluations),
            )
        effective_scores = scores.copy()
        clearance_preference_active = bool(
            apply_clearance_preference
            and self.config.clearance_preference_weight > 0.0
        )
        if clearance_preference_active:
            # This is a bounded preference, not another feasibility Gate.  It
            # only breaks close score decisions in favour of a little more
            # physical clearance; the benefit saturates so a very wide detour
            # cannot overwhelm the learned YOPO goal/smoothness objective.
            spare_clearance = []
            for item in evaluations:
                values = []
                if item.min_observed_clearance_m is not None:
                    values.append(
                        item.min_observed_clearance_m
                        - self.config.collision_floor_m
                    )
                if item.min_predicted_dynamic_clearance_m is not None:
                    # Dynamic clearance is already signed relative to the
                    # predicted occupied radius.
                    values.append(item.min_predicted_dynamic_clearance_m)
                spare_clearance.append(min(values) if values else 0.0)
            normalized = np.clip(
                np.asarray(spare_clearance, dtype=np.float64),
                0.0,
                self.config.clearance_preference_saturation_m,
            ) / self.config.clearance_preference_saturation_m
            effective_scores -= (
                self.config.clearance_preference_weight * normalized
            )
        safe_scores = np.where(eligible, effective_scores, np.inf)
        return SafetySelectionV1(
            int(np.argmin(safe_scores)),
            (
                "network_safe_clearance_preference"
                if clearance_preference_active
                else "network_safe"
            ),
            tuple(evaluations),
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
