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
    dynamic_risk_ranking_enabled: bool = False
    dynamic_risk_score_weight: float = 0.0
    dynamic_risk_score_cap: float = 1.0
    dynamic_risk_distance_scale_m: float = 0.75
    dynamic_risk_ttc_scale_s: float = 1.50
    dynamic_risk_closing_speed_scale_mps: float = 2.0
    dynamic_time_retiming_enabled: bool = False
    dynamic_time_retiming_scales: tuple[float, ...] = (1.0, 1.2, 1.4)
    dynamic_command_freshness_watchdog_enabled: bool = False
    dynamic_command_max_age_s: float = 0.25
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
        if self.dynamic_risk_score_weight < 0.0:
            raise ValueError("dynamic risk score weight must be non-negative")
        if self.dynamic_risk_score_cap <= 0.0:
            raise ValueError("dynamic risk score cap must be positive")
        if min(
            self.dynamic_risk_distance_scale_m,
            self.dynamic_risk_ttc_scale_s,
            self.dynamic_risk_closing_speed_scale_mps,
        ) <= 0.0:
            raise ValueError("dynamic risk scales must be positive")
        scales = tuple(float(value) for value in self.dynamic_time_retiming_scales)
        if not scales or not np.isclose(scales[0], 1.0):
            raise ValueError("dynamic time scales must start at 1.0")
        if any(not np.isfinite(value) for value in scales) \
                or any(value < 1.0 for value in scales) \
                or any(right <= left for left, right in zip(scales, scales[1:])):
            raise ValueError(
                "dynamic time scales must be finite and strictly increasing"
            )
        if self.dynamic_command_max_age_s <= 0.0:
            raise ValueError("dynamic command maximum age must be positive")
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
            "dynamic_selection_policy": (
                "normal_candidate_hard_predicted_occupancy_veto_plus_"
                "bounded_ttc_distance_closing_speed_ranking"
                if self.dynamic_risk_ranking_enabled else
                "hard_predicted_occupancy_veto"
            ),
            "dynamic_emergency_policy": (
                "certified_brake_first_else_dynamic_only_minimum_risk_"
                "brake_static_hardware_boundary_failures_forbidden"
                if self.dynamic_risk_ranking_enabled else
                "legacy_finite_horizon_braking"
            ),
            "dynamic_time_policy": (
                "ordered_global_dynamic_only_temporal_retiming"
                if self.dynamic_time_retiming_enabled else
                "no_dynamic_temporal_retiming"
            ),
            "dynamic_authorization_scope": (
                "rolling_certified_prefix_with_command_freshness_watchdog"
                if self.dynamic_command_freshness_watchdog_enabled else
                "rolling_certified_prefix_without_command_freshness_watchdog"
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
    predicted_dynamic_collision_ttc_s: float | None = None
    minimum_dynamic_ttc_s: float | None = None
    dynamic_closest_approach_time_s: float | None = None
    dynamic_closing_speed_at_closest_mps: float = 0.0
    dynamic_uncertainty_radius_at_closest_m: float = 0.0
    dynamic_risk_cost: float = 0.0
    dynamic_risk_track_id: int | None = None
    dynamic_certified_horizon_s: float | None = None
    dynamic_full_duration_certified: bool | None = None

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class SafetySelectionV1:
    action_id: int | None
    mode: str
    evaluations: tuple[CandidateSafetyV1, ...]


def dynamic_command_is_fresh_v1(
    last_certified_plan_monotonic_s, now_monotonic_s, maximum_age_s,
):
    """Return whether a rolling dynamic certificate is still actionable.

    Dynamic occupancy is deliberately certified only over a near-term prefix:
    long constant-velocity extrapolation is not trustworthy for interactive
    actors.  Therefore a controller may keep executing a longer polynomial
    only while fresh depth/perception replans continually replace that prefix.
    """
    maximum_age_s = float(maximum_age_s)
    now_monotonic_s = float(now_monotonic_s)
    if maximum_age_s <= 0.0 or not np.isfinite(maximum_age_s):
        raise ValueError("maximum dynamic command age must be positive")
    if not np.isfinite(now_monotonic_s):
        raise ValueError("dynamic command clock must be finite")
    if last_certified_plan_monotonic_s is None:
        return False
    last = float(last_certified_plan_monotonic_s)
    if not np.isfinite(last):
        return False
    age = now_monotonic_s - last
    return bool(0.0 <= age <= maximum_age_s)


def select_dynamic_braking_option_v1(evaluations):
    """Choose a certified brake, else the least-bad dynamic-only brake.

    Static collision, boundary and hardware failures are never eligible for
    the uncertified fallback.  If no fully feasible stop exists, later TTC,
    shallower predicted overlap and lower continuous risk are preferred and
    the result is explicitly marked uncertified for telemetry.
    """
    evaluations = tuple(evaluations)
    if not evaluations:
        return None, False
    for index, evaluation in enumerate(evaluations):
        if evaluation.feasible:
            return index, True
    eligible = [
        (index, evaluation)
        for index, evaluation in enumerate(evaluations)
        if tuple(evaluation.reasons) == ("predicted_dynamic_clearance",)
    ]
    if not eligible:
        return None, False

    def key(value):
        _, evaluation = value
        collision_ttc = evaluation.predicted_dynamic_collision_ttc_s
        clearance = evaluation.min_predicted_dynamic_clearance_m
        return (
            -np.inf if collision_ttc is None else float(collision_ttc),
            -np.inf if clearance is None else float(clearance),
            -float(evaluation.dynamic_risk_cost),
        )

    return int(max(eligible, key=key)[0]), False


def classify_dynamic_blocking_v1(evaluations):
    """Classify why a complete candidate pool has no executable action.

    This is diagnostic/control-state semantics, not another Gate.  A
    ``dynamic_only`` result means at least one candidate would be executable
    after removing only the causal moving-occupancy reason.  ``mixed`` means
    dynamic predictions are present but every such candidate also violates a
    non-dynamic contract.
    """
    evaluations = tuple(evaluations)
    dynamic_only_count = sum(
        tuple(item.reasons) == ("predicted_dynamic_clearance",)
        for item in evaluations
    )
    non_dynamic_feasible_count = sum(
        not tuple(
            reason for reason in item.reasons
            if reason != "predicted_dynamic_clearance"
        )
        for item in evaluations
    )
    dynamic_veto_count = sum(
        "predicted_dynamic_clearance" in item.reasons
        for item in evaluations
    )
    if any(item.feasible for item in evaluations):
        cause = "none"
    elif dynamic_only_count:
        cause = "dynamic_only"
    elif dynamic_veto_count:
        cause = "mixed"
    else:
        cause = "non_dynamic_only"
    return {
        "cause": cause,
        "dynamic_only_veto_candidate_count": int(dynamic_only_count),
        "non_dynamic_feasible_candidate_count": int(
            non_dynamic_feasible_count
        ),
        "dynamic_hard_veto_count": int(dynamic_veto_count),
    }


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


def _dynamic_prediction_batch_v1(
    positions, velocities, durations, prepared_tracks, config,
):
    """Vectorize candidate/track prediction for bounded runtime latency."""
    candidate_count, sample_count, _ = positions.shape
    track_count = len(prepared_tracks)
    if track_count == 0:
        return tuple(None for _ in range(candidate_count))

    sample_fraction = np.linspace(
        0.0, 1.0, sample_count, dtype=np.float64,
    )
    times = durations[:, None] * sample_fraction[None, :]
    state_ages = np.asarray(
        [value[1] for value in prepared_tracks], dtype=np.float64,
    )
    track_positions = np.stack(
        [value[2] for value in prepared_tracks], axis=0,
    )
    track_velocities = np.stack(
        [value[3] for value in prepared_tracks], axis=0,
    )
    track_radii = np.asarray(
        [value[4] for value in prepared_tracks], dtype=np.float64,
    )
    covariance_radii = np.asarray(
        [value[5] for value in prepared_tracks], dtype=np.float64,
    )

    prediction_time = np.maximum(
        0.0,
        state_ages[None, :, None] + times[:, None, :],
    )
    active = np.broadcast_to(
        times[:, None, :]
        <= config.dynamic_track_prediction_horizon_s + config.limit_tolerance,
        (candidate_count, track_count, sample_count),
    )
    centers = (
        track_positions[None, :, None, :]
        + prediction_time[:, :, :, None]
        * track_velocities[None, :, None, :]
    )
    occupied_radius = (
        config.vehicle_radius_m
        + track_radii[None, :, None]
        + config.dynamic_track_prediction_margin_m
        + covariance_radii[None, :, None]
        + config.dynamic_track_uncertainty_growth_mps * prediction_time
    )
    relative = positions[:, None, :, :] - centers
    separation = np.linalg.norm(relative, axis=3)
    signed = separation - occupied_radius
    signed = np.where(active, signed, np.inf)
    closest_index = np.argmin(signed, axis=2)
    closest_signed = np.take_along_axis(
        signed, closest_index[:, :, None], axis=2,
    )[:, :, 0]

    relative_velocity = (
        velocities[:, None, :, :]
        - track_velocities[None, :, None, :]
    )
    radial_rate = np.sum(relative * relative_velocity, axis=3) \
        / np.maximum(separation, 1.0e-6)
    closing_speed = np.where(active, np.maximum(0.0, -radial_rate), 0.0)
    closest_closing = np.take_along_axis(
        closing_speed, closest_index[:, :, None], axis=2,
    )[:, :, 0]
    closest_time = np.take_along_axis(
        np.broadcast_to(times[:, None, :], signed.shape),
        closest_index[:, :, None], axis=2,
    )[:, :, 0]
    closest_prediction_time = np.take_along_axis(
        prediction_time, closest_index[:, :, None], axis=2,
    )[:, :, 0]

    collision_times = np.full(
        (candidate_count, track_count), np.inf, dtype=np.float64,
    )
    for candidate_index in range(candidate_count):
        for track_index in range(track_count):
            colliding = np.flatnonzero(
                active[candidate_index, track_index]
                & (signed[candidate_index, track_index] <= 0.0)
            )
            if not len(colliding):
                continue
            crossing_index = int(colliding[0])
            if crossing_index == 0:
                collision_times[candidate_index, track_index] = float(
                    times[candidate_index, 0]
                )
                continue
            before = float(signed[
                candidate_index, track_index, crossing_index - 1
            ])
            after = float(signed[
                candidate_index, track_index, crossing_index
            ])
            fraction = before / max(before - after, 1.0e-12)
            collision_times[candidate_index, track_index] = float(
                times[candidate_index, crossing_index - 1]
                + fraction * (
                    times[candidate_index, crossing_index]
                    - times[candidate_index, crossing_index - 1]
                )
            )

    surface_ttc = np.full_like(signed, np.inf)
    approaching = active & (closing_speed > 1.0e-6)
    surface_ttc[approaching] = (
        np.broadcast_to(times[:, None, :], signed.shape)[approaching]
        + np.maximum(signed[approaching], 0.0)
        / closing_speed[approaching]
    )
    minimum_surface_ttc = np.min(surface_ttc, axis=2)
    ttc_for_risk = np.where(
        np.isfinite(collision_times), collision_times, minimum_surface_ttc,
    )

    distance_risk = np.exp(
        -np.maximum(closest_signed, 0.0)
        / config.dynamic_risk_distance_scale_m
    )
    ttc_risk = np.where(
        np.isfinite(ttc_for_risk),
        np.exp(
            -np.maximum(ttc_for_risk, 0.0)
            / config.dynamic_risk_ttc_scale_s
        ),
        0.0,
    )
    closing_risk = np.clip(
        closest_closing / config.dynamic_risk_closing_speed_scale_mps,
        0.0, 1.0,
    ) * distance_risk
    track_risk = np.clip(
        0.50 * distance_risk + 0.35 * ttc_risk + 0.15 * closing_risk,
        0.0, config.dynamic_risk_score_cap,
    )
    if not config.dynamic_risk_ranking_enabled:
        track_risk.fill(0.0)
    best_risk_track = np.argmax(track_risk, axis=1)

    results = []
    for candidate_index in range(candidate_count):
        candidate_collision_times = collision_times[candidate_index]
        candidate_ttc = ttc_for_risk[candidate_index]
        finite_collision = candidate_collision_times[
            np.isfinite(candidate_collision_times)
        ]
        finite_ttc = candidate_ttc[np.isfinite(candidate_ttc)]
        track_index = int(best_risk_track[candidate_index])
        track = prepared_tracks[track_index][0]
        results.append({
            "predicted_clearance": float(np.min(
                closest_signed[candidate_index]
            )),
            "collision_ttc": (
                float(np.min(finite_collision))
                if len(finite_collision) else None
            ),
            "minimum_ttc": (
                float(np.min(finite_ttc)) if len(finite_ttc) else None
            ),
            "closest_time": (
                float(closest_time[candidate_index, track_index])
                if config.dynamic_risk_ranking_enabled else None
            ),
            "closing_speed": (
                float(closest_closing[candidate_index, track_index])
                if config.dynamic_risk_ranking_enabled else 0.0
            ),
            "uncertainty_radius": (
                float(
                    covariance_radii[track_index]
                    + config.dynamic_track_uncertainty_growth_mps
                    * closest_prediction_time[candidate_index, track_index]
                ) if config.dynamic_risk_ranking_enabled else 0.0
            ),
            "risk_cost": float(track_risk[candidate_index, track_index]),
            "risk_track_id": (
                int(track.track_id)
                if config.dynamic_risk_ranking_enabled
                and getattr(track, "track_id", None) is not None
                else None
            ),
            "certified_horizon": float(min(
                durations[candidate_index],
                config.dynamic_track_prediction_horizon_s,
            )),
            "full_duration_certified": bool(
                durations[candidate_index]
                <= config.dynamic_track_prediction_horizon_s
                + config.limit_tolerance
            ),
        })
    return tuple(results)


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
        relative_positions = (positions - origin[None, None, :]) @ rotation
        observed_clearance_samples = (
            None if tree is None else
            tree.query(
                relative_positions.reshape(-1, 3), workers=1
            )[0].reshape(relative_positions.shape[:2])
        )
        tracks = tuple(dynamic_tracks or ())
        goal_direction = None
        if goal_world is not None:
            goal_delta = np.asarray(goal_world, dtype=np.float64).reshape(3) - origin
            goal_norm = float(np.linalg.norm(goal_delta))
            if np.isfinite(goal_norm) and goal_norm >= 1.0e-6:
                goal_direction = goal_delta / goal_norm
        if tracks and query_timestamp is None:
            raise ValueError("query_timestamp is required with dynamic tracks")
        prepared_tracks = []
        if self.config.dynamic_track_prediction_enabled and tracks:
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
                if (
                    measurement_age < -1.0e-6
                    or measurement_age > self.config.dynamic_track_max_age_s
                    or state_age < -1.0e-6
                ):
                    continue
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
                            np.max(np.linalg.eigvalsh(covariance[:3, :3])),
                            0.0,
                        )))
                    )
                prepared_tracks.append((
                    track,
                    state_age,
                    np.asarray(track.position_world, dtype=np.float64),
                    np.asarray(track.velocity_world, dtype=np.float64),
                    track_radius,
                    covariance_radius,
                ))
        dynamic_predictions = _dynamic_prediction_batch_v1(
            positions, velocities, durations, prepared_tracks, self.config,
        )
        results = []
        for candidate_index, (
            candidate_position, candidate_velocity, candidate_acceleration,
            candidate_duration, relative_body,
        ) in enumerate(zip(
            positions, velocities, accelerations, durations,
            relative_positions,
        )):
            times = np.linspace(
                0.0, float(candidate_duration),
                self.config.trajectory_samples, dtype=np.float64,
            )
            speed = np.linalg.norm(candidate_velocity, axis=1)
            acceleration = np.linalg.norm(candidate_acceleration, axis=1)
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
                clearance_samples = observed_clearance_samples[
                    candidate_index
                ]
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
            predicted_dynamic_collision_ttc = None
            minimum_dynamic_ttc = None
            dynamic_closest_approach_time = None
            dynamic_closing_speed_at_closest = 0.0
            dynamic_uncertainty_radius_at_closest = 0.0
            dynamic_risk_cost = 0.0
            dynamic_risk_track_id = None
            dynamic_certified_horizon = None
            dynamic_full_duration_certified = None
            prediction = dynamic_predictions[candidate_index]
            if prediction is not None:
                predicted_dynamic_clearance = prediction[
                    "predicted_clearance"
                ]
                predicted_dynamic_collision_ttc = prediction[
                    "collision_ttc"
                ]
                minimum_dynamic_ttc = prediction["minimum_ttc"]
                dynamic_closest_approach_time = prediction["closest_time"]
                dynamic_closing_speed_at_closest = prediction[
                    "closing_speed"
                ]
                dynamic_uncertainty_radius_at_closest = prediction[
                    "uncertainty_radius"
                ]
                dynamic_risk_cost = prediction["risk_cost"]
                dynamic_risk_track_id = prediction["risk_track_id"]
                dynamic_certified_horizon = prediction[
                    "certified_horizon"
                ]
                dynamic_full_duration_certified = prediction[
                    "full_duration_certified"
                ]
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
                predicted_dynamic_collision_ttc_s=(
                    predicted_dynamic_collision_ttc
                ),
                minimum_dynamic_ttc_s=minimum_dynamic_ttc,
                dynamic_closest_approach_time_s=(
                    dynamic_closest_approach_time
                ),
                dynamic_closing_speed_at_closest_mps=(
                    dynamic_closing_speed_at_closest
                ),
                dynamic_uncertainty_radius_at_closest_m=(
                    dynamic_uncertainty_radius_at_closest
                ),
                dynamic_risk_cost=dynamic_risk_cost,
                dynamic_risk_track_id=dynamic_risk_track_id,
                dynamic_certified_horizon_s=dynamic_certified_horizon,
                dynamic_full_duration_certified=(
                    dynamic_full_duration_certified
                ),
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

    def retime_dynamic_only_candidates(
        self, polynomials, duration_s, evaluations, scale,
    ):
        """Retain geometry while slowing candidates blocked only by actors.

        This is a separate operation from hardware retiming.  It is permitted
        only for candidates whose complete base evaluation contains exactly
        ``predicted_dynamic_clearance``.  All other candidates and durations
        are preserved byte-for-byte so temporal scaling cannot wash out a
        static collision, boundary, visibility or vehicle-limit failure.

        The caller must run :meth:`evaluate` again on the returned pool.  V4.9
        searches one global scale at a time (1.2 before 1.4), rather than
        mixing a slower candidate into a pool where a faster safe action was
        already available.
        """
        candidates = tuple(polynomials)
        durations = _candidate_durations(duration_s, len(candidates))
        if len(evaluations) != len(candidates):
            raise ValueError("candidate/evaluation counts differ")
        scale = float(scale)
        if not np.isfinite(scale) or scale <= 1.0:
            raise ValueError("dynamic time scale must be finite and above one")
        allowed = tuple(float(value) for value in (
            self.config.dynamic_time_retiming_scales
        ))
        if not any(np.isclose(scale, value) for value in allowed[1:]):
            raise ValueError("dynamic time scale is not in the profile contract")

        rebuilt_pool = []
        rebuilt_durations = []
        applied_scales = []
        eligible = []
        for candidate, candidate_duration, evaluation in zip(
            candidates, durations, evaluations
        ):
            is_dynamic_only = tuple(evaluation.reasons) == (
                "predicted_dynamic_clearance",
            )
            eligible.append(is_dynamic_only)
            if not is_dynamic_only:
                rebuilt_pool.append(candidate)
                rebuilt_durations.append(float(candidate_duration))
                applied_scales.append(1.0)
                continue
            if len(candidate) != 3:
                raise ValueError("candidate must contain three axis polynomials")
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
                axis.get_position(candidate_duration) for axis in candidate
            ], dtype=np.float64)
            end_velocity = np.asarray([
                axis.get_velocity(candidate_duration) for axis in candidate
            ], dtype=np.float64)
            end_acceleration = np.asarray([
                axis.get_acceleration(candidate_duration) for axis in candidate
            ], dtype=np.float64)
            scaled_duration = float(candidate_duration) * scale
            rebuilt = tuple(
                Poly5Solver(
                    start_position[axis], start_velocity[axis],
                    start_acceleration[axis], end_position[axis],
                    end_velocity[axis] / scale,
                    end_acceleration[axis] / scale ** 2,
                    scaled_duration,
                )
                for axis in range(3)
            )
            rebuilt_pool.append(rebuilt)
            rebuilt_durations.append(scaled_duration)
            applied_scales.append(scale)
        return (
            tuple(rebuilt_pool), tuple(rebuilt_durations),
            tuple(applied_scales), tuple(eligible),
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
        apply_dynamic_risk=True,
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
        dynamic_risk_active = bool(
            apply_dynamic_risk
            and self.config.dynamic_risk_ranking_enabled
            and self.config.dynamic_risk_score_weight > 0.0
        )
        if dynamic_risk_active:
            # Continuous dynamic risk is a bounded ranking term among actions
            # that have already passed the hard predicted-occupancy veto.  It
            # cannot make an intersecting trajectory eligible and cannot
            # outweigh an arbitrarily better learned score.
            risk = np.clip(
                np.asarray([
                    item.dynamic_risk_cost for item in evaluations
                ], dtype=np.float64),
                0.0, self.config.dynamic_risk_score_cap,
            )
            effective_scores += self.config.dynamic_risk_score_weight * risk
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
                if (
                    not dynamic_risk_active
                    and item.min_predicted_dynamic_clearance_m is not None
                ):
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
            "network_safe" + (
                "_dynamic_risk" if dynamic_risk_active else ""
            ) + (
                "_clearance_preference"
                if clearance_preference_active else ""
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

    def braking_trajectory_options(
        self, position, velocity, acceleration, option_count=5,
    ):
        """Return bounded stop timings for dynamic minimum-risk fallback.

        A crossing actor can make the shortest stop unsafe while a slightly
        slower or longer stop is clear (or vice versa).  V4.9 therefore
        compares a small deterministic set rather than blindly installing the
        first limit-compliant brake.  This method owns only kinematics; every
        option must still undergo the complete static/dynamic evaluation.
        """
        option_count = int(option_count)
        if option_count < 2:
            raise ValueError("braking option count must be at least two")
        position = np.asarray(position, dtype=np.float64).reshape(3)
        velocity = np.asarray(velocity, dtype=np.float64).reshape(3)
        acceleration = np.asarray(acceleration, dtype=np.float64).reshape(3)
        acc_norm = float(np.linalg.norm(acceleration))
        if acc_norm > self.config.max_acceleration_mps2:
            acceleration *= self.config.max_acceleration_mps2 / acc_norm
        durations = np.linspace(
            self.config.braking_duration_min_s,
            self.config.braking_duration_max_s,
            option_count,
            dtype=np.float64,
        )
        options = []
        option_durations = []
        for duration in durations:
            endpoint = position + 0.5 * velocity * float(duration)
            candidate = tuple(
                Poly5Solver(
                    position[axis], velocity[axis], acceleration[axis],
                    endpoint[axis], 0.0, 0.0, float(duration),
                )
                for axis in range(3)
            )
            _, sampled_velocity, sampled_acceleration = _sample_polynomials(
                [candidate], duration, self.config.trajectory_samples,
            )
            if (
                float(np.max(np.linalg.norm(sampled_velocity[0], axis=1)))
                <= self.config.max_speed_mps + self.config.limit_tolerance
                and float(np.max(np.linalg.norm(
                    sampled_acceleration[0], axis=1,
                ))) <= (
                    self.config.max_acceleration_mps2
                    + self.config.limit_tolerance
                )
            ):
                options.append(candidate)
                option_durations.append(float(duration))
        return tuple(options), tuple(option_durations)

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
