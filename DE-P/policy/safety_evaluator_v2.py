"""Offline versioned Safety Evaluator V2.

V1 remains importable and unchanged.  This class owns only validation-time
geometry, timeline and label semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
from ruamel.yaml import YAML

from loss.safety_geometry_v2 import (
    actor_physical_clearance,
    continuous_sphere_segment_gap,
    directional_covariance_margin,
    maximum_eigenvalue_margin,
    planning_clearance,
    static_physical_clearance,
)


@dataclass(frozen=True)
class SafetyEvaluatorV2Config:
    evaluator_version: str
    uav_radius_m: float
    static_planning_margin_m: float
    dynamic_planning_margin_m: float
    tracking_control_margin_m: float
    estimated_uncertainty_policy: str
    confidence_sigma: float
    latency_s: float
    wall_clock_horizon_s: float
    horizon_semantics: str
    controlled_samples: int
    subdivisions: int
    contact_tolerance_m: float
    static_tolerance_m: float

    @classmethod
    def load(cls, config_path, controller_report):
        values = YAML(typ="safe").load(Path(config_path))
        controller = json.loads(Path(controller_report).read_text())
        return cls(
            evaluator_version=str(values["evaluator_version"]),
            uav_radius_m=float(values["physical_geometry"]["uav_radius_m"]),
            static_planning_margin_m=float(
                values["planning_margins"]["static_margin_m"]
            ),
            dynamic_planning_margin_m=float(
                values["planning_margins"]["dynamic_margin_m"]
            ),
            tracking_control_margin_m=float(
                values["planning_margins"]["tracking_control_margin_m"]
            ),
            estimated_uncertainty_policy=str(
                values["uncertainty"]["estimated_policy"]
            ),
            confidence_sigma=float(values["uncertainty"]["confidence_sigma"]),
            latency_s=float(controller["first_controllable_time_s"]),
            wall_clock_horizon_s=float(
                values["timeline"]["wall_clock_horizon_s"]
            ),
            horizon_semantics=str(values["timeline"]["horizon_semantics"]),
            controlled_samples=int(values["timeline"]["controlled_samples"]),
            subdivisions=int(
                values["continuous_collision"][
                    "adaptive_subdivisions_per_base_segment"
                ]
            ),
            contact_tolerance_m=float(
                values["physical_geometry"]["contact_tolerance_m"]
            ),
            static_tolerance_m=float(
                values["continuous_collision"]["static_tolerance_m"]
            ),
        )

    def canonical_hash(self):
        return hashlib.sha256(
            json.dumps(
                self.__dict__, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()


def _quintic_matrix(duration):
    value = float(duration)
    return np.asarray([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0],
        [1, value, value**2, value**3, value**4, value**5],
        [0, 1, 2*value, 3*value**2, 4*value**3, 5*value**4],
        [0, 0, 2, 6*value, 12*value**2, 20*value**3],
    ], dtype=float)


def quintic_coefficients(start_state, end_state, duration):
    start_state = np.asarray(start_state, dtype=float)
    end_state = np.asarray(end_state, dtype=float)
    if start_state.shape != (3, 3) or end_state.shape != (3, 3):
        raise ValueError("states must be [xyz,pva]")
    boundary = np.concatenate((start_state, end_state), axis=1)
    return (boundary @ _quintic_inverse(float(duration)).T)


@lru_cache(maxsize=8)
def _quintic_inverse(duration):
    return np.linalg.inv(_quintic_matrix(duration))


@lru_cache(maxsize=8)
def _position_fit_pseudoinverse(count, duration):
    times = np.linspace(float(duration) / count, float(duration), count)
    normalized = times / float(duration)
    vandermonde = np.stack(
        [normalized**power for power in range(6)], axis=1
    )
    return np.linalg.pinv(vandermonde)


def evaluate_quintic(coefficients, times):
    times = np.asarray(times, dtype=float)
    powers = np.stack([
        np.ones_like(times), times, times**2, times**3, times**4, times**5
    ], axis=-1)
    velocity_powers = np.stack([
        np.zeros_like(times), np.ones_like(times), 2*times,
        3*times**2, 4*times**3, 5*times**4,
    ], axis=-1)
    acceleration_powers = np.stack([
        np.zeros_like(times), np.zeros_like(times), 2*np.ones_like(times),
        6*times, 12*times**2, 20*times**3,
    ], axis=-1)
    return (
        powers @ coefficients.T,
        velocity_powers @ coefficients.T,
        acceleration_powers @ coefficients.T,
    )


def infer_terminal_state(sampled_positions, duration):
    """Recover terminal P/V/A from V1's exact quintic position samples."""
    positions = np.asarray(sampled_positions, dtype=float)
    count = positions.shape[0]
    coefficients = (_position_fit_pseudoinverse(count, float(duration)) @ positions).T
    terminal_position = coefficients.sum(axis=1)
    terminal_velocity = (
        coefficients[:, 1] + 2*coefficients[:, 2] + 3*coefficients[:, 3]
        + 4*coefficients[:, 4] + 5*coefficients[:, 5]
    ) / float(duration)
    terminal_acceleration = (
        2*coefficients[:, 2] + 6*coefficients[:, 3]
        + 12*coefficients[:, 4] + 20*coefficients[:, 5]
    ) / float(duration)**2
    return np.stack(
        (terminal_position, terminal_velocity, terminal_acceleration), axis=1
    )


class SafetyEvaluatorV2:
    def __init__(self, config):
        self.config = config

    def timeline(self, current_state, v1_positions, semantics=None):
        semantics = semantics or self.config.horizon_semantics
        current = np.asarray(current_state, dtype=float)
        latency = self.config.latency_s
        prefix_times = np.asarray([0.0, latency / 2.0, latency])
        prefix_position = (
            current[:, 0][None]
            + prefix_times[:, None] * current[:, 1][None]
            + 0.5 * prefix_times[:, None] ** 2 * current[:, 2][None]
        )
        first_state = current.copy()
        first_state[:, 0] = prefix_position[-1]
        first_state[:, 1] = current[:, 1] + latency * current[:, 2]
        end = infer_terminal_state(v1_positions, self.config.wall_clock_horizon_s)
        controlled_duration = (
            self.config.wall_clock_horizon_s - latency
            if semantics == "fixed_wall_clock"
            else self.config.wall_clock_horizon_s
        )
        base_count = self.config.controlled_samples * self.config.subdivisions
        local_times = np.linspace(0.0, controlled_duration, base_count + 1)
        coefficient = quintic_coefficients(first_state, end, controlled_duration)
        controlled_position, velocity, acceleration = evaluate_quintic(
            coefficient, local_times
        )
        # Prefix includes the first-controllable state; avoid duplicating it.
        wall_times = np.concatenate((prefix_times[:-1], latency + local_times))
        positions = np.vstack((prefix_position[:-1], controlled_position))
        return {
            "times": wall_times,
            "positions": positions,
            "velocities": velocity,
            "accelerations": acceleration,
            "first_controllable_state": first_state,
            "terminal_state": end,
            "controlled_duration_s": controlled_duration,
            "wall_clock_duration_s": float(wall_times[-1]),
            "semantics": semantics,
        }

    def uncertainty_margin(
        self, uav_position, actor_position, covariance, suite, policy=None
    ):
        if suite == "valid_gt" or covariance is None:
            return 0.0
        policy = policy or self.config.estimated_uncertainty_policy
        if policy == "none":
            return 0.0
        if policy == "directional_covariance":
            return directional_covariance_margin(
                uav_position, actor_position, covariance,
                self.config.confidence_sigma,
            )
        if policy == "maximum_eigenvalue":
            return maximum_eigenvalue_margin(
                covariance, self.config.confidence_sigma
            )
        raise ValueError(f"unknown uncertainty policy: {policy}")

    def evaluate_dynamic(
        self, timeline, actors_by_time, suite, covariance_by_actor=None,
        uncertainty_policy=None,
    ):
        positions = timeline["positions"]
        physical = np.full(len(positions), np.inf)
        planning = np.full(len(positions), np.inf)
        covariance_by_actor = covariance_by_actor or {}
        for time_index, (position, actors) in enumerate(
            zip(positions, actors_by_time)
        ):
            for actor in actors:
                if not actor.get("active", True) or not actor.get("dynamic", True):
                    continue
                gap = actor_physical_clearance(
                    position, actor, self.config.uav_radius_m
                )
                covariance = covariance_by_actor.get(int(actor["object_id"]))
                uncertainty = self.uncertainty_margin(
                    position, actor["position_world"], covariance, suite,
                    uncertainty_policy,
                )
                planned = planning_clearance(
                    gap,
                    planning_margin=self.config.dynamic_planning_margin_m,
                    tracking_control_margin=self.config.tracking_control_margin_m,
                    uncertainty_margin=uncertainty,
                )
                physical[time_index] = min(physical[time_index], gap)
                planning[time_index] = min(planning[time_index], float(planned))
        continuous = float(np.min(physical))
        for index in range(len(positions) - 1):
            left = {
                int(actor["object_id"]): actor
                for actor in actors_by_time[index]
                if actor.get("active", True)
            }
            right = {
                int(actor["object_id"]): actor
                for actor in actors_by_time[index + 1]
                if actor.get("active", True)
            }
            for identifier in left.keys() & right.keys():
                if left[identifier].get("type") != "sphere":
                    continue
                gap, _ = continuous_sphere_segment_gap(
                    positions[index], positions[index + 1],
                    left[identifier]["position_world"],
                    right[identifier]["position_world"],
                    self.config.uav_radius_m,
                    left[identifier]["radius"],
                )
                continuous = min(continuous, gap)
        return {
            "physical_clearance_by_time": physical,
            "planning_clearance_by_time": planning,
            "discrete_physical_min": float(np.min(physical)),
            "continuous_physical_min": continuous,
            "planning_min": float(np.min(planning)),
        }

    def evaluate_static(self, raw_esdf_by_time):
        physical = static_physical_clearance(
            raw_esdf_by_time, self.config.uav_radius_m
        )
        planning = planning_clearance(
            physical,
            planning_margin=self.config.static_planning_margin_m,
            tracking_control_margin=self.config.tracking_control_margin_m,
        )
        return {
            "physical_clearance_by_time": physical,
            "planning_clearance_by_time": planning,
            "physical_min": float(np.min(physical)),
            "planning_min": float(np.min(planning)),
        }

    @staticmethod
    def safety_first_label(
        dynamic_clearance, static_clearance, secondary, collision_priority=1e6,
        penetration_scale=1e3,
    ):
        dynamic = np.asarray(dynamic_clearance, dtype=float)
        static = np.asarray(static_clearance, dtype=float)
        secondary = np.asarray(secondary, dtype=float)
        violation = np.maximum(-dynamic, 0) + np.maximum(-static, 0)
        unsafe = violation > 1e-9
        spread = max(float(np.ptp(secondary)), 1e-9)
        normalized = (secondary - float(np.min(secondary))) / spread
        return (
            unsafe.astype(float) * collision_priority
            + violation * penetration_scale
            + normalized
        )
