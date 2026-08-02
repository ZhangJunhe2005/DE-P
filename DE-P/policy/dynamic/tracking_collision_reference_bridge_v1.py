"""Development shadow bridge from surface tracks to collision geometry.

The bridge never mutates TrackManager or its Kalman filter.  It consumes only
current causal component geometry plus declared runtime priors.  Its output
keeps surface, reference-transform and final occupancy uncertainty separate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import time
from typing import Mapping, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares

from .measurement_geometry_adapter_v1 import DynamicMeasurementGeometryV1


BRIDGE_VERSION = "tracking_collision_reference_bridge_v1"
CONTRACT_VERSION = "tracking_collision_reference_contract_v1_candidate"


class ReferenceObservability(str, Enum):
    CENTER_OBSERVABLE = "CENTER_OBSERVABLE"
    CENTER_WEAKLY_OBSERVABLE = "CENTER_WEAKLY_OBSERVABLE"
    SUPPORT_ONLY = "SUPPORT_ONLY"
    REFERENCE_UNOBSERVABLE = "REFERENCE_UNOBSERVABLE"


@dataclass(frozen=True)
class ReferenceBridgeConfigV1:
    radius_prior_min_m: float = .18
    radius_prior_max_m: float = .42
    vertical_half_extent_upper_m: float = .80
    minimum_fit_points: int = 12
    minimum_angular_span_rad: float = .025
    maximum_fit_residual_m: float = .055
    maximum_fit_condition_number: float = 2.0e6
    maximum_fit_evaluations: int = 40
    border_guard_px: int = 2
    reference_model_floor_m: float = .035
    weak_reference_model_floor_m: float = .10
    interval_lateral_floor_m: float = .06
    fixed_shift_scales: Tuple[float, ...] = (.5, .75, 1.0)

    def validate(self):
        if not (
            0 < self.radius_prior_min_m < self.radius_prior_max_m
            and self.vertical_half_extent_upper_m >= self.radius_prior_max_m
            and self.minimum_fit_points >= 4
            and self.minimum_angular_span_rad > 0
            and self.maximum_fit_residual_m > 0
            and self.maximum_fit_condition_number > 1
            and self.maximum_fit_evaluations > 0
            and self.reference_model_floor_m > 0
            and self.weak_reference_model_floor_m > 0
            and self.interval_lateral_floor_m > 0
        ):
            raise ValueError("invalid reference bridge configuration")

    @classmethod
    def from_mapping(cls, values: Mapping):
        allowed = set(cls.__dataclass_fields__)
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown reference bridge fields: {sorted(unknown)}")
        result = cls(**dict(values))
        result.validate()
        return result


@dataclass(frozen=True)
class ReferenceEvidenceV1:
    valid: bool
    reference_mode: str
    observability: ReferenceObservability
    timestamp: float
    observation_id: int
    center_estimate_world: Optional[np.ndarray]
    center_interval_world: Optional[np.ndarray]
    estimated_radius_m: Optional[float]
    radius_interval_m: Tuple[float, float]
    surface_covariance_world: np.ndarray
    reference_transform_covariance_world: np.ndarray
    center_covariance_world: np.ndarray
    support_half_extent_world: np.ndarray
    fit_residual_m: Optional[float]
    fit_condition_number: Optional[float]
    angular_support_rad: Tuple[float, float]
    occupancy_model: str
    extent_provenance: str
    center_velocity_source: str
    fallback_reason: str
    runtime_gt_used: bool = False

    def __post_init__(self):
        if self.runtime_gt_used:
            raise ValueError("runtime GT is forbidden")
        for name in (
            "surface_covariance_world", "reference_transform_covariance_world",
            "center_covariance_world",
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (3, 3) or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite [3,3]")
            object.__setattr__(self, name, value.copy())
        half = np.asarray(self.support_half_extent_world, dtype=np.float64)
        if half.shape != (3,) or not np.isfinite(half).all() or np.any(half < 0):
            raise ValueError("support_half_extent_world must be nonnegative [3]")
        object.__setattr__(self, "support_half_extent_world", half.copy())
        if self.center_estimate_world is not None:
            center = np.asarray(self.center_estimate_world, dtype=np.float64)
            if center.shape != (3,) or not np.isfinite(center).all():
                raise ValueError("center estimate must be finite [3]")
            object.__setattr__(self, "center_estimate_world", center.copy())
        if self.center_interval_world is not None:
            interval = np.asarray(self.center_interval_world, dtype=np.float64)
            if interval.shape != (2, 3) or not np.isfinite(interval).all():
                raise ValueError("center interval must be finite [2,3]")
            object.__setattr__(self, "center_interval_world", interval.copy())


def _ray(geometry):
    delta = (
        geometry.cluster_centroid_world - geometry.camera_position_world
    )
    norm = float(np.linalg.norm(delta))
    if norm <= 1e-9:
        return None
    return delta / norm


def fixed_radial_shift(geometry, radius_prior_m, scale):
    """R1 diagnostic; it is never selected by ``build_hybrid``."""
    unit = _ray(geometry)
    if unit is None:
        raise ValueError("surface centroid coincides with camera")
    return (
        geometry.cluster_centroid_world
        + float(scale) * float(radius_prior_m) * unit
    )


def _sphere_fit(geometry, config):
    points = geometry.points_world
    unit = _ray(geometry)
    if unit is None or len(points) < config.minimum_fit_points:
        return None, "insufficient_geometry"
    initial_radius = float(np.clip(
        np.linalg.norm(geometry.cluster_centroid_world
                       - geometry.camera_position_world)
        * math.sin(.5 * max(geometry.angular_span_rad)),
        config.radius_prior_min_m, config.radius_prior_max_m,
    ))
    initial_center = geometry.cluster_centroid_world + initial_radius * unit
    lower = np.concatenate((
        geometry.cluster_centroid_world - config.radius_prior_max_m,
        [config.radius_prior_min_m],
    ))
    upper = np.concatenate((
        geometry.cluster_centroid_world + config.radius_prior_max_m,
        [config.radius_prior_max_m],
    ))
    result = least_squares(
        lambda value: np.linalg.norm(points-value[:3], axis=1)-value[3],
        np.concatenate((initial_center, [initial_radius])),
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=.02,
        max_nfev=config.maximum_fit_evaluations,
    )
    center, radius = result.x[:3], float(result.x[3])
    residuals = np.linalg.norm(points-center, axis=1)-radius
    residual = float(np.sqrt(np.mean(np.square(residuals))))
    jacobian = np.asarray(result.jac, dtype=np.float64)
    information = jacobian.T @ jacobian
    condition = float(np.linalg.cond(information))
    behind_surface = float((center-geometry.cluster_centroid_world) @ unit)
    at_bound = min(
        radius-config.radius_prior_min_m,
        config.radius_prior_max_m-radius,
    ) < .005
    fit = {
        "center": center,
        "radius": radius,
        "residual": residual,
        "condition": condition,
        "information": information,
        "behind_surface": behind_surface,
        "at_radius_bound": at_bound,
    }
    if not result.success:
        return fit, "optimizer_failed"
    if behind_surface <= 0:
        return fit, "center_not_behind_visible_surface"
    if not np.isfinite(condition) or condition > config.maximum_fit_condition_number:
        return fit, "fit_ill_conditioned"
    if residual > config.maximum_fit_residual_m:
        return fit, "fit_residual_exceeded"
    return fit, ""


def _interval(geometry, config, fitted_radius=None):
    unit = _ray(geometry)
    if unit is None:
        return None
    distance = float(np.linalg.norm(
        geometry.cluster_centroid_world-geometry.camera_position_world
    ))
    apparent = float(np.clip(
        distance * math.sin(.5 * max(geometry.angular_span_rad)),
        config.radius_prior_min_m, config.radius_prior_max_m,
    ))
    nominal = apparent if fitted_radius is None else float(fitted_radius)
    axial_min = max(
        .25 * config.radius_prior_min_m,
        nominal - .5 * (config.radius_prior_max_m-config.radius_prior_min_m),
    )
    axial_max = min(
        config.radius_prior_max_m,
        nominal + .5 * (config.radius_prior_max_m-config.radius_prior_min_m),
    )
    axial_max = max(axial_max, axial_min + .02)
    endpoints = np.stack((
        geometry.cluster_centroid_world + axial_min * unit,
        geometry.cluster_centroid_world + axial_max * unit,
    ))
    lateral = max(
        config.interval_lateral_floor_m,
        .5 * float(np.linalg.norm(geometry.visible_support_extent_world)),
    )
    if geometry.image_border_clipped:
        lateral += .5 * config.radius_prior_max_m
    return {
        "endpoints": endpoints,
        "midpoint": endpoints.mean(axis=0),
        "axial_width": axial_max-axial_min,
        "lateral_uncertainty": lateral,
        "apparent_radius": apparent,
    }


class TrackingCollisionReferenceBridgeV1:
    """Bounded R2/R3/R4/R6 implementation."""

    version = BRIDGE_VERSION

    def __init__(self, config=None):
        self.config = config or ReferenceBridgeConfigV1()
        self.config.validate()
        self.last_runtime_ms = 0.0

    def evaluate_geometry(self, geometry):
        started = time.perf_counter()
        if not isinstance(geometry, DynamicMeasurementGeometryV1):
            raise TypeError("geometry must be DynamicMeasurementGeometryV1")
        if not geometry.valid:
            output = self._invalid(geometry, geometry.failure_reason)
            self.last_runtime_ms = (time.perf_counter()-started)*1000
            return output
        fit, fit_failure = _sphere_fit(geometry, self.config)
        interval = _interval(
            geometry, self.config,
            None if fit is None else fit["radius"],
        )
        if interval is None:
            output = self._invalid(geometry, "view_ray_unavailable")
            self.last_runtime_ms = (time.perf_counter()-started)*1000
            return output
        angular = max(geometry.angular_span_rad)
        full_center_observable = bool(
            fit is not None
            and not fit_failure
            and not fit["at_radius_bound"]
            and angular >= self.config.minimum_angular_span_rad
            and geometry.image_border_distance_px > self.config.border_guard_px
        )
        weak_center = bool(
            fit is not None
            and not fit_failure
            and angular >= .5 * self.config.minimum_angular_span_rad
        )
        if full_center_observable:
            observability = ReferenceObservability.CENTER_OBSERVABLE
            mode = "geometry_center_fit"
            center = fit["center"]
            model_std = self.config.reference_model_floor_m
            reason = ""
        elif weak_center:
            observability = ReferenceObservability.CENTER_WEAKLY_OBSERVABLE
            mode = "hybrid_weak_center_interval"
            center = .5 * (fit["center"] + interval["midpoint"])
            model_std = self.config.weak_reference_model_floor_m
            reason = (
                "border_or_radius_identifiability"
                if fit["at_radius_bound"] or geometry.image_border_clipped
                else "weak_angular_support"
            )
        else:
            observability = ReferenceObservability.SUPPORT_ONLY
            mode = "surface_support_envelope"
            center = interval["midpoint"]
            model_std = max(
                self.config.weak_reference_model_floor_m,
                interval["lateral_uncertainty"],
                .5 * interval["axial_width"],
            )
            reason = fit_failure or "center_not_observable"
        reference_covariance = np.eye(3) * model_std**2
        if interval["axial_width"] > 0:
            unit = _ray(geometry)
            reference_covariance += (
                np.outer(unit, unit) * (interval["axial_width"]**2/12)
            )
        center_covariance = (
            geometry.cluster_covariance_world + reference_covariance
        )
        half_extent = np.asarray([
            self.config.radius_prior_max_m
            + interval["lateral_uncertainty"],
            self.config.radius_prior_max_m
            + interval["lateral_uncertainty"],
            self.config.vertical_half_extent_upper_m
            + interval["lateral_uncertainty"],
        ])
        output = ReferenceEvidenceV1(
            valid=True,
            reference_mode=mode,
            observability=observability,
            timestamp=geometry.timestamp,
            observation_id=geometry.observation_id,
            center_estimate_world=center,
            center_interval_world=interval["endpoints"],
            estimated_radius_m=(
                None if fit is None else float(fit["radius"])
            ),
            radius_interval_m=(
                self.config.radius_prior_min_m,
                self.config.radius_prior_max_m,
            ),
            surface_covariance_world=geometry.cluster_covariance_world,
            reference_transform_covariance_world=reference_covariance,
            center_covariance_world=center_covariance,
            support_half_extent_world=half_extent,
            fit_residual_m=(None if fit is None else fit["residual"]),
            fit_condition_number=(
                None if fit is None else fit["condition"]
            ),
            angular_support_rad=geometry.angular_span_rad,
            occupancy_model=(
                "bounded_vertical_extent_ellipsoid_over_center_interval"
            ),
            extent_provenance="fixed_contract_prior",
            center_velocity_source="not_available_single_measurement",
            fallback_reason=reason,
        )
        self.last_runtime_ms = (time.perf_counter()-started)*1000
        return output

    def _invalid(self, geometry, reason):
        zero = np.zeros((3, 3), dtype=np.float64)
        return ReferenceEvidenceV1(
            valid=False,
            reference_mode="INVALID",
            observability=ReferenceObservability.REFERENCE_UNOBSERVABLE,
            timestamp=geometry.timestamp,
            observation_id=geometry.observation_id,
            center_estimate_world=None,
            center_interval_world=None,
            estimated_radius_m=None,
            radius_interval_m=(
                self.config.radius_prior_min_m,
                self.config.radius_prior_max_m,
            ),
            surface_covariance_world=geometry.cluster_covariance_world,
            reference_transform_covariance_world=zero,
            center_covariance_world=geometry.cluster_covariance_world,
            support_half_extent_world=np.zeros(3),
            fit_residual_m=None,
            fit_condition_number=None,
            angular_support_rad=geometry.angular_span_rad,
            occupancy_model="none",
            extent_provenance="fixed_contract_prior",
            center_velocity_source="unavailable",
            fallback_reason=str(reason),
        )


__all__ = [
    "BRIDGE_VERSION", "CONTRACT_VERSION", "ReferenceObservability",
    "ReferenceBridgeConfigV1", "ReferenceEvidenceV1",
    "TrackingCollisionReferenceBridgeV1", "fixed_radial_shift",
]
