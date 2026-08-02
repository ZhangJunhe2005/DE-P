"""Fast analytic sphere/cylinder hypotheses from causal depth geometry."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional, Tuple

import numpy as np

from .measurement_geometry_adapter_v2 import DynamicMeasurementGeometryV2


MODEL_VERSION = "dynamic_object_geometry_model_v1"


class ShapeObservability(str, Enum):
    SPHERE_OBSERVABLE = "SPHERE_OBSERVABLE"
    CYLINDER_OBSERVABLE = "CYLINDER_OBSERVABLE"
    SHAPE_AMBIGUOUS = "SHAPE_AMBIGUOUS"
    SUPPORT_ONLY = "SUPPORT_ONLY"
    GEOMETRY_INVALID = "GEOMETRY_INVALID"


@dataclass(frozen=True)
class GeometryModelConfigV1:
    sphere_radius_interval_m: Tuple[float, float] = (.18, .42)
    cylinder_radius_interval_m: Tuple[float, float] = (.34, .42)
    cylinder_half_height_interval_m: Tuple[float, float] = (.70, .85)
    minimum_points: int = 12
    maximum_sphere_residual_m: float = .045
    maximum_cylinder_residual_m: float = .045
    maximum_condition_number: float = 2e6
    minimum_relative_fit_margin: float = .20
    cylinder_minimum_silhouette_aspect: float = 1.35
    cylinder_minimum_vertical_extent_m: float = .55
    sphere_maximum_silhouette_aspect: float = 1.45
    border_evidence_weight: float = .35
    occlusion_evidence_weight: float = .50
    robust_reweighting_iterations: int = 2

    @classmethod
    def from_mapping(cls, priors: Mapping, observability: Mapping, features: Mapping):
        result = cls(
            sphere_radius_interval_m=tuple(
                float(x) for x in priors["sphere_radius_interval_m"]
            ),
            cylinder_radius_interval_m=tuple(
                float(x) for x in priors["cylinder_radius_interval_m"]
            ),
            cylinder_half_height_interval_m=tuple(
                float(x) for x in priors[
                    "cylinder_half_height_interval_m"
                ]
            ),
            minimum_points=int(observability["minimum_points"]),
            maximum_sphere_residual_m=float(
                observability["maximum_sphere_residual_m"]
            ),
            maximum_cylinder_residual_m=float(
                observability["maximum_cylinder_residual_m"]
            ),
            maximum_condition_number=float(
                observability["maximum_condition_number"]
            ),
            minimum_relative_fit_margin=float(
                observability["minimum_relative_fit_margin"]
            ),
            cylinder_minimum_silhouette_aspect=float(
                observability["cylinder_minimum_silhouette_aspect"]
            ),
            cylinder_minimum_vertical_extent_m=float(
                observability["cylinder_minimum_vertical_extent_m"]
            ),
            sphere_maximum_silhouette_aspect=float(
                observability["sphere_maximum_silhouette_aspect"]
            ),
            border_evidence_weight=float(
                observability["border_evidence_weight"]
            ),
            occlusion_evidence_weight=float(
                observability["occlusion_evidence_weight"]
            ),
            robust_reweighting_iterations=int(
                features["robust_reweighting_iterations"]
            ),
        )
        result.validate()
        return result

    def validate(self):
        if not (
            0 < self.sphere_radius_interval_m[0]
            < self.sphere_radius_interval_m[1]
            and 0 < self.cylinder_radius_interval_m[0]
            < self.cylinder_radius_interval_m[1]
            and 0 < self.cylinder_half_height_interval_m[0]
            < self.cylinder_half_height_interval_m[1]
            and self.minimum_points >= 4
            and self.robust_reweighting_iterations in (1, 2)
        ):
            raise ValueError("invalid dynamic geometry model config")


@dataclass(frozen=True)
class ShapeHypothesisV1:
    geometry_type: str
    center_world: np.ndarray
    center_z_interval_m: Tuple[float, float]
    radius_interval_m: Tuple[float, float]
    half_height_interval_m: Tuple[float, float]
    model_covariance_world: np.ndarray
    fit_residual_m: float
    fit_condition_number: float
    evidence_score: float
    observability: str
    temporal_consistency: float = 0.0
    expiry_frames: int = 2
    runtime_gt_used: bool = False

    def __post_init__(self):
        if self.geometry_type not in {"sphere", "vertical_cylinder"}:
            raise ValueError("unsupported geometry hypothesis")
        center = np.asarray(self.center_world, dtype=np.float64)
        covariance = np.asarray(
            self.model_covariance_world, dtype=np.float64
        )
        if (
            center.shape != (3,) or covariance.shape != (3, 3)
            or not np.isfinite(center).all()
            or not np.isfinite(covariance).all()
        ):
            raise ValueError("invalid hypothesis geometry")
        if self.runtime_gt_used:
            raise ValueError("runtime GT is forbidden")
        object.__setattr__(self, "center_world", center.copy())
        object.__setattr__(
            self, "model_covariance_world", covariance.copy()
        )


@dataclass(frozen=True)
class ShapeGeometryEvaluationV1:
    valid: bool
    timestamp: float
    observation_id: int
    observability: ShapeObservability
    hypotheses: Tuple[ShapeHypothesisV1, ...]
    sphere_score: float
    cylinder_score: float
    absolute_quality_pass: bool
    relative_margin: float
    fallback_reason: str
    runtime_gt_used: bool = False


def _weighted_sphere(points, iterations):
    origin = points.mean(axis=0)
    geometry_scale = max(
        float(np.linalg.norm(np.ptp(points, axis=0))), 1e-6
    )
    normalized = (points-origin)/geometry_scale
    weights = np.ones(len(points))
    for _ in range(iterations):
        matrix = np.column_stack((
            2*normalized, np.ones(len(normalized))
        ))
        target = np.sum(normalized*normalized, axis=1)
        root = np.sqrt(weights)
        solution, *_ = np.linalg.lstsq(
            matrix*root[:, None], target*root, rcond=None
        )
        center_normalized = solution[:3]
        radius_normalized = float(np.sqrt(max(
            solution[3]+center_normalized@center_normalized, 0.
        )))
        center = origin+geometry_scale*center_normalized
        radius = geometry_scale*radius_normalized
        residual = np.linalg.norm(points-center, axis=1)-radius
        robust_scale = max(
            float(np.median(np.abs(residual)))*1.4826, 1e-4
        )
        weights = 1/np.maximum(
            1, np.abs(residual)/(2.5*robust_scale)
        )
    rmse = float(np.sqrt(np.mean(residual**2)))
    condition = float(np.linalg.cond(
        (matrix*root[:, None]).T@(matrix*root[:, None])
    ))
    return center, radius, rmse, condition


def _weighted_circle(points, iterations):
    horizontal = points[:, :2]
    origin = horizontal.mean(axis=0)
    geometry_scale = max(
        float(np.linalg.norm(np.ptp(horizontal, axis=0))), 1e-6
    )
    normalized = (horizontal-origin)/geometry_scale
    weights = np.ones(len(points))
    for _ in range(iterations):
        matrix = np.column_stack((
            2*normalized, np.ones(len(points))
        ))
        target = np.sum(normalized*normalized, axis=1)
        root = np.sqrt(weights)
        solution, *_ = np.linalg.lstsq(
            matrix*root[:, None], target*root, rcond=None
        )
        center_normalized = solution[:2]
        radius_normalized = float(np.sqrt(max(
            solution[2]+center_normalized@center_normalized, 0.
        )))
        center = origin+geometry_scale*center_normalized
        radius = geometry_scale*radius_normalized
        residual = np.linalg.norm(
            horizontal-center, axis=1
        )-radius
        robust_scale = max(
            float(np.median(np.abs(residual)))*1.4826, 1e-4
        )
        weights = 1/np.maximum(
            1, np.abs(residual)/(2.5*robust_scale)
        )
    rmse = float(np.sqrt(np.mean(residual**2)))
    condition = float(np.linalg.cond(
        (matrix*root[:, None]).T@(matrix*root[:, None])
    ))
    return center, radius, rmse, condition


class DynamicObjectGeometryModelV1:
    """G1/G2/G3/G4 bounded analytic geometry model."""

    version = MODEL_VERSION

    def __init__(self, config=None):
        self.config = config or GeometryModelConfigV1()
        self.config.validate()

    def evaluate(self, geometry):
        if not isinstance(geometry, DynamicMeasurementGeometryV2):
            raise TypeError("geometry must be DynamicMeasurementGeometryV2")
        if not geometry.valid or len(geometry.points_world) < self.config.minimum_points:
            return ShapeGeometryEvaluationV1(
                False, geometry.timestamp, geometry.observation_id,
                ShapeObservability.GEOMETRY_INVALID, (), np.inf, np.inf,
                False, 0., "insufficient_or_invalid_geometry",
            )
        points = geometry.points_world
        try:
            sphere = _weighted_sphere(
                points, self.config.robust_reweighting_iterations
            )
            cylinder = _weighted_circle(
                points, self.config.robust_reweighting_iterations
            )
        except np.linalg.LinAlgError:
            return ShapeGeometryEvaluationV1(
                False, geometry.timestamp, geometry.observation_id,
                ShapeObservability.GEOMETRY_INVALID, (), np.inf, np.inf,
                False, 0., "linear_fit_failed",
            )
        sphere_hypothesis = self._sphere_hypothesis(geometry, sphere)
        cylinder_hypothesis = self._cylinder_hypothesis(
            geometry, cylinder
        )
        sphere_quality = bool(
            sphere_hypothesis is not None
            and sphere[2] <= self.config.maximum_sphere_residual_m
            and sphere[3] <= self.config.maximum_condition_number
        )
        cylinder_quality = bool(
            cylinder_hypothesis is not None
            and cylinder[2] <= self.config.maximum_cylinder_residual_m
            and cylinder[3] <= self.config.maximum_condition_number
            and geometry.vertical_extent_m
                >= self.config.cylinder_minimum_vertical_extent_m
            and geometry.silhouette_aspect_ratio is not None
            and geometry.silhouette_aspect_ratio
                >= self.config.cylinder_minimum_silhouette_aspect
        )
        sphere_score = sphere[2]/self.config.maximum_sphere_residual_m
        cylinder_score = cylinder[2]/self.config.maximum_cylinder_residual_m
        if geometry.silhouette_aspect_ratio is not None:
            sphere_score += max(
                0., geometry.silhouette_aspect_ratio
                - self.config.sphere_maximum_silhouette_aspect
            )
            cylinder_score += max(
                0., self.config.cylinder_minimum_silhouette_aspect
                - geometry.silhouette_aspect_ratio
            )
        evidence_weight = 1.
        if geometry.image_border_clipped:
            evidence_weight *= self.config.border_evidence_weight
        if (
            geometry.depth_edge_clipping_fraction is not None
            and geometry.depth_edge_clipping_fraction > .5
        ):
            evidence_weight *= self.config.occlusion_evidence_weight
        best = min(sphere_score, cylinder_score)
        margin = abs(sphere_score-cylinder_score)/max(
            max(sphere_score, cylinder_score), 1e-9
        )
        decisive = (
            margin*evidence_weight >=
            self.config.minimum_relative_fit_margin
        )
        if sphere_quality and decisive and sphere_score < cylinder_score:
            observability = ShapeObservability.SPHERE_OBSERVABLE
            hypotheses = (sphere_hypothesis,)
            reason = ""
        elif cylinder_quality and decisive and cylinder_score < sphere_score:
            observability = ShapeObservability.CYLINDER_OBSERVABLE
            hypotheses = (cylinder_hypothesis,)
            reason = ""
        elif sphere_hypothesis is not None and cylinder_hypothesis is not None:
            observability = ShapeObservability.SHAPE_AMBIGUOUS
            hypotheses = (sphere_hypothesis, cylinder_hypothesis)
            reason = "relative_fit_margin_or_observability_insufficient"
        elif sphere_hypothesis is not None:
            observability = ShapeObservability.SUPPORT_ONLY
            hypotheses = (sphere_hypothesis,)
            reason = "only_bounded_sphere_support_available"
        elif cylinder_hypothesis is not None:
            observability = ShapeObservability.SUPPORT_ONLY
            hypotheses = (cylinder_hypothesis,)
            reason = "only_bounded_cylinder_support_available"
        else:
            observability = ShapeObservability.GEOMETRY_INVALID
            hypotheses = ()
            reason = "no_bounded_hypothesis"
        return ShapeGeometryEvaluationV1(
            bool(hypotheses), geometry.timestamp, geometry.observation_id,
            observability, hypotheses, float(sphere_score),
            float(cylinder_score), bool(sphere_quality or cylinder_quality),
            float(margin), reason,
        )

    def _sphere_hypothesis(self, geometry, fit):
        center, radius, residual, condition = fit
        lower, upper = self.config.sphere_radius_interval_m
        if not np.isfinite(radius) or radius < lower-.05 or radius > upper+.05:
            return None
        radius = float(np.clip(radius, lower, upper))
        uncertainty = max(residual, .015)
        interval = (
            max(lower, radius-2*uncertainty),
            min(upper, radius+2*uncertainty),
        )
        covariance = (
            geometry.cluster_covariance_world
            + np.eye(3)*uncertainty**2
        )
        return ShapeHypothesisV1(
            "sphere", center, (center[2], center[2]), interval,
            interval, covariance, residual, condition,
            1/(1+residual/self.config.maximum_sphere_residual_m),
            "sphere_fit",
        )

    def _cylinder_hypothesis(self, geometry, fit):
        horizontal_center, radius, residual, condition = fit
        lower, upper = self.config.cylinder_radius_interval_m
        if not np.isfinite(radius) or radius < lower-.08 or radius > upper+.08:
            return None
        radius = float(np.clip(radius, lower, upper))
        z_min, z_max = (
            float(geometry.points_world[:, 2].min()),
            float(geometry.points_world[:, 2].max()),
        )
        half_lower, half_upper = (
            self.config.cylinder_half_height_interval_m
        )
        center_interval = (z_max-half_upper, z_min+half_upper)
        if center_interval[0] > center_interval[1]:
            midpoint = .5*(z_min+z_max)
            center_interval = (midpoint-.05, midpoint+.05)
        center_z = .5*(center_interval[0]+center_interval[1])
        uncertainty = max(
            residual, .5*(center_interval[1]-center_interval[0]), .02
        )
        center = np.asarray([
            horizontal_center[0], horizontal_center[1], center_z
        ])
        radius_uncertainty = max(residual*2, .015)
        radius_interval = (
            max(lower, radius-radius_uncertainty),
            min(upper, radius+radius_uncertainty),
        )
        covariance = np.diag([
            max(residual, .02)**2,
            max(residual, .02)**2,
            uncertainty**2,
        ])
        return ShapeHypothesisV1(
            "vertical_cylinder", center, center_interval,
            radius_interval, (half_lower, half_upper), covariance,
            residual, condition,
            1/(1+residual/self.config.maximum_cylinder_residual_m),
            (
                "height_observable"
                if geometry.vertical_extent_m >= 2*half_lower*.8
                else "height_weakly_observable"
            ),
        )


__all__ = [
    "MODEL_VERSION", "ShapeObservability", "GeometryModelConfigV1",
    "ShapeHypothesisV1", "ShapeGeometryEvaluationV1",
    "DynamicObjectGeometryModelV1",
]
