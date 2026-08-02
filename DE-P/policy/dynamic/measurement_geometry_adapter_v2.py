"""Bounded causal shape features layered on frozen measurement geometry v1."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Mapping, Tuple

import numpy as np

from .measurement_geometry_adapter_v1 import (
    DynamicMeasurementGeometryV1, MeasurementGeometryAdapterV1,
)


ADAPTER_VERSION = "measurement_geometry_adapter_v2"


def _copy_v1(value):
    return {item.name: getattr(value, item.name) for item in fields(value)}


def _algebraic_sphere(points):
    origin = points.mean(axis=0)
    geometry_scale = max(
        float(np.linalg.norm(np.ptp(points, axis=0))), 1e-6
    )
    normalized = (points-origin)/geometry_scale
    matrix = np.column_stack((
        2*normalized, np.ones(len(normalized))
    ))
    target = np.sum(normalized*normalized, axis=1)
    solution, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    center_normalized = solution[:3]
    radius_sq = solution[3]+center_normalized@center_normalized
    center = origin+geometry_scale*center_normalized
    radius = geometry_scale*float(np.sqrt(max(radius_sq, 0.)))
    residual = np.linalg.norm(points-center, axis=1)-radius
    return center, radius, float(np.sqrt(np.mean(residual**2))), float(
        np.linalg.cond(matrix.T@matrix)
    )


def _algebraic_vertical_cylinder(points):
    horizontal = points[:, :2]
    origin = horizontal.mean(axis=0)
    geometry_scale = max(
        float(np.linalg.norm(np.ptp(horizontal, axis=0))), 1e-6
    )
    normalized = (horizontal-origin)/geometry_scale
    matrix = np.column_stack((
        2*normalized, np.ones(len(normalized))
    ))
    target = np.sum(normalized*normalized, axis=1)
    solution, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    center_normalized = solution[:2]
    radius_sq = solution[2]+center_normalized@center_normalized
    center = origin+geometry_scale*center_normalized
    radius = geometry_scale*float(np.sqrt(max(radius_sq, 0.)))
    residual = np.linalg.norm(horizontal-center, axis=1)-radius
    return center, radius, float(np.sqrt(np.mean(residual**2))), float(
        np.linalg.cond(matrix.T@matrix)
    )


def _bounded_normals(points, maximum_points=128, neighbors=8):
    if len(points) < neighbors+1:
        return np.empty((0, 3)), "insufficient_points"
    indices = np.linspace(
        0, len(points)-1, min(len(points), maximum_points), dtype=int
    )
    selected = points[indices]
    distances = np.sum(
        (selected[:, None, :]-points[None, :, :])**2, axis=2
    )
    nearest = np.argpartition(
        distances, min(neighbors, len(points)-1), axis=1
    )[:, :neighbors]
    normals = []
    for row in nearest:
        local = points[row]
        covariance = np.cov(local-local.mean(0), rowvar=False)
        values, vectors = np.linalg.eigh(covariance)
        normal = vectors[:, int(np.argmin(values))]
        normals.append(normal/np.linalg.norm(normal))
    return np.asarray(normals), "available"


@dataclass(frozen=True)
class DynamicMeasurementGeometryV2(DynamicMeasurementGeometryV1):
    pca_eigenvalues: np.ndarray = None
    pca_eigenvectors: np.ndarray = None
    vertical_axis_alignment: float = 0.0
    horizontal_radial_extent_m: float = 0.0
    vertical_extent_m: float = 0.0
    point_to_ray_depth_distribution_m: Tuple[float, ...] = ()
    normal_vertical_abs_mean: float | None = None
    normal_radial_abs_mean: float | None = None
    local_curvature_proxy: float | None = None
    sphere_algebraic_residual_m: float | None = None
    sphere_condition_number: float | None = None
    cylinder_radial_residual_m: float | None = None
    cylinder_condition_number: float | None = None
    cylinder_vertical_residual_m: float | None = None
    silhouette_aspect_ratio: float | None = None
    angular_width_rad: float = 0.0
    angular_height_rad: float = 0.0
    border_clipping_directions: Tuple[str, ...] = ()
    depth_edge_clipping_fraction: float | None = None
    visible_cap_fraction_proxy: float | None = None
    temporal_geometry_history_id: str = "unassigned"
    geometry_feature_validity: Mapping[str, str] = None

    def __post_init__(self):
        super().__post_init__()
        eigenvalues = np.asarray(self.pca_eigenvalues, dtype=np.float64)
        eigenvectors = np.asarray(self.pca_eigenvectors, dtype=np.float64)
        if eigenvalues.shape != (3,) or eigenvectors.shape != (3, 3):
            raise ValueError("PCA fields have invalid shape")
        if not (
            np.isfinite(eigenvalues).all()
            and np.isfinite(eigenvectors).all()
        ):
            raise ValueError("PCA fields must be finite")
        object.__setattr__(self, "pca_eigenvalues", eigenvalues.copy())
        object.__setattr__(self, "pca_eigenvectors", eigenvectors.copy())
        if not isinstance(self.geometry_feature_validity, Mapping):
            raise ValueError("geometry_feature_validity must be a mapping")
        object.__setattr__(
            self, "geometry_feature_validity",
            {str(key): str(value)
             for key, value in self.geometry_feature_validity.items()},
        )


class MeasurementGeometryAdapterV2:
    """Export v1 geometry plus bounded analytic shape evidence."""

    version = ADAPTER_VERSION

    def __init__(self):
        self.v1 = MeasurementGeometryAdapterV1()

    def export(self, observation, frame, temporal_geometry_history_id="unassigned"):
        base = self.v1.export(observation, frame)
        if not base.valid or len(base.points_world) < 4:
            return DynamicMeasurementGeometryV2(
                **_copy_v1(base),
                pca_eigenvalues=np.zeros(3),
                pca_eigenvectors=np.eye(3),
                geometry_feature_validity={
                    "pca": "unavailable", "normals": "unavailable",
                    "sphere_fit": "unavailable",
                    "cylinder_fit": "unavailable",
                },
                temporal_geometry_history_id=str(
                    temporal_geometry_history_id
                ),
            )
        points = base.points_world
        covariance = np.cov(points-points.mean(0), rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        principal = eigenvectors[:, -1]
        vertical_alignment = float(abs(principal@[0., 0., 1.]))
        horizontal = np.linalg.norm(
            points[:, :2]-points[:, :2].mean(0), axis=1
        )
        ray = base.cluster_centroid_world-base.camera_position_world
        ray /= max(np.linalg.norm(ray), 1e-12)
        along_ray = (
            points-base.cluster_centroid_world
        )@ray
        normals, normal_status = _bounded_normals(points)
        if len(normals):
            vertical_normal = float(np.mean(np.abs(normals[:, 2])))
            radial_vectors = points[
                np.linspace(0, len(points)-1, len(normals), dtype=int), :2
            ]-points[:, :2].mean(0)
            radial_norm = np.linalg.norm(radial_vectors, axis=1)
            radial_unit = radial_vectors/np.maximum(
                radial_norm[:, None], 1e-12
            )
            radial_normal = float(np.mean(np.abs(
                np.sum(normals[:, :2]*radial_unit, axis=1)
            )))
            local_curvature = float(np.std(
                np.abs(np.sum(normals*ray, axis=1))
            ))
        else:
            vertical_normal = radial_normal = local_curvature = None
        try:
            _, sphere_radius, sphere_residual, sphere_condition = (
                _algebraic_sphere(points)
            )
            sphere_status = "available"
        except np.linalg.LinAlgError:
            sphere_radius = sphere_residual = sphere_condition = None
            sphere_status = "unavailable"
        try:
            _, cylinder_radius, cylinder_residual, cylinder_condition = (
                _algebraic_vertical_cylinder(points)
            )
            cylinder_status = "available"
        except np.linalg.LinAlgError:
            cylinder_radius = cylinder_residual = cylinder_condition = None
            cylinder_status = "unavailable"
        u0, v0, u1, v1 = base.pixel_bbox
        width = max(u1-u0+1, 1)
        height = max(v1-v0+1, 1)
        directions = []
        camera = frame.camera_model
        if u0 <= 1:
            directions.append("left")
        if u1 >= camera.width-2:
            directions.append("right")
        if v0 <= 1:
            directions.append("top")
        if v1 >= camera.height-2:
            directions.append("bottom")
        pixels = base.pixels_uv
        depth = frame.depth_m
        edge_hits = 0
        for u, v in pixels:
            local = depth[
                max(0, v-1):min(camera.height, v+2),
                max(0, u-1):min(camera.width, u+2),
            ]
            valid = local[np.isfinite(local)]
            if len(valid) and np.ptp(valid) > .12:
                edge_hits += 1
        extent = np.ptp(points, axis=0)
        vertical_residual = float(np.std(
            np.minimum(
                np.abs(points[:, 2]-points[:, 2].min()),
                np.abs(points[:, 2]-points[:, 2].max()),
            )
        ))
        visible_cap = (
            float(np.clip(
                max(base.angular_span_rad)
                / max(2*np.arcsin(min(
                    (sphere_radius or .42)
                    / max(np.linalg.norm(
                        base.cluster_centroid_world
                        - base.camera_position_world
                    ), 1e-9), 1.
                )), 1e-9),
                0., 1.,
            )) if sphere_radius is not None else None
        )
        return DynamicMeasurementGeometryV2(
            **_copy_v1(base),
            pca_eigenvalues=eigenvalues,
            pca_eigenvectors=eigenvectors,
            vertical_axis_alignment=vertical_alignment,
            horizontal_radial_extent_m=float(horizontal.max(initial=0)),
            vertical_extent_m=float(extent[2]),
            point_to_ray_depth_distribution_m=tuple(
                float(value) for value in np.percentile(
                    along_ray, [0, 10, 50, 90, 100]
                )
            ),
            normal_vertical_abs_mean=vertical_normal,
            normal_radial_abs_mean=radial_normal,
            local_curvature_proxy=local_curvature,
            sphere_algebraic_residual_m=sphere_residual,
            sphere_condition_number=sphere_condition,
            cylinder_radial_residual_m=cylinder_residual,
            cylinder_condition_number=cylinder_condition,
            cylinder_vertical_residual_m=vertical_residual,
            silhouette_aspect_ratio=float(height/width),
            angular_width_rad=float(base.angular_span_rad[0]),
            angular_height_rad=float(base.angular_span_rad[1]),
            border_clipping_directions=tuple(directions),
            depth_edge_clipping_fraction=float(edge_hits/len(pixels)),
            visible_cap_fraction_proxy=visible_cap,
            temporal_geometry_history_id=str(
                temporal_geometry_history_id
            ),
            geometry_feature_validity={
                "pca": "available",
                "normals": normal_status,
                "sphere_fit": sphere_status,
                "cylinder_fit": cylinder_status,
                "border_clipping": "available",
                "depth_edge_clipping": "available",
            },
        )


__all__ = [
    "ADAPTER_VERSION", "DynamicMeasurementGeometryV2",
    "MeasurementGeometryAdapterV2",
]
