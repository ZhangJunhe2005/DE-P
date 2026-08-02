"""Cached direct geometry updates and hypothesis-preserving risk queries."""

from __future__ import annotations

from dataclasses import dataclass, fields
import time
from typing import Dict, Tuple

import numpy as np

from .dynamic_object_geometry_model_v1 import DynamicObjectGeometryModelV1
from .measurement_geometry_adapter_v1 import MeasurementGeometryAdapterV1
from .measurement_geometry_adapter_v2 import DynamicMeasurementGeometryV2
from .shape_motion_hypothesis_tracker_v1 import TrackedShapeMotionV1
from .support_reachable_occupancy_v1 import point_aabb_signed_distance
from tools.evaluate_dynamic_geometry_risk_v1 import (
    finite_vertical_cylinder_signed_distance, sphere_signed_distance,
)


OCCUPANCY_VERSION = "shape_aware_dynamic_occupancy_v1"


def _copy(value):
    return {field.name: getattr(value, field.name) for field in fields(value)}


class FastGeometryUpdateCacheV1:
    """One closed-form geometry fit per direct observation ID."""

    version = "geometry_direct_update_fast_path_v1"

    def __init__(self, geometry_model=None):
        self.base_adapter = MeasurementGeometryAdapterV1()
        self.geometry_model = geometry_model or DynamicObjectGeometryModelV1()
        self._cache: Dict[Tuple[str, int], Tuple[object, object]] = {}
        self.fit_calls = 0
        self.cache_hits = 0
        self.profile = []

    def clear(self):
        self._cache.clear()

    def update(self, observation, frame, generation):
        key = (str(generation), int(observation.observation_id))
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        started = time.perf_counter()
        base = self.base_adapter.export(observation, frame)
        points = base.points_world
        if base.valid and len(points) >= 4:
            covariance = np.cov(points-points.mean(0), rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            extent = np.ptp(points, axis=0)
            principal = eigenvectors[:, -1]
        else:
            eigenvalues, eigenvectors = np.zeros(3), np.eye(3)
            extent, principal = np.zeros(3), np.asarray([0., 0., 1.])
        u0, v0, u1, v1 = base.pixel_bbox
        width, height = max(u1-u0+1, 1), max(v1-v0+1, 1)
        camera = frame.camera_model
        directions = tuple(
            name for name, hit in (
                ("left", u0 <= 1), ("right", u1 >= camera.width-2),
                ("top", v0 <= 1), ("bottom", v1 >= camera.height-2),
            ) if hit
        )
        geometry = DynamicMeasurementGeometryV2(
            **_copy(base), pca_eigenvalues=eigenvalues,
            pca_eigenvectors=eigenvectors,
            vertical_axis_alignment=float(abs(principal[2])),
            horizontal_radial_extent_m=float(
                np.linalg.norm(
                    points[:, :2]-points[:, :2].mean(0), axis=1
                ).max(initial=0) if len(points) else 0.
            ),
            vertical_extent_m=float(extent[2]),
            point_to_ray_depth_distribution_m=(),
            normal_vertical_abs_mean=None,
            normal_radial_abs_mean=None, local_curvature_proxy=None,
            sphere_algebraic_residual_m=None,
            sphere_condition_number=None,
            cylinder_radial_residual_m=None,
            cylinder_condition_number=None,
            cylinder_vertical_residual_m=None,
            silhouette_aspect_ratio=float(height/width),
            angular_width_rad=float(base.angular_span_rad[0]),
            angular_height_rad=float(base.angular_span_rad[1]),
            border_clipping_directions=directions,
            depth_edge_clipping_fraction=None,
            visible_cap_fraction_proxy=None,
            temporal_geometry_history_id=str(generation),
            geometry_feature_validity={
                "pca": "available" if base.valid else "unavailable",
                "normals": "unavailable_fast_path_not_required",
                "sphere_fit": "computed_once_by_geometry_model",
                "cylinder_fit": "computed_once_by_geometry_model",
            },
        )
        evaluation = self.geometry_model.evaluate(geometry)
        self.fit_calls += 1
        elapsed = (time.perf_counter()-started)*1000
        self.profile.append({
            "observation_id": int(observation.observation_id),
            "point_count": int(len(points)), "elapsed_ms": elapsed,
        })
        self._cache[key] = (geometry, evaluation)
        return geometry, evaluation


@dataclass(frozen=True)
class ReachabilityConfigV1:
    acceleration_bound_mps2: float
    reference_drift_rate_mps: float


def evaluate_shape_aware_risk(
    candidates, times, tracked_states, reachability,
    *, uav_radius_m=.30, required_margin_m=.10,
):
    candidates = np.asarray(candidates, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    rows = []
    for candidate_id, candidate in enumerate(candidates):
        best = float("inf")
        limiting = None
        for state in tracked_states:
            if not isinstance(state, TrackedShapeMotionV1):
                raise TypeError("tracked state type mismatch")
            for item in state.hypotheses:
                motion = item.motion
                age = max(0., state.timestamp-motion.last_direct_timestamp)
                horizon = times+age
                center = (
                    motion.position_world[None, :]
                    + horizon[:, None]*motion.velocity_world[None, :]
                )
                growth = (
                    .5*reachability.acceleration_bound_mps2*horizon**2
                    + reachability.reference_drift_rate_mps*horizon
                )
                geometry = item.geometry_hypothesis
                if item.geometry_type == "sphere":
                    signed = sphere_signed_distance(
                        candidate, center,
                        geometry.radius_interval_m[1],
                    )-growth
                else:
                    radial = geometry.radius_interval_m[1]+growth
                    vertical = (
                        geometry.half_height_interval_m[1]+growth
                    )
                    signed = np.asarray([
                        finite_vertical_cylinder_signed_distance(
                            candidate[index:index+1],
                            center[index:index+1],
                            radial[index], vertical[index],
                        )[0] for index in range(len(times))
                    ])
                clearance = signed-uav_radius_m-required_margin_m
                index = int(np.argmin(clearance))
                if clearance[index] < best:
                    best = float(clearance[index])
                    limiting = {
                        "track_id": state.track_id,
                        "shape": item.geometry_type,
                        "time": float(times[index]),
                    }
        rows.append({
            "candidate_trajectory_id": candidate_id,
            "predicted_minimum_clearance_m": best,
            "would_veto": bool(best < 0),
            "limiting": limiting,
        })
    return {
        "runtime_gt_used": False,
        "hypothesis_preserving": True,
        "candidate_rows": rows,
    }


__all__ = [
    "OCCUPANCY_VERSION", "FastGeometryUpdateCacheV1",
    "ReachabilityConfigV1", "evaluate_shape_aware_risk",
]
