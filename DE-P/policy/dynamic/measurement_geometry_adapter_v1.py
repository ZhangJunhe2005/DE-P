"""Runtime-safe export of current-frame dynamic component geometry.

This module is deliberately separate from the legacy measurement and tracking
path.  It reconstructs only geometry that is already present in the current
raw depth frame and a ``ClusterObservation``.  Ground-truth identity, masks,
centres, radii and future frames are not accepted by the public interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .pointcloud import camera_points_to_world
from .types import ClusterObservation, DepthFrame


ADAPTER_VERSION = "measurement_geometry_adapter_v1"
MEASUREMENT_REFERENCE = "visible_surface_cluster_centroid"
FORBIDDEN_RUNTIME_FIELDS = frozenset({
    "gt_actor_id", "gt_actor_center", "gt_actor_radius", "gt_mask",
    "actor_owner", "future_points", "future_depth",
})


def _array(value, shape, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result.copy()


@dataclass(frozen=True)
class DynamicMeasurementGeometryV1:
    """A causally exported component with explicit reference semantics."""

    timestamp: float
    observation_id: int
    measurement_reference: str
    component_provenance: str
    points_camera: np.ndarray
    points_world: np.ndarray
    pixels_uv: np.ndarray
    depth_values_m: np.ndarray
    camera_position_world: np.ndarray
    rotation_world_from_camera: np.ndarray
    cluster_centroid_camera: np.ndarray
    cluster_centroid_world: np.ndarray
    cluster_covariance_world: np.ndarray
    point_count: int
    depth_min_m: float
    depth_max_m: float
    depth_span_m: float
    angular_span_rad: Tuple[float, float]
    pixel_bbox: Tuple[int, int, int, int]
    image_border_distance_px: int
    image_border_clipped: bool
    visible_support_extent_world: np.ndarray
    valid: bool
    failure_reason: str = ""

    def __post_init__(self):
        points_camera = np.asarray(self.points_camera, dtype=np.float64)
        points_world = np.asarray(self.points_world, dtype=np.float64)
        pixels = np.asarray(self.pixels_uv, dtype=np.int32)
        depths = np.asarray(self.depth_values_m, dtype=np.float64)
        if points_camera.ndim != 2 or points_camera.shape[1:] != (3,):
            raise ValueError("points_camera must be [N,3]")
        if points_world.shape != points_camera.shape:
            raise ValueError("points_world must match points_camera")
        if pixels.shape != (len(points_camera), 2):
            raise ValueError("pixels_uv must be [N,2]")
        if depths.shape != (len(points_camera),):
            raise ValueError("depth_values_m must be [N]")
        if not (
            np.isfinite(points_camera).all()
            and np.isfinite(points_world).all()
            and np.isfinite(depths).all()
        ):
            raise ValueError("measurement geometry must be finite")
        if self.measurement_reference != MEASUREMENT_REFERENCE:
            raise ValueError("unexpected measurement reference")
        if int(self.point_count) != len(points_camera):
            raise ValueError("point_count does not match geometry")
        if len(self.pixel_bbox) != 4 or len(self.angular_span_rad) != 2:
            raise ValueError("invalid pixel/angular extent")
        object.__setattr__(self, "points_camera", points_camera.copy())
        object.__setattr__(self, "points_world", points_world.copy())
        object.__setattr__(self, "pixels_uv", pixels.copy())
        object.__setattr__(self, "depth_values_m", depths.copy())
        object.__setattr__(self, "camera_position_world", _array(
            self.camera_position_world, (3,), "camera_position_world"
        ))
        object.__setattr__(self, "rotation_world_from_camera", _array(
            self.rotation_world_from_camera, (3, 3),
            "rotation_world_from_camera",
        ))
        object.__setattr__(self, "cluster_centroid_camera", _array(
            self.cluster_centroid_camera, (3,), "cluster_centroid_camera"
        ))
        object.__setattr__(self, "cluster_centroid_world", _array(
            self.cluster_centroid_world, (3,), "cluster_centroid_world"
        ))
        object.__setattr__(self, "cluster_covariance_world", _array(
            self.cluster_covariance_world, (3, 3),
            "cluster_covariance_world",
        ))
        object.__setattr__(self, "visible_support_extent_world", _array(
            self.visible_support_extent_world, (3,),
            "visible_support_extent_world",
        ))


class MeasurementGeometryAdapterV1:
    """Export current component pixels/points without changing legacy output."""

    version = ADAPTER_VERSION

    def export(
        self, observation: ClusterObservation, frame: DepthFrame
    ) -> DynamicMeasurementGeometryV1:
        if not isinstance(observation, ClusterObservation):
            raise TypeError("observation must be ClusterObservation")
        if not isinstance(frame, DepthFrame):
            raise TypeError("frame must be DepthFrame")
        if abs(float(observation.timestamp) - frame.timestamp) > 1e-9:
            raise ValueError("observation and depth frame timestamps differ")
        width, height = frame.camera_model.width, frame.camera_model.height
        flat = np.asarray(observation.pixel_indices, dtype=np.int64)
        if not len(flat):
            return self._invalid(observation, frame, "component_pixels_unavailable")
        if np.any(flat < 0) or np.any(flat >= width * height):
            raise ValueError("component pixel index outside current depth frame")
        v, u = np.divmod(flat, width)
        pixels = np.stack((u, v), axis=1).astype(np.int32)
        depths = frame.depth_m[v, u].astype(np.float64)
        valid = (
            np.isfinite(depths)
            & (depths >= frame.camera_model.min_depth)
            & (depths < frame.camera_model.max_depth)
        )
        pixels, depths = pixels[valid], depths[valid]
        if not len(depths):
            return self._invalid(observation, frame, "no_valid_component_depth")
        camera = frame.camera_model
        points_camera = np.stack((
            (pixels[:, 0] - camera.cx) * depths / camera.fx,
            (pixels[:, 1] - camera.cy) * depths / camera.fy,
            depths,
        ), axis=1)
        points_world = camera_points_to_world(
            points_camera, frame.camera_pose_world
        )
        ray_x = np.arctan2(points_camera[:, 0], points_camera[:, 2])
        ray_y = np.arctan2(points_camera[:, 1], points_camera[:, 2])
        bbox = (
            int(pixels[:, 0].min()), int(pixels[:, 1].min()),
            int(pixels[:, 0].max()), int(pixels[:, 1].max()),
        )
        border = int(min(
            bbox[0], bbox[1], width - 1 - bbox[2], height - 1 - bbox[3]
        ))
        return DynamicMeasurementGeometryV1(
            timestamp=frame.timestamp,
            observation_id=int(observation.observation_id),
            measurement_reference=MEASUREMENT_REFERENCE,
            component_provenance="current_causal_depth_component",
            points_camera=points_camera,
            points_world=points_world,
            pixels_uv=pixels,
            depth_values_m=depths,
            camera_position_world=frame.camera_pose_world.position_world,
            rotation_world_from_camera=(
                frame.camera_pose_world.rotation_world_from_camera
            ),
            cluster_centroid_camera=observation.centroid_camera,
            cluster_centroid_world=observation.centroid_world,
            cluster_covariance_world=observation.position_covariance,
            point_count=len(points_world),
            depth_min_m=float(depths.min()),
            depth_max_m=float(depths.max()),
            depth_span_m=float(np.ptp(depths)),
            angular_span_rad=(
                float(np.ptp(ray_x)), float(np.ptp(ray_y))
            ),
            pixel_bbox=bbox,
            image_border_distance_px=border,
            image_border_clipped=border <= 1,
            visible_support_extent_world=np.ptp(points_world, axis=0),
            valid=True,
        )

    @staticmethod
    def _invalid(observation, frame, reason):
        empty_points = np.empty((0, 3), dtype=np.float64)
        return DynamicMeasurementGeometryV1(
            timestamp=frame.timestamp,
            observation_id=int(observation.observation_id),
            measurement_reference=MEASUREMENT_REFERENCE,
            component_provenance="geometry_unavailable",
            points_camera=empty_points,
            points_world=empty_points,
            pixels_uv=np.empty((0, 2), dtype=np.int32),
            depth_values_m=np.empty((0,), dtype=np.float64),
            camera_position_world=frame.camera_pose_world.position_world,
            rotation_world_from_camera=(
                frame.camera_pose_world.rotation_world_from_camera
            ),
            cluster_centroid_camera=observation.centroid_camera,
            cluster_centroid_world=observation.centroid_world,
            cluster_covariance_world=observation.position_covariance,
            point_count=0,
            depth_min_m=0.0,
            depth_max_m=0.0,
            depth_span_m=0.0,
            angular_span_rad=(0.0, 0.0),
            pixel_bbox=(-1, -1, -1, -1),
            image_border_distance_px=-1,
            image_border_clipped=True,
            visible_support_extent_world=np.zeros(3),
            valid=False,
            failure_reason=str(reason),
        )


__all__ = [
    "ADAPTER_VERSION", "MEASUREMENT_REFERENCE",
    "FORBIDDEN_RUNTIME_FIELDS", "DynamicMeasurementGeometryV1",
    "MeasurementGeometryAdapterV1",
]
