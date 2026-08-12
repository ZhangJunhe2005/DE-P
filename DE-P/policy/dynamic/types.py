"""Strongly typed public data structures for dynamic perception."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Tuple

import numpy as np
import torch


def _finite_array(name: str, value, shape) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


@dataclass(frozen=True)
class CameraModel:
    """Pinhole camera using optical axes +X right, +Y down, +Z forward."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale: float
    min_depth: float
    max_depth: float

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera width and height must be positive")
        values = (self.fx, self.fy, self.cx, self.cy, self.depth_scale,
                  self.min_depth, self.max_depth)
        if not np.isfinite(values).all():
            raise ValueError("camera intrinsics and depth limits must be finite")
        if self.fx <= 0 or self.fy <= 0 or self.depth_scale <= 0:
            raise ValueError("fx, fy, and depth_scale must be positive")
        if self.min_depth < 0 or self.max_depth <= self.min_depth:
            raise ValueError("camera depth range is invalid")


@dataclass(frozen=True)
class Pose:
    """Camera pose: p_world = R_world_from_camera @ p_camera + position_world."""

    position_world: np.ndarray
    rotation_world_from_camera: np.ndarray
    timestamp: float

    def __post_init__(self):
        object.__setattr__(self, "position_world", _finite_array(
            "position_world", self.position_world, (3,)
        ))
        rotation = _finite_array(
            "rotation_world_from_camera", self.rotation_world_from_camera, (3, 3)
        )
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
            raise ValueError("rotation_world_from_camera must be orthonormal")
        if np.linalg.det(rotation) < 0.999999 or np.linalg.det(rotation) > 1.000001:
            raise ValueError("rotation_world_from_camera must be a proper rotation")
        if not np.isfinite(self.timestamp):
            raise ValueError("pose timestamp must be finite")
        object.__setattr__(self, "rotation_world_from_camera", rotation)


@dataclass(frozen=True)
class DepthFrame:
    """Raw metric depth plus a reversible sampled optical-camera point set.

    ``pixels_uv[i]`` is the source pixel of ``points_camera[i]``.  Geometry is
    always computed from the raw sensor image; resized CNN inputs must never be
    placed in this structure.
    """

    depth_m: np.ndarray
    valid_mask: np.ndarray
    points_camera: np.ndarray
    pixels_uv: np.ndarray
    camera_model: CameraModel
    camera_pose_world: Pose
    timestamp: float
    stride: int

    def __post_init__(self):
        if not isinstance(self.camera_model, CameraModel):
            raise TypeError("camera_model must be CameraModel")
        if not isinstance(self.camera_pose_world, Pose):
            raise TypeError("camera_pose_world must be Pose")
        expected = (self.camera_model.height, self.camera_model.width)
        depth = np.asarray(self.depth_m)
        valid = np.asarray(self.valid_mask)
        points = np.asarray(self.points_camera)
        pixels = np.asarray(self.pixels_uv)
        if depth.shape != expected or depth.dtype != np.float32:
            raise ValueError(f"depth_m must be float32 with shape {expected}")
        if valid.shape != expected or valid.dtype != np.bool_:
            raise ValueError(f"valid_mask must be bool with shape {expected}")
        if points.ndim != 2 or points.shape[1:] != (3,) or points.dtype != np.float32:
            raise ValueError("points_camera must be float32 [N,3]")
        if pixels.shape != (len(points), 2) or pixels.dtype != np.int32:
            raise ValueError("pixels_uv must be int32 [N,2] and match points_camera")
        if not np.isfinite(points).all():
            raise ValueError("points_camera must be finite")
        if len(pixels) and (np.any(pixels[:, 0] < 0)
                            or np.any(pixels[:, 0] >= self.camera_model.width)
                            or np.any(pixels[:, 1] < 0)
                            or np.any(pixels[:, 1] >= self.camera_model.height)):
            raise ValueError("pixels_uv lies outside the raw depth image")
        if not isinstance(self.stride, int) or self.stride <= 0:
            raise ValueError("stride must be a positive integer")
        timestamp = float(self.timestamp)
        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        object.__setattr__(self, "depth_m", np.ascontiguousarray(depth.copy()))
        object.__setattr__(self, "valid_mask", np.ascontiguousarray(valid.copy()))
        object.__setattr__(self, "points_camera", np.ascontiguousarray(points.copy()))
        object.__setattr__(self, "pixels_uv", np.ascontiguousarray(pixels.copy()))
        object.__setattr__(self, "timestamp", timestamp)


@dataclass(frozen=True)
class ClusterObservation:
    temporary_cluster_id: int
    centroid_world: np.ndarray
    centroid_camera: np.ndarray
    point_count: int
    bounding_box_world: np.ndarray
    position_covariance: np.ndarray
    timestamp: float
    extent: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    covariance_eigenvalues: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    shape_ratios: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    voxel_signature: Tuple[Tuple[int, int, int], ...] = ()
    foreground_support: float = 1.0
    observation_id: int = -1
    frame_index: int = -1
    pixel_indices: Tuple[int, ...] = ()
    pixel_bbox: Tuple[int, int, int, int] = (-1, -1, -1, -1)
    seed_pixel_count: int = 0
    range_seed_count: int = 0
    free_space_seed_count: int = 0
    component_pixel_count: int = 0
    history_support: float = 0.0
    direct_image_evidence: bool = True
    camera_inside_fraction: float = 0.0
    depth_consistency: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "centroid_world", _finite_array(
            "centroid_world", self.centroid_world, (3,)
        ))
        object.__setattr__(self, "centroid_camera", _finite_array(
            "centroid_camera", self.centroid_camera, (3,)
        ))
        object.__setattr__(self, "bounding_box_world", _finite_array(
            "bounding_box_world", self.bounding_box_world, (2, 3)
        ))
        covariance = _finite_array(
            "position_covariance", self.position_covariance, (3, 3)
        )
        if self.point_count <= 0:
            raise ValueError("point_count must be positive")
        if not np.allclose(covariance, covariance.T, atol=1e-9):
            raise ValueError("position_covariance must be symmetric")
        if not np.isfinite(self.timestamp):
            raise ValueError("observation timestamp must be finite")
        object.__setattr__(self, "position_covariance", covariance)
        object.__setattr__(self, "extent", _finite_array("extent", self.extent, (3,)))
        object.__setattr__(self, "covariance_eigenvalues", _finite_array(
            "covariance_eigenvalues", self.covariance_eigenvalues, (3,)
        ))
        object.__setattr__(self, "shape_ratios", _finite_array(
            "shape_ratios", self.shape_ratios, (2,)
        ))
        if not 0.0 <= float(self.foreground_support) <= 1.0:
            raise ValueError("foreground_support must be in [0,1]")
        object.__setattr__(self, "voxel_signature", tuple(tuple(int(x) for x in row)
                                                           for row in self.voxel_signature))
        if self.observation_id < -1 or self.frame_index < -1:
            raise ValueError("observation_id and frame_index must be >= -1")
        object.__setattr__(self, "pixel_indices", tuple(int(x) for x in self.pixel_indices))
        if len(self.pixel_bbox) != 4:
            raise ValueError("pixel_bbox must be (u_min,v_min,u_max,v_max)")
        object.__setattr__(self, "pixel_bbox", tuple(int(x) for x in self.pixel_bbox))
        for name in ("seed_pixel_count", "range_seed_count", "free_space_seed_count",
                     "component_pixel_count"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("history_support", "camera_inside_fraction", "depth_consistency"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must be finite and in [0,1]")


@dataclass(frozen=True)
class DynamicTrack:
    track_id: int
    position_world: np.ndarray
    velocity_world: np.ndarray
    state_covariance: np.ndarray
    age: int
    hit_count: int
    missed_count: int
    is_confirmed: bool
    is_dynamic: bool
    timestamp: float
    dynamic_reason: str
    confidence: float = 1.0
    confidence_components: Mapping[str, float] = field(default_factory=dict)
    birth_observation_id: int = -1
    birth_frame: int = -1
    last_observation_id: int = -1
    last_direct_observation_frame: int = -1
    direct_observation_count: int = 0
    consecutive_direct_hits: int = 0
    prediction_only_age: int = 0
    ever_directly_observed: bool = False
    last_pixel_bbox: Tuple[int, int, int, int] = (-1, -1, -1, -1)
    last_pixel_mask_signature: str = ""
    attention_authorized: bool = False
    visibility_state: str = "unknown"
    ever_confirmed_dynamic: bool = False
    last_direct_observation_timestamp: float | None = None
    last_observed_extent: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self):
        object.__setattr__(self, "position_world", _finite_array(
            "position_world", self.position_world, (3,)
        ))
        object.__setattr__(self, "velocity_world", _finite_array(
            "velocity_world", self.velocity_world, (3,)
        ))
        covariance = _finite_array(
            "state_covariance", self.state_covariance, (6, 6)
        )
        if not np.allclose(covariance, covariance.T, atol=1e-8):
            raise ValueError("state_covariance must be symmetric")
        object.__setattr__(self, "state_covariance", covariance)
        if not np.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("track confidence must be finite and in [0,1]")
        components = {str(key): float(value) for key, value in self.confidence_components.items()}
        if any(not np.isfinite(value) or value < 0 or value > 1
               for value in components.values()):
            raise ValueError("track confidence components must be finite in [0,1]")
        object.__setattr__(self, "confidence_components", components)
        if self.direct_observation_count < 0 or self.consecutive_direct_hits < 0:
            raise ValueError("direct observation counters must be non-negative")
        if self.prediction_only_age < 0:
            raise ValueError("prediction_only_age must be non-negative")
        if len(self.last_pixel_bbox) != 4:
            raise ValueError("last_pixel_bbox must contain four integers")
        object.__setattr__(self, "last_pixel_bbox",
                           tuple(int(x) for x in self.last_pixel_bbox))
        if (self.last_direct_observation_timestamp is not None
                and not np.isfinite(self.last_direct_observation_timestamp)):
            raise ValueError(
                "last_direct_observation_timestamp must be finite or None"
            )
        extent = tuple(float(value) for value in self.last_observed_extent)
        if len(extent) != 3 or not np.isfinite(extent).all() \
                or min(extent) < 0.0:
            raise ValueError("last_observed_extent must be finite non-negative XYZ")
        object.__setattr__(self, "last_observed_extent", extent)


@dataclass(frozen=True)
class ProjectedDynamicTrack:
    track_id: int
    pixel: np.ndarray
    feature_coordinate: np.ndarray
    depth_camera: float
    attention_weight: float


@dataclass(frozen=True)
class DynamicPerceptionResult:
    all_tracks: Tuple[DynamicTrack, ...]
    confirmed_tracks: Tuple[DynamicTrack, ...]
    dynamic_tracks: Tuple[DynamicTrack, ...]
    projected_dynamic_tracks: Tuple[ProjectedDynamicTrack, ...]
    attention_map: torch.Tensor
    observations: Tuple[ClusterObservation, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DynamicPerceptionConfig:
    enabled: bool
    source: str
    use_attention: bool
    fallback_to_static: bool
    publish_debug: bool
    max_pose_time_offset: float
    attention_alpha: float
    depth_stride: int
    depth_topic: str
    pointcloud_topic: str
    pointcloud_body_topic: str
    pointcloud_optical_topic: str
    pointcloud_world_topic: str
    camera_info_topic: str
    odom_topic: str
    use_camera_info: bool
    use_tf_extrinsics: bool
    tf_lookup_timeout: float
    allow_empty_sensor_frame: bool
    camera_frame_id: str
    body_frame_id: str
    world_frame_id: str
    sync_queue_size: int
    sync_slop: float
    camera_width: int
    camera_height: int
    camera_fx: float
    camera_fy: float
    camera_cx: float
    camera_cy: float
    camera_depth_scale: float
    camera_min_depth: float
    camera_max_depth: float
    camera_position_body: Tuple[float, float, float]
    camera_rotation_body_from_camera: Tuple[Tuple[float, float, float], ...]
    cluster_eps: float
    cluster_min_samples: int
    cluster_max_iterations: int
    min_dt: float
    max_dt: float
    process_noise_acceleration: float
    measurement_noise: float
    initial_position_variance: float
    initial_velocity_variance: float
    association_distance_threshold: float
    association_mahalanobis_threshold: float
    max_missed_frames: int
    min_confirmed_hits: int
    dynamic_enter_speed: float
    dynamic_exit_speed: float
    dynamic_min_confirmed_hits: int
    dynamic_max_velocity_std: float
    dynamic_min_cluster_points: int
    dynamic_min_cluster_extent: float
    dynamic_max_cluster_extent: float
    dynamic_max_cluster_min_extent: float
    attention_sigma: float
    attention_max: float
    attention_distance_scale: float
    attention_uncertainty_scale: float
    foreground_mode: str
    background_voxel_size: float
    background_match_distance: float
    background_min_persistence: int
    background_max_age: int
    background_max_voxels: int
    foreground_min_persistence: int
    foreground_reset_gap: float
    background_update_speed_limit: float
    centroid_trim_fraction: float
    centroid_extent_covariance_scale: float
    association_point_count_ratio_max: float
    association_extent_ratio_max: float
    association_bbox_margin: float
    association_shape_weight: float
    association_size_weight: float
    association_overlap_weight: float
    physically_plausible_speed_max: float
    physically_plausible_acceleration_max: float
    velocity_covariance_floor: float
    motion_consistency_frames: int
    track_confidence_threshold: float
    confidence_missed_decay: float
    range_history_frames: int
    range_min_history_support: int
    range_abs_residual_threshold: float
    range_rel_residual_threshold: float
    range_static_consistency_threshold: float
    range_edge_guard_threshold: float
    range_min_seed_pixels: int
    range_min_seed_fraction: float
    range_component_connectivity: int
    range_growth_abs_depth: float
    range_growth_rel_depth: float
    range_growth_max_3d_neighbor_distance: float
    range_max_component_pixels: int
    range_max_component_depth_span: float
    range_component_min_persistence: int
    range_history_max_age: float
    range_reset_gap: float

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]):
        unknown = set(values) - set(cls.__dataclass_fields__)
        missing = set(cls.__dataclass_fields__) - set(values)
        if unknown or missing:
            raise ValueError(f"dynamic_perception config unknown={sorted(unknown)}, missing={sorted(missing)}")
        config = cls(**values)
        config.validate()
        return config

    @classmethod
    def from_global_config(cls):
        from config.config import cfg

        return cls.from_mapping(cfg["dynamic_perception"])

    def validate(self):
        for name in ("enabled", "use_attention", "fallback_to_static", "publish_debug",
                     "use_camera_info", "use_tf_extrinsics", "allow_empty_sensor_frame"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if self.source not in {"depth", "pointcloud"}:
            raise ValueError("source must be 'depth' or 'pointcloud'")
        if self.foreground_mode not in {"none", "temporal_voxel", "range_image_hybrid"}:
            raise ValueError("foreground_mode must be none, temporal_voxel, or range_image_hybrid")
        for name in ("depth_topic", "pointcloud_topic", "pointcloud_body_topic",
                     "pointcloud_optical_topic", "pointcloud_world_topic",
                     "camera_info_topic", "odom_topic", "camera_frame_id",
                     "body_frame_id", "world_frame_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        positive = (
            "max_pose_time_offset", "sync_queue_size", "sync_slop", "tf_lookup_timeout", "camera_width",
            "camera_height", "camera_fx", "camera_fy", "camera_depth_scale",
            "camera_max_depth", "depth_stride",
            "cluster_eps", "cluster_min_samples", "cluster_max_iterations", "min_dt",
            "max_dt", "process_noise_acceleration", "measurement_noise",
            "initial_position_variance", "initial_velocity_variance",
            "association_distance_threshold", "association_mahalanobis_threshold",
            "min_confirmed_hits", "dynamic_enter_speed", "dynamic_min_confirmed_hits",
            "dynamic_max_velocity_std", "dynamic_min_cluster_points",
            "dynamic_min_cluster_extent", "dynamic_max_cluster_extent", "attention_sigma",
            "dynamic_max_cluster_min_extent",
            "attention_max", "attention_distance_scale", "attention_uncertainty_scale",
            "background_voxel_size", "background_match_distance",
            "background_min_persistence", "background_max_age", "background_max_voxels",
            "foreground_min_persistence", "foreground_reset_gap",
            "background_update_speed_limit", "centroid_extent_covariance_scale",
            "association_point_count_ratio_max", "association_extent_ratio_max",
            "association_bbox_margin", "physically_plausible_speed_max",
            "physically_plausible_acceleration_max", "velocity_covariance_floor",
            "motion_consistency_frames", "track_confidence_threshold",
            "confidence_missed_decay",
            "range_history_frames", "range_min_history_support",
            "range_abs_residual_threshold", "range_rel_residual_threshold",
            "range_static_consistency_threshold", "range_edge_guard_threshold",
            "range_min_seed_pixels", "range_min_seed_fraction",
            "range_growth_abs_depth", "range_growth_rel_depth",
            "range_growth_max_3d_neighbor_distance", "range_max_component_pixels",
            "range_max_component_depth_span", "range_component_min_persistence",
            "range_history_max_age", "range_reset_gap",
        )
        for name in positive:
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_missed_frames < 0:
            raise ValueError("max_missed_frames must be non-negative")
        if not np.isfinite(self.attention_alpha) or self.attention_alpha < 0 or self.attention_alpha > 1:
            raise ValueError("attention_alpha must be finite and in [0,1]")
        if self.camera_min_depth < 0 or self.camera_max_depth <= self.camera_min_depth:
            raise ValueError("camera depth range is invalid")
        _finite_array("camera_position_body", self.camera_position_body, (3,))
        rotation = _finite_array(
            "camera_rotation_body_from_camera", self.camera_rotation_body_from_camera, (3, 3)
        )
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(
                np.linalg.det(rotation), 1.0, atol=1e-6):
            raise ValueError("camera_rotation_body_from_camera must be a proper rotation")
        if self.max_dt < self.min_dt:
            raise ValueError("max_dt must be >= min_dt")
        if self.dynamic_exit_speed < 0 or self.dynamic_enter_speed <= self.dynamic_exit_speed:
            raise ValueError("dynamic_enter_speed must be greater than dynamic_exit_speed >= 0")
        if self.dynamic_max_cluster_extent <= self.dynamic_min_cluster_extent:
            raise ValueError("dynamic cluster extent bounds are invalid")
        if self.dynamic_max_cluster_min_extent <= self.dynamic_min_cluster_extent:
            raise ValueError("dynamic cluster minimum-extent bounds are invalid")
        if self.attention_max > 1.0:
            raise ValueError("attention_max must be <= 1")
        for name in ("centroid_trim_fraction", "association_shape_weight",
                     "association_size_weight", "association_overlap_weight"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.centroid_trim_fraction >= 0.5:
            raise ValueError("centroid_trim_fraction must be below 0.5")
        if self.track_confidence_threshold > 1 or self.confidence_missed_decay > 1:
            raise ValueError("confidence values must be <= 1")
        if self.range_min_history_support > self.range_history_frames:
            raise ValueError("range_min_history_support exceeds bounded history")
        if self.range_component_connectivity not in {4, 8}:
            raise ValueError("range_component_connectivity must be 4 or 8")
        if self.range_min_seed_fraction > 1:
            raise ValueError("range_min_seed_fraction must be <= 1")
