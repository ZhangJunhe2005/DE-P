"""Independent single-stream dynamic perception pipeline."""

from __future__ import annotations

from dataclasses import replace
import numpy as np

from .attention import build_dynamic_attention_with_projection
from .camera_model import make_depth_frame
from .clustering import StandardDBSCANClustering
from .pointcloud import camera_points_to_world, validate_point_cloud_camera
from .track_manager import TrackManager
from .temporal_foreground import TemporalVoxelForeground
from .range_image_foreground import CausalRangeImageForeground
from .types import (
    CameraModel,
    DynamicPerceptionConfig,
    DynamicPerceptionResult,
    Pose,
)


class DynamicPerception:
    """One instance owns one temporal sensor stream; never share across training batches."""

    def __init__(self, config=None, feature_shape=(3, 5), attention_device="cpu"):
        self.config = config or DynamicPerceptionConfig.from_global_config()
        self.config.validate()
        self.feature_shape = (int(feature_shape[0]), int(feature_shape[1]))
        if min(self.feature_shape) <= 0:
            raise ValueError("feature_shape must be positive")
        self.attention_device = attention_device
        self.clustering = StandardDBSCANClustering(self.config)
        self.track_manager = TrackManager(self.config)
        self.foreground = TemporalVoxelForeground(self.config)
        self.range_foreground = CausalRangeImageForeground(self.config)
        self._last_timestamp = None
        self._frame_index = 0
        self._next_observation_id = 0

    def reset(self):
        self.track_manager.reset()
        self.foreground.reset()
        self.range_foreground.reset()
        self._last_timestamp = None
        self._frame_index = 0
        self._next_observation_id = 0

    def update(self, point_cloud_camera, camera_pose_world: Pose, timestamp,
               camera_model: CameraModel) -> DynamicPerceptionResult:
        return self._update_points(
            point_cloud_camera, camera_pose_world, timestamp, camera_model,
            input_kind="pointcloud",
        )

    def update_depth(self, depth, camera_pose_world: Pose, timestamp,
                     camera_model: CameraModel) -> DynamicPerceptionResult:
        """Update from raw sensor depth while retaining its pixel topology."""
        frame = make_depth_frame(
            depth, camera_model, camera_pose_world, timestamp, self.config.depth_stride
        )
        result = self._update_points(
            frame.points_camera, camera_pose_world, timestamp, camera_model,
            input_kind="depth", depth_frame=frame,
        )
        diagnostics = dict(result.diagnostics)
        diagnostics.update({
            "raw_depth_shape": list(frame.depth_m.shape),
            "raw_depth_dtype": str(frame.depth_m.dtype),
            "sampled_pixel_count": len(frame.pixels_uv),
            "pixel_point_identity_preserved": True,
        })
        return DynamicPerceptionResult(
            all_tracks=result.all_tracks,
            confirmed_tracks=result.confirmed_tracks,
            dynamic_tracks=result.dynamic_tracks,
            projected_dynamic_tracks=result.projected_dynamic_tracks,
            attention_map=result.attention_map,
            observations=result.observations,
            diagnostics=diagnostics,
        )

    def _update_points(self, point_cloud_camera, camera_pose_world: Pose, timestamp,
                       camera_model: CameraModel, input_kind="pointcloud",
                       depth_frame=None) -> DynamicPerceptionResult:
        if not isinstance(camera_pose_world, Pose):
            raise TypeError("camera_pose_world must be a Pose with explicit frame direction")
        if not isinstance(camera_model, CameraModel):
            raise TypeError("camera_model must be CameraModel")
        timestamp = float(timestamp)
        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        if abs(camera_pose_world.timestamp - timestamp) > self.config.max_pose_time_offset:
            raise ValueError("camera pose is missing or too stale for this sensor frame")
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            relation = "duplicate" if timestamp == self._last_timestamp else "out-of-order"
            raise ValueError(f"{relation} timestamp is not allowed")

        points_camera = validate_point_cloud_camera(point_cloud_camera, camera_model)
        points_world = camera_points_to_world(points_camera, camera_pose_world)
        if self.config.foreground_mode == "temporal_voxel":
            foreground_mask = self.foreground.extract(
                points_world, timestamp, camera_pose_world.position_world
            )
            clustering_world = points_world[foreground_mask]
            clustering_camera = points_camera[foreground_mask]
            observations, labels = self.clustering.cluster(
                clustering_world, clustering_camera, timestamp
            )
            foreground_diagnostics = self.foreground.last_diagnostics
        elif self.config.foreground_mode == "range_image_hybrid":
            if depth_frame is None:
                raise ValueError(
                    "range_image_hybrid requires update_depth(); pixel topology cannot be guessed"
                )
            # Query historical free space before any current occupied sample is
            # ingested.  Completed foreground pixels are then withheld from the
            # world background update so a moving actor is not immediately
            # promoted to stable background.
            free_mask = self.foreground.previously_free_mask(points_world, timestamp)
            free_seed = np.zeros(depth_frame.depth_m.shape, dtype=bool)
            free_pixels = depth_frame.pixels_uv[free_mask]
            free_seed[free_pixels[:, 1], free_pixels[:, 0]] = True
            observations, labels = self.range_foreground.extract(depth_frame, free_seed)
            foreground_mask = labels >= 0
            # The range image supplies dense geometry.  The complementary
            # world free-space memory therefore uses a deterministic sparse
            # ray lattice; this bounds CPU/memory without expanding seeds.
            background_points = points_world[~foreground_mask][::8]
            self.foreground.extract(
                background_points, timestamp,
                camera_pose_world.position_world,
            )
            foreground_diagnostics = dict(self.range_foreground.last_diagnostics)
            foreground_diagnostics["free_space_source"] = self.foreground.last_diagnostics
        else:
            foreground_mask = np.ones(len(points_world), dtype=bool)
            clustering_world, clustering_camera = points_world, points_camera
            observations, labels = self.clustering.cluster(
                clustering_world, clustering_camera, timestamp
            )
            foreground_diagnostics = {"mode": "none", "foreground_points": len(points_world)}
        identified = []
        for observation in observations:
            identified.append(replace(
                observation,
                observation_id=self._next_observation_id,
                frame_index=self._frame_index,
            ))
            self._next_observation_id += 1
        observations = tuple(identified)
        all_tracks = self.track_manager.update(
            observations, timestamp, depth_frame=depth_frame
        )
        confirmed = tuple(track for track in all_tracks if track.is_confirmed)
        dynamic = tuple(
            track for track in confirmed
            if track.is_dynamic and track.attention_authorized
        )
        attention, projected = build_dynamic_attention_with_projection(
            dynamic,
            camera_pose_world,
            camera_model,
            self.feature_shape,
            self.config,
            self.attention_device,
        )
        self._last_timestamp = timestamp
        self._frame_index += 1
        diagnostics = {
            "coordinate_convention": "camera optical +X right, +Y down, +Z forward; tracking in world",
            "clustering_algorithm": self.clustering.algorithm_name,
            "input_point_count": int(np.asarray(point_cloud_camera).shape[0]),
            "valid_point_count": len(points_camera),
            "foreground_point_count": int(foreground_mask.sum()),
            "foreground": foreground_diagnostics,
            "cluster_count": len(observations),
            "noise_point_count": int(np.sum(labels == -1)),
            "track_manager": self.track_manager.last_diagnostics,
            "batch_semantics": "one DynamicPerception instance per temporal stream",
            "input_kind": input_kind,
        }
        return DynamicPerceptionResult(
            all_tracks=all_tracks,
            confirmed_tracks=confirmed,
            dynamic_tracks=dynamic,
            projected_dynamic_tracks=projected,
            attention_map=attention,
            observations=observations,
            diagnostics=diagnostics,
        )
