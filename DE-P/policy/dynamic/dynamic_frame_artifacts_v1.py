"""Immutable, frame-scoped artifacts shared by DIRO1 runtime consumers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera_model import make_depth_frame
from .pointcloud import camera_points_to_world
from .types import CameraModel, DepthFrame, Pose


ARTIFACT_VERSION = "dynamic_frame_artifacts_v1"


def _readonly(value):
    array = np.ascontiguousarray(value)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class DynamicFrameArtifactsV1:
    frame_id: int
    timestamp: float
    depth_frame: DepthFrame
    finite_depth_mask: np.ndarray
    valid_depth_mask: np.ndarray
    points_camera: np.ndarray
    points_world: np.ndarray
    rotation_world_from_camera: np.ndarray
    rotation_camera_from_world: np.ndarray
    camera_position_world: np.ndarray
    runtime_gt_used: bool = False

    def __post_init__(self):
        if self.frame_id < 0:
            raise ValueError("frame_id must be non-negative")
        if self.runtime_gt_used:
            raise ValueError("runtime GT is forbidden")
        for name in (
            "finite_depth_mask", "valid_depth_mask", "points_camera",
            "points_world", "rotation_world_from_camera",
            "rotation_camera_from_world", "camera_position_world",
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name)))


class DynamicFrameArtifactBuilderV1:
    """Build each depth/point/world representation exactly once per frame."""

    def __init__(self, camera_model: CameraModel, stride: int):
        self.camera_model = camera_model
        self.stride = int(stride)
        self._last_frame_id = -1
        self.build_count = 0
        self.point_reconstruction_count = 0
        self.world_transform_count = 0

    def build(self, depth, pose: Pose, timestamp: float, frame_id: int):
        if frame_id <= self._last_frame_id:
            raise ValueError("frame_id must increase strictly")
        if not isinstance(pose, Pose):
            raise TypeError("pose must be Pose")
        frame = make_depth_frame(
            depth, self.camera_model, pose, timestamp, self.stride
        )
        points_camera = frame.points_camera
        points_world = camera_points_to_world(points_camera, pose)
        self._last_frame_id = int(frame_id)
        self.build_count += 1
        self.point_reconstruction_count += 1
        self.world_transform_count += 1
        return DynamicFrameArtifactsV1(
            frame_id=int(frame_id), timestamp=float(timestamp),
            depth_frame=frame,
            finite_depth_mask=np.isfinite(frame.depth_m),
            valid_depth_mask=frame.valid_mask,
            points_camera=points_camera, points_world=points_world,
            rotation_world_from_camera=pose.rotation_world_from_camera,
            rotation_camera_from_world=pose.rotation_world_from_camera.T,
            camera_position_world=pose.position_world,
        )


__all__ = [
    "ARTIFACT_VERSION", "DynamicFrameArtifactsV1",
    "DynamicFrameArtifactBuilderV1",
]
