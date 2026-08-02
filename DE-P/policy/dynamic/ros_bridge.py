"""ROS-message-shaped conversions kept free of rospy for offline testing."""

from __future__ import annotations

import numpy as np

from .types import CameraModel, DynamicPerceptionConfig, Pose


def stamp_to_seconds(stamp) -> float:
    if hasattr(stamp, "to_sec"):
        value = float(stamp.to_sec())
    else:
        value = float(stamp.secs) + float(stamp.nsecs) * 1e-9
    if not np.isfinite(value):
        raise ValueError("message timestamp must be finite")
    return value


def message_timestamp(message) -> float:
    return stamp_to_seconds(message.header.stamp)


def quaternion_to_rotation(x, y, z, w):
    quaternion = np.asarray([x, y, z, w], dtype=np.float64)
    if not np.isfinite(quaternion).all():
        raise ValueError("odometry quaternion must be finite")
    norm = np.linalg.norm(quaternion)
    if norm <= 1e-12:
        raise ValueError("odometry quaternion must be non-zero")
    x, y, z, w = quaternion / norm
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def camera_pose_from_odometry(odometry, config: DynamicPerceptionConfig) -> Pose:
    """Compose world<-body odometry with the configured body<-camera extrinsic."""
    frame_id = str(odometry.header.frame_id).lstrip("/")
    expected = config.world_frame_id.lstrip("/")
    if frame_id != expected:
        raise ValueError(f"odometry frame_id {frame_id!r} does not match world frame {expected!r}")
    pose = odometry.pose.pose
    rotation_world_from_body = quaternion_to_rotation(
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
    )
    position_world_body = np.asarray(
        [pose.position.x, pose.position.y, pose.position.z], dtype=np.float64
    )
    rotation_body_from_camera = np.asarray(
        config.camera_rotation_body_from_camera, dtype=np.float64
    )
    position_body_camera = np.asarray(config.camera_position_body, dtype=np.float64)
    return Pose(
        position_world=position_world_body + rotation_world_from_body @ position_body_camera,
        rotation_world_from_camera=rotation_world_from_body @ rotation_body_from_camera,
        timestamp=message_timestamp(odometry),
    )


def body_points_to_optical(points_body, config: DynamicPerceptionConfig):
    """Transform row-vector body-FLU points to optical-RDF coordinates."""
    points = np.asarray(points_body, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points_body must be finite [N,3]")
    rotation_body_from_camera = np.asarray(
        config.camera_rotation_body_from_camera, dtype=np.float64
    )
    translation_body_from_camera = np.asarray(config.camera_position_body, dtype=np.float64)
    return (points - translation_body_from_camera) @ rotation_body_from_camera


def optical_points_to_body(points_optical, config: DynamicPerceptionConfig):
    """Transform row-vector optical-RDF points to body-FLU coordinates."""
    points = np.asarray(points_optical, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points_optical must be finite [N,3]")
    rotation_body_from_camera = np.asarray(
        config.camera_rotation_body_from_camera, dtype=np.float64
    )
    translation_body_from_camera = np.asarray(config.camera_position_body, dtype=np.float64)
    return points @ rotation_body_from_camera.T + translation_body_from_camera


def body_points_to_world(points_body, odometry, config: DynamicPerceptionConfig):
    """Transform body points with the pose from the same stamped odometry."""
    frame_id = str(odometry.child_frame_id).lstrip("/")
    if frame_id != config.body_frame_id.lstrip("/"):
        raise ValueError(
            f"odometry child_frame_id {frame_id!r} does not match body frame "
            f"{config.body_frame_id!r}"
        )
    pose = odometry.pose.pose
    rotation_world_from_body = quaternion_to_rotation(
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
    )
    translation = np.asarray([pose.position.x, pose.position.y, pose.position.z])
    points = np.asarray(points_body, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points_body must be finite [N,3]")
    return points @ rotation_world_from_body.T + translation


def camera_model_from_config(config: DynamicPerceptionConfig) -> CameraModel:
    return CameraModel(
        width=config.camera_width,
        height=config.camera_height,
        fx=config.camera_fx,
        fy=config.camera_fy,
        cx=config.camera_cx,
        cy=config.camera_cy,
        depth_scale=config.camera_depth_scale,
        min_depth=config.camera_min_depth,
        max_depth=config.camera_max_depth,
    )


def camera_model_from_info(camera_info, config: DynamicPerceptionConfig) -> CameraModel:
    if len(camera_info.K) != 9:
        raise ValueError("CameraInfo.K must contain 9 values")
    return CameraModel(
        width=int(camera_info.width),
        height=int(camera_info.height),
        fx=float(camera_info.K[0]),
        fy=float(camera_info.K[4]),
        cx=float(camera_info.K[2]),
        cy=float(camera_info.K[5]),
        depth_scale=config.camera_depth_scale,
        min_depth=config.camera_min_depth,
        max_depth=config.camera_max_depth,
    )


def validate_sensor_frame(message, config: DynamicPerceptionConfig):
    frame_id = str(message.header.frame_id).lstrip("/")
    expected = config.camera_frame_id.lstrip("/")
    if not frame_id and config.allow_empty_sensor_frame:
        return
    if frame_id != expected:
        raise ValueError(
            f"sensor frame_id {frame_id!r} does not match configured optical frame {expected!r}"
        )


def validate_synchronized_timestamps(sensor_message, odometry, camera_info,
                                     config: DynamicPerceptionConfig):
    sensor_time = message_timestamp(sensor_message)
    offsets = {"odometry": abs(message_timestamp(odometry) - sensor_time)}
    if camera_info is not None:
        offsets["camera_info"] = abs(message_timestamp(camera_info) - sensor_time)
    largest = max(offsets.values(), default=0.0)
    if largest > config.max_pose_time_offset:
        raise ValueError(
            f"synchronized input offset {largest:.6f}s exceeds "
            f"max_pose_time_offset={config.max_pose_time_offset:.6f}s"
        )
    return sensor_time, offsets


def validate_depth_encoding(encoding: str):
    normalized = str(encoding).upper()
    if normalized not in {"32FC1", "16UC1", "MONO16"}:
        raise ValueError(f"unsupported depth encoding: {encoding!r}")
    return normalized
