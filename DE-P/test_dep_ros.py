import rospy
import message_filters
import std_msgs.msg
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Vector3
from visualization_msgs.msg import Marker, MarkerArray  # 导入 Marker 相关
from std_msgs.msg import ColorRGBA, Float32, String  # 用于颜色设置
from cv_bridge import CvBridge
from threading import Lock
from sensor_msgs.msg import CameraInfo, PointCloud2, PointField, Image
from sensor_msgs import point_cloud2

import cv2
import os
import time
import torch
import numpy as np
import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from config.config import cfg
from data.static_yopo_preprocessing_v1 import StaticYOPODepthPreprocessorV1
from control_msg import PositionCommand
from policy.dep_network import DepNetwork
from policy.backbone_variant import BACKBONE_VARIANTS, resolve_backbone_variant
from policy.checkpoint_utils import load_dep_checkpoint, validate_dep_checkpoint_variant
from policy.poly_solver import *
from policy.runtime_safety_v1 import (
    RuntimeSafetyConfigV1,
    RuntimeTrajectorySafetyV1,
    clamp_vector_norm_v1,
    depth_to_body_points_v1,
)
from policy.deadlock_recovery_v1 import (
    DeadlockRecoveryConfigV1,
    DeadlockRecoveryV1,
)
from policy.state_transform import *
from policy.dynamic.context import DynamicContext
from policy.dynamic.types import DynamicPerceptionConfig
from policy.dynamic.types import Pose as DynamicPose
from policy.dynamic.camera_model import depth_to_pointcloud
from policy.dynamic.ros_bridge import (
    camera_model_from_config,
    camera_model_from_info,
    camera_pose_from_odometry,
    message_timestamp,
    validate_depth_encoding,
    validate_sensor_frame,
    validate_synchronized_timestamps,
)

try:
    from torch2trt import TRTModule
except ImportError:
    print("tensorrt not found.")


class DepNet:
    def __init__(self, config, weight, dynamic_config=None):
        self.config = config
        self.dynamic_config = dynamic_config or DynamicPerceptionConfig.from_global_config()
        self.dynamic_config.validate()
        print("Dynamic perception config:", json.dumps(asdict(self.dynamic_config), indent=2))
        self.backbone_variant = resolve_backbone_variant(config.get("backbone_variant"))
        print(f"Backbone variant: {self.backbone_variant}")
        weight = os.path.abspath(os.path.expanduser(weight))
        if not os.path.isfile(weight):
            raise FileNotFoundError(f"DEP inference checkpoint does not exist: {weight}")
        print("Loading inference checkpoint:", weight)
        if not config["use_tensorrt"]:
            # Fail on architecture mismatch before touching the ROS runtime.
            validate_dep_checkpoint_variant(weight, self.backbone_variant)
        rospy.init_node('dep_net', anonymous=False)
        # load params
        cfg["train"] = False
        self.height = cfg['image_height']
        self.width = cfg['image_width']
        self.min_dis, self.max_dis = 0.04, 20.0
        self.scale = {'435': 0.001, 'simulation': 1.0}.get(self.config['env'], 1.0)
        self.static_depth_preprocessor = StaticYOPODepthPreprocessorV1(
            height=self.height, width=self.width, depth_scale=self.scale,
            min_depth_m=self.min_dis, max_depth_m=self.max_dis,
        )
        self.goal = np.array(self.config['goal'])
        self.goal_z = float(self.config.get("goal_z", self.goal[2]))
        self.arrival_radius = float(self.config.get("arrival_radius", 5.0))
        self.wait_for_goal = bool(self.config.get("wait_for_goal", False))
        self.hold_on_arrival = bool(self.config.get("hold_on_arrival", False))
        self.goal_received = not self.wait_for_goal
        self.plan_from_reference = self.config['plan_from_reference']
        self.use_trt = self.config['use_tensorrt']
        if self.use_trt and self.dynamic_config.enabled:
            raise RuntimeError(
                "dynamic perception is not supported by the static TensorRT artifact; "
                "disable dynamic_perception or use the PyTorch model"
            )
        self.verbose = self.config['verbose']
        self.visualize = self.config['visualize']
        self.Rotation_bc = R.from_euler('ZYX', [0, self.config['pitch_angle_deg'], 0], degrees=True).as_matrix()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # --------------- 新增：坐标系轨迹可视化参数 ---------------
        self.frame_decay_time = float(self.config.get("frame_decay_time", 3.0))
        self.frame_size = float(self.config.get("frame_size", 0.5))
        self.frame_line_width = 0.02  # 坐标系轴线宽度
        self.frame_max_count = 100  # 最大缓存坐标系数量（防止内存溢出）
        self.frame_list = []  # 存储坐标系数据：[(marker_id, position, orientation, timestamp), ...]
        self.next_frame_id = 0  # 下一个坐标系的唯一ID
        # 坐标系轴颜色（ROS标准：X红、Y绿、Z蓝）
        self.x_axis_color = ColorRGBA(1.0, 0.0, 0.0, 1.0)  # X轴：红色
        self.y_axis_color = ColorRGBA(0.0, 1.0, 0.0, 1.0)  # Y轴：绿色
        self.z_axis_color = ColorRGBA(0.0, 0.0, 1.0, 1.0)  # Z轴：蓝色
        # --------------------------------------------------------

        # variables
        self.bridge = CvBridge()
        self.odom = Odometry()
        self.odom_init = False
        self.last_yaw = 0.0
        self.ctrl_dt = 0.02
        self.ctrl_time = None
        self.desire_init = False
        self.arrive = False
        self.desire_pos = None
        self.desire_vel = None
        self.desire_acc = None
        self.optimal_poly_x = None
        self.optimal_poly_y = None
        self.optimal_poly_z = None
        self.lock = Lock()
        self.dynamic_lock = Lock()
        self.latest_dynamic_context = DynamicContext.invalid(
            self.dynamic_config.source, reason="no_sensor_update"
        )
        self.last_dynamic_update_wall = None
        self.dynamic_perception = None
        self.dynamic_network_attention_enabled = bool(
            config.get("dynamic_network_attention_enabled", True)
        )
        self.dynamic_camera_model = camera_model_from_config(self.dynamic_config)
        self.last_control_msg = None
        self.state_transform = StateTransform()
        self.lattice_primitive = LatticePrimitive.get_instance()
        self.traj_time = self.lattice_primitive.segment_time
        safety_mapping = dict(cfg["runtime_safety"])
        if config.get("runtime_safety_enabled") is not None:
            safety_mapping["enabled"] = bool(config["runtime_safety_enabled"])
        self.runtime_safety_config = RuntimeSafetyConfigV1.from_mapping(safety_mapping)
        self.runtime_safety = RuntimeTrajectorySafetyV1(self.runtime_safety_config)
        recovery_mapping = dict(cfg["deadlock_recovery"])
        if config.get("deadlock_recovery_enabled") is not None:
            recovery_mapping["enabled"] = bool(
                config["deadlock_recovery_enabled"]
            )
        self.deadlock_recovery_config = (
            DeadlockRecoveryConfigV1.from_mapping(recovery_mapping)
        )
        self.deadlock_recovery = DeadlockRecoveryV1(
            self.deadlock_recovery_config
        )
        self.recovery_retreat_installed = False
        self.active_traj_duration = self.traj_time
        telemetry_value = config.get("safety_telemetry")
        self.safety_telemetry_path = (
            Path(telemetry_value).expanduser().resolve() if telemetry_value else None
        )
        if self.safety_telemetry_path is not None:
            self.safety_telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        self.safety_command_clamp_count = 0
        print("Runtime safety contract:", json.dumps(
            self.runtime_safety_config.contract(), sort_keys=True
        ))
        print("Deadlock recovery contract:", json.dumps(
            self.deadlock_recovery_config.contract(), sort_keys=True
        ))

        # eval
        self.time_forward = 0.0
        self.time_process = 0.0
        self.time_prepare = 0.0
        self.time_interpolation = 0.0
        self.time_visualize = 0.0
        self.count = 0

        # Load Network
        if self.use_trt:
            self.policy = TRTModule()
            self.policy.load_state_dict(torch.load(weight))
        else:
            self.policy = DepNetwork(
                backbone_variant=self.backbone_variant,
                dynamic_config=self.dynamic_config,
            )
            load_dep_checkpoint(self.policy, weight, self.backbone_variant)
            self.policy = self.policy.to(self.device)
            self.policy.eval()
        self.warm_up()

        # ros publisher
        self.lattice_traj_pub = rospy.Publisher("/dep_net/lattice_trajs_visual", PointCloud2, queue_size=1)
        self.best_traj_pub = rospy.Publisher("/dep_net/best_traj_visual", PointCloud2, queue_size=1)
        self.all_trajs_pub = rospy.Publisher("/dep_net/trajs_visual", PointCloud2, queue_size=1)
        self.ctrl_pub = rospy.Publisher(self.config["ctrl_topic"], PositionCommand, queue_size=1)
        self.safety_decision_pub = rospy.Publisher(
            "/dep_net/safety_decision", String, queue_size=10
        )
        # --------------- 新增：坐标系轨迹发布者 ---------------
        self.frame_traj_pub = rospy.Publisher("/dep_net/frame_trajectory", MarkerArray, queue_size=1)
        # --------------------------------------------------------

        # ros subscriber
        self.odom_sub = rospy.Subscriber(self.dynamic_config.odom_topic, Odometry, self.callback_odometry, queue_size=1)
        self._configure_dynamic_subscribers()
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.callback_set_goal, queue_size=1)
        
        # --------------- 新增：坐标系更新定时器（5Hz，平衡实时性和性能） ---------------
        self.frame_update_timer = rospy.Timer(rospy.Duration(0.2), self.update_frame_trajectory)
        # --------------------------------------------------------

        # ros timer
        rospy.sleep(1.0)  # wait connection...
        self.timer_ctrl = rospy.Timer(rospy.Duration(self.ctrl_dt), self.control_pub)
        print("DE-P Net Node Ready!")
        if self.wait_for_goal:
            print(
                "Waiting for RViz '2D Nav Goal'; "
                f"goal altitude={self.goal_z:.2f} m, arrival radius={self.arrival_radius:.2f} m"
            )
        print(f"Frame Trajectory Visualization Enabled: Decay Time = {self.frame_decay_time}s, Frame Size = {self.frame_size}m")
        rospy.spin()

    def _configure_dynamic_subscribers(self):
        """Tracker ownership and temporal synchronization live outside DepNetwork."""
        self.depth_sub = None
        self.dynamic_sync = None
        if not self.dynamic_config.enabled:
            self.depth_sub = rospy.Subscriber(
                self.dynamic_config.depth_topic, Image, self.callback_depth, queue_size=1
            )
            return

        from policy.dynamic.dynamic_perception import DynamicPerception

        self.dynamic_perception = DynamicPerception(
            self.dynamic_config, feature_shape=(cfg["vertical_num"], cfg["horizon_num"]),
            attention_device="cpu",
        )
        print("Dynamic perception startup:", json.dumps({
            "source": self.dynamic_config.source,
            "foreground_mode": self.dynamic_config.foreground_mode,
            "raw_depth_shape": [self.dynamic_config.camera_height,
                                self.dynamic_config.camera_width],
            "depth_stride": self.dynamic_config.depth_stride,
            "range_history_frames": self.dynamic_config.range_history_frames,
            "range_min_history_support": self.dynamic_config.range_min_history_support,
            "causal": True,
        }, sort_keys=True))
        if self.dynamic_config.use_tf_extrinsics:
            import tf2_ros
            self.dynamic_tf_buffer = tf2_ros.Buffer(
                cache_time=rospy.Duration(10.0)
            )
            self.dynamic_tf_listener = tf2_ros.TransformListener(self.dynamic_tf_buffer)
        # Planning always remains depth-driven. This independent subscriber is
        # also the static fallback when synchronization or CameraInfo is absent.
        self.depth_sub = rospy.Subscriber(
            self.dynamic_config.depth_topic, Image, self.callback_depth, queue_size=1
        )
        if self.dynamic_config.publish_debug:
            self.dynamic_track_pub = rospy.Publisher(
                "/dep_net/dynamic_tracks", MarkerArray, queue_size=1
            )
            self.dynamic_attention_pub = rospy.Publisher(
                "/dep_net/dynamic_attention", Image, queue_size=1
            )
            self.dynamic_time_pub = rospy.Publisher(
                "/dep_net/dynamic_perception_ms", Float32, queue_size=1
            )
            self.dynamic_fallback_pub = rospy.Publisher(
                "/dep_net/dynamic_fallback", String, queue_size=1
            )

        sensor_type = Image if self.dynamic_config.source == "depth" else PointCloud2
        sensor_topic = (self.dynamic_config.depth_topic if self.dynamic_config.source == "depth"
                        else self.dynamic_config.pointcloud_topic)
        self.dynamic_sensor_sub = message_filters.Subscriber(sensor_topic, sensor_type)
        self.dynamic_odom_sub = message_filters.Subscriber(
            self.dynamic_config.odom_topic, Odometry
        )
        sync_inputs = [self.dynamic_sensor_sub, self.dynamic_odom_sub]
        if self.dynamic_config.use_camera_info:
            self.dynamic_camera_info_sub = message_filters.Subscriber(
                self.dynamic_config.camera_info_topic, CameraInfo
            )
            sync_inputs.append(self.dynamic_camera_info_sub)
        self.dynamic_sync = message_filters.ApproximateTimeSynchronizer(
            sync_inputs,
            queue_size=self.dynamic_config.sync_queue_size,
            slop=self.dynamic_config.sync_slop,
            allow_headerless=False,
        )
        callback = (self.callback_dynamic_depth if self.dynamic_config.source == "depth"
                    else self.callback_dynamic_pointcloud)
        self.dynamic_sync.registerCallback(callback)
        self.dynamic_watchdog_timer = rospy.Timer(
            rospy.Duration(0.5), self._dynamic_watchdog
        )

    def _warn_dynamic(self, reason, exception=None):
        detail = reason if exception is None else f"{reason}: {exception}"
        rospy.logwarn_throttle(2.0, f"DEP dynamic fallback: {detail}")
        if self.dynamic_config.publish_debug:
            self.dynamic_fallback_pub.publish(String(data=detail))

    def _set_dynamic_context(self, context):
        with self.dynamic_lock:
            self.latest_dynamic_context = context
            if context.valid:
                self.last_dynamic_update_wall = time.monotonic()

    def _dynamic_watchdog(self, _timer):
        if self.last_dynamic_update_wall is None:
            self._warn_dynamic("no synchronized dynamic sensor update received")
            return
        age = time.monotonic() - self.last_dynamic_update_wall
        stale_after = max(0.5, 4.0 * self.dynamic_config.max_pose_time_offset)
        if age > stale_after:
            self._invalidate_dynamic(
                self.dynamic_config.source, None,
                f"synchronized dynamic input stale for {age:.3f}s",
            )

    def _invalidate_dynamic(self, source, timestamp, reason, exception=None):
        detail = reason if exception is None else f"{reason}: {exception}"
        self._set_dynamic_context(DynamicContext.invalid(source, timestamp, detail))
        self._warn_dynamic(reason, exception)

    def _camera_model(self, camera_info):
        if camera_info is None:
            return camera_model_from_config(self.dynamic_config)
        return camera_model_from_info(camera_info, self.dynamic_config)

    def _camera_pose(self, odometry, sensor_message):
        if not self.dynamic_config.use_tf_extrinsics:
            return camera_pose_from_odometry(odometry, self.dynamic_config)
        stamp = sensor_message.header.stamp
        transform = self.dynamic_tf_buffer.lookup_transform(
            self.dynamic_config.world_frame_id,
            self.dynamic_config.camera_frame_id,
            stamp,
            rospy.Duration(self.dynamic_config.tf_lookup_timeout),
        )
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return DynamicPose(
            position_world=np.asarray(
                [translation.x, translation.y, translation.z], dtype=np.float64
            ),
            rotation_world_from_camera=R.from_quat(
                [rotation.x, rotation.y, rotation.z, rotation.w]
            ).as_matrix(),
            timestamp=message_timestamp(sensor_message),
        )

    def _decode_depth_message(self, data):
        encoding = validate_depth_encoding(data.encoding)
        try:
            depth = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        except Exception:
            dtype = np.float32 if encoding == "32FC1" else np.uint16
            depth = np.frombuffer(data.data, dtype=dtype).reshape(data.height, data.width)
        depth = np.asarray(depth)
        if depth.ndim != 2:
            raise ValueError(f"depth image must be 2-D, got {depth.shape}")
        return depth

    def _update_dynamic(self, sensor_data, sensor_message, odometry, camera_info,
                        input_kind="pointcloud"):
        timestamp, offsets = validate_synchronized_timestamps(
            sensor_message, odometry, camera_info, self.dynamic_config
        )
        validate_sensor_frame(sensor_message, self.dynamic_config)
        if camera_info is not None:
            validate_sensor_frame(camera_info, self.dynamic_config)
        camera_model = self._camera_model(camera_info)
        camera_pose = self._camera_pose(odometry, sensor_message)
        started = time.perf_counter()
        if input_kind == "depth":
            result = self.dynamic_perception.update_depth(
                sensor_data, camera_pose, timestamp, camera_model
            )
        else:
            result = self.dynamic_perception.update(
                sensor_data, camera_pose, timestamp, camera_model
            )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        diagnostics = dict(result.diagnostics)
        diagnostics.update({"time_offsets_seconds": offsets, "perception_ms": elapsed_ms})
        context = DynamicContext.from_perception_result(
            replace(result, diagnostics=diagnostics), timestamp, self.dynamic_config.source
        )
        self.dynamic_camera_model = camera_model
        self._set_dynamic_context(context)
        if self.dynamic_config.publish_debug:
            self.dynamic_time_pub.publish(Float32(data=elapsed_ms))
            self._publish_dynamic_debug(result, context)
        return context

    def callback_dynamic_depth(self, depth_message, odometry, camera_info=None):
        timestamp = message_timestamp(depth_message)
        try:
            depth = self._decode_depth_message(depth_message)
            camera_model = self._camera_model(camera_info)
            if depth.shape != (camera_model.height, camera_model.width):
                raise ValueError(
                    f"depth shape {depth.shape} does not match intrinsics "
                    f"{(camera_model.height, camera_model.width)}"
                )
            self._update_dynamic(
                depth, depth_message, odometry, camera_info, input_kind="depth"
            )
        except Exception as exc:
            self._invalidate_dynamic("depth", timestamp, "depth_dynamic_update_failed", exc)

    def callback_dynamic_pointcloud(self, cloud_message, odometry, camera_info=None):
        timestamp = message_timestamp(cloud_message)
        try:
            # Input PointCloud2 must genuinely contain optical-camera coordinates.
            # The current simulator labels body-frame samples as 'odom', so it is
            # intentionally rejected instead of applying an assumed axis mapping.
            points = np.asarray(list(point_cloud2.read_points(
                cloud_message, field_names=("x", "y", "z"), skip_nans=True
            )), dtype=np.float32).reshape(-1, 3)
            self._update_dynamic(points, cloud_message, odometry, camera_info)
        except Exception as exc:
            self._invalidate_dynamic(
                "pointcloud", timestamp, "pointcloud_dynamic_update_failed", exc
            )

    def _context_for_depth(self, depth_message):
        if not self.dynamic_config.enabled:
            return None
        with self.dynamic_lock:
            context = self.latest_dynamic_context
        if not context.valid or context.timestamp is None:
            return context
        offset = abs(message_timestamp(depth_message) - context.timestamp)
        if offset > self.dynamic_config.max_pose_time_offset:
            return DynamicContext.invalid(
                self.dynamic_config.source,
                message_timestamp(depth_message),
                f"dynamic_context_age={offset:.6f}s exceeds max_pose_time_offset",
            )
        return context

    def _enter_safe_state(self, reason):
        self.ctrl_time = None
        self._warn_dynamic(f"safe_state: {reason}")

    def _publish_dynamic_debug(self, result, context):
        attention = result.attention_map.detach().cpu().numpy()[0, 0]
        heatmap = np.uint8(np.clip(attention, 0, 1) * 255)
        message = self.bridge.cv2_to_imgmsg(heatmap, encoding="mono8")
        message.header.stamp = rospy.Time.from_sec(context.timestamp)
        message.header.frame_id = self.dynamic_config.camera_frame_id
        self.dynamic_attention_pub.publish(message)

        markers = MarkerArray()
        for track in result.dynamic_tracks:
            marker = Marker()
            marker.header.stamp = message.header.stamp
            marker.header.frame_id = self.dynamic_config.world_frame_id
            marker.ns = "dynamic_centers"
            marker.id = int(track.track_id)
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(track.position_world[0])
            marker.pose.position.y = float(track.position_world[1])
            marker.pose.position.z = float(track.position_world[2])
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = marker.scale.z = 0.35
            marker.color = ColorRGBA(1.0, 0.2, 0.0, 0.9)
            marker.lifetime = rospy.Duration(0.3)
            markers.markers.append(marker)
        self.dynamic_track_pub.publish(markers)

    def callback_set_goal(self, data):
        self.goal = np.asarray(
            [data.pose.position.x, data.pose.position.y, self.goal_z],
            dtype=np.float64,
        )
        self.goal_received = True
        self.arrive = False
        self.ctrl_time = None
        print(
            f"New Goal: ({self.goal[0]:.1f}, {self.goal[1]:.1f}, {self.goal[2]:.1f}); "
            f"arrival radius={self.arrival_radius:.1f} m"
        )

        # 可选：重置坐标系轨迹（到达新目标时清空历史坐标系）
        with self.lock:
            position = None
            if self.odom_init:
                position = np.asarray([
                    self.odom.pose.pose.position.x,
                    self.odom.pose.pose.position.y,
                    self.odom.pose.pose.position.z,
                ], dtype=np.float64)
            self.deadlock_recovery.reset(position)
            self.recovery_retreat_installed = False
            self.frame_list.clear()
            self.next_frame_id = 0

    # the first frame
    def callback_odometry(self, data):
        self.odom = data
        
        if not self.desire_init:
            self.desire_pos = np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
            self.desire_vel = np.array((self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z))
            self.desire_acc = np.array((0.0, 0.0, 0.0))
            ypr = R.from_quat([self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y,
                               self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w]).as_euler('ZYX', degrees=False)
            self.last_yaw = ypr[0]
        self.odom_init = True

        pos = np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
        if (
            self.goal_received
            and np.linalg.norm(pos - self.goal) < self.arrival_radius
            and not self.arrive
        ):
            print(
                f"Arrive! distance={np.linalg.norm(pos - self.goal):.2f} m "
                f"(threshold={self.arrival_radius:.2f} m)"
            )
            self.arrive = True

        # --------------- 新增：记录当前坐标系（位置+姿态） ---------------
        current_time = rospy.Time.now().to_sec()
        # 获取当前姿态（四元数）
        orientation = Quaternion()
        orientation.x = self.odom.pose.pose.orientation.x
        orientation.y = self.odom.pose.pose.orientation.y
        orientation.z = self.odom.pose.pose.orientation.z
        orientation.w = self.odom.pose.pose.orientation.w

        with self.lock:
            if self.deadlock_recovery.mode == DeadlockRecoveryV1.NORMAL:
                self.deadlock_recovery.record_position(pos)
            # 添加当前坐标系到缓存
            self.frame_list.append((
                self.next_frame_id,  # 唯一ID
                pos,  # 位置 (x,y,z)
                orientation,  # 姿态（四元数）
                current_time  # 时间戳
            ))
            self.next_frame_id += 1  # 更新下一个ID

            # 限制最大缓存数量
            if len(self.frame_list) > self.frame_max_count:
                self.frame_list.pop(0)  # 移除最早的坐标系
        # --------------------------------------------------------

    # --------------- 新增：更新并发布坐标系轨迹 ---------------
    def update_frame_trajectory(self, _timer):
        if not self.odom_init:
            return

        current_time = rospy.Time.now().to_sec()
        marker_array = MarkerArray()

        with self.lock:
            # 过滤超时的坐标系
            valid_frames = [
                (frame_id, pos, ori, ts) for (frame_id, pos, ori, ts) in self.frame_list
                if (current_time - ts) < self.frame_decay_time
            ]
            # 更新缓存（只保留有效坐标系）
            self.frame_list = valid_frames

            # 为每个有效坐标系创建 Marker（3个箭头：X/Y/Z轴）
            for (frame_id, pos, ori, ts) in valid_frames:
                # 1. X轴箭头（红色）
                x_marker = self.create_axis_marker(
                    marker_id=frame_id * 3,  # 每个坐标系占用3个Marker ID（X=0, Y=1, Z=2）
                    position=pos,
                    orientation=ori,
                    axis_direction=Vector3(1.0, 0.0, 0.0),  # X轴方向
                    color=self.x_axis_color
                )
                # 2. Y轴箭头（绿色）
                y_marker = self.create_axis_marker(
                    marker_id=frame_id * 3 + 1,
                    position=pos,
                    orientation=ori,
                    axis_direction=Vector3(0.0, 1.0, 0.0),  # Y轴方向
                    color=self.y_axis_color
                )
                # 3. Z轴箭头（蓝色）
                z_marker = self.create_axis_marker(
                    marker_id=frame_id * 3 + 2,
                    position=pos,
                    orientation=ori,
                    axis_direction=Vector3(0.0, 0.0, 1.0),  # Z轴方向
                    color=self.z_axis_color
                )

                marker_array.markers.append(x_marker)
                marker_array.markers.append(y_marker)
                marker_array.markers.append(z_marker)

            # 添加删除超时Marker的指令（如果有被过滤的坐标系）
            if len(self.frame_list) < len(valid_frames):
                delete_marker = Marker()
                delete_marker.action = Marker.DELETEALL
                marker_array.markers.append(delete_marker)

        # 发布坐标系数组
        self.frame_traj_pub.publish(marker_array)

    # --------------- 辅助函数：创建单轴箭头Marker ---------------
    def create_axis_marker(self, marker_id, position, orientation, axis_direction, color):
        marker = Marker()
        marker.header.frame_id = "world"  # 与里程计统一坐标系
        marker.header.stamp = rospy.Time.now()
        marker.id = marker_id
        marker.type = Marker.ARROW  # 箭头类型（表示坐标轴）
        marker.action = Marker.ADD
        marker.lifetime = rospy.Duration(self.frame_decay_time)  # Marker生命周期（与衰减时间一致）

        # 坐标系位置和姿态
        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = position[2]
        marker.pose.orientation = orientation

        # 箭头尺寸（长度=frame_size，直径=line_width）
        marker.scale.x = self.frame_size  # 箭头长度（轴长）
        marker.scale.y = self.frame_line_width  # 箭头头部直径
        marker.scale.z = self.frame_line_width  # 箭头尾部直径

        # 箭头颜色
        marker.color = color

        return marker
    # --------------------------------------------------------

    def process_odom(self):
        # Rwb -> Rwc -> Rcw
        Rotation_wb = R.from_quat([self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y,
                                   self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w]).as_matrix()
        self.Rotation_wc = np.dot(Rotation_wb, self.Rotation_bc)
        Rotation_cw = self.Rotation_wc.T

        # vel and acc
        vel_w = self.desire_vel if self.plan_from_reference else np.array([self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z])
        vel_c = np.dot(Rotation_cw, vel_w)
        acc_w = self.desire_acc
        acc_c = np.dot(Rotation_cw, acc_w)

        # goal_dir
        goal_w = self.goal - self.desire_pos
        goal_c = np.dot(Rotation_cw, goal_w)

        obs = np.concatenate((vel_c, acc_c, goal_c), axis=0).astype(np.float32)
        obs_norm = self.state_transform.normalize_obs(torch.from_numpy(obs[None, :]))
        return obs_norm.to(self.device, non_blocking=True)

    def _write_safety_telemetry(self, payload):
        payload = {
            "contract_version": "runtime_trajectory_safety_v1",
            "timestamp": time.time(),
            **payload,
        }
        encoded = json.dumps(payload, sort_keys=True)
        self.safety_decision_pub.publish(String(data=encoded))
        if self.safety_telemetry_path is not None:
            with self.safety_telemetry_path.open("a", encoding="utf-8") as handle:
                handle.write(encoded + "\n")

    @staticmethod
    def _candidate_polynomials(start_pos, start_vel, start_acc, endstate_w, duration):
        candidates = []
        for endstate in endstate_w:
            candidates.append(tuple(
                Poly5Solver(
                    start_pos[axis], start_vel[axis], start_acc[axis],
                    start_pos[axis] + endstate[axis, 0],
                    endstate[axis, 1], endstate[axis, 2], duration,
                )
                for axis in range(3)
            ))
        return tuple(candidates)

    def _install_trajectory(self, polynomials, duration):
        self.optimal_poly_x, self.optimal_poly_y, self.optimal_poly_z = polynomials
        self.active_traj_duration = float(duration)
        self.ctrl_time = 0.0

    @torch.inference_mode()
    def callback_depth(self, data, dynamic_context=None):
        if not self.odom_init: return
        if not self.goal_received or self.arrive:
            return

        # 1. Depth Image Process
        try:
            depth_raw = self._decode_depth_message(data)
        except Exception as exc:
            rospy.logerr_throttle(2.0, f"DEP depth conversion failed: {exc}")
            return

        time0 = time.time()
        depth = self.static_depth_preprocessor(depth_raw)[None, :, :, :]
        # cv2.imshow("1", depth[0][0])
        # cv2.waitKey(1)

        # 2. DE-P Network Inference
        # input prepare
        time1 = time.time()
        depth_input = torch.from_numpy(depth).to(self.device, non_blocking=True)  # (non_blocking: copying speed 3x)
        obs_norm = self.process_odom()
        obs_input = self.state_transform.prepare_input(obs_norm)
        obs_input = obs_input.to(self.device, non_blocking=True)
        # torch.cuda.synchronize()

        time2 = time.time()
        # Forward (TensorRT: inference speed increased by 5x)
        if dynamic_context is None:
            dynamic_context = self._context_for_depth(data)
        runtime_dynamic_context = dynamic_context
        network_dynamic_context = (
            dynamic_context if self.dynamic_network_attention_enabled else None
        )
        try:
            if self.use_trt:
                endstate_pred, score_pred = self.policy(depth_input, obs_input)
            else:
                endstate_pred, score_pred = self.policy(
                    depth_input, obs_input,
                    dynamic_context=network_dynamic_context
                )
            if not bool(torch.isfinite(endstate_pred).all()) or not bool(torch.isfinite(score_pred).all()):
                raise FloatingPointError("network output contains NaN/Inf")
        except Exception as exc:
            if self.dynamic_config.enabled and not self.dynamic_config.fallback_to_static:
                self._enter_safe_state(str(exc))
                return
            rospy.logerr_throttle(2.0, f"DEP inference rejected unsafe output: {exc}")
            return
        if getattr(self.policy, "last_dynamic_fallback_reason", None):
            self._warn_dynamic(self.policy.last_dynamic_fallback_reason)
        endstate_pred, score_pred = endstate_pred.cpu().numpy(), score_pred.cpu().numpy()
        time3 = time.time()

        # 3. Post-Processing
        # Replacing PyTorch operation on CUDA with NumPy operation on CPU (speed increased by 10x)
        return_all = self.visualize or self.runtime_safety_config.enabled
        endstate, score = self.process_output(
            endstate_pred, score_pred, return_all_preds=return_all
        )
        if not np.isfinite(endstate).all() or not np.isfinite(score).all():
            self._enter_safe_state("post-processed trajectory contains NaN/Inf")
            return
        # Vectorization: transform the prediction(P V A in body frame) to the world frame with the attitude (without the position)
        endstate_c = endstate.reshape(-1, 3, 3).transpose(0, 2, 1)  # [N, 9] -> [N, 3, 3] -> [px vx ax, py vy ay, pz vz az]
        endstate_w = np.matmul(self.Rotation_wc, endstate_c)

        raw_scores = np.asarray(score_pred).reshape(-1)
        with self.lock:  # Python3.8: threads are scheduled using time slices, add the lock to ensure safety
            start_pos = self.desire_pos if self.plan_from_reference else np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
            start_vel = self.desire_vel if self.plan_from_reference else np.array((self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z))
            start_acc = np.asarray(self.desire_acc, dtype=np.float64)
            candidates = self._candidate_polynomials(
                start_pos, start_vel, start_acc, endstate_w, self.traj_time
            )
            if self.runtime_safety_config.enabled:
                try:
                    candidates, projection_scales, projection_succeeded = (
                        self.runtime_safety.project_endstate_candidates(
                            start_pos, start_vel, start_acc, endstate_w,
                            self.traj_time,
                        )
                    )
                    camera_model = self.dynamic_camera_model
                    if depth_raw.shape != (camera_model.height, camera_model.width):
                        raise ValueError(
                            f"raw depth {depth_raw.shape} does not match runtime camera "
                            f"{(camera_model.height, camera_model.width)}"
                        )
                    obstacle_points = depth_to_body_points_v1(
                        depth_raw, camera_model,
                        self.dynamic_config.camera_rotation_body_from_camera,
                        stride=self.runtime_safety_config.depth_stride,
                    )
                    evaluations = self.runtime_safety.evaluate(
                        candidates, self.traj_time, obstacle_points,
                        start_pos, self.Rotation_wc,
                        dynamic_tracks=(
                            runtime_dynamic_context.dynamic_tracks
                            if runtime_dynamic_context is not None
                            and runtime_dynamic_context.valid else ()
                        ),
                        query_timestamp=message_timestamp(data),
                    )
                    selection = self.runtime_safety.select(raw_scores, evaluations)
                    feasible_candidate_count = int(sum(
                        item.feasible for item in evaluations
                    ))
                    current_observed_clearance = (
                        float(np.min(np.linalg.norm(obstacle_points, axis=1)))
                        if len(obstacle_points) else None
                    )
                    collision_floor_present = (
                        current_observed_clearance is not None
                        and current_observed_clearance < max(
                            0.0,
                            self.runtime_safety_config.vehicle_radius_m
                            - self.runtime_safety_config.clearance_sensor_tolerance_m,
                        )
                    )
                    recovery = self.deadlock_recovery.observe(
                        feasible_candidate_count=feasible_candidate_count,
                        collision_floor_present=collision_floor_present,
                        speed_mps=float(np.linalg.norm(start_vel)),
                        position_world=start_pos,
                        depth=depth_raw,
                    )
                    if recovery.transition is not None:
                        rospy.logwarn(
                            "DE-P recovery transition: %s", recovery.transition
                        )
                    action_id = (
                        selection.action_id
                        if recovery.mode == DeadlockRecoveryV1.NORMAL
                        else None
                    )
                    braking_compliant = None
                    retreat_compliant = None
                    if recovery.mode == DeadlockRecoveryV1.RETREAT:
                        needs_install = (
                            recovery.transition
                            == "braking_to_breadcrumb_retreat"
                            or not self.recovery_retreat_installed
                            or self.ctrl_time is None
                            or self.ctrl_time >= self.active_traj_duration
                        )
                        if needs_install:
                            retreat, duration, retreat_compliant = (
                                self.runtime_safety.recovery_trajectory(
                                    start_pos, start_vel, start_acc,
                                    recovery.retreat_target_world,
                                )
                            )
                            if retreat is None or not retreat_compliant:
                                self.deadlock_recovery.mode = (
                                    DeadlockRecoveryV1.YAW_SCAN
                                )
                                self.recovery_retreat_installed = False
                                rospy.logerr(
                                    "DE-P breadcrumb retreat was not limit-compliant; "
                                    "falling back to stationary yaw scan"
                                )
                            else:
                                self._install_trajectory(retreat, duration)
                                self.recovery_retreat_installed = True
                    elif action_id is None:
                        self.recovery_retreat_installed = False
                        braking, duration, braking_compliant = (
                            self.runtime_safety.braking_trajectory(
                                start_pos, start_vel, start_acc
                            )
                        )
                        if braking is None:
                            self._enter_safe_state("unable to construct braking trajectory")
                            return
                        self._install_trajectory(braking, duration)
                    else:
                        self.recovery_retreat_installed = False
                        self._install_trajectory(candidates[action_id], self.traj_time)
                    self._write_safety_telemetry({
                        "mode": recovery.mode,
                        "network_selection_mode": selection.mode,
                        "recovery_transition": recovery.transition,
                        "recovery_retreat_target_world": (
                            recovery.retreat_target_world
                        ),
                        "action_id": action_id,
                        "network_best_action_id": int(np.argmin(raw_scores)),
                        "network_best_rejected": (
                            not evaluations[int(np.argmin(raw_scores))].feasible
                        ),
                        "feasible_candidate_count": feasible_candidate_count,
                        "collision_floor_present": collision_floor_present,
                        "current_observed_clearance_m": (
                            current_observed_clearance
                        ),
                        "observed_point_count": int(len(obstacle_points)),
                        "predicted_dynamic_track_count": int(
                            len(runtime_dynamic_context.dynamic_tracks)
                            if runtime_dynamic_context is not None
                            and runtime_dynamic_context.valid else 0
                        ),
                        "braking_limit_compliant": braking_compliant,
                        "retreat_limit_compliant": retreat_compliant,
                        "projection_scales": projection_scales,
                        "projection_succeeded": projection_succeeded,
                        "projected_candidate_count": int(sum(
                            projection_succeeded
                        )),
                        "evaluations": [item.as_dict() for item in evaluations],
                    })
                except Exception as exc:
                    self._enter_safe_state(f"runtime safety evaluation failed: {exc}")
                    return
            else:
                action_id = int(np.argmin(raw_scores)) if return_all else 0
                self._install_trajectory(candidates[action_id], self.traj_time)
        time4 = time.time()
        self.visualize_trajectory(score_pred, endstate_w)
        time5 = time.time()

        if self.verbose:
            self.time_interpolation = self.time_interpolation + (time1 - time0)
            self.time_prepare = self.time_prepare + (time2 - time1)
            self.time_forward = self.time_forward + (time3 - time2)
            self.time_process = self.time_process + (time4 - time3)
            self.time_visualize = self.time_visualize + (time5 - time4)
            self.count = self.count + 1
            print(f"Time Consuming:"
                  f"depth-interpolation: {1000 * self.time_interpolation / self.count:.2f}ms;"
                  f"data-prepare: {1000 * self.time_prepare / self.count:.2f}ms; "
                  f"network-inference: {1000 * self.time_forward / self.count:.2f}ms; "
                  f"post-process: {1000 * self.time_process / self.count:.2f}ms;"
                  f"visualize-trajectory: {1000 * self.time_visualize / self.count:.2f}ms")

    def control_pub(self, _timer):
        if self.odom_init and (not self.goal_received or (self.arrive and self.hold_on_arrival)):
            self._publish_hold()
            return
        if self.ctrl_time is None or self.ctrl_time > self.active_traj_duration:
            return
        if self.arrive and self.last_control_msg is not None:
            self.desire_init = False   # ready for next rollout
            self.last_control_msg.trajectory_flag = self.last_control_msg.TRAJECTORY_STATUS_EMPTY
            self.ctrl_pub.publish(self.last_control_msg)
            return

        with self.lock:  # Python3.8: threads are scheduled using time slices, add the lock to ensure safety and publish frequency
            self.ctrl_time += self.ctrl_dt
            control_msg = PositionCommand()
            control_msg.header.stamp = rospy.Time.now()
            control_msg.trajectory_flag = control_msg.TRAJECTORY_STATUS_READY
            control_msg.position.x = self.optimal_poly_x.get_position(self.ctrl_time)
            control_msg.position.y = self.optimal_poly_y.get_position(self.ctrl_time)
            control_msg.position.z = self.optimal_poly_z.get_position(self.ctrl_time)
            velocity, velocity_clamped = clamp_vector_norm_v1(np.array([
                self.optimal_poly_x.get_velocity(self.ctrl_time),
                self.optimal_poly_y.get_velocity(self.ctrl_time),
                self.optimal_poly_z.get_velocity(self.ctrl_time),
            ]), self.runtime_safety_config.max_speed_mps)
            acceleration, acceleration_clamped = clamp_vector_norm_v1(np.array([
                self.optimal_poly_x.get_acceleration(self.ctrl_time),
                self.optimal_poly_y.get_acceleration(self.ctrl_time),
                self.optimal_poly_z.get_acceleration(self.ctrl_time),
            ]), self.runtime_safety_config.max_acceleration_mps2)
            control_msg.velocity.x, control_msg.velocity.y, control_msg.velocity.z = velocity
            (control_msg.acceleration.x, control_msg.acceleration.y,
             control_msg.acceleration.z) = acceleration
            if velocity_clamped or acceleration_clamped:
                self.safety_command_clamp_count += 1
                self._write_safety_telemetry({
                    "mode": "command_limit_backstop",
                    "velocity_clamped": velocity_clamped,
                    "acceleration_clamped": acceleration_clamped,
                    "clamp_count": self.safety_command_clamp_count,
                })
            self.desire_pos = np.array([control_msg.position.x, control_msg.position.y, control_msg.position.z])
            self.desire_vel = np.array([control_msg.velocity.x, control_msg.velocity.y, control_msg.velocity.z])
            self.desire_acc = np.array([control_msg.acceleration.x, control_msg.acceleration.y, control_msg.acceleration.z])
            goal_dir = self.goal - self.desire_pos
            if self.deadlock_recovery.mode == DeadlockRecoveryV1.YAW_SCAN:
                yaw, yaw_dot = self.deadlock_recovery.yaw_command(
                    self.last_yaw, self.ctrl_dt
                )
            elif self.deadlock_recovery.mode == DeadlockRecoveryV1.RETREAT:
                yaw, yaw_dot = self.last_yaw, 0.0
            else:
                yaw, yaw_dot = calculate_yaw(
                    self.desire_vel, goal_dir, self.last_yaw, self.ctrl_dt
                )
            self.last_yaw = yaw
            control_msg.yaw = yaw
            control_msg.yaw_dot = yaw_dot
            self.desire_init = True
            self.last_control_msg = control_msg
            self.ctrl_pub.publish(control_msg)

    def _publish_hold(self):
        """Keep the position controller active while waiting or after arrival."""
        position = np.array((
            self.odom.pose.pose.position.x,
            self.odom.pose.pose.position.y,
            self.odom.pose.pose.position.z,
        ))
        control_msg = PositionCommand()
        control_msg.header.stamp = rospy.Time.now()
        control_msg.trajectory_flag = control_msg.TRAJECTORY_STATUS_READY
        control_msg.position.x = float(position[0])
        control_msg.position.y = float(position[1])
        control_msg.position.z = float(position[2])
        control_msg.velocity.x = control_msg.velocity.y = control_msg.velocity.z = 0.0
        control_msg.acceleration.x = control_msg.acceleration.y = control_msg.acceleration.z = 0.0
        control_msg.yaw = float(self.last_yaw)
        control_msg.yaw_dot = 0.0
        self.desire_pos = position
        self.desire_vel = np.zeros(3, dtype=np.float64)
        self.desire_acc = np.zeros(3, dtype=np.float64)
        self.last_control_msg = control_msg
        self.ctrl_pub.publish(control_msg)

    def process_output(self, endstate_pred, score_pred, return_all_preds=False):
        endstate_pred = endstate_pred.reshape(9, self.lattice_primitive.traj_num).T
        score_pred = score_pred.reshape(self.lattice_primitive.traj_num)

        if not return_all_preds:
            action_id = np.argmin(score_pred)
            lattice_id = self.lattice_primitive.traj_num - 1 - action_id
            endstate = self.state_transform.pred_to_endstate_cpu(endstate_pred[action_id, :][np.newaxis, :], lattice_id)
            score = score_pred[action_id]
        else:
            score = score_pred
            endstate = self.state_transform.pred_to_endstate_cpu(endstate_pred, torch.arange(self.lattice_primitive.traj_num-1, -1, -1))

        return endstate, score

    def visualize_trajectory(self, pred_score, pred_endstate):
        dt = self.traj_time / 20.0
        start_pos = self.desire_pos if self.plan_from_reference else np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
        start_vel = self.desire_vel if self.plan_from_reference else np.array((self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z))
        # best predicted trajectory
        if self.best_traj_pub.get_num_connections() > 0:
            t_values = np.arange(0, self.traj_time, dt)
            points_array = np.stack((
                self.optimal_poly_x.get_position(t_values),
                self.optimal_poly_y.get_position(t_values),
                self.optimal_poly_z.get_position(t_values)
            ), axis=-1)
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = 'world'
            point_cloud_msg = point_cloud2.create_cloud_xyz32(header, points_array)
            self.best_traj_pub.publish(point_cloud_msg)
        # lattice primitive
        if self.visualize and self.lattice_traj_pub.get_num_connections() > 0:
            lattice_endstate = self.lattice_primitive.lattice_pos_node.cpu().numpy()
            lattice_endstate = np.dot(lattice_endstate, self.Rotation_wc.T)
            zero_state = np.zeros_like(lattice_endstate)
            t_values = np.arange(0, self.traj_time, dt)
            num_lattice = lattice_endstate.shape[0]  # 晶格轨迹数量
            lattice_points_list = []

            for i in range(num_lattice):
                # 第i条晶格轨迹的求解器
                lattice_poly_x = Poly5Solver(start_pos[0], start_vel[0], self.desire_acc[0],
                                            lattice_endstate[i, 0] + start_pos[0], zero_state[i, 0], zero_state[i, 0], self.traj_time)
                lattice_poly_y = Poly5Solver(start_pos[1], start_vel[1], self.desire_acc[1],
                                            lattice_endstate[i, 1] + start_pos[1], zero_state[i, 1], zero_state[i, 1], self.traj_time)
                lattice_poly_z = Poly5Solver(start_pos[2], start_vel[2], self.desire_acc[2],
                                            lattice_endstate[i, 2] + start_pos[2], zero_state[i, 2], zero_state[i, 2], self.traj_time)

                # 采样第i条晶格轨迹
                lattice_traj = np.stack((
                    lattice_poly_x.get_position(t_values),
                    lattice_poly_y.get_position(t_values),
                    lattice_poly_z.get_position(t_values)
                ), axis=-1)
                lattice_points_list.append(lattice_traj)

            # 合并所有晶格轨迹
            lattice_points_array = np.concatenate(lattice_points_list, axis=0)
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = 'world'
            point_cloud_msg = point_cloud2.create_cloud_xyz32(header, lattice_points_array)
            self.lattice_traj_pub.publish(point_cloud_msg)
        # all predicted trajectories
        if self.visualize and self.all_trajs_pub.get_num_connections() > 0:
            t_values = np.arange(0, self.traj_time, dt)  # 时间采样点：(20,)
            num_trajs = pred_endstate.shape[0]  # 轨迹数量（例如15条）
            points_list = []  # 存储所有轨迹的点

            # 逐条处理轨迹
            for i in range(num_trajs):
                # 第i条轨迹的终止状态
                end_x = pred_endstate[i, 0, 0] + start_pos[0]
                end_vx = pred_endstate[i, 0, 1]
                end_ax = pred_endstate[i, 0, 2]
        
                end_y = pred_endstate[i, 1, 0] + start_pos[1]
                end_vy = pred_endstate[i, 1, 1]
                end_ay = pred_endstate[i, 1, 2]
        
                end_z = pred_endstate[i, 2, 0] + start_pos[2]
                end_vz = pred_endstate[i, 2, 1]
                end_az = pred_endstate[i, 2, 2]

                # 为第i条轨迹创建求解器
                poly_x = Poly5Solver(start_pos[0], start_vel[0], self.desire_acc[0],
                                    end_x, end_vx, end_ax, self.traj_time)
                poly_y = Poly5Solver(start_pos[1], start_vel[1], self.desire_acc[1],
                                    end_y, end_vy, end_ay, self.traj_time)
                poly_z = Poly5Solver(start_pos[2], start_vel[2], self.desire_acc[2],
                                    end_z, end_vz, end_az, self.traj_time)

                # 采样第i条轨迹的位置（形状(20,3)）
                traj_points = np.stack((
                    poly_x.get_position(t_values),
                    poly_y.get_position(t_values),
                    poly_z.get_position(t_values)
                ), axis=-1)  # 单条轨迹形状：(20, 3)
                points_list.append(traj_points)

            # 合并所有轨迹的点（形状：(15, 20, 3) → 展平为(300, 3)，15×20=300）
            points_array = np.concatenate(points_list, axis=0)

            # 处理分数（与轨迹点数量匹配）
            scores = np.repeat(pred_score, t_values.size)  # 形状：(15×20,) = (300,)
            points_array = np.column_stack((points_array, scores))  # 形状：(300, 4)

            # 发布点云（保持原逻辑）
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = 'world'
            fields = [PointField('x', 0, PointField.FLOAT32, 1), 
                    PointField('y', 4, PointField.FLOAT32, 1),
                    PointField('z', 8, PointField.FLOAT32, 1), 
                    PointField('intensity', 12, PointField.FLOAT32, 1)]
            point_cloud_msg = point_cloud2.create_cloud(header, fields, points_array)
            self.all_trajs_pub.publish(point_cloud_msg)

    def warm_up(self):
        depth = torch.zeros((1, 1, self.height, self.width), dtype=torch.float32, device=self.device)
        obs = torch.zeros((1, 9), dtype=torch.float32, device=self.device)
        obs = self.state_transform.prepare_input(obs)
        endstate_pred, score_pred = self.policy(depth, obs)
        _ = self.state_transform.pred_to_endstate(endstate_pred)


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_tensorrt", type=int, default=0, help="use tensorrt or not")
    parser.add_argument("--trial", type=int, default=0, help="trial number")
    parser.add_argument("--epoch", type=int, default=10, help="epoch number")
    parser.add_argument("--backbone-variant", choices=BACKBONE_VARIANTS, default=None,
                        help="override config/traj_opt.yaml backbone_variant")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="explicit PyTorch or TensorRT checkpoint path")
    parser.add_argument("--dynamic-enabled", type=int, choices=(0, 1), default=None,
                        help="override dynamic_perception.enabled")
    parser.add_argument("--dynamic-source", choices=("depth", "pointcloud"), default=None,
                        help="override dynamic_perception.source")
    parser.add_argument(
        "--dynamic-network-attention-enabled", type=int, choices=(0, 1),
        default=1,
        help="allow tracker attention into CNN; set 0 for deterministic safety only",
    )
    parser.add_argument("--goal-z", type=float, default=2.0,
                        help="fixed goal altitude used by the RViz 2D Nav Goal tool")
    parser.add_argument("--arrival-radius", type=float, default=5.0,
                        help="3D distance threshold for declaring arrival")
    parser.add_argument("--wait-for-goal", type=int, choices=(0, 1), default=0,
                        help="hold position until an RViz 2D Nav Goal is received")
    parser.add_argument("--hold-on-arrival", type=int, choices=(0, 1), default=0,
                        help="publish a zero-velocity position hold after arrival")
    parser.add_argument("--runtime-safety-enabled", type=int, choices=(0, 1), default=None,
                        help="override the versioned runtime trajectory safety shield")
    parser.add_argument("--deadlock-recovery-enabled", type=int, choices=(0, 1), default=None,
                        help="override deterministic brake/scan/breadcrumb recovery")
    parser.add_argument("--safety-telemetry", type=Path, default=None,
                        help="optional JSONL path for per-replan safety decisions")
    # --------------- 新增：坐标系相关命令行参数 ---------------
    parser.add_argument("--frame_decay_time", type=float, default=3.0, help="frame trajectory decay time (seconds)")
    parser.add_argument("--frame_size", type=float, default=0.5, help="frame axis length (meters)")
    # --------------------------------------------------------
    return parser


def resolve_weight_path(use_tensorrt, trial, epoch, base_dir, backbone_variant=None):
    base = Path(base_dir).resolve()
    variant = resolve_backbone_variant(backbone_variant)
    if use_tensorrt:
        weight = base / ("dep_trt.pth" if variant == "legacy" else "dep_trt_corrected.pth")
    else:
        directory = f"DEP_{trial}" if variant == "legacy" else f"DEP_corrected_{trial}"
        weight = base / "saved" / directory / f"epoch{epoch}.pth"
    weight = weight.resolve()
    if not weight.is_file():
        available = sorted(str(path.resolve()) for path in (base / "saved").glob("DEP_*/epoch*.pth"))
        available_text = "\n  ".join(available) if available else "(none)"
        raise FileNotFoundError(
            f"Requested DEP inference checkpoint does not exist: {weight}\n"
            f"Available PyTorch checkpoints:\n  {available_text}"
        )
    if use_tensorrt:
        metadata_path = Path(str(weight) + ".metadata.json")
        if metadata_path.is_file():
            with metadata_path.open("r", encoding="utf-8") as metadata_file:
                declared = json.load(metadata_file).get("backbone_variant")
            if declared != variant:
                raise ValueError(
                    f"TensorRT variant mismatch: artifact={declared!r}, requested={variant!r}"
                )
        elif variant != "legacy":
            raise FileNotFoundError(
                f"Corrected TensorRT artifact requires variant metadata: {metadata_path}"
            )
    return str(weight)


if __name__ == "__main__":
    args = parser().parse_args()
    dynamic_config = DynamicPerceptionConfig.from_global_config()
    dynamic_overrides = {}
    if args.dynamic_enabled is not None:
        dynamic_overrides["enabled"] = bool(args.dynamic_enabled)
    if args.dynamic_source is not None:
        dynamic_overrides["source"] = args.dynamic_source
    if dynamic_overrides:
        dynamic_config = replace(dynamic_config, **dynamic_overrides)
        dynamic_config.validate()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    backbone_variant = resolve_backbone_variant(args.backbone_variant)
    print(f"Backbone variant: {backbone_variant}")
    if args.checkpoint is not None:
        weight = args.checkpoint.expanduser().resolve()
        if not weight.is_file():
            raise FileNotFoundError(f"Requested DEP inference checkpoint does not exist: {weight}")
        if args.use_tensorrt:
            # Reuse the normal TensorRT sidecar check with the explicit path below.
            metadata_path = Path(str(weight) + ".metadata.json")
            if metadata_path.is_file():
                with metadata_path.open("r", encoding="utf-8") as metadata_file:
                    declared = json.load(metadata_file).get("backbone_variant")
                if declared != backbone_variant:
                    raise ValueError(
                        f"TensorRT variant mismatch: artifact={declared!r}, "
                        f"requested={backbone_variant!r}"
                    )
            elif backbone_variant != "legacy":
                raise FileNotFoundError(
                    f"Corrected TensorRT artifact requires variant metadata: {metadata_path}"
                )
        weight = str(weight)
    else:
        weight = resolve_weight_path(
            args.use_tensorrt, args.trial, args.epoch, base_dir, backbone_variant
        )
    print("load weight from:", weight)

    settings = {'use_tensorrt': args.use_tensorrt,
                'backbone_variant': backbone_variant,
                'goal': [50, 0, 2],      # 目标点位置
                'goal_z': args.goal_z,
                'arrival_radius': args.arrival_radius,
                'wait_for_goal': bool(args.wait_for_goal),
                'hold_on_arrival': bool(args.hold_on_arrival),
                'runtime_safety_enabled': args.runtime_safety_enabled,
                'deadlock_recovery_enabled': args.deadlock_recovery_enabled,
                'dynamic_network_attention_enabled': bool(
                    args.dynamic_network_attention_enabled
                ),
                'safety_telemetry': args.safety_telemetry,
                'frame_decay_time': args.frame_decay_time,
                'frame_size': args.frame_size,
                'env': 'simulation',     # 深度图来源 ('435' or 'simulation', 和深度单位有关)
                'pitch_angle_deg': -0,   # 相机俯仰角(仰为负)
                'odom_topic': '/sim/odom',                   # 里程计话题
                'depth_topic': '/depth_image',               # 深度图话题
                'ctrl_topic': '/so3_control/pos_cmd',        # 控制器话题
                'plan_from_reference': False,   # 从参考状态规划？位置控制器: True, 神经网络直接控制: False
                'verbose': True,               # 打印耗时？
                'visualize': True               # 可视化所有轨迹？(实飞改为False节省计算)
                }
    
    # 初始化节点
    dep_net = DepNet(settings, weight, dynamic_config=dynamic_config)
