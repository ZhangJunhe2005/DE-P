# 阶段 4：独立动态障碍感知核心报告

日期：2026-07-21  
工程：`/home/zjh/YOPO/DE-P`

## 1. 结论

阶段 4 已建立不依赖 ROS、不依赖 DepNetwork 的连续动态感知接口：

```python
result = dynamic_perception.update(
    point_cloud_camera=point_cloud_camera,
    camera_pose_world=camera_pose_world,
    timestamp=timestamp,
    camera_model=camera_model,
)
```

结果包含 `all_tracks`、`confirmed_tracks`、`dynamic_tracks`、`projected_dynamic_tracks`、`attention_map` 和 diagnostics。正式实现使用标准 DBSCAN、世界坐标系、线性 Kalman Filter、gated Hungarian 一对一关联、稳定单调 track ID 和动态迟滞判定。

CPU 全量回归：

```text
Ran 83 tests in 8.136s
OK (skipped=1, expected failures=1)
```

其中阶段 4 动态专项 36 项全部通过。旧基线中的 1 个 CUDA skip 只代表 workspace sandbox；本阶段新增 GPU 脚本已通过 full-access 在宿主 RTX 5070 Ti 上实际运行并得到 `status: PASS`。

本阶段没有接入 DepNetwork、CNN、训练 Dataset、动态 loss、ROS 或控制器，也没有开始阶段 5。

## 2. 修改文件

正式模块：

```text
policy/dynamic/
├── __init__.py
├── types.py
├── camera_model.py
├── pointcloud.py
├── clustering.py
├── kalman_tracker.py
├── association.py
├── track_manager.py
├── projection.py
├── attention.py
└── dynamic_perception.py
```

配置和依赖说明：

- `config/traj_opt.yaml`
- `requirements-dynamic.txt`

兼容入口：

- `policy/models/MonteCarloCutting.py`
- `policy/models/EkfDynPercept.py`
- `policy/models/backbone.py` 中的实验 `AttentionDepBackbone`

测试和宿主入口：

- `tests/dynamic_helpers.py`
- `tests/test_dynamic_pointcloud_clustering.py`
- `tests/test_dynamic_kalman.py`
- `tests/test_dynamic_tracking.py`
- `tests/test_dynamic_projection_attention.py`
- `tests/test_dynamic_perception_interface.py`
- `tests/test_dynamic_failures.py`（已从失败复现改为兼容边界测试）
- `tests/run_host_dynamic_perception_validation.py`
- `tests/run_host_dynamic_perception_validation.sh`

未修改静态 DepNetwork、DepHead、状态分支、运动基元、safety/guidance loss、legacy/corrected stem 和已有 checkpoint。

## 3. 模块边界和强类型结构

公开数据结构全部位于 `policy/dynamic/types.py`：

- `CameraModel`：width、height、fx、fy、cx、cy、depth_scale、min/max depth。
- `Pose`：`position_world`、`rotation_world_from_camera`、timestamp。
- `ClusterObservation`：临时 cluster ID、世界/相机质心、点数、世界包围盒、位置协方差、timestamp。
- `DynamicTrack`：稳定 track ID、世界位置/速度、6×6 协方差、age/hit/missed、confirmed/dynamic、timestamp、动态判定理由。
- `ProjectedDynamicTrack`：track ID、pixel、feature coordinate、相机深度、attention 权重。
- `DynamicPerceptionResult`：完整结果和 diagnostics。
- `DynamicPerceptionConfig`：唯一正式动态配置结构，未知/缺失/非法字段立即失败。

一个 `DynamicPerception` 实例只对应一个时间序列流。训练 batch 中不同序列不得共享 tracker；`reset()` 会清空状态并把 ID 计数器重置为新流的 0。

## 4. 坐标系定义

相机使用标准 optical frame：

```text
+X：图像向右
+Y：图像向下
+Z：相机前方
```

Pose 的方向固定为：

```text
p_world = R_world_from_camera @ p_camera + position_world
```

逆变换为：

```text
p_camera = R_world_from_camera.T @ (p_world - position_world)
```

代码使用带坐标系后缀的字段名，不再把相机系质心位移直接解释为目标速度。外部相机到机体的外参必须由调用方显式合入 `Pose`，本阶段没有猜测外参。

位姿 timestamp 与传感器 timestamp 的差必须不大于 `max_pose_time_difference=0.05 s`；无 Pose、非法旋转或过期 Pose 会在 tracker 更新前失败。

## 5. 深度反投影和外部点云

支持：

1. 外部 `point_cloud_camera: [N,3]`，明确位于 optical camera frame；
2. `depth_to_pointcloud(depth, camera_model, stride)` 从深度图生成点云。

透视反投影：

```text
Z = depth * depth_scale
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
```

- 32FC1 米单位使用 `depth_scale=1.0`；
- 16UC1 毫米使用 `depth_scale=0.001`；
- 过滤 NaN、Inf、0 和 min/max depth 外的值；
- stride 控制点数；
- 输出固定为 `[N,3] float32`；
- NumPy 输入返回 CPU NumPy；Torch 输入保持在原 Torch device；
- CPU DBSCAN/KF 接口拒绝 Torch Tensor，不做隐式 CUDA→NumPy 拷贝。

## 6. 聚类和 cluster_size 修复

正式算法名称：`sklearn.cluster.DBSCAN`。

正式配置来自唯一 YAML 节点：

```yaml
dynamic_perception:
  cluster_eps: 0.35
  cluster_min_samples: 5
  cluster_max_iterations: 300
```

标准 DBSCAN 是穷尽式算法，不使用随机抽样，也没有 iterations 参数。`cluster_max_iterations` 仅作为旧配置迁移时经过范围校验的兼容限制，不传给 DBSCAN；正式 diagnostics 明确报告算法为 sklearn DBSCAN。

噪声标签 `-1` 被正确忽略。每个有效 cluster 输出质心、点数、世界包围盒、质心协方差和 timestamp。

正式 `AttentionDepBackbone` 构造代码不再传 `cluster_size`。旧 wrapper 只有在显式收到 `cluster_size` 时发出 deprecation warning，并映射为 `cluster_min_samples`；未知正式配置包含 `cluster_size` 会立即失败。

旧 `monte_carlo_clustering` 名称保留为带弃用警告的 DBSCAN wrapper，并明确说明原算法从未随机。旧 random、matplotlib 和自定义 dense-seed 实现已退出正式路径。

聚类测试覆盖空点云、全噪声、单目标、双目标、相近但不合并、稀疏噪声和固定输入重复稳定性。

## 7. 线性 Kalman Filter

正式实现使用 `filterpy.kalman.KalmanFilter`，不是 ExtendedKalmanFilter。

状态和观测：

```text
x = [px, py, pz, vx, vy, vz]^T
z = [px, py, pz]^T
```

状态转移与观测矩阵：

```text
F(dt) = [[I3, dt*I3],
         [ 0,    I3]]

H = [I3, 0]
```

白噪声加速度模型的 Q：

```text
Q = sigma_a^2 * [[dt^4/4 I3, dt^3/2 I3],
                 [dt^3/2 I3, dt^2   I3]]
```

R：`measurement_noise² * I3 + cluster centroid covariance`。  
P 初值：位置方差 `0.25`，速度方差 `4.0`。

dt 取 timestamp 实际差值，再限制到 `[min_dt=0.001, max_dt=1.0]`。同时保留 raw dt 和 effective dt 供诊断。重复和乱序 timestamp 明确拒绝；状态转移中没有随机噪声采样。支持连续仅 predict、不 update。

测试覆盖静止、匀速、带噪匀速、单帧缺测、多帧缺测、极小/大 dt、重复/乱序时间、有限状态和对称协方差。

## 8. Ego-motion 补偿

每帧先把所有有效 camera points 转换到世界系，再进行 DBSCAN 和跟踪。四个 16 帧合成场景的最终世界速度：

| 场景 | 估计 velocity_world (m/s) |
|---|---|
| 相机静止、目标静止 | `[0.0, 0.0, 0.0]` |
| 相机 0.5 m/s、目标静止 | `[-6.11e-9, -8.05e-13, -2.45e-12]` |
| 相机静止、目标 0.4 m/s | `[0.4000549, -3.47e-8, -1.06e-7]` |
| 相机 0.25 m/s、目标 0.4 m/s | `[0.4000549, -3.47e-8, -1.06e-7]` |

结果确认相机自身平移不会被误当作目标速度。

## 9. 数据关联和 track 生命周期

关联过程：

1. 按单调 track ID 和临时 cluster ID 排序，结果不依赖 dict 遍历顺序；
2. 每个 track 先用真实 dt 预测；
3. 同时使用 Euclidean distance 和 Mahalanobis squared distance gating；
4. 对有效 cost 使用 Hungarian algorithm；
5. 一个 observation 最多匹配一个 track，一个 track 最多匹配一个 observation；
6. 未匹配 observation 创建新 track；未匹配 track 只 predict 并增加 missed count。

默认门限：

```text
association_distance_threshold = 1.5 m
association_mahalanobis_threshold = 11.345
```

新 ID 单调递增且退出后不复用。track 在 `missed_count > max_missed_frames=3` 时才删除；确认需要 `min_confirmed_hits=3`。测试覆盖新目标、短暂遮挡恢复、超时删除、平行目标、交叉目标、双向互斥和遮挡前后 ID 保持。

## 10. 动态判定迟滞

默认配置：

```text
dynamic_enter_speed = 0.30 m/s
dynamic_exit_speed  = 0.15 m/s
dynamic_min_confirmed_hits = 4
dynamic_max_velocity_std = 0.75 m/s
```

只有 confirmed、命中数足够且速度协方差足够低的 track 才可进入 dynamic。速度达到 enter threshold 后进入；已动态 track 只有降到 exit threshold 才退出；两阈值之间保持状态。

每个输出 track 包含以下当前理由之一：

```text
insufficient_confirmed_hits
velocity_uncertainty_too_high
speed_above_enter_threshold
hysteresis_hold_dynamic
speed_below_exit_threshold
speed_below_enter_threshold
```

非法的 `enter <= exit` 配置立即失败。测试验证阈值附近不闪烁、低 hit 和高不确定性不轻易标动态。

## 11. 投影

世界点先按 Pose 逆变换到 optical camera frame，再执行：

```text
u = fx * X/Z + cx
v = fy * Y/Z + cy
```

过滤相机后方、深度越界和图像边界外点。没有直接把 world x/y 映射到 feature map，也没有忽略 Z。

图像到特征图使用像素中心、`align_corners=False` 约定：

```text
feature_x = (u + 0.5) * feature_width/image_width - 0.5
feature_y = (v + 0.5) * feature_height/image_height - 0.5
```

支持任意 H×W。测试覆盖主点、左右上下方向、后方过滤、边界、世界—相机变换、3×5 映射和有限输出。

## 12. 独立 attention

attention 只读取 confirmed dynamic tracks，输出 `[1,1,H,W]` Torch Tensor，不与 CNN 融合。

单 track 权重结合：

- 与相机距离的指数衰减；
- 径向接近速度；
- hit confidence；
- 位置协方差；
- 当前透视投影位置。

远离相机的目标乘以 0.25；接近目标可增强但最终受 `attention_max` 限制。空间响应为 feature map 上的 Gaussian，多目标使用逐元素 max，不做无界求和。

- 无动态目标：全 0；
- 默认范围：`[0,1]`；
- 任意 feature shape；
- 所有值有限；
- 近且接近的目标权重大于远或远离目标。

本阶段只反映当前投影和接近趋势，不预测未来时空碰撞。

## 13. 旧实验兼容边界

- `PointCloudProcessor` 和 `monte_carlo_clustering`：deprecated DBSCAN wrapper。
- `EKFTracker`：deprecated `LinearKalmanTracker` wrapper，旧 velocity 强制修正被移除。
- `DynamicObstacleAttention`：可导入，但 forward 明确拒绝缺少 Pose/timestamp/intrinsics 的旧接口。
- `AttentionDepBackbone`：构造时发出 unintegrated experimental warning；动态 forward 不会悄悄进入正式路径。

这意味着旧类名仍可定位迁移说明，但错误实验实现不会伪装成已接通的生产功能。

## 14. CPU 测试结果

动态专项 36 项覆盖：

- 深度反投影和明确设备边界；
- 标准 DBSCAN；
- 线性 KF 和 dt；
- Hungarian 关联和生命周期；
- 动态迟滞；
- ego-motion；
- 透视投影；
- bounded attention；
- 独立连续接口、reset、空输入和错误输入；
- 旧类名兼容/弃用行为。

全量 83 项还复测了阶段 2A/3：

- safety 单调、d0 连续、地图外高风险：PASS；
- eval/BN 验证行为：PASS；
- requirements 分层：保留；
- legacy checkpoint strict load 与 golden：PASS；
- corrected/legacy 双架构与迁移：PASS；
- 唯一 expected failure：历史 `YOPOLoss` 名称；
- 唯一 sandbox skip：旧 CUDA 单测，已由下述 full-access 专项覆盖。

## 15. Full-access GPU 结果

执行：

```bash
bash tests/run_host_dynamic_perception_validation.sh
```

最终宿主结果：

```text
status: PASS
device: NVIDIA GeForce RTX 5070 Ti Laptop GPU
compute capability: 12.0
torch: 2.7.0+cu128
CUDA build: 12.8

CUDA depth point cloud: [3839,3], cuda:0, finite
CUDA direct attention: [1,1,3,5], max 1.0
pipeline frames: 12
dynamic tracks: 1
projected dynamic tracks: 1
pipeline attention: [1,1,3,5], cuda:0, max 0.1409107
peak GPU memory: 189952 bytes
```

设备边界：DBSCAN/KF 保持 NumPy CPU；Torch depth projection 和 attention 保持 CUDA。`depth_pointcloud_ms=76.88` 是首次功能调用耗时，不是预热后的性能 benchmark；`pipeline_total_ms=16.89` 也仅用于冒烟，不作为部署性能承诺。

首次 launcher 使用宿主默认 Python 而非 yopo，第二次在 CUDA context 初始化前调用 memory-stat API；脚本现已固定使用 `conda run -n yopo`，并在重置显存统计前显式初始化 CUDA context。最终上述同一脚本完整 PASS。

## 16. 尚未接入和后续接口建议

明确尚未接入：

- DepNetwork forward；
- legacy/corrected backbone feature fusion；
- 训练 Dataset 和 batch 时序；
- 动态 loss；
- ROS depth/point cloud/odometry 同步；
- PositionCommand 或控制频率。

建议下一阶段只通过显式适配器接入：

```text
sensor adapter
  → synchronized point_cloud_camera + Pose + CameraModel + timestamp
  → one DynamicPerception instance per stream
  → DynamicPerceptionResult.attention_map
  → explicit, configurable CNN fusion adapter
```

接入时必须保持 perception state 在网络 batch 之外，禁止多个环境共享 tracker；同时保留静态 DepNetwork 路径和旧 checkpoint 的默认行为。本报告不代表已开始该接入。
