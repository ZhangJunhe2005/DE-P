# DE-P 阶段 5：动态网络与 ROS 接入报告

日期：2026-07-21  
工程：`/home/zjh/YOPO/DE-P`

## 1. 结论

阶段 4 的动态感知结果已经通过正式 `DynamicContext` 接入 `DepNetwork`、普通 `DepBackbone` 的可选注意力融合以及 ROS 推理节点。默认配置仍为：

```yaml
dynamic_perception:
  enabled: false
```

因此默认训练、PyTorch ROS 推理和静态 TensorRT 路径仍是阶段 3 的静态路径。动态开启时，tracker 只存在于 ROS/预处理侧，网络只读取有界 Torch attention；网络 forward 内没有 DBSCAN、Kalman、FilterPy、scikit-learn 或 NumPy 聚类。

CPU 全量回归通过 93 项测试；宿主机 RTX 5070 Ti 的最终 GPU 集成验证为 PASS。没有启动完整训练，也没有在无 ROS Master 时伪造仿真结果。

## 2. 修改文件

核心和配置：

- `config/traj_opt.yaml`
- `policy/dep_network.py`
- `policy/models/backbone.py`
- `policy/dynamic/__init__.py`
- `policy/dynamic/types.py`
- `policy/dynamic/dynamic_perception.py`
- `policy/dynamic/context.py`（新增）
- `policy/dynamic/ros_bridge.py`（新增）
- `test_dep_ros.py`

测试、脚本和报告：

- `tests/test_dynamic_network_integration.py`（新增）
- `tests/test_dynamic_ros_bridge.py`（新增）
- `tests/test_dynamic_perception_interface.py`
- `tests/run_host_dynamic_network_validation.py`（新增）
- `tests/run_host_dynamic_network_validation.sh`（新增）
- `scripts/run_dynamic_ros_smoke.sh`（新增）
- `reports/phase5_dynamic_network_ros_integration.md`（新增）

没有修改 `/home/zjh/YOPO/YOPO`、阶段 2A loss、原 checkpoint、Controller 或 Simulator。

## 3. 统一配置

`DynamicPerceptionConfig.from_global_config()` 是训练、网络、动态核心、ROS 和测试的唯一解析入口。YAML 包含：

- 总开关：`enabled=false`；
- 输入：`source=depth|pointcloud`；
- 融合与回退：`use_attention`、`fallback_to_static`、`attention_alpha`；
- 调试：`publish_debug`；
- 同步：`max_pose_time_offset`、`sync_queue_size`、`sync_slop`；
- topic：depth、pointcloud、CameraInfo、odom；
- 相机模型：尺寸、内参、深度比例和有效范围；
- frame/extrinsic：world frame、optical-camera frame、body 到 camera 的显式外参；
- 阶段 4 的聚类、关联、跟踪、动态判定和 attention 参数。

非法 source、非布尔开关、未知或缺失字段、非法内参、非法旋转矩阵、非法 attention 范围均立即报错。`DepNetwork` 和 ROS 启动时打印完整动态配置。

当前 Simulator 源码的显式相机配置为 `160×90, fx=80, fy=80, cx=80, cy=45`；DEP 网络输入仍在原规划路径中 resize 为 `160×96`。这两种尺寸职责不同：前者用于真实深度反投影，后者用于 CNN。

## 4. DynamicContext

正式结构在 `policy/dynamic/context.py`：

```text
DynamicContext
├── attention_maps_by_level: Mapping[str, Tensor[B,1,H,W]]
├── dynamic_tracks: Tuple[DynamicTrackSummary, ...]
├── timestamp: float | None
├── source: none | depth | pointcloud | test
├── valid: bool
└── diagnostics: JSON-compatible Mapping
```

网络当前只读取 `attention_maps_by_level["backbone_output"]`。track 会先转换为只含数值和 tuple 的 `DynamicTrackSummary`，不会把 ROS message、tracker 或 NumPy 跟踪对象传入 GPU 网络。diagnostics 会转换为 JSON 可序列化值。

batch 语义是严格的：attention 必须是 `[B,1,H,W]`，并且 B 必须与 depth batch 完全一致；不允许把一个时序流的 context 静默广播给其他 batch 样本。构造时检查 shape、NaN/Inf 和 `[0,1]` 边界。

## 5. DepNetwork 和 backbone

正式接口为：

```python
forward(depth, obs, dynamic_context=None)
inference(depth, obs, dynamic_context=None)
```

融合位置：

```text
depth [B,1,96,160]
  -> legacy/corrected MobileNetV3 + 原 1×1 output conv
  -> base feature [B,64,3,5]
  -> feature * (1 + attention_alpha * bounded_attention)
  -> 与 state feature [B,9,3,5] 拼接
  -> 原 DepHead
  -> endstate [B,9,3,5] + score [B,3,5]
```

这是阶段要求允许的“普通 backbone 可选注意力步骤”。正式路径不再使用旧 `AttentionDepBackbone`；该类继续只作为弃用兼容定位，不进入主流程。

融合没有新增模块或训练参数，`attention_alpha=1.0` 是固定配置值。attention 会自动 resize 到目标 feature shape，并在融合前转到 feature 的 device/dtype。全零 attention 与静态 feature 逐元素完全相同。

## 6. 静态兼容和 checkpoint

当 dynamic disabled 时：

- `dynamic_context` 被忽略；
- legacy/corrected 输出 shape 不变；
- state_dict key 和参数量不变；
- legacy 仍为 1,046,698 参数，corrected 仍为 1,046,666 参数；
- 原 `saved/DEP_0/epoch10.pth` 在 legacy 下 strict=True；
- converted corrected checkpoint 在 corrected 下 strict=True；
- 阶段 3 legacy CPU golden 继续通过；
- scikit-learn、FilterPy、matplotlib 被屏蔽时静态网络仍可导入运行。

`policy.dynamic.__init__` 改为 lazy export，避免静态 `DepNetwork` 因导入 context/config 而连带导入动态可选依赖。

静态 TensorRT 的两输入接口和 artifact 规则不变。动态感知开启时 ROS 会明确拒绝静态 TensorRT artifact，不会假装 TensorRT 已支持第三个动态输入。

## 7. ROS 输入与同步

### 7.1 depth 模式

- depth 继续独立驱动规划，因此同步失败或 CameraInfo 缺失时仍能静态回退；
- `message_filters.ApproximateTimeSynchronizer` 同步 depth 与 odom，并可选同步 CameraInfo；
- 支持 `32FC1`、`16UC1` 和 `MONO16`；
- 使用消息原始时间戳；
- 使用 CameraInfo，或 YAML 中明确给出的针孔模型；
- 按 `depth_stride` 反投影到 optical-camera 点云；
- 用同步 odom 和显式 `R_body_from_camera/t_body_camera` 得到 world-camera pose；
- tracker/attention 更新由同步后的传感器帧频率决定，规划仍由 depth 频率决定。

### 7.2 pointcloud 模式

- 同步可配置 PointCloud2、odom 和可选 CameraInfo；
- 只接受 frame 与配置的 optical-camera frame 一致的真实输入点云；
- 不订阅 `/dep_net/*_visual` 等可视化点云作为输入；
- depth 仍独立触发网络规划，并读取时间上有效的最近 DynamicContext。

只读检查当前 Simulator 实现发现：`/lidar_points` 的点值由 world 点转换到机体系后发布，但 message 的 `frame_id` 写成 `odom`。这不是一个可安全解释的 optical-camera PointCloud2。正式 pointcloud 路径会拒绝它并按配置回退，不能把该错误 frame 静默映射。当前 Simulator 不修改的前提下，阶段 5 的可运行动态来源是 depth；pointcloud 实测需先由 Simulator 发布语义正确的 frame/坐标，或增加经过验证的 TF 转换输入。

### 7.3 frame 和时间

- odom header 必须匹配配置 world frame；
- sensor/CameraInfo frame 必须匹配配置 camera optical frame；
- 当前 Simulator depth header 为空，已通过 `allow_empty_sensor_frame=true` 明确声明兼容，而不是误认为某个 frame；
- 任一 odom/CameraInfo 时间差超过 `max_pose_time_offset=0.05s` 时拒绝该动态更新；
- watchdog 检测长期无同步更新并产生限频 warning；
- 原 50 Hz `rospy.Timer(0.02s)` 控制发布没有改变。

## 8. 回退和失败保护

| 情况 | fallback_to_static=true | fallback_to_static=false |
|---|---|---|
| dynamic disabled | 原静态路径 | 原静态路径（开关优先） |
| 无 context/点云/CameraInfo | 标记原因并静态规划 | 停止生成新轨迹，进入明确 safe state |
| 时间不同步/context 过期 | 限频 warning，静态规划 | safe state |
| 动态感知异常 | 无效 context + 静态规划 | safe state |
| 无动态障碍/全零 attention | 精确等价静态 | 精确等价静态 |
| attention 非有限/越界 | context 边界拒绝，静态回退 | 抛错并进入 safe state |
| 网络或 post-process 非有限 | 不发布 NaN 新轨迹 | 不发布 NaN 新轨迹 |

fallback 原因记录在 `DynamicContext.diagnostics` 或 `DepNetwork.last_dynamic_fallback_reason`，ROS 使用 `logwarn_throttle`，异常不会被无声吞掉。

## 9. ROS 调试输出

`publish_debug=false` 为默认值。开启后当前实现发布：

- `/dep_net/dynamic_tracks`：world frame 动态中心 Marker，带有限 lifetime；
- `/dep_net/dynamic_attention`：optical-camera frame mono8 attention 热图；
- `/dep_net/dynamic_perception_ms`：感知耗时；
- `/dep_net/dynamic_fallback`：fallback 状态和原因。

当前阶段没有声称已经发布 track ID 文字、速度箭头或未来位置 Marker；这些仍可在后续可视化阶段增加。调试 topic 不会被重新订阅为输入。

## 10. CPU 测试

最终全量命令：

```bash
source /opt/ros/noetic/setup.bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/dep-phase5-mpl \
  conda run --no-capture-output -n yopo python tests/run_baseline.py
```

结果：

```text
Ran 93 tests in 8.336s
OK (skipped=1, expected failures=1)
```

- skip：受限 Codex sandbox 不可见 CUDA；宿主 GPU 结果见下一节；
- expected failure：工程既有的 `YOPOLoss` 名称不存在，正式类名为 `DEPLoss`；
- 通过内容包括 legacy golden、corrected、strict checkpoint、阶段 2A loss、阶段 4 动态核心、dynamic disabled/空/单目标/多目标/invalid/零/boundary、两种 backbone、CPU forward/backward、严格 batch、无 tracker、ROS mock conversion/sync/frame 检查。

ROS 无 Master smoke：

```text
import_without_ros_master: PASS
control_period_seconds: 0.02
ROS Master not running: topic inspection skipped as designed.
```

未启动仿真器或 ROS Master。

## 11. full-access 宿主 GPU

执行：

```bash
bash tests/run_host_dynamic_network_validation.sh
```

最终结果：

```text
HOST_DYNAMIC_NETWORK_VALIDATION_RESULT status=PASS
device=NVIDIA GeForce RTX 5070 Ti Laptop GPU
compute_capability=[12,0]
torch=2.7.0+cu128
all_outputs_finite=true
peak_gpu_memory_bytes=44,465,152
```

| 路径 | 平均延迟 | P95 | strict load |
|---|---:|---:|---|
| static legacy | 1.3358 ms | 1.6545 ms | PASS |
| static corrected | 1.3383 ms | 1.6282 ms | PASS |
| dynamic corrected | 1.6687 ms | 1.9299 ms | PASS |

动态细节：

- 空 context：精确静态回退；
- 全零 attention：精确静态等价；
- 1 个真实阶段 4 动态 track；
- attention `[1,1,3,5]`，范围 `[0.006666, 0.140806]`；
- attention 生成平均 `0.9831 ms`；
- CPU→GPU attention 传输 `0.0839 ms`；
- 相对静态 endstate 最大差 `0.076331`；
- 相对静态 score 最大差 `0.657862`；
- forward/backward/optimizer step：PASS；
- 有限梯度张量数：145。

上述数值证明阶段 4 attention 已真正改变网络结果，而不是只生成后丢弃。

## 12. 当前训练边界和下一阶段数据格式

本阶段没有加入动态 loss，也没有启动完整训练。现有 `/home/zjh/YOPO/dataset` 是独立静态帧，训练入口不产生 `DynamicContext`；默认 dynamic disabled 时继续按原静态方式训练。converted corrected checkpoint 仍只是初始化权重，不能视为已训练好的 corrected/dynamic 模型。

下一阶段动态训练数据至少需要按连续序列提供：

```text
sequence_id / frame_index / timestamp
raw depth 或 optical-camera point cloud
CameraInfo 或固定标定 ID
world<-body pose + body<-camera extrinsic
当前 9D observation、静态 map_id
动态目标 track/速度标注或可重建的逐帧实例标注
可选未来位置/占据/碰撞时间标签
```

batch collate 必须为每个样本独立生成 `[1,1,H,W]` attention 后组成 `[B,1,H,W]`，不能跨 sequence 共用 tracker。若要训练未来时空避障，还需要统一预测时域及 future trajectory/occupancy 标签；当前 attention 仍只表达阶段 4 的当前投影与接近趋势。

## 13. 验收结论

- 动态感知是否真正进入网络：**是（PyTorch 路径，配置开启且 context 有效时）**。
- 是否保留静态回退：**是，且零 attention/关闭开关为精确静态等价**。
- 是否新增训练参数：**否**。
- 旧权重是否 strict 兼容：**是**。
- 静态 TensorRT 是否保留：**是；动态 TensorRT 尚未实现并会明确拒绝**。
- ROS depth 动态入口：**已接通并通过离线转换/同步检查，尚未启动完整仿真**。
- ROS pointcloud 通用入口：**已接通正确 optical-frame PointCloud2；当前 Simulator 的错误 frame/content 被明确拒绝**。
- 动态 loss：**未加入**。
- 完整训练：**未运行**。
