# DE-P 修复前自动化基线

生成日期：2026-07-20  
工程：`/home/zjh/YOPO/DE-P`  
对照工程：`/home/zjh/YOPO/YOPO`（本阶段未修改）

## 1. 范围与运行方式

本基线只新增 `tests/` 和本报告，没有修改训练、网络、loss、ROS 或动态障碍核心实现。测试不启动 ROS Master，不运行完整 Trainer/epoch，不遍历全部 100,000 张训练图像。

`yopo` 环境未安装 pytest，因此测试使用 Python 标准库 `unittest`，不新增依赖。统一运行命令：

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/dep-baseline-mpl \
conda run --no-capture-output -n yopo python tests/run_baseline.py
```

数据集测试通过运行时 patch 只向真实 `DEPDataset` 暴露地图 0 的前 10 张图；train split 实际加载 9 张。loss 和一步训练只建立真实 `pointcloud-0.ply` 的 ESDF。

## 2. 环境版本

| 项目 | 基线值 |
|---|---|
| Kernel | Linux 5.15.0-139-generic x86_64 |
| glibc | 2.31 |
| ROS | Noetic |
| Python | 3.10.20 |
| PyTorch | 2.7.0+cu128 |
| torchvision | 0.22.0+cu128 |
| PyTorch CUDA build | 12.8 |
| NumPy | 1.22.3 |
| SciPy | 1.10.1 |
| OpenCV | 4.11.0 |
| Open3D | 0.19.0 |
| FilterPy | 1.4.5 |
| scikit-learn | 1.7.2 |
| ruamel.yaml | 0.17.21 |

当前受控执行进程中 `torch.cuda.is_available() == False`，`nvidia-smi` 无法与驱动通信。因此 GPU 前向被条件跳过，GPU 峰值记为 0；这只说明本次测试沙箱未暴露 GPU，不否定宿主机的 RTX 5070 Ti 配置。CPU 基线完整执行成功。

## 3. 汇总

```text
Ran 21 tests in 5.279s
OK (skipped=1, expected failures=3)
```

| 分类 | 数量 | 结果 |
|---|---:|---|
| 普通测试 | 17 | 全部通过 |
| 条件跳过 | 1 | CUDA 前向；当前进程无 CUDA |
| 预期失败 | 3 | YOPOLoss 符号、safety 单调性、safety 连续性 |

动态模块的构造/API 缺陷采用“断言当前必然抛出指定异常”的失败复现测试，所以这些用例在 unittest 汇总中显示为通过；它们不代表动态模块可运行。

## 4. 模块导入结果

| 模块/符号 | 结果 | 备注 |
|---|---|---|
| `config.config.cfg` | PASS | YAML 可加载 |
| `DepNetwork` | PASS | 其导入会连带导入动态依赖 |
| `DepBackbone` | PASS | 普通静态 backbone |
| `DynamicObstacleAttention` | PASS | 类可导入，不代表可完成第二帧 |
| `PointCloudProcessor` | PASS | 可导入 |
| `DepTrainer` | PASS | 未实例化完整数据集/Trainer |
| `DEPDataset` | PASS | 实际类名是全大写 `DEPDataset` |
| `DEPLoss` | PASS | DE-P 当前 loss 类 |
| `YOPOLoss` | EXPECTED FAIL | `loss.loss_function` 不导出该原版类名 |
| `SafetyLoss` | PASS | 仅导入；数值测试另行实例化单地图版本 |
| `SmoothnessLoss` | PASS |  |
| `GuidanceLoss` | PASS |  |
| `test_dep_ros` | PASS | 无 ROS Master；未构造 `DepNet`，未调用 `rospy.init_node/spin` |

ROS 导入打印 `tensorrt not found.`，来自可选 `torch2trt` 导入保护，不阻塞普通 PyTorch ROS 节点的模块导入。

## 5. 网络前向

| 检查项 | 结果 |
|---|---|
| CPU backbone 输入 | `(1,1,96,160)` |
| CPU backbone 输出 | `(1,64,3,5)`，有限 |
| CPU 完整 inference observation | `(1,9)` |
| CPU endstate | `(1,9,3,5)`，无 NaN/Inf |
| CPU score | `(1,3,5)`，无 NaN/Inf |
| CUDA 完整 inference | SKIP：当前进程不可见 CUDA |
| checkpoint | `saved/DEP_0/epoch10.pth` 存在 |
| strict state_dict | PASS：missing keys 0，unexpected keys 0 |

结论：当前权重与当前普通 `DepNetwork/DepBackbone` 架构严格匹配，静态网络至少可在 CPU 完成一次完整 inference。

## 6. 数据集少量真实样本

受控加载耗时：`0.023089 s`（地图 0 前 10 张中 train split 的 9 张，不是全数据集耗时）。

```text
depth shape: (1, 96, 160)
depth dtype: float32
depth range: [0.0519111939, 1.0]
pos shape:   (3,)
rot shape:   (3, 3)
obs shape:   (9,)
map_id:      0 (Python int)
```

结果：

- depth、pos、rot、obs 均无 NaN/Inf；
- 深度在 `[0,1]`；
- `dataset/0/` 存在；
- `dataset/pointcloud-0.ply` 存在；
- map_id 与实际地图、点云对应。

风险：生产 `DEPDataset` 构造仍会全量预加载数据；本测试刻意限制文件可见范围，不能当作全量内存/耗时基准。

## 7. Loss 数值与梯度

简单轨迹、真实地图 0 ESDF 的结果：

```text
smoothness mean = 0.0910074413
safety mean     = 9.3222045898
guidance mean   = 0.9000005126
total           = 10.3132123947
```

三个分量、总损失和终态梯度均为有限值，backward 成功。

### Safety cost 扫描

从 0 m 到 10 m 以 0.1 m 计算，下面每 0.5 m 摘录一次：

| d (m) | cost | d (m) | cost |
|---:|---:|---:|---:|
| 0.0 | 54.59814835 | 5.0 | 0.09982271 |
| 0.5 | 10.31226063 | 5.5 | 0.09992287 |
| 1.0 | 1.94773436 | 6.0 | 0.09996647 |
| 1.5 | 0.06224594 | 6.5 | 0.09998542 |
| 2.0 | 0.07913915 | 7.0 | 0.09999366 |
| 2.5 | 0.08972161 | 7.5 | 0.09999724 |
| 3.0 | 0.09525742 | 8.0 | 0.09999881 |
| 3.5 | 0.09788209 | 8.5 | 0.09999948 |
| 4.0 | 0.09906840 | 9.0 | 0.09999978 |
| 4.5 | 0.09959299 | 10.0 | 0.09999996 |

必现失败：

- **不单调下降**：超过 `d0=1.2 m` 后代价随距离增长，最终逼近 0.1；
- **在 d0 不连续**：`cost(d0-1e-5)≈1.000033`，`cost(d0)=0.05`，跳变约 0.950033。

本阶段仅记录，没有修改公式。

## 8. 动态模块失败复现

| 检查 | 基线结论 |
|---|---|
| `AttentionDepBackbone()` | 必现 `TypeError: unexpected keyword argument 'cluster_size'` |
| `EKFTracker.predict()` | 必现 `TypeError: unexpected keyword argument 'fx'` |
| `EKFTracker.update()` | 必现 FilterPy 参数接口 `TypeError` |
| forward 接口 | `AttentionDepBackbone.forward` 必需 `pcl`，`DepNetwork.forward` 不接受 `pcl` |
| 是否实际随机聚类 | 否；不同 `random.seed` 输出相同，函数源码不调用 `random`/`np.random` |
| 单次关联重复 track | 未复现；`used_pred_ids` 保证同一次 `_match_clusters` 内一个 track 最多分配一次 |
| 数据关联风险 | 仍是无门限、顺序相关的贪心匹配 |
| 目标短暂消失 | 确认一帧无匹配后立即删除 tracker 和 obstacle |

结论：动态类能够导入，但动态 backbone 不能构造；绕过构造参数后，tracker 第二帧仍因 FilterPy API 失败。当前动态模块不能独立完成连续两帧运行。

## 9. 单 batch 训练步骤

测试执行了：一个真实深度样本、网络 forward、三个生产 loss、score label、backward、AdamW `fused=True` 的一次 `optimizer.step()`。

```text
device                    = cpu
limited data load         = 0.023378 s
forward + losses          = 0.019049 s
backward + optimizer.step = 0.131407 s

trajectory_loss = 13.2206792831
score_loss      = 12.0245370865
smooth mean     = 0.27306994796
safety mean     = 9.4211101531
guidance mean   = 3.5264995098

gradient tensors          = 147
gradient NaN/Inf          = false
BatchNorm all training    = true
peak GPU memory           = 0 bytes (CPU run)
```

注意：耗时不包括首次单地图 ESDF 构建；统一测试中 loss 对象被缓存并由 loss 测试先建立。该结果是冒烟基线，不是训练吞吐 benchmark。

## 10. 已通过项

- 核心静态模块及 ROS 文件无 Master 导入。
- 少量真实数据读取、形状、dtype、范围、有限性和 map_id 对应关系。
- CPU backbone 和完整网络前向。
- 现有 checkpoint 严格加载。
- smoothness/safety/guidance 在简单轨迹上输出有限值。
- loss 对终态的 backward 产生有限梯度。
- 单 batch forward/backward/optimizer.step。
- 147 个网络参数张量获得有限梯度。
- 训练步骤中所有 BatchNorm 处于 train 模式。
- 动态失败条件可以稳定、精确复现。

## 11. 必现失败项

- DE-P 不导出请求中的旧类名 `YOPOLoss`；实际类名为 `DEPLoss`。
- safety cost 不随距离单调下降。
- safety cost 在 `d0` 不连续。
- 动态 backbone 的 `cluster_size` 构造参数错误。
- EKF predict/update 的 FilterPy API 错误。
- 动态 backbone 与 `DepNetwork` 点云参数接口不一致。
- 障碍物缺失一帧即删除 track。

## 12. 非必现或本环境未覆盖风险

- CUDA 前向和 GPU 训练步骤：当前沙箱不暴露 GPU，需在宿主 GPU 会话复跑同一命令。
- GPU 动态路径的 NumPy/CUDA Tensor 混用：被更早的构造/API 错误遮挡。
- ROS topic、消息频率和控制发布：未启动 Master/仿真器，按约束不测试。
- 全量 Dataset 构造的内存峰值和耗时：刻意不遍历全数据集。
- 10 张 ESDF 同时构建的 CPU/GPU 内存：只测试地图 0。
- 无门限贪心关联可能造成错误匹配，但单次调用不会把同一个 track 重复分配两次。
- 静态目标因无人机自身运动被误判为动态目标：缺少连续真实传感器数据，本阶段未做在线实验。

## 13. 当前可运行性判断

### 当前权重能否加载

能。`saved/DEP_0/epoch10.pth` 与当前静态网络严格匹配。

### 当前静态 YOPO 主链路能否运行

能完成 CPU 网络 inference 和一个包含真实 ESDF safety loss 的训练步骤，forward、backward、optimizer.step 均成功且数值有限。ROS 模块能在无 Master 时导入。但这不代表默认 ROS checkpoint 参数、完整训练 epoch、仿真闭环或 safety 算法逻辑正确。

### 动态模块能否独立运行

不能。构造参数和 FilterPy API 都是确定性阻塞，且正式网络/ROS 没有点云接口。

## 14. 后续修复验收指标

第一批静态修复建议以以下条件为准：

1. 全部静态导入、CPU 前向、shape、finite、checkpoint 测试继续通过。
2. CUDA 可见环境中 GPU 前向和一步训练通过，所有梯度有限，并记录真实峰值显存。
3. safety cost 在 `[0,10] m` 上单调不增，远距离趋近 0，而不是趋近 0.1。
4. `d0±1e-5` 的最大跳变小于 `1e-2`，后续最好同时约束一阶导数。
5. 一步训练的三个 loss 和总 loss 有限，147 个参数梯度继续有限。
6. 验证入口显式进入 eval，训练入口显式恢复 train；分别新增 BN 状态断言。
7. 现有 checkpoint 的兼容策略明确：若修复 MobileNet 第一层导致键变化，必须提供转换或明确拒绝旧权重。
8. ROS 文件继续能够在无 Master 情况下导入，不在 import 阶段初始化节点或 spin。

动态模块未来验收条件：

1. `AttentionDepBackbone` 可使用合法参数构造。
2. 动态 forward 接口在网络、Dataset、ROS 三处一致，并明确点云 batch/时间戳语义。
3. 连续至少 3 帧执行不出现 FilterPy、NumPy/Tensor 或设备错误。
4. 每个当前 cluster 最多关联一个 track，每个 track 最多关联一个 cluster，并有明确门限。
5. track 短暂缺失 1 帧后保留，超过配置 `max_age` 才删除。
6. 聚类名称和实际算法一致；若声称随机/Monte Carlo，测试应能观测受控随机采样。
7. 加入 ego-motion 和相机投影后，用静态场景运动相机测试静态目标不被误判为动态。

## 15. 新增测试文件

- `tests/run_baseline.py`
- `tests/baseline_helpers.py`
- `tests/test_imports.py`
- `tests/test_network.py`
- `tests/test_dataset_smoke.py`
- `tests/test_losses.py`
- `tests/test_dynamic_failures.py`
- `tests/test_training_step.py`

