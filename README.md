# DE-P：基于 YOPO 的轻量化与动态避障增强

> 本仓库在原版 YOPO 基础上进行二次开发，核心改动为：
> 1) 将主干 CNN 替换/扩展为 **MobileNetV3** 体系；
> 2) 增加面向动态场景的障碍感知与避障能力（以**蒙特卡洛思想的时空证据聚合 + 线性 Kalman 跟踪**为主线）。

## 1. 项目简介

DE-P（`ZhangJunhe2005/DE-P`）是一个面向无人机实时规划的实验与工程仓库，继承 YOPO 的“一阶段学习规划”范式，并在网络轻量化和动态障碍处理方面做了系统增强。

仓库描述：`DE-P route planning algorithm based on YOPO`

## 2. 相比原版 YOPO 的主要改动

### 2.1 网络骨干：引入 MobileNetV3

代码中已包含完整 MobileNetV3 实现与接入：

- `DE-P/policy/models/MobileNetV3.py`
- `DE-P/policy/models/backbone.py`
- `DE-P/policy/dep_network.py`

并在阶段报告中明确记录了 MobileNetV3 双架构（legacy/corrected）演进：

- `DE-P/reports/phase3_mobilenet_dual_architecture.md`

这部分改动目标是：在保持规划质量的前提下，降低模型计算与部署负担，提升嵌入式/边缘平台可用性。

### 2.2 动态障碍能力：蒙特卡洛 + Kalman 跟踪链路

动态模块位于：

- `DE-P/policy/dynamic/`

关键能力包括：

1. **动态感知统一入口**（独立于主干网络前向）
   - `dynamic_perception.update(...)`
   - 参考：`DE-P/reports/phase4_dynamic_perception_core.md`

2. **聚类与时空证据聚合**
   - 支持深度/点云来源、连通域/聚类处理、前景支持度统计。
   - 工程中包含“MonteCarlo”兼容/过渡入口（如 `policy/models/MonteCarloCutting.py`）。

3. **多目标关联与跟踪**
   - 一对一 gated Hungarian 关联
   - 线性常速度 Kalman Filter（状态 `[x,y,z,vx,vy,vz]`）
   - 核心实现：`DE-P/policy/dynamic/kalman_tracker.py`

4. **动态判定与注意力投影**
   - 依据速度阈值、确认次数、连续命中、可见性等策略做动态迟滞判定
   - 将动态结果投影为网络可消费的 attention 上下文

> 说明：当前实现是“**蒙特卡洛式不确定性/证据处理 + 线性 Kalman 目标状态估计**”的工程化组合，相关 EKF/实验入口也在仓库中保留（如 `policy/models/EkfDynPercept.py`）。

## 3. 总体代码结构（重点）

```text
DE-P/
├── policy/
│   ├── dep_network.py
│   ├── models/
│   │   ├── MobileNetV3.py
│   │   ├── backbone.py
│   │   ├── MonteCarloCutting.py
│   │   └── EkfDynPercept.py
│   └── dynamic/
│       ├── dynamic_perception.py
│       ├── kalman_tracker.py
│       ├── association.py
│       ├── track_manager.py
│       ├── attention.py
│       ├── projection.py
│       └── types.py
├── config/
│   └── traj_opt.yaml
├── tests/
└── reports/
```

## 4. 环境与安装（沿用 YOPO 工程流）

建议环境：Ubuntu 20.04 + ROS Noetic + CUDA + Conda。

### 4.1 克隆

```bash
git clone https://github.com/ZhangJunhe2005/DE-P.git
cd DE-P
```

### 4.2 Python 环境

```bash
conda create -n yopo python=3.8 -y
conda activate yopo
pip install -r requirements.txt
```

> 若使用动态模块，请额外确认 `filterpy`、`scikit-learn`、`numpy`、`scipy` 等依赖已安装；必要时可参考仓库中的动态依赖说明文件与测试脚本。

### 4.3 ROS/仿真依赖

工程包含 `Controller/`、`Simulator/` 相关流程（继承 YOPO 运行方式），可按各子模块说明分别 `catkin_make`。

## 5. 运行说明（最小路径）

### 5.1 静态/基础规划推理

按原 YOPO 路径运行测试脚本（示例）：

```bash
python test_yopo_ros.py --trial=1 --epoch=50
```

### 5.2 启用动态感知（实验）

动态配置集中在：

- `config/traj_opt.yaml`

其中 `dynamic_perception` 段控制是否启用、输入源、关联门限、跟踪参数、attention 融合等。

建议先跑动态单测与集成测试，再进行 ROS 闭环：

- `tests/test_dynamic_kalman.py`
- `tests/test_dynamic_tracking.py`
- `tests/test_dynamic_perception_interface.py`
- `tests/test_dynamic_network_integration.py`
- `tests/test_dynamic_ros_bridge.py`

## 6. 方法说明（简版）

### 6.1 MobileNetV3 轻量骨干

- 使用 MobileNetV3 的深度可分离卷积与 SE/HS 机制降低计算开销。
- 在 DE-P 中提供了 legacy/corrected 兼容路径，便于和历史权重、历史行为做对照验证。

### 6.2 蒙特卡洛 + Kalman 的动态避障链路

1. 从深度/点云中提取候选动态目标；
2. 基于时序证据与统计阈值进行前景/动态候选筛选（蒙特卡洛思想体现在随机化与不确定性处理范式）；
3. 用匈牙利匹配进行观测-轨迹关联；
4. 用线性 Kalman 估计目标位置/速度；
5. 通过动态迟滞和置信度策略输出稳定动态目标；
6. 将动态风险映射为规划注意力，引导轨迹选择与避障。

## 7. 结果与阶段报告

仓库中保留了较完整的阶段性工程报告，建议从以下文档开始：

- `DE-P/reports/phase3_mobilenet_dual_architecture.md`
- `DE-P/reports/phase4_dynamic_perception_core.md`
- `DE-P/reports/phase5_dynamic_network_ros_integration.md`

这些文档记录了：改动边界、测试覆盖、已验证能力与未纳入范围。

## 8. 注意事项

1. 动态模块与静态主干支持分层开关，默认配置可能仍以静态路径为主；请以 `traj_opt.yaml` 实际参数为准。
2. legacy/corrected 架构切换可能影响 checkpoint 兼容性，请遵循仓库中的转换/加载约束。
3. 真实机部署前，建议先完成：
   - 单元测试
   - GPU 主机集成验证
   - ROS 时序同步验证
   - 传感器外参/深度尺度标定核对

## 9. 致谢

- 原始 YOPO 工作与开源实现为本项目提供了坚实基础。
- 本仓库在其上持续推进轻量化与动态避障方向的工程化实践。
