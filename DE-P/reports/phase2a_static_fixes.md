# DE-P 阶段 2A：静态主链路确定性修复报告

日期：2026-07-20  
工程：`/home/zjh/YOPO/DE-P`  
修复前基线：`reports/baseline_before_fix.md`

## 1. 结论

阶段 2A 规定的静态确定性问题已经修复，并保持现有网络参数结构不变：

- `saved/DEP_0/epoch10.pth` 继续通过 `strict=True` 加载；
- CPU backbone、完整 inference、真实单样本、单地图三个 loss、backward 和一次 optimizer step 全部通过；
- safety cost 已变为全局连续、单调不增、远距离趋零；
- 地图边界和地图外不再是零代价，而是配置化高风险未知区域；
- GuidanceLoss 只有一个正式实现，并使用 YAML 的 `guidance_perp_max`；
- 验证模式由 `eval_one_epoch()` 自己切换到 eval，BN 统计在验证前后保持不变；
- checkpoint 统一使用 `DEP_<trial>/epoch<epoch>.pth`，显式缺失不再静默回退；
- ROS 默认权重指向实际存在的 `DEP_0/epoch10.pth`；
- 静态 DepNetwork 不再强制导入 FilterPy、matplotlib 或 scikit-learn；
- requirements 已安全分层，未安装或替换任何包。

Codex 沙箱没有 NVIDIA 设备透传。本报告的 GPU 状态为：

```text
NOT MEASURED IN CODEX SANDBOX
```

没有声称 GPU 已通过，也没有把 CPU 运行的 0 字节当成 GPU benchmark。

## 2. 修改范围

### 核心与配置

- `config/config.py`：用上下文管理器读取 YAML，避免文件句柄泄漏。
- `config/traj_opt.yaml`：接入 `safety_full_ratio`，新增 `out_of_bounds_cost`，删除未使用且语义含混的 `danger_decay_ratio`。
- `policy/dep_trainer.py`：验证自主进入 eval；显式 checkpoint 不存在时直接失败。
- `loss/guidance_loss.py`：合并重复函数，正式使用 `guidance_perp_max`。
- `loss/safety_loss.py`：连续单调代价、地图外高风险处理、配置比例校验。
- `loss/loss_function.py`：打印实际 safety/guidance 配置。
- `policy/models/backbone.py`：动态依赖延迟导入。
- `train_dep.py`：统一 checkpoint 目录和显式解析。
- `test_dep_ros.py`：默认 `DEP_0/epoch10.pth`、加载前检查、缺失时列出可用权重。

### 依赖说明

- `requirements.txt`
- `requirements-core.txt`
- `requirements-dynamic.txt`
- `requirements-dev.txt`
- `requirements-ros.txt`

### 测试与宿主入口

- 更新：`tests/test_losses.py`
- 更新：`tests/test_network.py`
- 更新：`tests/test_training_step.py`
- 新增：`tests/test_guidance.py`
- 新增：`tests/test_train_eval_modes.py`
- 新增：`tests/test_checkpoint_paths.py`
- 新增：`tests/test_static_optional_imports.py`
- 新增：`tests/run_host_gpu_validation.py`
- 新增：`tests/run_host_gpu_validation.sh`

没有修改：

- `policy/models/MobileNetV3.py`
- 普通 `DepBackbone` 的有效网络层或参数名
- `policy/dep_network.py`
- `policy/models/head.py`
- `policy/state_transform.py`
- `saved/DEP_0/epoch10.pth`
- 动态注意力、EKF、聚类或动态数据接口
- `/home/zjh/YOPO/YOPO`

## 3. 修改前后测试

修改前重新执行阶段 1 基线：

```text
Ran 21 tests
OK (skipped=1, expected failures=3)
```

三个 expected failure 是：旧 `YOPOLoss` 符号、safety 单调性、safety 在 d0 的连续性。

修改后完整测试：

```bash
PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=/tmp/dep-phase2a-mpl \
conda run --no-capture-output -n yopo \
python tests/run_baseline.py
```

结果：

```text
Ran 36 tests in 6.694s
OK (skipped=1, expected failures=1)
```

- 唯一 skip：CUDA，标记为 `NOT MEASURED IN CODEX SANDBOX`。
- 唯一 expected failure：DE-P 不导出原版名称 `YOPOLoss`，实际正式类名为 `DEPLoss`。
- 动态 backbone/EKF 的失败复现仍保留；这些测试通过表示“指定失败被精确复现”，不表示动态算法可运行。

## 4. 训练和验证模式

### 根因

训练循环在每轮前调用了 `policy.train()`，但验证入口自身不调用 `policy.eval()`。调用者如果遗漏模式切换，BatchNorm 会在验证数据上继续更新 running statistics。

### 修改

`eval_one_epoch()` 在任何验证计算前显式调用：

```python
self.policy.eval()
```

并保留 `@torch.inference_mode()`。下一轮 epoch 进入训练前，训练循环再次显式调用 `self.policy.train()`。

### 验收

- 训练阶段所有 BatchNorm：`training=True`；
- 验证阶段所有 BatchNorm：`training=False`；
- 验证前后 `running_mean`、`running_var`、`num_batches_tracked` 逐项完全相等；
- 连续两轮训练均在 train 模式开始，验证均在 eval 模式执行。

## 5. GuidanceLoss

### 根因

文件中有两个同名 `terminal_aware_similarity_loss()`，Python 只保留第二个。第一个函数不可达，其中对 `guidance_perp_max` 的读取也不生效；真正执行的第二个函数硬编码最大权重 0.6。

### 最终公式

设：

```text
L       = ||goal_dir||
g_hat   = goal_dir / (L + 1e-8)
s       = traj_dir · g_hat
perp    = ||traj_dir - s * g_hat||
w       = guidance_perp_max * clamp(1 - L / goal_length, 0, 1)
loss    = (1 - w) * |L - s| + w * perp
```

配置：

```yaml
guidance_perp_max: 0.5
```

语义：

- `L >= goal_length` 时 `w=0`，远距离允许横向绕行；
- 接近目标时 `w` 平滑增加；
- `L=0` 时 `w=guidance_perp_max`，加强终点横向收敛；
- 权重始终在 `[0, guidance_perp_max]`；
- `goal_dir=0` 和极小目标距离均通过有限 loss/梯度测试。

构造时要求 `guidance_perp_max ∈ [0,1]`，避免混合权重产生负系数。

## 6. SafetyLoss

### 6.1 根因

旧函数在 `d<d0` 使用指数，在 `d>=d0` 使用方向错误的 sigmoid：

- `d0` 左侧约为 1；
- `d0` 及右侧约为 0.05；
- 之后随距离增加而上升；
- 无穷远趋近 0.1。

### 6.2 新代价函数

采用单一可解释指数：

```text
cost(d) = exp(clamp((d0 - d) / r, -60, 60))
```

其中保持原配置：

```text
d0 = 1.2 m
r  = 0.6 m
```

性质：

- 全定义域使用同一公式；
- 在 d0 连续且一阶连续；
- 距离增加时单调下降；
- 负距离/障碍物内部快速增大；
- `d→+∞` 时趋近 0；
- 指数限制避免极端负距离溢出；
- 没有新增网络参数或修改 `wc`。

### 6.3 修改前后采样

| d (m) | 修复前 | 修复后 |
|---:|---:|---:|
| 0.0 | 54.59814835 | 7.38905621 |
| 0.5 | 10.31226063 | 3.21127081 |
| 1.0 | 1.94773436 | 1.39561248 |
| 1.5 | 0.06224594 | 0.60653073 |
| 2.0 | 0.07913915 | 0.26359716 |
| 3.0 | 0.09525742 | 0.04978708 |
| 4.0 | 0.09906840 | 0.00940356 |
| 5.0 | 0.09982271 | 0.00177610 |
| 6.0 | 0.09996647 | 0.00033546 |
| 7.0 | 0.09999366 | 0.00006336 |
| 8.0 | 0.09999881 | 0.00001197 |
| 9.0 | 0.09999978 | 0.00000226 |
| 10.0 | 0.09999996 | 0.0000004269 |

101 点扫描中所有相邻差值均小于等于浮点容差，`cost(10m)=4.2692e-7`，不再趋近 0.1。

### 6.4 d0 连续性和导数

```text
cost(d0 - 1e-5) = 1.0000166893
cost(d0)         = 1.0000000000
cost(d0 + 1e-5) = 0.9999834895
最大相邻跳变      = 1.66893e-5  (< 1e-3)

左侧数值导数 = -1.66809547
右侧数值导数 = -1.66517484
导数差        = 0.00292063
```

导数方向为负：增大距离会降低代价。

### 6.5 地图外和未知区域

新增配置：

```yaml
out_of_bounds_cost: 10.0
```

`grid_sample` 有效区域之外的点现在被赋予该高风险代价，不再 mask 为 0。受控 3×3×3 距离场测试：

```text
地图内远离障碍：4.2692e-7
边界未知区域：  10.0
地图外区域：    10.0
```

未知区域不会比地图内远离障碍区域更安全。

### 6.6 配置一致性

`safety_full_ratio: 0.3` 现正式控制：

```text
0.3 * 完整轨迹成本 + 0.7 * 首次占据样本前成本
```

配置会校验在 `[0,1]`。旧 `danger_decay_ratio` 没有清晰且唯一的数学语义，且新的单一指数已由 `r` 控制衰减，因此本阶段删除该未使用配置，而不是继续留下伪配置。

训练日志现在打印：

```text
full ratio = 0.3000
OOB cost   = 10.0000
guide perp = 0.5000
```

## 7. 距离场术语诊断

真实 `pointcloud-0.ply` 构建结果：

```text
sdf_min       = -1.1661903858
negative_count = 914949
zero_count     = 0
```

距离场确实含负值，所以当前实现继续称为 signed distance field 有依据。本阶段没有重构距离场生成方式。

## 8. Checkpoint 规则

### 训练

最终命名统一为：

```text
saved/DEP_<trial>/epoch<epoch>.pth
```

- `--pretrained 0`：明确从零训练，不解析 checkpoint；
- `--pretrained 1`：打印绝对路径；文件不存在立即抛出 `FileNotFoundError`；
- `DepTrainer` 自身也会检查显式 checkpoint，不能由其他调用入口绕过；
- 不再把缺失 checkpoint 静默解释为从零训练。

### ROS

采用方案 A，默认值为：

```text
--trial 0 --epoch 10
/home/zjh/YOPO/DE-P/saved/DEP_0/epoch10.pth
```

加载前检查文件并打印绝对路径。若文件不存在，错误信息会列出 `saved/DEP_*/epoch*.pth` 中可用权重。默认仍实例化静态 `DepNetwork`，未改变 ROS 消息或网络输入输出。

### 兼容性

阶段 2A 后再次执行：

```text
strict=True load: PASS
missing_keys: 0
unexpected_keys: 0
```

## 9. 静态 backbone 与动态可选依赖

### 根因

普通 `DepBackbone` 不使用动态注意力，但 `backbone.py` 顶层导入 `EkfDynPercept` 和 `MonteCarloCutting`，导致静态网络被迫依赖 FilterPy、scikit-learn 和 matplotlib。

### 修改

动态 import 移入 `AttentionDepBackbone.__init__()`。普通 `DepBackbone` 的代码、层、参数名和 state_dict key 完全未变。

验收：

- 在子进程中主动阻断 FilterPy、matplotlib、scikit-learn 后，`DepNetwork` 仍可导入并完成 CPU forward；
- 构造动态 backbone 且缺依赖时，抛出带 `requirements-dynamic.txt` 指引的明确 ImportError；
- 依赖存在时，原有 `cluster_size` TypeError 仍被精确复现，异常没有被吞掉。

## 10. Requirements 安全分层

`requirements.txt` 现在仅包含强警告和分层入口说明，不再是一键替换环境的危险清单。

- `requirements-core.txt`：静态训练/推理依赖；不含 torch、torchvision、CUDA、empy。
- `requirements-dynamic.txt`：FilterPy、matplotlib 等实验动态依赖。
- `requirements-dev.txt`：说明基线使用标准库 unittest，无需额外测试包。
- `requirements-ros.txt`：明确 ROS Noetic、cv_bridge、genpy、empy 等由 apt/现有工作空间提供，禁止 pip 覆盖。

本阶段没有运行任何 pip 安装，也没有修改现有 Conda、torch、CUDA 或 ROS empy。

## 11. CPU 静态主链路结果

最终单 batch：

```text
device                    = cpu
limited data load         = 0.023431 s
forward + losses          = 0.039071 s
backward + optimizer.step = 0.267774 s

trajectory_loss = 6.0570092201
score_loss      = 4.8608679771
smooth mean     = 0.2730699480
safety mean     = 2.2574396133
guidance mean   = 3.5264995098

gradient tensors       = 147
gradient NaN/Inf       = false
BatchNorm all training = true
```

GPU 峰值显存：

```text
NOT MEASURED IN CODEX SANDBOX
```

## 12. 宿主机 GPU 复测入口

Python 脚本：

```text
/home/zjh/YOPO/DE-P/tests/run_host_gpu_validation.py
```

Shell 包装：

```text
/home/zjh/YOPO/DE-P/tests/run_host_gpu_validation.sh
```

脚本检查：torch/CUDA 版本、CUDA 可用性、GPU 名称、compute capability、checkpoint strict load、backbone、完整 inference、真实单 batch 三个 loss、score loss、backward、optimizer step、有限输出/梯度、峰值显存，以及预热后 100 次 inference 的平均和 P95 延迟。

脚本不会安装包、修改模型、启动 ROS 或运行完整 epoch。CUDA 不可用时会输出诊断并以非零状态退出，不会回退到 CPU。

### 第一步：进入工程目录

```bash
cd /home/zjh/YOPO/DE-P
```

### 第二步：确认 Conda 环境

```bash
conda activate yopo
```

### 第三步：确认 GPU 可用

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

预期第一项为 `True`，设备名称为 NVIDIA GeForce RTX 5070 Ti Laptop GPU。

### 第四步：运行 GPU 验证脚本

推荐：

```bash
bash tests/run_host_gpu_validation.sh
```

也可直接运行：

```bash
conda run --no-capture-output -n yopo python tests/run_host_gpu_validation.py
```

### 第五步：如何判断成功

日志应同时满足：

- `cuda_available: True`；
- GPU 名称和 compute capability 正常输出；
- `checkpoint_strict_load: PASS`；
- `gpu_forward: PASS`；
- 最终出现 `HOST_GPU_VALIDATION_RESULT`；
- JSON 中 `status` 为 `PASS`；
- forward、三个 loss、score loss、backward、optimizer step 无异常；
- `peak_gpu_memory_bytes > 0`；
- `inference_average_ms` 和 `inference_p95_ms` 为正数且正常输出；
- 无 NaN/Inf 或缺失梯度错误。

### 第六步：如果失败怎么办

#### CUDA=False

先运行：

```bash
nvidia-smi
conda activate yopo
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

确认是在普通宿主终端而不是 Codex 沙箱中运行。不要重装 torch、CUDA 或驱动；先保留完整输出用于定位环境激活或设备透传问题。

#### checkpoint 找不到

检查：

```bash
ls -lh /home/zjh/YOPO/DE-P/saved/DEP_0/epoch10.pth
```

不要用其他 checkpoint 覆盖现有权重。

#### OOM

当前验证 batch 已固定为 1。先关闭其他占用 GPU 的程序并用 `nvidia-smi` 检查显存；不要通过改网络结构规避。若仍 OOM，请提供完整日志和显存占用。

#### 其他错误

保存从 `torch_version` 开始到 traceback 结束的完整日志，并同时提供：

```bash
nvidia-smi
conda list | grep -E 'torch|cuda|open3d|filterpy'
```

不要在没有定位根因前执行 requirements 全量安装。

## 13. 本阶段明确未修复

- MobileNetV3 第一层 legacy 重复 BN/激活；
- `AttentionDepBackbone` 的 `cluster_size` 参数；
- FilterPy EKF API；
- EKF/KF 模型选择；
- ego-motion 补偿；
- 动态数据关联、track 生命周期；
- 点云聚类；
- 相机投影和动态注意力；
- 动态训练数据和 ROS 点云；
- 自适应全状态规划；
- 两阶段 yaw。

## 14. 下一阶段判断

可以开始设计 MobileNetV3 legacy/corrected 双架构，但建议先由用户在宿主终端运行本报告第 12 节的 GPU 验证并提供输出。

原因是当前 `epoch10.pth` 与 legacy 结构严格匹配。下一阶段如果修正第一层，必须：

1. 保留 legacy 模式用于旧 checkpoint；
2. 新增 corrected 模式而不是原地改变旧结构；
3. 明确两种 state_dict 规则或提供可验证转换；
4. 分别测试 CPU/GPU forward、BN 结构和 checkpoint 兼容性；
5. 不把 corrected 随机初始化结果与 legacy 已训练权重混用。
