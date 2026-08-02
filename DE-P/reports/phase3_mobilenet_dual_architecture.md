# 阶段 3：MobileNetV3 legacy/corrected 双架构报告

日期：2026-07-20  
工程：`/home/zjh/YOPO/DE-P`

## 1. 结论

- `legacy` 完整保留，并继续作为 YAML、`DepNetwork()`、训练和 ROS 的默认架构。
- 原权重 `saved/DEP_0/epoch10.pth` 未被改写，SHA256 仍为 `615c40c638cf4741d19d232687514ee7e33ffb68b414b8d154263caed5a41223`，在 legacy 模式下 `strict=True` 加载通过。
- legacy 固定输入 golden regression：endstate 最大绝对差为 `0.0`，score 最大绝对差为 `0.0`。
- corrected stem 只有一次 `Conv2d(1→16) → BatchNorm2d → Hardswish`，不存在嵌套、重复 BN 或重复激活。
- legacy→corrected 转换后的 checkpoint 可由 corrected 模型 `strict=True` 加载，但转换只用于初始化，不是数学等价变换，必须重新训练或微调。
- 阶段 2A 与阶段 3 全量 CPU 回归通过；Codex 沙箱未暴露 CUDA，因此阶段 3 GPU 性能为 `NOT MEASURED IN CODEX SANDBOX`。
- 动态注意力、EKF、聚类、点云接口和 ROS 动态感知均未修改，仍未开始动态障碍模块修复。

## 2. 阶段 2A 宿主机基线

用户在真实宿主机已验证 legacy：

- GPU：NVIDIA GeForce RTX 5070 Ti Laptop GPU，compute capability 12.0
- torch：2.7.0+cu128
- checkpoint：`saved/DEP_0/epoch10.pth`
- backbone：`[1,64,3,5]`
- endstate：`[1,9,3,5]`
- score：`[1,3,5]`
- 梯度张量：147
- 峰值显存：76,318,720 bytes
- 平均推理：2.2902528 ms
- P95：2.3416881 ms

这是 legacy 的正式宿主机性能基线。本阶段未在沙箱内推导或伪造 corrected GPU 性能。

## 3. 修改文件

核心和入口：

- `config/traj_opt.yaml`：增加唯一 YAML 默认项 `backbone_variant: legacy`。
- `policy/backbone_variant.py`：统一解析和校验 `legacy/corrected`。
- `policy/models/backbone.py`：保留原 legacy stem，新增 corrected stem。
- `policy/dep_network.py`：接收统一 variant，并在构造时打印架构。
- `policy/checkpoint_utils.py`：checkpoint 类型识别、严格加载、元数据封装和 ROS 启动前预检。
- `policy/dep_trainer.py`：按 variant 构造/加载/保存；corrected 使用带元数据 checkpoint。
- `train_dep.py`：增加 `--backbone-variant` 和任意 `--checkpoint`；区分保存路径。
- `test_dep_ros.py`：默认 legacy，可显式选择 corrected；在 `rospy.init_node` 前拒绝不匹配权重。
- `dep_trt_transfer.py`：统一 variant/checkpoint 参数，延迟导入 torch2trt，写入 TensorRT variant sidecar。

诊断、迁移、测试和报告：

- `tools/inspect_mobilenet_stem.py`
- `tools/convert_legacy_to_corrected.py`
- `tests/fixtures/legacy_forward_reference.pt`
- `tests/test_backbone_variants.py`
- `tests/test_variant_checkpoints.py`
- `tests/test_checkpoint_paths.py`
- `tests/run_backbone_variant_cpu_validation.py`
- `tests/run_host_backbone_variant_validation.py`
- `tests/run_host_backbone_variant_validation.sh`
- `reports/phase3_mobilenet_dual_architecture.md`

未修改 `policy/models/MobileNetV3.py`、`policy/models/head.py`、状态分支、15 个运动基元、loss 数学、动态模块和原版 YOPO。

## 4. 修改前 legacy stem 的实际结构

只读诊断确认 `cnn.features[0]` 为：

```text
ConvBNActivation                         # 外层
├── ConvBNActivation                     # 被塞进原 Conv2d 位置的内层
│   ├── Conv2d(1,16,3,stride=2,padding=1)
│   ├── BatchNorm2d(16, eps=1e-5, momentum=0.1)
│   └── Hardswish
├── BatchNorm2d(16, eps=1e-3, momentum=0.01)
└── Hardswish
```

实际执行顺序：

```text
Conv2d → BatchNorm2d → Hardswish → BatchNorm2d → Hardswish
```

计数：1 个 Conv2d、2 个 BatchNorm2d、2 个 Hardswish，且存在嵌套 `ConvBNActivation`。

## 5. corrected stem

```text
ConvBNActivation
├── Conv2d(1,16,kernel=3,stride=2,padding=1,bias=False)
├── BatchNorm2d(16, eps=1e-3, momentum=0.01)
└── Hardswish
```

实际执行顺序且仅为：

```text
Conv2d → BatchNorm2d → Hardswish
```

后续 inverted residual、SE、下采样、末端 1×1 输出层和 DepHead 保持不变。

## 6. 参数量、shape 和 state_dict

| 项目 | legacy | corrected |
|---|---:|---:|
| DepBackbone 参数量 | 959,392 | 959,360 |
| 完整 DepNetwork 参数量 | 1,046,698 | 1,046,666 |
| state_dict key 数 | 252 | 247 |
| backbone 输出 | `[B,64,3,5]` | `[B,64,3,5]` |
| endstate 输出 | `[B,9,3,5]` | `[B,9,3,5]` |
| score 输出 | `[B,3,5]` | `[B,3,5]` |

legacy 独有 key：

```text
image_backbone.backbone.0.0.0.0.weight
image_backbone.backbone.0.0.0.1.weight
image_backbone.backbone.0.0.0.1.bias
image_backbone.backbone.0.0.0.1.running_mean
image_backbone.backbone.0.0.0.1.running_var
image_backbone.backbone.0.0.0.1.num_batches_tracked
```

corrected 独有 key：

```text
image_backbone.backbone.0.0.0.weight
```

corrected BN 的 5 个 key 与 legacy 外层 BN 字符串名称相同，但语义不同。迁移器因此不会按同名直接复制，而是显式从 legacy 内层、第一次 Hardswish 之前的 BN 映射。

## 7. Legacy 回归

fixture 内容：固定随机种子 314159、固定 depth、固定原始 9 维 obs、legacy endstate、legacy score、PyTorch 版本、checkpoint SHA256、参数量和完整 key 顺序；不包含模型或权重副本。

结果：

- 默认 `DepNetwork()`：legacy
- checkpoint：plain state_dict，识别为 legacy
- strict load：PASS，missing `[]`，unexpected `[]`
- 参数量：1,046,698，与修改前一致
- key 数和顺序：与修改前一致
- endstate max abs diff：`0.0`
- score max abs diff：`0.0`

## 8. Checkpoint 识别和加载规则

识别依据：

- 含 `image_backbone.backbone.0.0.0.0.weight`：legacy
- 含 `image_backbone.backbone.0.0.0.weight` 且不含 legacy 独有 key：corrected
- 两者同时存在或都不存在：拒绝
- 元数据 variant 与 key 推断不一致：拒绝

加载器兼容旧 plain state_dict 和新 `{state_dict, metadata}` 格式。所有允许的 PyTorch 模型加载都使用 `strict=True`，并打印检测 variant、请求 variant、missing keys、unexpected keys 和最终结果。

| 模型 | checkpoint | 结果 |
|---|---|---|
| legacy | legacy | strict=True 允许 |
| corrected | corrected | strict=True 允许 |
| corrected | legacy | 拒绝并要求先迁移 |
| legacy | corrected | 拒绝 |
| 任意 | 无法识别 | 拒绝并显示 key 样本 |

## 9. legacy → corrected 映射策略

策略名：`inner_conv_and_pre_activation_bn`。

特殊映射 6 项：

```text
legacy ...0.0.0.0.weight              → corrected ...0.0.0.weight
legacy ...0.0.0.1.weight              → corrected ...0.0.1.weight
legacy ...0.0.0.1.bias                → corrected ...0.0.1.bias
legacy ...0.0.0.1.running_mean        → corrected ...0.0.1.running_mean
legacy ...0.0.0.1.running_var         → corrected ...0.0.1.running_var
legacy ...0.0.0.1.num_batches_tracked → corrected ...0.0.1.num_batches_tracked
```

其中 `...` 为 `image_backbone.backbone.0.0.`。

直接复制 241 项：所有 stem 之外、名称和 shape 完全一致的 target key。精确分组为：

```text
features[1] 16 keys; features[2] 18; features[3] 18
features[4..11] each 22 keys; features[12] 6 keys
image_backbone output Conv2d 1 key; DepHead 6 keys
```

迁移脚本会在控制台打印完整 241-key 列表，并把完整列表写入新 checkpoint metadata。

明确丢弃 legacy 外层、第一次 Hardswish 之后的 BN 5 项：

```text
image_backbone.backbone.0.0.1.weight
image_backbone.backbone.0.0.1.bias
image_backbone.backbone.0.0.1.running_mean
image_backbone.backbone.0.0.1.running_var
image_backbone.backbone.0.0.1.num_batches_tracked
```

初始化而未映射：`[]`。转换后 corrected model strict load：PASS。

原 checkpoint 不会被覆盖；输入输出路径相同或输出已存在时，工具会拒绝执行。

### 为什么不是数学等价转换

legacy 的两个 BN 之间存在 Hardswish，第二个 Hardswish 又位于外层 BN 之后。非线性激活不能像纯线性/仿射层那样折叠到单个 Conv-BN 中。因此丢弃外层 BN 后必然改变函数；迁移结果只是更合理的 corrected 初始值。

## 10. 同输入迁移差异

CPU、固定 fixture、比较 `legacy + epoch10` 与 `corrected + converted init`：

| 指标 | 值 |
|---|---:|
| endstate MAE | 2.1262066364 |
| endstate 最大绝对差 | 8.8875217438 |
| endstate cosine similarity | 0.7812860608 |
| score MAE | 1.8185843229 |
| score 排名位置变化数 | 12 / 15 |
| 最优 primitive 是否相同 | 是（仅此固定样本） |

该结果说明迁移不能作为等价部署权重；最优 primitive 在此样本相同也不能外推到其他输入。

## 11. 训练行为

- 默认 legacy 保存目录继续为 `saved/DEP_<trial>/`，避免破坏现有路径。
- corrected 保存目录为 `saved/DEP_corrected_<trial>/`。
- 新 corrected 训练 checkpoint 使用带 metadata 的格式；legacy 保留 plain state_dict。
- corrected 直接加载 legacy 会在训练开始前失败。
- corrected 可从零训练，也可通过 `--checkpoint` 加载转换后的初始化。
- 本阶段仅执行单 batch 冒烟，没有启动完整 epoch。

示例：

```bash
python train_dep.py --backbone-variant legacy --pretrained 1 --trial 0 --epoch 10
python train_dep.py --backbone-variant corrected --pretrained 0
python train_dep.py --backbone-variant corrected \
  --checkpoint saved/DEP_corrected_init/epoch10_converted.pth
```

## 12. ROS 行为

- 默认仍是 `legacy + saved/DEP_0/epoch10.pth`，旧命令和话题、PositionCommand、50 Hz 控制逻辑不变。
- corrected 可显式指定；转换初始化路径可通过 `--checkpoint` 使用。
- PyTorch checkpoint 的 variant 会在 `rospy.init_node` 之前预检，不匹配时无需 ROS Master 即明确失败。
- TensorRT corrected 使用独立默认文件 `dep_trt_corrected.pth`，且必须有 `.metadata.json` sidecar；legacy 继续兼容原 `dep_trt.pth`。

```bash
python test_dep_ros.py
python test_dep_ros.py --backbone-variant corrected \
  --checkpoint saved/DEP_corrected_init/epoch10_converted.pth
```

未接入动态点云，未改变 ROS 订阅、发布和控制频率。

## 13. TensorRT

`dep_trt_transfer.py` 现在使用与训练/ROS 相同的 variant 解析和 checkpoint 严格加载；torch2trt 延迟到实际转换时导入，因此缺少 TensorRT 不会被误报为网络导入失败。转换产物旁写入架构和源 checkpoint sidecar。

```bash
python dep_trt_transfer.py --backbone-variant corrected \
  --checkpoint saved/DEP_corrected_init/epoch10_converted.pth \
  --dir dep_trt_corrected.pth
```

沙箱未安装/运行 TensorRT，本阶段只验证了参数解析、模型构造和导入路径。

## 14. CPU 测试

双架构专项：

```text
Ran 12 tests in 2.256s
OK
CORRECTED_CPU_GRADIENT_TENSOR_COUNT: 145
```

阶段 2A + 阶段 3 全量回归：

```text
Ran 54 tests in 7.988s
OK (skipped=1, expected failures=1)
```

- legacy golden regression：PASS
- legacy checkpoint strict load：PASS
- corrected stem 结构：PASS
- corrected forward/backward/optimizer step：PASS
- 所有 corrected 输出和梯度有限：PASS
- checkpoint 交叉加载拒绝：PASS
- plain/wrapped checkpoint 识别：PASS
- 转换、源文件哈希不变、映射和 strict load：PASS
- 动态模块既有 expected failure：保留

## 15. GPU 状态和宿主机验证入口

Codex 沙箱：

```text
torch: 2.7.0+cu128
CUDA build: 12.8
cuda_available: False
NOT MEASURED IN CODEX SANDBOX
```

没有重装或更改 torch、torchvision、CUDA 或驱动。

宿主脚本：

- `tests/run_host_backbone_variant_validation.py`
- `tests/run_host_backbone_variant_validation.sh`

脚本输出 `HOST_BACKBONE_VARIANT_VALIDATION_RESULT` JSON，覆盖：legacy 原权重严格加载和 golden、corrected 随机初始化、corrected 转换权重严格加载、两种 corrected 的 forward/backward/optimizer step、shape、有限值、梯度数、参数量、平均/P95、峰值显存和转换差异。

### 2026-07-21 宿主首次执行校正

首次宿主执行确认 CUDA 可用且 legacy checkpoint strict load 通过，但旧脚本错误地用 `1e-4` 硬门限比较 CPU fixture 和 CUDA/cuDNN 输出，得到：

```text
endstate CPU↔CUDA max abs diff: 0.0341618061
score CPU↔CUDA max abs diff:    0.0216846466
```

fixture 明确生成于 CPU；宿主 CUDA 走不同的 cuDNN 内核和浮点累计顺序，并且环境默认允许 cuDNN TF32。跨设备差异不能证明 legacy 结构回退。同期同设备 CPU golden 仍为 endstate `0.0`、score `0.0`。

验证脚本现已改为：

1. 使用 `CUDA_VISIBLE_DEVICES=""` 的隔离 CPU 子进程执行权威 golden regression，硬门限仍为 `1e-7`；
2. GPU 执行严格加载、shape、有限值、训练步、显存和性能测试；
3. CPU↔CUDA 差异写入 `cpu_cuda_numerical_difference`，标记为诊断项，不再作为架构回归门限；
4. 不关闭 TF32 来伪装成日常部署性能，GPU 性能仍按当前宿主实际运行配置测量。

## 16. 宿主机完整执行步骤

### 第一步：进入目录

```bash
cd /home/zjh/YOPO/DE-P
```

### 第二步：激活环境

```bash
conda activate yopo
```

### 第三步：确认旧 checkpoint

```bash
ls -lh saved/DEP_0/epoch10.pth
```

### 第四步：执行转换

若目标文件已经存在，请先使用一个新的输出文件名；工具不会覆盖已有输出。

```bash
python tools/convert_legacy_to_corrected.py \
  --input saved/DEP_0/epoch10.pth \
  --output saved/DEP_corrected_init/epoch10_converted.pth
```

### 第五步：运行双架构 GPU 验证

```bash
bash tests/run_host_backbone_variant_validation.sh
```

### 第六步：成功标准

- JSON 顶层 `status` 为 `PASS`。
- legacy 原 checkpoint `strict_load: true`，隔离 CPU 的 `golden_regression_cpu.status` 为 `PASS`，差值不超过 `1e-7`。
- corrected converted checkpoint `strict_load: true`。
- 三种模型 shape 正确且无 NaN/Inf。
- 两种 corrected 模型的 optimizer step 为 PASS、梯度有限。
- 三组 `peak_gpu_memory_bytes` 均大于 0。
- average/P95 有实际值。
- `cpu_cuda_numerical_difference` 明确报告跨设备数值差异但不把它误判为结构回退。
- `comparison_before_optimizer_step` 明确报告 legacy/corrected 差异。

常见失败处理：

- `converted checkpoint not found`：先执行第四步。
- `Refusing to overwrite existing output`：保留已有文件并换新输出名，或确认后手工选择另一路径；不要覆盖原 legacy 权重。
- variant mismatch：检查 `--backbone-variant` 与权重 metadata/key 是否一致。
- CUDA unavailable：检查当前是否确实为宿主 `yopo` 环境及 NVIDIA 设备是否可见，不要重装 torch/CUDA。

## 17. 动态模块边界

本阶段没有修改：

- `policy/models/EkfDynPercept.py`
- `policy/models/MonteCarloCutting.py`
- `AttentionDepBackbone`
- 动态注意力、跟踪、数据关联、点云投影
- ROS 动态点云订阅或动态避障接线

阶段 1 已复现的动态模块失败和算法风险继续存在。

## 18. 最终回答

- legacy 模式是否完全保留：是，结构、参数名、参数量、key 和 golden 数值均保留。
- 原 `epoch10.pth` 是否继续严格加载：是，`strict=True`，missing/unexpected 均为空。
- corrected stem 是否只有一次 Conv-BN-Hardswish：是。
- 转换后的 corrected checkpoint 是否严格加载：是。
- 转换是否仅用于初始化：是，不保证函数等价，必须训练或微调。
- 是否可以开始 corrected 短程训练：CPU 入口和安全加载已具备；建议宿主阶段 3 GPU 双架构验证 PASS 后开始。
- 是否已开始动态障碍模块修复：否。
