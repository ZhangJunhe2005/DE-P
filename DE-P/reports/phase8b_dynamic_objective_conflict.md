# 阶段 8B：动态目标梯度冲突诊断与修复报告

## 结论

阶段 8B 软件 Gate 已重新关闭并通过：`reports/phase8b_preflight_result.json` 为
`status: PASS`、`production_ready: true`。最终方案保留原始候选均值目标，不把
CVaR 加入训练目标，不扩大解冻范围，并采用仅在训练期启用的
`dynamic_priority_pcgrad`；非动态梯度总范数上限为动态梯度范数的 `0.25`。

这是“软件与 bounded preflight 已就绪”，不是“已获准开始生产”。容量计划仍为
`pending_capacity_approval`，216 条正式数据没有生成，production training 也没有启动。

## 范围与不可变输入

- 初始化权重：`saved/DEP_corrected_init/epoch10_converted.pth`。
- hard-risk：`diagnostics/fixed_hard_risk.json`，8 个训练窗口。
- no-target：`diagnostics/fixed_no_target.json`，10 个跨 Phase 7/8、train/valid/test 窗口。
- temporal separation：`diagnostics/fixed_temporal_separation.json`，4 个窗口。
- occluded-but-tracked：`diagnostics/fixed_occluded_tracked.json`，10 个窗口。
- 训练比较期间列表不重挖；误用 float32 绝对 ROS 时间戳挖出的旧结果保存在
  `reports/phase8b_gradient_diagnostics/mining_attempt_pre_timestamp_fix/`，没有参与正式比较。
- 未覆盖 `reports/phase8_preflight_result.json`。

## 指标语义核对

`DynamicCollisionLoss` 先在 30 个轨迹采样时刻计算安全代价，按时间做折扣平均，
再对有效障碍取 `max`；由此得到每个 batch、每个 primitive 的原始 candidate cost。
报告中的 `raw_dynamic_mean` 是该张量在 batch 和 15 个 primitive 上的平均值；它未乘
`dynamic_loss.weight`。`weighted_dynamic_mean = 0.1 * raw_dynamic_mean`。

最终训练目标为：

```text
L_dynamic_train = 0.1 * (1.0 * mean(candidate_cost) / 1.0)
```

因此，报告原始均值和反向项来自完全相同、未 detach 的 candidate-cost 张量，但最终
参与反向的标量还乘了权重 0.1。`score label` 单独 detach，只隔离标签生成，不切断
trajectory dynamic loss。recorded-future GT、valid/observable mask 在前后评估中固定。
无目标时 raw、weighted、CVaR 均精确为 0。

另修复了一个会污染所有时间风险诊断的确定性错误：约 `1.784e9` 的绝对 ROS 时间戳
若先转为 float32，30 个未来时刻会坍缩为同一值。现在 dataset/collate 保留 float64
绝对时间，loss 先以 float64 相减，再把相对时间转为轨迹 dtype。

风险阈值不再使用 `dynamic_cost > 0`：

- `clearance_threshold = 0 m`，碰撞违反定义为 `d_safe - distance > 0`。
- `cost_epsilon = log(2)^2 = 0.4804530139182014`，由 clearance 为 0 时的 loss 公式得到。
- `cvar_fraction = 4/15`，固定取 15 条候选中风险最高的 4 条。
- `hard_window_threshold = 0.15 m`，等于 UAV 半径 0.3 m 的一半。

## 梯度诊断

在多个 hard-risk/no-target 场景上对每个 loss 分量独立使用
`torch.autograd.grad(..., retain_graph=True)`。全参数组平均梯度范数：

| 分量 | 平均梯度范数 |
|---|---:|
| dynamic raw | 5.267937 |
| dynamic weighted | 0.526747 |
| smoothness | 2.114236 |
| static | 12.841027 |
| guidance | 6.132331 |
| score | 8.399653 |
| total | 18.581592 |

动态加权梯度并非数值为零，但仅为 static 的约 4.1%，且方向经常冲突：

| 梯度对 | mean cosine | 负 cosine 比例 |
|---|---:|---:|
| dynamic vs static | -0.354393 | 80% |
| dynamic vs guidance | -0.032095 | 30% |
| dynamic vs score | -0.013948 | 50% |
| dynamic vs smoothness | -0.264519 | 70% |

分支诊断也符合网络边界：score branch 只收到 score 梯度；trajectory branch 的 score
梯度为 0；shared trunk 同时收到二者。主要冲突来自 static，其次是 smoothness；score
在共享 trunk 上有干扰，但不是唯一或最大的冲突源。

CSV、参数组明细及 heatmap 位于：

- `reports/phase8b_gradient_diagnostics/gradients/gradient_norms.csv`
- `reports/phase8b_gradient_diagnostics/gradients/gradient_cosines.csv`
- `reports/phase8b_gradient_diagnostics/gradients/gradient_cosine_heatmap.png`

## 候选风险与消融结论

基线 hard-risk 集的 raw mean 为 `0.147667`，CVaR 为 `0.474619`，安全候选比例
为 `0.625`，score-selected 碰撞比例为 `0.625`，最小 clearance 为 `-0.849331 m`。
逐候选 CSV 位于 `reports/phase8b_gradient_diagnostics/candidate_risk/candidates.csv`。

同一初始化、同一 fixed set、同一 batch 顺序、40 bounded steps 的 A-G 消融：

| 配置 | after raw mean | after CVaR | safe fraction | top1 risk | static |
|---|---:|---:|---:|---:|---:|
| A dynamic only | 0.005519 | 0.015661 | 0.9750 | 0.003821 | 2.736547 |
| B dynamic + score | 0.023930 | 0.073448 | 0.8000 | 0.008700 | 2.580468 |
| C dynamic + static | 0.113990 | 0.380520 | 0.7167 | 0.011927 | 1.083889 |
| D dynamic + guidance | 0.030387 | 0.093945 | 0.7833 | 0.025360 | 2.499918 |
| E dynamic + static + guidance | 0.171554 | 0.484509 | 0.5917 | 0.301326 | 1.597743 |
| F full weighted sum | 0.228327 | 0.625409 | 0.5500 | 0.273366 | 1.617646 |
| G full without attention | 0.232584 | 0.610490 | 0.5167 | 0.528384 | 1.614617 |

A 显著下降，证明 dynamic loss 数学、mask 和梯度链路本身可优化；E/F 上升则确认根因
是多目标梯度方向冲突与尺度压制，而不是网络完全没有动态容量。关闭 attention 更差，
所以保留 attention。

旧 F 的风险不是所有候选普遍上升。primitive 5、6、7、8、9、10、11、12、13、14
分别变化 `+0.1110,+0.5020,+0.3469,+0.0293,+0.0046,+0.6064,+0.5412,+0.4110,+0.0199,+0.0053`，
其中 6、7、10、11、12 的大幅上升主导均值；primitive 0、1、2、3 反而明显下降。

当前 `max` obstacle aggregation 确实只向获胜障碍/时刻传梯度，属于结构性稀疏；但
dynamic-only 能稳定大幅下降、duplicate-obstacle 不变性测试通过，因此没有证据证明
它是本次 Gate 的阻塞根因。未引入未经需要的 smooth-max。

## 方案选择过程

- mean-only 权重 `0.3/0.6/1.0` 的普通 weighted sum 均未通过 full-objective Gate。
- mean+CVaR（固定 baseline reference scale、CVaR 权重 `0.1/0.3/0.6/1.0`）均使原始
  mean 变差；CVaR 会把优化过度集中到少数候选。因此仍报告并 Gate CVaR，但最终训练
  `cvar_coefficient = 0`。
- 第一版无范数上限的 PCGrad 仍被与动态梯度近正交、但幅值更大的非动态梯度压制。
- `pcgrad_other_norm_ratio = 0.25/0.5/1.0` 均通过；0.25 的动态指标最好且 static 回退
  未超过 20%，因此选 0.25。
- `reference_scale = 1.0`，没有为了让数字好看而启用多余 normalization。
- `freeze_policy=late` 已可同时关闭 Gate，因此按要求没有直接扩大到全量解冻。

最终 PCGrad 将 dynamic 定义为独立任务，将 smooth/static/guidance/score 合成
non-dynamic 任务；只在训练时投影冲突的 non-dynamic 分量，并把其全局范数限制为动态
梯度的 0.25。验证路径不变，策略与配置写入 checkpoint metadata。默认工程配置仍可用
`weighted_sum`，只有 Phase 8B production 配置显式启用该策略。

## 五随机种子最终 preflight

每个种子从相同 corrected checkpoint 开始，使用相同 fixed data 和 40 steps；训练 seed
不同，fixed sample 顺序按 seed 确定且 resume 精确。

| seed | raw mean | CVaR | safe fraction | top1 risk | min clearance (m) | static | 结果 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 8101 | 0.009081 | 0.025833 | 0.8833 | 0.001426 | -0.177409 | 2.662089 | PASS |
| 8102 | 0.008040 | 0.022790 | 0.9250 | 0.003740 | -0.155413 | 2.667445 | PASS |
| 8103 | 0.007808 | 0.022614 | 0.9333 | 0.003088 | -0.185019 | 2.671051 | PASS |
| 8104 | 0.008933 | 0.025724 | 0.9083 | 0.003715 | -0.191476 | 2.676162 | PASS |
| 8105 | 0.008786 | 0.025080 | 0.9000 | 0.003947 | -0.178626 | 2.674312 | PASS |

汇总结果：

- raw mean：`0.008530 ± 0.000509`，median `0.008786`，相对 baseline `0.147667` 明显下降。
- CVaR：`0.024408 ± 0.001418`，相对 baseline `0.474619` 明显下降。
- safe fraction：`0.9100 ± 0.0178`，相对 baseline `0.625` 提高。
- top1 risk：均值 `0.003183`；5 个 seed 的 top1 collision fraction 均从 `0.625` 降为 0。
- minimum clearance：均值 `-0.177589 m`，虽最坏候选仍可能侵入，但较 baseline
  `-0.849331 m` 显著改善；score-selected clearance 已通过碰撞 Gate。
- Spearman：均值 `0.168393`，高于 baseline `0.027679`。
- oracle regret：均值 `0.002919`，低于 baseline `0.055601`。
- static：均值 `2.670212`，相对 baseline `2.433117` 增加约 9.75%，低于 20% 上限。
- no-target dynamic raw/weighted/CVaR 始终精确为 0，静态回退在允许范围内。
- 5/5 seed 全 Gate 通过，无 NaN/Inf，checkpoint resume 精确。

资源稳定性：warm-up 后 FD 恒为 44，CUDA allocated 恒为 17,041,408 bytes；GPU peak
419,356,160 bytes。各 seed 后 RSS 约 2.6071–2.6076 GB，无随 seed 单调增长。进程初始
FD 增量 25 来自 CUDA/Open3D 一次性初始化，不是逐轮泄漏。

## 地图生成器加固

Simulator 的 `dataset_generator.cpp` 已改为显式 `map_seed/pose_seed/actor_seed`，移除
`random_device`；支持 `--save-path`、`--overwrite`、`--yes`。默认拒绝既有输出；覆盖前
列出目标并要求确认；先写唯一 staging，完整后原子 rename，incomplete 不进入 manifest。

`tools/validate_static_map_reachability.py` 对静态体素地图按 UAV 半径膨胀，做三维自由空间
连通分量与 BFS 最短路径检查，要求 start/goal 位于同一分量且路径超过阈值，并把结果
写入 metadata。production map plan 声明 train/valid/test 隔离、每个 split 六类场景齐全，
test 使用 held-out actor 范围，自动生成关闭。

`reports/phase8b_map_generator_validation.json` 为 PASS，Simulator 当前 binary 编译通过。
确定性已由显式种子源码检查和 bounded fixture 验证；由于容量尚未批准，没有执行两套完整
production 数据的双生成与 PLY 全量 hash 对比。因此“生成机制确定且 bounded 验证通过”，
但不能声称“正式 216 条数据已经完成全量复现验证”。

## 验证结果

- Phase 8B CPU/host 单元回归：共 136 tests，整体 OK（1 skip，1 expected failure）。
- Phase 8B shell 入口全部通过 `bash -n`。
- host backbone variant：legacy CPU golden exact，legacy/corrected strict load 与 GPU forward PASS。
- Simulator `catkin_make --pkg sensor_simulator` PASS。
- 地图生成器 hardening fixture：3 tests PASS。
- GPU bounded five-seed preflight：PASS，设备为 RTX 5070 Ti Laptop、capability `(12, 0)`。

## 提示词要求的 22 项回答

1. **是否同一 dynamic term：** 同一未 detach candidate-cost 张量；报告 raw 未加权，反向项乘 0.1，二者不应混称为同一数值。
2. **raw/weighted gradient norm：** 5.267937 / 0.526747。
3. **cosine：** dynamic-static -0.354393，dynamic-guidance -0.032095，dynamic-score -0.013948；另 dynamic-smoothness -0.264519。
4. **哪些 primitive 上升：** 主要是 6、7、10、11、12；不是全部候选普遍上升。
5. **dynamic-only：** 能，raw mean 0.147667 → 0.005519。
6. **主要冲突：** static 最大，smoothness 次之；score 在共享 trunk 有次要干扰。
7. **max 稀疏：** 存在结构性稀疏，但消融证明不是当前阻塞根因，未替换。
8. **实施内容：** 最终实施配置化 dynamic-priority PCGrad 和非动态梯度范数上限；CVaR/normalization 完成消融但最终关闭；未扩大解冻。
9. **选择原因：** ratio 0.25 在保留 mean Gate 的前提下动态指标最好，static 回退仍小于 20%，且 5 seed 稳定。
10. **5 seed：** 8101–8105 的 raw mean 分别为 0.009081、0.008040、0.007808、0.008933、0.008786，5/5 PASS。
11. **mean 是否真下降：** 是，aggregate 0.147667 → 0.008530。
12. **CVaR 是否下降：** 是，0.474619 → 0.024408。
13. **safe fraction：** 0.625 → 0.910，提高。
14. **hard top1：** risk 0.061639 → 均值 0.003183，碰撞比例 0.625 → 0。
15. **no-target：** raw、weighted、CVaR 精确为 0。
16. **static 回退：** +9.75%，在 20% Gate 内，不属于明显回退。
17. **地图确定性：** 显式三 seed 和 bounded fixture PASS；正式全量双生成未执行，见上文限定。
18. **非破坏性：** 是，默认拒绝、显式覆盖确认、staging、原子 rename。
19. **全局可达：** 是，膨胀体素连通分量和 BFS 最短路径验证并写 metadata。
20. **production data：** 未自动生成。
21. **production training：** 未自动启动。
22. **production_ready：** `true`，仅表示所有 Phase 8B 软件硬 Gate 通过；容量审批仍独立阻止正式生产。

## 最终状态与后续边界

`tools/run_managed_dynamic_training.py` 的 production 软件 Gate 只认
`reports/phase8b_preflight_result.json`（不再接受旧 Phase 8 结果），并同时要求该文件满足
`status=PASS`、`production_ready=true`。它还会单独检查
`reports/phase8_dataset_capacity_plan.json` 必须为 `status=APPROVED`，最后仍要求显式
`--yes`。当前容量状态为 `AWAITING_USER_APPROVAL`，因此 production launcher 仍被机器锁定；
生产配置的数据 split 也仍标为 `pending_capacity_approval`，不存在正式 manifest 时同样会拒绝
启动。下一步必须由用户单独批准容量计划后，才可执行正式地图/序列生成；本阶段没有越过该边界。
已实际调用 launcher 的 `--yes` 路径验证，它在创建 run 目录和检查 CUDA 之前按预期报
`separate dataset capacity approval is required`。
