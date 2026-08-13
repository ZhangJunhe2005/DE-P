# Route A V4.9 Dynamic Motion-Preserving Safety

## 结论

V4.9 已在冻结的 V4.8.5 恢复基线之上实现。没有训练网络，没有修改
V4.8.3 权重，也没有改变 V4.8.5 的 60/90/120 度扫描参数。当前状态是
“实现与无 ROS 回归通过，等待四场景闭环验证”，不能报告为生产合格。

冻结权重仍为：

`22e5c63c273d751c15479d70c99d9b85ad615b7b4c62063946a5b1683776ac60`

## 动态运动保持策略

动态目标预测占用的硬碰撞否决对正常网络候选保持不变。对已经通过所有硬约束的候选，
V4.9 只增加一个上限为 0.20 的连续排序修正：

```text
risk = clip(
    0.50 * exp(-max(signed_clearance, 0) / 0.75)
  + 0.35 * exp(-TTC / 1.50)
  + 0.15 * clip(closing_speed / 2.0, 0, 1) * distance_risk,
  0, 1)

effective_score = network_score + 0.20 * risk
```

该项只能在硬安全候选之间改变次序，不能重新放行预计相交的轨迹，也不会
覆盖差异很大的冻结网络分数。

当且仅当所有基础候选都不可执行、且至少一条轨迹仅因动态预测相交而被拒
绝时，依次尝试 1.2 倍和 1.4 倍时长。每次都重新检查静态深度、地图边界、
6 m/s、6 m/s² 以及动态目标；1.2 倍已有可行轨迹时不会继续尝试 1.4 倍。

## 滚动认证与紧急动作

常速度动态外推仍只信任未来 1.5 秒，避免把交互 actor 不可靠地外推到
4 秒以上。较长多项式只能作为高频滚动重规划的近端命令执行；若 0.20 秒
内没有新的有效动态上下文与规划，控制入口会停止旧轨迹并保持当前位置，
不会继续执行未经观测的长尾。

动态目标阻塞全部候选时，系统比较五条 0.8–2.5 秒的硬件约束制动轨迹：

- 优先执行完整通过静态/动态检查的制动；
- 若只剩动态相交风险，选择碰撞更晚、穿透更浅、风险更小的方案，并明确
  标记为 `minimum_risk_uncertified_bounded_braking`；
- 静态碰撞、边界或硬件不合格的制动不会作为后备动作执行。

第二项是不可避免动态相交时唯一、显式的紧急例外：它不是“硬否决通过”，
也不是正常候选。控制器仍以 0.20 秒滚动时效要求重新感知和重选；遥测同时
记录 `normal_candidate_dynamic_hard_veto_preserved=true` 与
`dynamic_hard_veto_emergency_brake_exception=true`，避免把紧急最小风险动作
误报为已认证动作。

## 与恢复状态机的边界

动态 actor 的短暂横穿等待不再累计静态 `zero feasible` 或里程计停滞证据，
也会打断等待前后的候选连续确认，避免将两个不连续的片段拼成一次恢复交接。
若 actor 恰好在 1.2 秒 provisional handoff 窗口中出现，该窗口和已有静态
证据会冻结；恢复后只续跑剩余时间，暂停期间的制动漂移和视野变化不会被
误算成脱困证据。扫描失败次数和当前 60/90/120 度级别会保留。该行为通过
公开 pause API 实现，ROS 节点不再直接修改恢复对象的私有成员。

## Actor 与碰撞诊断

固定验证入口使用 16 个沿完整路线分布的 actor：crossing、head-on、
same-direction-slow、staggered-crossing 各四个。canonical relocation 后会
重新验证实际 waypoint、速度、首次出现方向和时空相遇，不再依赖场景元数据
自证。

碰撞报告区分 `visible / tracked / unobservable / unknown`。不可见 actor 与
track 的绑定只在监控器中做双向唯一的 post-hoc 时空关联；GT 不会进入规划器，
也不会影响控制。歧义关联保守记为 `unknown`。

## 自动验证

- V4.9、动态感知、恢复和旧 runtime profile 相关测试：215 passed，
  1 expected xfail，10 subtests passed。
- cave、forest、pillar、wall 四场景 preflight：全部退出码 0，冻结权重严格
  加载通过。
- 16 tracks、15 candidates、81 samples、3600 depth points 的 CPU 诊断：
  单次评估 p95 约 7.23 ms；保守执行 base + 1.2 + 1.4 三次的上界 p95
  约 23.70 ms。该结果不是资格 Gate，且不包含网络、深度准备和可视化耗时。
- 全仓测试仍受到已删除的历史 Phase 8 fixture 与冻结 hash 合同影响；一次
  尝试得到 2688 passed、31 failed、109 errors。V4.9 相关测试单独全部通过，
  这些历史失败未被伪装成 V4.9 通过项。

## 宿主机闭环验证

每次只运行一个场景，正常关闭后再运行下一个：

```bash
cd /home/zjh/YOPO/DE-P
conda activate yopo

bash scripts/route_a_v4_9_dynamic_rviz_host.sh cave
bash scripts/route_a_v4_9_dynamic_rviz_host.sh forest
bash scripts/route_a_v4_9_dynamic_rviz_host.sh pillar
bash scripts/route_a_v4_9_dynamic_rviz_host.sh wall
```

交互式目标可追加：

```bash
bash scripts/route_a_v4_9_dynamic_rviz_host.sh forest \
  --goal-mode interactive
```

四个固定场景完成后汇总：

```bash
bash scripts/route_a_v4_9_dynamic_summary.sh
```

独立复测安全评估延迟：

```bash
bash scripts/route_a_v4_9_dynamic_runtime_benchmark.sh
```

机器可读合同位于 `configs/route_a_v4_9_dynamic_motion_safety.yaml`。

核心实现文件 SHA256 清单再哈希为：

`41e66b8432d2fd71b046c257c9ad41bedb03175e4a09a613577e971edc94f388`
