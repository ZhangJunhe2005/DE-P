# Phase 8D 最终生产训练前验收

结论：**FAIL**；`production_ready=false`。
长期训练未启动，正式 launcher 因 Gate 未通过而保持未更新/不可授权。

## 已通过

- capacity_approved
- formal_dataset_validated
- formal_dataset_distribution_passed
- map_catalog_isolation_passed
- mixed_epoch_schedule_passed
- lazy_static_dataset_passed
- fixed_validation_suites_passed
- risk_sets_frozen
- resume_passed
- resource_stability_passed
- real_static_smoke_passed

## 硬失败

- train: detection precision below 0.5
- train: no-target false-positive frames exceed 0.1
- valid: detection precision below 0.5
- valid: no-target false-positive frames exceed 0.1
- test: detection precision below 0.5
- test: velocity RMSE exceeds 1.5 m/s
- test: no-target false-positive frames exceed 0.1
- no strategy meets fixed-validation hard constraints

## 关键事实

- 9 runs × 2 epochs × 1026 steps = 18468 optimizer steps。
- resume：True（epoch 边界 1026 → 2052）。
- 每 epoch：513 static + 513 dynamic；dynamic equivalent passes=1.0；无 loader recycling。
- validation：固定 valid_estimated / valid_gt / valid_static；test 不参与选模。
- estimated-context：训练 curriculum 已禁用；fixed valid_estimated 仍保留。
- PCGrad：三种策略均未达到动态 collision 硬阈值，因此没有获批策略。
- static：9/9 run 的真实 static smoke 通过，no-target dynamic cost 精确为 0。
- 正式数据分布与 9 个独立风险集通过并冻结。

## 后续解锁条件

1. 修复因果动态感知的静态簇误报与 test 速度估计。
2. 重新运行 estimated-context 全量质量分析并通过硬阈值。
3. 重新执行有限 shakedown，使 ValidEstimated collision ≤ 0.05。
4. 重新生成本 Gate；仅在 PASS 后更新并交付长期训练 launcher。

## 产物

- `reports/phase8d_final_preproduction_result.json`
- `reports/phase8d_estimated_context_quality.json`
- `reports/phase8d_shakedown_summary.json`
- `reports/phase8d_formal_dataset_distribution.json`
- `diagnostics/phase8c_production_risk_set_manifest.json`
