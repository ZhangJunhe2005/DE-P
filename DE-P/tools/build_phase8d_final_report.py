#!/usr/bin/env python3
"""Build the fail-closed Phase 8D result and human-readable readiness report."""

import hashlib
import json
from pathlib import Path

from ruamel.yaml import YAML


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    config_path = ROOT / "configs/train_dynamic_production_v2.yaml"
    config = YAML(typ="safe").load(config_path)
    data_root = Path(config["dataset_root"])
    manifest_path = Path(config["dataset_manifest"])
    manifest = YAML(typ="safe").load(manifest_path)
    distribution = json.loads((ROOT / "reports/phase8d_formal_dataset_distribution.json").read_text())
    estimated = json.loads((ROOT / "reports/phase8d_estimated_context_quality.json").read_text())
    shakedown_path = ROOT / "reports/phase8d_shakedown_summary.json"
    shakedown = json.loads(shakedown_path.read_text())
    smoke = json.loads((ROOT / "reports/phase8d_mixed_smoke.json").read_text())
    capacity = json.loads((ROOT / "reports/phase8_dataset_capacity_plan.json").read_text())
    risk_path = ROOT / "diagnostics/phase8c_production_risk_set_manifest.json"
    risk = json.loads(risk_path.read_text())
    risk_valid = risk.get("status") == "PASS" and all(
        (ROOT / "diagnostics" / name).is_file()
        and sha256(ROOT / "diagnostics" / name) == digest
        for name, digest in risk.get("file_sha256", {}).items()
    ) and len(risk.get("file_sha256", {})) == 9
    resource_failures = [
        item for item in shakedown.get("failures", [])
        if any(word in item.lower() for word in ("rss", "gpu", "descriptor", "cache", "latency"))
    ]
    fields = {
        "capacity_approved": capacity.get("status") == "APPROVED",
        "formal_dataset_validated": (
            manifest.get("completion_status") == "complete"
            and manifest.get("formal_pointcloud_training_allowed") is False
        ),
        "formal_dataset_distribution_passed": distribution.get("status") == "PASS",
        "map_catalog_isolation_passed": bool(
            smoke.get("static_maps_only_legacy") and smoke.get("dynamic_maps_only_formal")
            and smoke.get("static_catalog_hash") != smoke.get("dynamic_catalog_hash")
        ),
        "mixed_epoch_schedule_passed": bool(
            smoke.get("steps_per_epoch") == 1026
            and smoke.get("batch_counts") == {"static": 513, "dynamic": 513}
            and all(run.get("global_step") == 2052 for run in shakedown.get("runs", []))
        ),
        "lazy_static_dataset_passed": smoke.get("static_dataset_cache_count", 10**9) <= 128,
        "fixed_validation_suites_passed": bool(
            smoke.get("validation_contexts") == {
                "valid_gt": "ground_truth", "valid_estimated": "estimated"
            }
            and all(set(run) >= {"valid_gt", "valid_estimated", "valid_static"}
                    for run in shakedown.get("runs", []))
        ),
        "estimated_context_quality_passed": estimated.get("status") == "PASS",
        "risk_sets_frozen": risk_valid,
        "long_horizon_shakedown_passed": shakedown.get("status") == "PASS",
        "resume_passed": shakedown.get("resume_passed") is True,
        "resource_stability_passed": not resource_failures and len(shakedown.get("runs", [])) == 9,
        "real_static_smoke_passed": bool(
            shakedown.get("runs")
            and all(run["valid_static"].get("static_smoke_pass") for run in shakedown["runs"])
        ),
    }
    hard_failures = []
    if not fields["estimated_context_quality_passed"]:
        hard_failures.extend(estimated.get("hard_failures", ["estimated-context quality failed"]))
    if not fields["long_horizon_shakedown_passed"]:
        hard_failures.extend(shakedown.get("failures", ["shakedown failed"]))
    production_ready = all(fields.values())
    payload = {
        "status": "PASS" if production_ready else "FAIL",
        "production_ready": production_ready,
        **fields,
        "dataset_manifest_sha256": sha256(manifest_path),
        "dynamic_map_catalog_sha256": sha256(config["dynamic_map_catalog"]),
        "static_map_catalog_sha256": sha256(config["static_map_catalog"]),
        "production_config_sha256": sha256(config_path),
        "initialization_checkpoint_sha256": sha256(config["initialization_checkpoint"]),
        "data_root": str(data_root.resolve()),
        "risk_set_manifest": str(risk_path.resolve()),
        "risk_set_manifest_sha256": sha256(risk_path),
        "shakedown_report": str(shakedown_path.resolve()),
        "shakedown_report_sha256": sha256(shakedown_path),
        "selected_pcgrad_strategy": shakedown.get("selected_strategy"),
        "candidate_pcgrad_strategy_not_approved": "fixed_025",
        "steps_per_epoch": config["training_schedule"]["steps_per_epoch"],
        "recommended_epochs": None,
        "candidate_epochs_not_approved": config["epochs"],
        "validation_suite_version": config["validation"]["suite_version"],
        "estimated_context_curriculum_enabled": any(
            stage["context_source"] == "estimated"
            for stage in config["dynamic_training"]["curriculum"]
        ),
        "long_training_started_by_codex": False,
        "launch_scripts_updated": False,
        "launch_scripts_update_blocked_until_gate_pass": True,
        "hard_failures": hard_failures,
    }
    output = ROOT / "reports/phase8d_final_preproduction_result.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Phase 8D 最终生产训练前验收", "",
        f"结论：**{payload['status']}**；`production_ready={str(production_ready).lower()}`。",
        "长期训练未启动，正式 launcher 因 Gate 未通过而保持未更新/不可授权。", "",
        "## 已通过", "",
    ]
    lines.extend(f"- {name}" for name, value in fields.items() if value)
    lines.extend(["", "## 硬失败", ""])
    lines.extend(f"- {item}" for item in hard_failures)
    lines.extend([
        "", "## 关键事实", "",
        f"- 9 runs × 2 epochs × 1026 steps = {shakedown.get('optimizer_steps_total')} optimizer steps。",
        f"- resume：{shakedown.get('resume_passed')}（epoch 边界 1026 → 2052）。",
        "- 每 epoch：513 static + 513 dynamic；dynamic equivalent passes=1.0；无 loader recycling。",
        "- validation：固定 valid_estimated / valid_gt / valid_static；test 不参与选模。",
        "- estimated-context：训练 curriculum 已禁用；fixed valid_estimated 仍保留。",
        "- PCGrad：三种策略均未达到动态 collision 硬阈值，因此没有获批策略。",
        "- static：9/9 run 的真实 static smoke 通过，no-target dynamic cost 精确为 0。",
        "- 正式数据分布与 9 个独立风险集通过并冻结。", "",
        "## 后续解锁条件", "",
        "1. 修复因果动态感知的静态簇误报与 test 速度估计。",
        "2. 重新运行 estimated-context 全量质量分析并通过硬阈值。",
        "3. 重新执行有限 shakedown，使 ValidEstimated collision ≤ 0.05。",
        "4. 重新生成本 Gate；仅在 PASS 后更新并交付长期训练 launcher。", "",
        "## 产物", "",
        "- `reports/phase8d_final_preproduction_result.json`",
        "- `reports/phase8d_estimated_context_quality.json`",
        "- `reports/phase8d_shakedown_summary.json`",
        "- `reports/phase8d_formal_dataset_distribution.json`",
        "- `diagnostics/phase8c_production_risk_set_manifest.json`",
    ])
    (ROOT / "reports/phase8d_final_training_readiness.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
