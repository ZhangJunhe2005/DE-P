#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zjh/YOPO/DE-P
CONFIG="$ROOT/configs/phase8_authoritative_v1_generation.yaml"
PLAN="$ROOT/reports/phase8jqv2_4_generation_plan.json"
SPLIT="$ROOT/reports/phase8jqv2_4_formal_split_manifest.json"
OUTPUT="$ROOT/data/phase8_authoritative_v1"
SIM="$ROOT/../Simulator/devel/lib/sensor_simulator/static_authority_contract_test"

cd "$ROOT"
echo "cwd: $PWD"
conda run --no-capture-output -n yopo python - <<'PY'
import json, os, platform, resource
from pathlib import Path
import torch, yaml
from authoritative_dataset.generate_v1 import sha256

root = Path("/home/zjh/YOPO/DE-P")
config_path = root/"configs/phase8_authoritative_v1_generation.yaml"
plan = json.loads((root/"reports/phase8jqv2_4_generation_plan.json").read_text())
config = yaml.safe_load(config_path.read_text())
assert torch.cuda.is_available(), "CUDA is unavailable"
assert sha256(config_path) == plan["config_hash"], "formal config changed"
assert sha256(root/"reports/phase8jqv2_4_formal_split_manifest.json") == plan["split_manifest_hash"]
assert config["test_disabled"] and config["blind_disabled"]
mismatches = {
    name: {"expected": expected, "actual": sha256(name)}
    for name, expected in plan["source_hashes"].items()
    if sha256(name) != expected}
if mismatches:
    hotfix_path = root/"reports/phase8jqv2_4_visibility_hotfix.json"
    assert hotfix_path.is_file(), f"unfrozen generation sources: {mismatches}"
    hotfix = json.loads(hotfix_path.read_text())
    assert hotfix["status"] == "PASS"
    assert hotfix["base_source_hash"] == config["frozen_hashes"]["source_hash"]
    authorized = hotfix["authorized_source_hashes"]
    for name, values in mismatches.items():
        assert authorized.get(name) == values["actual"], (
            f"source not authorized by compatible hotfix: {name}")
output = Path(config["output_root"]).resolve()
assert output.name == "phase8_authoritative_v1"
assert output not in {
    (root/"data/phase8_authoritative_pilot_v1").resolve(),
    (root/"data/phase8_dynamic_production").resolve()}
stat = os.statvfs(output.parent)
free = stat.f_bavail*stat.f_frsize
inodes = stat.f_favail
required = int(plan["projected_storage_gib"]*1.25*(1024**3))
assert free >= required, f"disk free {free/2**30:.1f} GiB < required {required/2**30:.1f} GiB"
assert inodes >= 100000, f"insufficient free inodes: {inodes}"
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
assert soft >= 1024, f"file descriptor soft limit too small: {soft}"
probe = output.parent/".phase8jqv2_4_write_probe"
probe.write_text("ok"); probe.unlink()
locks = list((output/"generation_state/locks").glob("*.lock")) if output.exists() else []
assert not locks, f"generation lock exists: {locks}"
if output.exists():
    plan_path = output/"generation_state/generation_plan.json"
    journal = output/"generation_state/generation_journal.jsonl"
    if journal.is_file() and journal.stat().st_size:
        assert plan_path.is_file(), (
            "existing formal root predates CUDA generation-plan binding; "
            "quarantine it before restart")
    if plan_path.is_file():
        bound = json.loads(plan_path.read_text())
        assert bound["config_hash"] == sha256(config_path)
        assert bound["source_hash"] == config["frozen_hashes"]["source_hash"]
        assert bound["renderer_version"] == (
            "canonical_occupancy_cuda_raycast_v1")
print(json.dumps({
  "status":"PASS", "python":platform.python_version(),
  "torch":torch.__version__, "cuda_build":torch.version.cuda,
  "gpu":torch.cuda.get_device_name(0),
  "compute_capability":torch.cuda.get_device_capability(0),
  "config_hash":plan["config_hash"],
  "split_manifest_hash":plan["split_manifest_hash"],
  "free_disk_gib":round(free/2**30,2),
  "required_with_margin_gib":round(required/2**30,2),
  "free_inodes":inodes, "nofile_soft":soft,
  "git_commit":config["parent_git_commit"],
}, indent=2))
PY

test -x "$SIM"
conda run --no-capture-output -n yopo python \
  tools/validate_phase8jqv2_3_empty.py --no-write
conda run --no-capture-output -n yopo python \
  tools/validate_phase8jqv2_4_cuda_renderer.py --no-write
conda run --no-capture-output -n yopo python -m authoritative_dataset.generate_v1 \
  --config "$CONFIG" --output "$OUTPUT" --split train --resume --dry-run \
  --workers 8 --device 0
conda run --no-capture-output -n yopo python -m authoritative_dataset.generate_v1 \
  --config "$CONFIG" --output "$OUTPUT" --split valid --resume --dry-run \
  --workers 8 --device 0
echo "PHASE8JQV2_4_PREFLIGHT_RESULT: PASS"
