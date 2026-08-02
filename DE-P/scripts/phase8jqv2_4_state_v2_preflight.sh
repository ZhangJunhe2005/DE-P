#!/usr/bin/env bash
set -euo pipefail
cd /home/zjh/YOPO/DE-P

conda run --no-capture-output -n yopo python - <<'PY'
import hashlib, json
from pathlib import Path
import torch

root = Path("/home/zjh/YOPO/DE-P")
old = root/"data/phase8_authoritative_v1/manifests/dataset_manifest.json"
smoke_manifest = (
    root/"data/phase8_authoritative_v2_smoke/manifests/dataset_manifest.json"
)
expected = "a81319d734871c48d49058e22a36a733baead3d85db2ec39206221d3b154f723"
actual = hashlib.sha256(old.read_bytes()).hexdigest()
if actual != expected:
    raise SystemExit("immutable V1 root manifest changed")
if not smoke_manifest.is_file():
    raise SystemExit("V2 smoke root manifest is missing")
smoke_actual = hashlib.sha256(smoke_manifest.read_bytes()).hexdigest()
smoke = json.loads((
    root/"reports/phase8jqv2_4_smoke_semantic_validation.json"
).read_text())
integrity = json.loads((
    root/"reports/phase8jqv2_4_smoke_generation_summary.json"
).read_text())
if smoke["status"] != "PASS" or integrity["status"] != "PASS":
    raise SystemExit("V2 smoke gates are not PASS")
reported_hashes = {
    smoke.get("root_manifest_hash"),
    integrity.get("root_manifest_hash"),
}
if reported_hashes != {smoke_actual}:
    raise SystemExit(
        "V2 smoke reports are stale or disagree with the current smoke dataset"
    )
if not torch.cuda.is_available():
    raise SystemExit("host CUDA is unavailable")
print(json.dumps({
    "status": "PASS",
    "old_root_manifest_hash": actual,
    "smoke_root_manifest_hash": smoke_actual,
    "smoke_reports_current": True,
    "cuda": torch.cuda.get_device_name(0),
    "compute_capability": torch.cuda.get_device_capability(0),
}, indent=2))
PY
