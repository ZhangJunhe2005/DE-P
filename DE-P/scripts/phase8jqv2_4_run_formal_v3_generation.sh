#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zjh/YOPO/DE-P"
FORMAL_ROOT="$ROOT/data/phase8_dynamic_evidence_formal_v3"
AUTHORIZE=false
RESUME=false
WORKERS=8
DEVICE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --authorize-formal-generation) AUTHORIZE=true; shift ;;
    --resume) RESUME=true; shift ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$AUTHORIZE" == true ]] || { echo "Refusing: --authorize-formal-generation is required" >&2; exit 2; }
cd "$ROOT"
VERIFY_ARGS=(--verify-only)
if [[ "$RESUME" == true ]]; then
  VERIFY_ARGS+=(--allow-partial-formal)
fi
conda run --no-capture-output -n yopo python \
  tools/run_phase8jqv2_4v3fgp1_preflight.py "${VERIFY_ARGS[@]}"
python - "$FORMAL_ROOT" "$RESUME" "$WORKERS" <<'PY'
import json, os, pathlib, shutil, socket, sys
root=pathlib.Path(sys.argv[1])
resume=sys.argv[2].lower()=="true"
workers=int(sys.argv[3])
if root.exists() and any(root.iterdir()) and not resume:
    raise SystemExit("Formal root is non-empty; use --resume only after status audit")
def audit_lock(path):
    if not path.is_file():
        return
    if not resume:
        raise SystemExit(f"Generation lock exists and requires audit: {path}")
    try:
        payload=json.loads(path.read_text())
        pid=int(payload["pid"])
        host=str(payload["host"])
    except Exception as error:
        raise SystemExit(f"Unreadable generation lock {path}: {error}")
    if host != socket.gethostname():
        raise SystemExit(
            f"Generation lock belongs to another host and cannot be recovered: {path}")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        path.unlink()
        print({"stale_lock_recovered":str(path),"dead_pid":pid})
    except PermissionError:
        raise SystemExit(f"Cannot audit generation PID {pid}: {path}")
    else:
        raise SystemExit(f"Generation process is still active (PID {pid}): {path}")
audit_lock(root/"generation_state/formal_v3.lock")
locks=root/"raw_authoritative/generation_state/locks"
if locks.is_dir():
    for path in sorted(locks.glob("phase8jqv2_4_generation_*.lock")):
        audit_lock(path)
free=shutil.disk_usage(root.parent).free
if free < 50_000_000_000:
    raise SystemExit(f"Insufficient free disk: {free} < 50000000000")
print({"formal_root":str(root),"free_bytes":free,"workers":workers,"training":False,"resume":resume})
PY
echo "Plan: 420000 chain samples; train=360000; calibration/validation/internal-test share 60000 by map groups."
echo "This command generates data only. It contains no training command."
read -r -p "Type GENERATE_FORMAL_V3 to continue: " CONFIRM
[[ "$CONFIRM" == "GENERATE_FORMAL_V3" ]] || { echo "Cancelled"; exit 1; }
CMD=(conda run --no-capture-output -n yopo python tools/generate_phase8_dynamic_evidence_formal_v3.py --authorize-formal-generation --workers "$WORKERS" --device "$DEVICE")
if [[ "$RESUME" == true ]]; then CMD+=(--resume); fi
"${CMD[@]}"
