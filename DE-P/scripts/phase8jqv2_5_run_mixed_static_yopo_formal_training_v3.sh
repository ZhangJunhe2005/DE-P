#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zjh/YOPO/DE-P"
CONFIG="$ROOT/configs/phase8jqv2_5_mixed_static_yopo_training_v3.yaml"
VERIFY_ONLY=false
DRY_RUN=false
AUTHORIZED=false
RESUME=""

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --verify-only) VERIFY_ONLY=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    --authorize-training) AUTHORIZED=true; shift ;;
    --resume)
      [[ "$#" -ge 2 ]] || { echo "--resume requires a checkpoint" >&2; exit 2; }
      RESUME="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

cd "$ROOT"
conda run --no-capture-output -n yopo \
  python tools/train_mixed_static_yopo_v1.py \
  --config "$CONFIG" --verify-only

if "$VERIFY_ONLY"; then
  [[ "$DRY_RUN" == false && "$AUTHORIZED" == false && -z "$RESUME" ]] || {
    echo "--verify-only cannot be combined with training arguments" >&2
    exit 2
  }
  exit 0
fi

COMPILE_GATE="$ROOT/reports/phase8jqv2_5smgsstr1_compileall.json"
if [[ ! -f "$COMPILE_GATE" ]] || \
  ! grep -q '"status": "PASS"' "$COMPILE_GATE"; then
  echo "Refusing: SMGSS-TR1 compileall approval/check is not complete." >&2
  exit 2
fi

"$AUTHORIZED" || {
  echo "Refusing: --authorize-training is required." >&2
  exit 2
}

if [[ -n "$RESUME" && ! -f "$RESUME" ]]; then
  echo "Resume checkpoint does not exist: $RESUME" >&2
  exit 2
fi

if find "$ROOT/runs/phase8_mixed_static_yopo_v3" -name training.lock \
  -type f -print -quit 2>/dev/null | grep -q .; then
  echo "An active training.lock already exists." >&2
  exit 2
fi

AVAILABLE_KIB="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
if [[ "$AVAILABLE_KIB" -lt 10485760 ]]; then
  echo "At least 10 GiB free disk is required." >&2
  exit 2
fi

read -r -p "Type TRAIN_MIXED_STATIC_YOPO_V3 to continue: " CONFIRMATION
[[ "$CONFIRMATION" == "TRAIN_MIXED_STATIC_YOPO_V3" ]] || {
  echo "Confirmation mismatch; nothing started." >&2
  exit 2
}

ARGS=(python tools/train_mixed_static_yopo_v1.py --config "$CONFIG" --authorized)
"$DRY_RUN" && ARGS+=(--dry-run)
[[ -n "$RESUME" ]] && ARGS+=(--resume "$RESUME")
exec conda run --no-capture-output -n yopo "${ARGS[@]}"
