#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zjh/YOPO/DE-P
CONFIG="$ROOT/configs/phase8_authoritative_v1_generation.yaml"
OUTPUT="$ROOT/data/phase8_authoritative_v1"
LOGDIR="$ROOT/logs/phase8jqv2_4"
WORKERS=8
DEVICE=0
MODE=resume
DO_TRAIN=1
DO_VALID=1
DRY_RUN=0

while (($#)); do
  case "$1" in
    --resume) MODE=resume ;;
    --fresh) MODE=fresh ;;
    --train-only) DO_VALID=0 ;;
    --valid-only) DO_TRAIN=0 ;;
    --workers) shift; WORKERS="${1:?missing worker count}" ;;
    --device) shift; DEVICE="${1:?missing GPU device}" ;;
    --dry-run) DRY_RUN=1 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

cd "$ROOT"
mkdir -p "$LOGDIR"
touch "$LOGDIR/errors.log" "$LOGDIR/progress.jsonl" \
  "$LOGDIR/resource_usage.jsonl"
exec 2> >(tee -a "$LOGDIR/errors.log" >&2)
conda run --no-capture-output -n yopo python - <<'PY' \
  > "$LOGDIR/environment.json"
import json, platform, torch
print(json.dumps({
  "python": platform.python_version(), "torch": torch.__version__,
  "cuda_build": torch.version.cuda,
  "cuda_available": torch.cuda.is_available(),
  "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
  "compute_capability": (
    torch.cuda.get_device_capability(0) if torch.cuda.is_available()
    else None),
}, indent=2))
PY
if [[ "$MODE" == fresh && -e "$OUTPUT" ]]; then
  echo "--fresh is permitted only for a completely new output root; nothing was deleted." >&2
  exit 2
fi
if [[ "$MODE" == fresh ]]; then
  read -r -p "Type CREATE_NEW to create the new formal root: " answer
  [[ "$answer" == CREATE_NEW ]] || exit 2
fi
bash scripts/phase8jqv2_4_preflight.sh
if ((DRY_RUN)); then
  echo "dry-run complete; formal generation not started"
  exit 0
fi

mkdir -p "$OUTPUT/generation_state/locks"
GLOBAL_LOCK="$OUTPUT/generation_state/phase8jqv2_4_generation.lock"
if ! ( set -o noclobber; printf '{"pid":%s,"host":"%s","started":"%s"}\n' \
  "$$" "$(hostname)" "$(date --iso-8601=seconds)" > "$GLOBAL_LOCK" ) 2>/dev/null; then
  echo "Another generation owns $GLOBAL_LOCK:" >&2
  cat "$GLOBAL_LOCK" >&2
  exit 3
fi
trap 'rm -f "$GLOBAL_LOCK"' EXIT INT TERM
export CUDA_VISIBLE_DEVICES="$DEVICE"

run_split() {
  local split="$1"
  conda run --no-capture-output -n yopo python \
    -m authoritative_dataset.generate_v1 \
    --config "$CONFIG" --output "$OUTPUT" --split "$split" --resume \
    --workers "$WORKERS" --device 0 --fail-fast
}
((DO_TRAIN)) && run_split train
((DO_VALID)) && run_split valid
if [[ -f "$OUTPUT/generation_state/completion/TRAIN_COMPLETE" \
   && -f "$OUTPUT/generation_state/completion/VALID_COMPLETE" ]]; then
  printf '{"completed":"%s"}\n' "$(date --iso-8601=seconds)" \
    > "$OUTPUT/generation_state/completion/FULL_GENERATION_COMPLETE"
fi
echo "Generation command finished. Run phase8jqv2_4_status.sh."
