#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zjh/YOPO/DE-P
CONFIG="$ROOT/configs/phase8_authoritative_v2_generation.yaml"
OUTPUT="$ROOT/data/phase8_authoritative_v2"
LOGDIR="$ROOT/logs/phase8jqv2_4_state_v2"
DEVICE=0
WORKERS=8
RESUME=0

while (($#)); do
  case "$1" in
    --resume) RESUME=1 ;;
    --device) shift; DEVICE="${1:?missing device}" ;;
    --workers) shift; WORKERS="${1:?missing workers}" ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

cd "$ROOT"
bash scripts/phase8jqv2_4_state_v2_preflight.sh
if [[ -e "$OUTPUT" && "$RESUME" -ne 1 ]]; then
  echo "$OUTPUT exists; use --resume after inspecting status." >&2
  exit 2
fi
mkdir -p "$OUTPUT/generation_state" "$LOGDIR"
LOCK="$OUTPUT/generation_state/phase8jqv2_4_state_v2_generation.lock"
if ! ( set -o noclobber; printf '{"pid":%s,"host":"%s","started":"%s"}\n' \
  "$$" "$(hostname)" "$(date --iso-8601=seconds)" > "$LOCK" ) 2>/dev/null; then
  echo "Generation lock exists: $LOCK" >&2
  cat "$LOCK" >&2
  exit 3
fi
trap 'rm -f "$LOCK"' EXIT INT TERM
export CUDA_VISIBLE_DEVICES="$DEVICE"

run_split() {
  local split="$1"
  conda run --no-capture-output -n yopo python \
    -m authoritative_dataset.generate_v1 \
    --config "$CONFIG" --output "$OUTPUT" --split "$split" \
    --resume --workers "$WORKERS" --device 0 --fail-fast
}

run_split train 2>&1 | tee -a "$LOGDIR/train.log"
run_split valid 2>&1 | tee -a "$LOGDIR/valid.log"
echo "Formal V2 generation finished. Run the validate script next."
