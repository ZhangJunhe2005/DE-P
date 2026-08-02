#!/usr/bin/env bash
set -euo pipefail
cd /home/zjh/YOPO/DE-P

run_root="runs/phase8d_shakedown"
if [[ -e "${run_root}" ]]; then
  echo "refusing to overwrite existing ${run_root}" >&2
  exit 1
fi
mkdir -p "${run_root}"

resume_name="fixed_025_seed8401"
stage_config="${run_root}/resume_stage1.yaml"
conda run --no-capture-output -n yopo python tools/make_phase8d_resume_stage_config.py \
  --source "configs/phase8d_shakedown/${resume_name}.yaml" --output "${stage_config}"
conda run --no-capture-output -n yopo python tools/run_managed_dynamic_training.py \
  --config "${stage_config}" --run-dir "${run_root}/${resume_name}"
conda run --no-capture-output -n yopo python tools/audit_phase8d_resume.py \
  --checkpoint "${run_root}/${resume_name}/checkpoints/latest.pt" \
  --output "${run_root}/${resume_name}/resume_audit_before.json"
conda run --no-capture-output -n yopo python tools/run_managed_dynamic_training.py \
  --config "configs/phase8d_shakedown/${resume_name}.yaml" \
  --run-dir "${run_root}/${resume_name}" \
  --resume "${run_root}/${resume_name}/checkpoints/latest.pt"

for config in configs/phase8d_shakedown/*.yaml; do
  name="$(basename "${config}" .yaml)"
  if [[ "${name}" == "${resume_name}" ]]; then
    continue
  fi
  conda run --no-capture-output -n yopo python tools/run_managed_dynamic_training.py \
    --config "${config}" --run-dir "${run_root}/${name}"
done

conda run --no-capture-output -n yopo python tools/summarize_phase8d_shakedown.py \
  --runs "${run_root}" --output reports/phase8d_shakedown_summary.json \
  --estimated-quality reports/phase8d_estimated_context_quality.json
