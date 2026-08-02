#!/usr/bin/env bash
set -euo pipefail
cd /home/zjh/YOPO/DE-P
resume_name="c_risk_gated_025_seed8010"
resume_stage="configs/phase8c_resume_stage1.yaml"
if [[ -e runs/phase8c_ablation || -e "${resume_stage}" ]]; then
  echo "refusing to overwrite an existing Phase-8C ablation run or resume config" >&2
  exit 1
fi
conda run --no-capture-output -n yopo python tools/make_phase8c_resume_stage_config.py \
  --source "configs/phase8c_ablation/${resume_name}.yaml" --output "${resume_stage}"
for config in configs/phase8c_ablation/*.yaml; do
  name="$(basename "${config}" .yaml)"
  run_dir="/home/zjh/YOPO/DE-P/runs/phase8c_ablation/${name}"
  if [[ "${name}" == "${resume_name}" ]]; then
    conda run --no-capture-output -n yopo python tools/run_managed_dynamic_training.py \
      --config "${resume_stage}" --run-dir "${run_dir}" --yes
    conda run --no-capture-output -n yopo python tools/run_managed_dynamic_training.py \
      --config "${config}" --run-dir "${run_dir}" \
      --resume "${run_dir}/checkpoints/latest.pt" --yes
  else
    conda run --no-capture-output -n yopo python tools/run_managed_dynamic_training.py \
      --config "${config}" --run-dir "${run_dir}" --yes
  fi
done
