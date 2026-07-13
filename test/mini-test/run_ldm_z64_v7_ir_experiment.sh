#!/bin/bash
# 一键运行 Z=64 近场 LDM v7 红外视锥结构监督实验。
#
# v7 在 v6 顶部结构监督基础上，额外加入 IR frustum 内 target 正样本
# occupancy/top-height 辅助监督，并依赖训练日志中的 mock IR/calib 比例判断
# 红外条件是否可信。此脚本只配置实验，不在 Codex 中自动长训。

set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SELF_DIR}/../.." && pwd)"

export EXP_DIR="${EXP_DIR:-test/result/ldm/ablation/ldm_near40_500_z64_v7_ir}"
export BASE_VAE_CKPT="${BASE_VAE_CKPT:-test/result/vae/reconstruction/vae_near40_500_z64_upper_bound/vae/vae_best.pt}"
export MINI_TARGET_SIZE="${MINI_TARGET_SIZE:-64,128,128}"
export MINI_MODEL_PC_RANGE="${MINI_MODEL_PC_RANGE:-0,-20,-6,40,20,10}"
export MINI_SOURCE_PC_RANGE="${MINI_SOURCE_PC_RANGE:-0,-20,-6,120,20,10}"
export SAMPLES_PER_SCENE="${SAMPLES_PER_SCENE:-500}"
export MINI_LDM_EPOCHS="${MINI_LDM_EPOCHS:-10}"
export MINI_NUM_WORKERS="${MINI_NUM_WORKERS:-2}"
export MINI_LDM_DECODED_WEIGHT="${MINI_LDM_DECODED_WEIGHT:-0.12}"
export MINI_LDM_DECODED_FP_WEIGHT="${MINI_LDM_DECODED_FP_WEIGHT:-0.20}"
export MINI_LDM_DECODED_MASS_WEIGHT="${MINI_LDM_DECODED_MASS_WEIGHT:-0.08}"
export MINI_LDM_HEIGHT_WEIGHT="${MINI_LDM_HEIGHT_WEIGHT:-0.04}"
export MINI_LDM_TOP_WEIGHT="${MINI_LDM_TOP_WEIGHT:-0.08}"
export MINI_LDM_CONTINUITY_WEIGHT="${MINI_LDM_CONTINUITY_WEIGHT:-0.02}"
export MINI_LDM_DENSITY_WEIGHT="${MINI_LDM_DENSITY_WEIGHT:-0.015}"
export MINI_LDM_IR_FRUSTUM_OCC_WEIGHT="${MINI_LDM_IR_FRUSTUM_OCC_WEIGHT:-0.04}"
export MINI_LDM_IR_FRUSTUM_TOP_WEIGHT="${MINI_LDM_IR_FRUSTUM_TOP_WEIGHT:-0.05}"
export MINI_LDM_UNCERTAINTY_WEIGHT="${MINI_LDM_UNCERTAINTY_WEIGHT:-0.0}"
export OCC_THRESHOLD="${OCC_THRESHOLD:-0.85}"
export TARGET_THRESHOLD="${TARGET_THRESHOLD:-0.5}"
export ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"

if [[ -n "${CUDA_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${CUDA_DEVICES}}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_DEVICES="${CUDA_VISIBLE_DEVICES}"
else
  export CUDA_DEVICES="0"
  export CUDA_VISIBLE_DEVICES="0"
fi

if [[ "${EXP_DIR}" = /* ]]; then
  EXP_DIR_ABS="${EXP_DIR}"
else
  EXP_DIR_ABS="${ROOT_DIR}/${EXP_DIR}"
fi
if [[ "${ALLOW_OVERWRITE}" != "1" && -e "${EXP_DIR_ABS}/ldm/ldm_best.pt" ]]; then
  echo "Error: experiment output already exists: ${EXP_DIR_ABS}"
  echo "Set EXP_DIR to a new directory, or ALLOW_OVERWRITE=1 to overwrite intentionally."
  exit 1
fi

bash "${SELF_DIR}/run_ldm_vertical_experiment.sh"
