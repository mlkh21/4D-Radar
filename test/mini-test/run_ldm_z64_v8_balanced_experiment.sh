#!/bin/bash
# 一键运行 Z=64 近场 LDM v8 平衡结构监督实验。
#
# v8 同时约束顶部过冲和 IR 视锥负样本，并降低 IR 正样本及顶部权重，
# 用于平衡竖向恢复与假阳性抑制。此脚本只配置实验，不自动启动额外消融或 CD。

set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SELF_DIR}/../.." && pwd)"

export EXP_DIR="${EXP_DIR:-test/result/ldm/ablation/ldm_near40_500_z64_v8_balanced}"
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
export MINI_LDM_TOP_OVERSHOOT_WEIGHT="${MINI_LDM_TOP_OVERSHOOT_WEIGHT:-0.05}"
export MINI_LDM_CONTINUITY_WEIGHT="${MINI_LDM_CONTINUITY_WEIGHT:-0.02}"
export MINI_LDM_DENSITY_WEIGHT="${MINI_LDM_DENSITY_WEIGHT:-0.015}"
export MINI_LDM_IR_FRUSTUM_OCC_WEIGHT="${MINI_LDM_IR_FRUSTUM_OCC_WEIGHT:-0.02}"
export MINI_LDM_IR_FRUSTUM_NEGATIVE_WEIGHT="${MINI_LDM_IR_FRUSTUM_NEGATIVE_WEIGHT:-0.02}"
export MINI_LDM_IR_FRUSTUM_TOP_WEIGHT="${MINI_LDM_IR_FRUSTUM_TOP_WEIGHT:-0.03}"
export MINI_LDM_UNCERTAINTY_WEIGHT="${MINI_LDM_UNCERTAINTY_WEIGHT:-0.0}"
export OCC_THRESHOLD="${OCC_THRESHOLD:-0.99}"
export TARGET_THRESHOLD="${TARGET_THRESHOLD:-0.5}"
export ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"

if [[ -n "${CUDA_DEVICES:-}" ]]; then
  SELECTED_CUDA_DEVICES="${CUDA_DEVICES}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  SELECTED_CUDA_DEVICES="${CUDA_VISIBLE_DEVICES}"
else
  SELECTED_CUDA_DEVICES="0"
fi
export CUDA_DEVICES="${SELECTED_CUDA_DEVICES}"
export CUDA_VISIBLE_DEVICES="${SELECTED_CUDA_DEVICES}"

EXP_DIR_INPUT="${EXP_DIR}"
if [[ "${EXP_DIR_INPUT}" != /* ]]; then
  EXP_DIR_INPUT="${ROOT_DIR}/${EXP_DIR_INPUT}"
fi
if [[ -L "${EXP_DIR_INPUT}" ]]; then
  echo "Error: EXP_DIR must not be a symlink: ${EXP_DIR_INPUT}"
  exit 1
fi
EXP_DIR_ABS="$(realpath -m -- "${EXP_DIR_INPUT}")"
export EXP_DIR="${EXP_DIR_ABS}"
export MINI_DATASET_DIR="${MINI_DATASET_DIR:-${EXP_DIR_ABS}/.tmp_mini_train_dataset}"
export MINI_CONFIG_PATH="${MINI_CONFIG_PATH:-${EXP_DIR_ABS}/.tmp_ldm_config.yaml}"

if [[ "${ALLOW_OVERWRITE}" != "1" && -e "${EXP_DIR_ABS}" ]]; then
  if [[ ! -d "${EXP_DIR_ABS}" || -n "$(find "${EXP_DIR_ABS}" -mindepth 1 -print -quit)" ]]; then
    echo "Error: experiment output already exists and is not empty: ${EXP_DIR_ABS}"
    echo "Set EXP_DIR to a new directory, or ALLOW_OVERWRITE=1 to overwrite intentionally."
    exit 1
  fi
fi

bash "${SELF_DIR}/run_ldm_vertical_experiment.sh"
