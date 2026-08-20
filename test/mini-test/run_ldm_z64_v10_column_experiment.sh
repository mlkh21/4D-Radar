#!/bin/bash
# 运行 Z=64 LDM v10/v11 列级损失隔离变量训练实验，不执行推理、消融或 CD。

set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SELF_DIR}/../.." && pwd)"
V10_VARIANT="${V10_VARIANT:-A}"

case "${V10_VARIANT}" in
  A)
    DEFAULT_EXP_DIR="${ROOT_DIR}/test/result/ldm/ablation/ldm_near40_500_z64_v10a_column_screen"
    COLUMN_POSITIVE_WEIGHT="0.02"
    COLUMN_NEGATIVE_WEIGHT="0.01"
    COLUMN_CURRICULUM_ENABLED="false"
    COLUMN_POSITIVE_START_WEIGHT="${COLUMN_POSITIVE_WEIGHT}"
    COLUMN_NEGATIVE_START_WEIGHT="${COLUMN_NEGATIVE_WEIGHT}"
    ;;
  B)
    DEFAULT_EXP_DIR="${ROOT_DIR}/test/result/ldm/ablation/ldm_near40_500_z64_v10b_column_screen"
    COLUMN_POSITIVE_WEIGHT="0.02"
    COLUMN_NEGATIVE_WEIGHT="0.02"
    COLUMN_CURRICULUM_ENABLED="false"
    COLUMN_POSITIVE_START_WEIGHT="${COLUMN_POSITIVE_WEIGHT}"
    COLUMN_NEGATIVE_START_WEIGHT="${COLUMN_NEGATIVE_WEIGHT}"
    ;;
  C)
    DEFAULT_EXP_DIR="${ROOT_DIR}/test/result/ldm/ablation/ldm_near40_500_z64_v10c_pos003_screen"
    COLUMN_POSITIVE_WEIGHT="0.03"
    COLUMN_NEGATIVE_WEIGHT="0.01"
    COLUMN_CURRICULUM_ENABLED="false"
    COLUMN_POSITIVE_START_WEIGHT="${COLUMN_POSITIVE_WEIGHT}"
    COLUMN_NEGATIVE_START_WEIGHT="${COLUMN_NEGATIVE_WEIGHT}"
    ;;
  D)
    DEFAULT_EXP_DIR="${ROOT_DIR}/test/result/ldm/ablation/ldm_near40_500_z64_v10d_neg0005_screen"
    COLUMN_POSITIVE_WEIGHT="0.02"
    COLUMN_NEGATIVE_WEIGHT="0.005"
    COLUMN_CURRICULUM_ENABLED="false"
    COLUMN_POSITIVE_START_WEIGHT="${COLUMN_POSITIVE_WEIGHT}"
    COLUMN_NEGATIVE_START_WEIGHT="${COLUMN_NEGATIVE_WEIGHT}"
    ;;
  V11)
    DEFAULT_EXP_DIR="${ROOT_DIR}/test/result/ldm/ablation/ldm_near40_500_z64_column_curriculum_v11_screen"
    COLUMN_CURRICULUM_ENABLED="true"
    COLUMN_POSITIVE_START_WEIGHT="0.03"
    COLUMN_POSITIVE_WEIGHT="0.02"
    COLUMN_NEGATIVE_START_WEIGHT="0.0"
    COLUMN_NEGATIVE_WEIGHT="0.01"
    ;;
  *)
    echo "Error: V10_VARIANT must be A, B, C, D, or V11, got: ${V10_VARIANT}"
    exit 1
    ;;
esac

EXP_DIR_INPUT="${EXP_DIR:-${DEFAULT_EXP_DIR}}"
if [[ "${EXP_DIR_INPUT}" != /* ]]; then
  EXP_DIR_INPUT="${ROOT_DIR}/${EXP_DIR_INPUT}"
fi
if [[ -L "${EXP_DIR_INPUT}" ]]; then
  echo "Error: EXP_DIR must not be a symlink: ${EXP_DIR_INPUT}"
  exit 1
fi
EXP_DIR="$(realpath -m -- "${EXP_DIR_INPUT}")"
case "${EXP_DIR}" in
  "${ROOT_DIR}/test/result/"*|/tmp/*) ;;
  *) echo "Error: unsafe EXP_DIR: ${EXP_DIR}"; exit 1 ;;
esac
if [[ "${EXP_DIR}" == "${ROOT_DIR}/test/result" || "${EXP_DIR}" == "/tmp" ]]; then
  echo "Error: unsafe EXP_DIR: ${EXP_DIR}"
  exit 1
fi
if [[ -e "${EXP_DIR}" ]] && { [[ ! -d "${EXP_DIR}" ]] || [[ -n "$(find "${EXP_DIR}" -mindepth 1 -print -quit)" ]]; }; then
  echo "Error: experiment output already exists and is not empty: ${EXP_DIR}"
  exit 1
fi

BASE_VAE_CKPT="${BASE_VAE_CKPT:-${ROOT_DIR}/test/result/vae/reconstruction/vae_near40_500_z64_upper_bound/vae/vae_best.pt}"
if [[ "${BASE_VAE_CKPT}" != /* ]]; then
  BASE_VAE_CKPT="${ROOT_DIR}/${BASE_VAE_CKPT}"
fi
if [[ ! -s "${BASE_VAE_CKPT}" ]]; then
  echo "Error: VAE checkpoint not found: ${BASE_VAE_CKPT}"
  exit 1
fi

LOCK_PATH="${EXP_DIR}.v10.lock"
mkdir -p -- "$(dirname "${LOCK_PATH}")"
if ! mkdir -- "${LOCK_PATH}" 2>/dev/null; then
  echo "Error: experiment is already running: ${EXP_DIR}"
  exit 1
fi
cleanup_lock() { rmdir -- "${LOCK_PATH}" 2>/dev/null || true; }
trap cleanup_lock EXIT INT TERM

check_exp_empty_after_lock() {
  if [[ -L "${EXP_DIR}" ]]; then
    echo "Error: unsafe EXP_DIR after lock: path is a symlink: ${EXP_DIR}"
    return 1
  fi
  if [[ -e "${EXP_DIR}" ]]; then
    if [[ ! -d "${EXP_DIR}" || -n "$(find "${EXP_DIR}" -mindepth 1 -print -quit)" ]]; then
      echo "Error: experiment output became non-empty after lock: ${EXP_DIR}"
      return 1
    fi
  fi
}

check_exp_empty_after_lock

VAE_DIR="${EXP_DIR}/vae"
VAE_CKPT_PATH="${VAE_DIR}/vae_best.pt"
LDM_DIR="${EXP_DIR}/ldm"
LDM_CKPT_PATH="${LDM_DIR}/ldm_best.pt"
MINI_DATASET_DIR="${EXP_DIR}/.tmp_mini_train_dataset"
MINI_CONFIG_PATH="${EXP_DIR}/.tmp_ldm_config.yaml"

audit_write_path() {
  local label="$1"
  local path="$2"
  local parent
  if [[ -L "${path}" ]]; then
    echo "Error: unsafe ${label}: path must not be a symlink: ${path}"
    return 1
  fi
  parent="$(realpath -m -- "$(dirname -- "${path}")")"
  case "${parent}" in
    "${EXP_DIR}" | "${EXP_DIR}"/*) ;;
    *) echo "Error: unsafe ${label}: parent escapes EXP_DIR: ${parent}"; return 1 ;;
  esac
}

ensure_exp_entity() {
  local canonical_now
  if [[ -L "${EXP_DIR}" ]]; then
    echo "Error: unsafe EXP_DIR: path became a symlink: ${EXP_DIR}"
    return 1
  fi
  if [[ ! -e "${EXP_DIR}" ]]; then
    canonical_now="$(realpath -m -- "$(dirname -- "${EXP_DIR}")")/$(basename -- "${EXP_DIR}")"
    if [[ "${canonical_now}" != "${EXP_DIR}" ]]; then
      echo "Error: unsafe EXP_DIR parent changed: ${canonical_now}"
      return 1
    fi
    mkdir -- "${EXP_DIR}"
  fi
  if [[ -L "${EXP_DIR}" || ! -d "${EXP_DIR}" ]]; then
    echo "Error: unsafe EXP_DIR entity: ${EXP_DIR}"
    return 1
  fi
  canonical_now="$(realpath -m -- "${EXP_DIR}")"
  if [[ "${canonical_now}" != "${EXP_DIR}" ]]; then
    echo "Error: unsafe EXP_DIR canonical path changed: ${canonical_now}"
    return 1
  fi
}

audit_runtime_paths() {
  ensure_exp_entity
  audit_write_path "VAE directory" "${VAE_DIR}"
  audit_write_path "VAE checkpoint" "${VAE_CKPT_PATH}"
  audit_write_path "LDM directory" "${LDM_DIR}"
  audit_write_path "LDM checkpoint" "${LDM_CKPT_PATH}"
  audit_write_path "MINI_DATASET_DIR" "${MINI_DATASET_DIR}"
  audit_write_path "MINI_CONFIG_PATH" "${MINI_CONFIG_PATH}"
}

audit_runtime_paths
mkdir -p -- "${EXP_DIR}/vae"
audit_runtime_paths
cp -a -- "${BASE_VAE_CKPT}" "${VAE_CKPT_PATH}"
audit_runtime_paths
if [[ ! -s "${VAE_CKPT_PATH}" ]]; then
  echo "Error: copied VAE checkpoint is empty: ${VAE_CKPT_PATH}"
  exit 1
fi

if [[ -n "${CUDA_DEVICES:-}" ]]; then
  SELECTED_CUDA_DEVICES="${CUDA_DEVICES}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  SELECTED_CUDA_DEVICES="${CUDA_VISIBLE_DEVICES}"
else
  SELECTED_CUDA_DEVICES="0"
fi
export CUDA_DEVICES="${SELECTED_CUDA_DEVICES}"
export CUDA_VISIBLE_DEVICES="${SELECTED_CUDA_DEVICES}"
export EXP_DIR
export BASE_VAE_CKPT="${VAE_CKPT_PATH}"
export MINI_DATASET_DIR MINI_CONFIG_PATH
export PREPROCESSED_ROOT="${ROOT_DIR}/Data/NTU4DRadLM_Pre_sensor_aware"
export CALIB_CONFIG_DIR="${ROOT_DIR}/Data/config"
export MINI_REQUIRE_FRESH_SCRATCH="1"
export MINI_REQUIRE_FRESH_CONFIG="1"
export SAMPLES_PER_SCENE="500"
export MINI_BATCH_SIZE="1"
export MINI_NUM_WORKERS="2"
export MINI_GRAD_ACCUM="1"
export MINI_USE_AUG="false"
export MINI_LDM_EPOCHS="3"
export MINI_TARGET_SIZE="64,128,128"
export MINI_SOURCE_PC_RANGE="0,-20,-6,120,20,10"
export MINI_MODEL_PC_RANGE="0,-20,-6,40,20,10"
export MINI_VAE_CONFIG_TYPE="ultra_lightweight"
export MINI_VAE_LATENT_DIM=""
export MINI_VAE_OCC_LOSS="bce_dice"
export MINI_TRAIN_SPLIT="0.8"
export MINI_SPLIT_SEED="42"
export TRAIN_SCENES_OVERRIDE="garden"
export MINI_LDM_DECODED_WEIGHT="0.12"
export MINI_LDM_DECODED_FP_WEIGHT="0.20"
export MINI_LDM_DECODED_MASS_WEIGHT="0.08"
export MINI_LDM_HEIGHT_WEIGHT="0.04"
export MINI_LDM_TOP_WEIGHT="0.08"
export MINI_LDM_TOP_OVERSHOOT_WEIGHT="0.02"
export MINI_LDM_CONTINUITY_WEIGHT="0.02"
export MINI_LDM_DENSITY_WEIGHT="0.0"
export MINI_LDM_IR_FRUSTUM_OCC_WEIGHT="0.02"
export MINI_LDM_IR_FRUSTUM_NEGATIVE_WEIGHT="0.02"
export MINI_LDM_IR_FRUSTUM_TOP_WEIGHT="0.03"
export MINI_LDM_UNCERTAINTY_WEIGHT="0.0"
export MINI_LDM_COLUMN_CURRICULUM_ENABLED="${COLUMN_CURRICULUM_ENABLED}"
export MINI_LDM_COLUMN_POSITIVE_START_WEIGHT="${COLUMN_POSITIVE_START_WEIGHT}"
export MINI_LDM_COLUMN_NEGATIVE_START_WEIGHT="${COLUMN_NEGATIVE_START_WEIGHT}"
export MINI_LDM_COLUMN_POSITIVE_WEIGHT="${COLUMN_POSITIVE_WEIGHT}"
export MINI_LDM_COLUMN_NEGATIVE_WEIGHT="${COLUMN_NEGATIVE_WEIGHT}"
export MINI_LDM_COLUMN_TEMPERATURE="1.0"
export LDM_TRAIN_ONLY="1"
# v10 已在持锁状态下完成空目录审计，允许通用 runner 使用刚复制的 VAE。
export ALLOW_OVERWRITE="1"

echo "column variant=${V10_VARIANT}, curriculum=${MINI_LDM_COLUMN_CURRICULUM_ENABLED}, start pos/neg=${MINI_LDM_COLUMN_POSITIVE_START_WEIGHT}/${MINI_LDM_COLUMN_NEGATIVE_START_WEIGHT}, final pos/neg/temp=${MINI_LDM_COLUMN_POSITIVE_WEIGHT}/${MINI_LDM_COLUMN_NEGATIVE_WEIGHT}/${MINI_LDM_COLUMN_TEMPERATURE}"
audit_runtime_paths
bash "${SELF_DIR}/run_ldm_vertical_experiment.sh"
audit_runtime_paths

if [[ ! -s "${LDM_CKPT_PATH}" ]]; then
  echo "Error: final LDM checkpoint missing or empty: ${LDM_CKPT_PATH}"
  exit 1
fi
echo "v10 training complete: ${LDM_CKPT_PATH}"
