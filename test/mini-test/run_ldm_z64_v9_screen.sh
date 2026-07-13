#!/bin/bash
# 运行 Z=64 LDM v9 A/B 单变量 screen 实验，并在训练后执行 32 帧 IR 消融。
#
# A/B 共用 v8 的数据、网格、VAE、随机种子和其余损失权重，只改变顶部过冲与
# IR 负样本权重。脚本不运行正式推理、可视化或 CD。

set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SELF_DIR}/../.." && pwd)"

V9_VARIANT="${V9_VARIANT:-A}"
case "${V9_VARIANT}" in
  A)
    DEFAULT_EXP_DIR="test/result/ldm/ablation/ldm_near40_500_z64_v9a_top_screen"
    TOP_OVERSHOOT_WEIGHT="0.02"
    IR_NEGATIVE_WEIGHT="0.02"
    ;;
  B)
    DEFAULT_EXP_DIR="test/result/ldm/ablation/ldm_near40_500_z64_v9b_irneg_screen"
    TOP_OVERSHOOT_WEIGHT="0.05"
    IR_NEGATIVE_WEIGHT="0.01"
    ;;
  *)
    echo "Error: V9_VARIANT must be A or B: ${V9_VARIANT}"
    exit 1
    ;;
esac

EXP_DIR="${EXP_DIR:-${DEFAULT_EXP_DIR}}"
BASE_VAE_CKPT="${BASE_VAE_CKPT:-test/result/vae/reconstruction/vae_near40_500_z64_upper_bound/vae/vae_best.pt}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-${ROOT_DIR}/Data/NTU4DRadLM_Pre_sensor_aware}"
TRAIN_SCENES_OVERRIDE="${TRAIN_SCENES_OVERRIDE:-garden}"
SAMPLES_PER_SCENE="${SAMPLES_PER_SCENE:-500}"
MINI_LDM_EPOCHS="${MINI_LDM_EPOCHS:-3}"
MINI_NUM_WORKERS="${MINI_NUM_WORKERS:-2}"
MINI_SPLIT_SEED="${MINI_SPLIT_SEED:-42}"
MINI_TARGET_SIZE="${MINI_TARGET_SIZE:-64,128,128}"
MINI_MODEL_PC_RANGE="${MINI_MODEL_PC_RANGE:-0,-20,-6,40,20,10}"
MINI_SOURCE_PC_RANGE="${MINI_SOURCE_PC_RANGE:-0,-20,-6,120,20,10}"
MINI_LDM_DECODED_WEIGHT="0.12"
MINI_LDM_DECODED_FP_WEIGHT="0.20"
MINI_LDM_DECODED_MASS_WEIGHT="0.08"
MINI_LDM_HEIGHT_WEIGHT="0.04"
MINI_LDM_TOP_WEIGHT="0.08"
MINI_LDM_TOP_OVERSHOOT_WEIGHT="${TOP_OVERSHOOT_WEIGHT}"
MINI_LDM_CONTINUITY_WEIGHT="0.02"
MINI_LDM_DENSITY_WEIGHT="0.015"
MINI_LDM_IR_FRUSTUM_OCC_WEIGHT="0.02"
MINI_LDM_IR_FRUSTUM_NEGATIVE_WEIGHT="${IR_NEGATIVE_WEIGHT}"
MINI_LDM_IR_FRUSTUM_TOP_WEIGHT="0.03"
MINI_LDM_UNCERTAINTY_WEIGHT="0.0"

# CUDA 只从一个选定值导出，避免上下游观察到不同设备。
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
EXP_DIR="$(realpath -m -- "${EXP_DIR_INPUT}")"
export EXP_DIR

# 实验目录只允许放在项目 test/result 的子目录或独立 /tmp 子目录中。
RESULT_ROOT="$(realpath -m -- "${ROOT_DIR}/test/result")"
case "${EXP_DIR}" in
  "${RESULT_ROOT}"/* | /tmp/*)
    ;;
  *)
    echo "Error: unsafe EXP_DIR: ${EXP_DIR}"
    echo "EXP_DIR must be a child of ${RESULT_ROOT} or /tmp."
    exit 1
    ;;
esac

canonical_exp_child() {
  local variable_name="$1"
  local input_path="$2"
  local kind="$3"
  local candidate
  if [[ "${input_path}" == /* ]]; then
    candidate="${input_path}"
  else
    candidate="${EXP_DIR}/${input_path}"
  fi
  if [[ -L "${candidate}" ]]; then
    echo "Error: unsafe ${variable_name}: path must not be a symlink: ${candidate}" >&2
    return 1
  fi
  candidate="$(realpath -m -- "${candidate}")"
  if [[ "${candidate}" != "${EXP_DIR}/"* ]]; then
    echo "Error: unsafe ${variable_name}: ${candidate}" >&2
    return 1
  fi
  if [[ "$(basename -- "${candidate}")" != .tmp_* ]]; then
    echo "Error: unsafe ${variable_name}: basename must start with .tmp_: ${candidate}" >&2
    return 1
  fi
  if [[ "${kind}" == "dataset" && -e "${candidate}" && ! -d "${candidate}" ]]; then
    echo "Error: unsafe ${variable_name}: dataset scratch must be a directory: ${candidate}" >&2
    return 1
  fi
  if [[ "${kind}" == "config" && -d "${candidate}" ]]; then
    echo "Error: unsafe ${variable_name}: config scratch must be a file path: ${candidate}" >&2
    return 1
  fi
  printf '%s\n' "${candidate}"
}

MINI_DATASET_DIR="$(canonical_exp_child \
  MINI_DATASET_DIR "${MINI_DATASET_DIR:-.tmp_mini_train_dataset}" dataset)"
MINI_CONFIG_PATH="$(canonical_exp_child \
  MINI_CONFIG_PATH "${MINI_CONFIG_PATH:-.tmp_ldm_config.yaml}" config)"
export MINI_DATASET_DIR MINI_CONFIG_PATH

VAE_DIR="${EXP_DIR}/vae"
VAE_CKPT_PATH="${VAE_DIR}/vae_best.pt"
LDM_DIR="${EXP_DIR}/ldm"
ABLATION_OUTPUT_DIR="${EXP_DIR}/ir_target_ablation_32_thr099"

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
    "${EXP_DIR}" | "${EXP_DIR}"/*)
      ;;
    *)
      echo "Error: unsafe ${label}: parent escapes EXP_DIR: ${parent}"
      return 1
      ;;
  esac
}

# 在首次创建目录前审计每个固定写入目标，复用实验目录时也不信任已有路径。
audit_write_path "VAE directory" "${VAE_DIR}"
audit_write_path "VAE checkpoint" "${VAE_CKPT_PATH}"
audit_write_path "LDM directory" "${LDM_DIR}"
audit_write_path "ablation output" "${ABLATION_OUTPUT_DIR}"
audit_write_path "MINI_DATASET_DIR" "${MINI_DATASET_DIR}"
audit_write_path "MINI_CONFIG_PATH" "${MINI_CONFIG_PATH}"

ensure_exp_entity() {
  local canonical_now
  if [[ -L "${EXP_DIR}" ]]; then
    echo "Error: unsafe EXP_DIR: path became a symlink: ${EXP_DIR}"
    return 1
  fi
  if [[ ! -e "${EXP_DIR}" ]]; then
    mkdir -- "${EXP_DIR}"
  fi
  if [[ -L "${EXP_DIR}" || ! -d "${EXP_DIR}" ]]; then
    echo "Error: unsafe EXP_DIR: expected an entity directory: ${EXP_DIR}"
    return 1
  fi
  canonical_now="$(realpath -m -- "${EXP_DIR}")"
  if [[ "${canonical_now}" != "${EXP_DIR}" ]]; then
    echo "Error: unsafe EXP_DIR: canonical path changed to ${canonical_now}"
    return 1
  fi
}

audit_runtime_paths() {
  ensure_exp_entity
  audit_write_path "VAE directory" "${VAE_DIR}"
  audit_write_path "VAE checkpoint" "${VAE_CKPT_PATH}"
  audit_write_path "LDM directory" "${LDM_DIR}"
  audit_write_path "ablation output" "${ABLATION_OUTPUT_DIR}"
  audit_write_path "MINI_DATASET_DIR" "${MINI_DATASET_DIR}"
  audit_write_path "MINI_CONFIG_PATH" "${MINI_CONFIG_PATH}"
}

if [[ "${BASE_VAE_CKPT}" != /* ]]; then
  BASE_VAE_CKPT="${ROOT_DIR}/${BASE_VAE_CKPT}"
fi
BASE_VAE_CKPT="$(realpath -m -- "${BASE_VAE_CKPT}")"

LOCK_PATH="${EXP_DIR}.lock"
LOCK_CREATED=0
cleanup_lock() {
  if [[ "${LOCK_CREATED}" == "1" ]]; then
    rmdir -- "${LOCK_PATH}" 2>/dev/null || true
  fi
}
mkdir -p -- "$(dirname "${LOCK_PATH}")"
if ! mkdir -- "${LOCK_PATH}" 2>/dev/null; then
  echo "Error: experiment is already running: ${EXP_DIR}"
  exit 1
fi
LOCK_CREATED=1
trap cleanup_lock EXIT INT TERM

# 锁只协调本 runner；持锁后创建并确认 EXP 是 canonical 的实体目录。
ensure_exp_entity

if [[ "${ALLOW_OVERWRITE}" != "1" && -e "${EXP_DIR}" ]]; then
  if [[ ! -d "${EXP_DIR}" || -n "$(find "${EXP_DIR}" -mindepth 1 -print -quit)" ]]; then
    echo "Error: experiment output already exists and is not empty: ${EXP_DIR}"
    echo "Set EXP_DIR to a new directory, or ALLOW_OVERWRITE=1 to overwrite intentionally."
    exit 1
  fi
fi
if [[ ! -f "${BASE_VAE_CKPT}" ]]; then
  echo "Error: base VAE checkpoint not found: ${BASE_VAE_CKPT}"
  exit 1
fi

# 每个写入/调用边界都重新 lstat 并检查 canonical 父路径，缩小非协作替换窗口。
audit_runtime_paths
mkdir -p -- "${VAE_DIR}"
audit_runtime_paths
cp -a -- "${BASE_VAE_CKPT}" "${VAE_CKPT_PATH}"

audit_runtime_paths
MINI_RESULTS_DIR="${EXP_DIR}" \
MINI_DATASET_DIR="${MINI_DATASET_DIR}" \
MINI_CONFIG_PATH="${MINI_CONFIG_PATH}" \
PREPROCESSED_ROOT="${PREPROCESSED_ROOT}" \
TRAIN_SCENES_OVERRIDE="${TRAIN_SCENES_OVERRIDE}" \
SAMPLES_PER_SCENE="${SAMPLES_PER_SCENE}" \
MINI_LDM_EPOCHS="${MINI_LDM_EPOCHS}" \
MINI_NUM_WORKERS="${MINI_NUM_WORKERS}" \
MINI_SPLIT_SEED="${MINI_SPLIT_SEED}" \
MINI_TARGET_SIZE="${MINI_TARGET_SIZE}" \
MINI_MODEL_PC_RANGE="${MINI_MODEL_PC_RANGE}" \
MINI_SOURCE_PC_RANGE="${MINI_SOURCE_PC_RANGE}" \
MINI_LDM_DECODED_WEIGHT="${MINI_LDM_DECODED_WEIGHT}" \
MINI_LDM_DECODED_FP_WEIGHT="${MINI_LDM_DECODED_FP_WEIGHT}" \
MINI_LDM_DECODED_MASS_WEIGHT="${MINI_LDM_DECODED_MASS_WEIGHT}" \
MINI_LDM_HEIGHT_WEIGHT="${MINI_LDM_HEIGHT_WEIGHT}" \
MINI_LDM_TOP_WEIGHT="${MINI_LDM_TOP_WEIGHT}" \
MINI_LDM_TOP_OVERSHOOT_WEIGHT="${MINI_LDM_TOP_OVERSHOOT_WEIGHT}" \
MINI_LDM_CONTINUITY_WEIGHT="${MINI_LDM_CONTINUITY_WEIGHT}" \
MINI_LDM_DENSITY_WEIGHT="${MINI_LDM_DENSITY_WEIGHT}" \
MINI_LDM_IR_FRUSTUM_OCC_WEIGHT="${MINI_LDM_IR_FRUSTUM_OCC_WEIGHT}" \
MINI_LDM_IR_FRUSTUM_NEGATIVE_WEIGHT="${MINI_LDM_IR_FRUSTUM_NEGATIVE_WEIGHT}" \
MINI_LDM_IR_FRUSTUM_TOP_WEIGHT="${MINI_LDM_IR_FRUSTUM_TOP_WEIGHT}" \
MINI_LDM_UNCERTAINTY_WEIGHT="${MINI_LDM_UNCERTAINTY_WEIGHT}" \
bash "${SELF_DIR}/train_minimal.sh" ldm

audit_runtime_paths
if [[ -L "${LDM_DIR}/ldm_best.pt" || ! -f "${LDM_DIR}/ldm_best.pt" ]]; then
  echo "Error: training completed without a safe LDM checkpoint: ${LDM_DIR}/ldm_best.pt"
  exit 1
fi

audit_runtime_paths
V7_DIR="${EXP_DIR}" \
DATASET_ROOT="${PREPROCESSED_ROOT}" \
OUTPUT_DIR="${ABLATION_OUTPUT_DIR}" \
ABLATION_MAX_SAMPLES="32" \
ABLATION_STEPS="20" \
OCC_THRESHOLD="0.99" \
TARGET_THRESHOLD="0.5" \
CUDA_DEVICES="${SELECTED_CUDA_DEVICES}" \
bash "${SELF_DIR}/run_ldm_z64_v7_target_ablation.sh"
