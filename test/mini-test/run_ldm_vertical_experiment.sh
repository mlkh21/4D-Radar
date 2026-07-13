#!/bin/bash
# 一键运行近场 LDM 竖向结构恢复实验。
#
# 默认实验：500 帧、LDM 10 epoch、height loss=0.05、continuity loss=0.02。
# 脚本会顺序执行：准备 VAE checkpoint -> 训练 LDM -> 推理 -> 竖向结构评估 -> 生成 raw LiDAR 3D HTML。

set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SELF_DIR}/../.." && pwd)"

EXP_DIR="${EXP_DIR:-test/result/ldm/vertical_structure/ldm_near40_500_vertical_v2}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
SAMPLES_PER_SCENE="${SAMPLES_PER_SCENE:-500}"
MINI_LDM_EPOCHS="${MINI_LDM_EPOCHS:-10}"
MINI_NUM_WORKERS="${MINI_NUM_WORKERS:-2}"
MINI_LDM_DECODED_WEIGHT="${MINI_LDM_DECODED_WEIGHT:-}"
MINI_LDM_DECODED_FP_WEIGHT="${MINI_LDM_DECODED_FP_WEIGHT:-}"
MINI_LDM_DECODED_MASS_WEIGHT="${MINI_LDM_DECODED_MASS_WEIGHT:-}"
MINI_LDM_HEIGHT_WEIGHT="${MINI_LDM_HEIGHT_WEIGHT:-0.05}"
MINI_LDM_TOP_WEIGHT="${MINI_LDM_TOP_WEIGHT:-0.0}"
MINI_LDM_TOP_OVERSHOOT_WEIGHT="${MINI_LDM_TOP_OVERSHOOT_WEIGHT:-0.0}"
MINI_LDM_CONTINUITY_WEIGHT="${MINI_LDM_CONTINUITY_WEIGHT:-0.02}"
MINI_LDM_DENSITY_WEIGHT="${MINI_LDM_DENSITY_WEIGHT:-0.0}"
MINI_LDM_IR_FRUSTUM_OCC_WEIGHT="${MINI_LDM_IR_FRUSTUM_OCC_WEIGHT:-0.0}"
MINI_LDM_IR_FRUSTUM_NEGATIVE_WEIGHT="${MINI_LDM_IR_FRUSTUM_NEGATIVE_WEIGHT:-0.0}"
MINI_LDM_IR_FRUSTUM_TOP_WEIGHT="${MINI_LDM_IR_FRUSTUM_TOP_WEIGHT:-0.0}"
MINI_LDM_UNCERTAINTY_WEIGHT="${MINI_LDM_UNCERTAINTY_WEIGHT:-}"
MINI_LDM_COLUMN_POSITIVE_WEIGHT="${MINI_LDM_COLUMN_POSITIVE_WEIGHT:-0.0}"
MINI_LDM_COLUMN_NEGATIVE_WEIGHT="${MINI_LDM_COLUMN_NEGATIVE_WEIGHT:-0.0}"
MINI_LDM_COLUMN_TEMPERATURE="${MINI_LDM_COLUMN_TEMPERATURE:-1.0}"
LDM_TRAIN_ONLY="${LDM_TRAIN_ONLY:-0}"

SCENE="${SCENE:-loop3}"
TRAIN_SCENES_OVERRIDE="${TRAIN_SCENES_OVERRIDE:-garden}"
MAX_INFER_FILES="${MAX_INFER_FILES:-500}"
OCC_THRESHOLD="${OCC_THRESHOLD:-0.05}"
TARGET_THRESHOLD="${TARGET_THRESHOLD:-0.5}"
DEVICE="${DEVICE:-cuda}"

MINI_TARGET_SIZE="${MINI_TARGET_SIZE:-32,128,128}"
MINI_SOURCE_PC_RANGE="${MINI_SOURCE_PC_RANGE:-0,-20,-6,120,20,10}"
MINI_MODEL_PC_RANGE="${MINI_MODEL_PC_RANGE:-0,-20,-6,40,20,10}"

VIS_FRAMES="${VIS_FRAMES:-000037,000103,000195,000229,000280,000303,000311,000431,000454,000493}"
Z_MIN="${Z_MIN:--1}"
X_MAX="${X_MAX:-40}"

PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-${ROOT_DIR}/Data/NTU4DRadLM_Pre_sensor_aware}"
RAW_LIDAR_DIR="${RAW_LIDAR_DIR:-${ROOT_DIR}/Data/NTU4DRadLM_Raw/${SCENE}/livox_lidar}"
LIDAR_INDEX_FILE="${LIDAR_INDEX_FILE:-${ROOT_DIR}/Data/NTU4DRadLM_Raw/${SCENE}/lidar_index_sequence.txt}"
BASE_VAE_CKPT="${BASE_VAE_CKPT:-}"

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
RESULT_ROOT="$(realpath -m -- "${ROOT_DIR}/test/result")"
case "${EXP_DIR}" in
  "${RESULT_ROOT}"/* | /tmp/*) ;;
  *)
    echo "Error: unsafe EXP_DIR: ${EXP_DIR}"
    echo "EXP_DIR must be a child of ${RESULT_ROOT} or /tmp."
    exit 1
    ;;
esac
if [[ -n "${BASE_VAE_CKPT}" && "${BASE_VAE_CKPT}" != /* ]]; then
  BASE_VAE_CKPT="${ROOT_DIR}/${BASE_VAE_CKPT}"
fi
MINI_DATASET_DIR="${MINI_DATASET_DIR:-${EXP_DIR}/.tmp_mini_train_dataset}"
MINI_CONFIG_PATH="${MINI_CONFIG_PATH:-${EXP_DIR}/.tmp_ldm_config.yaml}"

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
    echo "Error: unsafe ${variable_name}: path must be inside EXP_DIR: ${candidate}" >&2
    return 1
  fi
  if [[ "$(basename -- "${candidate}")" != .tmp_* ]]; then
    echo "Error: unsafe ${variable_name}: basename must start with .tmp_: ${candidate}" >&2
    return 1
  fi
  if [[ "${kind}" == "dataset" && -e "${candidate}" && ! -d "${candidate}" ]]; then
    echo "Error: unsafe ${variable_name}: dataset path must not be a file: ${candidate}" >&2
    return 1
  fi
  if [[ "${kind}" == "config" && -d "${candidate}" ]]; then
    echo "Error: unsafe ${variable_name}: config path must not be a directory: ${candidate}" >&2
    return 1
  fi
  printf '%s\n' "${candidate}"
}

MINI_DATASET_DIR="$(canonical_exp_child MINI_DATASET_DIR "${MINI_DATASET_DIR}" dataset)"
MINI_CONFIG_PATH="$(canonical_exp_child MINI_CONFIG_PATH "${MINI_CONFIG_PATH}" config)"

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


if [[ "${ALLOW_OVERWRITE}" != "1" && -e "${EXP_DIR}" ]]; then
  if [[ ! -d "${EXP_DIR}" || -n "$(find "${EXP_DIR}" -mindepth 1 -print -quit)" ]]; then
    echo "Error: experiment output already exists and is not empty: ${EXP_DIR}"
    echo "Set EXP_DIR to a new directory, or ALLOW_OVERWRITE=1 to overwrite intentionally."
    exit 1
  fi
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD=("${PYTHON_BIN}")
elif command -v conda >/dev/null 2>&1; then
  PYTHON_CMD=(conda run -n Radar-Diffusion python)
else
  PYTHON_CMD=(python)
fi

IFS=',' read -r -a TARGET_SIZE_ARGS <<< "${MINI_TARGET_SIZE}"
IFS=',' read -r -a SOURCE_PC_RANGE_ARGS <<< "${MINI_SOURCE_PC_RANGE}"
IFS=',' read -r -a MODEL_PC_RANGE_ARGS <<< "${MINI_MODEL_PC_RANGE}"

if [[ ${#TARGET_SIZE_ARGS[@]} -ne 3 ]]; then
  echo "Error: MINI_TARGET_SIZE must contain 3 values: ${MINI_TARGET_SIZE}"
  exit 1
fi
if [[ ${#SOURCE_PC_RANGE_ARGS[@]} -ne 6 || ${#MODEL_PC_RANGE_ARGS[@]} -ne 6 ]]; then
  echo "Error: MINI_SOURCE_PC_RANGE and MINI_MODEL_PC_RANGE must contain 6 values"
  exit 1
fi

mkdir -p "${EXP_DIR}/vae"

find_base_vae() {
  local candidates=()
  if [[ -n "${BASE_VAE_CKPT}" ]]; then
    candidates+=("${BASE_VAE_CKPT}")
  fi
  candidates+=(
    "${EXP_DIR}/vae/vae_best.pt"
    "${ROOT_DIR}/test/result/ldm/vertical_structure/ldm_near40_500_vertical_v1/vae/vae_best.pt"
    "${ROOT_DIR}/test/mini-test/train_results_near40_loop3/vae/vae_best.pt"
    "${ROOT_DIR}/test/mini-test/train_results_mini_calibrated/vae/vae_best.pt"
    "${ROOT_DIR}/test/mini-test/train_results_mini/vae/vae_best.pt"
  )

  local path
  for path in "${candidates[@]}"; do
    if [[ -s "${path}" ]]; then
      echo "${path}"
      return 0
    fi
  done
  return 1
}

if [[ ! -s "${EXP_DIR}/vae/vae_best.pt" ]]; then
  if BASE_FOUND="$(find_base_vae)"; then
    echo "Using base VAE checkpoint: ${BASE_FOUND}"
    cp -a "${BASE_FOUND}" "${EXP_DIR}/vae/vae_best.pt"
  else
    echo "Error: no VAE checkpoint found."
    echo "Set BASE_VAE_CKPT=/path/to/vae_best.pt, or run VAE first into ${EXP_DIR}."
    exit 1
  fi
fi
if [[ ! -s "${EXP_DIR}/vae/vae_best.pt" ]]; then
  echo "Error: VAE checkpoint missing or empty: ${EXP_DIR}/vae/vae_best.pt"
  exit 1
fi

echo "=========================================="
echo "LDM vertical experiment"
echo "exp dir: ${EXP_DIR}"
echo "train scenes: ${TRAIN_SCENES_OVERRIDE}"
echo "test scene: ${SCENE}"
echo "samples per scene: ${SAMPLES_PER_SCENE}"
echo "ldm epochs: ${MINI_LDM_EPOCHS}"
echo "decoded/FP/mass weights: ${MINI_LDM_DECODED_WEIGHT:-default}/${MINI_LDM_DECODED_FP_WEIGHT:-default}/${MINI_LDM_DECODED_MASS_WEIGHT:-default}"
echo "height/top/continuity/density weights: ${MINI_LDM_HEIGHT_WEIGHT}/${MINI_LDM_TOP_WEIGHT}/${MINI_LDM_CONTINUITY_WEIGHT}/${MINI_LDM_DENSITY_WEIGHT}"
echo "top overshoot weight: ${MINI_LDM_TOP_OVERSHOOT_WEIGHT}"
echo "IR frustum occupancy/top weights: ${MINI_LDM_IR_FRUSTUM_OCC_WEIGHT}/${MINI_LDM_IR_FRUSTUM_TOP_WEIGHT}"
echo "IR frustum negative weight: ${MINI_LDM_IR_FRUSTUM_NEGATIVE_WEIGHT}"
echo "uncertainty weight: ${MINI_LDM_UNCERTAINTY_WEIGHT:-default}"
echo "column positive/negative weights: ${MINI_LDM_COLUMN_POSITIVE_WEIGHT}/${MINI_LDM_COLUMN_NEGATIVE_WEIGHT}"
echo "column temperature: ${MINI_LDM_COLUMN_TEMPERATURE}"
echo "occ threshold: ${OCC_THRESHOLD}"
echo "model pc range: ${MINI_MODEL_PC_RANGE}"
echo "target size: ${MINI_TARGET_SIZE}"
echo "=========================================="

MINI_RESULTS_DIR="${EXP_DIR}" \
MINI_DATASET_DIR="${MINI_DATASET_DIR}" \
MINI_CONFIG_PATH="${MINI_CONFIG_PATH}" \
SAMPLES_PER_SCENE="${SAMPLES_PER_SCENE}" \
TRAIN_SCENES_OVERRIDE="${TRAIN_SCENES_OVERRIDE}" \
MINI_LDM_EPOCHS="${MINI_LDM_EPOCHS}" \
MINI_NUM_WORKERS="${MINI_NUM_WORKERS}" \
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
MINI_LDM_COLUMN_POSITIVE_WEIGHT="${MINI_LDM_COLUMN_POSITIVE_WEIGHT}" \
MINI_LDM_COLUMN_NEGATIVE_WEIGHT="${MINI_LDM_COLUMN_NEGATIVE_WEIGHT}" \
MINI_LDM_COLUMN_TEMPERATURE="${MINI_LDM_COLUMN_TEMPERATURE}" \
MINI_TARGET_SIZE="${MINI_TARGET_SIZE}" \
MINI_SOURCE_PC_RANGE="${MINI_SOURCE_PC_RANGE}" \
MINI_MODEL_PC_RANGE="${MINI_MODEL_PC_RANGE}" \
PREPROCESSED_ROOT="${PREPROCESSED_ROOT}" \
bash "${SELF_DIR}/train_minimal.sh" ldm

if [[ ! -s "${EXP_DIR}/ldm/ldm_best.pt" ]]; then
  echo "Error: final LDM checkpoint not found: ${EXP_DIR}/ldm/ldm_best.pt"
  exit 1
fi
if [[ "${LDM_TRAIN_ONLY}" == "1" ]]; then
  echo "LDM training complete: ${EXP_DIR}/ldm/ldm_best.pt"
  exit 0
fi

MINI_RESULTS_DIR="${EXP_DIR}" \
MINI_INFERENCE_RESULTS_DIR="${EXP_DIR}" \
MAX_INFER_FILES="${MAX_INFER_FILES}" \
OCC_THRESHOLD="${OCC_THRESHOLD}" \
DEVICE="${DEVICE}" \
SCENE="${SCENE}" \
MINI_TARGET_SIZE="${MINI_TARGET_SIZE}" \
MINI_SOURCE_PC_RANGE="${MINI_SOURCE_PC_RANGE}" \
MINI_MODEL_PC_RANGE="${MINI_MODEL_PC_RANGE}" \
PREPROCESSED_ROOT="${PREPROCESSED_ROOT}" \
bash "${SELF_DIR}/inference_minimal.sh" ldm

"${PYTHON_CMD[@]}" "${ROOT_DIR}/test/evaluation/ldm/evaluate_ldm_vertical_structure.py" \
  --pred_voxel_dir "${EXP_DIR}/${SCENE}_ldm_eval" \
  --target_voxel_dir "${PREPROCESSED_ROOT}/${SCENE}/target_voxel" \
  --output_dir "${EXP_DIR}/vertical_structure_eval" \
  --occ_threshold "${OCC_THRESHOLD}" \
  --target_threshold "${TARGET_THRESHOLD}" \
  --pc_range "${MODEL_PC_RANGE_ARGS[@]}" \
  --source_pc_range "${SOURCE_PC_RANGE_ARGS[@]}" \
  --target_size "${TARGET_SIZE_ARGS[@]}"

if [[ -d "${RAW_LIDAR_DIR}" && -f "${LIDAR_INDEX_FILE}" ]]; then
  "${PYTHON_CMD[@]}" "${ROOT_DIR}/test/visualization/generate_interactive_inference_compare.py" \
    --pre_dir "${PREPROCESSED_ROOT}/${SCENE}" \
    --raw_lidar_dir "${RAW_LIDAR_DIR}" \
    --lidar_index_file "${LIDAR_INDEX_FILE}" \
    --ldm_dir "${EXP_DIR}/${SCENE}_ldm_eval" \
    --output_dir "${EXP_DIR}/raw_lidar_visuals" \
    --frames "${VIS_FRAMES}" \
    --pc_range "${MODEL_PC_RANGE_ARGS[@]}" \
    --z_min "${Z_MIN}" \
    --x_max "${X_MAX}"
else
  echo "Warning: raw LiDAR directory or index file not found; skip HTML visualization."
fi

echo "=========================================="
echo "Experiment complete."
echo "Inference metrics: ${EXP_DIR}/${SCENE}_ldm_eval/inference_metrics.csv"
echo "Vertical report: ${EXP_DIR}/vertical_structure_eval/vertical_structure_report.md"
echo "3D HTML dir: ${EXP_DIR}/raw_lidar_visuals"
echo "=========================================="
