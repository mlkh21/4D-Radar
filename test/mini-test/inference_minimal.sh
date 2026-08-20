#!/bin/bash
# Fast minimal inference with outputs isolated under test/mini-test.
# Usage:
#   bash test/mini-test/inference_minimal.sh ldm
#   bash test/mini-test/inference_minimal.sh cd

set -euo pipefail

if [[ -n "${ADAPTIVE_OCC_FROM_TARGET+x}" || -n "${ADAPTIVE_TARGET_THRESHOLD+x}" ]]; then
  echo "Error: adaptive target threshold 已从推理入口移除；请运行 test/diagnostics/occupancy/diagnose_oracle_target_adaptation.py"
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_DIR="${ROOT_DIR}/diffusion_consistency_radar"

INFER_SCRIPT="${PROJECT_DIR}/scripts/inference.py"
DATA_LOADING_CONFIG="${PROJECT_DIR}/config/data_loading_config.yml"
MINI_RADAR_PROTOCOL="${MINI_RADAR_PROTOCOL:-legacy}"
case "${MINI_RADAR_PROTOCOL}" in
  legacy)
    DEFAULT_PREPROCESSED_ROOT="${ROOT_DIR}/Data/NTU4DRadLM_Pre_sensor_aware"
    DEFAULT_RAW_ROOT="${ROOT_DIR}/Data/NTU4DRadLM_Raw"
    DEFAULT_MODEL_PC_RANGE="0,-20,-6,40,20,10"
    ;;
  formal)
    DEFAULT_PREPROCESSED_ROOT="${ROOT_DIR}/Data/NTU4DRadLM_Pre_sensor_aware_p1_04_candidate"
    DEFAULT_RAW_ROOT="${ROOT_DIR}/Data/NTU4DRadLM_Raw_p1_01_candidate"
    DEFAULT_MODEL_PC_RANGE="0,-20,-6,120,20,10"
    ;;
  *)
    echo "Error: MINI_RADAR_PROTOCOL must be legacy or formal"
    exit 1
    ;;
esac
PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-${DEFAULT_PREPROCESSED_ROOT}}"
RAW_ROOT="${RAW_ROOT:-${DEFAULT_RAW_ROOT}}"
MINI_RESULTS_DIR="${MINI_RESULTS_DIR:-${SCRIPT_DIR}/train_results_mini}"
MINI_INFERENCE_RESULTS_DIR="${MINI_INFERENCE_RESULTS_DIR:-${SCRIPT_DIR}/inference_results_mini}"
MINI_TARGET_SIZE="${MINI_TARGET_SIZE:-32,128,128}"
MINI_SOURCE_PC_RANGE="${MINI_SOURCE_PC_RANGE:-0,-20,-6,120,20,10}"
MINI_MODEL_PC_RANGE="${MINI_MODEL_PC_RANGE:-${DEFAULT_MODEL_PC_RANGE}}"

MODEL_TYPE="${1:-ldm}"
MAX_INFER_FILES="${MAX_INFER_FILES:-20}"
# Sensor-aware mini validation selected 0.5 by voxel F1. Override with
# OCC_THRESHOLD when calibrating a different checkpoint/protocol.
OCC_THRESHOLD="${OCC_THRESHOLD:-0.5}"
EMPTY_FALLBACK_TOPK="${EMPTY_FALLBACK_TOPK:-2000}"
TRAIN_DURATION_SECONDS="${TRAIN_DURATION_SECONDS:--1}"
DEVICE="${DEVICE:-cuda}"
USE_MINI_CHECKPOINTS="${USE_MINI_CHECKPOINTS:-1}"
USER_OUTPUT_DIR="${OUTPUT_DIR:-}"

RADAR_PROTOCOL_ARGS=()
REAL_IR_ARGS=()
if [[ "${MINI_RADAR_PROTOCOL}" == "legacy" ]]; then
  RADAR_PROTOCOL_ARGS+=(--allow_legacy_radar_units)
else
  REAL_IR_ARGS+=(--require_real_ir)
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD=("${PYTHON_BIN}")
elif python -c "import torch" >/dev/null 2>&1; then
  PYTHON_CMD=(python)
elif command -v conda >/dev/null 2>&1; then
  PYTHON_CMD=(conda run -n Radar-Diffusion python)
else
  PYTHON_CMD=(python)
fi

if python3 -c "import yaml" >/dev/null 2>&1; then
  CONFIG_PYTHON_CMD=(python3)
elif python -c "import yaml" >/dev/null 2>&1; then
  CONFIG_PYTHON_CMD=(python)
else
  CONFIG_PYTHON_CMD=("${PYTHON_CMD[@]}")
fi

if [[ "${USE_MINI_CHECKPOINTS}" == "1" ]]; then
  DEFAULT_RESULT_DIR="${MINI_RESULTS_DIR}"
  DEFAULT_OUTPUT_ROOT="${MINI_INFERENCE_RESULTS_DIR}"
else
  DEFAULT_RESULT_DIR="${ROOT_DIR}/test/result"
  DEFAULT_OUTPUT_ROOT="${ROOT_DIR}/test/result/ldm/visualization/mini_inference_compare"
fi

VAE_CKPT="${VAE_CKPT:-${DEFAULT_RESULT_DIR}/vae/vae_best.pt}"
if [[ "${MODEL_TYPE}" == "cd" ]]; then
  MODEL_CKPT="${MODEL_CKPT:-${DEFAULT_RESULT_DIR}/cd/cd_best.pt}"
  STEPS="${STEPS:-1}"
  SAMPLER="${SAMPLER:-euler}"
else
  MODEL_CKPT="${MODEL_CKPT:-${DEFAULT_RESULT_DIR}/ldm/ldm_best.pt}"
  STEPS="${STEPS:-40}"
  SAMPLER="${SAMPLER:-heun}"
fi

if [[ ! -f "${VAE_CKPT}" ]]; then
  echo "Error: VAE checkpoint not found: ${VAE_CKPT}"
  exit 1
fi

if [[ ! -f "${MODEL_CKPT}" ]]; then
  echo "Error: model checkpoint not found: ${MODEL_CKPT}"
  exit 1
fi

if [[ ! -f "${DATA_LOADING_CONFIG}" ]]; then
  echo "Error: config not found: ${DATA_LOADING_CONFIG}"
  exit 1
fi

MULTIMODAL_META_ARGS=()
if "${PYTHON_CMD[@]}" -c 'import sys, torch
path = sys.argv[1]
try:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
except TypeError:
    ckpt = torch.load(path, map_location="cpu")
except Exception:
    ckpt = torch.load(path, map_location="cpu")
state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else {}
keys = tuple(state.keys()) if isinstance(state, dict) else ()
prefixes = ("unet_3d.", "ir_extractor.", "projection_layer.", "radar_encoder.", "uncertainty_head.", "fusion_conv.")
raise SystemExit(0 if any(k.startswith(prefixes) for k in keys) else 1)' "${MODEL_CKPT}"
then
  MULTIMODAL_META_ARGS+=(--use_multimodal_meta)
fi

if [[ -n "${SCENE:-}" ]]; then
  TEST_SCENES=("${SCENE}")
else
mapfile -t TEST_SCENES < <("${CONFIG_PYTHON_CMD[@]}" - "${DATA_LOADING_CONFIG}" <<'PY'
import sys
import yaml

with open(sys.argv[1], 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f) or {}

scenes = (cfg.get('data') or {}).get('test') or []
if isinstance(scenes, str):
    scenes = [scenes]

for scene in scenes:
    s = str(scene).strip()
    if s:
        print(s)
PY
)
fi

if [[ ${#TEST_SCENES[@]} -eq 0 ]]; then
  echo "Error: data.test is empty in ${DATA_LOADING_CONFIG}"
  exit 1
fi

echo "=========================================="
echo "Minimal inference setup"
echo "model_type: ${MODEL_TYPE}"
echo "use_mini_checkpoints: ${USE_MINI_CHECKPOINTS}"
echo "project dir: ${PROJECT_DIR}"
echo "preprocessed root: ${PREPROCESSED_ROOT}"
echo "model_ckpt: ${MODEL_CKPT}"
echo "vae_ckpt: ${VAE_CKPT}"
echo "steps/sampler: ${STEPS}/${SAMPLER}"
echo "max files per scene: ${MAX_INFER_FILES}"
echo "occ_threshold: ${OCC_THRESHOLD}"
echo "empty_fallback_topk: ${EMPTY_FALLBACK_TOPK}"
echo "target size [Z,X,Y]: ${MINI_TARGET_SIZE}"
echo "source pc range: ${MINI_SOURCE_PC_RANGE}"
echo "model pc range: ${MINI_MODEL_PC_RANGE}"
echo "radar protocol: ${MINI_RADAR_PROTOCOL}"
echo "=========================================="

for SCENE in "${TEST_SCENES[@]}"; do
  RADAR_VOXEL_DIR="${PREPROCESSED_ROOT}/${SCENE}/radar_voxel"
  TARGET_VOXEL_DIR="${PREPROCESSED_ROOT}/${SCENE}/target_voxel"
  RAW_LIVOX_DIR="${RAW_ROOT}/${SCENE}/livox_lidar"
  RAW_RADAR_DIR="${RAW_ROOT}/${SCENE}/radar_pcl"
  LIDAR_INDEX_FILE="${RAW_ROOT}/${SCENE}/lidar_index_sequence.txt"
  RADAR_INDEX_FILE="${RAW_ROOT}/${SCENE}/radar_index_sequence.txt"
  if [[ -n "${USER_OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="${USER_OUTPUT_DIR}"
  else
    OUTPUT_DIR="${DEFAULT_OUTPUT_ROOT}/${SCENE}_${MODEL_TYPE}_eval"
  fi

  if [[ ! -d "${RADAR_VOXEL_DIR}" ]]; then
    echo "Warning: skip scene ${SCENE}, missing ${RADAR_VOXEL_DIR}"
    continue
  fi

  EXTRA_COMPARE_ARGS=()
  if [[ -d "${RAW_LIVOX_DIR}" && -f "${LIDAR_INDEX_FILE}" ]]; then
    EXTRA_COMPARE_ARGS+=(--compare_with_lidar)
    EXTRA_COMPARE_ARGS+=(--raw_livox_dir "${RAW_LIVOX_DIR}")
    EXTRA_COMPARE_ARGS+=(--lidar_index_file "${LIDAR_INDEX_FILE}")
  fi
  if [[ -d "${TARGET_VOXEL_DIR}" ]]; then
    EXTRA_COMPARE_ARGS+=(--target_voxel_dir "${TARGET_VOXEL_DIR}")
    EXTRA_COMPARE_ARGS+=(--compare_with_target)
  fi

  echo "Running minimal inference for scene: ${SCENE}"
  "${PYTHON_CMD[@]}" "${INFER_SCRIPT}" \
    --vae_ckpt "${VAE_CKPT}" \
    --model_ckpt "${MODEL_CKPT}" \
    --model_type "${MODEL_TYPE}" \
    --steps "${STEPS}" \
    --sampler "${SAMPLER}" \
    --device "${DEVICE}" \
    --train_duration_seconds "${TRAIN_DURATION_SECONDS}" \
    --radar_voxel_dir "${RADAR_VOXEL_DIR}" \
    --raw_radar_dir "${RAW_RADAR_DIR}" \
    --radar_index_file "${RADAR_INDEX_FILE}" \
    --max_files "${MAX_INFER_FILES}" \
    --occ_threshold "${OCC_THRESHOLD}" \
    --empty_fallback_topk "${EMPTY_FALLBACK_TOPK}" \
    --target_size ${MINI_TARGET_SIZE//,/ } \
    --source_pc_range ${MINI_SOURCE_PC_RANGE//,/ } \
    --pc_range ${MINI_MODEL_PC_RANGE//,/ } \
    --report_task_metrics \
    --save_voxel \
    --save_uncertainty \
    --save_pointcloud \
    --output_dir "${OUTPUT_DIR}" \
    "${RADAR_PROTOCOL_ARGS[@]}" \
    "${REAL_IR_ARGS[@]}" \
    "${MULTIMODAL_META_ARGS[@]}" \
    "${EXTRA_COMPARE_ARGS[@]}"

done

echo "Minimal inference done."
