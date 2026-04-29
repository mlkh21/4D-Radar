#!/bin/bash
# Fast minimal inference with unified metrics output.
# Usage:
#   bash diffusion_consistency_radar/launch/inference_minimal.sh ldm
#   bash diffusion_consistency_radar/launch/inference_minimal.sh cd

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"

INFER_SCRIPT="${PROJECT_DIR}/scripts/inference.py"
DATA_LOADING_CONFIG="${PROJECT_DIR}/config/data_loading_config.yml"
PREPROCESSED_ROOT="${ROOT_DIR}/Data/NTU4DRadLM_Pre"
RAW_ROOT="${ROOT_DIR}/Data/NTU4DRadLM_Raw"

MODEL_TYPE="${1:-ldm}"
MAX_INFER_FILES="${MAX_INFER_FILES:-20}"
OCC_THRESHOLD="${OCC_THRESHOLD:-0.2}"
EMPTY_FALLBACK_TOPK="${EMPTY_FALLBACK_TOPK:-2000}"
ADAPTIVE_OCC_FROM_TARGET="${ADAPTIVE_OCC_FROM_TARGET:-0}"
ADAPTIVE_TARGET_THRESHOLD="${ADAPTIVE_TARGET_THRESHOLD:--1}"
TRAIN_DURATION_SECONDS="${TRAIN_DURATION_SECONDS:--1}"
DEVICE="${DEVICE:-cuda}"
USE_MINI_CHECKPOINTS="${USE_MINI_CHECKPOINTS:-0}"
USER_OUTPUT_DIR="${OUTPUT_DIR:-}"

if [[ "${USE_MINI_CHECKPOINTS}" == "1" ]]; then
  DEFAULT_RESULT_DIR="${ROOT_DIR}/Result/train_results_mini"
  DEFAULT_OUTPUT_ROOT="${ROOT_DIR}/Result/inference_results_mini"
else
  DEFAULT_RESULT_DIR="${ROOT_DIR}/Result/train_results"
  DEFAULT_OUTPUT_ROOT="${ROOT_DIR}/Result/inference_results"
fi

VAE_CKPT="${VAE_CKPT:-${DEFAULT_RESULT_DIR}/vae/vae_best.pt}"
if [[ "${MODEL_TYPE}" == "cd" ]]; then
  MODEL_CKPT="${MODEL_CKPT:-${ROOT_DIR}/Result/train_results/cd/cd_best.pt}"
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

mapfile -t TEST_SCENES < <(python - "${DATA_LOADING_CONFIG}" <<'PY'
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

if [[ ${#TEST_SCENES[@]} -eq 0 ]]; then
  echo "Error: data.test is empty in ${DATA_LOADING_CONFIG}"
  exit 1
fi

echo "=========================================="
echo "Minimal inference setup"
echo "model_type: ${MODEL_TYPE}"
echo "use_mini_checkpoints: ${USE_MINI_CHECKPOINTS}"
echo "model_ckpt: ${MODEL_CKPT}"
echo "vae_ckpt: ${VAE_CKPT}"
echo "steps/sampler: ${STEPS}/${SAMPLER}"
echo "max files per scene: ${MAX_INFER_FILES}"
echo "occ_threshold: ${OCC_THRESHOLD}"
echo "empty_fallback_topk: ${EMPTY_FALLBACK_TOPK}"
echo "adaptive_occ_from_target: ${ADAPTIVE_OCC_FROM_TARGET} (adaptive_target_threshold=${ADAPTIVE_TARGET_THRESHOLD})"
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

  EXTRA_ADAPTIVE_ARGS=()
  if [[ "${ADAPTIVE_OCC_FROM_TARGET}" == "1" ]]; then
    if [[ ! -d "${TARGET_VOXEL_DIR}" ]]; then
      echo "Warning: adaptive mode requested but missing ${TARGET_VOXEL_DIR}; fallback to fixed occ_threshold"
    else
      EXTRA_ADAPTIVE_ARGS+=(--adaptive_occ_from_target)
      EXTRA_ADAPTIVE_ARGS+=(--target_voxel_dir "${TARGET_VOXEL_DIR}")
      EXTRA_ADAPTIVE_ARGS+=(--adaptive_target_threshold "${ADAPTIVE_TARGET_THRESHOLD}")
    fi
  fi

  echo "Running minimal inference for scene: ${SCENE}"
  python "${INFER_SCRIPT}" \
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
    --save_pointcloud \
    --output_dir "${OUTPUT_DIR}" \
    "${EXTRA_ADAPTIVE_ARGS[@]}" \
    "${EXTRA_COMPARE_ARGS[@]}"

done

echo "Minimal inference done."
