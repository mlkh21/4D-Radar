#!/bin/bash
# 文件功能：运行 Z=64 近场 VAE 上界实验，并在训练完成后执行 VAE 重建诊断。
# 说明：本脚本只定义可复现实验入口；请按需手动执行，避免在自动测试中启动长训练。

set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SELF_DIR}/../.." && pwd)"
TRAIN_SCRIPT="${SELF_DIR}/train_minimal.sh"
DIAG_SCRIPT="${ROOT_DIR}/diffusion_consistency_radar/scripts/diagnose_vae_reconstruction.py"

EXP_DIR="${EXP_DIR:-test/result/vae/reconstruction/vae_near40_500_z64_upper_bound}"
SCENE="${SCENE:-garden}"
MINI_TARGET_SIZE="${MINI_TARGET_SIZE:-64,128,128}"
MINI_MODEL_PC_RANGE="${MINI_MODEL_PC_RANGE:-0,-20,-6,40,20,10}"
MINI_SOURCE_PC_RANGE="${MINI_SOURCE_PC_RANGE:-0,-20,-6,120,20,10}"
SAMPLES_PER_SCENE="${SAMPLES_PER_SCENE:-500}"
MINI_VAE_EPOCHS="${MINI_VAE_EPOCHS:-10}"
MINI_VAE_CONFIG_TYPE="${MINI_VAE_CONFIG_TYPE:-lightweight}"
MINI_VAE_LATENT_DIM="${MINI_VAE_LATENT_DIM:-8}"
MINI_VAE_OCC_LOSS="${MINI_VAE_OCC_LOSS:-bce_dice}"
MINI_NUM_WORKERS="${MINI_NUM_WORKERS:-2}"
MAX_DIAG_FILES="${MAX_DIAG_FILES:-0}"
DIAG_DEVICE="${DIAG_DEVICE:-cuda}"

if [[ -n "${CUDA_DEVICES:-}" ]]; then
	CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${CUDA_DEVICES}}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
	CUDA_DEVICES="${CUDA_VISIBLE_DEVICES}"
else
	CUDA_DEVICES="0"
	CUDA_VISIBLE_DEVICES="0"
fi

if [[ "${EXP_DIR}" = /* ]]; then
	EXP_DIR_ABS="${EXP_DIR}"
else
	EXP_DIR_ABS="${ROOT_DIR}/${EXP_DIR}"
fi

MINI_RESULTS_DIR="${MINI_RESULTS_DIR:-${EXP_DIR_ABS}}"
MINI_DATASET_DIR="${MINI_DATASET_DIR:-${EXP_DIR_ABS}/mini_dataset}"
MINI_CONFIG_PATH="${MINI_CONFIG_PATH:-${EXP_DIR_ABS}/default_config.z64_upper_bound.yaml}"
DIAG_OUTPUT_DIR="${DIAG_OUTPUT_DIR:-${EXP_DIR_ABS}/vae_upper_bound}"
TARGET_VOXEL_DIR="${TARGET_VOXEL_DIR:-${MINI_DATASET_DIR}/${SCENE}/target_voxel}"
VAE_CKPT="${VAE_CKPT:-${MINI_RESULTS_DIR}/vae/vae_best.pt}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
	PYTHON_CMD=("${PYTHON_BIN}")
elif python -c "import torch" >/dev/null 2>&1; then
	PYTHON_CMD=(python)
elif command -v conda >/dev/null 2>&1; then
	PYTHON_CMD=(conda run -n Radar-Diffusion python)
else
	PYTHON_CMD=(python)
fi

mkdir -p "${EXP_DIR_ABS}" "${DIAG_OUTPUT_DIR}"

export CUDA_DEVICES CUDA_VISIBLE_DEVICES
export SCENE SAMPLES_PER_SCENE
export MINI_RESULTS_DIR MINI_DATASET_DIR MINI_CONFIG_PATH
export MINI_TARGET_SIZE MINI_SOURCE_PC_RANGE MINI_MODEL_PC_RANGE
export MINI_VAE_EPOCHS MINI_VAE_CONFIG_TYPE MINI_VAE_LATENT_DIM MINI_VAE_OCC_LOSS
export MINI_NUM_WORKERS

echo "=========================================="
echo "Z=64 VAE upper-bound experiment"
echo "scene: ${SCENE}"
echo "experiment dir: ${EXP_DIR_ABS}"
echo "mini results dir: ${MINI_RESULTS_DIR}"
echo "mini dataset dir: ${MINI_DATASET_DIR}"
echo "mini config path: ${MINI_CONFIG_PATH}"
echo "target size [Z,X,Y]: ${MINI_TARGET_SIZE}"
echo "source pc range: ${MINI_SOURCE_PC_RANGE}"
echo "model pc range: ${MINI_MODEL_PC_RANGE}"
echo "samples per scene: ${SAMPLES_PER_SCENE}"
echo "vae epochs: ${MINI_VAE_EPOCHS}"
echo "vae config type: ${MINI_VAE_CONFIG_TYPE}"
echo "vae latent dim: ${MINI_VAE_LATENT_DIM}"
echo "vae occupancy loss: ${MINI_VAE_OCC_LOSS}"
echo "num workers: ${MINI_NUM_WORKERS}"
echo "cuda devices: ${CUDA_DEVICES}"
echo "diagnostic max files: ${MAX_DIAG_FILES}"
echo "diagnostic output dir: ${DIAG_OUTPUT_DIR}"
echo "=========================================="

bash "${TRAIN_SCRIPT}" vae

if [[ ! -f "${VAE_CKPT}" ]]; then
	echo "Error: VAE checkpoint not found after training: ${VAE_CKPT}"
	exit 1
fi

if [[ ! -d "${TARGET_VOXEL_DIR}" ]]; then
	echo "Error: target voxel directory not found for diagnosis: ${TARGET_VOXEL_DIR}"
	exit 1
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PYTHON_CMD[@]}" "${DIAG_SCRIPT}" \
	--vae_ckpt "${VAE_CKPT}" \
	--target_voxel_dir "${TARGET_VOXEL_DIR}" \
	--output_dir "${DIAG_OUTPUT_DIR}" \
	--max_files "${MAX_DIAG_FILES}" \
	--target_size "${MINI_TARGET_SIZE}" \
	--source_pc_range "${MINI_SOURCE_PC_RANGE}" \
	--model_pc_range "${MINI_MODEL_PC_RANGE}" \
	--device "${DIAG_DEVICE}"

echo "Z=64 VAE upper-bound diagnosis done: ${DIAG_OUTPUT_DIR}"
