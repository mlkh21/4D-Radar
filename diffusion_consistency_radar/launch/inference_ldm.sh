#!/bin/bash
# LDM 推理脚本 - 固定为逐文件推理模式（1 个输入文件 -> 1 个生成点云文件）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"

INFER_SCRIPT="${PROJECT_DIR}/scripts/inference.py"
VAE_CKPT="${ROOT_DIR}/Result/train_results/vae/vae_best.pt"
LDM_CKPT="${ROOT_DIR}/Result/train_results/ldm/ldm_best.pt"
DATA_LOADING_CONFIG="${PROJECT_DIR}/config/data_loading_config.yml"
PREPROCESSED_ROOT="${ROOT_DIR}/Data/NTU4DRadLM_Pre"
RAW_ROOT="${ROOT_DIR}/Data/NTU4DRadLM_Raw"

if [ ! -f "${VAE_CKPT}" ]; then
    echo "错误: VAE 模型不存在: ${VAE_CKPT}"
    exit 1
fi

if [ ! -f "${LDM_CKPT}" ]; then
    echo "错误: LDM 模型不存在: ${LDM_CKPT}"
    exit 1
fi

if [ ! -f "${DATA_LOADING_CONFIG}" ]; then
    echo "错误: 配置文件不存在: ${DATA_LOADING_CONFIG}"
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

if [ ${#TEST_SCENES[@]} -eq 0 ]; then
    echo "错误: data_loading_config.yml 中 data.test 为空"
    exit 1
fi

for SCENE in "${TEST_SCENES[@]}"; do
    RADAR_VOXEL_DIR="${PREPROCESSED_ROOT}/${SCENE}/radar_voxel"
    RAW_LIVOX_DIR="${RAW_ROOT}/${SCENE}/livox_lidar"
    LIDAR_INDEX_FILE="${RAW_ROOT}/${SCENE}/lidar_index_sequence.txt"
    OUTPUT_DIR="${ROOT_DIR}/Result/inference_results/${SCENE}_ldm_eval"

    if [ ! -d "${RADAR_VOXEL_DIR}" ]; then
        echo "错误: radar_voxel 目录不存在: ${RADAR_VOXEL_DIR}"
        exit 1
    fi

    if [ ! -d "${RAW_LIVOX_DIR}" ]; then
        echo "错误: livox_lidar 目录不存在: ${RAW_LIVOX_DIR}"
        exit 1
    fi

    if [ ! -f "${LIDAR_INDEX_FILE}" ]; then
        echo "错误: lidar 索引文件不存在: ${LIDAR_INDEX_FILE}"
        exit 1
    fi

    echo "开始 LDM 推理场景: ${SCENE}"
    python "${INFER_SCRIPT}" \
        --vae_ckpt "${VAE_CKPT}" \
        --model_ckpt "${LDM_CKPT}" \
        --model_type ldm \
        --steps 40 \
        --sampler heun \
        --radar_voxel_dir "${RADAR_VOXEL_DIR}" \
        --save_pointcloud \
        --compare_with_lidar \
        --raw_livox_dir "${RAW_LIVOX_DIR}" \
        --lidar_index_file "${LIDAR_INDEX_FILE}" \
        --output_dir "${OUTPUT_DIR}" \
        --device cuda

    echo "完成: ${OUTPUT_DIR}"
done
