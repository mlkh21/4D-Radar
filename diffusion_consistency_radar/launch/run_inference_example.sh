#!/bin/bash
# 完整推理示例 - 演示如何使用 LDM 和 CD 模型

set -euo pipefail  # 遇到错误立即退出

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"
INFER_SCRIPT="${PROJECT_DIR}/scripts/inference.py"
DATA_LOADING_CONFIG="${PROJECT_DIR}/config/data_loading_config.yml"
DEFAULT_CONFIG="${PROJECT_DIR}/config/default_config.yaml"
PREPROCESSED_ROOT="${ROOT_DIR}/Data/NTU4DRadLM_Pre"
RAW_ROOT="${ROOT_DIR}/Data/NTU4DRadLM_Raw"

INFER_DEFAULTS=$(python - "${DEFAULT_CONFIG}" <<'PY'
import sys
import yaml

cfg_path = sys.argv[1]
defaults = {
    'max_infer_files': 0,
    'empty_fallback_topk': 0,
}

try:
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    infer = cfg.get('inference') or {}
    max_files = int(infer.get('max_infer_files', 0) or 0)
    topk = int(infer.get('empty_fallback_topk', 0) or 0)
except Exception:
    max_files = defaults['max_infer_files']
    topk = defaults['empty_fallback_topk']

print(max_files)
print(topk)
PY
)

DEFAULT_MAX_INFER_FILES=$(echo "${INFER_DEFAULTS}" | sed -n '1p')
DEFAULT_EMPTY_FALLBACK_TOPK=$(echo "${INFER_DEFAULTS}" | sed -n '2p')

MAX_INFER_FILES="${MAX_INFER_FILES:-${DEFAULT_MAX_INFER_FILES}}"
EMPTY_FALLBACK_TOPK="${EMPTY_FALLBACK_TOPK:-${DEFAULT_EMPTY_FALLBACK_TOPK}}"

echo "=========================================="
echo "4D Radar 推理"
echo "=========================================="
echo "default config: ${DEFAULT_CONFIG}"
echo "max files per scene: ${MAX_INFER_FILES} (0 means all)"
echo "empty fallback top-k: ${EMPTY_FALLBACK_TOPK} (0 means disabled)"

# 检查模型是否存在
VAE_CKPT="${ROOT_DIR}/Result/train_results/vae/vae_best.pt"
LDM_CKPT="${ROOT_DIR}/Result/train_results/ldm/ldm_best.pt"
CD_CKPT="${ROOT_DIR}/Result/train_results/cd/cd_best.pt"

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

if [ ! -f "$VAE_CKPT" ]; then
    echo "错误: VAE 模型不存在: $VAE_CKPT"
    echo "请先运行: bash launch/train_unified.sh vae"
    exit 1
fi

if [ ! -f "$LDM_CKPT" ]; then
    echo "警告: LDM 模型不存在: $LDM_CKPT"
    echo "跳过 LDM 推理..."
    RUN_LDM=false
else
    RUN_LDM=true
fi

if [ ! -f "$CD_CKPT" ]; then
    echo "警告: CD 模型不存在: $CD_CKPT"
    echo "跳过 CD 推理..."
    RUN_CD=false
else
    RUN_CD=true
fi

# LDM 推理
if [ "$RUN_LDM" = true ]; then
    echo ""
    echo "=========================================="
    echo "1. LDM 推理 (40步 Heun 采样)"
    echo "=========================================="
    
    for SCENE in "${TEST_SCENES[@]}"; do
        RADAR_VOXEL_DIR="${PREPROCESSED_ROOT}/${SCENE}/radar_voxel"
        RAW_LIVOX_DIR="${RAW_ROOT}/${SCENE}/livox_lidar"
        LIDAR_INDEX_FILE="${RAW_ROOT}/${SCENE}/lidar_index_sequence.txt"
        LDM_OUTPUT_DIR="${ROOT_DIR}/Result/inference_results/${SCENE}_ldm_eval"

        echo "  - 场景: ${SCENE}"
        python "${INFER_SCRIPT}" \
            --vae_ckpt "$VAE_CKPT" \
            --model_ckpt "$LDM_CKPT" \
            --model_type ldm \
            --steps 40 \
            --sampler heun \
            --radar_voxel_dir "${RADAR_VOXEL_DIR}" \
            --max_files "${MAX_INFER_FILES}" \
            --empty_fallback_topk "${EMPTY_FALLBACK_TOPK}" \
            --save_pointcloud \
            --compare_with_lidar \
            --raw_livox_dir "${RAW_LIVOX_DIR}" \
            --lidar_index_file "${LIDAR_INDEX_FILE}" \
            --output_dir "${LDM_OUTPUT_DIR}" \
            --device cuda
    done
    
    echo "✓ LDM 推理完成"
fi

# CD 推理
if [ "$RUN_CD" = true ]; then
    echo ""
    echo "=========================================="
    echo "2. CD 推理 (2步快速生成)"
    echo "=========================================="
    
    for SCENE in "${TEST_SCENES[@]}"; do
        RADAR_VOXEL_DIR="${PREPROCESSED_ROOT}/${SCENE}/radar_voxel"
        RAW_LIVOX_DIR="${RAW_ROOT}/${SCENE}/livox_lidar"
        LIDAR_INDEX_FILE="${RAW_ROOT}/${SCENE}/lidar_index_sequence.txt"
        CD_OUTPUT_DIR="${ROOT_DIR}/Result/inference_results/${SCENE}_cd_2step_eval"

        echo "  - 场景: ${SCENE}"
        python "${INFER_SCRIPT}" \
            --vae_ckpt "$VAE_CKPT" \
            --model_ckpt "$CD_CKPT" \
            --model_type cd \
            --steps 1 \
            --sampler euler \
            --radar_voxel_dir "${RADAR_VOXEL_DIR}" \
            --max_files "${MAX_INFER_FILES}" \
            --empty_fallback_topk "${EMPTY_FALLBACK_TOPK}" \
            --save_pointcloud \
            --compare_with_lidar \
            --raw_livox_dir "${RAW_LIVOX_DIR}" \
            --lidar_index_file "${LIDAR_INDEX_FILE}" \
            --output_dir "${CD_OUTPUT_DIR}" \
            --device cuda
    done
    
    echo "✓ CD 2步推理完成"
    
    # CD 4步推理（提升质量）
    echo ""
    echo "=========================================="
    echo "3. CD 推理 (4步高质量生成)"
    echo "=========================================="
    
    for SCENE in "${TEST_SCENES[@]}"; do
        RADAR_VOXEL_DIR="${PREPROCESSED_ROOT}/${SCENE}/radar_voxel"
        RAW_LIVOX_DIR="${RAW_ROOT}/${SCENE}/livox_lidar"
        LIDAR_INDEX_FILE="${RAW_ROOT}/${SCENE}/lidar_index_sequence.txt"
        CD4_OUTPUT_DIR="${ROOT_DIR}/Result/inference_results/${SCENE}_cd_4step_eval"

        echo "  - 场景: ${SCENE}"
        python "${INFER_SCRIPT}" \
            --vae_ckpt "$VAE_CKPT" \
            --model_ckpt "$CD_CKPT" \
            --model_type cd \
            --steps 4 \
            --sampler euler \
            --radar_voxel_dir "${RADAR_VOXEL_DIR}" \
            --max_files "${MAX_INFER_FILES}" \
            --empty_fallback_topk "${EMPTY_FALLBACK_TOPK}" \
            --save_pointcloud \
            --compare_with_lidar \
            --raw_livox_dir "${RAW_LIVOX_DIR}" \
            --lidar_index_file "${LIDAR_INDEX_FILE}" \
            --output_dir "${CD4_OUTPUT_DIR}" \
            --device cuda
    done
    
    echo "✓ CD 4步推理完成"
fi

echo ""
echo "=========================================="
echo "推理完成！"
echo "=========================================="
echo "test 场景列表: ${TEST_SCENES[*]}"
echo "输入根目录: ${PREPROCESSED_ROOT}"
echo "输出根目录: ${ROOT_DIR}/Result/inference_results"
echo "每个输出目录包含: *_pcl.npy + comparison_metrics.csv"
echo ""