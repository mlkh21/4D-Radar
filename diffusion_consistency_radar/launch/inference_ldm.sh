#!/bin/bash
# LDM 正式部署生成脚本：只读取 sensor-aware Radar+IR，不读取离线真值

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"

INFER_SCRIPT="${PROJECT_DIR}/scripts/inference.py"
DEPLOYMENT_VIEW_SCRIPT="${PROJECT_DIR}/scripts/build_deployment_view.py"
CHECKPOINT_CHAIN_SCRIPT="${PROJECT_DIR}/scripts/diagnose_checkpoint_chain.py"
PROTOCOL_TAG="${PROTOCOL_TAG:-formal_v2_80m_86p8_v1}"
RESULTS_DIR="${ROOT_DIR}/Result/train_results/${PROTOCOL_TAG}"
VAE_CKPT="${RESULTS_DIR}/vae/vae_best.pt"
LDM_CKPT="${RESULTS_DIR}/ldm/ldm_best.pt"
THRESHOLD_ARTIFACT="${LDM_THRESHOLD_ARTIFACT:-${RESULTS_DIR}/ldm/occupancy_threshold.json}"
DATA_LOADING_CONFIG="${PROJECT_DIR}/config/data_loading_config.yml"
PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-${ROOT_DIR}/Data/NTU4DRadLM_Deploy_formal_v2_80m_86p8_v1}"
CALIBRATION_DIR="${CALIBRATION_DIR:-${ROOT_DIR}/Data/config}"
TRAIN_DURATION_SECONDS="${TRAIN_DURATION_SECONDS:--1}"
INFERENCE_SEED="${INFERENCE_SEED:-42}"

if [ ! -f "${VAE_CKPT}" ]; then
    echo "错误: VAE 模型不存在: ${VAE_CKPT}"
    exit 1
fi

if [ ! -f "${LDM_CKPT}" ]; then
    echo "错误: LDM 模型不存在: ${LDM_CKPT}"
    exit 1
fi
if [ ! -f "${THRESHOLD_ARTIFACT}" ]; then
    echo "错误: LDM validation threshold artifact 不存在: ${THRESHOLD_ARTIFACT}"
    exit 1
fi

echo "校验正式 VAE/LDM checkpoint 父链"
python "${CHECKPOINT_CHAIN_SCRIPT}" validate \
    --target_stage ldm \
    --vae_ckpt "${VAE_CKPT}" \
    --ldm_ckpt "${LDM_CKPT}"

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

# 根收据、精确场景集合、deployment v3、父 training manifest 与逐帧内容
# 必须一次性全部通过，之后才允许产生任一正式推理结果。
DEPLOYMENT_VALIDATE_ARGS=(validate --dataset_dir "${PREPROCESSED_ROOT}")
for SCENE in "${TEST_SCENES[@]}"; do
    DEPLOYMENT_VALIDATE_ARGS+=(--scene "${SCENE}")
done
python "${DEPLOYMENT_VIEW_SCRIPT}" "${DEPLOYMENT_VALIDATE_ARGS[@]}"

for SCENE in "${TEST_SCENES[@]}"; do
    RADAR_VOXEL_DIR="${PREPROCESSED_ROOT}/${SCENE}/radar_voxel"
    OUTPUT_DIR="${ROOT_DIR}/Result/inference_results/${SCENE}_${PROTOCOL_TAG}_ldm_deploy"

    if [ ! -d "${RADAR_VOXEL_DIR}" ]; then
        echo "错误: radar_voxel 目录不存在: ${RADAR_VOXEL_DIR}"
        exit 1
    fi

    echo "开始 LDM 正式部署生成场景: ${SCENE}"
    python "${INFER_SCRIPT}" \
        --vae_ckpt "${VAE_CKPT}" \
        --model_ckpt "${LDM_CKPT}" \
        --model_type ldm \
        --steps 40 \
        --sampler heun \
        --train_duration_seconds "${TRAIN_DURATION_SECONDS}" \
        --threshold_artifact "${THRESHOLD_ARTIFACT}" \
        --seed "${INFERENCE_SEED}" \
        --radar_voxel_dir "${RADAR_VOXEL_DIR}" \
        --deployment_scene_dir "${PREPROCESSED_ROOT}/${SCENE}" \
        --calibration_dir "${CALIBRATION_DIR}" \
        --require_real_ir \
        --save_voxel \
        --save_pointcloud \
        --save_uncertainty \
        --output_dir "${OUTPUT_DIR}" \
        --device cuda

    echo "完成: ${OUTPUT_DIR}"
done
