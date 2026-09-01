#!/bin/bash
# 统一正式部署生成入口：LDM/CD 均只读取 sensor-aware Radar+IR

set -euo pipefail  # 遇到错误立即退出

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"
INFER_SCRIPT="${PROJECT_DIR}/scripts/inference.py"
DEPLOYMENT_VIEW_SCRIPT="${PROJECT_DIR}/scripts/build_deployment_view.py"
CHECKPOINT_CHAIN_SCRIPT="${PROJECT_DIR}/scripts/diagnose_checkpoint_chain.py"
DATA_LOADING_CONFIG="${PROJECT_DIR}/config/data_loading_config.yml"
DEFAULT_CONFIG="${PROJECT_DIR}/config/default_config.yaml"
PROTOCOL_TAG="${PROTOCOL_TAG:-formal_v2_80m_86p8_v1}"
RESULTS_DIR="${ROOT_DIR}/Result/train_results/${PROTOCOL_TAG}"
PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-${ROOT_DIR}/Data/NTU4DRadLM_Deploy_formal_v2_80m_86p8_v1}"
CALIBRATION_DIR="${CALIBRATION_DIR:-${ROOT_DIR}/Data/config}"

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
INFERENCE_SEED="${INFERENCE_SEED:-42}"

echo "=========================================="
echo "4D Radar 推理"
echo "=========================================="
echo "default config: ${DEFAULT_CONFIG}"
echo "max files per scene: ${MAX_INFER_FILES} (0 means all)"
echo "empty fallback top-k: ${EMPTY_FALLBACK_TOPK} (0 means disabled)"
echo "occ threshold: validation artifact"

# 检查模型是否存在
VAE_CKPT="${RESULTS_DIR}/vae/vae_best.pt"
LDM_CKPT="${RESULTS_DIR}/ldm/ldm_best.pt"
CD_CKPT="${RESULTS_DIR}/cd/cd_best.pt"
LDM_THRESHOLD_ARTIFACT="${LDM_THRESHOLD_ARTIFACT:-${RESULTS_DIR}/ldm/occupancy_threshold.json}"
CD_THRESHOLD_ARTIFACT="${CD_THRESHOLD_ARTIFACT:-${RESULTS_DIR}/cd/occupancy_threshold.json}"

for ARTIFACT in "${LDM_THRESHOLD_ARTIFACT}" "${CD_THRESHOLD_ARTIFACT}"; do
    if [ ! -f "${ARTIFACT}" ]; then
        echo "错误: validation threshold artifact 不存在: ${ARTIFACT}"
        exit 1
    fi
done

echo "校验正式 VAE/LDM/CD checkpoint 链"
python "${CHECKPOINT_CHAIN_SCRIPT}" validate \
    --target_stage cd \
    --vae_ckpt "${VAE_CKPT}" \
    --ldm_ckpt "${LDM_CKPT}" \
    --cd_ckpt "${CD_CKPT}"

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

# 统一入口只校验一次完整 deployment dataset，后续各采样分支复用。
DEPLOYMENT_VALIDATE_ARGS=(validate --dataset_dir "${PREPROCESSED_ROOT}")
for SCENE in "${TEST_SCENES[@]}"; do
    DEPLOYMENT_VALIDATE_ARGS+=(--scene "${SCENE}")
done
python "${DEPLOYMENT_VIEW_SCRIPT}" "${DEPLOYMENT_VALIDATE_ARGS[@]}"

RUN_LDM=true
RUN_CD=true

# LDM 推理
if [ "$RUN_LDM" = true ]; then
    echo ""
    echo "=========================================="
    echo "1. LDM 推理 (40步 Heun 采样)"
    echo "=========================================="
    
    for SCENE in "${TEST_SCENES[@]}"; do
        RADAR_VOXEL_DIR="${PREPROCESSED_ROOT}/${SCENE}/radar_voxel"
        LDM_OUTPUT_DIR="${ROOT_DIR}/Result/inference_results/${SCENE}_${PROTOCOL_TAG}_ldm_deploy"

        echo "  - 场景: ${SCENE}"
        python "${INFER_SCRIPT}" \
            --vae_ckpt "$VAE_CKPT" \
            --model_ckpt "$LDM_CKPT" \
            --model_type ldm \
            --steps 40 \
            --sampler heun \
            --radar_voxel_dir "${RADAR_VOXEL_DIR}" \
            --deployment_scene_dir "${PREPROCESSED_ROOT}/${SCENE}" \
            --calibration_dir "${CALIBRATION_DIR}" \
            --max_files "${MAX_INFER_FILES}" \
            --threshold_artifact "${LDM_THRESHOLD_ARTIFACT}" \
            --seed "${INFERENCE_SEED}" \
            --empty_fallback_topk "${EMPTY_FALLBACK_TOPK}" \
            --require_real_ir \
            --save_voxel \
            --save_pointcloud \
            --save_uncertainty \
            --output_dir "${LDM_OUTPUT_DIR}" \
            --device cuda
    done
    
    echo "✓ LDM 推理完成"
fi

# CD 推理
if [ "$RUN_CD" = true ]; then
    echo ""
    echo "=========================================="
    echo "2. CD 推理 (1步快速生成)"
    echo "=========================================="
    
    for SCENE in "${TEST_SCENES[@]}"; do
        RADAR_VOXEL_DIR="${PREPROCESSED_ROOT}/${SCENE}/radar_voxel"
        CD_OUTPUT_DIR="${ROOT_DIR}/Result/inference_results/${SCENE}_${PROTOCOL_TAG}_cd_1step_deploy"

        echo "  - 场景: ${SCENE}"
        python "${INFER_SCRIPT}" \
            --vae_ckpt "$VAE_CKPT" \
            --model_ckpt "$CD_CKPT" \
            --model_type cd \
            --steps 1 \
            --sampler euler \
            --radar_voxel_dir "${RADAR_VOXEL_DIR}" \
            --deployment_scene_dir "${PREPROCESSED_ROOT}/${SCENE}" \
            --calibration_dir "${CALIBRATION_DIR}" \
            --max_files "${MAX_INFER_FILES}" \
            --threshold_artifact "${CD_THRESHOLD_ARTIFACT}" \
            --seed "${INFERENCE_SEED}" \
            --empty_fallback_topk "${EMPTY_FALLBACK_TOPK}" \
            --require_real_ir \
            --save_voxel \
            --save_pointcloud \
            --save_uncertainty \
            --output_dir "${CD_OUTPUT_DIR}" \
            --device cuda
    done
    
    echo "✓ CD 1步推理完成"
    
    # CD 4步推理（提升质量）
    echo ""
    echo "=========================================="
    echo "3. CD 推理 (4步高质量生成)"
    echo "=========================================="
    
    for SCENE in "${TEST_SCENES[@]}"; do
        RADAR_VOXEL_DIR="${PREPROCESSED_ROOT}/${SCENE}/radar_voxel"
        CD4_OUTPUT_DIR="${ROOT_DIR}/Result/inference_results/${SCENE}_${PROTOCOL_TAG}_cd_4step_deploy"

        echo "  - 场景: ${SCENE}"
        python "${INFER_SCRIPT}" \
            --vae_ckpt "$VAE_CKPT" \
            --model_ckpt "$CD_CKPT" \
            --model_type cd \
            --steps 4 \
            --sampler euler \
            --radar_voxel_dir "${RADAR_VOXEL_DIR}" \
            --deployment_scene_dir "${PREPROCESSED_ROOT}/${SCENE}" \
            --calibration_dir "${CALIBRATION_DIR}" \
            --max_files "${MAX_INFER_FILES}" \
            --threshold_artifact "${CD_THRESHOLD_ARTIFACT}" \
            --seed "${INFERENCE_SEED}" \
            --empty_fallback_topk "${EMPTY_FALLBACK_TOPK}" \
            --require_real_ir \
            --save_voxel \
            --save_pointcloud \
            --save_uncertainty \
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
echo "每个输出目录包含: *_pcl.npy + *_voxel.npy + 可用的 *_uncertainty.npy"
echo "运行协议文件: inference_runtime.csv + inference_run.json"
echo ""
