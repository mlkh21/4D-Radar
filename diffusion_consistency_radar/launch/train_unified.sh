#!/bin/bash
# ==============================================================================
# 统一训练脚本 - VAE / LDM / CD 一站式训练
# ==============================================================================
#
# 使用方法:
#   bash diffusion_consistency_radar/launch/train_unified.sh vae    # 训练 VAE
#   bash diffusion_consistency_radar/launch/train_unified.sh ldm    # 训练 LDM
#   bash diffusion_consistency_radar/launch/train_unified.sh cd     # 蒸馏 CD
#   bash diffusion_consistency_radar/launch/train_unified.sh all    # 完整流程
#
# 配置文件: config/default_config.yaml
#
# ==============================================================================

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 默认路径
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SELF_DIR}/.." && pwd)"
ROOT_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"
SCRIPT_DIR="${PROJECT_DIR}/scripts"
MANIFEST_SCRIPT="${PROJECT_DIR}/scripts/dataset_manifest.py"
DEFAULT_CONFIG_PATH="${PROJECT_DIR}/config/default_config.yaml"
DATA_LOADING_CONFIG="${PROJECT_DIR}/config/data_loading_config.yml"
PROTOCOL_TAG="formal_p1_04_full120_86p8_v1"
PREPROCESSED_ROOT="${ROOT_DIR}/Data/NTU4DRadLM_Pre_sensor_aware_p1_04_candidate"
RADAR_NORMALIZATION_ARTIFACT="${PROJECT_DIR}/config/radar_normalization_garden_32x128x128_full120_86p8_v1.json"
EXPECTED_ARTIFACT_SHA256="2c9c92650b98ec686d621b53eccb5e7f376cb6b8ea1047d4fb594349af90c4d5"
TRAIN_DATASET_DIR="${PROJECT_DIR}/.tmp_train_dataset"
CONFIG_PATH="${PROJECT_DIR}/config/.default_config.train_override.yaml"
RESULTS_DIR="${ROOT_DIR}/Result/train_results/${PROTOCOL_TAG}"
ALLOW_RESUME="${ALLOW_RESUME:-0}"
MODE="${1:-vae}"

case "${ALLOW_RESUME}" in
    0|1) ;;
    *)
        echo "错误：ALLOW_RESUME 只能为 0 或 1"
        exit 2
        ;;
esac

case "${MODE}" in
    vae|ldm|cd|all) ;;
    *)
        echo "Usage: $0 [vae|ldm|cd|all]"
        exit 2
        ;;
esac

stage_output_has_entries() {
    local stage_dir="$1"
    [ -d "${stage_dir}" ] \
        && [ -n "$(find "${stage_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]
}

guard_stage_output() {
    local stage="$1"
    local checkpoint="$2"
    local stage_dir="${RESULTS_DIR}/${stage}"

    if ! stage_output_has_entries "${stage_dir}"; then
        return
    fi
    if [ "${ALLOW_RESUME}" != "1" ]; then
        echo "错误：${stage_dir} 已包含结果，拒绝隐式续训或覆盖。"
        echo "确认恢复当前协议 checkpoint 后，显式设置 ALLOW_RESUME=1。"
        exit 1
    fi
    if [ ! -f "${checkpoint}" ]; then
        echo "错误：${stage_dir} 非空但缺少恢复 checkpoint：${checkpoint}"
        exit 1
    fi
}

case "${MODE}" in
    vae)
        guard_stage_output vae "${RESULTS_DIR}/vae/vae_best.pt"
        ;;
    ldm)
        guard_stage_output ldm "${RESULTS_DIR}/ldm/ldm_best.pt"
        ;;
    cd)
        guard_stage_output cd "${RESULTS_DIR}/cd/cd_best.pt"
        ;;
    all)
        guard_stage_output vae "${RESULTS_DIR}/vae/vae_best.pt"
        guard_stage_output ldm "${RESULTS_DIR}/ldm/ldm_best.pt"
        guard_stage_output cd "${RESULTS_DIR}/cd/cd_best.pt"
        ;;
esac

if [ ! -f "${DATA_LOADING_CONFIG}" ]; then
    echo "Error: data loading config not found: ${DATA_LOADING_CONFIG}"
    exit 1
fi

mapfile -t TRAIN_SCENES < <(python - "${DATA_LOADING_CONFIG}" <<'PY'
import sys
import yaml

with open(sys.argv[1], 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f) or {}

scenes = (cfg.get('data') or {}).get('train') or []
if isinstance(scenes, str):
    scenes = [scenes]

for scene in scenes:
    s = str(scene).strip()
    if s:
        print(s)
PY
)

if [ ${#TRAIN_SCENES[@]} -eq 0 ]; then
    echo "Error: data_loading_config.yml data.train is empty"
    exit 1
fi

# 在改动临时训练目录前，先验证所有正式训练场景的内容协议。
for SCENE in "${TRAIN_SCENES[@]}"; do
    SRC_SCENE_DIR="${PREPROCESSED_ROOT}/${SCENE}"
    if [ ! -d "${SRC_SCENE_DIR}" ]; then
        echo "Error: train scene directory not found: ${SRC_SCENE_DIR}"
        exit 1
    fi
    if ! python "${MANIFEST_SCRIPT}" validate \
        --scene_dir "${SRC_SCENE_DIR}" \
        --expected_scene "${SCENE}"; then
        echo "Error: dataset manifest validation failed: ${SRC_SCENE_DIR}"
        exit 1
    fi
done

if [ ! -f "${RADAR_NORMALIZATION_ARTIFACT}" ]; then
    echo "错误：Radar normalization artifact 不存在：${RADAR_NORMALIZATION_ARTIFACT}"
    exit 1
fi

# 在改动训练临时目录前校验正式 artifact 的网格、量程和固定文件身份。
python - "${ROOT_DIR}" "${RADAR_NORMALIZATION_ARTIFACT}" "${EXPECTED_ARTIFACT_SHA256}" <<'PY'
import sys

root_dir, artifact_path, expected_sha256 = sys.argv[1:4]
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from diffusion_consistency_radar.radar_normalization import (
    load_radar_normalization_artifact,
)

spec, digest = load_radar_normalization_artifact(
    artifact_path,
    target_size=(32, 128, 128),
    source_pc_range=(0, -20, -6, 120, 20, 10),
    model_pc_range=(0, -20, -6, 120, 20, 10),
    doppler_scale_mps=86.8,
    require_formal=True,
)
if digest != expected_sha256:
    raise RuntimeError(
        f"Radar normalization artifact SHA-256 不匹配: {digest} != {expected_sha256}"
    )
if spec.get("training_scenes") != ["garden"] or spec.get("frame_count") != 4013:
    raise RuntimeError("Radar normalization artifact 训练场景或帧数不匹配")
print(f"Radar normalization artifact 校验通过: {digest}")
PY

# 生成本次正式协议的绝对路径配置，避免启动位置改变相对路径语义。
python - \
    "${DEFAULT_CONFIG_PATH}" \
    "${CONFIG_PATH}" \
    "${TRAIN_DATASET_DIR}" \
    "${RADAR_NORMALIZATION_ARTIFACT}" \
    "${RESULTS_DIR}" <<'PY'
import os
import sys
import yaml

src_cfg, dst_cfg, dataset_dir, artifact_path, results_dir = sys.argv[1:6]
with open(src_cfg, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

cfg.setdefault('data', {})
cfg['data']['dataset_dir'] = dataset_dir
cfg['data']['target_size'] = [32, 128, 128]
cfg['data']['source_pc_range'] = [0, -20, -6, 120, 20, 10]
cfg['data']['model_pc_range'] = [0, -20, -6, 120, 20, 10]
cfg['data']['radar_normalization_path'] = artifact_path
cfg['data']['doppler_scale_mps'] = 86.8
for stage in ('vae', 'ldm', 'cd'):
    cfg.setdefault(stage, {})
    cfg[stage]['save_dir'] = os.path.join(results_dir, stage)

temp_path = f"{dst_cfg}.tmp-{os.getpid()}"
try:
    with open(temp_path, 'x', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    os.replace(temp_path, dst_cfg)
finally:
    if os.path.exists(temp_path):
        os.remove(temp_path)
PY

rm -rf "${TRAIN_DATASET_DIR}"
mkdir -p "${TRAIN_DATASET_DIR}"

for SCENE in "${TRAIN_SCENES[@]}"; do
    SRC_SCENE_DIR="${PREPROCESSED_ROOT}/${SCENE}"
    ln -s "${SRC_SCENE_DIR}" "${TRAIN_DATASET_DIR}/${SCENE}"
done

echo "Using train scenes: ${TRAIN_SCENES[*]}"
echo "Training dataset root: ${TRAIN_DATASET_DIR}"
echo "Training config: ${CONFIG_PATH}"
echo "Training protocol: ${PROTOCOL_TAG}"
echo "Training results: ${RESULTS_DIR}"

case "$MODE" in
    vae)
        echo "=========================================="
        echo "Stage 1: Training VAE"
        echo "=========================================="
        
        VAE_RESUME="${RESULTS_DIR}/vae/vae_best.pt"
        RESUME_ARGS=()
        if [ "${ALLOW_RESUME}" = "1" ] && [ -f "${VAE_RESUME}" ]; then
            RESUME_ARGS+=(--resume "${VAE_RESUME}")
        fi
        CUDA_VISIBLE_DEVICES=0,1 python "${SCRIPT_DIR}/unified_train.py" \
            --mode vae \
            --config "${CONFIG_PATH}" \
            "${RESUME_ARGS[@]}"
        ;;
        
    ldm)
        echo "=========================================="
        echo "Stage 2: Training LDM"
        echo "=========================================="
        
        VAE_CKPT="${RESULTS_DIR}/vae/vae_best.pt"
        if [ ! -f "$VAE_CKPT" ]; then
            echo "Error: VAE checkpoint not found at $VAE_CKPT"
            echo "Please train VAE first: bash $0 vae"
            exit 1
        fi
        
        LDM_RESUME="${RESULTS_DIR}/ldm/ldm_best.pt"
        RESUME_ARGS=()
        if [ "${ALLOW_RESUME}" = "1" ] && [ -f "${LDM_RESUME}" ]; then
            RESUME_ARGS+=(--resume "${LDM_RESUME}")
        fi
        CUDA_VISIBLE_DEVICES=0,1 python "${SCRIPT_DIR}/unified_train.py" \
            --mode ldm \
            --config "${CONFIG_PATH}" \
            --vae_ckpt "${VAE_CKPT}" \
            "${RESUME_ARGS[@]}"
        ;;
        
    cd)
        echo "=========================================="
        echo "Stage 3: Consistency Distillation"
        echo "=========================================="
        
        VAE_CKPT="${RESULTS_DIR}/vae/vae_best.pt"
        LDM_CKPT="${RESULTS_DIR}/ldm/ldm_best.pt"
        
        if [ ! -f "$VAE_CKPT" ]; then
            echo "Error: VAE checkpoint not found at $VAE_CKPT"
            exit 1
        fi
        
        if [ ! -f "$LDM_CKPT" ]; then
            echo "Error: LDM checkpoint not found at $LDM_CKPT"
            exit 1
        fi
        
        CD_RESUME="${RESULTS_DIR}/cd/cd_best.pt"
        RESUME_ARGS=()
        if [ "${ALLOW_RESUME}" = "1" ] && [ -f "${CD_RESUME}" ]; then
            RESUME_ARGS+=(--resume "${CD_RESUME}")
        fi
        CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/unified_train.py" \
            --mode cd \
            --config "${CONFIG_PATH}" \
            --vae_ckpt "${VAE_CKPT}" \
            --ldm_ckpt "${LDM_CKPT}" \
            "${RESUME_ARGS[@]}"
        ;;
        
    all)
        echo "=========================================="
        echo "Running Full Training Pipeline"
        echo "=========================================="
        
        # Stage 1: VAE
        bash "$0" vae
        
        # Stage 2: LDM
        bash "$0" ldm
        
        # Stage 3: CD
        bash "$0" cd
        ;;
        
esac

echo "Done!"
