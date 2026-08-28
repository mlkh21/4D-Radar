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
PROTOCOL_TAG="${PROTOCOL_TAG:-formal_v2_80m_86p8_v1}"
PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-${ROOT_DIR}/Data/NTU4DRadLM_Pre_formal_v2_80m_86p8_v1}"
RADAR_NORMALIZATION_ARTIFACT="${RADAR_NORMALIZATION_ARTIFACT:-${PROJECT_DIR}/config/radar_normalization_garden_32x128x128_80m_train80_purge3s_86p8_v2.json}"
EXPECTED_ARTIFACT_SHA256="${EXPECTED_ARTIFACT_SHA256:-}"
TEMPORAL_SPLIT_ARTIFACT="${TEMPORAL_SPLIT_ARTIFACT:-${PREPROCESSED_ROOT}/temporal_split_garden_train80_purge3s_v1.json}"
DATA_PROTOCOL_ARTIFACT="${DATA_PROTOCOL_ARTIFACT:-${PREPROCESSED_ROOT}/formal_data_protocol_garden_train80_purge3s_v1.json}"
CALIBRATION_DIR="${CALIBRATION_DIR:-${ROOT_DIR}/Data/config}"
CONFIG_PATH="${PROJECT_DIR}/config/.default_config.train_override.yaml"
RESULTS_DIR="${ROOT_DIR}/Result/train_results/${PROTOCOL_TAG}"
ALLOW_RESUME="${ALLOW_RESUME:-0}"
CUDA_DEVICES="${CUDA_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
FORMAL_EPOCHS="${FORMAL_EPOCHS:-20}"
MODE="${1:-vae}"

case "${ALLOW_RESUME}" in
    0|1) ;;
    *)
        echo "错误：ALLOW_RESUME 只能为 0 或 1"
        exit 2
        ;;
esac

case "${PREFLIGHT_ONLY}" in
    0|1) ;;
    *)
        echo "错误：PREFLIGHT_ONLY 只能为 0 或 1"
        exit 2
        ;;
esac

if [[ ! "${FORMAL_EPOCHS}" =~ ^[0-9]+$ || "${FORMAL_EPOCHS}" -ne 20 ]]; then
    echo "错误：正式服务器训练固定 VAE/LDM/CD epochs=20/20/20，实际为 ${FORMAL_EPOCHS}"
    exit 2
fi

if [[ ! "${CUDA_DEVICES}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "错误：CUDA_DEVICES 必须是逗号分隔的非负 GPU 编号，例如 0 或 0,1"
    exit 2
fi

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

if [ "${PREFLIGHT_ONLY}" != "1" ]; then
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
fi

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

# 在生成运行配置前，先验证所有正式训练场景的内容协议。
for SCENE in "${TRAIN_SCENES[@]}"; do
    SRC_SCENE_DIR="${PREPROCESSED_ROOT}/${SCENE}"
    if [ ! -d "${SRC_SCENE_DIR}" ]; then
        echo "Error: train scene directory not found: ${SRC_SCENE_DIR}"
        exit 1
    fi
    if ! python "${MANIFEST_SCRIPT}" validate \
        --scene_dir "${SRC_SCENE_DIR}" \
        --expected_scene "${SCENE}" \
        --expected_profile training; then
        echo "Error: dataset manifest validation failed: ${SRC_SCENE_DIR}"
        exit 1
    fi
done

if [ ! -f "${DATA_PROTOCOL_ARTIFACT}" ]; then
    echo "错误：formal v2 data protocol artifact 不存在：${DATA_PROTOCOL_ARTIFACT}"
    echo "需先完成范围决策、observed/split 固化和 v2 数据重建。"
    exit 1
fi
if [ ! -f "${TEMPORAL_SPLIT_ARTIFACT}" ]; then
    echo "错误：formal v2 temporal split artifact 不存在：${TEMPORAL_SPLIT_ARTIFACT}"
    exit 1
fi

if [ ! -f "${RADAR_NORMALIZATION_ARTIFACT}" ]; then
    echo "错误：Radar normalization artifact 不存在：${RADAR_NORMALIZATION_ARTIFACT}"
    exit 1
fi
if [ -z "${EXPECTED_ARTIFACT_SHA256}" ]; then
    echo "错误：必须显式设置 EXPECTED_ARTIFACT_SHA256，禁止按文件名信任 normalization。"
    exit 1
fi

# 在改动训练临时目录前校验正式 artifact 的网格、量程和固定文件身份。
python - "${ROOT_DIR}" "${RADAR_NORMALIZATION_ARTIFACT}" "${EXPECTED_ARTIFACT_SHA256}" "${TEMPORAL_SPLIT_ARTIFACT}" <<'PY'
import sys

root_dir, artifact_path, expected_sha256, split_path = sys.argv[1:5]
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from diffusion_consistency_radar.radar_normalization import (
    load_radar_normalization_artifact,
)
from diffusion_consistency_radar.dataset_manifest import sha256_file

split_sha256 = sha256_file(split_path)

spec, digest = load_radar_normalization_artifact(
    artifact_path,
    target_size=(32, 128, 128),
    source_pc_range=(0, -20, -6, 80, 20, 10),
    model_pc_range=(0, -20, -6, 80, 20, 10),
    doppler_scale_mps=86.8,
    require_formal=True,
    expected_split_artifact_sha256=split_sha256,
)
if digest != expected_sha256:
    raise RuntimeError(
        f"Radar normalization artifact SHA-256 不匹配: {digest} != {expected_sha256}"
    )
if spec.get("training_scenes") != ["garden"] or int(spec.get("frame_count", 0)) <= 0:
    raise RuntimeError("Radar normalization artifact 训练场景或帧数无效")
print(f"Radar normalization artifact 校验通过: {digest}")
PY

# 全量核对 manifest 记录所指向的 Radar statistics，并重建 data protocol 身份。
python - \
    "${ROOT_DIR}" \
    "${PREPROCESSED_ROOT}" \
    "${DATA_PROTOCOL_ARTIFACT}" \
    "${TEMPORAL_SPLIT_ARTIFACT}" \
    "$(IFS=,; echo "${TRAIN_SCENES[*]}")" <<'PY'
import json
import os
import sys

root_dir, dataset_dir, protocol_path, split_path, scenes_csv = sys.argv[1:6]
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from diffusion_consistency_radar.dataset_manifest import validate_scene_manifest
from diffusion_consistency_radar.formal_data_protocol import (
    load_formal_data_protocol_artifact,
)
from diffusion_consistency_radar.radar_statistics import (
    RADAR_STATISTICS_PROTOCOL,
    validate_sparse_radar_statistics,
)

scenes = [scene for scene in scenes_csv.split(',') if scene]
checked_frames = 0
occupied_voxels = 0
point_count = 0
doppler_valid_count = 0
for scene in scenes:
    scene_dir = os.path.join(dataset_dir, scene)
    manifest = validate_scene_manifest(
        scene_dir,
        scene,
        expected_profile="training",
    )
    policy_path = os.path.join(scene_dir, "preprocess_policy.json")
    with open(policy_path, "r", encoding="utf-8") as handle:
        policy = json.load(handle)
    if policy.get("radar_statistics_protocol") != RADAR_STATISTICS_PROTOCOL:
        raise RuntimeError(f"场景 {scene!r} Radar statistics policy 协议不匹配")
    if policy.get("radar_statistics_model_consumed") is not False:
        raise RuntimeError(
            f"场景 {scene!r} 必须声明 radar_statistics_model_consumed=false"
        )
    records = manifest.get("modalities", {}).get("radar_voxel")
    if not isinstance(records, list) or not records:
        raise RuntimeError(f"场景 {scene!r} manifest 缺少 Radar records")
    for record in records:
        relative_path = record.get("path") if isinstance(record, dict) else None
        if not isinstance(relative_path, str) or not relative_path:
            raise RuntimeError(f"场景 {scene!r} Radar manifest path 无效")
        radar_path = os.path.join(scene_dir, relative_path)
        summary = validate_sparse_radar_statistics(radar_path)
        checked_frames += 1
        occupied_voxels += int(summary["occupied_voxels"])
        point_count += int(summary["total_point_count"])
        doppler_valid_count += int(summary["total_doppler_valid_count"])

_protocol, protocol_sha256 = load_formal_data_protocol_artifact(
    protocol_path,
    dataset_dir=dataset_dir,
    scenes=scenes,
    split_artifact_path=split_path,
    stage="vae",
)
print(
    json.dumps(
        {
            "status": "Radar statistics 预检通过",
            "protocol": RADAR_STATISTICS_PROTOCOL,
            "scenes": scenes,
            "checked_frames": checked_frames,
            "occupied_voxels": occupied_voxels,
            "total_point_count": point_count,
            "total_doppler_valid_count": doppler_valid_count,
            "formal_data_protocol_sha256": protocol_sha256,
            "model_consumed": False,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
)
PY

echo "Formal epochs per stage: ${FORMAL_EPOCHS}"
echo "CUDA allocator: ${PYTORCH_CUDA_ALLOC_CONF}"

if [ "${PREFLIGHT_ONLY}" = "1" ]; then
    echo "正式训练预检完成；PREFLIGHT_ONLY=1，未生成训练配置且未启动训练。"
    exit 0
fi

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"

# 生成本次正式协议的绝对路径配置，避免启动位置改变相对路径语义。
python - \
    "${DEFAULT_CONFIG_PATH}" \
    "${CONFIG_PATH}" \
    "${PREPROCESSED_ROOT}" \
    "${RADAR_NORMALIZATION_ARTIFACT}" \
    "${RESULTS_DIR}" \
    "${CALIBRATION_DIR}" \
    "${DATA_PROTOCOL_ARTIFACT}" \
    "${TEMPORAL_SPLIT_ARTIFACT}" \
    "$(IFS=,; echo "${TRAIN_SCENES[*]}")" \
    "${FORMAL_EPOCHS}" <<'PY'
import os
import sys
import yaml

(
    src_cfg,
    dst_cfg,
    dataset_dir,
    artifact_path,
    results_dir,
    calibration_dir,
    data_protocol_path,
    temporal_split_path,
    scenes_csv,
    formal_epochs,
) = sys.argv[1:11]
with open(src_cfg, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
allocator_conf = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '').strip()
if allocator_conf != 'max_split_size_mb:128':
    raise RuntimeError(f"正式训练 CUDA allocator 配置异常: {allocator_conf!r}")
cfg.setdefault('hardware', {})
cfg['hardware']['cuda_allocator_conf'] = allocator_conf
cfg.setdefault('data', {})
cfg['data']['dataset_dir'] = dataset_dir
cfg['data']['scene_names'] = [scene for scene in scenes_csv.split(',') if scene]
cfg['data']['calibration_dir'] = calibration_dir
cfg['data']['require_real_ir'] = True
cfg['data']['require_real_calibration'] = True
cfg['data']['require_persisted_observed_mask'] = True
cfg['data']['require_radar_statistics'] = True
cfg['data']['voxel_coordinate_frame'] = 'lidar'
cfg['data']['checkpoint_protocol'] = 'formal_chain_v2'
# 正式服务器配置必须完整消费 temporal split，禁止继承任何笔记本 mini 截断字段。
cfg['data'].pop('mini_train_frames_per_scene', None)
cfg['data'].pop('mini_validation_frames_per_scene', None)
cfg['data'].pop('data_protocol', None)
cfg['data']['data_protocol_path'] = data_protocol_path
cfg['data']['temporal_split_artifact'] = temporal_split_path
cfg['data']['target_size'] = [32, 128, 128]
cfg['data']['source_pc_range'] = [0, -20, -6, 80, 20, 10]
cfg['data']['model_pc_range'] = [0, -20, -6, 80, 20, 10]
cfg['data']['radar_normalization_path'] = artifact_path
cfg['data']['doppler_scale_mps'] = 86.8
for stage in ('vae', 'ldm', 'cd'):
    cfg.setdefault(stage, {})
    cfg[stage]['epochs'] = int(formal_epochs)
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

echo "Using train scenes: ${TRAIN_SCENES[*]}"
echo "Training dataset root: ${PREPROCESSED_ROOT}"
echo "Calibration directory: ${CALIBRATION_DIR}"
echo "Training config: ${CONFIG_PATH}"
echo "Training protocol: ${PROTOCOL_TAG}"
echo "Training results: ${RESULTS_DIR}"
echo "CUDA devices: ${CUDA_VISIBLE_DEVICES}"

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
        python "${SCRIPT_DIR}/unified_train.py" \
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
        python "${SCRIPT_DIR}/unified_train.py" \
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
        python "${SCRIPT_DIR}/unified_train.py" \
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
