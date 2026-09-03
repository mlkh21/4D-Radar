#!/usr/bin/env bash
# 文件功能：在服务器安全删除旧 formal-v2 数据，重新生成 formal-v2.1，并依次执行多卡 smoke 与完整训练。

set -Eeuo pipefail

on_error() {
    local status="$?"
    local line="$1"
    echo "失败：第 ${line} 行，退出码 ${status}；后续阶段未执行。" >&2
    exit "${status}"
}
trap 'on_error "${LINENO}"' ERR

ROOT="${PROJECT_ROOT:-/home/ps/zxj_workspace/src/4D-Radar}"
CONDA_ENV="${CONDA_ENV:-Radar}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
RADAR_FIELD_SCHEMA="${RADAR_FIELD_SCHEMA:-${ROOT}/Data/config/radar_field_schema_v2.json}"

CONFIRM_DELETE_OLD_V2="${CONFIRM_DELETE_OLD_V2:-NO}"
CONFIRM_FULL_TRAINING="${CONFIRM_FULL_TRAINING:-NO}"
SMOKE_TRAIN_FRAMES="${SMOKE_TRAIN_FRAMES:-32}"
SMOKE_VALIDATION_FRAMES="${SMOKE_VALIDATION_FRAMES:-16}"
FULL_EPOCHS="${FULL_EPOCHS:-20}"

INPUT_ROOT="${ROOT}/Data/NTU4DRadLM"
OLD_RAW_ROOT="${ROOT}/Data/NTU4DRadLM_Raw_p1_01_candidate"
OLD_PRE_ROOT="${ROOT}/Data/NTU4DRadLM_Pre_formal_v2_80m_86p8_v1"
OLD_DEPLOY_ROOT="${ROOT}/Data/NTU4DRadLM_Deploy_formal_v2_80m_86p8_v1"

NEW_RAW_ROOT="${ROOT}/Data/NTU4DRadLM_Raw_formal_v2_1_80m_86p8_db_snr_v1"
NEW_PRE_ROOT="${ROOT}/Data/NTU4DRadLM_Pre_formal_v2_1_80m_86p8_db_snr_v1"
NEW_DEPLOY_ROOT="${ROOT}/Data/NTU4DRadLM_Deploy_formal_v2_1_80m_86p8_db_snr_v1"
NORMALIZATION_ARTIFACT="${ROOT}/diffusion_consistency_radar/config/radar_normalization_garden_32x128x128_80m_train80_purge3s_86p8_db_snr_v2.json"
TEMPORAL_SPLIT_ARTIFACT="${NEW_PRE_ROOT}/temporal_split_garden_train80_purge3s_v1.json"
DATA_PROTOCOL_ARTIFACT="${NEW_PRE_ROOT}/formal_data_protocol_garden_train80_purge3s_v4.json"

SMOKE_TAG="formal_v2_1_80m_86p8_db_snr_smoke_v1"
FORMAL_TAG="formal_v2_1_80m_86p8_db_snr_v1"
SMOKE_RESULTS="${ROOT}/Result/train_results/${SMOKE_TAG}"
FORMAL_RESULTS="${ROOT}/Result/train_results/${FORMAL_TAG}"
LAUNCHER="${ROOT}/diffusion_consistency_radar/launch/train_unified.sh"
PREPROCESSOR="${ROOT}/NTU4DRadLM_pre_processing/preprocess-v2.sh"

fail() {
    echo "错误：$*" >&2
    exit 1
}

require_file() {
    [[ -f "$1" ]] || fail "缺少文件：$1"
}

require_fresh_path() {
    [[ ! -e "$1" ]] || fail "目标已存在，拒绝覆盖：$1"
}

safe_delete_old_v2() {
    local target="$1"
    case "${target}" in
        "${OLD_RAW_ROOT}"|"${OLD_PRE_ROOT}"|"${OLD_DEPLOY_ROOT}") ;;
        *) fail "删除路径不在旧 v2 白名单：${target}" ;;
    esac
    [[ ! -L "${target}" ]] || fail "旧 v2 路径是符号链接，拒绝递归删除：${target}"
    if [[ -e "${target}" ]]; then
        echo "删除旧 v2 数据：${target}"
        rm -rf --one-file-system -- "${target}"
    else
        echo "旧 v2 路径不存在，跳过：${target}"
    fi
}

[[ "${ROOT}" == "/home/ps/zxj_workspace/src/4D-Radar" || -n "${PROJECT_ROOT:-}" ]] \
    || fail "项目路径不是服务器默认值；请显式设置 PROJECT_ROOT"
[[ "${ROOT}" != "/" && "${ROOT}" != "/home" && "${ROOT}" == */4D-Radar* ]] \
    || fail "PROJECT_ROOT 安全检查失败：${ROOT}"

cd "${ROOT}"
require_file "${PREPROCESSOR}"
require_file "${LAUNCHER}"
require_file "${RADAR_FIELD_SCHEMA}"
[[ -d "${INPUT_ROOT}" ]] || fail "原始 rosbag 数据目录不存在：${INPUT_ROOT}"

for calibration in \
    "${ROOT}/Data/config/calib_radar_to_livox.txt" \
    "${ROOT}/Data/config/calib_radar_to_thermal.txt" \
    "${ROOT}/Data/config/calib_livox_to_thermal.txt" \
    "${ROOT}/Data/config/calib_cam_thermal.txt"; do
    require_file "${calibration}"
done

[[ "${CONFIRM_DELETE_OLD_V2}" == "YES" ]] \
    || fail "必须显式设置 CONFIRM_DELETE_OLD_V2=YES 才允许删除旧 v2 数据"
[[ "${CONFIRM_FULL_TRAINING}" == "YES" ]] \
    || fail "必须显式设置 CONFIRM_FULL_TRAINING=YES 才允许 smoke 通过后启动长训练"

[[ "${CUDA_DEVICES}" =~ ^[0-9]+(,[0-9]+)*$ ]] \
    || fail "CUDA_DEVICES 必须为逗号分隔的 GPU 编号"
IFS=',' read -r -a GPU_IDS <<< "${CUDA_DEVICES}"
[[ "${#GPU_IDS[@]}" -ge 2 && "${#GPU_IDS[@]}" -le 4 ]] \
    || fail "该脚本要求使用 2--4 个 GPU，实际为 ${#GPU_IDS[@]}"
declare -A SEEN_GPU_IDS=()
for gpu_id in "${GPU_IDS[@]}"; do
    [[ -z "${SEEN_GPU_IDS[${gpu_id}]:-}" ]] \
        || fail "CUDA_DEVICES 含重复编号：${gpu_id}"
    SEEN_GPU_IDS["${gpu_id}"]=1
done
[[ "${SMOKE_TRAIN_FRAMES}" =~ ^[1-9][0-9]*$ ]] \
    || fail "SMOKE_TRAIN_FRAMES 必须为正整数"
[[ "${SMOKE_VALIDATION_FRAMES}" =~ ^[1-9][0-9]*$ ]] \
    || fail "SMOKE_VALIDATION_FRAMES 必须为正整数"
[[ "${FULL_EPOCHS}" =~ ^[1-9][0-9]*$ ]] \
    || fail "FULL_EPOCHS 必须为正整数"

for gpu_id in "${GPU_IDS[@]}"; do
    nvidia-smi -i "${gpu_id}" --query-gpu=name,memory.total --format=csv,noheader >/dev/null \
        || fail "GPU ${gpu_id} 不可用"
done

# 先验证 Conda、代码导入和权威 schema；这些检查必须早于删除旧数据。
conda run --no-capture-output -n "${CONDA_ENV}" python - \
    "${ROOT}" "${RADAR_FIELD_SCHEMA}" <<'PY'
import sys

root, schema_path = sys.argv[1:3]
sys.path.insert(0, root)
from diffusion_consistency_radar.radar_field_schema import (
    load_radar_field_schema_artifact,
)

schema, digest = load_radar_field_schema_artifact(
    schema_path,
    require_verified=True,
)
if schema.get("protocol") != "radar_raw_field_semantics_v2":
    raise RuntimeError("formal-v2.1 必须使用 radar_raw_field_semantics_v2")
print(f"Radar field schema 校验通过：{digest}")
PY

# 新数据和两组训练结果都必须是 fresh，避免覆盖成功产物。
for fresh_path in \
    "${NEW_RAW_ROOT}" \
    "${NEW_PRE_ROOT}" \
    "${NEW_DEPLOY_ROOT}" \
    "${NORMALIZATION_ARTIFACT}" \
    "${SMOKE_RESULTS}" \
    "${FORMAL_RESULTS}"; do
    require_fresh_path "${fresh_path}"
done

if pgrep -af 'unified_train.py|cd_train_optimized.py|unpack_rosbag.py|NTU4DRadLM_pre_processing.py' >/dev/null; then
    pgrep -af 'unified_train.py|cd_train_optimized.py|unpack_rosbag.py|NTU4DRadLM_pre_processing.py' || true
    fail "检测到训练或预处理进程，请停止后再执行"
fi

echo "即将删除的旧 v2 数据（不会删除 Data/NTU4DRadLM、Result 或 checkpoint）："
du -sh "${OLD_RAW_ROOT}" "${OLD_PRE_ROOT}" "${OLD_DEPLOY_ROOT}" 2>/dev/null || true
safe_delete_old_v2 "${OLD_RAW_ROOT}"
safe_delete_old_v2 "${OLD_PRE_ROOT}"
safe_delete_old_v2 "${OLD_DEPLOY_ROOT}"

echo "开始生成 formal-v2.1 数据。"
export CONDA_ENV
export INPUT_ROOT
export FORMAL_V2_RAW_ROOT="${NEW_RAW_ROOT}"
export FORMAL_V2_PREPROCESSED_ROOT="${NEW_PRE_ROOT}"
export FORMAL_V2_DEPLOY_ROOT="${NEW_DEPLOY_ROOT}"
export FORMAL_V2_NORMALIZATION_ARTIFACT="${NORMALIZATION_ARTIFACT}"
export RADAR_FIELD_SCHEMA
export DOPPLER_SCALE_MPS=86.8
bash "${PREPROCESSOR}"

NORMALIZATION_SHA256="$(sha256sum "${NORMALIZATION_ARTIFACT}" | awk '{print $1}')"
[[ "${NORMALIZATION_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
    || fail "无法计算 normalization SHA-256"
echo "v2.1 normalization SHA-256：${NORMALIZATION_SHA256}"

# 这些变量必须 export；train_unified.sh all 的三个子阶段才能保持同一 v2.1 身份。
export PREPROCESSED_ROOT="${NEW_PRE_ROOT}"
export RADAR_NORMALIZATION_ARTIFACT="${NORMALIZATION_ARTIFACT}"
export EXPECTED_ARTIFACT_SHA256="${NORMALIZATION_SHA256}"
export TEMPORAL_SPLIT_ARTIFACT
export DATA_PROTOCOL_ARTIFACT
export CALIBRATION_DIR="${ROOT}/Data/config"
export CUDA_DEVICES

# 防止调用者遗留的阶段专用变量覆盖下面的 smoke/full 公共参数。
unset VAE_EPOCHS LDM_EPOCHS CD_EPOCHS
unset VAE_TRAIN_FRAMES_PER_EPOCH LDM_TRAIN_FRAMES_PER_EPOCH CD_TRAIN_FRAMES_PER_EPOCH
unset VAE_VALIDATION_FRAMES_PER_EPOCH LDM_VALIDATION_FRAMES_PER_EPOCH CD_VALIDATION_FRAMES_PER_EPOCH

echo "执行 v2.1 正式只读 preflight。"
PROTOCOL_TAG="${FORMAL_TAG}" \
PREFLIGHT_ONLY=1 \
ALLOW_RESUME=0 \
FORMAL_EPOCHS="${FULL_EPOCHS}" \
FORMAL_TRAIN_FRAMES_PER_EPOCH=0 \
FORMAL_VALIDATION_FRAMES_PER_EPOCH=0 \
conda run --no-capture-output -n "${CONDA_ENV}" bash "${LAUNCHER}" all

echo "执行独立 1-epoch、${SMOKE_TRAIN_FRAMES}/${SMOKE_VALIDATION_FRAMES} 帧多卡 smoke。"
PROTOCOL_TAG="${SMOKE_TAG}" \
PREFLIGHT_ONLY=0 \
ALLOW_RESUME=0 \
FORMAL_EPOCHS=1 \
FORMAL_TRAIN_FRAMES_PER_EPOCH="${SMOKE_TRAIN_FRAMES}" \
FORMAL_VALIDATION_FRAMES_PER_EPOCH="${SMOKE_VALIDATION_FRAMES}" \
conda run --no-capture-output -n "${CONDA_ENV}" bash "${LAUNCHER}" all

echo "smoke 全部阶段通过，开始 fresh 全量训练：VAE/LDM/CD 各 ${FULL_EPOCHS} epoch。"
PROTOCOL_TAG="${FORMAL_TAG}" \
PREFLIGHT_ONLY=0 \
ALLOW_RESUME=0 \
FORMAL_EPOCHS="${FULL_EPOCHS}" \
FORMAL_TRAIN_FRAMES_PER_EPOCH=0 \
FORMAL_VALIDATION_FRAMES_PER_EPOCH=0 \
conda run --no-capture-output -n "${CONDA_ENV}" bash "${LAUNCHER}" all

echo "全部完成。"
echo "v2.1 training 数据：${NEW_PRE_ROOT}"
echo "v2.1 deployment 数据：${NEW_DEPLOY_ROOT}"
echo "smoke 结果：${SMOKE_RESULTS}"
echo "正式训练结果：${FORMAL_RESULTS}"

