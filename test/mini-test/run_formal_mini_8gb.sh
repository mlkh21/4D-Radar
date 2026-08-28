#!/bin/bash
# 文件功能：在 8 GB 单卡笔记本上分阶段运行正式数据协议 mini 训练，并提供温度、显存和时长门禁。

set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SELF_DIR}/../.." && pwd)"
TRAIN_SCRIPT="${SELF_DIR}/train_minimal.sh"

if [[ "$#" -gt 2 ]]; then
  echo "Usage: $0 [vae|ldm|cd] [smoke|short_train|medium_train]"
  echo "Error: 受保护入口最多接受阶段和 profile 两个位置参数"
  exit 2
fi

MODE="${1:-vae}"
PROFILE="${2:-smoke}"
case "${MODE}" in
  vae|ldm|cd) ;;
  *)
    echo "Usage: $0 [vae|ldm|cd] [smoke|short_train|medium_train]"
    echo "为避免笔记本持续满载，本入口只允许逐阶段运行。"
    exit 2
    ;;
esac

case "${PROFILE}" in
  smoke)
    PROFILE_TRAIN_FRAMES_PER_SCENE=8
    PROFILE_VALIDATION_FRAMES_PER_SCENE=4
    PROFILE_REQUIRE_EXACT_FRAMES=0
    PROFILE_MAX_TRAIN_FRAMES_PER_SCENE=32
    PROFILE_MAX_VALIDATION_FRAMES_PER_SCENE=16
    PROFILE_VAE_EPOCHS=1
    PROFILE_LDM_EPOCHS=1
    PROFILE_CD_EPOCHS=1
    PROFILE_MAX_GPU_TEMP_C=80
    PROFILE_MAX_START_TEMP_C=65
    PROFILE_MAX_STAGE_MINUTES=20
    PROFILE_MIN_FREE_GPU_MEMORY_MIB=6000
    PROFILE_REQUIRED_GPU_NAME=""
    DEFAULT_MINI_RESULTS_DIR="${ROOT_DIR}/test/result/formal_mini_v2_80m_8gb_v1"
    ;;
  short_train)
    if [[ "${MODE}" != "vae" ]]; then
      echo "Error: short_train 目前只允许 VAE；LDM/CD 继续使用 smoke 档逐阶段验证"
      exit 2
    fi
    PROFILE_TRAIN_FRAMES_PER_SCENE=8
    PROFILE_VALIDATION_FRAMES_PER_SCENE=4
    PROFILE_REQUIRE_EXACT_FRAMES=0
    PROFILE_MAX_TRAIN_FRAMES_PER_SCENE=32
    PROFILE_MAX_VALIDATION_FRAMES_PER_SCENE=16
    PROFILE_VAE_EPOCHS=3
    PROFILE_LDM_EPOCHS=1
    PROFILE_CD_EPOCHS=1
    PROFILE_MAX_GPU_TEMP_C=75
    PROFILE_MAX_START_TEMP_C=60
    PROFILE_MAX_STAGE_MINUTES=20
    PROFILE_MIN_FREE_GPU_MEMORY_MIB=6000
    PROFILE_REQUIRED_GPU_NAME=""
    DEFAULT_MINI_RESULTS_DIR="${ROOT_DIR}/test/result/formal_mini_v2_80m_8gb_short_v1"
    ;;
  medium_train)
    # 500 帧由正式时间切分中的 400 train + 100 validation 组成，三阶段各训练 20 epoch。
    PROFILE_TRAIN_FRAMES_PER_SCENE=400
    PROFILE_VALIDATION_FRAMES_PER_SCENE=100
    PROFILE_REQUIRE_EXACT_FRAMES=1
    PROFILE_MAX_TRAIN_FRAMES_PER_SCENE=400
    PROFILE_MAX_VALIDATION_FRAMES_PER_SCENE=100
    PROFILE_VAE_EPOCHS=20
    PROFILE_LDM_EPOCHS=20
    PROFILE_CD_EPOCHS=20
    PROFILE_MAX_GPU_TEMP_C=72
    PROFILE_MAX_START_TEMP_C=55
    PROFILE_MAX_STAGE_MINUTES=180
    PROFILE_MIN_FREE_GPU_MEMORY_MIB=6500
    PROFILE_REQUIRED_GPU_NAME="NVIDIA GeForce RTX 4070 Laptop GPU"
    DEFAULT_MINI_RESULTS_DIR="${ROOT_DIR}/test/result/formal_medium_v2_80m_laptop_500f_20ep_v2"
    ;;
  *)
    echo "Error: profile 必须为 smoke、short_train 或 medium_train，实际为 ${PROFILE}"
    exit 2
    ;;
esac

MINI_TRAIN_FRAMES_PER_SCENE="${MINI_TRAIN_FRAMES_PER_SCENE:-${PROFILE_TRAIN_FRAMES_PER_SCENE}}"
MINI_VALIDATION_FRAMES_PER_SCENE="${MINI_VALIDATION_FRAMES_PER_SCENE:-${PROFILE_VALIDATION_FRAMES_PER_SCENE}}"
MINI_VAE_EPOCHS="${MINI_VAE_EPOCHS:-${PROFILE_VAE_EPOCHS}}"
MINI_LDM_EPOCHS="${MINI_LDM_EPOCHS:-${PROFILE_LDM_EPOCHS}}"
MINI_CD_EPOCHS="${MINI_CD_EPOCHS:-${PROFILE_CD_EPOCHS}}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-1}"
MINI_NUM_WORKERS="${MINI_NUM_WORKERS:-0}"
MINI_GRAD_ACCUM="${MINI_GRAD_ACCUM:-1}"
MINI_MAX_GPU_TEMP_C="${MINI_MAX_GPU_TEMP_C:-${PROFILE_MAX_GPU_TEMP_C}}"
MINI_MAX_START_TEMP_C="${MINI_MAX_START_TEMP_C:-${PROFILE_MAX_START_TEMP_C}}"
MINI_MAX_STAGE_MINUTES="${MINI_MAX_STAGE_MINUTES:-${PROFILE_MAX_STAGE_MINUTES}}"
MINI_MIN_GPU_MEMORY_MIB="${MINI_MIN_GPU_MEMORY_MIB:-7500}"
MINI_MIN_FREE_GPU_MEMORY_MIB="${MINI_MIN_FREE_GPU_MEMORY_MIB:-${PROFILE_MIN_FREE_GPU_MEMORY_MIB}}"
MINI_THERMAL_POLL_SECONDS="${MINI_THERMAL_POLL_SECONDS:-5}"
MINI_STOP_GRACE_SECONDS="${MINI_STOP_GRACE_SECONDS:-5}"
MINI_PREFLIGHT_ONLY="${MINI_PREFLIGHT_ONLY:-0}"

CUDA_DEVICES="${CUDA_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
if [[ ! "${CUDA_DEVICES}" =~ ^[0-9]+$ ]]; then
  echo "Error: 8 GB formal mini 只允许一个物理 GPU 编号，实际为 ${CUDA_DEVICES}"
  exit 2
fi
export CUDA_DEVICES
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"

for setting in \
  MINI_TRAIN_FRAMES_PER_SCENE MINI_VALIDATION_FRAMES_PER_SCENE \
  MINI_VAE_EPOCHS MINI_LDM_EPOCHS MINI_CD_EPOCHS \
  MINI_BATCH_SIZE MINI_GRAD_ACCUM MINI_MAX_GPU_TEMP_C MINI_MAX_START_TEMP_C \
  MINI_MAX_STAGE_MINUTES MINI_MIN_GPU_MEMORY_MIB \
  MINI_MIN_FREE_GPU_MEMORY_MIB MINI_THERMAL_POLL_SECONDS \
  MINI_STOP_GRACE_SECONDS; do
  value="${!setting}"
  if [[ ! "${value}" =~ ^[0-9]+$ || "${value}" -lt 1 ]]; then
    echo "Error: ${setting} 必须为正整数，实际为 ${value}"
    exit 2
  fi
done
if [[ ! "${MINI_NUM_WORKERS}" =~ ^[0-9]+$ ]]; then
  echo "Error: MINI_NUM_WORKERS 必须为非负整数，实际为 ${MINI_NUM_WORKERS}"
  exit 2
fi
if [[ "${MINI_VAE_EPOCHS}" -ne "${PROFILE_VAE_EPOCHS}" ]] ||
  [[ "${MINI_LDM_EPOCHS}" -ne "${PROFILE_LDM_EPOCHS}" ]] ||
  [[ "${MINI_CD_EPOCHS}" -ne "${PROFILE_CD_EPOCHS}" ]]; then
  echo "Error: ${PROFILE} 固定 VAE/LDM/CD epochs=${PROFILE_VAE_EPOCHS}/${PROFILE_LDM_EPOCHS}/${PROFILE_CD_EPOCHS}"
  exit 2
fi
if [[ "${MINI_BATCH_SIZE}" -ne 1 || "${MINI_GRAD_ACCUM}" -ne 1 ]]; then
  echo "Error: 8 GB formal mini 固定 MINI_BATCH_SIZE=1 且 MINI_GRAD_ACCUM=1"
  exit 2
fi
if [[ "${MINI_NUM_WORKERS}" -ne 0 ]]; then
  echo "Error: 8 GB formal mini 固定 MINI_NUM_WORKERS=0，以降低 CPU 持续负载"
  exit 2
fi
if [[ "${PROFILE_REQUIRE_EXACT_FRAMES}" -eq 1 ]]; then
  if [[ "${MINI_TRAIN_FRAMES_PER_SCENE}" -ne "${PROFILE_TRAIN_FRAMES_PER_SCENE}" ]] ||
    [[ "${MINI_VALIDATION_FRAMES_PER_SCENE}" -ne "${PROFILE_VALIDATION_FRAMES_PER_SCENE}" ]]; then
    echo "Error: ${PROFILE} 固定 train/validation frames per scene=${PROFILE_TRAIN_FRAMES_PER_SCENE}/${PROFILE_VALIDATION_FRAMES_PER_SCENE}"
    exit 2
  fi
elif [[ "${MINI_TRAIN_FRAMES_PER_SCENE}" -gt "${PROFILE_MAX_TRAIN_FRAMES_PER_SCENE}" ]] ||
  [[ "${MINI_VALIDATION_FRAMES_PER_SCENE}" -gt "${PROFILE_MAX_VALIDATION_FRAMES_PER_SCENE}" ]]; then
  echo "Error: ${PROFILE} 的 train/validation frames per scene 不得高于 ${PROFILE_MAX_TRAIN_FRAMES_PER_SCENE}/${PROFILE_MAX_VALIDATION_FRAMES_PER_SCENE}"
  exit 2
fi
if [[ "${MINI_MAX_GPU_TEMP_C}" -gt "${PROFILE_MAX_GPU_TEMP_C}" ||
  "${MINI_MAX_START_TEMP_C}" -gt "${PROFILE_MAX_START_TEMP_C}" ]]; then
  echo "Error: ${PROFILE} 受保护入口不得提高 ${PROFILE_MAX_GPU_TEMP_C} C 运行温度或 ${PROFILE_MAX_START_TEMP_C} C 启动温度上限"
  exit 2
fi
if [[ "${MINI_MAX_STAGE_MINUTES}" -gt "${PROFILE_MAX_STAGE_MINUTES}" ]]; then
  echo "Error: ${PROFILE} 的单阶段时长不得高于 ${PROFILE_MAX_STAGE_MINUTES} 分钟"
  exit 2
fi
if [[ "${MINI_MIN_GPU_MEMORY_MIB}" -lt 7500 ||
  "${MINI_MIN_FREE_GPU_MEMORY_MIB}" -lt "${PROFILE_MIN_FREE_GPU_MEMORY_MIB}" ]]; then
  echo "Error: ${PROFILE} 受保护入口不得降低总显存 7500 MiB 或可用显存 ${PROFILE_MIN_FREE_GPU_MEMORY_MIB} MiB 门槛"
  exit 2
fi
if [[ "${MINI_THERMAL_POLL_SECONDS}" -gt 10 || "${MINI_STOP_GRACE_SECONDS}" -gt 15 ]]; then
  echo "Error: 温度轮询不得慢于 10 秒，中止宽限不得长于 15 秒"
  exit 2
fi
if [[ "${MINI_MAX_START_TEMP_C}" -ge "${MINI_MAX_GPU_TEMP_C}" ]]; then
  echo "Error: 启动温度阈值必须低于运行温度阈值"
  exit 2
fi
if [[ "${MINI_PREFLIGHT_ONLY}" != "0" && "${MINI_PREFLIGHT_ONLY}" != "1" ]]; then
  echo "Error: MINI_PREFLIGHT_ONLY 必须为 0 或 1"
  exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "Error: 找不到 nvidia-smi，无法启用 GPU 温度门禁"
  exit 1
fi
if ! command -v setsid >/dev/null 2>&1; then
  echo "Error: 找不到 setsid，无法可靠终止过热训练进程组"
  exit 1
fi

query_gpu_state() {
  nvidia-smi -i "${CUDA_DEVICES}" \
    --query-gpu=name,memory.total,memory.free,temperature.gpu \
    --format=csv,noheader,nounits
}

GPU_STATE="$(query_gpu_state)" || {
  echo "Error: 无法读取 GPU ${CUDA_DEVICES} 状态"
  exit 1
}
IFS=',' read -r GPU_NAME GPU_TOTAL_MIB GPU_FREE_MIB GPU_TEMP_C <<< "${GPU_STATE}"
GPU_NAME="${GPU_NAME#${GPU_NAME%%[![:space:]]*}}"
GPU_NAME="${GPU_NAME%${GPU_NAME##*[![:space:]]}}"
GPU_TOTAL_MIB="${GPU_TOTAL_MIB//[[:space:]]/}"
GPU_FREE_MIB="${GPU_FREE_MIB//[[:space:]]/}"
GPU_TEMP_C="${GPU_TEMP_C//[[:space:]]/}"
for value in "${GPU_TOTAL_MIB}" "${GPU_FREE_MIB}" "${GPU_TEMP_C}"; do
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    echo "Error: nvidia-smi 返回了无法解析的状态：${GPU_STATE}"
    exit 1
  fi
done
if [[ -n "${PROFILE_REQUIRED_GPU_NAME}" &&
  "${GPU_NAME}" != "${PROFILE_REQUIRED_GPU_NAME}" ]]; then
  echo "Error: ${PROFILE} 仅允许 ${PROFILE_REQUIRED_GPU_NAME}，实际为 ${GPU_NAME}"
  exit 1
fi
if [[ "${GPU_TOTAL_MIB}" -lt "${MINI_MIN_GPU_MEMORY_MIB}" ]]; then
  echo "Error: GPU 总显存 ${GPU_TOTAL_MIB} MiB 低于门槛 ${MINI_MIN_GPU_MEMORY_MIB} MiB"
  exit 1
fi
if [[ "${GPU_FREE_MIB}" -lt "${MINI_MIN_FREE_GPU_MEMORY_MIB}" ]]; then
  echo "Error: GPU 可用显存仅 ${GPU_FREE_MIB} MiB；请关闭占用 GPU 的程序后重试"
  exit 1
fi
if [[ "${GPU_TEMP_C}" -gt "${MINI_MAX_START_TEMP_C}" ]]; then
  echo "Error: GPU 当前 ${GPU_TEMP_C} C，高于启动门槛 ${MINI_MAX_START_TEMP_C} C；请先冷却"
  exit 1
fi

MINI_RESULTS_DIR="${MINI_RESULTS_DIR:-${DEFAULT_MINI_RESULTS_DIR}}"
MINI_RESULTS_DIR="$(realpath -m -- "${MINI_RESULTS_DIR}")"
RESULT_ROOT="$(realpath -m -- "${ROOT_DIR}/test/result")"
case "${MINI_RESULTS_DIR}" in
  "${RESULT_ROOT}"/* | /tmp/*) ;;
  *)
    echo "Error: unsafe MINI_RESULTS_DIR: ${MINI_RESULTS_DIR}"
    exit 1
    ;;
esac
if [[ -L "${MINI_RESULTS_DIR}" ]]; then
  echo "Error: MINI_RESULTS_DIR 拒绝符号链接: ${MINI_RESULTS_DIR}"
  exit 1
fi

STAGE_DIR="${MINI_RESULTS_DIR}/${MODE}"
if [[ -d "${STAGE_DIR}" ]] &&
  [[ -n "$(find "${STAGE_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Error: ${STAGE_DIR} 已包含输出；formal mini 默认拒绝覆盖或隐式续训"
  exit 1
fi
if [[ "${MODE}" == "ldm" && ! -f "${MINI_RESULTS_DIR}/vae/vae_best.pt" ]]; then
  echo "Error: 请先完成 formal mini VAE: ${MINI_RESULTS_DIR}/vae/vae_best.pt"
  exit 1
fi
if [[ "${MODE}" == "cd" ]]; then
  for checkpoint in \
    "${MINI_RESULTS_DIR}/vae/vae_best.pt" \
    "${MINI_RESULTS_DIR}/ldm/ldm_best.pt"; do
    if [[ ! -f "${checkpoint}" ]]; then
      echo "Error: formal mini CD 缺少父 checkpoint: ${checkpoint}"
      exit 1
    fi
  done
fi

export MINI_RESULTS_DIR
export MINI_PREFLIGHT_ONLY
export MINI_CONFIG_PATH="${MINI_RESULTS_DIR}/mini_${MODE}_config.yaml"
export MINI_RADAR_PROTOCOL="formal"
export MINI_CHECKPOINT_PROTOCOL="formal_mini_chain_v2"
export PREPROCESSED_ROOT="${ROOT_DIR}/Data/NTU4DRadLM_Pre_formal_v2_80m_86p8_v1"
export MINI_DATASET_DIR="${PREPROCESSED_ROOT}"
export CALIB_CONFIG_DIR="${ROOT_DIR}/Data/config"
export TRAIN_SCENES_OVERRIDE="garden"
export MINI_RADAR_NORMALIZATION_PATH="${ROOT_DIR}/diffusion_consistency_radar/config/radar_normalization_garden_32x128x128_80m_train80_purge3s_86p8_v2.json"
export EXPECTED_FORMAL_ARTIFACT_SHA256="11f59d84cc186c39256c112154faf458ec9ead5fec9b08b997abd5058b68e97c"
export MINI_TEMPORAL_SPLIT_ARTIFACT="${PREPROCESSED_ROOT}/temporal_split_garden_train80_purge3s_v1.json"
export MINI_DATA_PROTOCOL_PATH="${PREPROCESSED_ROOT}/formal_data_protocol_garden_train80_purge3s_v1.json"
export MINI_DOPPLER_SCALE_MPS="86.8"
export MINI_TARGET_SIZE="32,128,128"
export MINI_SOURCE_PC_RANGE="0,-20,-6,80,20,10"
export MINI_MODEL_PC_RANGE="0,-20,-6,80,20,10"
export MINI_VAE_CONFIG_TYPE="ultra_lightweight"
export MINI_VAE_LATENT_DIM=""
export MINI_VAE_OCC_LOSS="bce_dice"
export MINI_TRAIN_SPLIT="0.8"
export MINI_SPLIT_SEED="42"
export MINI_USE_AUG="false"
export MINI_REQUIRE_FRESH_SCRATCH="0"
export MINI_REQUIRE_FRESH_CONFIG="1"
export MINI_TRAIN_FRAMES_PER_SCENE MINI_VALIDATION_FRAMES_PER_SCENE
export MINI_VAE_EPOCHS MINI_LDM_EPOCHS MINI_CD_EPOCHS
export MINI_BATCH_SIZE MINI_NUM_WORKERS MINI_GRAD_ACCUM
# PyTorch 2.4.1 在本机 expandable segment 路径触发过 allocator 内部断言；固定使用 native allocator 分块策略。
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

case "${MODE}" in
  vae) ACTIVE_STAGE_EPOCHS="${MINI_VAE_EPOCHS}" ;;
  ldm) ACTIVE_STAGE_EPOCHS="${MINI_LDM_EPOCHS}" ;;
  cd) ACTIVE_STAGE_EPOCHS="${MINI_CD_EPOCHS}" ;;
esac

echo "=========================================="
echo "Formal mini 8 GB guarded run"
echo "stage: ${MODE}"
echo "profile: ${PROFILE}"
echo "GPU: ${GPU_NAME}"
echo "GPU memory total/free: ${GPU_TOTAL_MIB}/${GPU_FREE_MIB} MiB"
echo "GPU start/max temperature: ${GPU_TEMP_C}/${MINI_MAX_GPU_TEMP_C} C"
echo "train/validation frames per scene: ${MINI_TRAIN_FRAMES_PER_SCENE}/${MINI_VALIDATION_FRAMES_PER_SCENE}"
echo "selected frames per scene: $((MINI_TRAIN_FRAMES_PER_SCENE + MINI_VALIDATION_FRAMES_PER_SCENE))"
echo "epochs/batch: ${ACTIVE_STAGE_EPOCHS}/${MINI_BATCH_SIZE}"
echo "max stage runtime: ${MINI_MAX_STAGE_MINUTES} min"
echo "CUDA allocator: ${PYTORCH_CUDA_ALLOC_CONF}"
echo "results: ${MINI_RESULTS_DIR}"
echo "=========================================="

if [[ "${MINI_PREFLIGHT_ONLY}" == "1" ]]; then
  bash "${TRAIN_SCRIPT}" "${MODE}"
  echo "Formal mini 8 GB preflight passed; training was not started."
  exit 0
fi

mkdir -p "${MINI_RESULTS_DIR}"

stop_training_group() {
  local signal="$1"
  if [[ -n "${TRAIN_PID:-}" ]] && kill -0 "${TRAIN_PID}" 2>/dev/null; then
    kill -"${signal}" -- "-${TRAIN_PID}" 2>/dev/null || true
  fi
}

terminate_training_group() {
  local reason="$1"
  echo "Error: ${reason}"
  stop_training_group INT
  sleep "${MINI_STOP_GRACE_SECONDS}"
  if kill -0 "${TRAIN_PID}" 2>/dev/null; then
    echo "训练进程未响应 INT，升级为 TERM"
    stop_training_group TERM
    sleep "${MINI_STOP_GRACE_SECONDS}"
  fi
  if kill -0 "${TRAIN_PID}" 2>/dev/null; then
    echo "训练进程未响应 TERM，升级为 KILL"
    stop_training_group KILL
  fi
}

cleanup_guard() {
  if [[ -n "${TRAIN_PID:-}" ]] && kill -0 "${TRAIN_PID}" 2>/dev/null; then
    terminate_training_group "收到中止信号，停止训练进程组"
  fi
  if [[ -n "${MONITOR_PID:-}" ]]; then
    kill "${MONITOR_PID}" 2>/dev/null || true
  fi
}
trap cleanup_guard INT TERM

START_SECONDS="${SECONDS}"
setsid bash "${TRAIN_SCRIPT}" "${MODE}" &
TRAIN_PID=$!

(
  while kill -0 "${TRAIN_PID}" 2>/dev/null; do
    sleep "${MINI_THERMAL_POLL_SECONDS}"
    if ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
      break
    fi
    if ! CURRENT_STATE="$(query_gpu_state)"; then
      terminate_training_group "训练期间无法读取 GPU 状态，执行保护性中止"
      exit 97
    fi
    CURRENT_TEMP="${CURRENT_STATE##*,}"
    CURRENT_TEMP="${CURRENT_TEMP//[[:space:]]/}"
    if [[ ! "${CURRENT_TEMP}" =~ ^[0-9]+$ ]]; then
      terminate_training_group "训练期间温度值无法解析，执行保护性中止：${CURRENT_STATE}"
      exit 97
    fi
    if [[ "${CURRENT_TEMP}" -ge "${MINI_MAX_GPU_TEMP_C}" ]]; then
      terminate_training_group "GPU 达到 ${CURRENT_TEMP} C，触发 ${MINI_MAX_GPU_TEMP_C} C 温度门禁"
      exit 97
    fi
    if ((SECONDS - START_SECONDS >= MINI_MAX_STAGE_MINUTES * 60)); then
      terminate_training_group "阶段达到 ${MINI_MAX_STAGE_MINUTES} 分钟上限，执行保护性中止"
      exit 97
    fi
  done
) &
MONITOR_PID=$!

set +e
wait "${TRAIN_PID}"
TRAIN_STATUS=$?
set -e
set +e
wait "${MONITOR_PID}" 2>/dev/null
MONITOR_STATUS=$?
set -e
trap - INT TERM

if [[ "${MONITOR_STATUS}" -eq 97 ]]; then
  echo "Formal mini ${MODE} 已由硬件保护门禁中止；保留日志和 checkpoint 供诊断。"
  exit 97
fi

if [[ "${TRAIN_STATUS}" -ne 0 ]]; then
  echo "Formal mini ${MODE} 未完成，退出码 ${TRAIN_STATUS}；保留日志和 checkpoint 供诊断。"
  exit "${TRAIN_STATUS}"
fi

FINAL_STATE="$(query_gpu_state)" || FINAL_STATE="unavailable"
echo "Formal mini ${MODE} completed. GPU final state: ${FINAL_STATE}"
echo "运行下一阶段前请让 GPU 温度回落到 ${MINI_MAX_START_TEMP_C} C 以下。"
