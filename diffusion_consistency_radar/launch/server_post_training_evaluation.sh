#!/usr/bin/env bash
# 文件功能：在服务器完成正式训练后，安全执行合同校验、loop3 推理 smoke、全量推理与离线评价。

set -Eeuo pipefail

on_error() {
    local status="$?"
    local line="$1"
    echo "失败：第 ${line} 行，退出码 ${status}；后续阶段未执行。" >&2
    exit "${status}"
}
trap 'on_error "${LINENO}"' ERR

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SELF_DIR}/.." && pwd)"
ROOT_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"

CONDA_ENV="${CONDA_ENV:-Radar}"
TRAIN_TAG="${TRAIN_TAG:-formal_v2_1_80m_86p8_db_snr_v1}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
INFERENCE_SEED="${INFERENCE_SEED:-42}"
SMOKE_FILES="${SMOKE_FILES:-16}"
RUN_SMOKE="${RUN_SMOKE:-1}"
RUN_CD_FULL="${RUN_CD_FULL:-1}"
RUN_LDM_FULL="${RUN_LDM_FULL:-1}"
CONFIRM_FULL_EVALUATION="${CONFIRM_FULL_EVALUATION:-NO}"

TRAIN_RESULTS="${TRAIN_RESULTS:-${ROOT_DIR}/Result/train_results/${TRAIN_TAG}}"
TRAIN_DATA="${TRAIN_DATA:-${ROOT_DIR}/Data/NTU4DRadLM_Pre_formal_v2_1_80m_86p8_db_snr_v1}"
DEPLOY_DATA="${DEPLOY_DATA:-${ROOT_DIR}/Data/NTU4DRadLM_Deploy_formal_v2_1_80m_86p8_db_snr_v1}"
RAW_DATA="${RAW_DATA:-${ROOT_DIR}/Data/NTU4DRadLM_Raw_formal_v2_1_80m_86p8_db_snr_v1}"
CALIBRATION_DIR="${CALIBRATION_DIR:-${ROOT_DIR}/Data/config}"

VAE_CKPT="${TRAIN_RESULTS}/vae/vae_best.pt"
LDM_CKPT="${TRAIN_RESULTS}/ldm/ldm_best.pt"
CD_CKPT="${TRAIN_RESULTS}/cd/cd_best.pt"
LDM_THRESHOLD="${TRAIN_RESULTS}/ldm/occupancy_threshold.json"
CD_THRESHOLD="${TRAIN_RESULTS}/cd/occupancy_threshold.json"

LOOP3_DEPLOY="${DEPLOY_DATA}/loop3"
LOOP3_TRAIN="${TRAIN_DATA}/loop3"
LOOP3_RAW="${RAW_DATA}/loop3"

INFERENCE_SCRIPT="${PROJECT_DIR}/scripts/inference.py"
EVALUATION_SCRIPT="${PROJECT_DIR}/scripts/evaluate_saved_predictions.py"
DEPLOYMENT_VIEW_SCRIPT="${PROJECT_DIR}/scripts/build_deployment_view.py"

CONDA_RUN=(conda run --no-capture-output -n "${CONDA_ENV}")

fail() {
    echo "错误：$*" >&2
    exit 1
}

require_file() {
    [[ -f "$1" && ! -L "$1" ]] || fail "缺少普通文件：$1"
}

require_dir() {
    [[ -d "$1" && ! -L "$1" ]] || fail "缺少普通目录：$1"
}

require_fresh_path() {
    [[ ! -e "$1" && ! -L "$1" ]] || fail "输出已存在，拒绝覆盖：$1"
}

validate_binary_flag() {
    local name="$1"
    local value="$2"
    [[ "${value}" == "0" || "${value}" == "1" ]] \
        || fail "${name} 只能为 0 或 1，实际为 ${value}"
}

validate_binary_flag RUN_SMOKE "${RUN_SMOKE}"
validate_binary_flag RUN_CD_FULL "${RUN_CD_FULL}"
validate_binary_flag RUN_LDM_FULL "${RUN_LDM_FULL}"
[[ "${SMOKE_FILES}" =~ ^[1-9][0-9]*$ ]] \
    || fail "SMOKE_FILES 必须是正整数"
[[ "${INFERENCE_SEED}" =~ ^[0-9]+$ ]] \
    || fail "INFERENCE_SEED 必须是非负整数"
[[ "${CUDA_DEVICE}" =~ ^[0-9]+$ ]] \
    || fail "CUDA_DEVICE 必须是单个非负 GPU 编号"
case "${CONFIRM_FULL_EVALUATION}" in
    YES|NO) ;;
    *) fail "CONFIRM_FULL_EVALUATION 只能为 YES 或 NO" ;;
esac

cd "${ROOT_DIR}"

for file in \
    "${VAE_CKPT}" \
    "${LDM_CKPT}" \
    "${CD_CKPT}" \
    "${LDM_THRESHOLD}" \
    "${CD_THRESHOLD}" \
    "${LOOP3_RAW}/lidar_index_sequence.txt" \
    "${INFERENCE_SCRIPT}" \
    "${EVALUATION_SCRIPT}" \
    "${DEPLOYMENT_VIEW_SCRIPT}"; do
    require_file "${file}"
done

for directory in \
    "${LOOP3_DEPLOY}/radar_voxel" \
    "${LOOP3_TRAIN}/radar_voxel" \
    "${LOOP3_TRAIN}/target_voxel" \
    "${LOOP3_RAW}/livox_lidar" \
    "${CALIBRATION_DIR}"; do
    require_dir "${directory}"
done

command -v conda >/dev/null 2>&1 || fail "未找到 conda"
nvidia-smi -i "${CUDA_DEVICE}" --query-gpu=name,memory.total \
    --format=csv,noheader >/dev/null \
    || fail "GPU ${CUDA_DEVICE} 不可用"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${ROOT_DIR}/Result/post_training_runs/${TRAIN_TAG}_${RUN_ID}}"
require_fresh_path "${RUN_LOG_DIR}"
mkdir -p "${RUN_LOG_DIR}"

echo "步骤 1/7：读取正式 validation threshold 选择。"
"${CONDA_RUN[@]}" python - \
    "${LDM_THRESHOLD}" "${CD_THRESHOLD}" <<'PY'
import json
import sys

for path in sys.argv[1:]:
    with open(path, "r", encoding="utf-8") as handle:
        artifact = json.load(handle)
    print(
        json.dumps(
            {
                "path": path,
                "stage": artifact.get("stage"),
                "selected_threshold": artifact.get("selected_threshold"),
                "safety_recall_qualification": artifact.get(
                    "safety_recall_qualification"
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
PY

echo "步骤 2/7：校验 VAE/LDM/CD checkpoint 父链。"
"${CONDA_RUN[@]}" python - \
    "${VAE_CKPT}" "${LDM_CKPT}" "${CD_CKPT}" <<'PY'
import json
import sys

from diffusion_consistency_radar.checkpoint_chain import (
    validate_formal_checkpoint_chain,
)

report = validate_formal_checkpoint_chain(
    sys.argv[1],
    sys.argv[2],
    sys.argv[3],
    require_multimodal=True,
    target_stage="cd",
)
print(
    json.dumps(
        {
            "status": "正式 checkpoint 链校验通过",
            "protocol": report.get("protocol"),
            "stages": report.get("stages"),
            "grid": report.get("grid"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
)
PY

echo "步骤 3/7：只读验证完整 loop3 deployment view。"
"${CONDA_RUN[@]}" python "${DEPLOYMENT_VIEW_SCRIPT}" validate \
    --dataset_dir "${DEPLOY_DATA}" \
    --scene loop3

run_inference() {
    local label="$1"
    local model_type="$2"
    local model_ckpt="$3"
    local steps="$4"
    local sampler="$5"
    local max_files="$6"
    local threshold_artifact="$7"
    local output_dir="$8"

    require_fresh_path "${output_dir}"
    echo "开始推理：${label}"
    "${CONDA_RUN[@]}" python "${INFERENCE_SCRIPT}" \
        --vae_ckpt "${VAE_CKPT}" \
        --model_ckpt "${model_ckpt}" \
        --model_type "${model_type}" \
        --steps "${steps}" \
        --sampler "${sampler}" \
        --seed "${INFERENCE_SEED}" \
        --max_files "${max_files}" \
        --threshold_artifact "${threshold_artifact}" \
        --empty_fallback_topk 0 \
        --radar_voxel_dir "${LOOP3_DEPLOY}/radar_voxel" \
        --deployment_scene_dir "${LOOP3_DEPLOY}" \
        --calibration_dir "${CALIBRATION_DIR}" \
        --require_real_ir \
        --save_voxel \
        --save_pointcloud \
        --save_uncertainty \
        --output_dir "${output_dir}" \
        --device cuda \
        2>&1 | tee "${RUN_LOG_DIR}/${label}_inference.log"
}

run_evaluation() {
    local label="$1"
    local prediction_dir="$2"
    local output_dir="$3"

    require_file "${prediction_dir}/inference_run.json"
    require_fresh_path "${output_dir}"
    echo "开始离线评价：${label}"
    "${CONDA_RUN[@]}" python "${EVALUATION_SCRIPT}" \
        --pred_voxel_dir "${prediction_dir}" \
        --radar_voxel_dir "${LOOP3_TRAIN}/radar_voxel" \
        --target_voxel_dir "${LOOP3_TRAIN}/target_voxel" \
        --output_dir "${output_dir}" \
        --run_metadata_path "${prediction_dir}/inference_run.json" \
        --raw_livox_dir "${LOOP3_RAW}/livox_lidar" \
        --lidar_index_file "${LOOP3_RAW}/lidar_index_sequence.txt" \
        --target_threshold 0.5 \
        --max_files 0 \
        2>&1 | tee "${RUN_LOG_DIR}/${label}_evaluation.log"
    require_file "${output_dir}/evaluation_summary.json"
}

if [[ "${RUN_SMOKE}" == "1" ]]; then
    echo "步骤 4/7：执行 ${SMOKE_FILES} 帧 CD/LDM 推理 smoke。"
    CD_SMOKE_OUT="${ROOT_DIR}/Result/inference_results/loop3_${TRAIN_TAG}_cd_1step_smoke${SMOKE_FILES}_${RUN_ID}"
    LDM_SMOKE_OUT="${ROOT_DIR}/Result/inference_results/loop3_${TRAIN_TAG}_ldm_smoke${SMOKE_FILES}_${RUN_ID}"
    CD_SMOKE_EVAL="${CD_SMOKE_OUT}_evaluation"
    LDM_SMOKE_EVAL="${LDM_SMOKE_OUT}_evaluation"

    run_inference \
        "cd_smoke${SMOKE_FILES}" cd "${CD_CKPT}" 1 euler \
        "${SMOKE_FILES}" "${CD_THRESHOLD}" "${CD_SMOKE_OUT}"
    run_evaluation \
        "cd_smoke${SMOKE_FILES}" "${CD_SMOKE_OUT}" "${CD_SMOKE_EVAL}"

    run_inference \
        "ldm_smoke${SMOKE_FILES}" ldm "${LDM_CKPT}" 40 heun \
        "${SMOKE_FILES}" "${LDM_THRESHOLD}" "${LDM_SMOKE_OUT}"
    run_evaluation \
        "ldm_smoke${SMOKE_FILES}" "${LDM_SMOKE_OUT}" "${LDM_SMOKE_EVAL}"
else
    echo "步骤 4/7：RUN_SMOKE=0，按请求跳过 smoke。"
fi

if [[ "${CONFIRM_FULL_EVALUATION}" != "YES" ]]; then
    echo "smoke/预检已完成；未启动全量 loop3。"
    echo "确认磁盘空间和输出路径后，使用以下命令启动全量评价："
    echo "CONFIRM_FULL_EVALUATION=YES RUN_SMOKE=0 bash $0"
    echo "运行日志目录：${RUN_LOG_DIR}"
    exit 0
fi

SUMMARY_FILES=()

echo "步骤 5/7：执行正式 CD 1-step 全量 loop3 推理与评价。"
if [[ "${RUN_CD_FULL}" == "1" ]]; then
    CD_FULL_OUT="${ROOT_DIR}/Result/inference_results/loop3_${TRAIN_TAG}_cd_1step_deploy"
    CD_FULL_EVAL="${ROOT_DIR}/Result/inference_results/loop3_${TRAIN_TAG}_cd_1step_evaluation"
    run_inference \
        cd_full cd "${CD_CKPT}" 1 euler 0 \
        "${CD_THRESHOLD}" "${CD_FULL_OUT}"
    run_evaluation cd_full "${CD_FULL_OUT}" "${CD_FULL_EVAL}"
    SUMMARY_FILES+=("${CD_FULL_EVAL}/evaluation_summary.json")
else
    echo "RUN_CD_FULL=0，跳过 CD 全量评价。"
fi

echo "步骤 6/7：执行正式 LDM 40-step 全量 loop3 推理与评价。"
if [[ "${RUN_LDM_FULL}" == "1" ]]; then
    LDM_FULL_OUT="${ROOT_DIR}/Result/inference_results/loop3_${TRAIN_TAG}_ldm_deploy"
    LDM_FULL_EVAL="${ROOT_DIR}/Result/inference_results/loop3_${TRAIN_TAG}_ldm_evaluation"
    run_inference \
        ldm_full ldm "${LDM_CKPT}" 40 heun 0 \
        "${LDM_THRESHOLD}" "${LDM_FULL_OUT}"
    run_evaluation ldm_full "${LDM_FULL_OUT}" "${LDM_FULL_EVAL}"
    SUMMARY_FILES+=("${LDM_FULL_EVAL}/evaluation_summary.json")
else
    echo "RUN_LDM_FULL=0，跳过 LDM 全量评价。"
fi

echo "步骤 7/7：汇总正式外部评价结果。"
if [[ "${#SUMMARY_FILES[@]}" -eq 0 ]]; then
    fail "未选择任何全量评价方法"
fi

"${CONDA_RUN[@]}" python - "${SUMMARY_FILES[@]}" <<'PY' \
    | tee "${RUN_LOG_DIR}/formal_evaluation_summary.txt"
import json
import os
import sys

for path in sys.argv[1:]:
    with open(path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    print("=" * 72)
    print(os.path.basename(os.path.dirname(path)))
    print("frame_count:", summary.get("frame_count"))
    print("occ_threshold:", summary.get("occ_threshold"))
    print("threshold_source:", summary.get("occ_threshold_source"))
    print(
        "safety_recall_qualification:",
        summary.get("threshold_safety_recall_qualification"),
    )
    print("formal_metrics:")
    print(
        json.dumps(
            summary.get("formal_metrics", {}),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    print("occupancy_3d_global:")
    print(
        json.dumps(
            summary.get("occupancy_3d_metrics", {}).get("global", {}),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
PY

echo "全部完成。"
echo "训练产物：${TRAIN_RESULTS}"
echo "运行日志目录：${RUN_LOG_DIR}"
echo "推理与评价目录：${ROOT_DIR}/Result/inference_results"
echo "注意：离线评价通过不等同于 ROS/PX4 飞行安全闭环验证。"
