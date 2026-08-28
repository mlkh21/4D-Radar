#!/bin/bash
# 正式离线评价入口：只评价部署阶段已保存的 voxel，不加载或重跑生成模型

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"

EVALUATE_SCRIPT="${PROJECT_DIR}/scripts/evaluate_saved_predictions.py"
MANIFEST_SCRIPT="${PROJECT_DIR}/scripts/dataset_manifest.py"
DATA_LOADING_CONFIG="${PROJECT_DIR}/config/data_loading_config.yml"
PROTOCOL_TAG="formal_v2_80m_86p8_v1"
PREPROCESSED_ROOT="${ROOT_DIR}/Data/NTU4DRadLM_Pre_formal_v2_80m_86p8_v1"
RAW_ROOT="${ROOT_DIR}/Data/NTU4DRadLM_Raw_p1_01_candidate"
TARGET_THRESHOLD="${TARGET_THRESHOLD:-0.5}"
MAX_EVAL_FILES="${MAX_EVAL_FILES:-0}"

MODE="${1:-}"
case "${MODE}" in
  ldm)
    DEPLOY_SUFFIX="${PROTOCOL_TAG}_ldm_deploy"
    EVALUATION_SUFFIX="${PROTOCOL_TAG}_ldm_evaluation"
    ;;
  cd)
    DEPLOY_SUFFIX="${PROTOCOL_TAG}_cd_1step_deploy"
    EVALUATION_SUFFIX="${PROTOCOL_TAG}_cd_1step_evaluation"
    ;;
  cd4)
    DEPLOY_SUFFIX="${PROTOCOL_TAG}_cd_4step_deploy"
    EVALUATION_SUFFIX="${PROTOCOL_TAG}_cd_4step_evaluation"
    ;;
  *)
    echo "用法: $0 ldm|cd|cd4"
    exit 2
    ;;
esac

if [ ! -f "${DATA_LOADING_CONFIG}" ]; then
  echo "错误: 配置文件不存在: ${DATA_LOADING_CONFIG}"
  exit 1
fi

if [ ! -f "${EVALUATE_SCRIPT}" ]; then
  echo "错误: 离线评价脚本不存在: ${EVALUATE_SCRIPT}"
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
    text = str(scene).strip()
    if text:
        print(text)
PY
)

if [ ${#TEST_SCENES[@]} -eq 0 ]; then
  echo "错误: data_loading_config.yml 中 data.test 为空"
  exit 1
fi

# 所有场景先通过含 observed mask 的 training manifest，再允许写评价目录。
for SCENE in "${TEST_SCENES[@]}"; do
  SCENE_DIR="${PREPROCESSED_ROOT}/${SCENE}"
  python "${MANIFEST_SCRIPT}" validate \
    --scene_dir "${SCENE_DIR}" \
    --expected_scene "${SCENE}" \
    --expected_profile training
done

for SCENE in "${TEST_SCENES[@]}"; do
  SCENE_DIR="${PREPROCESSED_ROOT}/${SCENE}"
  PRED_DIR="${ROOT_DIR}/Result/inference_results/${SCENE}_${DEPLOY_SUFFIX}"
  OUTPUT_DIR="${ROOT_DIR}/Result/inference_results/${SCENE}_${EVALUATION_SUFFIX}"
  RADAR_VOXEL_DIR="${SCENE_DIR}/radar_voxel"
  TARGET_VOXEL_DIR="${SCENE_DIR}/target_voxel"
  RAW_LIVOX_DIR="${RAW_ROOT}/${SCENE}/livox_lidar"
  LIDAR_INDEX_FILE="${RAW_ROOT}/${SCENE}/lidar_index_sequence.txt"
  RUN_METADATA_PATH="${PRED_DIR}/inference_run.json"

  if [ ! -d "${PRED_DIR}" ]; then
    echo "错误: 部署预测目录不存在: ${PRED_DIR}"
    exit 1
  fi
  if [ ! -f "${RUN_METADATA_PATH}" ]; then
    echo "错误: 部署运行协议不存在: ${RUN_METADATA_PATH}"
    exit 1
  fi
  if [ ! -d "${RADAR_VOXEL_DIR}" ] || [ ! -d "${TARGET_VOXEL_DIR}" ]; then
    echo "错误: sensor-aware Radar/target 目录不完整: ${SCENE_DIR}"
    exit 1
  fi
  if [ ! -d "${RAW_LIVOX_DIR}" ] || [ ! -f "${LIDAR_INDEX_FILE}" ]; then
    echo "错误: raw LiDAR 或索引缺失: ${RAW_ROOT}/${SCENE}"
    exit 1
  fi

  echo "开始离线评价已保存预测: ${SCENE} (${MODE})"
  conda run -n Radar-Diffusion python "${EVALUATE_SCRIPT}" \
    --pred_voxel_dir "${PRED_DIR}" \
    --radar_voxel_dir "${RADAR_VOXEL_DIR}" \
    --target_voxel_dir "${TARGET_VOXEL_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --run_metadata_path "${RUN_METADATA_PATH}" \
    --raw_livox_dir "${RAW_LIVOX_DIR}" \
    --lidar_index_file "${LIDAR_INDEX_FILE}" \
    --target_threshold "${TARGET_THRESHOLD}" \
    --max_files "${MAX_EVAL_FILES}"

  echo "完成离线评价: ${OUTPUT_DIR}"
done
