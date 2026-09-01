#!/bin/bash
# Legacy diagnostic-only 图像对比入口；不生成正式 evaluation summary。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"
SCENE="${SCENE:-loop3}"
METHOD="${METHOD:-ldm_eval}"
PRED_PCL_DIR="${PRED_PCL_DIR:-${ROOT_DIR}/Result/inference_results/${SCENE}_${METHOD}}"
RAW_LIVOX_DIR="${RAW_LIVOX_DIR:-${ROOT_DIR}/Data/NTU4DRadLM_Raw/${SCENE}/livox_lidar}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/Result/comparison_results/${SCENE}_${METHOD}}"

echo "警告: compare.sh 仅用于 legacy diagnostic 图像对比，不是正式评价入口。"
python "${PROJECT_DIR}/scripts/compare_radar_lidar_images.py" \
  --pred_pcl_dir "${PRED_PCL_DIR}" \
  --raw_livox_dir "${RAW_LIVOX_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_files "${MAX_FILES:-1}" \
  --mode "${MODE:-3d}" \
  --point_size "${POINT_SIZE:-0.8}"
