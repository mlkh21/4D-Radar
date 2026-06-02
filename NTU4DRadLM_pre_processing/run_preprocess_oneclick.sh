#!/bin/bash
# 一键预处理流水线：
# 1) 重新生成毫米波雷达/激光雷达时间戳索引
# 2) 运行具有可配置对齐选项的体素预处理

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

RAW_DATA_PATH="${RAW_DATA_PATH:-${ROOT_DIR}/Data/NTU4DRadLM_Raw}"
INDEX_PATH="${INDEX_PATH:-${RAW_DATA_PATH}}"
OUTPUT_PATH="${OUTPUT_PATH:-${ROOT_DIR}/Data/NTU4DRadLM_Pre}"
CALIB_PATH="${CALIB_PATH:-${ROOT_DIR}/Data/config/calib_radar_to_livox.txt}"

SCENE="${SCENE:-}"
MAX_FRAMES="${MAX_FRAMES:-0}"
ALIGN_TO="${ALIGN_TO:-lidar}"        # lidar | radar
INVERT_CALIB="${INVERT_CALIB:-0}"    # 0 | 1
RADAR_Z_SHIFT="${RADAR_Z_SHIFT:-0.0}"

PYTHON_BIN=()
if [[ -n "${PYTHON_BIN_OVERRIDE:-}" ]]; then
  PYTHON_BIN=("${PYTHON_BIN_OVERRIDE}")
elif command -v conda >/dev/null 2>&1; then
  PYTHON_BIN=(conda run -n Radar-Diffusion python)
else
  PYTHON_BIN=(python)
fi

echo "=========================================="
echo "4D-Radar 预处理"
echo "=========================================="
echo "raw_data_path: ${RAW_DATA_PATH}"
echo "index_path: ${INDEX_PATH}"
echo "output_path: ${OUTPUT_PATH}"
echo "calib_path: ${CALIB_PATH}"
echo "scene: ${SCENE:-<all>}"
echo "max_frames: ${MAX_FRAMES}"
echo "align_to: ${ALIGN_TO}"
echo "invert_calib: ${INVERT_CALIB}"
echo "radar_z_shift: ${RADAR_Z_SHIFT}"
echo "python: ${PYTHON_BIN[*]}"
echo "=========================================="

echo "[1/2] 生成时间戳索引..."
"${PYTHON_BIN[@]}" "${SCRIPT_DIR}/NTU4DRadLM_timestamp_index.py"

echo "[2/2] 运行体素预处理..."
CMD=(
  "${PYTHON_BIN[@]}"
  "${SCRIPT_DIR}/NTU4DRadLM_pre_processing.py"
  --raw_data_path "${RAW_DATA_PATH}"
  --index_path "${INDEX_PATH}"
  --output_path "${OUTPUT_PATH}"
  --calib_path "${CALIB_PATH}"
  --max_frames "${MAX_FRAMES}"
  --align_to "${ALIGN_TO}"
  --radar_z_shift "${RADAR_Z_SHIFT}"
)

if [[ -n "${SCENE}" ]]; then
  CMD+=(--scene "${SCENE}")
fi

if [[ "${INVERT_CALIB}" == "1" ]]; then
  CMD+=(--invert_calib)
fi

"${CMD[@]}"

echo "=========================================="
echo "预处理完成."
echo "输出路径: ${OUTPUT_PATH}"
echo "=========================================="

