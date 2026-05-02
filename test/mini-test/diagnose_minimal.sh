#!/bin/bash
# Diagnose minimal inference outputs inside test/mini-test.
# Usage:
#   bash test/mini-test/diagnose_minimal.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_DIR="${ROOT_DIR}/diffusion_consistency_radar"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD=("${PYTHON_BIN}")
elif python -c "import torch" >/dev/null 2>&1; then
  PYTHON_CMD=(python)
elif command -v conda >/dev/null 2>&1; then
  PYTHON_CMD=(conda run -n Radar-Diffusion python)
else
  PYTHON_CMD=(python)
fi

DIAG_SCRIPT="${PROJECT_DIR}/scripts/diagnose_generation_quality.py"
PREPROCESSED_ROOT="${ROOT_DIR}/Data/NTU4DRadLM_Pre"
SCENE="${SCENE:-loop3}"
PRED_DIR="${PRED_DIR:-${SCRIPT_DIR}/inference_results_mini/${SCENE}_ldm_eval}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/diagnostics/${SCENE}_ldm_eval}"
MAX_FILES="${MAX_FILES:-20}"
OCC_THRESHOLD="${OCC_THRESHOLD:-0.1}"

"${PYTHON_CMD[@]}" "${DIAG_SCRIPT}" \
  --radar_voxel_dir "${PREPROCESSED_ROOT}/${SCENE}/radar_voxel" \
  --target_voxel_dir "${PREPROCESSED_ROOT}/${SCENE}/target_voxel" \
  --pred_dir "${PRED_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_files "${MAX_FILES}" \
  --occ_threshold "${OCC_THRESHOLD}"

echo "Mini-test diagnosis done: ${OUTPUT_DIR}"
