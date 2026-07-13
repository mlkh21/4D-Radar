#!/bin/bash
# 使用固定 validation 子集评估 LDM 每个 epoch，并按任务/结构门槛推荐 checkpoint。
# 该脚本只写诊断结果，不复制、删除或覆盖任何 checkpoint。

set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SELF_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
EXP_DIR="${EXP_DIR:-${ROOT_DIR}/test/result/ldm/ablation/ldm_near40_500_z64_v9a_top_full}"
DATASET_ROOT="${DATASET_ROOT:-${EXP_DIR}/.tmp_mini_train_dataset}"
VAE_CKPT="${VAE_CKPT:-${EXP_DIR}/vae/vae_best.pt}"
LDM_DIR="${LDM_DIR:-${EXP_DIR}/ldm}"
OUTPUT_DIR="${OUTPUT_DIR:-${EXP_DIR}/checkpoint_selection_32_thr099}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Error: Python executable not found: ${PYTHON_BIN}"
  exit 1
fi
if [[ ! -f "${VAE_CKPT}" || ! -d "${LDM_DIR}" || ! -d "${DATASET_ROOT}" ]]; then
  echo "Error: missing VAE, LDM directory, or mini dataset under ${EXP_DIR}"
  exit 1
fi
if [[ -L "${OUTPUT_DIR}" ]]; then
  echo "Error: OUTPUT_DIR must not be a symlink: ${OUTPUT_DIR}"
  exit 1
fi
mkdir -p "$(dirname "${OUTPUT_DIR}")"
LOCK_PATH="${OUTPUT_DIR}.lock"
if ! mkdir -- "${LOCK_PATH}" 2>/dev/null; then
  echo "Error: checkpoint selection is already running: ${OUTPUT_DIR}"
  exit 1
fi
cleanup_lock() {
  rmdir -- "${LOCK_PATH}" 2>/dev/null || true
}
trap cleanup_lock EXIT INT TERM

if [[ -e "${OUTPUT_DIR}" && -n "$(find "${OUTPUT_DIR}" -mindepth 1 -print -quit)" ]]; then
  echo "Error: output already exists and is not empty: ${OUTPUT_DIR}"
  echo "Use a new OUTPUT_DIR; existing diagnostic results are never deleted or silently reused."
  exit 1
fi
mkdir -p -- "${OUTPUT_DIR}"
OUTPUT_CANONICAL="$(realpath -m -- "${OUTPUT_DIR}")"

mapfile -t CHECKPOINTS < <(find "${LDM_DIR}" -maxdepth 1 -type f -name 'ldm_epoch*.pt' | sort -V)
if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
  echo "Error: no ldm_epoch*.pt checkpoints found in ${LDM_DIR}"
  exit 1
fi

SELECT_ARGS=()
for checkpoint in "${CHECKPOINTS[@]}"; do
  name="$(basename "${checkpoint}" .pt)"
  candidate_dir="${OUTPUT_DIR}/${name}"
  summary_csv="${candidate_dir}/ir_condition_ablation_summary.csv"
  if [[ -L "${candidate_dir}" ]]; then
    echo "Error: candidate output must not be a symlink: ${candidate_dir}"
    exit 1
  fi
  candidate_canonical="$(realpath -m -- "${candidate_dir}")"
  case "${candidate_canonical}" in
    "${OUTPUT_CANONICAL}"/*) ;;
    *) echo "Error: candidate output escapes OUTPUT_DIR: ${candidate_canonical}"; exit 1 ;;
  esac
  "${PYTHON_BIN}" "${ROOT_DIR}/test/ablation/diagnose_ir_condition_ablation.py" \
      --dataset_root "${DATASET_ROOT}" \
      --vae_ckpt "${VAE_CKPT}" \
      --model_ckpt "${checkpoint}" \
      --output_dir "${candidate_dir}" \
      --split validation \
      --variants real \
      --max_samples 32 \
      --require_sample_count 32 \
      --steps 20 \
      --sampler euler \
      --seed 42 \
      --target_size 64 128 128 \
      --source_pc_range 0 -20 -6 120 20 10 \
      --model_pc_range 0 -20 -6 40 20 10 \
      --occ_threshold 0.99 \
      --target_threshold 0.5
  SELECT_ARGS+=(--candidate "${name}" "${checkpoint}" "${summary_csv}")
done

"${PYTHON_BIN}" "${ROOT_DIR}/test/evaluation/ldm/select_ldm_checkpoint.py" \
  "${SELECT_ARGS[@]}" \
  --output_dir "${OUTPUT_DIR}"

echo "Checkpoint validation completed: ${OUTPUT_DIR}"
