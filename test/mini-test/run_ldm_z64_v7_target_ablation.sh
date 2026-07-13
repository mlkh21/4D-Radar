#!/bin/bash
# 一键运行 Z=64 LDM v7 的 32 帧 target-aware IR 条件消融。
#
# 同一验证帧分别输入真实 IR、全零 IR 和 mock IR，并比较它们相对 LiDAR target
# 的占用、BEV 与竖向结构指标。脚本只做推理诊断，不训练或覆盖 checkpoint。

set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SELF_DIR}/../.." && pwd)"

V7_DIR="${V7_DIR:-test/result/ldm/ablation/ldm_near40_500_z64_v7_ir}"
DATASET_ROOT="${DATASET_ROOT:-Data/NTU4DRadLM_Pre_sensor_aware}"
OUTPUT_DIR="${OUTPUT_DIR:-${V7_DIR}/ir_target_ablation_32}"
ABLATION_MAX_SAMPLES="${ABLATION_MAX_SAMPLES:-32}"
ABLATION_STEPS="${ABLATION_STEPS:-20}"
OCC_THRESHOLD="${OCC_THRESHOLD:-0.99}"
TARGET_THRESHOLD="${TARGET_THRESHOLD:-0.5}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"

cd "${ROOT_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${CUDA_DEVICES}}"

conda run -n Radar-Diffusion python test/ablation/diagnose_ir_condition_ablation.py \
  --dataset_root "${DATASET_ROOT}" \
  --vae_ckpt "${V7_DIR}/vae/vae_best.pt" \
  --model_ckpt "${V7_DIR}/ldm/ldm_best.pt" \
  --output_dir "${OUTPUT_DIR}" \
  --split validation \
  --model_type ldm \
  --max_samples "${ABLATION_MAX_SAMPLES}" \
  --steps "${ABLATION_STEPS}" \
  --sampler euler \
  --seed 42 \
  --occ_threshold "${OCC_THRESHOLD}" \
  --target_threshold "${TARGET_THRESHOLD}" \
  --target_size 64 128 128 \
  --source_pc_range 0 -20 -6 120 20 10 \
  --model_pc_range 0 -20 -6 40 20 10

echo "Target-aware IR ablation completed: ${OUTPUT_DIR}"
echo "Report: ${OUTPUT_DIR}/ir_condition_ablation_report.md"
