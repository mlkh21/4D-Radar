#!/bin/bash
set -euo pipefail

# 获取脚本所在目录和项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"

# 需要分析的场景和使用的方法（可根据需要修改为 cd_2step_eval, cd_4step_eval 等）
SCENE="loop3"
METHOD="ldm_eval"

# 配置相关的输入输出路径
RADAR_VOXEL_DIR="${ROOT_DIR}/Data/NTU4DRadLM_Pre/${SCENE}/radar_voxel"
TARGET_VOXEL_DIR="${ROOT_DIR}/Data/NTU4DRadLM_Pre/${SCENE}/target_voxel"
PRED_DIR="${ROOT_DIR}/Result/inference_results/${SCENE}_${METHOD}"
OUTPUT_DIR="${ROOT_DIR}/Result/diagnosis_results/${SCENE}_${METHOD}"

echo "开始仿真结果诊断分析..."
echo "预测数据目录: ${PRED_DIR}"
echo "输出结果目录: ${OUTPUT_DIR}"

# 运行质量诊断脚本
python "${PROJECT_DIR}/scripts/diagnose_generation_quality.py" \
    --radar_voxel_dir "${RADAR_VOXEL_DIR}" \
    --target_voxel_dir "${TARGET_VOXEL_DIR}" \
    --pred_dir "${PRED_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_files 20 \
    --pred_kind "pcl" \
    --occ_threshold 0.1

echo "诊断分析完成！结果存放在: ${OUTPUT_DIR}"
