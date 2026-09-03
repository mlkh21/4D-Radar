#!/usr/bin/env bash
# 文件功能：在全新目录中生成带 ROS layout、字段语义和 dB SNR normalization 的 formal-v2.1 数据链。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONDA_ENV="${CONDA_ENV:-Radar-Diffusion}"
INPUT_ROOT="${INPUT_ROOT:-$ROOT/Data/NTU4DRadLM}"
RAW_ROOT="${FORMAL_V2_RAW_ROOT:-$ROOT/Data/NTU4DRadLM_Raw_formal_v2_1_80m_86p8_db_snr_v1}"
NEW_ROOT="${FORMAL_V2_PREPROCESSED_ROOT:-$ROOT/Data/NTU4DRadLM_Pre_formal_v2_1_80m_86p8_db_snr_v1}"
DEPLOY_ROOT="${FORMAL_V2_DEPLOY_ROOT:-$ROOT/Data/NTU4DRadLM_Deploy_formal_v2_1_80m_86p8_db_snr_v1}"
RADAR_FIELD_SCHEMA="${RADAR_FIELD_SCHEMA:-}"

SPLIT_ARTIFACT="$NEW_ROOT/temporal_split_garden_train80_purge3s_v1.json"
DATA_PROTOCOL_ARTIFACT="$NEW_ROOT/formal_data_protocol_garden_train80_purge3s_v4.json"
NORMALIZATION_ARTIFACT="${FORMAL_V2_NORMALIZATION_ARTIFACT:-$ROOT/diffusion_consistency_radar/config/radar_normalization_garden_32x128x128_80m_train80_purge3s_86p8_db_snr_v2.json}"

DOPPLER_SCALE_MPS="${DOPPLER_SCALE_MPS:-86.8}"
TARGET_SIZE=(32 128 128)
PC_RANGE=(0 -20 -6 80 20 10)
PY=(conda run --no-capture-output -n "$CONDA_ENV" python)

# 所有会创建输出的步骤之前完成输入、schema、标定和输出隔离门禁。
if [[ ! -d "$INPUT_ROOT" ]]; then
    echo "错误：原始 rosbag 数据目录不存在：$INPUT_ROOT"
    exit 1
fi
if [[ -z "$RADAR_FIELD_SCHEMA" || ! -f "$RADAR_FIELD_SCHEMA" ]]; then
    echo "错误：formal-v2.1 必须通过 RADAR_FIELD_SCHEMA 指定带权威证据的普通 JSON 文件。"
    exit 1
fi
for file in \
    "$ROOT/Data/config/calib_radar_to_livox.txt" \
    "$ROOT/Data/config/calib_radar_to_thermal.txt" \
    "$ROOT/Data/config/calib_livox_to_thermal.txt" \
    "$ROOT/Data/config/calib_cam_thermal.txt"; do
    if [[ ! -f "$file" ]]; then
        echo "错误：缺少标定文件：$file"
        exit 1
    fi
done
for output in \
    "$RAW_ROOT" \
    "$NEW_ROOT" \
    "$DEPLOY_ROOT" \
    "$NORMALIZATION_ARTIFACT"; do
    if [[ -e "$output" ]]; then
        echo "错误：发现已有输出，拒绝覆盖：$output"
        exit 1
    fi
done

"${PY[@]}" -c \
    'import sys; from diffusion_consistency_radar.radar_field_schema import load_radar_field_schema_artifact; load_radar_field_schema_artifact(sys.argv[1], require_verified=True); print("Radar field schema v2 权威证据校验通过")' \
    "$RADAR_FIELD_SCHEMA"

echo "当前磁盘空间："
df -h "$ROOT"

echo "步骤 1/9：从原始 bag 解包并生成逐场景 ROS layout/extraction receipt"
"${PY[@]}" NTU4DRadLM_pre_processing/unpack_rosbag.py \
    --input "$INPUT_ROOT" \
    --output "$RAW_ROOT"

echo "步骤 2/9：生成严格 Radar--LiDAR 时间索引"
"${PY[@]}" NTU4DRadLM_pre_processing/NTU4DRadLM_timestamp_index.py \
    --directory "$RAW_ROOT" \
    --radar_lidar_max_delta 0.045 \
    --skip_unmatched \
    --max_rejected_fraction 0.01

PREPROCESS=(
    "${PY[@]}"
    NTU4DRadLM_pre_processing/NTU4DRadLM_pre_processing.py
    --raw_data_path "$RAW_ROOT"
    --index_path "$RAW_ROOT"
    --output_path "$NEW_ROOT"
    --calib_path "$ROOT/Data/config/calib_radar_to_livox.txt"
    --radar_to_thermal_path "$ROOT/Data/config/calib_radar_to_thermal.txt"
    --lidar_to_thermal_path "$ROOT/Data/config/calib_livox_to_thermal.txt"
    --thermal_intrinsics_path "$ROOT/Data/config/calib_cam_thermal.txt"
    --radar_field_schema "$RADAR_FIELD_SCHEMA"
    --require_verified_radar_field_schema
    --require_complete_extraction_receipt
    --align_to lidar
    --velocity_mode none
    --velocity_frame radar
    --radar_lidar_max_delta 0.045
    --radar_ir_max_delta 0.025
    --pc_range "${PC_RANGE[@]}"
    --z_min -1.0
    --x_max 80.0
    --visibility_mode preserve
    --radar_visibility_radius 2
    --doppler_radius 1
    --max_frames 0
)

echo "步骤 3/9：全量预处理 garden 到全新 formal-v2.1 目录"
"${PREPROCESS[@]}" --scene garden

echo "步骤 4/9：全量预处理 loop3 到全新 formal-v2.1 目录"
"${PREPROCESS[@]}" --scene loop3

echo "步骤 5/9：验证两个场景的 training manifest"
for scene in garden loop3; do
    "${PY[@]}" diffusion_consistency_radar/scripts/dataset_manifest.py validate \
        --scene_dir "$NEW_ROOT/$scene" \
        --expected_scene "$scene" \
        --expected_profile training
done

echo "步骤 6/9：生成唯一 garden temporal split"
"${PY[@]}" diffusion_consistency_radar/scripts/build_temporal_split.py \
    --dataset_dir "$NEW_ROOT" \
    --scene garden \
    --output "$SPLIT_ARTIFACT" \
    --train_fraction 0.8 \
    --purge_seconds 3.0

echo "步骤 7/9：仅使用 split.train 生成 dB SNR normalization v2"
"${PY[@]}" diffusion_consistency_radar/scripts/build_radar_normalization.py \
    --dataset_dir "$NEW_ROOT" \
    --scene garden \
    --output "$NORMALIZATION_ARTIFACT" \
    --target_size "${TARGET_SIZE[@]}" \
    --source_pc_range "${PC_RANGE[@]}" \
    --model_pc_range "${PC_RANGE[@]}" \
    --doppler_scale_mps "$DOPPLER_SCALE_MPS" \
    --intensity_transform identity_robust_zscore \
    --intensity_quantity signal_to_noise_ratio \
    --intensity_unit dB \
    --split_artifact "$SPLIT_ARTIFACT" \
    --max_frames 0

echo "步骤 8/9：生成绑定 schema v2 的 formal-data-v4 身份 artifact"
"${PY[@]}" diffusion_consistency_radar/scripts/build_formal_data_protocol.py \
    --dataset_dir "$NEW_ROOT" \
    --scene garden \
    --split_artifact "$SPLIT_ARTIFACT" \
    --output "$DATA_PROTOCOL_ARTIFACT" \
    --protocol_version v4

echo "步骤 9/9：从 loop3 training 数据生成严格 Radar+IR deployment 视图"
"${PY[@]}" diffusion_consistency_radar/scripts/build_deployment_view.py create \
    --training_dataset_dir "$NEW_ROOT" \
    --output_dataset_dir "$DEPLOY_ROOT" \
    --calibration_dir "$ROOT/Data/config" \
    --preprocess_script "$ROOT/NTU4DRadLM_pre_processing/NTU4DRadLM_pre_processing.py" \
    --scene loop3 \
    --link_mode hardlink

echo "候选数据：$NEW_ROOT"
echo "候选 Raw：$RAW_ROOT"
echo "Normalization artifact：$NORMALIZATION_ARTIFACT"
echo "Temporal split artifact：$SPLIT_ARTIFACT"
echo "Formal data protocol：$DATA_PROTOCOL_ARTIFACT"
echo "Deployment dataset：$DEPLOY_ROOT"
echo "处理完成。暂时不要启动正式训练，请先执行 preflight 和短 smoke。"
