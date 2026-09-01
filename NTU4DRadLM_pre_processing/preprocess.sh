#!/bin/bash

set -euo pipefail

ROOT=/home/ps/zxj_workspace/src/4D-Radar
cd "$ROOT"

INPUT_ROOT="$ROOT/Data/NTU4DRadLM"
RAW_ROOT="$ROOT/Data/NTU4DRadLM_Raw_p1_01_candidate"
NEW_ROOT="$ROOT/Data/NTU4DRadLM_Pre_formal_v2_80m_86p8_v1"
DEPLOY_ROOT="$ROOT/Data/NTU4DRadLM_Deploy_formal_v2_80m_86p8_v1"

SPLIT_ARTIFACT="$NEW_ROOT/temporal_split_garden_train80_purge3s_v1.json"
DATA_PROTOCOL_ARTIFACT="$NEW_ROOT/formal_data_protocol_garden_train80_purge3s_v1.json"
ARTIFACT="$ROOT/diffusion_consistency_radar/config/radar_normalization_garden_32x128x128_80m_train80_purge3s_86p8_server_v1.json"

for path in "$RAW_ROOT" "$NEW_ROOT" "$DEPLOY_ROOT" "$ARTIFACT"; do
    if [[ -e "$path" ]]; then
        echo "错误：发现已有输出，拒绝覆盖：$path"
        exit 1
    fi
done

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

df -h "$ROOT"

PY=(conda run --no-capture-output -n Radar python)

echo "步骤 1/9：从原始 bag 按 header timestamp 解包"
"${PY[@]}" NTU4DRadLM_pre_processing/unpack_rosbag.py \
    --input "$INPUT_ROOT" \
    --output "$RAW_ROOT"

echo "步骤 2/9：生成严格 Radar-LiDAR 时间索引"
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
    --align_to lidar
    --velocity_mode none
    --velocity_frame radar
    --radar_lidar_max_delta 0.045
    --radar_ir_max_delta 0.025
    --pc_range 0 -20 -6 80 20 10
    --z_min -1.0
    --x_max 80.0
    --visibility_mode preserve
    --radar_visibility_radius 2
    --doppler_radius 1
    --max_frames 0
)

echo "步骤 3/9：全量预处理 garden"
"${PREPROCESS[@]}" --scene garden

echo "步骤 4/9：全量预处理 loop3"
"${PREPROCESS[@]}" --scene loop3

echo "步骤 5/9：验证 training manifest"
for scene in garden loop3; do
    "${PY[@]}" diffusion_consistency_radar/scripts/dataset_manifest.py validate \
        --scene_dir "$NEW_ROOT/$scene" \
        --expected_scene "$scene" \
        --expected_profile training
done

echo "步骤 6/9：生成 garden temporal split"
"${PY[@]}" diffusion_consistency_radar/scripts/build_temporal_split.py \
    --dataset_dir "$NEW_ROOT" \
    --scene garden \
    --output "$SPLIT_ARTIFACT" \
    --train_fraction 0.8 \
    --purge_seconds 3.0

echo "步骤 7/9：使用 split.train 生成 normalization artifact"
"${PY[@]}" diffusion_consistency_radar/scripts/build_radar_normalization.py \
    --dataset_dir "$NEW_ROOT" \
    --scene garden \
    --output "$ARTIFACT" \
    --target_size 32 128 128 \
    --source_pc_range 0 -20 -6 80 20 10 \
    --model_pc_range 0 -20 -6 80 20 10 \
    --doppler_scale_mps 86.8 \
    --split_artifact "$SPLIT_ARTIFACT" \
    --max_frames 0

echo "步骤 8/9：生成 formal data protocol"
"${PY[@]}" diffusion_consistency_radar/scripts/build_formal_data_protocol.py \
    --dataset_dir "$NEW_ROOT" \
    --scene garden \
    --split_artifact "$SPLIT_ARTIFACT" \
    --output "$DATA_PROTOCOL_ARTIFACT"

echo "步骤 9/9：生成 loop3 deployment view"
"${PY[@]}" diffusion_consistency_radar/scripts/build_deployment_view.py create \
    --training_dataset_dir "$NEW_ROOT" \
    --output_dataset_dir "$DEPLOY_ROOT" \
    --calibration_dir "$ROOT/Data/config" \
    --preprocess_script "$ROOT/NTU4DRadLM_pre_processing/NTU4DRadLM_pre_processing.py" \
    --scene loop3 \
    --link_mode hardlink

echo "预处理完成"
echo "Training data: $NEW_ROOT"
echo "Deployment data: $DEPLOY_ROOT"
echo "Normalization artifact:"
sha256sum "$ARTIFACT"
