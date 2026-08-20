#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zxj/catkin_ws/src/4D-Radar-Diffusion"
cd "$ROOT"

BAG_ROOT="$ROOT/Data/NTU4DRadLM"
RAW_ROOT="$ROOT/Data/NTU4DRadLM_Raw_p1_01_candidate"
NEW_ROOT="$ROOT/Data/NTU4DRadLM_Pre_sensor_aware_p1_04_candidate"
ARTIFACT="$ROOT/diffusion_consistency_radar/config/radar_normalization_garden_32x128x128_full120_86p8_v1.json"

DOPPLER_SCALE_MPS="86.8"

# 与当前 default_config.yaml 的正式训练网格保持一致。
TARGET_SIZE=(32 128 128)
SOURCE_PC_RANGE=(0 -20 -6 120 20 10)
MODEL_PC_RANGE=(0 -20 -6 120 20 10)

# 新 Raw 与体素都采用候选目录，防止覆盖旧 receipt-time 数据和旧体素。
if [[ -e "$RAW_ROOT" ]]; then
    echo "错误：候选 Raw 目录已经存在，请勿直接删除：$RAW_ROOT"
    exit 1
fi

if [[ -e "$NEW_ROOT" ]]; then
    echo "错误：候选数据目录已经存在，请勿直接删除：$NEW_ROOT"
    exit 1
fi

if [[ -e "$ARTIFACT" ]]; then
    echo "错误：normalization artifact 已经存在，拒绝覆盖：$ARTIFACT"
    exit 1
fi

echo "当前磁盘空间："
df -h "$ROOT"

echo "步骤 1/6：从 rosbag 重新解包 header-time Raw 数据"
conda run -n Radar-Diffusion python \
    NTU4DRadLM_pre_processing/unpack_rosbag.py \
    --input "$BAG_ROOT" \
    --output "$RAW_ROOT"

for SCENE in garden loop3; do
    if [[ ! -d "$RAW_ROOT/$SCENE/radar_pcl" \
        || ! -d "$RAW_ROOT/$SCENE/livox_lidar" \
        || ! -d "$RAW_ROOT/$SCENE/thermal_cam_thermal_image_compressed" ]]; then
        echo "错误：header-time Raw 解包不完整：$RAW_ROOT/$SCENE"
        exit 1
    fi
done

echo "步骤 2/6：生成严格 Radar-LiDAR 时间索引"
conda run -n Radar-Diffusion python \
    NTU4DRadLM_pre_processing/NTU4DRadLM_timestamp_index.py \
    --directory "$RAW_ROOT" \
    --radar_lidar_max_delta 0.045 \
    --skip_unmatched \
    --max_rejected_fraction 0.01

PREPROCESS=(
    conda run -n Radar-Diffusion python
    NTU4DRadLM_pre_processing/NTU4DRadLM_pre_processing.py
    --raw_data_path "$RAW_ROOT"
    --index_path "$RAW_ROOT"
    --output_path "$NEW_ROOT"
    --calib_path "$ROOT/Data/config/calib_radar_to_livox.txt"
    --align_to lidar
    --velocity_mode none
    --velocity_frame radar
    --radar_lidar_max_delta 0.045
    --radar_ir_max_delta 0.025
    --dt_sync 0.002
    --pc_range "${SOURCE_PC_RANGE[@]}"
    --z_min -1.0
    --x_max 80.0
    --visibility_mode preserve
    --radar_visibility_radius 2
    --doppler_radius 1
    --max_frames 0
)

echo "步骤 3/6：全量重建 garden"
"${PREPROCESS[@]}" --scene garden

echo "步骤 4/6：全量重建 loop3"
"${PREPROCESS[@]}" --scene loop3

echo "步骤 5/6：验证两个场景的严格 manifest"
conda run -n Radar-Diffusion python \
    diffusion_consistency_radar/scripts/dataset_manifest.py validate \
    --scene_dir "$NEW_ROOT/garden" \
    --expected_scene garden

conda run -n Radar-Diffusion python \
    diffusion_consistency_radar/scripts/dataset_manifest.py validate \
    --scene_dir "$NEW_ROOT/loop3" \
    --expected_scene loop3

echo "步骤 6/6：仅使用训练场景 garden 生成正式 normalization artifact"
conda run -n Radar-Diffusion python \
    diffusion_consistency_radar/scripts/build_radar_normalization.py \
    --dataset_dir "$NEW_ROOT" \
    --scene garden \
    --output "$ARTIFACT" \
    --target_size "${TARGET_SIZE[@]}" \
    --source_pc_range "${SOURCE_PC_RANGE[@]}" \
    --model_pc_range "${MODEL_PC_RANGE[@]}" \
    --doppler_scale_mps "$DOPPLER_SCALE_MPS" \
    --max_frames 0

echo "候选数据：$NEW_ROOT"
echo "候选 header-time Raw：$RAW_ROOT"
echo "Normalization artifact：$ARTIFACT"
echo "处理完成。暂时不要启动训练，请先检查并反馈最后三段输出。"
