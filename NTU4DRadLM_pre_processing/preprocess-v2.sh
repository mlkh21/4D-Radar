#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zxj/catkin_ws/src/4D-Radar-Diffusion"
cd "$ROOT"

# 复用已经由 header timestamp 解包并生成严格 Radar--LiDAR 索引的 Raw；
# 本脚本只创建新的 0--80 m 输出，不覆盖 Raw 或任何旧体素数据。
RAW_ROOT="$ROOT/Data/NTU4DRadLM_Raw_p1_01_candidate"
NEW_ROOT="$ROOT/Data/NTU4DRadLM_Pre_formal_v2_80m_86p8_v1"
DEPLOY_ROOT="$ROOT/Data/NTU4DRadLM_Deploy_formal_v2_80m_86p8_v1"
SPLIT_ARTIFACT="$NEW_ROOT/temporal_split_garden_train80_purge3s_v1.json"
DATA_PROTOCOL_ARTIFACT="$NEW_ROOT/formal_data_protocol_garden_train80_purge3s_v1.json"
ARTIFACT="$ROOT/diffusion_consistency_radar/config/radar_normalization_garden_32x128x128_80m_train80_purge3s_86p8_v2.json"

DOPPLER_SCALE_MPS="86.8"

# 与当前 default_config.yaml 的正式训练网格保持一致。
TARGET_SIZE=(32 128 128)
SOURCE_PC_RANGE=(0 -20 -6 80 20 10)
MODEL_PC_RANGE=(0 -20 -6 80 20 10)

if [[ -e "$NEW_ROOT" ]]; then
    echo "错误：候选数据目录已经存在，请勿直接删除：$NEW_ROOT"
    exit 1
fi

if [[ -e "$DEPLOY_ROOT" ]]; then
    echo "错误：deployment 数据目录已经存在，请勿直接删除：$DEPLOY_ROOT"
    exit 1
fi

if [[ -e "$ARTIFACT" ]]; then
    echo "错误：normalization artifact 已经存在，拒绝覆盖：$ARTIFACT"
    exit 1
fi

if [[ ! -d "$RAW_ROOT" ]]; then
    echo "错误：header-time Raw 不存在：$RAW_ROOT"
    exit 1
fi

echo "当前磁盘空间："
df -h "$ROOT"

echo "步骤 1/8：验证复用的 header-time Raw 与严格索引"
for SCENE in garden loop3; do
    if [[ ! -d "$RAW_ROOT/$SCENE/radar_pcl" \
        || ! -d "$RAW_ROOT/$SCENE/livox_lidar" \
        || ! -d "$RAW_ROOT/$SCENE/thermal_cam_thermal_image_compressed" ]]; then
        echo "错误：header-time Raw 解包不完整：$RAW_ROOT/$SCENE"
        exit 1
    fi
    if [[ ! -f "$RAW_ROOT/$SCENE/radar_lidar_sync.csv" ]]; then
        echo "错误：严格 Radar--LiDAR sync 不存在：$RAW_ROOT/$SCENE"
        exit 1
    fi
done

PREPROCESS=(
    conda run -n Radar-Diffusion python
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
    --pc_range "${SOURCE_PC_RANGE[@]}"
    --z_min -1.0
    --x_max 80.0
    --visibility_mode preserve
    --radar_visibility_radius 2
    --doppler_radius 1
    --max_frames 0
)

echo "步骤 2/8：全量重建 garden（0--80 m）"
"${PREPROCESS[@]}" --scene garden

echo "步骤 3/8：全量重建 loop3（0--80 m）"
"${PREPROCESS[@]}" --scene loop3

echo "步骤 4/8：验证两个场景的五模态 training manifest"
conda run -n Radar-Diffusion python \
    diffusion_consistency_radar/scripts/dataset_manifest.py validate \
    --scene_dir "$NEW_ROOT/garden" \
    --expected_scene garden \
    --expected_profile training

conda run -n Radar-Diffusion python \
    diffusion_consistency_radar/scripts/dataset_manifest.py validate \
    --scene_dir "$NEW_ROOT/loop3" \
    --expected_scene loop3 \
    --expected_profile training

echo "步骤 5/8：生成唯一 garden temporal split（train 80%，purge 3 秒）"
conda run -n Radar-Diffusion python \
    diffusion_consistency_radar/scripts/build_temporal_split.py \
    --dataset_dir "$NEW_ROOT" \
    --scene garden \
    --output "$SPLIT_ARTIFACT" \
    --train_fraction 0.8 \
    --purge_seconds 3.0

echo "步骤 6/8：仅使用 split.train 生成正式 normalization artifact"
conda run -n Radar-Diffusion python \
    diffusion_consistency_radar/scripts/build_radar_normalization.py \
    --dataset_dir "$NEW_ROOT" \
    --scene garden \
    --output "$ARTIFACT" \
    --target_size "${TARGET_SIZE[@]}" \
    --source_pc_range "${SOURCE_PC_RANGE[@]}" \
    --model_pc_range "${MODEL_PC_RANGE[@]}" \
    --doppler_scale_mps "$DOPPLER_SCALE_MPS" \
    --split_artifact "$SPLIT_ARTIFACT" \
    --max_frames 0

echo "步骤 7/8：从 manifest/split/observed/标定生成 formal data protocol"
conda run -n Radar-Diffusion python \
    diffusion_consistency_radar/scripts/build_formal_data_protocol.py \
    --dataset_dir "$NEW_ROOT" \
    --scene garden \
    --split_artifact "$SPLIT_ARTIFACT" \
    --output "$DATA_PROTOCOL_ARTIFACT"

echo "步骤 8/8：从 loop3 training v2 生成严格 Radar+IR deployment v3 视图"
conda run -n Radar-Diffusion python \
    diffusion_consistency_radar/scripts/build_deployment_view.py create \
    --training_dataset_dir "$NEW_ROOT" \
    --output_dataset_dir "$DEPLOY_ROOT" \
    --calibration_dir "$ROOT/Data/config" \
    --preprocess_script "$ROOT/NTU4DRadLM_pre_processing/NTU4DRadLM_pre_processing.py" \
    --scene loop3 \
    --link_mode hardlink

echo "候选数据：$NEW_ROOT"
echo "候选 header-time Raw：$RAW_ROOT"
echo "Normalization artifact：$ARTIFACT"
echo "Temporal split artifact：$SPLIT_ARTIFACT"
echo "Formal data protocol：$DATA_PROTOCOL_ARTIFACT"
echo "Deployment dataset：$DEPLOY_ROOT"
echo "处理完成。暂时不要启动训练，请先检查并反馈最后三段输出。"
