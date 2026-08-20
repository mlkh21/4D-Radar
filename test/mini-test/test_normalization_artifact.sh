#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zxj/catkin_ws/src/4D-Radar-Diffusion"
DATASET="$ROOT/Data/NTU4DRadLM_Pre_sensor_aware_p1_04_candidate"
ARTIFACT="$ROOT/diffusion_consistency_radar/config/radar_normalization_garden_32x128x128_full120_86p8_v1.json"

cd "$ROOT"

echo "步骤 1/3：验证 garden manifest"
conda run -n Radar-Diffusion python \
    diffusion_consistency_radar/scripts/dataset_manifest.py validate \
    --scene_dir "$DATASET/garden" \
    --expected_scene garden

echo "步骤 2/3：验证 loop3 manifest"
conda run -n Radar-Diffusion python \
    diffusion_consistency_radar/scripts/dataset_manifest.py validate \
    --scene_dir "$DATASET/loop3" \
    --expected_scene loop3

echo "步骤 3/3：验证 artifact 并加载一个正式训练样本"
conda run -n Radar-Diffusion python -c '
import json
import sys
import torch

from diffusion_consistency_radar.radar_normalization import (
    load_radar_normalization_artifact,
)
from diffusion_consistency_radar.cm.dataset_loader import (
    NTU4DRadLM_VoxelDataset,
)

dataset_root, artifact_path = sys.argv[1:3]
target_size = (32, 128, 128)
pc_range = (0, -20, -6, 120, 20, 10)

spec, digest = load_radar_normalization_artifact(
    artifact_path,
    target_size=target_size,
    source_pc_range=pc_range,
    model_pc_range=pc_range,
    doppler_scale_mps=86.8,
    require_formal=True,
)

dataset = NTU4DRadLM_VoxelDataset(
    dataset_root,
    split="train",
    return_path=True,
    use_augmentation=False,
    sequence_length=1,
    target_size=target_size,
    source_pc_range=pc_range,
    model_pc_range=pc_range,
    radar_normalization=spec,
    radar_normalization_sha256=digest,
)

target, radar, metadata, path = dataset[0]

assert len(dataset) == 4013
assert tuple(target.shape) == (4, 32, 128, 128)
assert tuple(radar.shape) == (4, 32, 128, 128)
assert torch.isfinite(target).all()
assert torch.isfinite(radar).all()
assert metadata["radar_normalization_sha256"] == digest
assert not bool(metadata["is_mock_ir"])
assert not bool(metadata["is_mock_calib"])

print(json.dumps({
    "status": "PASS",
    "train_samples": len(dataset),
    "sample_path": path,
    "target_shape": list(target.shape),
    "radar_shape": list(radar.shape),
    "radar_occupied_voxels": int((radar[0] > 0).sum()),
    "artifact_sha256": digest,
    "real_ir": True,
    "real_calibration": True,
}, ensure_ascii=False, indent=2))
' "$DATASET" "$ARTIFACT"

echo "正式训练输入验收完成。"