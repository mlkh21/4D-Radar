#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""文件功能：从已保存概率体素重导出整个任务域的预测点云，不运行模型或射线裁剪。"""

import argparse
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from diffusion_consistency_radar.observed_mask import load_task_spatial_domain, task_domain_zxy
from diffusion_consistency_radar.scripts.evaluate_saved_predictions import (
    _discover_prediction_frames, _load_prediction, _resolve_parameters,
    _validate_output_directory, _write_json_atomic,
)
from diffusion_consistency_radar.scripts.inference import voxel_to_pointcloud


def export_saved_pointclouds(pred_voxel_dir, preprocess_policy_path, output_dir, max_files=0):
    """只写 fresh 派生目录；保留源 voxel/mask/metadata，unknown 不转成 free。"""
    _validate_output_directory(output_dir)
    if int(max_files) < 0:
        raise ValueError("max_files 必须非负")
    resolved = _resolve_parameters(pred_voxel_dir, "", None, None, None, None)
    policy = load_task_spatial_domain(preprocess_policy_path, resolved["source_pc_range"])
    stored_policy = resolved["metadata"].get("task_spatial_domain")
    if stored_policy is not None and stored_policy != policy:
        raise ValueError("导出 policy 与源 prediction task spatial domain 不一致")
    task = task_domain_zxy(resolved["target_size"], resolved["model_pc_range"], policy)
    predictions, _, _ = _discover_prediction_frames(pred_voxel_dir)
    frame_ids = sorted(predictions)
    if int(max_files):
        frame_ids = frame_ids[:int(max_files)]
    for frame_id in frame_ids:
        _load_prediction(predictions[frame_id], resolved["target_size"])

    os.makedirs(output_dir, exist_ok=True)
    records = []
    for frame_id in frame_ids:
        voxel = _load_prediction(predictions[frame_id], resolved["target_size"])
        points, _ = voxel_to_pointcloud(
            voxel, resolved["voxel_size"], resolved["model_pc_range"],
            occ_threshold=resolved["occ_threshold"], empty_fallback_topk=0,
            valid_mask=task,
        )
        file_name = f"{frame_id}_pcl.npy"
        with open(os.path.join(output_dir, file_name), "xb") as handle:
            np.save(handle, points, allow_pickle=False)
        records.append({
            "frame_id": frame_id, "file": file_name,
            "point_count": int(points.shape[0]), "valid_voxels": int(task.sum()),
            "unmasked_occupied_voxels": int((voxel[0] >= resolved["occ_threshold"]).sum()),
        })
    report = {
        "protocol": "saved_prediction_task_pointcloud_export_v2",
        "pointcloud_domain": "full_task_domain_prediction",
        "formal_evaluation": False,
        "note": "整个任务域的模型预测，不代表传感器已确认占用；不把 unknown 当 free，不证明生成质量提高",
        "prediction_unchanged": True,
        "source_prediction_dir": os.path.abspath(pred_voxel_dir),
        "preprocess_policy_path": os.path.abspath(preprocess_policy_path),
        "task_spatial_domain": policy,
        "target_size": list(resolved["target_size"]),
        "model_pc_range": list(resolved["model_pc_range"]),
        "voxel_size": list(resolved["voxel_size"]),
        "occ_threshold": resolved["occ_threshold"],
        "frame_count": len(records), "frames": records,
    }
    _write_json_atomic(os.path.join(output_dir, "pointcloud_export.json"), report)
    return report


def main():
    parser = argparse.ArgumentParser(description="重导出已保存预测，不需要 GPU、checkpoint 或 LiDAR")
    parser.add_argument("--pred_voxel_dir", required=True)
    parser.add_argument("--preprocess_policy_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_files", type=int, default=0)
    args = parser.parse_args()
    try:
        report = export_saved_pointclouds(**vars(args))
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"frame_count": report["frame_count"], "output_dir": args.output_dir,
                      "prediction_unchanged": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
