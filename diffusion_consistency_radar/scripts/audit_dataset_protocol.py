#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""审计预处理数据集的 IR、标定和体素协议覆盖情况。"""

import argparse
import csv
import json
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cm.dataset_loader import CalibrationProvider  # noqa: E402

DEFAULT_PC_RANGE = (0.0, -20.0, -6.0, 120.0, 20.0, 10.0)
DEFAULT_TARGET_SIZE = (32, 128, 128)
DEFAULT_K = np.array(
    [[457.2, 0.0, 323.1], [0.0, 457.9, 242.5], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)
MOCK_R_CAM_TO_LIDAR = np.array(
    [[0.012, -0.999, -0.015], [0.024, -0.015, 0.999], [-0.999, -0.012, 0.024]],
    dtype=np.float32,
)
MOCK_T_CAM_TO_LIDAR = np.array([0.01, 0.0, 0.0], dtype=np.float32)


def _list_voxel_files(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    files = [f for f in os.listdir(folder) if f.endswith(".npy") or f.endswith(".npz")]
    return sorted(f for f in files if not f.endswith("_pcl.npy"))


def _has_ir(scene_dir: str, voxel_file: str) -> bool:
    stem = os.path.splitext(voxel_file)[0]
    return os.path.exists(os.path.join(scene_dir, "ir_image", f"{stem}_ir.npy"))


def _has_compatible_ir(scene_dir: str, voxel_file: str) -> bool:
    stem = os.path.splitext(voxel_file)[0]
    candidates = [
        os.path.join(scene_dir, "ir_image", f"{stem}_ir.npy"),
        os.path.join(scene_dir, "ir_image", f"{stem}.npy"),
        os.path.join(scene_dir, "ir_image", f"{stem}.npz"),
    ]
    return any(os.path.exists(path) for path in candidates)


def _frustum_voxel_ratio(r_mat, t_vec, k_mat, target_size, pc_range, img_shape=(480, 640)) -> float:
    """估算 IR 投影视锥覆盖的体素比例，用于协议审计而非训练。"""
    nz, nx, ny = [int(v) for v in target_size]
    xmin, ymin, zmin, xmax, ymax, zmax = [float(v) for v in pc_range]
    xs = xmin + (np.arange(nx, dtype=np.float32) + 0.5) * ((xmax - xmin) / max(nx, 1))
    ys = ymin + (np.arange(ny, dtype=np.float32) + 0.5) * ((ymax - ymin) / max(ny, 1))
    zs = zmin + (np.arange(nz, dtype=np.float32) + 0.5) * ((zmax - zmin) / max(nz, 1))
    grid_z, grid_x, grid_y = np.meshgrid(zs, xs, ys, indexing="ij")
    pts = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)
    pts_cam = pts @ r_mat.T + t_vec.reshape(1, 3)
    depth = pts_cam[:, 2]
    valid_depth = depth > 0.1
    safe_depth = np.maximum(depth, 1e-6)
    pixels = pts_cam @ k_mat.T
    u = pixels[:, 0] / safe_depth
    v = pixels[:, 1] / safe_depth
    h_img, w_img = img_shape
    valid = valid_depth & (u >= 0.0) & (u <= w_img - 1) & (v >= 0.0) & (v <= h_img - 1)
    return float(valid.mean()) if valid.size else 0.0


def audit_scene(dataset_root: str, scene: str) -> Dict[str, object]:
    scene_dir = os.path.join(dataset_root, scene)
    radar_dir = os.path.join(scene_dir, "radar_voxel")
    target_dir = os.path.join(scene_dir, "target_voxel")
    radar_files = _list_voxel_files(radar_dir)
    target_files = _list_voxel_files(target_dir)
    ir_count = sum(1 for f in radar_files if _has_ir(scene_dir, f))
    compatible_ir_count = sum(1 for f in radar_files if _has_compatible_ir(scene_dir, f))
    policy_path = os.path.join(scene_dir, "preprocess_policy.json")
    policy = {}
    if os.path.exists(policy_path):
        with open(policy_path, "r", encoding="utf-8") as f:
            policy = json.load(f)

    r_mat, t_vec, k_mat, calib_meta = CalibrationProvider(dataset_root).load_with_metadata()
    policy_target_size = policy.get("target_size", policy.get("voxel_shape", DEFAULT_TARGET_SIZE))
    policy_pc_range = policy.get("model_pc_range", policy.get("pc_range", DEFAULT_PC_RANGE))
    r_np = r_mat.detach().cpu().numpy()
    t_np = t_vec.detach().cpu().numpy()
    k_np = k_mat.detach().cpu().numpy() if k_mat is not None else DEFAULT_K
    if calib_meta["is_mock_calib"]:
        # 与 NTU4DRadLM_VoxelDataset._get_mock_calibration 的训练路径保持一致。
        r_np = MOCK_R_CAM_TO_LIDAR
        t_np = MOCK_T_CAM_TO_LIDAR
        k_np = DEFAULT_K
    frustum_ratio = _frustum_voxel_ratio(
        r_np,
        t_np,
        k_np,
        policy_target_size,
        policy_pc_range,
    )
    frame_count = len(radar_files)
    ir_coverage = (ir_count / frame_count) if frame_count else 0.0
    return {
        "scene": scene,
        "radar_frames": frame_count,
        "target_frames": len(target_files),
        "ir_frames": ir_count,
        "compatible_ir_frames": compatible_ir_count,
        "ir_coverage": ir_coverage,
        "compatible_ir_coverage": (compatible_ir_count / frame_count) if frame_count else 0.0,
        "mock_ir_ratio": 1.0 - ir_coverage if frame_count else 0.0,
        "has_preprocess_policy": bool(policy),
        "align_to": policy.get("align_to", ""),
        "is_mock_calib": bool(calib_meta["is_mock_calib"]),
        "mock_calib_ratio": 1.0 if calib_meta["is_mock_calib"] and frame_count else 0.0,
        "calib_source": calib_meta["calib_source"],
        "calib_is_thermal": bool(calib_meta["calib_is_thermal"]),
        "has_thermal_calib": bool(calib_meta["has_thermal_calib"]),
        "has_livox_calib": bool(calib_meta["has_livox_calib"]),
        "calib_fallback_reason": calib_meta["calib_fallback_reason"],
        "ir_frustum_voxel_ratio": frustum_ratio,
    }


def write_report(rows: List[Dict[str, object]], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "dataset_protocol_audit.csv")
    md_path = os.path.join(output_dir, "dataset_protocol_audit.md")
    headers = [
        "scene",
        "radar_frames",
        "target_frames",
        "ir_frames",
        "compatible_ir_frames",
        "ir_coverage",
        "compatible_ir_coverage",
        "mock_ir_ratio",
        "has_preprocess_policy",
        "align_to",
        "is_mock_calib",
        "mock_calib_ratio",
        "calib_source",
        "calib_is_thermal",
        "has_thermal_calib",
        "has_livox_calib",
        "calib_fallback_reason",
        "ir_frustum_voxel_ratio",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Dataset Protocol Audit\n\n")
        f.write("| scene | radar | target | IR coverage | mock IR | calib source | mock calib | frustum |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |\n")
        for row in rows:
            f.write(
                f"| {row['scene']} | {row['radar_frames']} | {row['target_frames']} | "
                f"{float(row['ir_coverage']):.3f} | {float(row['mock_ir_ratio']):.3f} | "
                f"{row['calib_source']} | {row['is_mock_calib']} | "
                f"{float(row['ir_frustum_voxel_ratio']):.3f} |\n"
            )
    print(f"Saved audit CSV to: {csv_path}")
    print(f"Saved audit report to: {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit dataset IR/calibration/preprocess protocol coverage")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="test/result/comparison/dataset_protocol_audit_v7",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已有的审计输出；默认拒绝覆盖非空目录。",
    )
    parser.add_argument("--scenes", type=str, default="", help="Comma-separated scene list. Defaults to all scene folders.")
    args = parser.parse_args()

    if os.path.exists(args.output_dir):
        if not os.path.isdir(args.output_dir):
            parser.error(f"output_dir 不是目录: {args.output_dir}")
        if os.listdir(args.output_dir) and not args.overwrite:
            parser.error(
                f"output_dir 已存在且非空，默认拒绝覆盖: {args.output_dir}; "
                "如确认覆盖请显式传入 --overwrite"
            )

    if args.scenes:
        scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    else:
        scenes = sorted(
            d for d in os.listdir(args.dataset_root)
            if os.path.isdir(os.path.join(args.dataset_root, d))
        )
    rows = [audit_scene(args.dataset_root, scene) for scene in scenes]
    write_report(rows, args.output_dir)


if __name__ == "__main__":
    main()
