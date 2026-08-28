#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""只读审计 0--80/80--120 m 的 LiDAR 密度、监督保留、可见域和标定叠加。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion_consistency_radar.cm.dataset_loader import CalibrationProvider  # noqa: E402


PC_RANGE = (0.0, -20.0, -6.0, 120.0, 20.0, 10.0)
RANGE_BANDS = (("near_0_80", 0.0, 80.0), ("far_80_120", 80.0, 120.0))


def evenly_spaced_indices(total: int, limit: int) -> List[int]:
    """确定性均匀抽样；limit<=0 或不小于总数时返回全集。"""
    if total <= 0:
        return []
    if limit <= 0 or limit >= total:
        return list(range(total))
    return np.unique(np.linspace(0, total - 1, limit, dtype=np.int64)).tolist()


def list_frame_files(directory: str, suffixes=(".npz", ".npy")) -> Dict[str, str]:
    """按六位 frame stem 建立普通文件映射，拒绝符号链接。"""
    if os.path.islink(directory) or not os.path.isdir(directory):
        raise ValueError(f"数据目录必须是普通目录: {directory}")
    result: Dict[str, str] = {}
    for name in sorted(os.listdir(directory)):
        if not name.endswith(suffixes):
            continue
        path = os.path.join(directory, name)
        if os.path.islink(path) or not os.path.isfile(path):
            raise ValueError(f"审计输入必须是普通文件: {path}")
        result[os.path.splitext(name)[0]] = path
    return result


def load_sparse_occupied(path: str) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """只读取 occupied 坐标，不构造四通道稠密体素。"""
    if not path.endswith(".npz"):
        raise ValueError(f"全场景审计要求稀疏 NPZ，收到: {path}")
    with np.load(path) as data:
        if not {"coords", "features", "shape"}.issubset(data.files):
            raise ValueError(f"稀疏体素字段不完整: {path}")
        coords = np.asarray(data["coords"], dtype=np.int64)
        features = np.asarray(data["features"])
        shape_raw = tuple(int(value) for value in np.asarray(data["shape"]).reshape(-1)[:3])
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] != features.shape[0]:
        raise ValueError(f"稀疏体素 coords/features 协议无效: {path}")
    if features.ndim != 2 or features.shape[1] < 1:
        raise ValueError(f"稀疏体素 features 协议无效: {path}")
    return coords[features[:, 0] > 0.0], shape_raw


def physical_centers(
    coords: np.ndarray,
    shape: Sequence[int],
    pc_range: Sequence[float] = PC_RANGE,
) -> np.ndarray:
    """把 (X,Y,Z) 体素索引转换成物理中心坐标。"""
    if coords.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    mins = np.asarray(pc_range[:3], dtype=np.float64)
    maxs = np.asarray(pc_range[3:], dtype=np.float64)
    step = (maxs - mins) / np.asarray(shape, dtype=np.float64)
    return mins + (coords.astype(np.float64) + 0.5) * step


def count_bands_x(x_values: np.ndarray) -> Dict[str, int]:
    """统计两个冻结距离带的数量。"""
    return {
        label: int(np.count_nonzero((x_values >= low) & (x_values < high)))
        for label, low, high in RANGE_BANDS
    }


def count_points_in_grid_bands(
    points: np.ndarray,
    pc_range: Sequence[float] = PC_RANGE,
) -> Dict[str, int]:
    """仅统计正式 XYZ 体素盒内的原始点，保证与体素 occupied 口径一致。"""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("原始点云必须为至少三列的二维数组")
    bounds = np.asarray(pc_range, dtype=np.float64)
    inside_yz = (
        (points[:, 1] >= bounds[1])
        & (points[:, 1] < bounds[4])
        & (points[:, 2] >= bounds[2])
        & (points[:, 2] < bounds[5])
    )
    return count_bands_x(points[inside_yz, 0])


def flattened_far_indices(coords: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    centers = physical_centers(coords, shape)
    far = coords[(centers[:, 0] >= 80.0) & (centers[:, 0] < 120.0)]
    if far.size == 0:
        return np.empty(0, dtype=np.int64)
    flat = np.ravel_multi_index(far.T, tuple(shape))
    return np.unique(flat)


def jaccard_sorted(first: np.ndarray, second: np.ndarray) -> float:
    """计算同一体素网格内两帧 occupied 集合 Jaccard。"""
    if first.size == 0 and second.size == 0:
        return float("nan")
    intersection = np.intersect1d(first, second, assume_unique=True).size
    union = first.size + second.size - intersection
    return float(intersection / union) if union else float("nan")


def sparse_ray_band_counts(
    occupied_coords: np.ndarray,
    shape: Sequence[int],
    pc_range: Sequence[float] = PC_RANGE,
    ray_step_fraction: float = 0.5,
) -> Dict[str, int]:
    """从稀疏端点构建 bool 可见域，仅返回距离带计数。"""
    shape_array = np.asarray(shape, dtype=np.int64)
    observed = np.zeros(tuple(shape), dtype=bool)
    if occupied_coords.size == 0:
        return {label: 0 for label, _low, _high in RANGE_BANDS}
    observed[tuple(occupied_coords.T)] = True
    bounds = np.asarray(pc_range, dtype=np.float64)
    voxel_size = (bounds[3:] - bounds[:3]) / shape_array
    origin_index = np.floor((np.zeros(3) - bounds[:3]) / voxel_size).astype(np.int64)
    directions = occupied_coords - origin_index
    gcd = np.gcd.reduce(np.abs(directions), axis=1)
    gcd[gcd == 0] = 1
    directions = directions // gcd[:, None]
    centers = physical_centers(occupied_coords, shape, pc_range)
    distances = np.linalg.norm(centers, axis=1)
    order = np.argsort(distances, kind="stable")
    _unique, nearest_positions = np.unique(directions[order], axis=0, return_index=True)
    selected = order[nearest_positions]
    min_step = max(float(np.min(voxel_size)) * ray_step_fraction, 1e-6)
    for endpoint in centers[selected]:
        distance = float(np.linalg.norm(endpoint))
        steps = max(1, int(np.ceil(distance / min_step)))
        samples = endpoint[None, :] * (
            np.arange(1, steps + 1, dtype=np.float64)[:, None] / float(steps)
        )
        indices = np.floor((samples - bounds[:3]) / voxel_size).astype(np.int64)
        valid = np.all((indices >= 0) & (indices < shape_array), axis=1)
        indices = indices[valid]
        observed[indices[:, 0], indices[:, 1], indices[:, 2]] = True
    x_centers = bounds[0] + (np.arange(shape[0]) + 0.5) * voxel_size[0]
    return {
        label: int(observed[(x_centers >= low) & (x_centers < high)].sum())
        for label, low, high in RANGE_BANDS
    }


def finite_summary(values: Iterable[float]) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p10": float("nan"), "p90": float("nan")}
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
    }


def radar_lidar_far_overlap(
    radar_coords: np.ndarray,
    radar_shape: Sequence[int],
    lidar_coords: np.ndarray,
    lidar_shape: Sequence[int],
) -> Dict[str, float]:
    """在已预对齐体素中统计远距 Radar 端点到 LiDAR 的最近邻。"""
    radar = physical_centers(radar_coords, radar_shape)
    lidar = physical_centers(lidar_coords, lidar_shape)
    radar = radar[(radar[:, 0] >= 80.0) & (radar[:, 0] < 120.0)]
    lidar = lidar[(lidar[:, 0] >= 80.0) & (lidar[:, 0] < 120.0)]
    if radar.shape[0] == 0 or lidar.shape[0] == 0:
        return {"radar_count": int(radar.shape[0]), "lidar_count": int(lidar.shape[0]), "match_1m": float("nan"), "match_2m": float("nan"), "nn_median_m": float("nan")}
    distances, _indices = cKDTree(lidar).query(radar, k=1)
    return {
        "radar_count": int(radar.shape[0]),
        "lidar_count": int(lidar.shape[0]),
        "match_1m": float(np.mean(distances <= 1.0)),
        "match_2m": float(np.mean(distances <= 2.0)),
        "nn_median_m": float(np.median(distances)),
    }


def raw_ir_audit(raw_scene_dir: str, processed_scene_dir: str, sample_count: int) -> Dict[str, object]:
    """抽样确认原始 PNG 位深/通道差异及预处理三通道是否完全复制。"""
    raw_dir = os.path.join(raw_scene_dir, "thermal_cam_thermal_image_compressed")
    raw_files = sorted(name for name in os.listdir(raw_dir) if name.endswith(".png"))
    processed = list_frame_files(os.path.join(processed_scene_dir, "ir_image"), (".npy",))
    raw_rows = []
    for index in evenly_spaced_indices(len(raw_files), sample_count):
        image = cv2.imread(os.path.join(raw_dir, raw_files[index]), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"无法读取原始 IR: {raw_files[index]}")
        channel_delta = 0.0
        if image.ndim == 3:
            channel_delta = float(np.mean(np.ptp(image.astype(np.float32), axis=2)))
        raw_rows.append((str(image.dtype), list(image.shape), channel_delta, float(np.mean(image == 0)), float(np.mean(image == np.iinfo(image.dtype).max))))
    processed_deltas = []
    processed_paths = [processed[key] for key in sorted(processed)]
    for index in evenly_spaced_indices(len(processed_paths), sample_count):
        image = np.load(processed_paths[index], mmap_mode="r")
        if image.ndim == 3 and image.shape[0] >= 3:
            processed_deltas.append(float(np.max(np.abs(image[0] - image[1]))))
            processed_deltas.append(float(np.max(np.abs(image[0] - image[2]))))
    return {
        "sampled_raw_frames": len(raw_rows),
        "raw_dtypes": sorted({row[0] for row in raw_rows}),
        "raw_shapes": sorted({str(row[1]) for row in raw_rows}),
        "raw_channel_delta_mean": float(np.mean([row[2] for row in raw_rows])),
        "raw_zero_fraction_mean": float(np.mean([row[3] for row in raw_rows])),
        "raw_saturation_fraction_mean": float(np.mean([row[4] for row in raw_rows])),
        "sampled_processed_frames": len(processed_deltas) // 2,
        "processed_channel_max_abs_delta": max(processed_deltas, default=float("nan")),
    }


def audit(args: argparse.Namespace) -> Dict[str, object]:
    scene_dir = os.path.join(args.dataset_root, args.scene)
    lidar_files = list_frame_files(os.path.join(scene_dir, "lidar_voxel"))
    radar_files = list_frame_files(os.path.join(scene_dir, "radar_voxel"))
    target_files = list_frame_files(os.path.join(scene_dir, "target_voxel"))
    frame_ids = sorted(set(lidar_files) & set(radar_files) & set(target_files))
    if args.max_frames > 0:
        frame_ids = frame_ids[: args.max_frames]
    if not frame_ids:
        raise RuntimeError("没有共同 Radar/LiDAR/target 帧")

    raw_lidar_dir = os.path.join(args.raw_scene_dir, "livox_lidar")
    raw_lidar_files = sorted(
        (name for name in os.listdir(raw_lidar_dir) if name.endswith(".npy")),
        key=lambda name: float(os.path.splitext(name)[0]),
    )
    sync_path = os.path.join(args.raw_scene_dir, "radar_lidar_sync.csv")
    with open(sync_path, "r", encoding="utf-8", newline="") as handle:
        sync_rows = list(csv.DictReader(handle))
    if len(sync_rows) < len(frame_ids):
        raise RuntimeError("Radar--LiDAR sync 行数少于预处理帧数")

    ray_indices = set(evenly_spaced_indices(len(frame_ids), args.ray_sample_frames))
    overlap_indices = set(
        evenly_spaced_indices(len(frame_ids), args.overlap_sample_frames)
    )
    rows = []
    ray_counts = {label: [] for label, _low, _high in RANGE_BANDS}
    overlap_rows = []
    jaccards = []
    previous_far = None
    for position, frame_id in enumerate(frame_ids):
        lidar_coords, lidar_shape = load_sparse_occupied(lidar_files[frame_id])
        target_coords, target_shape = load_sparse_occupied(target_files[frame_id])
        lidar_bands = count_bands_x(physical_centers(lidar_coords, lidar_shape)[:, 0])
        target_bands = count_bands_x(physical_centers(target_coords, target_shape)[:, 0])
        lidar_index = int(sync_rows[position]["lidar_index"])
        raw = np.load(os.path.join(raw_lidar_dir, raw_lidar_files[lidar_index]), mmap_mode="r")
        raw_bands = count_points_in_grid_bands(raw)
        current_far = flattened_far_indices(lidar_coords, lidar_shape)
        if previous_far is not None:
            jaccards.append(jaccard_sorted(previous_far, current_far))
        previous_far = current_far
        rows.append(
            {
                "frame_id": frame_id,
                "raw_lidar_points_0_80": raw_bands["near_0_80"],
                "raw_lidar_points_80_120": raw_bands["far_80_120"],
                "lidar_occupied_0_80": lidar_bands["near_0_80"],
                "lidar_occupied_80_120": lidar_bands["far_80_120"],
                "target_occupied_0_80": target_bands["near_0_80"],
                "target_occupied_80_120": target_bands["far_80_120"],
            }
        )
        if position in ray_indices:
            counts = sparse_ray_band_counts(lidar_coords, lidar_shape)
            for label, value in counts.items():
                ray_counts[label].append(value)
        if position in overlap_indices:
            radar_coords, radar_shape = load_sparse_occupied(radar_files[frame_id])
            overlap_rows.append(
                radar_lidar_far_overlap(
                    radar_coords,
                    radar_shape,
                    lidar_coords,
                    lidar_shape,
                )
            )
        if (position + 1) % 500 == 0 or position + 1 == len(frame_ids):
            print(
                f"只读审计进度: {position + 1}/{len(frame_ids)}",
                file=sys.stderr,
                flush=True,
            )

    os.makedirs(args.output_dir, exist_ok=False)
    frame_csv = os.path.join(args.output_dir, "far_range_frame_counts.csv")
    with open(frame_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    calibration = CalibrationProvider(
        args.dataset_root,
        calibration_dir=args.calibration_dir,
        require_real=True,
        voxel_coordinate_frame="lidar",
    ).load_with_metadata()[3]
    totals = {
        key: int(sum(int(row[key]) for row in rows))
        for key in rows[0]
        if key != "frame_id"
    }
    far_lidar = totals["lidar_occupied_80_120"]
    far_target = totals["target_occupied_80_120"]
    result = {
        "protocol": "far_range_supervision_audit_v2",
        "raw_point_domain": "inside_xyz_pc_range_before_x_band_count",
        "scene": args.scene,
        "frame_count": len(rows),
        "range_bands_m": [[0.0, 80.0], [80.0, 120.0]],
        "totals": totals,
        "frames_with_raw_lidar_points_80_120": int(
            sum(row["raw_lidar_points_80_120"] > 0 for row in rows)
        ),
        "frames_with_lidar_occupied_80_120": int(
            sum(row["lidar_occupied_80_120"] > 0 for row in rows)
        ),
        "frames_with_target_occupied_80_120": int(
            sum(row["target_occupied_80_120"] > 0 for row in rows)
        ),
        "far_target_retention_ratio": (
            float(far_target / far_lidar) if far_lidar else float("nan")
        ),
        "far_lidar_temporal_same_index_jaccard": finite_summary(jaccards),
        "temporal_stability_limitation": (
            "same-grid index Jaccard only; no pose compensation, so it is not a world-frame stability metric"
        ),
        "ray_coverage": {
            "sampled_frames": len(ray_indices),
            **{
                label: finite_summary(values)
                for label, values in ray_counts.items()
            },
        },
        "far_radar_lidar_overlap": {
            "sampled_frames": len(overlap_rows),
            "match_1m": finite_summary(row["match_1m"] for row in overlap_rows),
            "match_2m": finite_summary(row["match_2m"] for row in overlap_rows),
            "nn_median_m": finite_summary(row["nn_median_m"] for row in overlap_rows),
            "radar_count": int(sum(row["radar_count"] for row in overlap_rows)),
            "lidar_count": int(sum(row["lidar_count"] for row in overlap_rows)),
        },
        "calibration_closure": {
            key: calibration[key]
            for key in (
                "calibration_closure_available",
                "calibration_closure_rotation_max_abs",
                "calibration_closure_translation_l2_m",
                "calibration_closure_composition",
            )
        },
        "ir_audit": raw_ir_audit(
            args.raw_scene_dir,
            scene_dir,
            args.ir_sample_frames,
        ),
        "frame_csv": os.path.abspath(frame_csv),
    }
    report_path = os.path.join(args.output_dir, "far_range_audit.json")
    with open(report_path, "x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--raw_scene_dir", required=True)
    parser.add_argument("--calibration_dir", required=True)
    parser.add_argument("--scene", default="garden")
    parser.add_argument(
        "--output_dir",
        default="test/result/comparison/far_range_supervision_audit_v2",
    )
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--ray_sample_frames", type=int, default=32)
    parser.add_argument("--overlap_sample_frames", type=int, default=128)
    parser.add_argument("--ir_sample_frames", type=int, default=64)
    args = parser.parse_args()
    if os.path.lexists(args.output_dir):
        parser.error(f"output_dir 已存在，拒绝覆盖: {args.output_dir}")
    for name in ("max_frames", "ray_sample_frames", "overlap_sample_frames", "ir_sample_frames"):
        if getattr(args, name) < 0:
            parser.error(f"--{name} 必须是非负整数")
    return args


def main() -> None:
    result = audit(parse_args())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
