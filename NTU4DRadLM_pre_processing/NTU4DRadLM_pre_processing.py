# -*- coding: utf-8 -*-

import os
import json
import argparse
import csv
import sys
import traceback
from typing import Tuple, Optional, Sequence
import numpy as np
import cv2
from tqdm import tqdm
from scipy.spatial import cKDTree
import scipy.ndimage as ndimage
import pypatchworkpp
from multiprocessing import Pool, cpu_count


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from diffusion_consistency_radar.dataset_manifest import write_scene_manifest_atomic
from diffusion_consistency_radar.observed_mask import (
    OBSERVED_MASK_PROTOCOL,
    OBSERVED_MASK_SPATIAL_DOMAIN,
    build_lidar_observed_mask,
    build_spatial_valid_domain,
    save_observed_mask,
)
from diffusion_consistency_radar.radar_statistics import (
    RADAR_STATISTICS_PROTOCOL,
    save_sparse_radar_voxel,
)
from diffusion_consistency_radar.radar_field_schema import (
    load_radar_field_schema_artifact,
    load_radar_layout_schema,
)
from diffusion_consistency_radar.extraction_receipt import (
    load_extraction_receipt_artifact,
)

try:
    # 作为包导入时使用完整模块路径。
    from NTU4DRadLM_pre_processing.motion_protocol import (
        VELOCITY_MODES,
        VELOCITY_FRAMES,
        load_recorded_velocity_table,
        resolve_frame_velocity,
        sensor_to_reference_motion_delta,
        sha256_file,
        transform_velocity,
    )
except ModuleNotFoundError as exc:
    # 直接执行本文件时，脚本文件会遮蔽同名目录，回退到同目录模块。
    if exc.name != "NTU4DRadLM_pre_processing.motion_protocol":
        raise
    from motion_protocol import (  # type: ignore[no-redef]
        VELOCITY_MODES,
        VELOCITY_FRAMES,
        load_recorded_velocity_table,
        resolve_frame_velocity,
        sensor_to_reference_motion_delta,
        sha256_file,
        transform_velocity,
    )

if __package__:
    from .timestamp_alignment import nearest_timestamp_match
else:
    from timestamp_alignment import nearest_timestamp_match  # type: ignore[no-redef]

# ==============================================================================
# 全局参数配置与常驻内存声明
# ==============================================================================
RAW_DATA_PATH = "./Data/NTU4DRadLM_Raw"
INDEX_PATH = "./Data/NTU4DRadLM_Raw"
OUTPUT_PATH = "./Data/NTU4DRadLM_Pre_sensor_aware"
CALIB_PATH = "./Data/config/calib_radar_to_livox.txt"
RADAR_TO_THERMAL_PATH = "./Data/config/calib_radar_to_thermal.txt"
LIDAR_TO_THERMAL_PATH = "./Data/config/calib_livox_to_thermal.txt"
THERMAL_INTRINSICS_PATH = "./Data/config/calib_cam_thermal.txt"

VOXEL_SIZE = [0.2, 0.2, 0.2]
PC_RANGE = [0, -20, -6, 120, 20, 10]
SAVE_SPARSE = True

# 声明一个每个子进程独立的全局常驻 Patchwork 实例占位符
_process_patchwork = None

def _init_worker_patchwork():
    """每个 CPU 核心在启动时只调用一次该函数，完成 C++ 对象的常驻常驻内存绑定"""
    global _process_patchwork
    params = pypatchworkpp.Parameters()
    params.verbose = False
    params.enable_RNR = True
    _process_patchwork = pypatchworkpp.patchworkpp(params)

# ==============================================================================
# 高性能空间矩阵算子
# ==============================================================================

def build_sensor_aware_target_vectorized(
    lidar_voxel: np.ndarray, radar_voxel: np.ndarray, pc_range: tuple,
    z_min: Optional[float], x_max: Optional[float], require_radar_visibility: bool,
    radar_visibility_radius: int, doppler_radius: int,
    visibility_mode: str = "preserve",
    spatial_valid_domain: Optional[np.ndarray] = None,
) -> np.ndarray:
    target = np.zeros_like(lidar_voxel, dtype=np.float32)
    lidar_occ = lidar_voxel[..., 0] > 0
    keep = lidar_occ.copy()

    if spatial_valid_domain is None:
        spatial_valid_domain = build_spatial_valid_domain(
            lidar_voxel.shape[:3],
            pc_range,
            z_min=z_min,
            x_max=x_max,
        )
    else:
        spatial_valid_domain = np.asarray(spatial_valid_domain)
        if spatial_valid_domain.shape != lidar_voxel.shape[:3]:
            raise ValueError(
                "spatial_valid_domain shape 与 LiDAR voxel 不匹配: "
                f"{spatial_valid_domain.shape} != {lidar_voxel.shape[:3]}"
            )
        if spatial_valid_domain.dtype != np.bool_:
            if not np.all(np.isin(spatial_valid_domain, (0, 1))):
                raise ValueError("spatial_valid_domain 只能包含 bool 或 0/1")
            spatial_valid_domain = spatial_valid_domain.astype(bool)
    keep &= spatial_valid_domain

    radar_occ = radar_voxel[..., 0] > 0
    radar_occ_float = radar_occ.astype(np.float32)

    visibility_mode = str(visibility_mode).strip().lower()
    if visibility_mode not in {"preserve", "hard"}:
        raise ValueError(f"Unsupported visibility_mode: {visibility_mode}")

    if visibility_mode == "hard" or require_radar_visibility:
        if radar_visibility_radius > 0:
            k_size = 2 * int(radar_visibility_radius) + 1
            kernel_visible = np.ones((k_size, k_size, k_size), dtype=bool)
            visible = ndimage.binary_dilation(radar_occ, structure=kernel_visible)
            keep &= visible
        else:
            keep &= radar_occ

    target[..., 0] = keep.astype(np.float32)
    target[..., 1] = np.where(keep, lidar_voxel[..., 1], 0.0).astype(np.float32)

    if doppler_radius > 0:
        k_size_d = 2 * int(doppler_radius) + 1
        kernel_d = np.ones((k_size_d, k_size_d, k_size_d), dtype=np.float32)

        radar_doppler = radar_voxel[..., 2]
        radar_doppler_masked = radar_doppler * radar_occ_float

        sum_doppler = ndimage.convolve(radar_doppler_masked, kernel_d, mode='constant', cval=0.0)
        count_radar = ndimage.convolve(radar_occ_float, kernel_d, mode='constant', cval=0.0)

        valid_counts = count_radar > 0
        local_mean = np.zeros_like(radar_doppler)
        local_mean[valid_counts] = sum_doppler[valid_counts] / count_radar[valid_counts]

        final_mask = keep & valid_counts
        target[..., 2] = np.where(final_mask, local_mean, 0.0)
        target[..., 3] = final_mask.astype(np.float32)
    else:
        target[..., 2] = np.where(keep, radar_voxel[..., 2], 0.0)
        target[..., 3] = (keep & radar_occ).astype(np.float32)

    return target


def build_sensor_aware_supervision(
    lidar_voxel: np.ndarray,
    radar_voxel: np.ndarray,
    pc_range: tuple,
    z_min: Optional[float],
    x_max: Optional[float],
    require_radar_visibility: bool,
    radar_visibility_radius: int,
    doppler_radius: int,
    visibility_mode: str = "preserve",
) -> Tuple[np.ndarray, np.ndarray]:
    """从同一显式任务域构建 target 和 LiDAR observed mask。"""
    spatial_valid_domain = build_spatial_valid_domain(
        lidar_voxel.shape[:3],
        pc_range,
        z_min=z_min,
        x_max=x_max,
    )
    target = build_sensor_aware_target_vectorized(
        lidar_voxel=lidar_voxel,
        radar_voxel=radar_voxel,
        pc_range=pc_range,
        z_min=z_min,
        x_max=x_max,
        require_radar_visibility=require_radar_visibility,
        radar_visibility_radius=radar_visibility_radius,
        doppler_radius=doppler_radius,
        visibility_mode=visibility_mode,
        spatial_valid_domain=spatial_valid_domain,
    )
    observed = build_lidar_observed_mask(
        lidar_voxel,
        pc_range,
        valid_domain=spatial_valid_domain,
    )
    return target, observed

# ==============================================================================
# IO 与位姿转换工具
# ==============================================================================

def invert_r_t(r_mat, t_vec):
    return r_mat.T, -np.dot(r_mat.T, t_vec)


def audit_calibration_closure(
    r_radar_to_lidar,
    t_radar_to_lidar,
    radar_to_thermal_path,
    lidar_to_thermal_path,
):
    """比较直接 Radar→Thermal 与经 LiDAR 组合的外参，只记录不擅自择真。"""
    r_radar_to_thermal, t_radar_to_thermal = load_calib(radar_to_thermal_path)
    r_lidar_to_thermal, t_lidar_to_thermal = load_calib(lidar_to_thermal_path)
    composed_r = r_lidar_to_thermal @ r_radar_to_lidar
    composed_t = r_lidar_to_thermal @ t_radar_to_lidar + t_lidar_to_thermal
    return {
        "composition": "radar_to_lidar_then_lidar_to_thermal",
        "rotation_max_abs": float(
            np.max(np.abs(composed_r - r_radar_to_thermal))
        ),
        "translation_l2_m": float(
            np.linalg.norm(composed_t - t_radar_to_thermal)
        ),
        "authority_for_lidar_voxels": "direct_lidar_to_thermal",
        "status": "audit_only_requires_reprojection_review",
    }

def ensure_dir(path):
    if not os.path.exists(path): os.makedirs(path)


def _timestamped_files(directory: str, suffix: str):
    """按文件名时间戳排序并验证严格递增，保证索引文件与原始帧一致。"""
    names = [name for name in os.listdir(directory) if name.endswith(suffix)]
    try:
        names.sort(key=lambda name: float(os.path.splitext(name)[0]))
    except ValueError as exc:
        raise ValueError(f"目录 {directory} 中存在无法解析时间戳的文件") from exc
    timestamps = np.asarray(
        [float(os.path.splitext(name)[0]) for name in names], dtype=np.float64
    )
    if timestamps.size and (
        not np.all(np.isfinite(timestamps))
        or np.any(np.diff(timestamps) <= 0.0)
    ):
        raise ValueError(f"目录 {directory} 的文件名时间戳必须严格递增且为有限数")
    return names, timestamps


def _load_radar_lidar_sync(
    path: str,
    radar_indices,
    lidar_indices,
    radar_timestamps: np.ndarray,
    lidar_timestamps: np.ndarray,
    max_delta: float,
):
    """校验 Step 1 的 Radar/LiDAR 配对及其实际 delta，拒绝旧无阈值索引。"""
    try:
        max_delta = float(max_delta)
    except (TypeError, ValueError) as exc:
        raise ValueError("Radar-LiDAR 时间容差必须是有限非负数") from exc
    if not np.isfinite(max_delta) or max_delta < 0.0:
        raise ValueError("Radar-LiDAR 时间容差必须是有限非负数")
    if os.path.islink(path):
        raise ValueError(f"Radar-LiDAR 同步记录不允许使用符号链接: {path}")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Radar-LiDAR 同步记录不存在: {path}；请先重新运行 NTU4DRadLM_timestamp_index.py"
        )
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(radar_indices) or len(rows) != len(lidar_indices):
        raise ValueError(
            f"Radar-LiDAR 同步记录行数与索引文件不一致: "
            f"sync={len(rows)}, radar={len(radar_indices)}, lidar={len(lidar_indices)}"
        )

    for position, row in enumerate(rows):
        try:
            radar_index = int(row["radar_index"])
            lidar_index = int(row["lidar_index"])
            recorded_delta = float(row["delta_seconds"])
            recorded_signed_delta = float(row["signed_delta_seconds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Radar-LiDAR 同步记录第 {position} 行格式错误") from exc
        if radar_index != radar_indices[position] or lidar_index != lidar_indices[position]:
            raise ValueError(f"Radar-LiDAR 同步记录第 {position} 行与索引文件不匹配")
        if not np.isfinite(recorded_delta) or recorded_delta < 0.0:
            raise ValueError(f"Radar-LiDAR 同步记录第 {position} 行 delta 非法")
        if radar_index < 0 or radar_index >= len(radar_timestamps):
            raise ValueError(f"Radar-LiDAR 同步记录第 {position} 行 Radar 索引越界")
        if lidar_index < 0 or lidar_index >= len(lidar_timestamps):
            raise ValueError(f"Radar-LiDAR 同步记录第 {position} 行 LiDAR 索引越界")
        measured_delta = abs(float(radar_timestamps[radar_index]) - float(lidar_timestamps[lidar_index]))
        measured_signed_delta = (
            float(lidar_timestamps[lidar_index])
            - float(radar_timestamps[radar_index])
        )
        if abs(recorded_delta - measured_delta) > 1e-6:
            raise ValueError(f"Radar-LiDAR 同步记录第 {position} 行 delta 与文件名时间戳不一致")
        if (
            not np.isfinite(recorded_signed_delta)
            or abs(recorded_signed_delta - measured_signed_delta) > 1e-6
        ):
            raise ValueError(
                f"Radar-LiDAR 同步记录第 {position} 行 signed delta 与文件名时间戳不一致"
            )
        if recorded_delta > max_delta:
            raise ValueError(
                f"Radar-LiDAR 第 {position} 对时间差 {recorded_delta:.9f}s "
                f"超过时间容差 {max_delta:.9f}s"
            )
    return rows


def _ensure_motion_args(args):
    """为旧的 Namespace 调用补齐安全默认值，避免隐式恢复固定 50m/s。"""
    defaults = {
        "velocity_mode": "none",
        "velocity_frame": "radar",
        "velocity_file": "",
        "velocity_max_delta": 0.02,
        "radar_lidar_max_delta": 0.045,
        "radar_ir_max_delta": 0.025,
        "radar_to_thermal_path": RADAR_TO_THERMAL_PATH,
        "lidar_to_thermal_path": LIDAR_TO_THERMAL_PATH,
        "thermal_intrinsics_path": THERMAL_INTRINSICS_PATH,
        "radar_field_schema": "",
        "require_verified_radar_field_schema": False,
        "require_complete_extraction_receipt": False,
    }
    for name, default in defaults.items():
        if not hasattr(args, name):
            setattr(args, name, default)
    return args


def resolve_radar_field_schema(
    path,
    *,
    require_verified,
    velocity_mode,
    layout_schema_path=None,
):
    """解析字段语义；Doppler 运动补偿禁止沿用未验证符号假设。"""
    schema_path = os.fspath(path).strip() if path is not None else ""
    if not schema_path:
        if require_verified:
            raise ValueError("正式预处理缺少 --radar_field_schema")
        if velocity_mode != "none":
            raise ValueError("启用 Radar Doppler 运动补偿前必须提供 verified field schema")
        return None, None
    schema, digest = load_radar_field_schema_artifact(
        schema_path,
        require_verified=require_verified or velocity_mode != "none",
    )
    if not layout_schema_path:
        raise ValueError("Radar field schema 缺少对应 pointcloud layout sidecar")
    load_radar_layout_schema(layout_schema_path, schema)
    return schema, digest


def resolve_extraction_receipt(path, *, require_complete):
    """已有收据必须完整；formal-v3 还要求收据不得缺失。"""
    receipt_path = os.path.abspath(os.fspath(path))
    if not os.path.isfile(receipt_path) or os.path.islink(receipt_path):
        if require_complete:
            raise ValueError(f"正式预处理缺少 complete extraction receipt: {receipt_path}")
        return None, None
    return load_extraction_receipt_artifact(
        receipt_path,
        require_complete=True,
    )

def ensure_fresh_scene_output(scene_out_path):
    """仅允许向不存在或为空的普通场景目录写入，避免覆盖旧批次。"""
    if os.path.lexists(scene_out_path):
        if os.path.islink(scene_out_path):
            raise RuntimeError(f"输出场景目录不允许使用符号链接: {scene_out_path}")
        if not os.path.isdir(scene_out_path):
            raise RuntimeError(f"输出场景路径不是目录: {scene_out_path}")
        if os.listdir(scene_out_path):
            raise RuntimeError(f"输出场景目录必须为空: {scene_out_path}")
        return
    os.makedirs(scene_out_path)

def load_calib(calib_file):
    """严格读取 source-to-target 外参，拒绝缺字段时静默使用单位阵。"""
    R, T = None, None
    if os.path.islink(calib_file) or not os.path.isfile(calib_file):
        raise FileNotFoundError(f"Calibration file not found or not regular: {calib_file}.")
    with open(calib_file, 'r') as f:
        for line in f:
            if ':' not in line: continue
            parts = line.strip().split(':')
            if len(parts) < 2: continue
            key, raw_parts = parts[0].strip(), parts[1].strip().split()
            vals = []
            for x in raw_parts:
                try: vals.append(float(x))
                except ValueError: continue
            if len(vals) == 0: continue
            if key == 'R' and len(vals) == 9: R = np.array(vals).reshape(3, 3)
            elif key == 'T' and len(vals) == 3: T = np.array(vals)
    if R is None or T is None or not np.all(np.isfinite(R)) or not np.all(np.isfinite(T)):
        raise ValueError(f"Calibration R/T missing or non-finite: {calib_file}")
    if not np.allclose(R @ R.T, np.eye(3), atol=5e-3, rtol=0.0):
        raise ValueError(f"Calibration R is not orthogonal: {calib_file}")
    determinant = float(np.linalg.det(R))
    if abs(determinant - 1.0) > 5e-3:
        raise ValueError(
            f"Calibration R determinant must be near 1, got {determinant}: {calib_file}"
        )
    return R, T

def transform_pcl(pcl, R, T):
    if pcl.shape[0] == 0: return pcl
    pcl_trans = pcl.copy()
    pcl_trans[:, :3] = np.dot(pcl[:, :3], R.T) + T
    return pcl_trans


def align_radar_lidar_pointclouds(
    radar_pcl,
    lidar_pcl,
    *,
    radar_point_coordinate_frame,
    target_frame,
    radar_to_lidar_rotation,
    radar_to_lidar_translation,
):
    """按已验证的 Radar 点物理坐标系，把两种点云对齐到同一目标 frame。"""
    if radar_point_coordinate_frame not in {"radar", "lidar"}:
        raise ValueError(
            f"Radar 点位于 {radar_point_coordinate_frame!r}，但当前只具备 "
            "Radar↔LiDAR 外参；禁止把 base_link 或未知 frame 当作 radar"
        )
    if target_frame not in {"radar", "lidar"}:
        raise ValueError(f"target_frame 必须是 radar 或 lidar，当前为 {target_frame!r}")

    inverse_rotation = radar_to_lidar_rotation.T
    inverse_translation = -np.dot(inverse_rotation, radar_to_lidar_translation)
    if radar_point_coordinate_frame != target_frame:
        if radar_point_coordinate_frame == "radar":
            radar_pcl = transform_pcl(
                radar_pcl,
                radar_to_lidar_rotation,
                radar_to_lidar_translation,
            )
        else:
            radar_pcl = transform_pcl(
                radar_pcl,
                inverse_rotation,
                inverse_translation,
            )
    if target_frame == "radar":
        lidar_pcl = transform_pcl(
            lidar_pcl,
            inverse_rotation,
            inverse_translation,
        )
    return radar_pcl, lidar_pcl


def compensate_radar_doppler(
    pcl,
    radar_velocity,
    *,
    positive_direction="toward_sensor",
):
    """按 schema 声明的 Doppler 正方向剔除平台径向速度。"""
    if radar_velocity is None or pcl.shape[0] == 0 or pcl.shape[1] <= 4:
        return pcl
    velocity = np.asarray(radar_velocity, dtype=np.float32)
    if velocity.shape != (3,) or not np.all(np.isfinite(velocity)):
        raise ValueError("radar_velocity 必须是三个有限数")
    if positive_direction not in {"toward_sensor", "away_from_sensor"}:
        raise ValueError("Doppler positive_direction 必须明确 toward/away")
    corrected = pcl.copy()
    xyz = corrected[:, :3]
    radius = np.maximum(np.linalg.norm(xyz, axis=1), 1e-6)
    ego_radial = np.sum(xyz * velocity[None, :], axis=1) / radius
    direction_sign = -1.0 if positive_direction == "toward_sensor" else 1.0
    corrected[:, 4] = corrected[:, 4] + direction_sign * ego_radial
    return corrected


def move_pcl_to_reference_time(pcl, velocity, motion_delta_seconds):
    """在共享坐标系内按有符号时间量移动一份非参考点云。"""
    if velocity is None or pcl.shape[0] == 0:
        return pcl
    delta = float(motion_delta_seconds)
    if not np.isfinite(delta):
        raise ValueError("motion_delta_seconds 必须是有限数")
    if abs(delta) <= 1e-12:
        return pcl
    moved = pcl.copy()
    moved[:, :3] += np.asarray(velocity, dtype=np.float32) * delta
    return moved

def save_voxel(filename, voxel_grid, radar_statistics=None):
    """保存普通稀疏体素；Radar 可把统计合同绑定在同一 NPZ 内。"""
    if SAVE_SPARSE:
        if radar_statistics is not None:
            save_sparse_radar_voxel(filename, voxel_grid, radar_statistics)
            return
        occupied = voxel_grid[..., 0] > 0
        coords = np.column_stack(np.where(occupied))
        features = voxel_grid[occupied]
        np.savez(filename, coords=coords, features=features, shape=voxel_grid.shape)
    else:
        if radar_statistics is not None:
            raise ValueError("Radar statistics 只支持 SAVE_SPARSE=True")
        np.save(filename, voxel_grid.astype(np.float32))

def voxelize_pcl_airborne_optimized(
    pcl,
    voxel_size,
    pc_range,
    v_drone=None,
    dt_sync=0.0,
    return_statistics=False,
):
    """
    重构后的机载自适应点云体素化核心函数

    参数说明:
    pcl: np.ndarray (N, C) -> 原始点云数据。前3列为 x, y, z；第4列为强度特征；第5列为原始相对多普勒速度
    voxel_size: list [3] -> 体素网格的分辨率 [dx, dy, dz] 单位:米
    pc_range: list [6] -> 感知空间的边界 [x_min, y_min, z_min, x_max, y_max, z_max]
    v_drone: array_like [3] -> 无人机当前的瞬时速度绝对值向量 [vx, vy, vz], 单位: m/s
    dt_sync: float -> 当前点云时间减参考时刻的有符号时间量, 单位: 秒
    """
    dt_sync = float(dt_sync)
    if not np.isfinite(dt_sync):
        raise ValueError("dt_sync 必须是有限数")
    if v_drone is not None:
        v_drone = np.asarray(v_drone, dtype=np.float32)
        if v_drone.shape != (3,) or not np.all(np.isfinite(v_drone)):
            raise ValueError("v_drone 必须是三个有限数")

    pcl = np.asarray(pcl, dtype=np.float32)
    if pcl.ndim != 2 or pcl.shape[1] < 3:
        raise ValueError("pcl 必须是至少含 XYZ 三列的二维数组")
    # 坐标非有限的点无法定义体素索引，必须在运动补偿和距离
    # 计算前丢弃；强度/Doppler 则保留空间占用并分字段计数。
    pcl = pcl[np.all(np.isfinite(pcl[:, :3]), axis=1)]

    # 1. 将非参考传感器点云移动到明确参考时刻；参考传感器的 dt 恒为 0。
    if abs(dt_sync) > 1e-6 and v_drone is not None:
        pcl = pcl.copy()
        pcl[:, :3] += np.array(v_drone, dtype=np.float32) * dt_sync

    # 2. 物理级自身运动多普勒解耦 (Egomotion Compensation)
    # 剔除由于飞机自身运动造成的静态背景/障碍物多普勒污染
    if v_drone is not None and pcl.shape[1] > 4:
        pcl = pcl.copy()
        x, y, z = pcl[:, 0], pcl[:, 1], pcl[:, 2]
        # 计算雷达发射探束的径向物理距离
        r = np.maximum(np.sqrt(x**2 + y**2 + z**2), 1e-6)
        # 无人机速度向量在当前各个雷达束射线方向上的投影分量
        v_ego_projected = (x * v_drone[0] + y * v_drone[1] + z * v_drone[2]) / r
        # 从相对速度中剥离自车运动，恢复纯净的障碍物测速
        pcl[:, 4] = pcl[:, 4] - v_ego_projected

    # 3. 空间边界裁剪（向量化过滤）
    keep = (pcl[:, 0] >= pc_range[0]) & (pcl[:, 0] < pc_range[3]) & \
           (pcl[:, 1] >= pc_range[1]) & (pcl[:, 1] < pc_range[4]) & \
           (pcl[:, 2] >= pc_range[2]) & (pcl[:, 2] < pc_range[5])
    pcl = pcl[keep]

    grid_shape = (
        int((pc_range[3] - pc_range[0]) / voxel_size[0]),
        int((pc_range[4] - pc_range[1]) / voxel_size[1]),
        int((pc_range[5] - pc_range[2]) / voxel_size[2])
    )
    if pcl.shape[0] == 0:
        empty_voxel = np.zeros(grid_shape + (4,), dtype=np.float32)
        if not return_statistics:
            return empty_voxel
        return empty_voxel, {
            "protocol": RADAR_STATISTICS_PROTOCOL,
            "coords": np.empty((0, 3), dtype=np.int32),
            "point_count": np.empty((0,), dtype=np.uint32),
            "intensity_valid_count": np.empty((0,), dtype=np.uint32),
            "doppler_valid_count": np.empty((0,), dtype=np.uint32),
        }

    # 4. 展平 3D 矩阵，实现无 Python 循环的高性能散列填充 (Scatter Accumulate)
    x_idx = np.clip(((pcl[:, 0] - pc_range[0]) / voxel_size[0]).astype(np.int32), 0, grid_shape[0] - 1)
    y_idx = np.clip(((pcl[:, 1] - pc_range[1]) / voxel_size[1]).astype(np.int32), 0, grid_shape[1] - 1)
    z_idx = np.clip(((pcl[:, 2] - pc_range[2]) / voxel_size[2]).astype(np.int32), 0, grid_shape[2] - 1)

    voxel_grid = np.zeros(grid_shape + (4,), dtype=np.float32)
    flat_indices = x_idx * (grid_shape[1] * grid_shape[2]) + y_idx * grid_shape[2] + z_idx

    sort_order = np.argsort(flat_indices)
    flat_indices = flat_indices[sort_order]
    features = (
        pcl[sort_order, 3]
        if pcl.shape[1] > 3
        else np.ones(pcl.shape[0], dtype=np.float32)
    )
    intensity_valid = np.isfinite(features)
    has_doppler = pcl.shape[1] > 4
    doppler = (
        pcl[sort_order, 4]
        if has_doppler
        else np.zeros(pcl.shape[0], dtype=np.float32)
    )
    doppler_valid = (
        np.isfinite(pcl[sort_order, 4])
        if has_doppler
        else np.zeros(pcl.shape[0], dtype=bool)
    )

    unique_indices, unique_counts = np.unique(flat_indices, return_counts=True)
    uz_idx = unique_indices % grid_shape[2]
    uy_idx = (unique_indices // grid_shape[2]) % grid_shape[1]
    ux_idx = (unique_indices // (grid_shape[2] * grid_shape[1]))

    # 通道 0: 空间占用状态 (Occupancy Mask)
    voxel_grid[ux_idx, uy_idx, uz_idx, 0] = 1.0
    # 通道 1: 平均反射特征强度 (Mean Intensity)
    accumulator_size = int(np.prod(grid_shape))
    # 使用 float64 累加，避免多个有限 float32 大值在求和时溢出。
    sum_features = np.zeros(accumulator_size, dtype=np.float64)
    intensity_valid_accumulator = np.zeros(accumulator_size, dtype=np.uint32)
    np.add.at(sum_features, flat_indices, np.where(intensity_valid, features, 0.0))
    np.add.at(
        intensity_valid_accumulator,
        flat_indices,
        intensity_valid.astype(np.uint32, copy=False),
    )
    intensity_counts = intensity_valid_accumulator[unique_indices]
    intensity_mean = np.zeros(unique_indices.shape[0], dtype=np.float64)
    intensity_has_samples = intensity_counts > 0
    intensity_mean[intensity_has_samples] = (
        sum_features[unique_indices[intensity_has_samples]]
        / intensity_counts[intensity_has_samples]
    )
    voxel_grid[ux_idx, uy_idx, uz_idx, 1] = intensity_mean
    # 通道 2: 有限 Doppler 样本均值；是否补偿由外部 policy 审计。
    sum_doppler = np.zeros(accumulator_size, dtype=np.float64)
    doppler_valid_accumulator = np.zeros(accumulator_size, dtype=np.uint32)
    finite_doppler = np.where(doppler_valid, doppler, 0.0).astype(
        np.float64,
        copy=False,
    )
    np.add.at(sum_doppler, flat_indices, finite_doppler)
    np.add.at(
        doppler_valid_accumulator,
        flat_indices,
        doppler_valid.astype(np.uint32, copy=False),
    )
    doppler_counts = doppler_valid_accumulator[unique_indices]
    doppler_mean = np.zeros(unique_indices.shape[0], dtype=np.float64)
    doppler_has_samples = doppler_counts > 0
    doppler_mean[doppler_has_samples] = (
        sum_doppler[unique_indices[doppler_has_samples]]
        / doppler_counts[doppler_has_samples]
    )
    voxel_grid[ux_idx, uy_idx, uz_idx, 2] = doppler_mean
    # 通道 3: 有限 Doppler 样本的体素内方差。
    # Var = E[X^2] - (E[X])^2
    sum_doppler_sq = np.zeros(accumulator_size, dtype=np.float64)
    np.add.at(sum_doppler_sq, flat_indices, finite_doppler ** 2)
    mean_doppler_sq = np.zeros(unique_indices.shape[0], dtype=np.float64)
    mean_doppler_sq[doppler_has_samples] = (
        sum_doppler_sq[unique_indices[doppler_has_samples]]
        / doppler_counts[doppler_has_samples]
    )

    var_doppler = mean_doppler_sq - doppler_mean ** 2
   # 截断保护，防止随机噪点方差过大导致神经网络训练时出现 NaN 异常
    voxel_grid[ux_idx, uy_idx, uz_idx, 3] = np.clip(var_doppler, 0.0, 50.0)

    if not return_statistics:
        return voxel_grid
    statistics = {
        "protocol": RADAR_STATISTICS_PROTOCOL,
        "coords": np.column_stack((ux_idx, uy_idx, uz_idx)).astype(
            np.int32,
            copy=False,
        ),
        "point_count": unique_counts.astype(np.uint32, copy=False),
        "intensity_valid_count": intensity_counts.astype(np.uint32, copy=False),
        "doppler_valid_count": doppler_valid_accumulator[unique_indices].astype(
            np.uint32,
            copy=False,
        ),
    }
    return voxel_grid, statistics

# ==============================================================================
# 工作子进程单元
# ==============================================================================

def _parallel_frame_worker(task_args):
    global _process_patchwork

    (i, r_file, l_file, radar_timestamp, lidar_timestamp,
     scene_raw_path, scene_out_path,
     r_radar_to_lidar, t_radar_to_lidar, frame_velocity,
     thermal_timestamps, thermal_files, thermal_dir, args_dict,
     thermal_index, thermal_delta) = task_args

    radar_pcl = np.load(os.path.join(scene_raw_path, "radar_pcl", r_file))
    lidar_pcl = np.load(os.path.join(scene_raw_path, "livox_lidar", l_file))

    # 直接复用全局的 _process_patchwork 执行地面滤波
    if lidar_pcl.shape[0] > 0 and _process_patchwork is not None:
        _process_patchwork.estimateGround(lidar_pcl)
        try: lidar_pcl = lidar_pcl[_process_patchwork.getNongroundIndices()]
        except AttributeError:
            nonground = _process_patchwork.getNonground()
            if nonground.shape[0] > 0:
                tree = cKDTree(lidar_pcl[:, :3])
                _, idx = tree.query(nonground[:, :3], k=1)
                lidar_pcl = lidar_pcl[idx]

    radar_velocity = None
    if frame_velocity is not None:
        radar_velocity = transform_velocity(
            frame_velocity,
            source_frame=args_dict["velocity_frame"],
            target_frame="radar",
            radar_to_lidar_rotation=r_radar_to_lidar,
        )
        radar_pcl = compensate_radar_doppler(
            radar_pcl,
            radar_velocity,
            positive_direction=args_dict["radar_doppler_positive_direction"],
        )

    radar_pcl, lidar_pcl = align_radar_lidar_pointclouds(
        radar_pcl,
        lidar_pcl,
        radar_point_coordinate_frame=args_dict["radar_point_coordinate_frame"],
        target_frame=args_dict["align_to"],
        radar_to_lidar_rotation=r_radar_to_lidar,
        radar_to_lidar_translation=t_radar_to_lidar,
    )

    target_velocity = None
    if frame_velocity is not None:
        target_velocity = transform_velocity(
            frame_velocity,
            source_frame=args_dict["velocity_frame"],
            target_frame=args_dict["align_to"],
            radar_to_lidar_rotation=r_radar_to_lidar,
        )

    reference_timestamp = (
        lidar_timestamp if args_dict["align_to"] == "lidar" else radar_timestamp
    )
    radar_motion_delta = sensor_to_reference_motion_delta(
        radar_timestamp,
        reference_timestamp,
    )
    lidar_motion_delta = sensor_to_reference_motion_delta(
        lidar_timestamp,
        reference_timestamp,
    )
    radar_pcl = move_pcl_to_reference_time(
        radar_pcl,
        target_velocity,
        radar_motion_delta,
    )
    lidar_pcl = move_pcl_to_reference_time(
        lidar_pcl,
        target_velocity,
        lidar_motion_delta,
    )

    r_voxel, radar_statistics = voxelize_pcl_airborne_optimized(
        radar_pcl,
        VOXEL_SIZE,
        args_dict["pc_range"],
        v_drone=None,
        dt_sync=0.0,
        return_statistics=True,
    )
    l_voxel = voxelize_pcl_airborne_optimized(
        lidar_pcl,
        VOXEL_SIZE,
        args_dict["pc_range"],
        v_drone=None,
        dt_sync=0.0,
    )

    target_voxel, observed_mask = build_sensor_aware_supervision(
        lidar_voxel=l_voxel, radar_voxel=r_voxel, pc_range=args_dict["pc_range"],
        z_min=args_dict["z_min"], x_max=args_dict["x_max"],
        require_radar_visibility=args_dict["require_radar_visibility"],
        radar_visibility_radius=args_dict["radar_visibility_radius"],
        doppler_radius=args_dict["doppler_radius"],
        visibility_mode=args_dict["visibility_mode"],
    )

    # 红外帧索引和实际 delta 在主进程中预先计算并通过任务传入，
    # 避免 worker 各自静默选择超出容差的最近帧。
    if thermal_index is not None:
        img = cv2.imread(
            os.path.join(thermal_dir, thermal_files[thermal_index]),
            cv2.IMREAD_GRAYSCALE,
        )
        if img is None:
            raise RuntimeError(
                f"无法读取 IR 帧 {thermal_files[thermal_index]} "
                f"(Radar timestamp={radar_timestamp:.9f})"
            )
        img_3ch = np.stack(
            [cv2.resize(img, (640, 480)).astype(np.float32) / 255.0] * 3,
            axis=0,
        )
        np.save(os.path.join(scene_out_path, "ir_image", f"{i:06d}_ir.npy"), img_3ch)

    ext = ".npz" if SAVE_SPARSE else ".npy"
    save_voxel(
        os.path.join(scene_out_path, "radar_voxel", f"{i:06d}{ext}"),
        r_voxel,
        radar_statistics=radar_statistics,
    )
    save_voxel(os.path.join(scene_out_path, "lidar_voxel", f"{i:06d}{ext}"), l_voxel)
    save_voxel(os.path.join(scene_out_path, "target_voxel", f"{i:06d}{ext}"), target_voxel)
    save_observed_mask(
        os.path.join(scene_out_path, "observed_mask", f"{i:06d}.npz"),
        observed_mask,
        args_dict["pc_range"],
    )
    return True

# ==============================================================================
# 场景总控制中心
# ==============================================================================

def process_scene_task(scene_name, args, v_drone=None):
    _ensure_motion_args(args)
    print(f"\n⚡ 正在初始化多进程并行流水线，目标场景: {scene_name}")
    scene_raw_path = os.path.join(args.raw_data_path, scene_name)
    scene_index_path = os.path.join(args.index_path, scene_name)
    scene_out_path = os.path.join(args.output_path, scene_name)

    r_radar_to_lidar, t_radar_to_lidar = load_calib(args.calib_path)
    if args.invert_calib: r_radar_to_lidar, t_radar_to_lidar = invert_r_t(r_radar_to_lidar, t_radar_to_lidar)
    if abs(args.radar_z_shift) > 1e-8: t_radar_to_lidar[2] += float(args.radar_z_shift)
    calibration_closure = audit_calibration_closure(
        r_radar_to_lidar,
        t_radar_to_lidar,
        args.radar_to_thermal_path,
        args.lidar_to_thermal_path,
    )
    if os.path.islink(args.thermal_intrinsics_path) or not os.path.isfile(
        args.thermal_intrinsics_path
    ):
        raise FileNotFoundError(
            "Thermal intrinsics file not found or not regular: "
            f"{args.thermal_intrinsics_path}"
        )

    if args.velocity_mode not in VELOCITY_MODES:
        raise ValueError(f"velocity_mode 必须是 {VELOCITY_MODES} 之一")
    if args.velocity_frame not in VELOCITY_FRAMES:
        raise ValueError(f"velocity_frame 必须是 {VELOCITY_FRAMES} 之一")
    radar_field_schema, radar_field_schema_sha256 = resolve_radar_field_schema(
        args.radar_field_schema,
        require_verified=bool(args.require_verified_radar_field_schema),
        velocity_mode=args.velocity_mode,
        layout_schema_path=os.path.join(
            scene_raw_path,
            "radar_pcl",
            "pointcloud_schema.json",
        ),
    )
    radar_layout_schema_sha256 = (
        sha256_file(
            os.path.join(
                scene_raw_path,
                "radar_pcl",
                "pointcloud_schema.json",
            )
        )
        if radar_field_schema is not None
        else None
    )
    extraction_receipt, extraction_receipt_sha256 = resolve_extraction_receipt(
        os.path.join(scene_raw_path, "extraction_receipt.json"),
        require_complete=bool(args.require_complete_extraction_receipt),
    )
    radar_doppler_positive_direction = (
        radar_field_schema["fields"]["doppler"]["positive_direction"]
        if radar_field_schema is not None
        else None
    )
    radar_point_coordinate_frame = (
        radar_field_schema["fields"]["xyz"].get(
            "physical_coordinate_frame",
            radar_field_schema["fields"]["xyz"].get("coordinate_frame"),
        )
        if radar_field_schema is not None
        else "radar"
    )
    if radar_point_coordinate_frame not in {"radar", "lidar"}:
        raise ValueError(
            f"Radar 点物理坐标系 {radar_point_coordinate_frame!r} 缺少到 "
            "Radar/LiDAR 的已验证外参，拒绝创建预处理输出"
        )
    fixed_velocity = None
    recorded_table = None
    velocity_file_sha256 = None
    if args.velocity_mode == "fixed":
        fixed_velocity = (
            args.vx if v_drone is None else v_drone[0],
            args.vy if v_drone is None else v_drone[1],
            args.vz if v_drone is None else v_drone[2],
        )
    elif args.velocity_mode == "recorded":
        recorded_table = load_recorded_velocity_table(args.velocity_file)
        velocity_file_sha256 = sha256_file(args.velocity_file)

    thermal_dir = os.path.join(scene_raw_path, "thermal_cam_thermal_image_compressed")
    if not os.path.isdir(thermal_dir):
        raise FileNotFoundError(f"IR 数据目录不存在，拒绝无提示地跳过 IR: {thermal_dir}")
    thermal_files, thermal_timestamps = _timestamped_files(thermal_dir, ".png")
    if thermal_timestamps.size == 0:
        raise ValueError(f"IR 数据目录为空，拒绝无提示地跳过 IR: {thermal_dir}")

    radar_index_path = os.path.join(scene_index_path, "radar_index_sequence.txt")
    lidar_index_path = os.path.join(scene_index_path, "lidar_index_sequence.txt")
    try:
        with open(radar_index_path, 'r') as f:
            radar_indices = [int(line.strip()) for line in f.readlines()]
        with open(lidar_index_path, 'r') as f:
            lidar_indices = [int(line.strip()) for line in f.readlines()]
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Index files not found for {scene_name}"
        ) from exc

    radar_files, radar_timestamps = _timestamped_files(
        os.path.join(scene_raw_path, "radar_pcl"), ".npy"
    )
    lidar_files, lidar_timestamps = _timestamped_files(
        os.path.join(scene_raw_path, "livox_lidar"), ".npy"
    )
    radar_lidar_sync_path = os.path.join(scene_index_path, "radar_lidar_sync.csv")
    radar_lidar_sync_rows = _load_radar_lidar_sync(
        radar_lidar_sync_path,
        radar_indices,
        lidar_indices,
        radar_timestamps,
        lidar_timestamps,
        args.radar_lidar_max_delta,
    )

    min_len = min(len(radar_indices), len(lidar_indices))
    if args.max_frames > 0: min_len = min(min_len, int(args.max_frames))

    args_dict = dict(vars(args))
    args_dict["radar_doppler_positive_direction"] = (
        radar_doppler_positive_direction
    )
    args_dict["radar_point_coordinate_frame"] = radar_point_coordinate_frame
    worker_tasks = []
    ir_sync_records = []
    for i in range(min_len):
        r_idx, l_idx = radar_indices[i], lidar_indices[i]
        if r_idx >= len(radar_files) or l_idx >= len(lidar_files): continue
        radar_timestamp = float(radar_timestamps[r_idx])
        lidar_timestamp = float(lidar_timestamps[l_idx])
        thermal_index, thermal_delta = nearest_timestamp_match(
            thermal_timestamps,
            radar_timestamp,
            max_delta=args.radar_ir_max_delta,
        )
        frame_velocity = resolve_frame_velocity(
            mode=args.velocity_mode,
            fixed_velocity=fixed_velocity,
            frame_timestamp=(
                lidar_timestamp if args.align_to == "lidar" else radar_timestamp
            ),
            recorded_table=recorded_table,
            max_delta=args.velocity_max_delta,
        )

        worker_tasks.append((
            i, radar_files[r_idx], lidar_files[l_idx], radar_timestamp,
            lidar_timestamp,
            scene_raw_path, scene_out_path, r_radar_to_lidar, t_radar_to_lidar,
            frame_velocity, thermal_timestamps, thermal_files, thermal_dir, args_dict,
            thermal_index, thermal_delta,
        ))
        ir_sync_records.append(
            {
                "frame_index": i,
                "radar_timestamp": f"{radar_timestamp:.9f}",
                "ir_timestamp": f"{thermal_timestamps[thermal_index]:.9f}",
                "delta_seconds": f"{thermal_delta:.9f}",
                "signed_delta_seconds": (
                    f"{float(thermal_timestamps[thermal_index]) - radar_timestamp:.9f}"
                ),
            }
        )

    num_workers = min(cpu_count(), len(worker_tasks), 16)
    if num_workers <= 0:
        raise RuntimeError(
            f"场景 {scene_name} 没有可处理的 Radar/LiDAR 配对帧，拒绝启动零进程流水线"
        )

    # 只有所有索引和 IR 时间容差检查通过后才创建输出目录，避免失败场景
    # 留下无法再次运行的半成品目录。
    ensure_fresh_scene_output(scene_out_path)
    ensure_dir(os.path.join(scene_out_path, "radar_voxel"))
    ensure_dir(os.path.join(scene_out_path, "lidar_voxel"))
    ensure_dir(os.path.join(scene_out_path, "target_voxel"))
    ensure_dir(os.path.join(scene_out_path, "observed_mask"))
    ensure_dir(os.path.join(scene_out_path, "ir_image"))

    # 使用 initializer 绑定进程启动钩子，每个进程终生只打印一次初始化日志！
    print(f"🔥 正在拉起 {num_workers} 个并行的常驻感知 Worker...")
    written = 0
    with Pool(processes=num_workers, initializer=_init_worker_patchwork) as pool:
        for _ in tqdm(pool.imap_unordered(_parallel_frame_worker, worker_tasks), total=len(worker_tasks), desc=f"Parallel {scene_name}"):
            written += 1

    # 将每个实际使用的 Radar-IR 最近邻和 delta 持久化，供审计和复现实验使用。
    ir_sync_path = os.path.join(scene_out_path, "radar_ir_sync.csv")
    with open(ir_sync_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "frame_index",
                "radar_timestamp",
                "ir_timestamp",
                "delta_seconds",
                "signed_delta_seconds",
            ),
        )
        writer.writeheader()
        writer.writerows(ir_sync_records)

    metadata = {
        "source_scene": scene_name, "frames_written": written,
        "policy": {
            "z_min": args.z_min, "x_max": args.x_max,
            "visibility_mode": args.visibility_mode,
            "require_radar_visibility": args.require_radar_visibility,
            "radar_visibility_radius": args.radar_visibility_radius, "doppler_radius": args.doppler_radius,
            "velocity_mode": args.velocity_mode,
            "velocity_frame": args.velocity_frame,
            "velocity_max_delta": args.velocity_max_delta,
            "velocity_file_sha256": velocity_file_sha256,
            "radar_lidar_max_delta": args.radar_lidar_max_delta,
            "radar_ir_max_delta": args.radar_ir_max_delta,
        }
    }
    target_policy_path = os.path.join(scene_out_path, "target_policy.json")
    with open(target_policy_path, "w", encoding="utf-8") as h:
        json.dump(metadata, h, indent=2)
    radar_lidar_signed_deltas = [
        float(row["signed_delta_seconds"])
        for row in radar_lidar_sync_rows[:min_len]
    ]
    radar_ir_signed_deltas = [
        float(row["signed_delta_seconds"])
        for row in ir_sync_records
    ]
    preprocess_policy = {
        "source_scene": scene_name,
        "frames_written": written,
        "pc_range": list(args.pc_range),
        "voxel_size": list(VOXEL_SIZE),
        "align_to": args.align_to,
        "voxel_coordinate_frame": args.align_to,
        "time_reference_sensor": args.align_to,
        "invert_calib": bool(args.invert_calib),
        "radar_z_shift": float(args.radar_z_shift),
        "velocity_mode": args.velocity_mode,
        "velocity_frame": args.velocity_frame,
        "velocity_max_delta": float(args.velocity_max_delta),
        "v_drone": (
            [float(value) for value in fixed_velocity]
            if fixed_velocity is not None
            else None
        ),
        "velocity_file": (
            os.path.basename(args.velocity_file)
            if args.velocity_mode == "recorded"
            else None
        ),
        "velocity_file_sha256": velocity_file_sha256,
        "velocity_record_count": (
            int(recorded_table.shape[0]) if recorded_table is not None else None
        ),
        "radar_lidar_max_delta": float(args.radar_lidar_max_delta),
        "radar_lidar_sync_filename": os.path.basename(radar_lidar_sync_path),
        "radar_lidar_signed_delta_semantics": "lidar_timestamp_minus_radar_timestamp",
        "radar_lidar_signed_delta_min_seconds": min(radar_lidar_signed_deltas),
        "radar_lidar_signed_delta_max_seconds": max(radar_lidar_signed_deltas),
        "radar_ir_max_delta": float(args.radar_ir_max_delta),
        "radar_ir_sync_filename": os.path.basename(ir_sync_path),
        "radar_ir_signed_delta_semantics": "ir_timestamp_minus_radar_timestamp",
        "radar_ir_signed_delta_min_seconds": min(radar_ir_signed_deltas),
        "radar_ir_signed_delta_max_seconds": max(radar_ir_signed_deltas),
        "spatial_time_compensation": (
            "non_reference_sensor_only"
            if args.velocity_mode != "none"
            else "none_no_velocity"
        ),
        "radar_field_schema": radar_field_schema,
        "radar_field_schema_sha256": radar_field_schema_sha256,
        "radar_pointcloud_layout_sha256": radar_layout_schema_sha256,
        "radar_field_schema_status": (
            radar_field_schema["verification"]["status"]
            if radar_field_schema is not None
            else "absent_unverified"
        ),
        "radar_doppler_positive_direction": radar_doppler_positive_direction,
        "radar_point_coordinate_frame": radar_point_coordinate_frame,
        "extraction_receipt": extraction_receipt,
        "extraction_receipt_sha256": extraction_receipt_sha256,
        "extraction_receipt_status": (
            extraction_receipt["status"]
            if extraction_receipt is not None
            else "absent_legacy"
        ),
        "calibration_closure_audit": calibration_closure,
        "z_min": args.z_min,
        "x_max": args.x_max,
        "visibility_mode": args.visibility_mode,
        "require_radar_visibility": bool(args.require_radar_visibility),
        "radar_visibility_radius": int(args.radar_visibility_radius),
        "doppler_radius": int(args.doppler_radius),
        "observed_mask_protocol": OBSERVED_MASK_PROTOCOL,
        "observed_mask_source": (
            "lidar_ray_from_preprocessed_lidar_voxel_with_target_domain"
        ),
        "observed_mask_spatial_domain": OBSERVED_MASK_SPATIAL_DOMAIN,
        "observed_mask_pc_range": list(args.pc_range),
        "radar_statistics_protocol": RADAR_STATISTICS_PROTOCOL,
        "radar_statistics_storage": "radar_npz_aligned_with_coords",
        "radar_statistics_fields": {
            "point_count": "uint32_points_per_occupied_voxel",
            "intensity_valid_count": "uint32_finite_intensity_samples_per_occupied_voxel",
            "doppler_valid_count": "uint32_finite_doppler_samples_per_occupied_voxel",
        },
        "radar_aggregation_semantics": "per_field_finite_count_mean_and_doppler_variance_v2",
        "radar_statistics_model_consumed": False,
        "channels": {
            "0": "occupancy",
            "1": "mean_intensity",
            "2": (
                "raw_mean_doppler"
                if args.velocity_mode == "none"
                else "egomotion_compensated_mean_doppler"
            ),
            "3": "clipped_doppler_variance_0_50",
        },
    }
    with open(os.path.join(scene_out_path, "preprocess_policy.json"), "w", encoding="utf-8") as h:
        json.dump(preprocess_policy, h, indent=2)
    provenance_sources = {
        "preprocess_script": os.path.abspath(__file__),
        "radar_to_lidar": args.calib_path,
        "radar_to_thermal": args.radar_to_thermal_path,
        "lidar_to_thermal": args.lidar_to_thermal_path,
        "thermal_intrinsics": args.thermal_intrinsics_path,
        "radar_lidar_sync": radar_lidar_sync_path,
        "radar_ir_sync": ir_sync_path,
        "target_policy": target_policy_path,
    }
    write_scene_manifest_atomic(
        scene_out_path,
        scene_name,
        written,
        provenance_sources,
        profile="training",
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Integrated High-Speed Sensor-Aware Preprocessing")
    parser.add_argument("--raw_data_path", type=str, default=RAW_DATA_PATH)
    parser.add_argument("--index_path", type=str, default=INDEX_PATH)
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH)
    parser.add_argument("--calib_path", type=str, default=CALIB_PATH)
    parser.add_argument(
        "--radar_to_thermal_path",
        type=str,
        default=RADAR_TO_THERMAL_PATH,
    )
    parser.add_argument(
        "--lidar_to_thermal_path",
        type=str,
        default=LIDAR_TO_THERMAL_PATH,
    )
    parser.add_argument(
        "--thermal_intrinsics_path",
        type=str,
        default=THERMAL_INTRINSICS_PATH,
    )
    parser.add_argument("--scene", type=str, default="")
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--invert_calib", action="store_true")
    parser.add_argument("--radar_z_shift", type=float, default=0.0)
    parser.add_argument(
        "--radar_field_schema",
        type=str,
        default="",
        help="Radar 原始字段/单位/Doppler 正方向 JSON artifact",
    )
    parser.add_argument(
        "--require_verified_radar_field_schema",
        action="store_true",
        help="要求 schema 绑定可校验权威证据；formal-v3 必须启用",
    )
    parser.add_argument(
        "--require_complete_extraction_receipt",
        action="store_true",
        help="要求 raw 场景含 complete 的关键模态解包收据",
    )
    parser.add_argument(
        "--align_to",
        choices=VELOCITY_FRAMES,
        default="lidar",
        help="将 Radar/LiDAR 都转换到哪个共享坐标系",
    )
    parser.add_argument("--vx", type=float, default=50.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--vz", type=float, default=0.0)
    parser.add_argument(
        "--velocity_mode",
        choices=VELOCITY_MODES,
        default="none",
        help="运动补偿模式：none 不补偿，fixed 使用显式速度，recorded 按时间戳读取速度表",
    )
    parser.add_argument(
        "--velocity_frame",
        choices=VELOCITY_FRAMES,
        default="radar",
        help="--vx/速度表所在坐标系；只使用旋转转换到 align_to 坐标系",
    )
    parser.add_argument(
        "--velocity_file",
        type=str,
        default="",
        help="recorded 模式的 CSV/空白分隔速度表：timestamp,vx,vy,vz",
    )
    parser.add_argument(
        "--velocity_max_delta",
        type=float,
        default=0.02,
        help="recorded 速度与 Radar 帧时间戳允许的最大差值（秒）",
    )
    parser.add_argument(
        "--radar_lidar_max_delta",
        type=float,
        default=0.045,
        help="Radar-LiDAR 索引允许的最大时间差（秒；由 Step 1 写入记录）",
    )
    parser.add_argument(
        "--radar_ir_max_delta",
        type=float,
        default=0.025,
        help="Radar-IR 最近邻允许的最大时间差（秒），超限直接失败",
    )
    parser.add_argument("--pc_range", type=float, nargs=6, default=(0, -20, -6, 120, 20, 10))
    parser.add_argument("--z_min", type=float, default=-1.0)
    parser.add_argument("--x_max", type=float, default=80.0)
    parser.add_argument(
        "--visibility_mode",
        choices=("preserve", "hard"),
        default="preserve",
        help="preserve keeps cropped LiDAR obstacle structure; hard keeps only radar-neighbor cells",
    )
    parser.add_argument(
        "--require_radar_visibility",
        action="store_true",
        help="Deprecated compatibility alias for --visibility_mode hard",
    )
    parser.add_argument("--radar_visibility_radius", type=int, default=2)
    parser.add_argument("--doppler_radius", type=int, default=1)
    args = parser.parse_args()

    if args.scene:
        scenes = [args.scene]
    else:
        scenes = [d for d in os.listdir(args.raw_data_path) if os.path.isdir(os.path.join(args.raw_data_path, d))]
    print(f"Target integrated preprocessing activated. Scenes: {scenes}")

    failures = []
    for scene in scenes:
        try:
            process_scene_task(
                scene,
                args,
                [args.vx, args.vy, args.vz] if args.velocity_mode == "fixed" else None,
            )
        except Exception as e:
            failures.append((scene, e))
            print(f"Failed to process {scene}: {e}")
            traceback.print_exc()
    if failures:
        raise SystemExit(1)
