# -*- coding: utf-8 -*-

import os
import json
import argparse
from typing import Tuple, Optional, Sequence
import numpy as np
import cv2
from tqdm import tqdm
from scipy.spatial import cKDTree
import scipy.ndimage as ndimage
import pypatchworkpp
from multiprocessing import Pool, cpu_count

# ==============================================================================
# 全局参数配置与常驻内存声明
# ==============================================================================
RAW_DATA_PATH = "./Data/NTU4DRadLM_Raw"
INDEX_PATH = "./Data/NTU4DRadLM_Raw"
OUTPUT_PATH = "./Data/NTU4DRadLM_Pre_sensor_aware"
CALIB_PATH = "./Data/config/calib_radar_to_livox.txt"

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

def _voxel_centers(shape: Sequence[int], pc_range: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx, ny, nz = shape
    x_size = (pc_range[3] - pc_range[0]) / float(nx)
    y_size = (pc_range[4] - pc_range[1]) / float(ny)
    z_size = (pc_range[5] - pc_range[2]) / float(nz)
    x = pc_range[0] + (np.arange(nx, dtype=np.float32) + 0.5) * x_size
    y = pc_range[1] + (np.arange(ny, dtype=np.float32) + 0.5) * y_size
    z = pc_range[2] + (np.arange(nz, dtype=np.float32) + 0.5) * z_size
    return x, y, z

def build_sensor_aware_target_vectorized(
    lidar_voxel: np.ndarray, radar_voxel: np.ndarray, pc_range: tuple,
    z_min: Optional[float], x_max: Optional[float], require_radar_visibility: bool,
    radar_visibility_radius: int, doppler_radius: int
) -> np.ndarray:
    target = np.zeros_like(lidar_voxel, dtype=np.float32)
    lidar_occ = lidar_voxel[..., 0] > 0
    keep = lidar_occ.copy()

    x_centers, _, z_centers = _voxel_centers(lidar_voxel.shape[:3], pc_range)
    if z_min is not None: keep &= z_centers[None, None, :] >= float(z_min)
    if x_max is not None: keep &= x_centers[:, None, None] <= float(x_max)

    radar_occ = radar_voxel[..., 0] > 0
    radar_occ_float = radar_occ.astype(np.float32)

    if require_radar_visibility:
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

# ==============================================================================
# IO 与位姿转换工具
# ==============================================================================

def invert_r_t(r_mat, t_vec):
    return r_mat.T, -np.dot(r_mat.T, t_vec)

def ensure_dir(path):
    if not os.path.exists(path): os.makedirs(path)

def load_calib(calib_file):
    R, T = np.eye(3), np.zeros(3)
    if not os.path.exists(calib_file): raise FileNotFoundError(f"Calibration file not found: {calib_file}.")
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
    return R, T

def transform_pcl(pcl, R, T):
    if pcl.shape[0] == 0: return pcl
    pcl_trans = pcl.copy()
    pcl_trans[:, :3] = np.dot(pcl[:, :3], R.T) + T
    return pcl_trans

def save_voxel(filename, voxel_grid):
    if SAVE_SPARSE:
        occupied = voxel_grid[..., 0] > 0
        coords = np.column_stack(np.where(occupied))
        features = voxel_grid[occupied]
        np.savez(filename, coords=coords, features=features, shape=voxel_grid.shape)
    else:
        np.save(filename, voxel_grid.astype(np.float32))

def voxelize_pcl_airborne_optimized(pcl, voxel_size, pc_range, v_drone=None, dt_sync=0.0):
    """
    重构后的机载自适应点云体素化核心函数

    参数说明:
    pcl: np.ndarray (N, C) -> 原始点云数据。前3列为 x, y, z；第4列为强度特征；第5列为原始相对多普勒速度
    voxel_size: list [3] -> 体素网格的分辨率 [dx, dy, dz] 单位:米
    pc_range: list [6] -> 感知空间的边界 [x_min, y_min, z_min, x_max, y_max, z_max]
    v_drone: array_like [3] -> 无人机当前的瞬时速度绝对值向量 [vx, vy, vz], 单位: m/s
    dt_sync: float -> 红外图像快门或激光与4D雷达帧之间的绝对硬件时钟残差, 单位: 秒
    """
    # 1.机载高动态时空位置微秒级畸变修正
    # 在 70m/s 下，修正因传感器异步产生的帧内空间拉伸模糊
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
    if pcl.shape[0] == 0: return np.zeros(grid_shape + (4,), dtype=np.float32)

    # 4. 展平 3D 矩阵，实现无 Python 循环的高性能散列填充 (Scatter Accumulate)
    x_idx = np.clip(((pcl[:, 0] - pc_range[0]) / voxel_size[0]).astype(np.int32), 0, grid_shape[0] - 1)
    y_idx = np.clip(((pcl[:, 1] - pc_range[1]) / voxel_size[1]).astype(np.int32), 0, grid_shape[1] - 1)
    z_idx = np.clip(((pcl[:, 2] - pc_range[2]) / voxel_size[2]).astype(np.int32), 0, grid_shape[2] - 1)

    voxel_grid = np.zeros(grid_shape + (4,), dtype=np.float32)
    flat_indices = x_idx * (grid_shape[1] * grid_shape[2]) + y_idx * grid_shape[2] + z_idx

    sort_order = np.argsort(flat_indices)
    flat_indices = flat_indices[sort_order]
    features = pcl[sort_order, 3] if pcl.shape[1] > 3 else np.ones(pcl.shape[0])
    doppler = pcl[sort_order, 4] if pcl.shape[1] > 4 else np.zeros(pcl.shape[0])

    unique_indices, unique_counts = np.unique(flat_indices, return_counts=True)
    uz_idx = unique_indices % grid_shape[2]
    uy_idx = (unique_indices // grid_shape[2]) % grid_shape[1]
    ux_idx = (unique_indices // (grid_shape[2] * grid_shape[1]))

    # 通道 0: 空间占用状态 (Occupancy Mask)
    voxel_grid[ux_idx, uy_idx, uz_idx, 0] = 1.0
    # 通道 1: 平均反射特征强度 (Mean Intensity)
    sum_features = np.zeros(np.prod(grid_shape), dtype=np.float32)
    np.add.at(sum_features, flat_indices, features)
    voxel_grid[ux_idx, uy_idx, uz_idx, 1] = sum_features[unique_indices] / unique_counts
    # 通道 2: 经解耦后的纯净多普勒均值测速 (Mean Pure Doppler)
    sum_doppler = np.zeros(np.prod(grid_shape), dtype=np.float32)
    np.add.at(sum_doppler, flat_indices, doppler)
    voxel_grid[ux_idx, uy_idx, uz_idx, 2] = sum_doppler[unique_indices] / unique_counts
    # 通道 3: 多普勒空间方差量化（直接作为不确定性协方差输入）
    # Var = E[X^2] - (E[X])^2
    sum_doppler_sq = np.zeros(np.prod(grid_shape), dtype=np.float32)
    np.add.at(sum_doppler_sq, flat_indices, doppler ** 2)
    mean_doppler_sq = sum_doppler_sq[unique_indices] / unique_counts

    var_doppler = mean_doppler_sq - voxel_grid[ux_idx, uy_idx, uz_idx, 2] ** 2
   # 截断保护，防止随机噪点方差过大导致神经网络训练时出现 NaN 异常
    voxel_grid[ux_idx, uy_idx, uz_idx, 3] = np.clip(var_doppler, 0.0, 50.0)

    return voxel_grid

# ==============================================================================
# 工作子进程单元
# ==============================================================================

def _parallel_frame_worker(task_args):
    global _process_patchwork

    (i, r_file, l_file, current_ts, scene_raw_path, scene_out_path,
     r_radar_to_lidar, t_radar_to_lidar, v_drone, dt_sync,
     thermal_timestamps, thermal_files, thermal_dir, args_dict) = task_args

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

    if args_dict["align_to"] == "lidar":
        radar_pcl = transform_pcl(radar_pcl, r_radar_to_lidar, t_radar_to_lidar)

    r_voxel = voxelize_pcl_airborne_optimized(radar_pcl, VOXEL_SIZE, args_dict["pc_range"], v_drone=v_drone, dt_sync=dt_sync)
    l_voxel = voxelize_pcl_airborne_optimized(lidar_pcl, VOXEL_SIZE, args_dict["pc_range"], v_drone=v_drone, dt_sync=dt_sync)

    target_voxel = build_sensor_aware_target_vectorized(
        lidar_voxel=l_voxel, radar_voxel=r_voxel, pc_range=args_dict["pc_range"],
        z_min=args_dict["z_min"], x_max=args_dict["x_max"],
        require_radar_visibility=args_dict["require_radar_visibility"],
        radar_visibility_radius=args_dict["radar_visibility_radius"],
        doppler_radius=args_dict["doppler_radius"]
    )

    if len(thermal_timestamps) > 0:
        ir_idx = np.argmin(np.abs(thermal_timestamps - current_ts))
        img = cv2.imread(os.path.join(thermal_dir, thermal_files[ir_idx]), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img_3ch = np.stack([cv2.resize(img, (640, 480)).astype(np.float32) / 255.0] * 3, axis=0)
            np.save(os.path.join(scene_out_path, "ir_image", f"{i:06d}_ir.npy"), img_3ch)

    ext = ".npz" if SAVE_SPARSE else ".npy"
    save_voxel(os.path.join(scene_out_path, "radar_voxel", f"{i:06d}{ext}"), r_voxel)
    save_voxel(os.path.join(scene_out_path, "lidar_voxel", f"{i:06d}{ext}"), l_voxel)
    save_voxel(os.path.join(scene_out_path, "target_voxel", f"{i:06d}{ext}"), target_voxel)
    return True

# ==============================================================================
# 场景总控制中心
# ==============================================================================

def process_scene_task(scene_name, args, v_drone, dt_sync):
    print(f"\n⚡ 正在初始化多进程并行流水线，目标场景: {scene_name}")
    scene_raw_path = os.path.join(args.raw_data_path, scene_name)
    scene_index_path = os.path.join(args.index_path, scene_name)
    scene_out_path = os.path.join(args.output_path, scene_name)

    r_radar_to_lidar, t_radar_to_lidar = load_calib(args.calib_path)
    if args.invert_calib: r_radar_to_lidar, t_radar_to_lidar = invert_r_t(r_radar_to_lidar, t_radar_to_lidar)
    if abs(args.radar_z_shift) > 1e-8: t_radar_to_lidar[2] += float(args.radar_z_shift)

    ensure_dir(os.path.join(scene_out_path, "radar_voxel"))
    ensure_dir(os.path.join(scene_out_path, "lidar_voxel"))
    ensure_dir(os.path.join(scene_out_path, "target_voxel"))
    ensure_dir(os.path.join(scene_out_path, "ir_image"))

    thermal_dir = os.path.join(scene_raw_path, "thermal_cam_thermal_image_compressed")
    thermal_files = sorted([f for f in os.listdir(thermal_dir) if f.endswith('.png')]) if os.path.exists(thermal_dir) else []
    thermal_timestamps = np.array([float(os.path.splitext(f)[0]) for f in thermal_files])

    try:
        with open(os.path.join(scene_index_path, "radar_index_sequence.txt"), 'r') as f:
            radar_indices = [int(line.strip()) for line in f.readlines()]
        with open(os.path.join(scene_index_path, "lidar_index_sequence.txt"), 'r') as f:
            lidar_indices = [int(line.strip()) for line in f.readlines()]
    except FileNotFoundError:
        print(f"Index files not found for {scene_name}, skipping.")
        return

    radar_files = sorted([f for f in os.listdir(os.path.join(scene_raw_path, "radar_pcl")) if f.endswith('.npy')])
    lidar_files = sorted([f for f in os.listdir(os.path.join(scene_raw_path, "livox_lidar")) if f.endswith('.npy')])

    min_len = min(len(radar_indices), len(lidar_indices))
    if args.max_frames > 0: min_len = min(min_len, int(args.max_frames))

    args_dict = vars(args)
    worker_tasks = []
    for i in range(min_len):
        r_idx, l_idx = radar_indices[i], lidar_indices[i]
        if r_idx >= len(radar_files) or l_idx >= len(lidar_files): continue
        current_ts = float(os.path.splitext(radar_files[r_idx])[0])

        worker_tasks.append((
            i, radar_files[r_idx], lidar_files[l_idx], current_ts,
            scene_raw_path, scene_out_path, r_radar_to_lidar, t_radar_to_lidar,
            v_drone, dt_sync, thermal_timestamps, thermal_files, thermal_dir, args_dict
        ))

    num_workers = min(cpu_count(), len(worker_tasks), 16)

    # ──► 核心重构：使用 initializer 绑定进程启动钩子，每个进程终生只打印一次初始化日志！
    print(f"🔥 正在拉起 {num_workers} 个并行的常驻感知 Worker...")
    written = 0
    with Pool(processes=num_workers, initializer=_init_worker_patchwork) as pool:
        for _ in tqdm(pool.imap_unordered(_parallel_frame_worker, worker_tasks), total=len(worker_tasks), desc=f"Parallel {scene_name}"):
            written += 1

    metadata = {
        "source_scene": scene_name, "frames_written": written,
        "policy": {
            "z_min": args.z_min, "x_max": args.x_max,
            "require_radar_visibility": args.require_radar_visibility,
            "radar_visibility_radius": args.radar_visibility_radius, "doppler_radius": args.doppler_radius
        }
    }
    with open(os.path.join(scene_out_path, "target_policy.json"), "w", encoding="utf-8") as h:
        json.dump(metadata, h, indent=2)
    preprocess_policy = {
        "source_scene": scene_name,
        "frames_written": written,
        "pc_range": list(args.pc_range),
        "voxel_size": list(VOXEL_SIZE),
        "align_to": args.align_to,
        "invert_calib": bool(args.invert_calib),
        "radar_z_shift": float(args.radar_z_shift),
        "v_drone": [float(v_drone[0]), float(v_drone[1]), float(v_drone[2])],
        "dt_sync": float(dt_sync),
        "z_min": args.z_min,
        "x_max": args.x_max,
        "require_radar_visibility": bool(args.require_radar_visibility),
        "radar_visibility_radius": int(args.radar_visibility_radius),
        "doppler_radius": int(args.doppler_radius),
        "channels": {
            "0": "occupancy",
            "1": "mean_intensity",
            "2": "egomotion_compensated_mean_doppler",
            "3": "clipped_doppler_variance_0_50",
        },
    }
    with open(os.path.join(scene_out_path, "preprocess_policy.json"), "w", encoding="utf-8") as h:
        json.dump(preprocess_policy, h, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Integrated High-Speed Sensor-Aware Preprocessing")
    parser.add_argument("--raw_data_path", type=str, default=RAW_DATA_PATH)
    parser.add_argument("--index_path", type=str, default=INDEX_PATH)
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH)
    parser.add_argument("--calib_path", type=str, default=CALIB_PATH)
    parser.add_argument("--scene", type=str, default="")
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--invert_calib", action="store_true")
    parser.add_argument("--radar_z_shift", type=float, default=0.0)
    parser.add_argument("--align_to", type=str, default="lidar")
    parser.add_argument("--vx", type=float, default=50.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--vz", type=float, default=0.0)
    parser.add_argument("--dt_sync", type=float, default=0.002)

    parser.add_argument("--pc_range", type=float, nargs=6, default=(0, -20, -6, 120, 20, 10))
    parser.add_argument("--z_min", type=float, default=-1.0)
    parser.add_argument("--x_max", type=float, default=80.0)
    parser.add_argument("--require_radar_visibility", action="store_true")
    parser.add_argument("--radar_visibility_radius", type=int, default=2)
    parser.add_argument("--doppler_radius", type=int, default=1)
    args = parser.parse_args()

    if args.scene:
        scenes = [args.scene]
    else:
        scenes = [d for d in os.listdir(args.raw_data_path) if os.path.isdir(os.path.join(args.raw_data_path, d))]
    print(f"Target integrated preprocessing activated. Scenes: {scenes}")

    for scene in scenes:
        try: process_scene_task(scene, args, [args.vx, args.vy, args.vz], args.dt_sync)
        except Exception as e:
            print(f"Failed to process {scene}: {e}")
            import traceback; traceback.print_exc()
