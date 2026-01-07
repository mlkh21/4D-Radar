# -*- coding: utf-8 -*-

import os
import numpy as np
import argparse
from multiprocessing import Pool
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree
import pypatchworkpp

# 路径
RAW_DATA_PATH = "/home/zxj/catkin_ws/src/4D-Radar-Diffusion/NTU4DRadLM_pre_processing/NTU4DRadLM_Raw" # 原始数据存放路径
INDEX_PATH = "/home/zxj/catkin_ws/src/4D-Radar-Diffusion/NTU4DRadLM_pre_processing/NTU4DRadLM_Raw" # 时间戳索引文件存放路径
OUTPUT_PATH = "/home/zxj/catkin_ws/src/4D-Radar-Diffusion/NTU4DRadLM_pre_processing/NTU4DRadLM_Pre" # 预处理后数据存放路径
CALIB_PATH = "/home/zxj/catkin_ws/src/4D-Radar-Diffusion/NTU4DRadLM_pre_processing/config/calib_radar_to_livox.txt" # 标定文件路径

# 参数
VOXEL_SIZE = [0.4, 0.4, 0.4] # 体素像素 [x, y, z] 单位：米
PC_RANGE = [0, -20, -6, 120, 20, 10] # [x_min, y_min, z_min, x_max, y_max, z_max] 单位：米
MAX_RANGE = 250.0 # 最大探测距离，单位：米
RANGE_BINS = 256 # 距离方向网格数
AZIMUTH_BINS = 128 # -90 到 90 度

SAVE_SPARSE = True # 是否使用稀疏格式存储体素 (.npz)
GENERATE_VISUALIZATION = False # 是否生成可视化文件 (Mesh, Heatmap, BEV, PLY)

def ensure_dir(path):
    """
    输入: path (str) - 需要创建的目录路径
    输出: 无
    作用: 确保目录存在，避免后续写入失败。
    逻辑: 检查路径是否存在，若不存在则调用os.makedirs创建。
    """
    if not os.path.exists(path):
        os.makedirs(path)

def load_calib(calib_file):
    """
    输入: 
        calib_file (str) - 标定文件的绝对路径
    输出: 
        R (np.array 3x3) - 旋转矩阵
        T (np.array 3,) - 平移向量
    作用: 从标定文件中加载旋转和平移参数并返回。
    逻辑: 读取文件，解析以'R:'和'T:'开头的行，将数值转换为numpy数组。若文件不存在返回单位矩阵和零向量。
    """
    print(f"Loading calibration from: {calib_file}") # 添加这一行
    R = np.eye(3)
    T = np.zeros(3)
    
    if not os.path.exists(calib_file):
        print(f"Warning: Calib file {calib_file} not found. Using identity transform.")
        return R, T

    with open(calib_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split(':')
            if len(parts) < 2: continue
            
            key = parts[0].strip()
            raw_parts = parts[1].strip().split()
            vals = []
            for x in raw_parts:
                try:
                    vals.append(float(x))
                except ValueError:
                    continue # 跳过无法转换为 float 的部分（例如 'use'）
            
            if key == 'R':
                R = np.array(vals).reshape(3, 3)
            elif key == 'T':
                T = np.array(vals)
    return R, T

def transform_pcl(pcl, R, T):
    """
    输入: 
        pcl (np.array NxM) - 原始点云数据，前三列必须是x, y, z
        R (np.array 3x3) - 旋转矩阵
        T (np.array 3,) - 平移向量
    输出: 
        pcl_trans (np.array NxM) - 变换后的点云数据
    作用: 将点云从一个坐标系变换到另一个坐标系 (例如 Radar -> LiDAR)。
    逻辑: 应用公式 p_new = R * p_old + T 对点云前三列执行仿射变换，保留其余维度。
    """
    if pcl.shape[0] == 0:
        return pcl
        
    xyz = pcl[:, :3]
    # R is 3x3, xyz is Nx3. Formula is R * p_r.
    # So we need (R @ xyz.T).T + T  =>  xyz @ R.T + T
    xyz_trans = np.dot(xyz, R.T) + T
    
    pcl_trans = pcl.copy()
    pcl_trans[:, :3] = xyz_trans
    return pcl_trans

def save_sparse_voxel(filename, voxel_grid):
    """保存稀疏体素（只存储非空体素）"""
    occupied = voxel_grid[..., 0] > 0
    coords = np.column_stack(np.where(occupied))  # 获取非空体素坐标
    features = voxel_grid[occupied]  # 获取对应特征
    
    np.savez_compressed(
        filename,
        coords=coords,
        features=features,
        shape=voxel_grid.shape
    )

def voxelize_pcl(pcl, voxel_size, pc_range):
    """
    输入: 
        pcl (np.array NxC) - 点云数据，前三列为x,y,z，第四列为特征(如强度)，第五列为多普勒(可选)
        voxel_size (list [3]) - 体素大小 [x_size, y_size, z_size]
        pc_range (list [6]) - 点云范围 [x_min, y_min, z_min, x_max, y_max, z_max]
    输出: 
        voxel_grid (np.array H, W, Z, 4) - 体素网格
            通道0: 占用率 (Occupancy)
            通道1: 平均特征 (Mean Intensity)
            通道2: 平均多普勒 (Mean Doppler)
            通道3: 多普勒方差 (Doppler Variance)
    作用: 将点云体素化，生成占用网格和特征网格，便于可视化。
    逻辑: 过滤区域外点，计算栅格索引，累积特征并求均值/方差填充。
    """
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
        return np.zeros(grid_shape + (4,), dtype=np.float32)

    # 计算体素索引
    x_idx = ((pcl[:, 0] - pc_range[0]) / voxel_size[0]).astype(np.int32)
    y_idx = ((pcl[:, 1] - pc_range[1]) / voxel_size[1]).astype(np.int32)
    z_idx = ((pcl[:, 2] - pc_range[2]) / voxel_size[2]).astype(np.int32)
    
    # Clip indices to be safe
    x_idx = np.clip(x_idx, 0, grid_shape[0] - 1)
    y_idx = np.clip(y_idx, 0, grid_shape[1] - 1)
    z_idx = np.clip(z_idx, 0, grid_shape[2] - 1)
    
    voxel_grid = np.zeros(grid_shape + (4,), dtype=np.float32)
    
    flat_indices = x_idx * (grid_shape[1] * grid_shape[2]) + y_idx * grid_shape[2] + z_idx
    
    # Sort by index
    sort_order = np.argsort(flat_indices)
    flat_indices = flat_indices[sort_order]
    
    # Extract features
    features = pcl[sort_order, 3] if pcl.shape[1] > 3 else np.ones(pcl.shape[0]) # Use 4th column as feature (Intensity)
    doppler = pcl[sort_order, 4] if pcl.shape[1] > 4 else np.zeros(pcl.shape[0]) # Use 5th column as Doppler
    
    unique_indices, unique_counts = np.unique(flat_indices, return_counts=True)
    
    # Map back to 3D
    uz_idx = unique_indices % grid_shape[2]
    uy_idx = (unique_indices // grid_shape[2]) % grid_shape[1]
    ux_idx = (unique_indices // (grid_shape[2] * grid_shape[1]))
    
    # Fill occupancy (Channel 0)
    voxel_grid[ux_idx, uy_idx, uz_idx, 0] = 1.0 
    
    # Fill mean feature (Channel 1)
    sum_features = np.zeros(np.prod(grid_shape), dtype=np.float32)
    np.add.at(sum_features, flat_indices, features)
    mean_features = sum_features[unique_indices] / unique_counts
    voxel_grid[ux_idx, uy_idx, uz_idx, 1] = mean_features
    
    # Fill mean doppler (Channel 2)
    sum_doppler = np.zeros(np.prod(grid_shape), dtype=np.float32)
    np.add.at(sum_doppler, flat_indices, doppler)
    mean_doppler = sum_doppler[unique_indices] / unique_counts
    voxel_grid[ux_idx, uy_idx, uz_idx, 2] = mean_doppler
    
    # Fill doppler variance (Channel 3)
    # Var = E[X^2] - (E[X])^2
    sum_doppler_sq = np.zeros(np.prod(grid_shape), dtype=np.float32)
    np.add.at(sum_doppler_sq, flat_indices, doppler ** 2)
    mean_doppler_sq = sum_doppler_sq[unique_indices] / unique_counts
    var_doppler = mean_doppler_sq - mean_doppler ** 2
    voxel_grid[ux_idx, uy_idx, uz_idx, 3] = var_doppler
    
    return voxel_grid

def generate_ra_heatmap(pcl, max_range, r_bins, a_bins, feature_idx=3):
    """
    输入: 
        pcl (np.array NxC) - 点云数据
        max_range (float) - 最大探测距离
        r_bins (int) - 距离方向的网格数
        a_bins (int) - 方位角方向的网格数
        feature_idx (int) - 用于生成热图的特征列索引 (默认3: 强度, 4: 多普勒)
    输出: 
        heatmap (np.array r_bins, a_bins) - 距离-方位热图
    作用: 将点云投影到距离-方位(Range-Azimuth)平面生成热图。
    逻辑: 计算距离与方位，映射到网格并按特征累加。
    """
    r = np.sqrt(pcl[:, 0]**2 + pcl[:, 1]**2)
    azimuth = np.arctan2(pcl[:, 1], pcl[:, 0]) # -pi to pi
    
    keep = (r < max_range) & (np.abs(azimuth) < np.pi/2)
    pcl = pcl[keep]
    r = r[keep]
    azimuth = azimuth[keep]
    
    if pcl.shape[0] == 0:
        return np.zeros((r_bins, a_bins), dtype=np.float32)
        
    r_idx = (r / max_range * r_bins).astype(np.int32)
    a_idx = ((azimuth + np.pi/2) / np.pi * a_bins).astype(np.int32)
    
    # 将索引限制在安全范围
    r_idx = np.clip(r_idx, 0, r_bins - 1)
    a_idx = np.clip(a_idx, 0, a_bins - 1)
    
    heatmap = np.zeros((r_bins, a_bins), dtype=np.float32)
    
    # 
    feature = pcl[:, feature_idx] if pcl.shape[1] > feature_idx else np.ones(pcl.shape[0])
    
    # 将二维坐标转化为一维索引进行累加
    flat_indices = r_idx * a_bins + a_idx
    np.add.at(heatmap.ravel(), flat_indices, feature)
    
    return heatmap

def generate_bev(pcl, pc_range, voxel_size):
    """
    输入: 
        pcl (np.array NxC) - 点云数据
        pc_range (list [6]) - 点云范围
        voxel_size (list [3]) - 体素大小
    输出: 
        bev_map (np.array H, W, 3) - 鸟瞰图特征，通道分别为: 占用率, 平均强度, 平均高度
    作用: 生成鸟瞰图(BEV)特征图。
    逻辑: 过滤范围外点，统计每个xy网格的占用、强度与高度均值。
    """
    keep = (pcl[:, 0] >= pc_range[0]) & (pcl[:, 0] < pc_range[3]) & \
           (pcl[:, 1] >= pc_range[1]) & (pcl[:, 1] < pc_range[4])
    pcl = pcl[keep]
    
    grid_shape = (
        int((pc_range[3] - pc_range[0]) / voxel_size[0]),
        int((pc_range[4] - pc_range[1]) / voxel_size[1])
    )
    
    bev_map = np.zeros(grid_shape + (3,), dtype=np.float32) 
    
    if pcl.shape[0] == 0:
        return bev_map
        
    x_idx = ((pcl[:, 0] - pc_range[0]) / voxel_size[0]).astype(np.int32)
    y_idx = ((pcl[:, 1] - pc_range[1]) / voxel_size[1]).astype(np.int32)
    
    x_idx = np.clip(x_idx, 0, grid_shape[0] - 1)
    y_idx = np.clip(y_idx, 0, grid_shape[1] - 1)
    
    flat_indices = x_idx * grid_shape[1] + y_idx
    
    feature = pcl[:, 3] if pcl.shape[1] > 3 else np.ones(pcl.shape[0])
    height = pcl[:, 2]
    
    # 占用率计算（采用唯一索引方法，确保每个网格只计一次）
    unique_indices = np.unique(flat_indices) # 获取唯一索引
    ux_idx = unique_indices // grid_shape[1] # 获取x索引
    uy_idx = unique_indices % grid_shape[1]  # 获取y索引
    bev_map[ux_idx, uy_idx, 0] = 1.0 # 通道0: 占用率设为1
    
    # 计算平均强度与高度
    sum_feat = np.zeros(np.prod(grid_shape), dtype=np.float32)
    sum_height = np.zeros(np.prod(grid_shape), dtype=np.float32)
    count_feat = np.zeros(np.prod(grid_shape), dtype=np.float32)
    
    np.add.at(sum_feat, flat_indices, feature)
    np.add.at(sum_height, flat_indices, height)
    np.add.at(count_feat, flat_indices, 1)
    
    valid = count_feat > 0
    
    mean_feat = np.zeros_like(sum_feat)
    mean_feat[valid] = sum_feat[valid] / count_feat[valid]
    bev_map[:, :, 1] = mean_feat.reshape(grid_shape)
    
    mean_height = np.zeros_like(sum_height)
    mean_height[valid] = sum_height[valid] / count_feat[valid]
    bev_map[:, :, 2] = mean_height.reshape(grid_shape)
    
    return bev_map

def save_ply(filename, pcl):
    """
    输入:
        filename (str) - 输出PLY路径
        pcl (np.array Nx4) - 点云数据，包括强度
    输出:
        无
    作用:
        以ASCII格式导出点云和强度到PLY文件。
    逻辑:
        构造PLY头部并使用numpy.savetxt导出前4列。
    """
    if pcl.shape[0] == 0:
        return
    
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(pcl)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float intensity\n"
        "end_header"
    )
    
    # Ensure we have 4 columns
    if pcl.shape[1] < 4:
        pcl = np.hstack((pcl[:, :3], np.zeros((pcl.shape[0], 1))))
    
    np.savetxt(filename, pcl[:, :4], fmt='%.6f', header=header, comments='')

def save_heatmap_png(filename, heatmap):
    """
    输入:
        filename (str) - 输出PNG路径
        heatmap (np.array R x A) - 热图矩阵
    输出:
        无
    作用:
        将热图以伪彩色保存为PNG。
    逻辑:
        使用matplotlib imsave保存图片并指定‘jet’颜色映射。
    """
    plt.imsave(filename, heatmap, cmap='jet')

def save_bev_png(filename, bev_map):
    """
    输入:
        filename (str) - 输出PNG路径
        bev_map (np.array H x W x 3) - BEV特征图
    输出:
        无
    作用:
        将BEV的强度通道保存为灰度图像。
    逻辑:
        取bev_map[:, :, 1]并调用plt.imsave，使用灰度颜色映射。
    """
    # Use channel 1 (Intensity) for visualization
    plt.imsave(filename, bev_map[:, :, 1], cmap='gray')

def save_voxel_obj(filename, voxel_grid, voxel_size):
    """
    输入:
        filename (str) - 输出OBJ路径
        voxel_grid (np.array HxWxZx2) - 体素网格
        voxel_size (list [3]) - 体素尺寸
    输出:
        无
    作用:
        将被占据的体素以带颜色的立方体写入OBJ。
    逻辑:
        遍历被占据的体素，映射特征到jet颜色，写顶点和面。
    """
    occupied = voxel_grid[..., 0] > 0
    if not np.any(occupied):
        return

    cx, cy, cz = np.where(occupied)
    features = voxel_grid[cx, cy, cz, 1]
    
    # Normalize features for color mapping
    f_min, f_max = features.min(), features.max()
    if f_max > f_min:
        norm_features = (features - f_min) / (f_max - f_min)
    else:
        norm_features = np.zeros_like(features)
    
    # Get RGB colors from colormap (jet)
    cmap = plt.get_cmap('jet')
    rgba = cmap(norm_features) # N x 4
    rgb = rgba[:, :3] # N x 3
    
    # Voxel dimensions
    dx, dy, dz = voxel_size
    
    # Relative vertices of a cube
    v_rel = np.array([
        [0, 0, 0], [dx, 0, 0], [dx, dy, 0], [0, dy, 0],
        [0, 0, dz], [dx, 0, dz], [dx, dy, dz], [0, dy, dz]
    ])

    # Faces (indices into v_rel, 1-based for OBJ logic later)
    faces_rel = np.array([
        [1, 2, 3, 4], [5, 8, 7, 6], [1, 5, 6, 2],
        [2, 6, 7, 3], [3, 7, 8, 4], [4, 8, 5, 1]
    ])

    with open(filename, 'w') as f:
        f.write(f"# Voxel Grid OBJ\n")
        vertex_offset = 0
        for i in range(len(cx)):
            x, y, z = cx[i] * dx, cy[i] * dy, cz[i] * dz
            c = rgb[i]
            for v in v_rel:
                f.write(f"v {x + v[0]:.4f} {y + v[1]:.4f} {z + v[2]:.4f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\n")
            for face in faces_rel:
                f.write(f"f {face[0] + vertex_offset} {face[1] + vertex_offset} {face[2] + vertex_offset} {face[3] + vertex_offset}\n")
            vertex_offset += 8

def lidar_filtering(lidar_pcl, patchwork):
    """
    输入:
        lidar_pcl (np.array Nx4) - LiDAR点云（含强度）
        patchwork (pypatchworkpp instance) - 已初始化的Patchwork++对象
    输出:
        filtered (np.array N'x4) - 去除地面的点云
    作用:
        使用Patchwork++移除地面点并保留强度信息。
    逻辑:
        给Patchwork++传入完整点云，尝试获取非地面索引或根据坐标回溯。
    """
    if lidar_pcl.shape[0] == 0:
        return lidar_pcl
        
    
    patchwork.estimateGround(lidar_pcl)
    
    # 获取非地面点索引
    try:
        indices = patchwork.getNongroundIndices()
        return lidar_pcl[indices]
    except AttributeError:
        # Fallback if getNongroundIndices is not available
        nonground = patchwork.getNonground()
        if nonground.shape[0] == 0:
            return np.zeros((0, lidar_pcl.shape[1]))
            
        # If nonground preserves dimensions (Nx4), return it
        if nonground.shape[1] == lidar_pcl.shape[1]:
             return nonground

        # Match back to original to keep intensity
        tree = cKDTree(lidar_pcl[:, :3])
        _, idx = tree.query(nonground[:, :3] if nonground.shape[1] > 3 else nonground, k=1)
        return lidar_pcl[idx]

def radar_filtering_by_lidar(radar_pcl, lidar_pcl, threshold=0.5):
    """
    输入:
        radar_pcl (np.array Nx4) - Radar点云
        lidar_pcl (np.array Mx4) - LiDAR点云
        threshold (float) - 过滤阈值（米）
    输出:
        filtered (np.array Kx4) - 与LiDAR邻近的Radar点云
    作用:
        通过KDTree去除距离大于阈值的Radar点。
    逻辑:
        使用LiDAR点构建KDTree，查询每个Radar点的最近点并按距离过滤。
    """
    if radar_pcl.shape[0] == 0 or lidar_pcl.shape[0] == 0:
        return radar_pcl
    
    # 构建KDTree
    lidar_tree = cKDTree(lidar_pcl[:, :3])
    
    # Query nearest neighbors for Radar points
    # k=1 returns distance to the nearest neighbor
    dists, _ = lidar_tree.query(radar_pcl[:, :3], k=1)
    
    # Filter
    mask = dists < threshold
    return radar_pcl[mask]

def generate_target_voxel(lidar_voxel, radar_voxel):
    """
    输入:
        lidar_voxel (H, W, Z, 4) - LiDAR体素 (Occ, Int, 0, 0)
        radar_voxel (H, W, Z, 4) - Radar体素 (Occ, Int, Dop, Var)
    输出:
        target_voxel (H, W, Z, 4) - 融合后的训练目标
            Ch0: LiDAR Occupancy
            Ch1: LiDAR Intensity
            Ch2: Radar Doppler (Masked by LiDAR Occ)
            Ch3: Doppler Mask (1 if valid, 0 else)
    作用:
        生成用于训练的 Ground Truth。
        Doppler 通道仅在 LiDAR 存在的地方从 Radar 继承，
        或者如果 LiDAR 存在但 Radar 不存在，则设为 0 (或保持缺失)。
        这里采用策略：
        - Occ/Int 来自 LiDAR (高精度几何)
        - Doppler 来自 Radar，但仅保留与 LiDAR 重叠的部分。
    """
    target = np.zeros_like(lidar_voxel)
    
    # 1. Geometry from LiDAR
    target[..., 0] = lidar_voxel[..., 0] # Occupancy
    target[..., 1] = lidar_voxel[..., 1] # Intensity
    
    # 2. Doppler from Radar, masked by LiDAR Occupancy
    # 只有当 LiDAR 认为该处有物体 (Occ > 0) 且 Radar 也有读数 (Occ > 0) 时，才信任 Radar 的速度
    # 或者：只要 LiDAR 有物体，就尝试去取 Radar 的速度（如果 Radar 在该体素有值）
    
    lidar_occ = lidar_voxel[..., 0] > 0
    radar_occ = radar_voxel[..., 0] > 0
    
    # Mask: LiDAR 有物体 AND Radar 也有物体
    valid_doppler_mask = lidar_occ & radar_occ
    
    # 填入 Doppler
    target[..., 2][valid_doppler_mask] = radar_voxel[..., 2][valid_doppler_mask]
    
    # 填入 Mask (作为第4通道，供 Loss 使用)
    target[..., 3] = valid_doppler_mask.astype(np.float32)
    
    return target

def process_scene_task(scene_name):
    
    """
    输入: 
        scene_name (str) - 场景名称 (例如 'carpark', 'garden')
    输出: 
        无 (处理后的数据将保存到 OUTPUT_PATH 对应的场景目录下)
    作用: 处理单个场景的所有帧数据。
    逻辑:
    1. 加载标定参数。
    2. 读取Radar和LiDAR的时间戳索引文件。
    3. 遍历每一帧，加载Radar和LiDAR点云。
    4. 将Radar点云变换到LiDAR坐标系。
    5. 生成并保存: 原始点云, Voxel, Heatmap, BEV。
    """

    print(f"Processing scene: {scene_name}")
    
    scene_raw_path = os.path.join(RAW_DATA_PATH, scene_name)
    scene_index_path = os.path.join(INDEX_PATH, scene_name)
    scene_out_path = os.path.join(OUTPUT_PATH, scene_name)
    
    # 加载标定参数
    R, T = load_calib(CALIB_PATH)
    print(f"Loaded calibration for {scene_name}")

    # 使用 Patchwork++ 初始化地面移除器
    params = pypatchworkpp.Parameters()
    params.verbose = False
    params.enable_RNR = True
    patchwork = pypatchworkpp.patchworkpp(params)
    
    # 创建输出目录
    ensure_dir(os.path.join(scene_out_path, "radar_pcl"))
    ensure_dir(os.path.join(scene_out_path, "lidar_pcl"))
    ensure_dir(os.path.join(scene_out_path, "radar_voxel"))
    ensure_dir(os.path.join(scene_out_path, "radar_mesh"))
    ensure_dir(os.path.join(scene_out_path, "radar_heatmap"))
    ensure_dir(os.path.join(scene_out_path, "radar_doppler_heatmap")) 
    ensure_dir(os.path.join(scene_out_path, "radar_bev"))
    ensure_dir(os.path.join(scene_out_path, "lidar_voxel"))
    ensure_dir(os.path.join(scene_out_path, "lidar_mesh"))
    ensure_dir(os.path.join(scene_out_path, "lidar_bev"))
    ensure_dir(os.path.join(scene_out_path, "target_voxel")) 

    # 读取索引文件
    try:
        with open(os.path.join(scene_index_path, "radar_index_sequence.txt"), 'r') as f:
            radar_indices = [int(line.strip()) for line in f.readlines()]
        with open(os.path.join(scene_index_path, "lidar_index_sequence.txt"), 'r') as f:
            lidar_indices = [int(line.strip()) for line in f.readlines()]
    except FileNotFoundError:
        print(f"Index files not found for {scene_name}, skipping.")
        return

    # 列出文件
    radar_files = sorted([f for f in os.listdir(os.path.join(scene_raw_path, "radar_pcl")) if f.endswith('.npy')])
    lidar_files = sorted([f for f in os.listdir(os.path.join(scene_raw_path, "livox_lidar")) if f.endswith('.npy')])
    
    min_len = min(len(radar_indices), len(lidar_indices))
    
    for i in range(min_len):
        r_idx = radar_indices[i]
        l_idx = lidar_indices[i]
        
        if r_idx >= len(radar_files) or l_idx >= len(lidar_files):
            continue
            
        r_file = radar_files[r_idx]
        l_file = lidar_files[l_idx]
        
        # 加载原始数据
        radar_pcl = np.load(os.path.join(scene_raw_path, "radar_pcl", r_file))
        lidar_pcl = np.load(os.path.join(scene_raw_path, "livox_lidar", l_file))
        
        # 将Radar原始数据与LiDAR空间对齐
        radar_pcl = transform_pcl(radar_pcl, R, T)
        
        # 备份原始LiDAR数据
        lidar_pcl_raw = lidar_pcl.copy()
        
        # 移除LiDAR地面点
        lidar_pcl = lidar_filtering(lidar_pcl, patchwork)
        
        # ? 通过原始LiDAR数据对Radar点云进行过滤（判断Radar点云的是否在LiDAR点云附近）
        # radar_pcl = radar_filtering_by_lidar(radar_pcl, lidar_pcl_raw, threshold=0.5)

        # 将处理后的点云数据保存为.ply格式
        if GENERATE_VISUALIZATION:
            save_ply(os.path.join(scene_out_path, "radar_pcl", f"{i:06d}.ply"), radar_pcl)
            save_ply(os.path.join(scene_out_path, "lidar_pcl", f"{i:06d}.ply"), lidar_pcl)
        
        # 生成LiDAR和Radar的体素化、热图和BEV表示并保存 
        r_voxel = voxelize_pcl(radar_pcl, VOXEL_SIZE, PC_RANGE)
        
        if SAVE_SPARSE:
            save_sparse_voxel(os.path.join(scene_out_path, "radar_voxel", f"{i:06d}.npz"), r_voxel)
        else:
            np.save(os.path.join(scene_out_path, "radar_voxel", f"{i:06d}.npy"), r_voxel)
            
        if GENERATE_VISUALIZATION:
            save_voxel_obj(os.path.join(scene_out_path, "radar_mesh", f"{i:06d}.obj"), r_voxel, VOXEL_SIZE)
        
            r_heatmap = generate_ra_heatmap(radar_pcl, MAX_RANGE, RANGE_BINS, AZIMUTH_BINS, feature_idx=3)
            save_heatmap_png(os.path.join(scene_out_path, "radar_heatmap", f"{i:06d}.png"), r_heatmap)
            
            r_doppler_heatmap = generate_ra_heatmap(radar_pcl, MAX_RANGE, RANGE_BINS, AZIMUTH_BINS, feature_idx=4)
            save_heatmap_png(os.path.join(scene_out_path, "radar_doppler_heatmap", f"{i:06d}.png"), r_doppler_heatmap)
            
            r_bev = generate_bev(radar_pcl, PC_RANGE, VOXEL_SIZE)
            save_bev_png(os.path.join(scene_out_path, "radar_bev", f"{i:06d}.png"), r_bev)
        
        l_voxel = voxelize_pcl(lidar_pcl, VOXEL_SIZE, PC_RANGE)
        
        if SAVE_SPARSE:
             save_sparse_voxel(os.path.join(scene_out_path, "lidar_voxel", f"{i:06d}.npz"), l_voxel)
        else:
            np.save(os.path.join(scene_out_path, "lidar_voxel", f"{i:06d}.npy"), l_voxel)

        if GENERATE_VISUALIZATION:
            save_voxel_obj(os.path.join(scene_out_path, "lidar_mesh", f"{i:06d}.obj"), l_voxel, VOXEL_SIZE)
        
            l_bev = generate_bev(lidar_pcl, PC_RANGE, VOXEL_SIZE)
            save_bev_png(os.path.join(scene_out_path, "lidar_bev", f"{i:06d}.png"), l_bev)
        
        # 生成训练目标 (Target Voxel)
        target_voxel = generate_target_voxel(l_voxel, r_voxel)
        if SAVE_SPARSE:
            save_sparse_voxel(os.path.join(scene_out_path, "target_voxel", f"{i:06d}.npz"), target_voxel)
        else:
            np.save(os.path.join(scene_out_path, "target_voxel", f"{i:06d}.npy"), target_voxel)
        
        if i % 100 == 0:
            print(f"Scene {scene_name}: Processed {i}/{min_len} frames")

if __name__ == "__main__":
    scenes = [d for d in os.listdir(RAW_DATA_PATH) if os.path.isdir(os.path.join(RAW_DATA_PATH, d))]
    print(f"Found scenes: {scenes}")
    
    # 调试模式：单进程运行，以便捕获错误和防止内存溢出
    # 如果需要并行，请取消注释下方的 Pool 代码
    for scene in scenes:
        try:
            process_scene_task(scene)
        except Exception as e:
            print(f"Error processing scene {scene}: {e}")
            import traceback
            traceback.print_exc()

    # with Pool(processes=4) as pool:
    #     pool.map(process_scene_task, scenes)
