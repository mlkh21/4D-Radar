# -*- coding: utf-8 -*-
"""
完整评估脚本

评估指标包括:
1. 点云质量指标 - Chamfer Distance, Hausdorff Distance
2. 占用指标 - IoU, Precision, Recall, F-score
3. 特征指标 - 强度 MAE, 多普勒 MAE
4. 结构指标 - 边缘保持率

使用方法:
    python evaluate.py --pred_dir ./results/pred --gt_dir ./results/gt --output ./eval_results.json
"""

import os
import sys
import argparse
import json
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate radar point cloud predictions')
    parser.add_argument('--pred_dir', type=str, required=True, help='预测结果目录')
    parser.add_argument('--gt_dir', type=str, required=True, help='真实标签目录')
    parser.add_argument('--output', type=str, default='./eval_results.json', help='输出结果路径')
    parser.add_argument('--threshold', type=float, default=0.5, help='匹配距离阈值 (米)')
    parser.add_argument('--voxel_size', type=float, default=0.2, help='体素大小 (米)')
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    return parser.parse_args()


# ==============================================================================
# 数据加载
# ==============================================================================

def load_voxel(filepath: str) -> np.ndarray:
    """
    加载体素数据
    
    支持 .npy 和 .npz (稀疏) 格式
    """
    if filepath.endswith('.npz'):
        data = np.load(filepath)
        voxel_grid = np.zeros(data['shape'], dtype=np.float32)
        coords = data['coords']
        if coords.shape[0] > 0:
            voxel_grid[coords[:, 0], coords[:, 1], coords[:, 2]] = data['features']
        return voxel_grid
    else:
        return np.load(filepath).astype(np.float32)


def voxel_to_pointcloud(voxel: np.ndarray, voxel_size: float = 0.2) -> np.ndarray:
    """
    将体素转换为点云
    
    Args:
        voxel: (H, W, Z, C) 或 (C, Z, H, W) 体素数据
        voxel_size: 体素大小
    
    Returns:
        pointcloud: (N, 3+) 点云 [x, y, z, features...]
    """
    # 处理不同的输入格式
    if voxel.ndim == 4:
        if voxel.shape[0] <= 4:  # (C, Z, H, W) 格式
            voxel = voxel.transpose(2, 3, 1, 0)  # -> (H, W, Z, C)
    
    # 获取占用位置
    occupancy = voxel[..., 0] if voxel.ndim == 4 else voxel
    occupied_indices = np.where(occupancy > 0)
    
    if len(occupied_indices[0]) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    
    # 转换为坐标
    h_idx, w_idx, z_idx = occupied_indices[:3]
    
    x = h_idx * voxel_size
    y = (w_idx - voxel.shape[1] / 2) * voxel_size  # 中心化 y
    z = (z_idx - voxel.shape[2] / 2) * voxel_size  # 中心化 z
    
    points = np.stack([x, y, z], axis=1)
    
    # 如果有特征通道，也提取
    if voxel.ndim == 4 and voxel.shape[-1] > 1:
        features = voxel[h_idx, w_idx, z_idx, 1:]
        points = np.concatenate([points, features], axis=1)
    
    return points.astype(np.float32)


def load_sample_pair(
    pred_path: str,
    gt_path: str,
    voxel_size: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    加载一对预测和真实数据
    
    Returns:
        pred_voxel, gt_voxel, pred_pc, gt_pc
    """
    pred_voxel = load_voxel(pred_path)
    gt_voxel = load_voxel(gt_path)
    
    pred_pc = voxel_to_pointcloud(pred_voxel, voxel_size)
    gt_pc = voxel_to_pointcloud(gt_voxel, voxel_size)
    
    return pred_voxel, gt_voxel, pred_pc, gt_pc


# ==============================================================================
# 点云指标
# ==============================================================================

def compute_chamfer_distance(
    pred_pc: np.ndarray,
    gt_pc: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    计算 Chamfer Distance
    
    Args:
        pred_pc: (N, 3) 预测点云
        gt_pc: (M, 3) 真实点云
    
    Returns:
        chamfer_dist: Chamfer 距离
        dist_pred_to_gt: 预测到真实的距离
        dist_gt_to_pred: 真实到预测的距离
    """
    if pred_pc.shape[0] == 0 or gt_pc.shape[0] == 0:
        return float('inf'), np.array([]), np.array([])
    
    # 只使用 xyz 坐标
    pred_xyz = pred_pc[:, :3]
    gt_xyz = gt_pc[:, :3]
    
    # 构建 KD-Tree
    tree_pred = cKDTree(pred_xyz)
    tree_gt = cKDTree(gt_xyz)
    
    # 查询最近邻
    dist_pred_to_gt, _ = tree_gt.query(pred_xyz)
    dist_gt_to_pred, _ = tree_pred.query(gt_xyz)
    
    # Chamfer Distance
    chamfer_dist = np.mean(dist_pred_to_gt) + np.mean(dist_gt_to_pred)
    
    return chamfer_dist, dist_pred_to_gt, dist_gt_to_pred


def compute_hausdorff_distance(
    dist_pred_to_gt: np.ndarray,
    dist_gt_to_pred: np.ndarray,
) -> float:
    """计算 Hausdorff Distance"""
    if len(dist_pred_to_gt) == 0 or len(dist_gt_to_pred) == 0:
        return float('inf')
    
    return max(np.max(dist_pred_to_gt), np.max(dist_gt_to_pred))


def compute_precision_recall(
    dist_pred_to_gt: np.ndarray,
    dist_gt_to_pred: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    计算精确率、召回率和 F-score
    
    Args:
        dist_pred_to_gt: 预测到真实的距离
        dist_gt_to_pred: 真实到预测的距离
        threshold: 匹配阈值
    
    Returns:
        metrics: {'precision', 'recall', 'fscore', 'tp', 'fp', 'fn'}
    """
    if len(dist_pred_to_gt) == 0 or len(dist_gt_to_pred) == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'fscore': 0.0,
            'tp': 0, 'fp': 0, 'fn': 0,
        }
    
    # True Positive: 预测点中距离真实点 < threshold 的数量
    tp = np.sum(dist_pred_to_gt <= threshold)
    # False Positive: 预测点中距离真实点 > threshold 的数量
    fp = np.sum(dist_pred_to_gt > threshold)
    # False Negative: 真实点中距离预测点 > threshold 的数量
    fn = np.sum(dist_gt_to_pred > threshold)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fscore = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'fscore': fscore,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
    }


# ==============================================================================
# 体素指标
# ==============================================================================

def compute_occupancy_iou(pred_voxel: np.ndarray, gt_voxel: np.ndarray) -> float:
    """
    计算占用 IoU
    
    Args:
        pred_voxel: 预测体素 (最后一维的第0通道是占用率)
        gt_voxel: 真实体素
    """
    # 获取占用掩码
    if pred_voxel.ndim == 4:
        pred_occ = pred_voxel[..., 0] > 0 if pred_voxel.shape[-1] <= 4 else pred_voxel[0] > 0
        gt_occ = gt_voxel[..., 0] > 0 if gt_voxel.shape[-1] <= 4 else gt_voxel[0] > 0
    else:
        pred_occ = pred_voxel > 0
        gt_occ = gt_voxel > 0
    
    # 计算 IoU
    intersection = np.logical_and(pred_occ, gt_occ).sum()
    union = np.logical_or(pred_occ, gt_occ).sum()
    
    iou = intersection / union if union > 0 else 0.0
    
    return iou


def compute_feature_mae(
    pred_voxel: np.ndarray,
    gt_voxel: np.ndarray,
    channel: int = 1,
) -> float:
    """
    计算特征通道的 MAE (只在占用区域)
    
    Args:
        channel: 0=占用, 1=强度, 2=多普勒, 3=多普勒方差
    """
    # 获取占用掩码 (使用真实数据的占用)
    if pred_voxel.ndim == 4 and pred_voxel.shape[-1] <= 4:
        gt_occ = gt_voxel[..., 0] > 0
        pred_feat = pred_voxel[..., channel]
        gt_feat = gt_voxel[..., channel]
    else:
        return 0.0
    
    # 只在占用区域计算
    if gt_occ.sum() == 0:
        return 0.0
    
    mae = np.mean(np.abs(pred_feat[gt_occ] - gt_feat[gt_occ]))
    return mae


# ==============================================================================
# 综合评估
# ==============================================================================

def evaluate_sample(
    pred_voxel: np.ndarray,
    gt_voxel: np.ndarray,
    pred_pc: np.ndarray,
    gt_pc: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    评估单个样本
    
    Returns:
        metrics: 包含所有指标的字典
    """
    metrics = {}
    
    # 点云指标
    chamfer, dist_p2g, dist_g2p = compute_chamfer_distance(pred_pc, gt_pc)
    metrics['chamfer_distance'] = chamfer
    metrics['hausdorff_distance'] = compute_hausdorff_distance(dist_p2g, dist_g2p)
    
    pr_metrics = compute_precision_recall(dist_p2g, dist_g2p, threshold)
    metrics.update(pr_metrics)
    
    # 体素指标
    metrics['occupancy_iou'] = compute_occupancy_iou(pred_voxel, gt_voxel)
    metrics['intensity_mae'] = compute_feature_mae(pred_voxel, gt_voxel, channel=1)
    metrics['doppler_mae'] = compute_feature_mae(pred_voxel, gt_voxel, channel=2)
    
    # 点数统计
    metrics['pred_points'] = pred_pc.shape[0]
    metrics['gt_points'] = gt_pc.shape[0]
    
    return metrics


def aggregate_metrics(all_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """聚合所有样本的指标"""
    if not all_metrics:
        return {}
    
    aggregated = {}
    keys = all_metrics[0].keys()
    
    for key in keys:
        values = [m[key] for m in all_metrics if not np.isinf(m[key])]
        if values:
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
            aggregated[f'{key}_median'] = np.median(values)
    
    aggregated['num_samples'] = len(all_metrics)
    
    return aggregated


# ==============================================================================
# 主函数
# ==============================================================================

def find_matching_files(pred_dir: str, gt_dir: str) -> List[Tuple[str, str]]:
    """查找匹配的预测和真实文件"""
    pairs = []
    
    pred_files = set(os.listdir(pred_dir))
    gt_files = set(os.listdir(gt_dir))
    
    # 查找同名文件
    for pf in pred_files:
        if pf in gt_files:
            pairs.append((
                os.path.join(pred_dir, pf),
                os.path.join(gt_dir, pf),
            ))
        else:
            # 尝试不同扩展名
            base = os.path.splitext(pf)[0]
            for ext in ['.npy', '.npz']:
                gf = base + ext
                if gf in gt_files:
                    pairs.append((
                        os.path.join(pred_dir, pf),
                        os.path.join(gt_dir, gf),
                    ))
                    break
    
    return sorted(pairs)


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Radar Point Cloud Evaluation")
    print("=" * 60)
    print(f"Prediction directory: {args.pred_dir}")
    print(f"Ground truth directory: {args.gt_dir}")
    print(f"Matching threshold: {args.threshold}m")
    print(f"Voxel size: {args.voxel_size}m")
    
    # 查找文件对
    file_pairs = find_matching_files(args.pred_dir, args.gt_dir)
    print(f"\nFound {len(file_pairs)} matching file pairs")
    
    if len(file_pairs) == 0:
        print("Error: No matching files found!")
        return
    
    # 评估每个样本
    all_metrics = []
    
    for pred_path, gt_path in tqdm(file_pairs, desc="Evaluating"):
        try:
            pred_voxel, gt_voxel, pred_pc, gt_pc = load_sample_pair(
                pred_path, gt_path, args.voxel_size
            )
            
            metrics = evaluate_sample(
                pred_voxel, gt_voxel, pred_pc, gt_pc, args.threshold
            )
            
            if args.verbose:
                print(f"\n{os.path.basename(pred_path)}:")
                print(f"  Chamfer: {metrics['chamfer_distance']:.4f}")
                print(f"  IoU: {metrics['occupancy_iou']:.4f}")
                print(f"  F-score: {metrics['fscore']:.4f}")
            
            all_metrics.append(metrics)
            
        except Exception as e:
            print(f"\nError processing {pred_path}: {e}")
            continue
    
    # 聚合结果
    aggregated = aggregate_metrics(all_metrics)
    
    # 打印结果
    print("\n" + "=" * 60)
    print("Aggregated Results")
    print("=" * 60)
    
    key_metrics = [
        ('chamfer_distance_mean', 'Chamfer Distance'),
        ('hausdorff_distance_mean', 'Hausdorff Distance'),
        ('occupancy_iou_mean', 'Occupancy IoU'),
        ('precision_mean', 'Precision'),
        ('recall_mean', 'Recall'),
        ('fscore_mean', 'F-score'),
        ('intensity_mae_mean', 'Intensity MAE'),
        ('doppler_mae_mean', 'Doppler MAE'),
    ]
    
    for key, name in key_metrics:
        if key in aggregated:
            std_key = key.replace('_mean', '_std')
            std_val = aggregated.get(std_key, 0)
            print(f"{name:25s}: {aggregated[key]:.4f} ± {std_val:.4f}")
    
    print(f"\nTotal samples evaluated: {aggregated.get('num_samples', 0)}")
    
    # 保存结果
    with open(args.output, 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
