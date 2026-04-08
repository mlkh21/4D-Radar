# -*- coding: utf-8 -*-
# NOTE: 该脚本用于评估雷达点云预测结果，支持以下指标：
# NOTE: 1) Chamfer 距离
# NOTE: 2) Hausdorff 距离
# NOTE: 3) Precision / Recall / F-score
# NOTE: 默认假设 pred_path 与 gt_path 均为 .npy 点云目录，按同名文件进行配对。

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate radar point cloud predictions")
    parser.add_argument("--pred_path", type=str, required=True, help="预测点云目录")
    parser.add_argument("--gt_path", type=str, required=True, help="真实点云目录")
    parser.add_argument("--output_path", type=str, default="./eval_results.json", help="输出json路径")
    parser.add_argument("--distance_threshold", type=float, default=0.5, help="F-score匹配阈值(米)")
    # TODO: 增加Hausdorff分位数配置（例如95% Hausdorff），降低离群点对指标的极端影响。
    # TODO: 增加DEM相关指标（DEM RMSE、冲突度统计）并与点云指标统一汇总。
    return parser.parse_args()


def _list_npy_files(folder: str) -> Dict[str, str]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Directory not found: {folder}")

    mapping = {}
    for name in os.listdir(folder):
        if not name.endswith(".npy"):
            continue
        stem = os.path.splitext(name)[0]
        mapping[stem] = os.path.join(folder, name)
    return mapping


def _to_points_xyz(arr: np.ndarray) -> np.ndarray:
    """将输入数组转换为 (N, 3) 点云坐标。"""
    pts = np.asarray(arr, dtype=np.float32)

    if pts.ndim == 1:
        if pts.size % 3 != 0:
            raise ValueError(f"1D array cannot be reshaped into xyz points: shape={pts.shape}")
        pts = pts.reshape(-1, 3)
    elif pts.ndim == 2:
        if pts.shape[1] < 3:
            raise ValueError(f"2D array must have at least 3 columns: shape={pts.shape}")
        pts = pts[:, :3]
    else:
        # TODO: 如后续输入为体素或多维张量，增加专门的体素->点云转换入口。
        if pts.shape[-1] < 3:
            raise ValueError(f"Array last dimension must be >=3 for xyz: shape={pts.shape}")
        pts = pts.reshape(-1, pts.shape[-1])[:, :3]

    return pts


def compute_chamfer_distance(gt: np.ndarray, pred: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """计算 Chamfer 距离，并返回双向最近邻距离数组。"""
    kdtree_gt = cKDTree(gt)
    kdtree_pred = cKDTree(pred)

    distance_pred_to_gt, _ = kdtree_gt.query(pred)
    distance_gt_to_pred, _ = kdtree_pred.query(gt)

    chamfer_dist = float(np.mean(distance_gt_to_pred) + np.mean(distance_pred_to_gt))
    return chamfer_dist, distance_gt_to_pred, distance_pred_to_gt


def evaluate_matches(distance_a_to_b: np.ndarray, distance_b_to_a: np.ndarray, threshold: float):
    """基于距离阈值统计 TP/FN/FP/TN 及 precision/recall。"""
    tp = int(np.sum(distance_b_to_a <= threshold))
    fp = int(np.sum(distance_b_to_a > threshold))
    fn = int(np.sum(distance_a_to_b > threshold))
    tn = 0  # NOTE: 点云匹配任务通常不存在严格定义的 TN。

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    return tp, fn, fp, tn, precision, recall


def read_inference_data(pred_root: str, gt_root: str) -> List[Tuple[str, str, str]]:
    """按文件名（去扩展名）配对预测与真值，返回 (sample_id, pred_file, gt_file) 列表。"""
    pred_map = _list_npy_files(pred_root)
    gt_map = _list_npy_files(gt_root)

    shared_keys = sorted(set(pred_map.keys()) & set(gt_map.keys()))
    if not shared_keys:
        raise RuntimeError(
            f"No matched .npy files between pred_path={pred_root} and gt_path={gt_root}."
        )

    # TODO: 后续可扩展跨场景索引文件驱动配对，避免仅依赖文件名一致。
    pairs = [(k, pred_map[k], gt_map[k]) for k in shared_keys]
    return pairs


def main():
    args = parse_args()

    print(f"评估开始，预测目录: {args.pred_path}，真值目录: {args.gt_path}")
    pairs = read_inference_data(args.pred_path, args.gt_path)

    metrics = {
        "chamfer_distance": [],
        "hausdorff_distance": [],
        "precision": [],
        "recall": [],
        "fscore": [],
    }

    sample_results = []

    for sample_id, pred_file, gt_file in tqdm(pairs, desc="Evaluating", ncols=100):
        pred_pc = _to_points_xyz(np.load(pred_file))
        gt_pc = _to_points_xyz(np.load(gt_file))

        if pred_pc.shape[0] == 0 or gt_pc.shape[0] == 0:
            continue

        chamfer_i, distance_gt_to_pred, distance_pred_to_gt = compute_chamfer_distance(gt_pc, pred_pc)
        hausdorff_i = float(max(np.max(distance_gt_to_pred), np.max(distance_pred_to_gt)))

        _, _, _, _, precision_i, recall_i = evaluate_matches(
            distance_gt_to_pred,
            distance_pred_to_gt,
            threshold=args.distance_threshold,
        )
        if (precision_i + recall_i) > 0.0:
            fscore_i = float(2.0 * precision_i * recall_i / (precision_i + recall_i))
        else:
            fscore_i = 0.0

        metrics["chamfer_distance"].append(chamfer_i)
        metrics["hausdorff_distance"].append(hausdorff_i)
        metrics["precision"].append(precision_i)
        metrics["recall"].append(recall_i)
        metrics["fscore"].append(fscore_i)

        sample_results.append(
            {
                "sample_id": sample_id,
                "pred_points": int(pred_pc.shape[0]),
                "gt_points": int(gt_pc.shape[0]),
                "chamfer_distance": chamfer_i,
                "hausdorff_distance": hausdorff_i,
                "precision": precision_i,
                "recall": recall_i,
                "fscore": fscore_i,
            }
        )

    if not metrics["chamfer_distance"]:
        raise RuntimeError("No valid paired samples were evaluated (empty point clouds or no pairs).")

    summary = {
        "num_evaluated": len(metrics["chamfer_distance"]),
        "distance_threshold": args.distance_threshold,
        "mean_chamfer_distance": float(np.mean(metrics["chamfer_distance"])),
        "mean_hausdorff_distance": float(np.mean(metrics["hausdorff_distance"])),
        "mean_precision": float(np.mean(metrics["precision"])),
        "mean_recall": float(np.mean(metrics["recall"])),
        "mean_fscore": float(np.mean(metrics["fscore"])),
    }

    output = {
        "summary": summary,
        "samples": sample_results,
    }

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Chamfer 平均距离: {summary['mean_chamfer_distance']:.6f}")
    print(f"Hausdorff 平均距离: {summary['mean_hausdorff_distance']:.6f}")
    print(f"F-score 平均值: {summary['mean_fscore']:.6f}")
    print(f"评估结果已保存到: {args.output_path}")


if __name__ == "__main__":
    main()
