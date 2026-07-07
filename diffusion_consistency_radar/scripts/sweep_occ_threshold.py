#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
扫描已保存预测体素的占用阈值，并在可配置物理范围与目标网格上比较 target_voxel。
"""

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cm.dataset_loader import crop_voxel_channels_to_pc_range, resize_voxel_channels
from cm.evaluation_metrics import (
    bev_iou,
    filter_points_by_band,
    nearest_neighbor_metrics,
    occupancy_prf,
    parse_range_bins,
    voxel_to_points,
)


def load_sparse_voxel(path: str) -> np.ndarray:
    data = np.load(path)
    voxel = np.zeros(data["shape"], dtype=np.float32)
    coords = data["coords"]
    if coords.shape[0] > 0:
        voxel[coords[:, 0], coords[:, 1], coords[:, 2]] = data["features"]
    return voxel


def load_target_occ_resized(
    path: str,
    device: torch.device,
    source_pc_range: Sequence[float],
    model_pc_range: Sequence[float],
    target_size: Sequence[int],
) -> np.ndarray:
    if len(target_size) != 3 or any(int(value) <= 0 for value in target_size):
        raise ValueError(f"target_size 必须包含三个正整数，当前为 {target_size}")
    if path.endswith(".npz"):
        target = load_sparse_voxel(path)
    else:
        target = np.load(path).astype(np.float32)

    # target: (X,Y,Z,C) -> (C,Z,X,Y)，先按物理范围裁剪，再按训练协议重采样。
    tensor = torch.from_numpy(target).permute(3, 2, 0, 1).to(device)
    cropped = crop_voxel_channels_to_pc_range(
        tensor,
        source_pc_range=source_pc_range,
        model_pc_range=model_pc_range,
    )
    resized = resize_voxel_channels(cropped, tuple(int(v) for v in target_size), mask_channel=3)
    return resized[0].cpu().numpy()


def resolve_target_path(
    target_voxel_dir: str,
    frame_id: str,
    evaluation_split: str,
) -> Optional[str]:
    """解析 target 文件；训练/验证子集缺失时立即失败。"""
    for extension in (".npz", ".npy"):
        candidate = os.path.join(target_voxel_dir, f"{frame_id}{extension}")
        if os.path.exists(candidate):
            return candidate
    if evaluation_split != "all":
        raise RuntimeError(
            f"{evaluation_split} 子集中的 prediction {frame_id} 缺少对应 target 文件"
        )
    return None


def parse_thresholds(raw: str) -> List[float]:
    vals = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if not np.isfinite(value):
            raise ValueError("threshold 必须是有限数")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"threshold 必须位于 [0, 1]，当前为 {value}")
        vals.append(value)
    if not vals:
        raise ValueError("threshold list is empty")
    return vals


def select_evaluation_files(
    files: Sequence[str],
    evaluation_split: str,
    train_split: float,
    split_seed: int,
) -> List[str]:
    """使用与 unified_train 完全一致的 torch.randperm 规则选择评估文件。"""
    ordered = list(files)
    if evaluation_split == "all":
        return ordered
    frame_ids = []
    suffix = "_voxel.npy"
    for filename in ordered:
        frame_id = filename[:-len(suffix)] if filename.endswith(suffix) else ""
        if not frame_id.isdigit():
            raise ValueError(
                "train/validation 划分要求预测 frame ID 为纯数字；"
                "非数字命名请使用 --evaluation_split all"
            )
        frame_ids.append(int(frame_id))
    if any(current != previous + 1 for previous, current in zip(frame_ids, frame_ids[1:])):
        raise ValueError(
            "train/validation 划分要求排序后的预测 frame ID 严格连续；"
            "检测到缺帧，请补齐预测或使用 --evaluation_split all"
        )
    if len(ordered) < 2:
        raise ValueError("训练/验证划分至少需要 2 个样本")
    if not 0.0 < float(train_split) < 1.0:
        raise ValueError("train_split 必须严格位于 (0, 1)")
    train_size = int(len(ordered) * float(train_split))
    if train_size <= 0 or train_size >= len(ordered):
        raise ValueError(
            f"train_split={train_split} 导致空划分："
            f"dataset_size={len(ordered)}, train_size={train_size}"
        )
    generator = torch.Generator().manual_seed(int(split_seed))
    indices = torch.randperm(len(ordered), generator=generator).tolist()
    selected = indices[:train_size] if evaluation_split == "train" else indices[train_size:]
    return [ordered[index] for index in selected]


def prepare_evaluation_files(
    files: Sequence[str],
    evaluation_split: str,
    train_split: float,
    split_seed: int,
    max_files: int,
) -> List[str]:
    """先在完整预测清单上复现数据划分，再限制实际评估帧数。"""
    selected = select_evaluation_files(
        files,
        evaluation_split=evaluation_split,
        train_split=train_split,
        split_seed=split_seed,
    )
    if int(max_files) > 0:
        return selected[: int(max_files)]
    return selected


def validate_range_bins(
    range_bins: Sequence[Tuple[str, float, float]],
    model_pc_range: Sequence[float],
) -> None:
    """验证距离分段在模型 X 范围内有序且互不重叠。"""
    model_x_min = float(model_pc_range[0])
    model_x_max = float(model_pc_range[3])
    previous_max = model_x_min
    for _, band_min, band_max in range_bins:
        lo, hi = float(band_min), float(band_max)
        if not np.isfinite(lo) or not np.isfinite(hi):
            raise ValueError("range bins 边界必须是有限数")
        if lo < model_x_min or hi > model_x_max:
            raise ValueError(
                f"range bins 必须位于模型 X 范围 [{model_x_min}, {model_x_max}] 内"
            )
        if lo >= hi:
            raise ValueError("range bins 每段下界必须小于上界")
        if lo < previous_max:
            raise ValueError("range bins 必须按 X 有序且不重叠")
        previous_max = hi


def _metrics_from_counts(tp: float, fp: float, fn: float) -> Dict[str, float]:
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    iou = tp / max(tp + fp + fn, 1.0)
    return {
        "task_bev_precision": float(precision),
        "task_bev_recall": float(recall),
        "task_bev_f1": float(f1),
        "task_bev_iou": float(iou),
    }


def evaluate_task_thresholds(
    pred_occ: np.ndarray,
    target_occ: np.ndarray,
    thresholds: Sequence[float],
    target_threshold: float,
    pc_range: Sequence[float],
    z_min: float,
    range_bins: Sequence[Tuple[str, float, float]],
    bev_cell_size: float,
) -> Dict[float, Dict[str, object]]:
    """计算单帧严格体素指标及按距离分段的避障任务指标。"""
    voxel_metrics = evaluate_thresholds(
        pred_occ,
        target_occ,
        thresholds=thresholds,
        target_threshold=target_threshold,
        pc_range=pc_range,
        x_max=max((band[2] for band in range_bins), default=float(pc_range[3])),
        z_min=z_min,
    )["metrics"]
    target_points = voxel_to_points(
        np.asarray(target_occ, dtype=np.float32)[None, ...],
        pc_range=pc_range,
        occ_threshold=target_threshold,
    )
    target_bands = {
        label: filter_points_by_band(
            target_points,
            pc_range=pc_range,
            x_min=band_min,
            x_max=band_max,
            z_min=z_min,
        )
        for label, band_min, band_max in range_bins
    }
    results: Dict[float, Dict[str, object]] = {}
    for threshold in thresholds:
        pred_points = voxel_to_points(
            np.asarray(pred_occ, dtype=np.float32)[None, ...],
            pc_range=pc_range,
            occ_threshold=threshold,
        )
        band_results: Dict[str, Dict[str, float]] = {}
        total_tp = total_fp = total_fn = 0.0
        pred_total = target_total = 0
        total_matched = 0.0
        total_match_queries = 0
        for label, band_min, band_max in range_bins:
            pred_band = filter_points_by_band(
                pred_points, pc_range=pc_range, x_min=band_min, x_max=band_max, z_min=z_min
            )
            target_band = target_bands[label]
            prf = occupancy_prf(
                pred_band,
                target_band,
                pc_range=pc_range,
                cell_size=bev_cell_size,
            )
            bev = bev_iou(
                pred_band,
                target_band,
                pc_range=pc_range,
                cell_size=bev_cell_size,
            )
            nn = nearest_neighbor_metrics(pred_band, target_band, thresholds=(2.0,))
            tp, fp, fn = prf["tp"], prf["fp"], prf["fn"]
            total_tp += tp
            total_fp += fp
            total_fn += fn
            pred_total += int(pred_band.shape[0])
            target_total += int(target_band.shape[0])
            match_ratio = float(nn["match_ratio_2"])
            match_query_count = int(pred_band.shape[0])
            if match_query_count == 0:
                matched_pred_count = 0.0
                reported_match_ratio = None
            elif target_band.shape[0] == 0:
                matched_pred_count = 0.0
                reported_match_ratio = 0.0
            else:
                reported_match_ratio = match_ratio if np.isfinite(match_ratio) else 0.0
                matched_pred_count = float(reported_match_ratio * match_query_count)
            total_matched += matched_pred_count
            total_match_queries += match_query_count
            band_metrics = _metrics_from_counts(tp, fp, fn)
            if np.isfinite(bev["bev_iou"]):
                band_metrics["task_bev_iou"] = float(bev["bev_iou"])
            band_metrics.update(
                {
                    "task_match_ratio_2": reported_match_ratio,
                    "matched_pred_count": matched_pred_count,
                    "match_query_count": match_query_count,
                    "pred_count": int(pred_band.shape[0]),
                    "target_count": int(target_band.shape[0]),
                    "pred_to_target_ratio": (
                        float(pred_band.shape[0] / target_band.shape[0])
                        if target_band.shape[0] > 0
                        else None
                    ),
                    "bev_tp": int(tp),
                    "bev_fp": int(fp),
                    "bev_fn": int(fn),
                }
            )
            band_results[label] = band_metrics

        task_metrics = _metrics_from_counts(total_tp, total_fp, total_fn)
        strict = voxel_metrics[float(threshold)]
        results[float(threshold)] = {
            "voxel_precision": strict["precision"],
            "voxel_recall": strict["recall"],
            "voxel_f1": strict["f1"],
            "voxel_iou": strict["iou"],
            "voxel_tp": strict["tp"],
            "voxel_fp": strict["fp"],
            "voxel_fn": strict["fn"],
            **task_metrics,
            "task_match_ratio_2": (
                float(total_matched / total_match_queries)
                if total_match_queries > 0
                else None
            ),
            "matched_pred_count": total_matched,
            "match_query_count": total_match_queries,
            "pred_count": pred_total,
            "target_count": target_total,
            "pred_to_target_ratio": (
                float(pred_total / target_total) if target_total > 0 else None
            ),
            "bev_tp": int(total_tp),
            "bev_fp": int(total_fp),
            "bev_fn": int(total_fn),
            "bands": band_results,
        }
    return results


def select_recommended_threshold(
    metrics: Dict[float, Dict[str, object]],
    selection_metric: str = "task_bev_f1",
) -> float:
    """按任务指标或严格体素 F1 选择阈值，并使用稳定规则消除平局。"""
    if selection_metric == "voxel_f1":
        return float(
            max(
                metrics,
                key=lambda value: (
                    float(metrics[value].get("voxel_f1", metrics[value].get("f1", 0.0))),
                    float(metrics[value].get("voxel_precision", metrics[value].get("precision", 0.0))),
                    -abs(float(value) - 0.5),
                    -float(value),
                ),
            )
        )
    if selection_metric != "task_bev_f1":
        raise ValueError(f"不支持的 selection_metric: {selection_metric}")

    def task_key(value: float) -> Tuple[float, float, float, float, float]:
        item = metrics[value]
        ratio = item.get("pred_to_target_ratio")
        ratio_distance = abs(float(ratio) - 1.0) if ratio is not None else float("inf")
        return (
            float(item.get("task_bev_f1", 0.0)),
            float(item.get("task_bev_iou", 0.0)),
            -ratio_distance,
            -abs(float(value) - 0.5),
            -float(value),
        )

    return float(max(metrics, key=task_key))


def evaluate_thresholds(
    pred_occ: np.ndarray,
    target_occ: np.ndarray,
    thresholds: Sequence[float],
    target_threshold: float = 0.1,
    pc_range: Sequence[float] = (0, -20, -6, 120, 20, 10),
    x_max: float = 80.0,
    z_min: float = -1.0,
) -> Dict[str, object]:
    """Evaluate voxel occupancy thresholds inside the task-relevant region."""
    pred = np.asarray(pred_occ, dtype=np.float32)
    target = np.asarray(target_occ, dtype=np.float32)
    if pred.shape != target.shape or pred.ndim != 3:
        raise ValueError(f"Expected matching (Z,X,Y) arrays, got {pred.shape} and {target.shape}")

    nz, nx, _ = pred.shape
    z_centers = float(pc_range[2]) + (np.arange(nz, dtype=np.float32) + 0.5) * (
        (float(pc_range[5]) - float(pc_range[2])) / max(nz, 1)
    )
    x_centers = float(pc_range[0]) + (np.arange(nx, dtype=np.float32) + 0.5) * (
        (float(pc_range[3]) - float(pc_range[0])) / max(nx, 1)
    )
    region = (z_centers[:, None, None] >= float(z_min)) & (x_centers[None, :, None] < float(x_max))
    target_mask = (target > float(target_threshold)) & region

    metrics = {}
    for threshold in thresholds:
        pred_mask = (pred > float(threshold)) & region
        tp = int(np.count_nonzero(pred_mask & target_mask))
        fp = int(np.count_nonzero(pred_mask & ~target_mask))
        fn = int(np.count_nonzero(~pred_mask & target_mask))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        iou = tp / max(tp + fp + fn, 1)
        metrics[float(threshold)] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "iou": float(iou),
            "pred_count": int(np.count_nonzero(pred_mask)),
            "target_count": int(np.count_nonzero(target_mask)),
        }

    best_threshold = max(
        metrics,
        key=lambda value: (metrics[value]["f1"], metrics[value]["precision"], -abs(value - 0.5)),
    )
    return {"best_threshold": float(best_threshold), "metrics": metrics}


def main():
    parser = argparse.ArgumentParser(
        description="Sweep occ_threshold for saved voxels and compare with target_voxel."
    )
    parser.add_argument("--pred_voxel_dir", type=str, required=True, help="Directory with *_voxel.npy")
    parser.add_argument("--target_voxel_dir", type=str, required=True, help="Directory with target_voxel files")
    parser.add_argument("--thresholds", type=str, default="0.03,0.05,0.08,0.1",
                        help="Comma-separated thresholds")
    parser.add_argument("--output_csv", type=str, default="",
                        help="Output csv path, default: <pred_voxel_dir>/occ_sweep_metrics.csv")
    parser.add_argument("--max_files", type=int, default=0, help="Max files to evaluate (0 means all)")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda for resizing target_voxel")
    parser.add_argument("--target_threshold", type=float, default=0.1)
    parser.add_argument(
        "--source_pc_range",
        type=float,
        nargs=6,
        default=[0, -20, -6, 120, 20, 10],
        help="原始 target 体素的 XYZ 物理范围",
    )
    parser.add_argument(
        "--model_pc_range",
        type=float,
        nargs=6,
        default=[0, -20, -6, 40, 20, 10],
        help="预测模型输出对应的 XYZ 物理范围",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        nargs=3,
        default=[32, 128, 128],
        help="比较网格大小，顺序为 Z X Y",
    )
    parser.add_argument(
        "--evaluation_split",
        choices=("train", "validation", "all"),
        default="validation",
        help="按训练时相同随机规则选择评估子集",
    )
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--range_bins", type=str, default="0-20,20-40")
    parser.add_argument("--bev_cell_size", type=float, default=0.5)
    parser.add_argument(
        "--selection_metric",
        choices=("task_bev_f1", "voxel_f1"),
        default="task_bev_f1",
    )
    parser.add_argument(
        "--x_max",
        type=float,
        default=80.0,
        help="已弃用，仅保留在输出元数据中以兼容旧命令",
    )
    parser.add_argument("--z_min", type=float, default=-1.0)
    parser.add_argument("--output_json", type=str, default="",
                        help="Recommended threshold JSON, default: <pred_voxel_dir>/occ_threshold_recommendation.json")
    args = parser.parse_args()

    thresholds = parse_thresholds(args.thresholds)
    pred_files = sorted([f for f in os.listdir(args.pred_voxel_dir) if f.endswith("_voxel.npy")])
    if not pred_files:
        raise RuntimeError(f"No *_voxel.npy found in {args.pred_voxel_dir}")
    pred_files = prepare_evaluation_files(
        pred_files,
        evaluation_split=args.evaluation_split,
        train_split=args.train_split,
        split_seed=args.split_seed,
        max_files=args.max_files,
    )
    range_bins = parse_range_bins(args.range_bins)
    if not range_bins:
        raise ValueError("range_bins 不能为空")
    validate_range_bins(range_bins, args.model_pc_range)
    if args.bev_cell_size <= 0.0:
        raise ValueError("bev_cell_size 必须大于 0")

    output_csv = args.output_csv or os.path.join(args.pred_voxel_dir, "occ_sweep_metrics.csv")
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    device = torch.device(args.device)
    summary = {}
    for threshold in thresholds:
        summary[threshold] = {
            "pred": 0,
            "target": 0,
            "voxel_tp": 0,
            "voxel_fp": 0,
            "voxel_fn": 0,
            "bev_tp": 0,
            "bev_fp": 0,
            "bev_fn": 0,
            "matched_pred_count": 0.0,
            "match_query_count": 0,
            "n": 0,
            "bands": {
                label: {
                    "pred": 0,
                    "target": 0,
                    "bev_tp": 0,
                    "bev_fp": 0,
                    "bev_fn": 0,
                    "matched_pred_count": 0.0,
                    "match_query_count": 0,
                }
                for label, _, _ in range_bins
            },
        }

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index",
            "frame_id",
            "threshold",
            "pred_occ_count",
            "target_occ_count",
            "pred_to_target_ratio",
            "voxel_precision",
            "voxel_recall",
            "voxel_f1",
            "voxel_iou",
            "task_bev_precision",
            "task_bev_recall",
            "task_bev_f1",
            "task_bev_iou",
            "task_match_ratio_2",
        ])

        evaluated_frame_count = 0
        for i, pred_name in enumerate(pred_files):
            frame_id = pred_name[:-10]  # remove "_voxel.npy"
            pred_path = os.path.join(args.pred_voxel_dir, pred_name)
            pred_voxel = np.load(pred_path).astype(np.float32)
            pred_occ = pred_voxel[0]

            target_path = resolve_target_path(
                args.target_voxel_dir,
                frame_id,
                args.evaluation_split,
            )
            if target_path is None:
                # all 模式保留跳过无匹配 target 的兼容行为。
                continue
            evaluated_frame_count += 1
            target_occ = load_target_occ_resized(
                target_path,
                device,
                source_pc_range=args.source_pc_range,
                model_pc_range=args.model_pc_range,
                target_size=args.target_size,
            )
            evaluated = evaluate_task_thresholds(
                pred_occ,
                target_occ,
                thresholds=thresholds,
                target_threshold=args.target_threshold,
                pc_range=args.model_pc_range,
                z_min=args.z_min,
                range_bins=range_bins,
                bev_cell_size=args.bev_cell_size,
            )

            for th in thresholds:
                frame_metrics = evaluated[float(th)]
                pred_count = frame_metrics["pred_count"]
                target_count = frame_metrics["target_count"]
                ratio = (pred_count / target_count) if target_count > 0 else np.nan

                writer.writerow([
                    i,
                    frame_id,
                    f"{th:.6f}",
                    pred_count,
                    target_count,
                    f"{ratio:.6f}" if np.isfinite(ratio) else "",
                    f"{frame_metrics['voxel_precision']:.6f}",
                    f"{frame_metrics['voxel_recall']:.6f}",
                    f"{frame_metrics['voxel_f1']:.6f}",
                    f"{frame_metrics['voxel_iou']:.6f}",
                    f"{frame_metrics['task_bev_precision']:.6f}",
                    f"{frame_metrics['task_bev_recall']:.6f}",
                    f"{frame_metrics['task_bev_f1']:.6f}",
                    f"{frame_metrics['task_bev_iou']:.6f}",
                    (
                        f"{frame_metrics['task_match_ratio_2']:.6f}"
                        if frame_metrics["task_match_ratio_2"] is not None
                        else ""
                    ),
                ])

                summary[th]["pred"] += pred_count
                summary[th]["target"] += target_count
                summary[th]["voxel_tp"] += frame_metrics["voxel_tp"]
                summary[th]["voxel_fp"] += frame_metrics["voxel_fp"]
                summary[th]["voxel_fn"] += frame_metrics["voxel_fn"]
                summary[th]["bev_tp"] += frame_metrics["bev_tp"]
                summary[th]["bev_fp"] += frame_metrics["bev_fp"]
                summary[th]["bev_fn"] += frame_metrics["bev_fn"]
                summary[th]["matched_pred_count"] += frame_metrics["matched_pred_count"]
                summary[th]["match_query_count"] += frame_metrics["match_query_count"]
                summary[th]["n"] += 1
                for label, _, _ in range_bins:
                    band = frame_metrics["bands"][label]
                    band_summary = summary[th]["bands"][label]
                    band_summary["pred"] += band["pred_count"]
                    band_summary["target"] += band["target_count"]
                    band_summary["bev_tp"] += band["bev_tp"]
                    band_summary["bev_fp"] += band["bev_fp"]
                    band_summary["bev_fn"] += band["bev_fn"]
                    band_summary["matched_pred_count"] += band["matched_pred_count"]
                    band_summary["match_query_count"] += band["match_query_count"]

        recommendation_metrics = {}
        for th in thresholds:
            n = max(summary[th]["n"], 1)
            pred_mean = summary[th]["pred"] / n
            target_mean = summary[th]["target"] / n
            ratio = (pred_mean / target_mean) if target_mean > 0 else np.nan
            voxel_tp = summary[th]["voxel_tp"]
            voxel_fp = summary[th]["voxel_fp"]
            voxel_fn = summary[th]["voxel_fn"]
            voxel_precision = voxel_tp / max(voxel_tp + voxel_fp, 1)
            voxel_recall = voxel_tp / max(voxel_tp + voxel_fn, 1)
            voxel_f1 = (
                2.0 * voxel_precision * voxel_recall
                / max(voxel_precision + voxel_recall, 1e-12)
            )
            voxel_iou = voxel_tp / max(voxel_tp + voxel_fp + voxel_fn, 1)
            task_metrics = _metrics_from_counts(
                summary[th]["bev_tp"],
                summary[th]["bev_fp"],
                summary[th]["bev_fn"],
            )
            band_metrics = {}
            for label, _, _ in range_bins:
                band = summary[th]["bands"][label]
                item = _metrics_from_counts(band["bev_tp"], band["bev_fp"], band["bev_fn"])
                item.update(
                    {
                        "task_match_ratio_2": (
                            float(band["matched_pred_count"] / band["match_query_count"])
                            if band["match_query_count"] > 0
                            else None
                        ),
                        "matched_pred_count": float(band["matched_pred_count"]),
                        "match_query_count": int(band["match_query_count"]),
                        "pred_to_target_ratio": (
                            float(band["pred"] / band["target"])
                            if band["target"] > 0
                            else None
                        ),
                        "pred_count": int(band["pred"]),
                        "target_count": int(band["target"]),
                        "bev_tp": int(band["bev_tp"]),
                        "bev_fp": int(band["bev_fp"]),
                        "bev_fn": int(band["bev_fn"]),
                    }
                )
                band_metrics[label] = item
            recommendation_metrics[float(th)] = {
                "voxel_precision": float(voxel_precision),
                "voxel_recall": float(voxel_recall),
                "voxel_f1": float(voxel_f1),
                "voxel_iou": float(voxel_iou),
                **task_metrics,
                "task_match_ratio_2": (
                    float(
                        summary[th]["matched_pred_count"]
                        / summary[th]["match_query_count"]
                    )
                    if summary[th]["match_query_count"] > 0
                    else None
                ),
                "matched_pred_count": float(summary[th]["matched_pred_count"]),
                "match_query_count": int(summary[th]["match_query_count"]),
                "pred_to_target_ratio": float(ratio) if np.isfinite(ratio) else None,
                "pred_count_mean": float(pred_mean),
                "target_count_mean": float(target_mean),
                "bands": band_metrics,
            }
            writer.writerow([
                "__summary__",
                "",
                f"{th:.6f}",
                f"{pred_mean:.3f}",
                f"{target_mean:.3f}",
                f"{ratio:.6f}" if np.isfinite(ratio) else "",
                f"{voxel_precision:.6f}",
                f"{voxel_recall:.6f}",
                f"{voxel_f1:.6f}",
                f"{voxel_iou:.6f}",
                f"{task_metrics['task_bev_precision']:.6f}",
                f"{task_metrics['task_bev_recall']:.6f}",
                f"{task_metrics['task_bev_f1']:.6f}",
                f"{task_metrics['task_bev_iou']:.6f}",
                (
                    f"{recommendation_metrics[float(th)]['task_match_ratio_2']:.6f}"
                    if recommendation_metrics[float(th)]["task_match_ratio_2"] is not None
                    else ""
                ),
            ])

    best_threshold = select_recommended_threshold(
        recommendation_metrics,
        args.selection_metric,
    )
    output_json = args.output_json or os.path.join(args.pred_voxel_dir, "occ_threshold_recommendation.json")
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "recommended_threshold": float(best_threshold),
                "selection_metric": args.selection_metric,
                "target_threshold": float(args.target_threshold),
                "source_pc_range": [float(v) for v in args.source_pc_range],
                "model_pc_range": [float(v) for v in args.model_pc_range],
                "target_size": [int(v) for v in args.target_size],
                "deprecated_x_max": float(args.x_max),
                "z_min": float(args.z_min),
                "evaluation_split": args.evaluation_split,
                "train_split": float(args.train_split),
                "split_seed": int(args.split_seed),
                "range_bins": [
                    {"label": label, "x_min": float(lo), "x_max": float(hi)}
                    for label, lo, hi in range_bins
                ],
                "bev_cell_size": float(args.bev_cell_size),
                "selected_frame_count": int(len(pred_files)),
                "evaluated_frame_count": int(evaluated_frame_count),
                "metrics": {str(k): v for k, v in recommendation_metrics.items()},
            },
            handle,
            indent=2,
        )

    print(f"Saved threshold sweep metrics to: {output_csv}")
    print(f"Recommended threshold: {best_threshold:.6f}")
    print(f"Saved threshold recommendation to: {output_json}")


if __name__ == "__main__":
    main()
