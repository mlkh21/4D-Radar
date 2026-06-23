#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sweep occupancy thresholds on saved prediction voxels and compare with target_voxel
using the same resolution (C, Z, H, W) = (4, 32, 128, 128).
"""

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Sequence

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cm.dataset_loader import resize_voxel_channels


def load_sparse_voxel(path: str) -> np.ndarray:
    data = np.load(path)
    voxel = np.zeros(data["shape"], dtype=np.float32)
    coords = data["coords"]
    if coords.shape[0] > 0:
        voxel[coords[:, 0], coords[:, 1], coords[:, 2]] = data["features"]
    return voxel


def load_target_occ_resized(path: str, device: torch.device) -> np.ndarray:
    if path.endswith(".npz"):
        target = load_sparse_voxel(path)
    else:
        target = np.load(path).astype(np.float32)

    # target: (H, W, Z, C) -> (C, Z, H, W), then resize with the same sparse-aware
    # rule used by training and inference.
    tensor = torch.from_numpy(target).permute(3, 2, 0, 1).to(device)
    resized = resize_voxel_channels(tensor, (32, 128, 128), mask_channel=3)
    return resized[0].cpu().numpy()


def parse_thresholds(raw: str) -> List[float]:
    vals = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        vals.append(float(token))
    if not vals:
        raise ValueError("threshold list is empty")
    return vals


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
    parser.add_argument("--pc_range", type=float, nargs=6, default=[0, -20, -6, 120, 20, 10])
    parser.add_argument("--x_max", type=float, default=80.0)
    parser.add_argument("--z_min", type=float, default=-1.0)
    parser.add_argument("--output_json", type=str, default="",
                        help="Recommended threshold JSON, default: <pred_voxel_dir>/occ_threshold_recommendation.json")
    args = parser.parse_args()

    thresholds = parse_thresholds(args.thresholds)
    pred_files = sorted([f for f in os.listdir(args.pred_voxel_dir) if f.endswith("_voxel.npy")])
    if args.max_files > 0:
        pred_files = pred_files[:args.max_files]
    if not pred_files:
        raise RuntimeError(f"No *_voxel.npy found in {args.pred_voxel_dir}")

    output_csv = args.output_csv or os.path.join(args.pred_voxel_dir, "occ_sweep_metrics.csv")
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    device = torch.device(args.device)
    summary = {th: {"pred": 0, "target": 0, "tp": 0, "fp": 0, "fn": 0, "n": 0} for th in thresholds}

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index",
            "frame_id",
            "threshold",
            "pred_occ_count",
            "target_occ_count",
            "pred_to_target_ratio",
            "precision",
            "recall",
            "f1",
            "iou",
        ])

        for i, pred_name in enumerate(pred_files):
            frame_id = pred_name[:-10]  # remove "_voxel.npy"
            pred_path = os.path.join(args.pred_voxel_dir, pred_name)
            pred_voxel = np.load(pred_path).astype(np.float32)
            pred_occ = pred_voxel[0]

            target_path = os.path.join(args.target_voxel_dir, f"{frame_id}.npz")
            if not os.path.exists(target_path):
                target_path = os.path.join(args.target_voxel_dir, f"{frame_id}.npy")
            if not os.path.exists(target_path):
                # skip unmatched frame
                continue
            target_occ = load_target_occ_resized(target_path, device)
            evaluated = evaluate_thresholds(
                pred_occ,
                target_occ,
                thresholds=thresholds,
                target_threshold=args.target_threshold,
                pc_range=args.pc_range,
                x_max=args.x_max,
                z_min=args.z_min,
            )

            for th in thresholds:
                frame_metrics = evaluated["metrics"][float(th)]
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
                    f"{frame_metrics['precision']:.6f}",
                    f"{frame_metrics['recall']:.6f}",
                    f"{frame_metrics['f1']:.6f}",
                    f"{frame_metrics['iou']:.6f}",
                ])

                summary[th]["pred"] += pred_count
                summary[th]["target"] += target_count
                summary[th]["tp"] += frame_metrics["tp"]
                summary[th]["fp"] += frame_metrics["fp"]
                summary[th]["fn"] += frame_metrics["fn"]
                summary[th]["n"] += 1

        recommendation_metrics = {}
        for th in thresholds:
            n = max(summary[th]["n"], 1)
            pred_mean = summary[th]["pred"] / n
            target_mean = summary[th]["target"] / n
            ratio = (pred_mean / target_mean) if target_mean > 0 else np.nan
            tp, fp, fn = summary[th]["tp"], summary[th]["fp"], summary[th]["fn"]
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
            iou = tp / max(tp + fp + fn, 1)
            recommendation_metrics[float(th)] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "iou": float(iou),
                "pred_to_target_ratio": float(ratio) if np.isfinite(ratio) else None,
            }
            writer.writerow([
                "__summary__",
                "",
                f"{th:.6f}",
                f"{pred_mean:.3f}",
                f"{target_mean:.3f}",
                f"{ratio:.6f}" if np.isfinite(ratio) else "",
                f"{precision:.6f}",
                f"{recall:.6f}",
                f"{f1:.6f}",
                f"{iou:.6f}",
            ])

    best_threshold = max(
        recommendation_metrics,
        key=lambda value: (
            recommendation_metrics[value]["f1"],
            recommendation_metrics[value]["precision"],
            -abs(value - 0.5),
        ),
    )
    output_json = args.output_json or os.path.join(args.pred_voxel_dir, "occ_threshold_recommendation.json")
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "recommended_threshold": float(best_threshold),
                "selection_metric": "voxel_f1",
                "target_threshold": float(args.target_threshold),
                "pc_range": [float(v) for v in args.pc_range],
                "x_max": float(args.x_max),
                "z_min": float(args.z_min),
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
