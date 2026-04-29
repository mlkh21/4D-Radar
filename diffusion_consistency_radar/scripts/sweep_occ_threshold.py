#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sweep occupancy thresholds on saved prediction voxels and compare with target_voxel
using the same resolution (C, Z, H, W) = (4, 32, 128, 128).
"""

import argparse
import csv
import os
from typing import List

import numpy as np
import torch
import torch.nn.functional as F


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

    # target: (H, W, Z, C) -> (C, Z, H, W), then resize to training/inference size.
    tensor = torch.from_numpy(target).permute(3, 2, 0, 1).unsqueeze(0).to(device)
    resized = F.interpolate(
        tensor,
        size=(32, 128, 128),
        mode="trilinear",
        align_corners=False,
    ).squeeze(0).cpu().numpy()
    return resized[0]


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
    summary = {th: {"pred": 0, "target": 0, "n": 0} for th in thresholds}

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index",
            "frame_id",
            "threshold",
            "pred_occ_count",
            "target_occ_count",
            "pred_to_target_ratio",
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

            for th in thresholds:
                pred_count = int(np.count_nonzero(pred_occ > th))
                target_count = int(np.count_nonzero(target_occ > th))
                ratio = (pred_count / target_count) if target_count > 0 else np.nan

                writer.writerow([
                    i,
                    frame_id,
                    f"{th:.6f}",
                    pred_count,
                    target_count,
                    f"{ratio:.6f}" if np.isfinite(ratio) else "",
                ])

                summary[th]["pred"] += pred_count
                summary[th]["target"] += target_count
                summary[th]["n"] += 1

        for th in thresholds:
            n = max(summary[th]["n"], 1)
            pred_mean = summary[th]["pred"] / n
            target_mean = summary[th]["target"] / n
            ratio = (pred_mean / target_mean) if target_mean > 0 else np.nan
            writer.writerow([
                "__summary__",
                "",
                f"{th:.6f}",
                f"{pred_mean:.3f}",
                f"{target_mean:.3f}",
                f"{ratio:.6f}" if np.isfinite(ratio) else "",
            ])

    print(f"Saved threshold sweep metrics to: {output_csv}")


if __name__ == "__main__":
    main()
