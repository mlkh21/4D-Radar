#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Diagnose 4D radar densification quality with aligned projections and metrics."""

import argparse
import csv
import math
import os
import sys
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow direct `python diffusion_consistency_radar/scripts/*.py` execution.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cm.dataset_loader import resize_voxel_channels

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose radar densification outputs")
    parser.add_argument("--radar_voxel_dir", type=str, required=True)
    parser.add_argument("--target_voxel_dir", type=str, required=True)
    parser.add_argument("--pred_dir", type=str, required=True, help="Directory containing *_pcl.npy or *_voxel.npy")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_files", type=int, default=20)
    parser.add_argument("--pred_kind", choices=["auto", "pcl", "voxel"], default="auto")
    parser.add_argument("--occ_threshold", type=float, default=0.1)
    parser.add_argument("--pc_range", type=float, nargs=6, default=[0, -20, -6, 120, 20, 10])
    parser.add_argument("--x_range", type=float, nargs=2, default=[0, 120])
    parser.add_argument("--y_range", type=float, nargs=2, default=[-20, 20])
    parser.add_argument("--z_range", type=float, nargs=2, default=[-6, 10])
    parser.add_argument("--max_plot_points", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def list_frame_ids(folder: str) -> List[str]:
    frame_ids = []
    for name in os.listdir(folder):
        if name.endswith(".npz") or name.endswith(".npy"):
            stem = os.path.splitext(name)[0]
            if stem.endswith("_pcl") or stem.endswith("_voxel"):
                stem = stem.rsplit("_", 1)[0]
            frame_ids.append(stem)
    return sorted(set(frame_ids))


def load_sparse_npz(path: str) -> np.ndarray:
    data = np.load(path)
    dense = np.zeros(tuple(data["shape"]), dtype=np.float32)
    coords = data["coords"]
    if coords.shape[0] > 0:
        dense[coords[:, 0], coords[:, 1], coords[:, 2]] = data["features"]
    return dense


def load_array(path: str) -> np.ndarray:
    if path.endswith(".npz"):
        return load_sparse_npz(path)
    return np.load(path).astype(np.float32)


def resized_occ_from_voxel_xyzc(voxel: np.ndarray, target_size=(32, 128, 128), mask_channel: Optional[int] = None) -> np.ndarray:
    tensor = torch.from_numpy(voxel).permute(3, 2, 0, 1)
    resized = resize_voxel_channels(tensor, target_size, mask_channel=mask_channel)
    return resized[0].cpu().numpy()


def voxel_xyzc_to_points(
    voxel: np.ndarray,
    pc_range: List[float],
    occ_threshold: float,
) -> np.ndarray:
    if voxel.ndim != 4:
        raise ValueError(f"Expected XY-Z-C voxel, got {voxel.shape}")

    occ = voxel[..., 0]
    idx = np.argwhere(occ > occ_threshold)
    if idx.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32)

    nx, ny, nz = occ.shape
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range
    voxel_size = (
        (x_max - x_min) / max(nx, 1),
        (y_max - y_min) / max(ny, 1),
        (z_max - z_min) / max(nz, 1),
    )

    x = x_min + (idx[:, 0].astype(np.float32) + 0.5) * voxel_size[0]
    y = y_min + (idx[:, 1].astype(np.float32) + 0.5) * voxel_size[1]
    z = z_min + (idx[:, 2].astype(np.float32) + 0.5) * voxel_size[2]
    intensity = voxel[idx[:, 0], idx[:, 1], idx[:, 2], 1] if voxel.shape[-1] > 1 else occ[idx[:, 0], idx[:, 1], idx[:, 2]]
    return np.stack([x, y, z, intensity], axis=1).astype(np.float32)


def voxel_czxy_to_points(
    voxel: np.ndarray,
    pc_range: List[float],
    occ_threshold: float,
) -> np.ndarray:
    if voxel.ndim != 4:
        raise ValueError(f"Expected C-Z-X-Y voxel, got {voxel.shape}")

    occ = voxel[0]
    idx = np.argwhere(occ > occ_threshold)
    if idx.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32)

    nz, nx, ny = occ.shape
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range
    voxel_size = (
        (x_max - x_min) / max(nx, 1),
        (y_max - y_min) / max(ny, 1),
        (z_max - z_min) / max(nz, 1),
    )

    z_idx, x_idx, y_idx = idx[:, 0], idx[:, 1], idx[:, 2]
    x = x_min + (x_idx.astype(np.float32) + 0.5) * voxel_size[0]
    y = y_min + (y_idx.astype(np.float32) + 0.5) * voxel_size[1]
    z = z_min + (z_idx.astype(np.float32) + 0.5) * voxel_size[2]
    intensity = voxel[1, z_idx, x_idx, y_idx] if voxel.shape[0] > 1 else occ[z_idx, x_idx, y_idx]
    return np.stack([x, y, z, intensity], axis=1).astype(np.float32)


def load_pred_points(pred_dir: str, frame_id: str, pred_kind: str, pc_range: List[float], occ_threshold: float) -> Tuple[np.ndarray, str]:
    pcl_path = os.path.join(pred_dir, f"{frame_id}_pcl.npy")
    voxel_path = os.path.join(pred_dir, f"{frame_id}_voxel.npy")

    if pred_kind in ("auto", "pcl") and os.path.exists(pcl_path):
        pts = np.load(pcl_path).astype(np.float32)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError(f"Invalid point cloud shape for {pcl_path}: {pts.shape}")
        return pts[:, :4] if pts.shape[1] >= 4 else np.pad(pts[:, :3], ((0, 0), (0, 1))), pcl_path

    if pred_kind in ("auto", "voxel") and os.path.exists(voxel_path):
        voxel = np.load(voxel_path).astype(np.float32)
        return voxel_czxy_to_points(voxel, pc_range, occ_threshold), voxel_path

    raise FileNotFoundError(f"No prediction found for frame {frame_id} in {pred_dir}")


def filter_points(points: np.ndarray, x_range: List[float], y_range: List[float], z_range: List[float]) -> np.ndarray:
    if points.shape[0] == 0:
        return points
    keep = (
        (points[:, 0] >= x_range[0])
        & (points[:, 0] <= x_range[1])
        & (points[:, 1] >= y_range[0])
        & (points[:, 1] <= y_range[1])
        & (points[:, 2] >= z_range[0])
        & (points[:, 2] <= z_range[1])
    )
    return points[keep]


def downsample(points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


def point_stats(points: np.ndarray) -> Dict[str, float]:
    if points.shape[0] == 0:
        keys = ["count", "cx", "cy", "cz", "z_std", "x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]
        return {k: (0 if k == "count" else float("nan")) for k in keys}
    xyz = points[:, :3]
    return {
        "count": int(points.shape[0]),
        "cx": float(np.mean(xyz[:, 0])),
        "cy": float(np.mean(xyz[:, 1])),
        "cz": float(np.mean(xyz[:, 2])),
        "z_std": float(np.std(xyz[:, 2])),
        "x_min": float(np.min(xyz[:, 0])),
        "x_max": float(np.max(xyz[:, 0])),
        "y_min": float(np.min(xyz[:, 1])),
        "y_max": float(np.max(xyz[:, 1])),
        "z_min": float(np.min(xyz[:, 2])),
        "z_max": float(np.max(xyz[:, 2])),
    }


def chamfer(a: np.ndarray, b: np.ndarray) -> float:
    if cKDTree is None or a.shape[0] == 0 or b.shape[0] == 0:
        return float("nan")
    tree_a = cKDTree(a[:, :3])
    tree_b = cKDTree(b[:, :3])
    d_ab, _ = tree_b.query(a[:, :3], k=1)
    d_ba, _ = tree_a.query(b[:, :3], k=1)
    return float(np.mean(d_ab) + np.mean(d_ba))


def nearest_mean(src: np.ndarray, dst: np.ndarray) -> float:
    if cKDTree is None or src.shape[0] == 0 or dst.shape[0] == 0:
        return float("nan")
    tree = cKDTree(dst[:, :3])
    dists, _ = tree.query(src[:, :3], k=1)
    return float(np.mean(dists))


def scatter_projection(ax, points: np.ndarray, plane: str, title: str, color: str, ranges: Tuple[List[float], List[float], List[float]]) -> None:
    x_range, y_range, z_range = ranges
    if plane == "bev":
        a, b = points[:, 0], points[:, 1]
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_xlabel("X/m")
        ax.set_ylabel("Y/m")
    elif plane == "xz":
        a, b = points[:, 0], points[:, 2]
        ax.set_xlim(x_range)
        ax.set_ylim(z_range)
        ax.set_xlabel("X/m")
        ax.set_ylabel("Z/m")
    else:
        a, b = points[:, 1], points[:, 2]
        ax.set_xlim(y_range)
        ax.set_ylim(z_range)
        ax.set_xlabel("Y/m")
        ax.set_ylabel("Z/m")

    ax.set_title(title, fontsize=10)
    ax.grid(True, linewidth=0.3, alpha=0.35)
    if points.shape[0] > 0:
        ax.scatter(a, b, s=0.35, c=color, alpha=0.75, linewidths=0)
    ax.set_aspect("auto")


def write_frame_figure(
    out_path: str,
    frame_id: str,
    radar: np.ndarray,
    target: np.ndarray,
    pred: np.ndarray,
    stats: Dict[str, float],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> None:
    radar_plot = downsample(radar, args.max_plot_points, rng)
    target_plot = downsample(target, args.max_plot_points, rng)
    pred_plot = downsample(pred, args.max_plot_points, rng)
    ranges = (args.x_range, args.y_range, args.z_range)

    fig, axes = plt.subplots(3, 3, figsize=(14, 10), dpi=150)
    fig.suptitle(
        f"Frame {frame_id} | pred-target Chamfer={stats['pred_target_chamfer']:.3f} | "
        f"centroid dz={stats['pred_target_dz']:.3f}m",
        fontsize=12,
    )

    columns = [
        ("Radar input", radar_plot, "#ffb000"),
        ("LiDAR target", target_plot, "#2ca02c"),
        ("Prediction", pred_plot, "#1f77b4"),
    ]
    rows = [("bev", "BEV X-Y"), ("xz", "Side X-Z"), ("yz", "Side Y-Z")]

    for row_idx, (plane, row_title) in enumerate(rows):
        for col_idx, (name, pts, color) in enumerate(columns):
            scatter_projection(
                axes[row_idx, col_idx],
                pts,
                plane,
                f"{name} | {row_title} | n={pts.shape[0]}",
                color,
                ranges,
            )

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)


def build_row(
    frame_id: str, radar: np.ndarray, target: np.ndarray, pred: np.ndarray, pred_file: str
) -> Tuple[Dict[str, Union[str, float]], Dict[str, float]]:
    radar_s = point_stats(radar)
    target_s = point_stats(target)
    pred_s = point_stats(pred)
    metrics: Dict[str, float] = {
        "radar_count": radar_s["count"],
        "target_count": target_s["count"],
        "pred_count": pred_s["count"],
        "pred_target_chamfer": chamfer(pred, target),
        "radar_target_chamfer": chamfer(radar, target),
        "pred_to_target_nn_mean": nearest_mean(pred, target),
        "target_to_pred_nn_mean": nearest_mean(target, pred),
        "pred_target_dx": pred_s["cx"] - target_s["cx"],
        "pred_target_dy": pred_s["cy"] - target_s["cy"],
        "pred_target_dz": pred_s["cz"] - target_s["cz"],
        "radar_target_dx": radar_s["cx"] - target_s["cx"],
        "radar_target_dy": radar_s["cy"] - target_s["cy"],
        "radar_target_dz": radar_s["cz"] - target_s["cz"],
        "target_z_std": target_s["z_std"],
        "pred_z_std": pred_s["z_std"],
        "pred_target_count_ratio": pred_s["count"] / max(target_s["count"], 1),
    }
    for prefix, stats in (("radar", radar_s), ("target", target_s), ("pred", pred_s)):
        for key in ("cx", "cy", "cz", "x_min", "x_max", "y_min", "y_max", "z_min", "z_max"):
            metrics[f"{prefix}_{key}"] = stats[key]
    meta = {"frame_id": frame_id, "pred_file": pred_file}
    return {**meta, **metrics}, metrics


def safe_nanmean(values: List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def write_report(output_dir: str, rows: List[Dict[str, float]]) -> None:
    mean_chamfer = safe_nanmean([r["pred_target_chamfer"] for r in rows])
    mean_radar_chamfer = safe_nanmean([r["radar_target_chamfer"] for r in rows])
    mean_dz = safe_nanmean([abs(r["pred_target_dz"]) for r in rows])
    mean_radar_dy = safe_nanmean([abs(r["radar_target_dy"]) for r in rows])
    mean_radar_dz = safe_nanmean([abs(r["radar_target_dz"]) for r in rows])
    mean_count_ratio = safe_nanmean([r["pred_target_count_ratio"] for r in rows])
    mean_pred_z_std = safe_nanmean([r["pred_z_std"] for r in rows])
    mean_target_z_std = safe_nanmean([r["target_z_std"] for r in rows])

    flags = []
    if mean_count_ratio < 0.2:
        flags.append("Prediction is much sparser than target; check occupancy threshold and loss imbalance.")
    if mean_count_ratio > 2.0:
        flags.append("Prediction is much denser than target; check threshold/top-k fallback and false-positive penalty.")
    if mean_target_z_std > 0 and mean_pred_z_std / mean_target_z_std < 0.5:
        flags.append("Prediction height spread is compressed; add height-aware loss or inspect z-axis preprocessing.")
    if mean_dz > 1.0:
        flags.append("Prediction-target vertical centroid offset is large; inspect z origin, pc_range, and extrinsic alignment.")
    if mean_chamfer > mean_radar_chamfer:
        flags.append("Generated output is farther from target than raw radar; validate training target pairing and conditioning.")
    if mean_radar_dy > 1.0 or mean_radar_dz > 1.0:
        flags.append(
            "Raw radar and target centroids have a stable cross-axis offset; re-run preprocessing with verified calibration "
            "and inspect ground filtering/extrinsics before retraining."
        )
    if not flags:
        flags.append("No single coarse failure mode dominates; inspect per-frame PNGs for local structure errors.")

    report_path = os.path.join(output_dir, "diagnosis_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Generation Quality Diagnosis\n\n")
        f.write(f"- Frames evaluated: {len(rows)}\n")
        f.write(f"- Mean pred-target Chamfer: {mean_chamfer:.4f}\n")
        f.write(f"- Mean radar-target Chamfer: {mean_radar_chamfer:.4f}\n")
        f.write(f"- Mean |pred-target dz|: {mean_dz:.4f} m\n")
        f.write(f"- Mean |radar-target dy|: {mean_radar_dy:.4f} m\n")
        f.write(f"- Mean |radar-target dz|: {mean_radar_dz:.4f} m\n")
        f.write(f"- Mean pred/target count ratio: {mean_count_ratio:.4f}\n")
        f.write(f"- Mean pred z std / target z std: {mean_pred_z_std:.4f} / {mean_target_z_std:.4f}\n\n")
        f.write("## Diagnostic Flags\n\n")
        for flag in flags:
            f.write(f"- {flag}\n")


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    frame_ids = sorted(
        set(list_frame_ids(args.radar_voxel_dir))
        & set(list_frame_ids(args.target_voxel_dir))
        & set(list_frame_ids(args.pred_dir))
    )
    if args.max_files > 0:
        frame_ids = frame_ids[: args.max_files]
    if not frame_ids:
        raise RuntimeError("No shared frame ids found across radar, target, and prediction dirs")

    rows: List[Dict[str, Union[str, float]]] = []
    metrics_rows: List[Dict[str, float]] = []
    image_dir = os.path.join(args.output_dir, "frames")
    os.makedirs(image_dir, exist_ok=True)

    for frame_id in frame_ids:
        radar_path = os.path.join(args.radar_voxel_dir, f"{frame_id}.npz")
        target_path = os.path.join(args.target_voxel_dir, f"{frame_id}.npz")
        if not os.path.exists(radar_path):
            radar_path = os.path.join(args.radar_voxel_dir, f"{frame_id}.npy")
        if not os.path.exists(target_path):
            target_path = os.path.join(args.target_voxel_dir, f"{frame_id}.npy")

        radar = voxel_xyzc_to_points(load_array(radar_path), args.pc_range, args.occ_threshold)
        target = voxel_xyzc_to_points(load_array(target_path), args.pc_range, args.occ_threshold)
        pred, pred_file = load_pred_points(args.pred_dir, frame_id, args.pred_kind, args.pc_range, args.occ_threshold)

        radar = filter_points(radar, args.x_range, args.y_range, args.z_range)
        target = filter_points(target, args.x_range, args.y_range, args.z_range)
        pred = filter_points(pred, args.x_range, args.y_range, args.z_range)

        row, metrics = build_row(frame_id, radar, target, pred, pred_file)
        rows.append(row)
        metrics_rows.append(metrics)
        write_frame_figure(
            os.path.join(image_dir, f"{frame_id}_diagnosis.png"),
            frame_id,
            radar,
            target,
            pred,
            metrics,
            args,
            rng,
        )

    csv_path = os.path.join(args.output_dir, "diagnosis_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    write_report(args.output_dir, metrics_rows)
    print(f"Saved diagnosis images to: {image_dir}")
    print(f"Saved metrics to: {csv_path}")
    print(f"Saved report to: {os.path.join(args.output_dir, 'diagnosis_report.md')}")


if __name__ == "__main__":
    main()
