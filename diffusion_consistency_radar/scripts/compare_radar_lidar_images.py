#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate side-by-side BEV or 3D comparison images for predicted radar point clouds and raw LiDAR point clouds."""

import argparse
import csv
import io
import math
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate radar-vs-lidar BEV or 3D comparison images")
    parser.add_argument("--pred_pcl_dir", type=str, required=True, help="Directory containing predicted *_pcl.npy files")
    parser.add_argument("--raw_livox_dir", type=str, required=True, help="Directory containing raw livox .npy files")
    parser.add_argument("--lidar_index_file", type=str, default="", help="Optional lidar_index_sequence.txt for mapping")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for comparison images")
    parser.add_argument("--max_files", type=int, default=0, help="Max number of files to process (0 means all)")
    parser.add_argument("--x_range", type=float, nargs=2, default=[0.0, 120.0], help="X range in meters")
    parser.add_argument("--y_range", type=float, nargs=2, default=[-20.0, 20.0], help="Y range in meters")
    parser.add_argument("--resolution", type=float, default=0.2, help="BEV resolution in meters per pixel")
    parser.add_argument("--mode", type=str, default="bev", choices=["bev", "3d"], help="Comparison mode")
    parser.add_argument("--point_size", type=float, default=1.0, help="Point size for 3D scatter mode")
    parser.add_argument("--z_range", type=float, nargs=2, default=None, help="Optional Z range for 3D mode")
    return parser.parse_args()


def list_npy_files(folder: str, suffix: Optional[str] = None) -> List[str]:
    names = [n for n in os.listdir(folder) if n.endswith(".npy")]
    if suffix:
        names = [n for n in names if n.endswith(suffix)]
    names.sort()
    return names


def load_lidar_indices(index_file: str) -> List[int]:
    indices: List[int] = []
    with open(index_file, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                indices.append(int(s))
    return indices


def to_xyz(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Point cloud must be 2D with >=3 columns, got {arr.shape}")
    return arr[:, :3].astype(np.float32)


def pointcloud_to_bev(
    points_xyz: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    resolution: float,
) -> np.ndarray:
    width = int(math.ceil((x_max - x_min) / resolution))
    height = int(math.ceil((y_max - y_min) / resolution))
    bev = np.zeros((height, width), dtype=np.uint8)

    if points_xyz.shape[0] == 0:
        return bev

    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    valid = (x >= x_min) & (x < x_max) & (y >= y_min) & (y < y_max)
    if not np.any(valid):
        return bev

    x = x[valid]
    y = y[valid]

    col = ((x - x_min) / resolution).astype(np.int32)
    row = ((y - y_min) / resolution).astype(np.int32)

    row = np.clip(row, 0, height - 1)
    col = np.clip(col, 0, width - 1)
    bev[row, col] = 255

    # Make sparse points easier to see.
    bev = cv2.dilate(bev, np.ones((3, 3), dtype=np.uint8), iterations=1)
    bev = np.flipud(bev)
    return bev


def compose_side_by_side(left_gray: np.ndarray, right_gray: np.ndarray, left_title: str, right_title: str) -> np.ndarray:
    left = cv2.cvtColor(left_gray, cv2.COLOR_GRAY2BGR)
    right = cv2.cvtColor(right_gray, cv2.COLOR_GRAY2BGR)

    if left.shape != right.shape:
        right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_NEAREST)

    canvas = np.concatenate([left, right], axis=1)
    cv2.putText(canvas, left_title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, right_title, (left.shape[1] + 12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return canvas


def filter_points(points_xyz: np.ndarray, x_range: Tuple[float, float], y_range: Tuple[float, float], z_range: Optional[Tuple[float, float]] = None) -> np.ndarray:
    if points_xyz.shape[0] == 0:
        return points_xyz

    valid = (
        (points_xyz[:, 0] >= x_range[0])
        & (points_xyz[:, 0] < x_range[1])
        & (points_xyz[:, 1] >= y_range[0])
        & (points_xyz[:, 1] < y_range[1])
    )
    if z_range is not None:
        valid &= (points_xyz[:, 2] >= z_range[0]) & (points_xyz[:, 2] < z_range[1])
    return points_xyz[valid]


def pointcloud_to_3d_figure(
    pred_xyz: np.ndarray,
    lidar_xyz: np.ndarray,
    point_size: float = 1.0,
    z_range: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    fig = plt.figure(figsize=(14, 6), facecolor="black")
    ax_pred = fig.add_subplot(1, 2, 1, projection="3d")
    ax_lidar = fig.add_subplot(1, 2, 2, projection="3d")

    def downsample(points: np.ndarray, max_points: int = 20000) -> np.ndarray:
        if points.shape[0] <= max_points:
            return points
        indices = np.random.choice(points.shape[0], size=max_points, replace=False)
        return points[indices]

    pred_xyz = downsample(pred_xyz)
    lidar_xyz = downsample(lidar_xyz)

    for ax, title, points in (
        (ax_pred, "Pred Radar", pred_xyz),
        (ax_lidar, "Raw LiDAR", lidar_xyz),
    ):
        ax.set_facecolor("black")
        ax.set_title(title, color="yellow", fontsize=14, pad=12)
        ax.tick_params(colors="white")
        ax.xaxis.pane.set_facecolor((0, 0, 0, 1))
        ax.yaxis.pane.set_facecolor((0, 0, 0, 1))
        ax.zaxis.pane.set_facecolor((0, 0, 0, 1))
        ax.grid(True, color="white", alpha=0.15)
        if points.shape[0] > 0:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=point_size, c=points[:, 2], cmap="viridis", alpha=0.85, linewidths=0)
            ax.set_xlim(points[:, 0].min(), points[:, 0].max())
            ax.set_ylim(points[:, 1].min(), points[:, 1].max())
            ax.set_zlim(points[:, 2].min(), points[:, 2].max())
        if z_range is not None:
            ax.set_zlim(z_range[0], z_range[1])
        ax.set_xlabel("X", color="white")
        ax.set_ylabel("Y", color="white")
        ax.set_zlabel("Z", color="white")

    plt.tight_layout(pad=1.0)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    buffer.seek(0)
    image = cv2.imdecode(np.frombuffer(buffer.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
    plt.close(fig)
    if image is None:
        raise RuntimeError("Failed to decode 3D comparison figure")
    return image


def main() -> None:
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    pred_files = list_npy_files(args.pred_pcl_dir, suffix="_pcl.npy")
    if not pred_files:
        raise RuntimeError(f"No *_pcl.npy files found in {args.pred_pcl_dir}")

    if args.max_files > 0:
        pred_files = pred_files[: args.max_files]

    lidar_files = list_npy_files(args.raw_livox_dir)
    if not lidar_files:
        raise RuntimeError(f"No .npy LiDAR files found in {args.raw_livox_dir}")

    lidar_indices: List[int] = []
    if args.lidar_index_file:
        lidar_indices = load_lidar_indices(args.lidar_index_file)

    summary_csv = os.path.join(args.output_dir, "image_comparison_pairs.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "pred_file", "lidar_file", "output_image"])

        for i, pred_name in enumerate(pred_files):
            lidar_name: Optional[str] = None
            if lidar_indices:
                if i < len(lidar_indices):
                    idx = lidar_indices[i]
                    if 0 <= idx < len(lidar_files):
                        lidar_name = lidar_files[idx]
            else:
                if i < len(lidar_files):
                    lidar_name = lidar_files[i]

            if lidar_name is None:
                continue

            pred_path = os.path.join(args.pred_pcl_dir, pred_name)
            lidar_path = os.path.join(args.raw_livox_dir, lidar_name)

            pred_xyz = to_xyz(np.load(pred_path))
            lidar_xyz = to_xyz(np.load(lidar_path))

            if args.mode == "bev":
                pred_bev = pointcloud_to_bev(
                    pred_xyz,
                    x_min=args.x_range[0],
                    x_max=args.x_range[1],
                    y_min=args.y_range[0],
                    y_max=args.y_range[1],
                    resolution=args.resolution,
                )
                lidar_bev = pointcloud_to_bev(
                    lidar_xyz,
                    x_min=args.x_range[0],
                    x_max=args.x_range[1],
                    y_min=args.y_range[0],
                    y_max=args.y_range[1],
                    resolution=args.resolution,
                )
                merged = compose_side_by_side(pred_bev, lidar_bev, "Pred Radar", "Raw LiDAR")
            else:
                pred_filtered = filter_points(pred_xyz, tuple(args.x_range), tuple(args.y_range), tuple(args.z_range) if args.z_range else None)
                lidar_filtered = filter_points(lidar_xyz, tuple(args.x_range), tuple(args.y_range), tuple(args.z_range) if args.z_range else None)
                merged = pointcloud_to_3d_figure(pred_filtered, lidar_filtered, point_size=args.point_size, z_range=tuple(args.z_range) if args.z_range else None)

            sample_id = pred_name.replace("_pcl.npy", "")
            out_name = f"{sample_id}_compare.png"
            out_path = os.path.join(args.output_dir, out_name)
            cv2.imwrite(out_path, merged)

            writer.writerow([i, pred_name, lidar_name, out_name])

    print(f"Saved comparison images to: {args.output_dir}")
    print(f"Saved pair list to: {summary_csv}")


if __name__ == "__main__":
    main()
