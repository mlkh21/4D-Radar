#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compare radar/lidar/target voxel geometry after preprocessing."""

import argparse
import csv
import os
from typing import Dict, List, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None


def load_sparse_voxel(path: str) -> np.ndarray:
    data = np.load(path)
    voxel = np.zeros(tuple(data["shape"]), dtype=np.float32)
    coords = data["coords"]
    if coords.shape[0] > 0:
        voxel[coords[:, 0], coords[:, 1], coords[:, 2]] = data["features"]
    return voxel


def load_voxel(path: str) -> np.ndarray:
    if path.endswith(".npz"):
        return load_sparse_voxel(path)
    return np.load(path).astype(np.float32)


def find_voxel_file(folder: str, frame_id: str) -> str:
    for ext in (".npz", ".npy"):
        path = os.path.join(folder, frame_id + ext)
        if os.path.exists(path):
            return path
    return ""


def voxel_to_points(voxel: np.ndarray, pc_range: List[float], threshold: float) -> np.ndarray:
    occ = voxel[..., 0]
    coords = np.argwhere(occ > threshold)
    if coords.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    nx, ny, nz = occ.shape
    voxel_size = (
        (pc_range[3] - pc_range[0]) / max(nx, 1),
        (pc_range[4] - pc_range[1]) / max(ny, 1),
        (pc_range[5] - pc_range[2]) / max(nz, 1),
    )
    x = pc_range[0] + (coords[:, 0].astype(np.float32) + 0.5) * voxel_size[0]
    y = pc_range[1] + (coords[:, 1].astype(np.float32) + 0.5) * voxel_size[1]
    z = pc_range[2] + (coords[:, 2].astype(np.float32) + 0.5) * voxel_size[2]
    return np.stack([x, y, z], axis=1)


def chamfer(a: np.ndarray, b: np.ndarray) -> float:
    if cKDTree is None or a.shape[0] == 0 or b.shape[0] == 0:
        return float("nan")
    tree_a = cKDTree(a)
    tree_b = cKDTree(b)
    da, _ = tree_b.query(a, k=1)
    db, _ = tree_a.query(b, k=1)
    return float(np.mean(da) + np.mean(db))


def pair_metrics(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    if a.shape[0] == 0 or b.shape[0] == 0:
        return {
            "count_a": float(a.shape[0]),
            "count_b": float(b.shape[0]),
            "dx": float("nan"),
            "dy": float("nan"),
            "dz": float("nan"),
            "abs_dy": float("nan"),
            "abs_dz": float("nan"),
            "chamfer": float("nan"),
        }
    delta = np.mean(a, axis=0) - np.mean(b, axis=0)
    return {
        "count_a": float(a.shape[0]),
        "count_b": float(b.shape[0]),
        "dx": float(delta[0]),
        "dy": float(delta[1]),
        "dz": float(delta[2]),
        "abs_dy": float(abs(delta[1])),
        "abs_dz": float(abs(delta[2])),
        "chamfer": chamfer(a, b),
    }


def nanmean(values: List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare radar/lidar/target voxel geometry")
    parser.add_argument("--pre_dir", required=True, help="Preprocessed scene root, e.g. Data/NTU4DRadLM_Pre/loop3")
    parser.add_argument("--output_csv", default="", help="Output csv path")
    parser.add_argument("--max_files", type=int, default=120)
    parser.add_argument("--occ_threshold", type=float, default=0.1)
    parser.add_argument("--pc_range", type=float, nargs=6, default=[0, -20, -6, 120, 20, 10])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    radar_dir = os.path.join(args.pre_dir, "radar_voxel")
    lidar_dir = os.path.join(args.pre_dir, "lidar_voxel")
    target_dir = os.path.join(args.pre_dir, "target_voxel")
    for folder in (radar_dir, lidar_dir, target_dir):
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Missing voxel folder: {folder}")

    frame_ids = sorted(
        {os.path.splitext(name)[0] for name in os.listdir(radar_dir) if name.endswith((".npz", ".npy"))}
        & {os.path.splitext(name)[0] for name in os.listdir(lidar_dir) if name.endswith((".npz", ".npy"))}
        & {os.path.splitext(name)[0] for name in os.listdir(target_dir) if name.endswith((".npz", ".npy"))}
    )
    if args.max_files > 0:
        frame_ids = frame_ids[: args.max_files]
    if not frame_ids:
        raise RuntimeError("No shared voxel frame ids found")

    output_csv = args.output_csv or os.path.join(args.pre_dir, "voxel_triplet_metrics.csv")
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    pairs: List[Tuple[str, str]] = [
        ("radar", "lidar"),
        ("radar", "target"),
        ("lidar", "target"),
    ]
    rows = []
    summary: Dict[str, Dict[str, List[float]]] = {
        f"{a}_vs_{b}": {"dx": [], "dy": [], "dz": [], "abs_dy": [], "abs_dz": [], "chamfer": [], "count_a": [], "count_b": []}
        for a, b in pairs
    }

    for frame_id in frame_ids:
        radar = voxel_to_points(load_voxel(find_voxel_file(radar_dir, frame_id)), args.pc_range, args.occ_threshold)
        lidar = voxel_to_points(load_voxel(find_voxel_file(lidar_dir, frame_id)), args.pc_range, args.occ_threshold)
        target = voxel_to_points(load_voxel(find_voxel_file(target_dir, frame_id)), args.pc_range, args.occ_threshold)
        clouds = {"radar": radar, "lidar": lidar, "target": target}
        for a_name, b_name in pairs:
            pair_name = f"{a_name}_vs_{b_name}"
            metrics = pair_metrics(clouds[a_name], clouds[b_name])
            rows.append({"frame_id": frame_id, "pair": pair_name, **metrics})
            for key, value in metrics.items():
                summary[pair_name][key].append(value)

    fieldnames = ["frame_id", "pair", "count_a", "count_b", "dx", "dy", "dz", "abs_dy", "abs_dz", "chamfer"]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        for pair_name, values in summary.items():
            writer.writerow(
                {
                    "frame_id": "__summary__",
                    "pair": pair_name,
                    **{key: nanmean(val) for key, val in values.items()},
                }
            )

    print(f"Saved triplet voxel metrics to: {output_csv}")
    for pair_name, values in summary.items():
        print(
            f"{pair_name}: "
            f"dy={nanmean(values['dy']):.4f}, "
            f"dz={nanmean(values['dz']):.4f}, "
            f"abs_dy={nanmean(values['abs_dy']):.4f}, "
            f"abs_dz={nanmean(values['abs_dz']):.4f}, "
            f"chamfer={nanmean(values['chamfer']):.4f}"
        )


if __name__ == "__main__":
    main()

