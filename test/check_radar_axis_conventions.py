#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""针对激光雷达帧测试原始毫米波雷达的 XYZ 坐标轴约定。"""

import argparse
import csv
import itertools
import os
from typing import Dict, List, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None


def load_calib(path: str) -> Tuple[np.ndarray, np.ndarray]:
    r_mat = np.eye(3, dtype=np.float32)
    t_vec = np.zeros(3, dtype=np.float32)
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("R:"):
                r_mat = np.asarray([float(x) for x in line.split(":", 1)[1].split()], dtype=np.float32).reshape(3, 3)
            elif line.startswith("T:"):
                t_vec = np.asarray([float(x) for x in line.split(":", 1)[1].split()[:3]], dtype=np.float32)
    return r_mat, t_vec


def read_indices(path: str) -> List[int]:
    with open(path, "r", encoding="utf-8") as handle:
        return [int(line.strip()) for line in handle if line.strip()]


def transform(points: np.ndarray, r_mat: np.ndarray, t_vec: np.ndarray) -> np.ndarray:
    out = points.copy()
    out[:, :3] = np.dot(points[:, :3], r_mat.T) + t_vec
    return out


def apply_axis_candidate(points: np.ndarray, perm: Tuple[int, int, int], signs: Tuple[int, int, int]) -> np.ndarray:
    out = points.copy()
    xyz = points[:, list(perm)].copy()
    out[:, :3] = xyz * np.asarray(signs, dtype=np.float32)
    return out


def filter_range(points: np.ndarray, pc_range: List[float]) -> np.ndarray:
    mask = (
        (points[:, 0] >= pc_range[0])
        & (points[:, 0] < pc_range[3])
        & (points[:, 1] >= pc_range[1])
        & (points[:, 1] < pc_range[4])
        & (points[:, 2] >= pc_range[2])
        & (points[:, 2] < pc_range[5])
    )
    return points[mask]


def chamfer(a: np.ndarray, b: np.ndarray) -> float:
    if cKDTree is None or a.shape[0] == 0 or b.shape[0] == 0:
        return float("nan")
    tree_a = cKDTree(a[:, :3])
    tree_b = cKDTree(b[:, :3])
    da, _ = tree_b.query(a[:, :3], k=1)
    db, _ = tree_a.query(b[:, :3], k=1)
    return float(np.mean(da) + np.mean(db))


def metrics(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    if a.shape[0] == 0 or b.shape[0] == 0:
        return {
            "radar_count": float(a.shape[0]),
            "lidar_count": float(b.shape[0]),
            "dx": float("nan"),
            "dy": float("nan"),
            "dz": float("nan"),
            "abs_dx": float("nan"),
            "abs_dy": float("nan"),
            "abs_dz": float("nan"),
            "chamfer": float("nan"),
        }
    delta = np.mean(a[:, :3], axis=0) - np.mean(b[:, :3], axis=0)
    return {
        "radar_count": float(a.shape[0]),
        "lidar_count": float(b.shape[0]),
        "dx": float(delta[0]),
        "dy": float(delta[1]),
        "dz": float(delta[2]),
        "abs_dx": float(abs(delta[0])),
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
    parser = argparse.ArgumentParser(description="Check raw radar axis conventions")
    parser.add_argument("--raw_root", default="./Data/NTU4DRadLM_Raw")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--calib", default="./Data/config/calib_radar_to_livox.txt")
    parser.add_argument("--output_csv", default="")
    parser.add_argument("--max_frames", type=int, default=120)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--pc_range", type=float, nargs=6, default=[0, -20, -6, 120, 20, 10])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_root = os.path.join(args.raw_root, args.scene)
    radar_dir = os.path.join(scene_root, "radar_pcl")
    lidar_dir = os.path.join(scene_root, "livox_lidar")
    radar_indices = read_indices(os.path.join(scene_root, "radar_index_sequence.txt"))
    lidar_indices = read_indices(os.path.join(scene_root, "lidar_index_sequence.txt"))
    radar_files = sorted([name for name in os.listdir(radar_dir) if name.endswith(".npy")])
    lidar_files = sorted([name for name in os.listdir(lidar_dir) if name.endswith(".npy")])
    r_mat, t_vec = load_calib(args.calib)

    frame_ids = list(range(0, min(len(radar_indices), len(lidar_indices)), max(1, args.stride)))[: args.max_frames]
    rows = []
    for perm in itertools.permutations((0, 1, 2)):
        for signs in itertools.product((-1, 1), repeat=3):
            vals: Dict[str, List[float]] = {
                "radar_count": [],
                "lidar_count": [],
                "dx": [],
                "dy": [],
                "dz": [],
                "abs_dx": [],
                "abs_dy": [],
                "abs_dz": [],
                "chamfer": [],
            }
            for frame_id in frame_ids:
                radar_idx = radar_indices[frame_id]
                lidar_idx = lidar_indices[frame_id]
                if radar_idx >= len(radar_files) or lidar_idx >= len(lidar_files):
                    continue
                radar = np.load(os.path.join(radar_dir, radar_files[radar_idx])).astype(np.float32)
                lidar = np.load(os.path.join(lidar_dir, lidar_files[lidar_idx])).astype(np.float32)
                radar_candidate = transform(apply_axis_candidate(radar, perm, signs), r_mat, t_vec)
                radar_candidate = filter_range(radar_candidate, args.pc_range)
                lidar = filter_range(lidar, args.pc_range)
                item = metrics(radar_candidate, lidar)
                for key, value in item.items():
                    vals[key].append(value)

            row = {
                "perm": "".join("xyz"[i] for i in perm),
                "signs": "".join("+" if s > 0 else "-" for s in signs),
                **{key: nanmean(value) for key, value in vals.items()},
            }
            row["score_abs_xyz"] = row["abs_dx"] + row["abs_dy"] + row["abs_dz"]
            rows.append(row)

    rows.sort(key=lambda row: (row["score_abs_xyz"], row["chamfer"]))
    output_csv = args.output_csv or os.path.join("./Result/alignment_check", args.scene, "radar_axis_candidates.csv")
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "perm",
            "signs",
            "radar_count",
            "lidar_count",
            "dx",
            "dy",
            "dz",
            "abs_dx",
            "abs_dy",
            "abs_dz",
            "chamfer",
            "score_abs_xyz",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved radar axis candidates to: {output_csv}")
    for row in rows[:10]:
        print(
            f"perm={row['perm']} signs={row['signs']} "
            f"dx={row['dx']:.3f} dy={row['dy']:.3f} dz={row['dz']:.3f} "
            f"chamfer={row['chamfer']:.3f} count={row['radar_count']:.1f}"
        )


if __name__ == "__main__":
    main()

