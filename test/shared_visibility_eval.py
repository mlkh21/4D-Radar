#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate radar/LiDAR overlap on shared visible regions.

This diagnostic avoids using only global centroid offsets. It reports nearest
neighbor coverage and BEV occupancy IoU after optional range and height filters.
"""

import argparse
import csv
import math
import os
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None


DEFAULT_PC_RANGE = [0.0, -20.0, -6.0, 120.0, 20.0, 10.0]


def as_xyz(points: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Expected point array with shape (N, >=3), got {arr.shape}")
    return arr[:, :3]


def threshold_label(value: float) -> str:
    return f"{value:g}"


def nearest_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = as_xyz(a)
    b = as_xyz(b)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.asarray([], dtype=np.float32)
    if cKDTree is not None:
        tree = cKDTree(b)
        distances, _ = tree.query(a, k=1)
        return np.asarray(distances, dtype=np.float32)

    # Small fallback for environments without SciPy.
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2)).min(axis=1)


def nearest_neighbor_metrics(
    radar: Sequence[Sequence[float]],
    lidar: Sequence[Sequence[float]],
    thresholds: Iterable[float] = (0.5, 1.0, 2.0),
) -> Dict[str, float]:
    radar_xyz = as_xyz(radar)
    lidar_xyz = as_xyz(lidar)
    distances = nearest_distances(radar_xyz, lidar_xyz)
    metrics: Dict[str, float] = {
        "radar_count": int(radar_xyz.shape[0]),
        "lidar_count": int(lidar_xyz.shape[0]),
        "nn_mean": float("nan"),
        "nn_median": float("nan"),
        "nn_p90": float("nan"),
    }
    if distances.size:
        metrics["nn_mean"] = float(np.mean(distances))
        metrics["nn_median"] = float(np.median(distances))
        metrics["nn_p90"] = float(np.percentile(distances, 90))
    for threshold in thresholds:
        key = f"match_ratio_{threshold_label(float(threshold))}"
        metrics[key] = float(np.mean(distances <= threshold)) if distances.size else float("nan")
    return metrics


def _bev_cells(points: np.ndarray, pc_range: Sequence[float], cell_size: float) -> set:
    xyz = as_xyz(points)
    if xyz.shape[0] == 0:
        return set()
    mask = (
        (xyz[:, 0] >= pc_range[0])
        & (xyz[:, 0] < pc_range[3])
        & (xyz[:, 1] >= pc_range[1])
        & (xyz[:, 1] < pc_range[4])
    )
    xyz = xyz[mask]
    if xyz.shape[0] == 0:
        return set()
    ix = np.floor((xyz[:, 0] - pc_range[0]) / cell_size).astype(np.int32)
    iy = np.floor((xyz[:, 1] - pc_range[1]) / cell_size).astype(np.int32)
    return set(zip(ix.tolist(), iy.tolist()))


def bev_iou(
    radar: Sequence[Sequence[float]],
    lidar: Sequence[Sequence[float]],
    pc_range: Sequence[float] = DEFAULT_PC_RANGE,
    cell_size: float = 1.0,
) -> Dict[str, float]:
    radar_cells = _bev_cells(as_xyz(radar), pc_range, cell_size)
    lidar_cells = _bev_cells(as_xyz(lidar), pc_range, cell_size)
    intersection = len(radar_cells & lidar_cells)
    union = len(radar_cells | lidar_cells)
    return {
        "bev_intersection": intersection,
        "bev_union": union,
        "bev_iou": float(intersection / union) if union else float("nan"),
    }


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


def voxel_to_points(voxel: np.ndarray, pc_range: Sequence[float], threshold: float) -> np.ndarray:
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


def filter_points(
    points: np.ndarray,
    pc_range: Sequence[float],
    x_min: float = None,
    x_max: float = None,
    z_min: float = None,
) -> np.ndarray:
    xyz = as_xyz(points)
    if xyz.shape[0] == 0:
        return xyz
    lo_x = pc_range[0] if x_min is None else x_min
    hi_x = pc_range[3] if x_max is None else x_max
    lo_z = pc_range[2] if z_min is None else z_min
    mask = (
        (xyz[:, 0] >= lo_x)
        & (xyz[:, 0] < hi_x)
        & (xyz[:, 1] >= pc_range[1])
        & (xyz[:, 1] < pc_range[4])
        & (xyz[:, 2] >= lo_z)
        & (xyz[:, 2] < pc_range[5])
    )
    return xyz[mask]


def parse_range_bins(text: str) -> List[Tuple[str, float, float]]:
    bins = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        left, right = item.split(":", 1)
        x_min = float(left)
        x_max = float(right)
        bins.append((f"x{threshold_label(x_min)}_{threshold_label(x_max)}", x_min, x_max))
    return bins


def nanmean(values: List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def metric_fields(thresholds: Sequence[float]) -> List[str]:
    return [
        "frame_id",
        "pair",
        "range_bin",
        "z_min",
        "radar_count",
        "lidar_count",
        "nn_mean",
        "nn_median",
        "nn_p90",
        *[f"match_ratio_{threshold_label(float(t))}" for t in thresholds],
        "bev_intersection",
        "bev_union",
        "bev_iou",
    ]


def evaluate_frame(
    frame_id: str,
    clouds: Dict[str, np.ndarray],
    pc_range: Sequence[float],
    range_bins: Sequence[Tuple[str, float, float]],
    z_mins: Sequence[float],
    thresholds: Sequence[float],
    bev_cell_size: float,
) -> List[Dict[str, float]]:
    rows = []
    pairs = [("radar", "lidar"), ("radar", "target"), ("lidar", "target")]
    for range_label, x_min, x_max in range_bins:
        for z_min in z_mins:
            sliced = {
                name: filter_points(points, pc_range, x_min=x_min, x_max=x_max, z_min=z_min)
                for name, points in clouds.items()
            }
            for a_name, b_name in pairs:
                nn = nearest_neighbor_metrics(sliced[a_name], sliced[b_name], thresholds)
                biou = bev_iou(sliced[a_name], sliced[b_name], pc_range=pc_range, cell_size=bev_cell_size)
                rows.append(
                    {
                        "frame_id": frame_id,
                        "pair": f"{a_name}_vs_{b_name}",
                        "range_bin": range_label,
                        "z_min": z_min,
                        **nn,
                        **biou,
                    }
                )
    return rows


def write_summary(rows: List[Dict[str, float]], output_csv: str, fields: Sequence[str]) -> List[Dict[str, float]]:
    groups: Dict[Tuple[str, str, float], Dict[str, List[float]]] = {}
    for row in rows:
        key = (row["pair"], row["range_bin"], row["z_min"])
        groups.setdefault(key, {field: [] for field in fields if field not in ("frame_id", "pair", "range_bin", "z_min")})
        for field in groups[key]:
            groups[key][field].append(row[field])

    summary = []
    for (pair, range_bin, z_min), values in sorted(groups.items()):
        item = {"frame_id": "__summary__", "pair": pair, "range_bin": range_bin, "z_min": z_min}
        item.update({field: nanmean(vals) for field, vals in values.items()})
        summary.append(item)

    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate shared visible radar/LiDAR voxel regions")
    parser.add_argument("--pre_dir", required=True, help="Preprocessed scene root, e.g. Data/NTU4DRadLM_Pre/loop3")
    parser.add_argument("--output_dir", default="", help="Output directory")
    parser.add_argument("--max_files", type=int, default=120)
    parser.add_argument("--occ_threshold", type=float, default=0.1)
    parser.add_argument("--pc_range", type=float, nargs=6, default=DEFAULT_PC_RANGE)
    parser.add_argument("--range_bins", default="0:20,20:40,40:80,80:120")
    parser.add_argument("--z_mins", default="-6,-1,0", help="Comma-separated z lower bounds")
    parser.add_argument("--nn_thresholds", default="0.5,1.0,2.0")
    parser.add_argument("--bev_cell_size", type=float, default=1.0)
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

    output_dir = args.output_dir or os.path.join(args.pre_dir, "shared_visibility")
    os.makedirs(output_dir, exist_ok=True)

    range_bins = parse_range_bins(args.range_bins)
    z_mins = [float(item.strip()) for item in args.z_mins.split(",") if item.strip()]
    thresholds = [float(item.strip()) for item in args.nn_thresholds.split(",") if item.strip()]
    fields = metric_fields(thresholds)

    rows: List[Dict[str, float]] = []
    for frame_id in frame_ids:
        clouds = {
            "radar": voxel_to_points(load_voxel(find_voxel_file(radar_dir, frame_id)), args.pc_range, args.occ_threshold),
            "lidar": voxel_to_points(load_voxel(find_voxel_file(lidar_dir, frame_id)), args.pc_range, args.occ_threshold),
            "target": voxel_to_points(load_voxel(find_voxel_file(target_dir, frame_id)), args.pc_range, args.occ_threshold),
        }
        rows.extend(
            evaluate_frame(
                frame_id,
                clouds,
                args.pc_range,
                range_bins,
                z_mins,
                thresholds,
                args.bev_cell_size,
            )
        )

    frame_csv = os.path.join(output_dir, "frame_metrics.csv")
    with open(frame_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary_csv = os.path.join(output_dir, "summary_metrics.csv")
    summary = write_summary(rows, summary_csv, fields)

    best = [
        row
        for row in summary
        if row["pair"] == "radar_vs_lidar" and row["range_bin"] == range_bins[0][0]
    ]
    best.sort(key=lambda row: (-row.get(f"match_ratio_{threshold_label(thresholds[-1])}", -math.inf), row["nn_mean"]))

    report = os.path.join(output_dir, "shared_visibility_report.md")
    with open(report, "w", encoding="utf-8") as handle:
        handle.write("# Shared Visibility Evaluation\n\n")
        handle.write(f"- pre_dir: `{args.pre_dir}`\n")
        handle.write(f"- frames: `{len(frame_ids)}`\n")
        handle.write(f"- frame metrics: `{frame_csv}`\n")
        handle.write(f"- summary metrics: `{summary_csv}`\n\n")
        if best:
            top = best[0]
            key = f"match_ratio_{threshold_label(thresholds[-1])}"
            handle.write("## Radar vs LiDAR Near-Range Highlight\n\n")
            handle.write(f"- range_bin: `{top['range_bin']}`\n")
            handle.write(f"- z_min: `{top['z_min']}`\n")
            handle.write(f"- nn_mean: `{top['nn_mean']:.4f}` m\n")
            handle.write(f"- {key}: `{top[key]:.4f}`\n")
            handle.write(f"- bev_iou: `{top['bev_iou']:.4f}`\n")

    print(f"Saved frame metrics: {frame_csv}")
    print(f"Saved summary metrics: {summary_csv}")
    print(f"Saved report: {report}")


if __name__ == "__main__":
    main()
