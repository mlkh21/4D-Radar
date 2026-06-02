#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对原始帧上的毫米波雷达->激光雷达配对进行快速对齐健全性检查。"""

import argparse
import csv
import os
from typing import Dict, List, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None


def load_calib(calib_file: str) -> Tuple[np.ndarray, np.ndarray]:
    r_mat = np.eye(3, dtype=np.float32)
    t_vec = np.zeros(3, dtype=np.float32)
    with open(calib_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(":")
            if len(parts) < 2:
                continue
            key = parts[0].strip()
            vals = []
            for token in parts[1].strip().split():
                try:
                    vals.append(float(token))
                except ValueError:
                    continue
            if key == "R" and len(vals) == 9:
                r_mat = np.asarray(vals, dtype=np.float32).reshape(3, 3)
            if key == "T" and len(vals) >= 3:
                t_vec = np.asarray(vals[:3], dtype=np.float32)
    return r_mat, t_vec


def read_indices(path: str) -> List[int]:
    with open(path, "r", encoding="utf-8") as f:
        return [int(x.strip()) for x in f if x.strip()]


def centroid_xyz(p: np.ndarray) -> np.ndarray:
    if p.shape[0] == 0:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    return np.mean(p[:, :3], axis=0)


def chamfer(a: np.ndarray, b: np.ndarray) -> float:
    if cKDTree is None or a.shape[0] == 0 or b.shape[0] == 0:
        return float("nan")
    ta = cKDTree(a[:, :3])
    tb = cKDTree(b[:, :3])
    dab, _ = tb.query(a[:, :3], k=1)
    dba, _ = ta.query(b[:, :3], k=1)
    return float(np.mean(dab) + np.mean(dba))


def apply_transform(points: np.ndarray, r_mat: np.ndarray, t_vec: np.ndarray) -> np.ndarray:
    out = points.copy()
    xyz = points[:, :3]
    out[:, :3] = np.dot(xyz, r_mat.T) + t_vec
    return out


def candidate_bank(base_r: np.ndarray, base_t: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    # Base: radar -> lidar (as provided by calib file)
    candidates: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    flips = {
        "none": np.diag([1.0, 1.0, 1.0]).astype(np.float32),
        "flip_y": np.diag([1.0, -1.0, 1.0]).astype(np.float32),
        "flip_z": np.diag([1.0, 1.0, -1.0]).astype(np.float32),
        "flip_yz": np.diag([1.0, -1.0, -1.0]).astype(np.float32),
    }

    inv_r = base_r.T
    inv_t = -np.dot(inv_r, base_t)

    for fname, fmat in flips.items():
        candidates[f"forward_{fname}"] = (np.dot(fmat, base_r), np.dot(fmat, base_t))
        candidates[f"inverse_{fname}"] = (np.dot(fmat, inv_r), np.dot(fmat, inv_t))
    return candidates


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Alignment sanity check for radar/lidar raw pairs")
    p.add_argument("--raw_root", type=str, default="./Data/NTU4DRadLM_Raw")
    p.add_argument("--scene", type=str, required=True)
    p.add_argument("--calib", type=str, default="./Data/config/calib_radar_to_livox.txt")
    p.add_argument("--max_frames", type=int, default=200)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--z_shift_grid", type=str, default="-2.0,-1.0,0.0,1.0,2.0")
    p.add_argument("--output_dir", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    scene_root = os.path.join(args.raw_root, args.scene)
    radar_dir = os.path.join(scene_root, "radar_pcl")
    lidar_dir = os.path.join(scene_root, "livox_lidar")
    radar_idx_file = os.path.join(scene_root, "radar_index_sequence.txt")
    lidar_idx_file = os.path.join(scene_root, "lidar_index_sequence.txt")

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join("./Result/alignment_check", args.scene)
    os.makedirs(output_dir, exist_ok=True)

    for pth in (radar_dir, lidar_dir, radar_idx_file, lidar_idx_file, args.calib):
        if not os.path.exists(pth):
            raise FileNotFoundError(f"Missing required path: {pth}")

    r_mat, t_vec = load_calib(args.calib)
    candidates = candidate_bank(r_mat, t_vec)

    radar_files = sorted([f for f in os.listdir(radar_dir) if f.endswith(".npy")])
    lidar_files = sorted([f for f in os.listdir(lidar_dir) if f.endswith(".npy")])
    radar_indices = read_indices(radar_idx_file)
    lidar_indices = read_indices(lidar_idx_file)
    n = min(len(radar_indices), len(lidar_indices))

    frame_ids = list(range(0, n, max(1, args.stride)))[: max(1, args.max_frames)]
    z_shifts = [float(x.strip()) for x in args.z_shift_grid.split(",") if x.strip()]

    rows = []
    for name, (r_use, t_use) in candidates.items():
        dx_vals, dy_vals, dz_vals, chamfer_vals = [], [], [], []
        for i in frame_ids:
            ri = radar_indices[i]
            li = lidar_indices[i]
            if ri >= len(radar_files) or li >= len(lidar_files):
                continue
            radar = np.load(os.path.join(radar_dir, radar_files[ri])).astype(np.float32)
            lidar = np.load(os.path.join(lidar_dir, lidar_files[li])).astype(np.float32)
            if radar.shape[0] == 0 or lidar.shape[0] == 0:
                continue

            radar_t = apply_transform(radar, r_use, t_use)
            c_r = centroid_xyz(radar_t)
            c_l = centroid_xyz(lidar)
            d = c_r - c_l
            dx_vals.append(float(d[0]))
            dy_vals.append(float(d[1]))
            dz_vals.append(float(d[2]))
            ch = chamfer(radar_t, lidar)
            if np.isfinite(ch):
                chamfer_vals.append(ch)

        if len(dx_vals) == 0:
            continue
        rows.append(
            {
                "candidate": name,
                "frames": len(dx_vals),
                "mean_dx": float(np.mean(dx_vals)),
                "mean_dy": float(np.mean(dy_vals)),
                "mean_dz": float(np.mean(dz_vals)),
                "mean_abs_dy": float(np.mean(np.abs(dy_vals))),
                "mean_abs_dz": float(np.mean(np.abs(dz_vals))),
                "mean_chamfer": float(np.mean(chamfer_vals)) if chamfer_vals else float("nan"),
            }
        )

    rows.sort(key=lambda x: (x["mean_abs_dy"] + x["mean_abs_dz"], x["mean_chamfer"]))
    csv_path = os.path.join(output_dir, "alignment_candidates.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "candidate",
                "frames",
                "mean_dx",
                "mean_dy",
                "mean_dz",
                "mean_abs_dy",
                "mean_abs_dz",
                "mean_chamfer",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Estimate z offset for best candidate.
    best = rows[0] if rows else None
    z_rows = []
    if best is not None:
        r_use, t_use = candidates[best["candidate"]]
        for z_shift in z_shifts:
            dy_vals, dz_vals = [], []
            for i in frame_ids:
                ri = radar_indices[i]
                li = lidar_indices[i]
                if ri >= len(radar_files) or li >= len(lidar_files):
                    continue
                radar = np.load(os.path.join(radar_dir, radar_files[ri])).astype(np.float32)
                lidar = np.load(os.path.join(lidar_dir, lidar_files[li])).astype(np.float32)
                if radar.shape[0] == 0 or lidar.shape[0] == 0:
                    continue
                shifted_t = t_use.copy()
                shifted_t[2] += z_shift
                radar_t = apply_transform(radar, r_use, shifted_t)
                d = centroid_xyz(radar_t) - centroid_xyz(lidar)
                dy_vals.append(float(d[1]))
                dz_vals.append(float(d[2]))
            if dy_vals:
                z_rows.append(
                    {
                        "candidate": best["candidate"],
                        "z_shift": z_shift,
                        "mean_abs_dy": float(np.mean(np.abs(dy_vals))),
                        "mean_abs_dz": float(np.mean(np.abs(dz_vals))),
                    }
                )
        z_rows.sort(key=lambda x: (x["mean_abs_dy"] + x["mean_abs_dz"]))
        with open(os.path.join(output_dir, "alignment_z_shift_sweep.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["candidate", "z_shift", "mean_abs_dy", "mean_abs_dz"])
            writer.writeheader()
            writer.writerows(z_rows)

    report = os.path.join(output_dir, "alignment_report.md")
    with open(report, "w", encoding="utf-8") as f:
        f.write("# Alignment Sanity Check\n\n")
        f.write(f"- scene: `{args.scene}`\n")
        f.write(f"- frames sampled: `{len(frame_ids)}`\n")
        f.write(f"- calib: `{args.calib}`\n")
        f.write(f"- candidates csv: `{csv_path}`\n\n")
        if rows:
            top = rows[0]
            f.write("## Best Candidate\n\n")
            f.write(f"- candidate: `{top['candidate']}`\n")
            f.write(f"- mean_dy: `{top['mean_dy']:.4f}` m\n")
            f.write(f"- mean_dz: `{top['mean_dz']:.4f}` m\n")
            f.write(f"- mean_abs_dy: `{top['mean_abs_dy']:.4f}` m\n")
            f.write(f"- mean_abs_dz: `{top['mean_abs_dz']:.4f}` m\n")
            f.write(f"- mean_chamfer: `{top['mean_chamfer']:.4f}`\n")
        else:
            f.write("No valid frames were evaluated.\n")

    print(f"Saved candidate ranking: {csv_path}")
    if z_rows:
        print(f"Saved z-shift sweep: {os.path.join(output_dir, 'alignment_z_shift_sweep.csv')}")
    print(f"Saved report: {report}")


if __name__ == "__main__":
    main()

