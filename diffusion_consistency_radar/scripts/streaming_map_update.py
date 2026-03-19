#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Streaming map update entrypoint (v1)."""

import argparse
import csv
import os
import sys
import time
from typing import List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cm.probabilistic_mapping import (  # noqa: E402
    GridMapConfig,
    LazyLocalMapQuery,
    SlidingProbabilisticGridMap,
    load_sparse_voxel_npz,
)


def list_voxel_files(folder: str) -> List[str]:
    files = [f for f in os.listdir(folder) if f.endswith(".npy") or f.endswith(".npz")]
    files.sort()
    return files


def load_voxel(path: str) -> np.ndarray:
    if path.endswith(".npz"):
        arr = load_sparse_voxel_npz(path)
    else:
        arr = np.load(path).astype(np.float32)

    # Some inference outputs are batched: (N, C, Z, X, Y).
    # For streaming smoke tests we use the first sample.
    if arr.ndim == 5:
        arr = arr[0]

    return to_xyzc(arr)


def to_xyzc(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D voxel, got shape={arr.shape}")
    if arr.shape[-1] <= 8 and arr.shape[0] > 8 and arr.shape[1] > 8:
        return arr.astype(np.float32)
    if arr.shape[0] <= 8:
        return np.transpose(arr, (2, 3, 1, 0)).astype(np.float32)
    raise ValueError(f"Unsupported voxel layout: {arr.shape}")


def load_ir_bev(bev_path: str, target_shape_xy) -> np.ndarray:
    bev = np.load(bev_path).astype(np.float32)
    if bev.ndim == 3:
        bev = bev[..., 0]
    if bev.shape != target_shape_xy:
        raise ValueError(f"Infrared BEV shape mismatch. expected={target_shape_xy}, got={bev.shape}")
    if bev.max() > 1.0:
        bev = bev / 255.0
    return np.clip(bev, 0.0, 1.0)


def build_config(args, first_voxel_xyzc: np.ndarray) -> GridMapConfig:
    nx, ny, nz = first_voxel_xyzc.shape[:3]
    x_min, y_min, z_min, x_max, y_max, z_max = args.pc_range
    x_res = (x_max - x_min) / max(nx, 1)
    y_res = (y_max - y_min) / max(ny, 1)
    z_res = (z_max - z_min) / max(nz, 1)
    return GridMapConfig(
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
        x_resolution=float(x_res),
        y_resolution=float(y_res),
        z_min=z_min,
        z_max=z_max,
        z_resolution=float(z_res),
        window_size=args.window_size,
        decay_rate=args.decay_rate,
        prior_reliability=args.prior_reliability,
        radar_reliability=args.radar_reliability,
        infrared_reliability=args.infrared_reliability,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming probabilistic map update")
    parser.add_argument("--radar_voxel_dir", type=str, required=True)
    parser.add_argument("--infrared_bev_dir", type=str, default="")
    parser.add_argument("--prior_dem", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="./streaming_results")
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--window_size", type=int, default=12)
    parser.add_argument("--decay_rate", type=float, default=0.12)
    parser.add_argument("--prior_reliability", type=float, default=0.90)
    parser.add_argument("--radar_reliability", type=float, default=0.75)
    parser.add_argument("--infrared_reliability", type=float, default=0.65)
    parser.add_argument("--frame_limit", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument(
        "--pc_range",
        type=float,
        nargs=6,
        default=[0, -20, -6, 120, 20, 10],
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    radar_files = list_voxel_files(args.radar_voxel_dir)
    if not radar_files:
        raise RuntimeError(f"No voxel files found under {args.radar_voxel_dir}")
    if args.frame_limit > 0:
        radar_files = radar_files[: args.frame_limit]

    first_voxel = load_voxel(os.path.join(args.radar_voxel_dir, radar_files[0]))
    cfg = build_config(args, first_voxel)
    grid_map = SlidingProbabilisticGridMap(cfg)
    query = LazyLocalMapQuery(cfg)

    prior_dem = None
    if args.prior_dem:
        prior_dem = np.load(args.prior_dem).astype(np.float32)

    metric_path = os.path.join(args.output_dir, "streaming_metrics.csv")
    with open(metric_path, "w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["frame", "timestamp", "update_ms", "nearest_dist", "nearest_uncertainty", "is_risky"])

        for i, file_name in enumerate(radar_files):
            frame_start = time.perf_counter()
            timestamp = i * args.dt

            voxel = load_voxel(os.path.join(args.radar_voxel_dir, file_name))
            grid_map.update_from_voxel(voxel_xyzc=voxel, timestamp=timestamp, sensor="radar")

            if args.infrared_bev_dir:
                ir_path = os.path.join(args.infrared_bev_dir, file_name.replace("_voxel", "_bev"))
                if os.path.exists(ir_path):
                    bev = load_ir_bev(ir_path, target_shape_xy=grid_map.occ_prob.shape)
                    grid_map.update_from_ir_bev(bev_xy=bev, timestamp=timestamp)

            if prior_dem is not None and prior_dem.shape == grid_map.dem_mean.shape:
                grid_map.fuse_with_prior_dem(prior_dem=prior_dem, prior_confidence=0.6)

            snapshot = grid_map.snapshot()
            query.refresh(snapshot)
            prox = query.query_proximity(x_m=25.0, y_m=0.0, search_radius=30.0)

            frame_ms = (time.perf_counter() - frame_start) * 1000.0
            writer.writerow([
                i,
                f"{timestamp:.3f}",
                f"{frame_ms:.3f}",
                f"{prox['distance']:.3f}",
                f"{prox['uncertainty']:.3f}",
                int(prox["is_risky"] > 0.5),
            ])

            if i % max(1, args.save_every) == 0:
                np.savez_compressed(
                    os.path.join(args.output_dir, f"map_snapshot_{i:06d}.npz"),
                    occ_prob=snapshot["occ_prob"],
                    belief=snapshot["belief"],
                    plausibility=snapshot["plausibility"],
                    dem_mean=snapshot["dem_mean"],
                    dem_var=snapshot["dem_var"],
                )
                print(f"[frame {i}] update={frame_ms:.2f}ms dist={prox['distance']:.2f}m unc={prox['uncertainty']:.2f}")

    final_snapshot = grid_map.snapshot()
    np.savez_compressed(
        os.path.join(args.output_dir, "map_final.npz"),
        occ_prob=final_snapshot["occ_prob"],
        belief=final_snapshot["belief"],
        plausibility=final_snapshot["plausibility"],
        dem_mean=final_snapshot["dem_mean"],
        dem_var=final_snapshot["dem_var"],
    )
    print(f"Saved final map to: {os.path.join(args.output_dir, 'map_final.npz')}")
    print(f"Saved metrics to: {metric_path}")


if __name__ == "__main__":
    main()
