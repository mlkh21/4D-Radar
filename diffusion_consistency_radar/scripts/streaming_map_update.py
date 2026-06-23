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
from cm.evaluation_metrics import occupancy_prf, voxel_to_points


def list_voxel_files(folder: str) -> List[str]:
    files = [
        f for f in os.listdir(folder)
        if f.endswith(".npz")
        or (
            f.endswith(".npy")
            and not f.endswith("_pcl.npy")
            and not f.endswith("_uncertainty.npy")
            and (f.endswith("_voxel.npy") or not f.endswith("_bev.npy"))
        )
    ]
    files.sort()
    return files


def load_voxel(path: str) -> np.ndarray:
    if path.endswith(".npz"):
        arr = load_sparse_voxel_npz(path)
    else:
        arr = np.load(path).astype(np.float32)

    # NOTE: 部分推理输出带批次维度：(N, C, Z, X, Y)。
    # NOTE: 流式冒烟测试仅取批次中的第一个样本。
    # TODO: 支持批量样本并行更新与异步队列，面向Orin级实时部署。
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
    # TODO: 增加红外质量门控(噪声/模糊/失焦检测)并输出融合置信度。
    if bev.ndim == 3:
        bev = bev[..., 0]
    if bev.shape != target_shape_xy:
        raise ValueError(f"Infrared BEV shape mismatch. expected={target_shape_xy}, got={bev.shape}")
    if bev.max() > 1.0:
        bev = bev / 255.0
    return np.clip(bev, 0.0, 1.0)


def find_uncertainty_file(uncertainty_dir: str, voxel_file_name: str) -> str:
    if not uncertainty_dir:
        return ""
    stem = os.path.splitext(voxel_file_name)[0]
    base = stem[:-6] if stem.endswith("_voxel") else stem
    candidates = [
        os.path.join(uncertainty_dir, f"{base}_uncertainty.npy"),
        os.path.join(uncertainty_dir, f"{stem}_uncertainty.npy"),
        os.path.join(uncertainty_dir, f"{base}_voxel_uncertainty.npy"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return ""


def load_model_uncertainty(path: str) -> np.ndarray:
    if not path:
        return None
    arr = np.load(path).astype(np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 4 and arr.shape[0] <= 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
        arr = np.transpose(arr, (1, 2, 0))
    return arr


def map_occ_to_points(occ_prob: np.ndarray, cfg: GridMapConfig, threshold: float = 0.55) -> np.ndarray:
    idx = np.argwhere(occ_prob >= float(threshold))
    if idx.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    x = cfg.x_min + (idx[:, 0].astype(np.float32) + 0.5) * cfg.x_resolution
    y = cfg.y_min + (idx[:, 1].astype(np.float32) + 0.5) * cfg.y_resolution
    z = np.zeros_like(x)
    return np.stack([x, y, z], axis=1).astype(np.float32)


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
        speed_m_s=args.speed_m_s,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming probabilistic map update")
    parser.add_argument("--radar_voxel_dir", type=str, required=True)
    parser.add_argument("--uncertainty_dir", type=str, default="",
                        help="Directory containing *_uncertainty.npy files from multimodal inference")
    parser.add_argument("--infrared_bev_dir", type=str, default="")
    parser.add_argument("--prior_dem", type=str, default="")
    parser.add_argument("--target_voxel_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="./streaming_results")
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--window_size", type=int, default=12)
    parser.add_argument("--decay_rate", type=float, default=0.12)
    parser.add_argument("--prior_reliability", type=float, default=0.90)
    parser.add_argument("--radar_reliability", type=float, default=0.75)
    parser.add_argument("--infrared_reliability", type=float, default=0.65)
    parser.add_argument("--speed_m_s", type=float, default=50.0)
    parser.add_argument("--odom_cov_trace", type=float, default=0.0)
    parser.add_argument("--calib_confidence", type=float, default=1.0)
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
    odom_cov = None
    if args.odom_cov_trace > 0.0:
        odom_cov = np.eye(3, dtype=np.float32) * (float(args.odom_cov_trace) / 3.0)

    prior_dem = None
    if args.prior_dem:
        # TODO: 支持先验DEM多来源输入及置信度地图，而非单一栅格文件。
        prior_dem = np.load(args.prior_dem).astype(np.float32)

    metric_path = os.path.join(args.output_dir, "streaming_metrics.csv")
    with open(metric_path, "w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow([
            "frame",
            "timestamp",
            "update_ms",
            "nearest_dist",
            "nearest_uncertainty",
            "is_risky",
            "speed_m_s",
            "odom_cov_trace",
            "obstacle_precision",
            "obstacle_recall",
            "false_positive_rate",
            "mean_uncertainty",
        ])

        for i, file_name in enumerate(radar_files):
            frame_start = time.perf_counter()
            timestamp = i * args.dt

            voxel = load_voxel(os.path.join(args.radar_voxel_dir, file_name))
            unc_path = find_uncertainty_file(args.uncertainty_dir, file_name)
            model_uncertainty = load_model_uncertainty(unc_path) if unc_path else None
            grid_map.update_from_voxel(
                voxel_xyzc=voxel,
                timestamp=timestamp,
                sensor="radar",
                odom_cov=odom_cov,
                model_uncertainty=model_uncertainty,
                calib_confidence=args.calib_confidence,
            )

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
            obstacle_precision = ""
            obstacle_recall = ""
            false_positive_rate = ""
            if args.target_voxel_dir:
                stem = file_name.replace("_voxel", "")
                stem = os.path.splitext(stem)[0]
                target_path = ""
                for ext in (".npz", ".npy"):
                    candidate = os.path.join(args.target_voxel_dir, f"{stem}{ext}")
                    if os.path.exists(candidate):
                        target_path = candidate
                        break
                if target_path:
                    target_points = voxel_to_points(load_voxel(target_path), pc_range=args.pc_range, occ_threshold=0.1)
                    map_points = map_occ_to_points(snapshot["occ_prob"], cfg, threshold=0.55)
                    prf = occupancy_prf(map_points, target_points, pc_range=args.pc_range, cell_size=max(cfg.x_resolution, cfg.y_resolution))
                    obstacle_precision = f"{prf['precision']:.6f}"
                    obstacle_recall = f"{prf['recall']:.6f}"
                    denom = prf["fp"] + prf["tp"]
                    false_positive_rate = f"{(prf['fp'] / denom if denom else 0.0):.6f}"

            frame_ms = (time.perf_counter() - frame_start) * 1000.0
            # TODO: 增加端到端时延分解统计(读取/融合/查询/写盘)与资源监控(CPU/GPU/内存)。
            writer.writerow([
                i,
                f"{timestamp:.3f}",
                f"{frame_ms:.3f}",
                f"{prox['distance']:.3f}",
                f"{prox['uncertainty']:.3f}",
                int(prox["is_risky"] > 0.5),
                f"{args.speed_m_s:.3f}",
                f"{args.odom_cov_trace:.6f}",
                obstacle_precision,
                obstacle_recall,
                false_positive_rate,
                f"{float(np.mean(1.0 - snapshot['belief'])):.6f}",
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
    # TODO: 新增ROS1发布模式，将map_final/streaming指标同步为service/action可消费接口。


if __name__ == "__main__":
    main()
