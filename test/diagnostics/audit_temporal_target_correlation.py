#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""文件功能：只读审计稀疏 target 在不同时距下的体素重合，用于选择 purge。"""

import argparse
import csv
import json
import math
import os

import numpy as np


def _load_timestamps(path):
    timestamps = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {
            "frame_index",
            "radar_timestamp",
        }.issubset(reader.fieldnames):
            raise ValueError("sync CSV 缺少 frame_index/radar_timestamp")
        for expected_index, row in enumerate(reader):
            if int(row["frame_index"]) != expected_index:
                raise ValueError("sync frame_index 必须从 0 严格连续")
            timestamp = float(row["radar_timestamp"])
            if not math.isfinite(timestamp):
                raise ValueError("Radar timestamp 必须有限")
            timestamps.append(timestamp)
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("Radar timestamp 必须严格递增")
    return np.asarray(timestamps, dtype=np.float64)


def _load_occupied_linear_ids(path):
    with np.load(path, allow_pickle=False) as data:
        coords = np.asarray(data["coords"], dtype=np.int64)
        shape = tuple(int(value) for value in np.asarray(data["shape"])[:3])
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"target coords 必须是 (N,3): {path}")
    if coords.size == 0:
        return np.empty((0,), dtype=np.int64)
    return np.unique(np.ravel_multi_index(coords.T, shape))


def _pair_metrics(left, right):
    intersection = np.intersect1d(left, right, assume_unique=True).size
    union = left.size + right.size - intersection
    return {
        "jaccard": float(intersection / union) if union else 1.0,
        "overlap_min": (
            float(intersection / min(left.size, right.size))
            if min(left.size, right.size) > 0
            else 1.0 if max(left.size, right.size) == 0 else 0.0
        ),
    }


def _summary(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="审计 target occupancy 随时间间隔的同网格重合率"
    )
    parser.add_argument("--target_dir", required=True)
    parser.add_argument("--sync_csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--lags",
        nargs="+",
        type=int,
        default=(1, 2, 5, 10, 20, 30, 50, 100),
    )
    parser.add_argument("--max_pairs_per_lag", type=int, default=256)
    args = parser.parse_args()

    timestamps = _load_timestamps(args.sync_csv)
    frame_paths = [
        os.path.join(args.target_dir, f"{index:06d}.npz")
        for index in range(len(timestamps))
    ]
    missing = [path for path in frame_paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"target frame 缺失，首项: {missing[0]}")
    if args.max_pairs_per_lag <= 0:
        raise ValueError("max_pairs_per_lag 必须为正")

    cache = {}

    def load(index):
        if index not in cache:
            cache[index] = _load_occupied_linear_ids(frame_paths[index])
        return cache[index]

    lag_results = []
    for lag in args.lags:
        if lag <= 0 or lag >= len(frame_paths):
            raise ValueError(f"lag 必须位于 [1,{len(frame_paths) - 1}]")
        available = len(frame_paths) - lag
        pair_count = min(available, args.max_pairs_per_lag)
        indices = np.linspace(0, available - 1, pair_count, dtype=np.int64)
        delta_values = []
        jaccard_values = []
        overlap_values = []
        for index in indices:
            metrics = _pair_metrics(load(int(index)), load(int(index + lag)))
            delta_values.append(float(timestamps[index + lag] - timestamps[index]))
            jaccard_values.append(metrics["jaccard"])
            overlap_values.append(metrics["overlap_min"])
        lag_results.append(
            {
                "lag_frames": int(lag),
                "pair_count": int(pair_count),
                "delta_seconds": _summary(delta_values),
                "jaccard": _summary(jaccard_values),
                "overlap_min": _summary(overlap_values),
            }
        )

    report = {
        "protocol": "temporal_target_correlation_audit_v1",
        "metric_scope": "same_grid_target_occupied_voxels_no_pose_warp",
        "frame_count": len(frame_paths),
        "lags": lag_results,
        "interpretation_limit": (
            "同网格重合同时受平台运动影响；该结果用于设置防相邻帧泄漏的保守 "
            "purge 下界，不代表世界坐标场景独立性"
        ),
    }
    output = os.path.abspath(args.output)
    if os.path.lexists(output):
        raise FileExistsError(f"输出已存在，拒绝覆盖: {output}")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"output": output, "frame_count": len(frame_paths)}))


if __name__ == "__main__":
    main()
