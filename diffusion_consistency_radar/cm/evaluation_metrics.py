# -*- coding: utf-8 -*-
"""Task-oriented evaluation metrics for airborne occupancy mapping."""

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None


DEFAULT_PC_RANGE = (0.0, -20.0, -6.0, 120.0, 20.0, 10.0)


def _resize_nearest(array: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32).squeeze()
    if arr.shape == tuple(target_shape):
        return arr
    if arr.ndim != len(target_shape):
        raise ValueError(f"Cannot resize uncertainty shape {arr.shape} to {target_shape}")
    indices = [
        np.clip(
            np.round(np.linspace(0, arr.shape[axis] - 1, target_shape[axis])).astype(np.int64),
            0,
            arr.shape[axis] - 1,
        )
        for axis in range(arr.ndim)
    ]
    return arr[np.ix_(*indices)]


def uncertainty_calibration_metrics(
    pred_occ: np.ndarray,
    target_occ: np.ndarray,
    uncertainty: np.ndarray,
    occ_threshold: float = 0.5,
    n_bins: int = 10,
) -> Dict[str, float]:
    """Measure whether predicted variance identifies occupancy errors."""
    pred = np.asarray(pred_occ, dtype=np.float32).squeeze()
    target = np.asarray(target_occ, dtype=np.float32).squeeze()
    if pred.shape != target.shape or pred.ndim != 3:
        raise ValueError(f"Expected matching 3D occupancy arrays, got {pred.shape} and {target.shape}")
    variance = _resize_nearest(uncertainty, pred.shape)
    variance = np.nan_to_num(np.clip(variance, 0.0, 50.0), nan=50.0)
    error_probability = variance / (1.0 + variance)
    binary_error = ((pred > float(occ_threshold)) != (target > float(occ_threshold))).astype(np.float32)

    eps = 1e-6
    brier = float(np.mean((error_probability - binary_error) ** 2))
    p = np.clip(error_probability, eps, 1.0 - eps)
    nll = float(-np.mean(binary_error * np.log(p) + (1.0 - binary_error) * np.log(1.0 - p)))

    ece = 0.0
    edges = np.linspace(0.0, 1.0, max(int(n_bins), 1) + 1)
    for index in range(len(edges) - 1):
        if index == len(edges) - 2:
            mask = (error_probability >= edges[index]) & (error_probability <= edges[index + 1])
        else:
            mask = (error_probability >= edges[index]) & (error_probability < edges[index + 1])
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(float(np.mean(error_probability[mask])) - float(np.mean(binary_error[mask])))

    flat_unc = variance.reshape(-1)
    flat_error = binary_error.reshape(-1)
    if np.std(flat_unc) < eps or np.std(flat_error) < eps:
        correlation = float("nan")
    else:
        correlation = float(np.corrcoef(flat_unc, flat_error)[0, 1])
    return {
        "uncertainty_ece": float(ece),
        "uncertainty_brier": brier,
        "uncertainty_nll": nll,
        "uncertainty_error_corr": correlation,
        "observed_error_rate": float(np.mean(binary_error)),
        "mean_predicted_error_probability": float(np.mean(error_probability)),
    }


def _longest_consecutive_run(indices: np.ndarray) -> int:
    """计算升序索引中的最长连续段长度。"""
    if indices.size == 0:
        return 0
    longest = 1
    current = 1
    for delta in np.diff(indices):
        if int(delta) == 1:
            current += 1
        else:
            longest = max(longest, current)
            current = 1
    return int(max(longest, current))


def vertical_structure_metrics(
    pred_occ: np.ndarray,
    target_occ: np.ndarray,
    pc_range: Sequence[float] = DEFAULT_PC_RANGE,
    occ_threshold: float = 0.5,
    top_height_tolerance_m: float = 0.0,
    trunk_base_max_z: float = 1.0,
    trunk_min_height_m: float = 2.0,
    trunk_height_cap_m: float = 3.0,
) -> Dict[str, float]:
    """评估树木/垂直结构的高度覆盖、顶高、连通性与主干区域召回。"""
    pred = np.asarray(pred_occ, dtype=np.float32).squeeze()
    target = np.asarray(target_occ, dtype=np.float32).squeeze()
    if pred.shape != target.shape or pred.ndim != 3:
        raise ValueError(f"Expected matching 3D occupancy arrays, got {pred.shape} and {target.shape}")

    z_min = float(pc_range[2])
    z_max = float(pc_range[5])
    voxel_height = (z_max - z_min) / max(int(pred.shape[0]), 1)
    if voxel_height <= 0.0:
        raise ValueError(f"Invalid pc_range z extent for vertical metrics: {pc_range}")

    pred_mask = pred > float(occ_threshold)
    target_mask = target > float(occ_threshold)
    top_tolerance_vox = max(0, int(np.floor(float(top_height_tolerance_m) / voxel_height + 1e-6)))
    trunk_cap_vox = max(0, int(np.ceil(float(trunk_height_cap_m) / voxel_height - 1e-6)))

    height_num = 0.0
    height_den = 0.0
    top_num = 0.0
    top_den = 0.0
    conn_num = 0.0
    conn_den = 0.0
    trunk_num = 0.0
    trunk_den = 0.0

    target_columns = np.argwhere(np.any(target_mask, axis=0))
    for x_idx, y_idx in target_columns:
        target_z = np.flatnonzero(target_mask[:, x_idx, y_idx])
        pred_z = np.flatnonzero(pred_mask[:, x_idx, y_idx])
        overlap_mask = pred_mask[:, x_idx, y_idx] & target_mask[:, x_idx, y_idx]
        overlap_z = np.flatnonzero(overlap_mask)

        target_bottom = int(target_z[0])
        target_top = int(target_z[-1])
        target_span = target_top - target_bottom + 1
        if pred_z.size:
            pred_top = int(pred_z[-1])
        else:
            pred_top = None

        # 高度覆盖显式统计目标列中被预测命中的真实占用体素数量，
        # 避免包络接近但中间有空洞、或整体错层时出现虚高分数。
        height_num += float(overlap_z.size)
        height_den += float(target_z.size)

        top_den += 1.0
        # 顶高召回采用“不能高估目标顶部”的定义：
        # 预测顶部必须不高于目标顶部，且只能在容差范围内略低。
        if pred_top is not None and int(pred_top) <= target_top and int(pred_top) >= (target_top - top_tolerance_vox):
            top_num += 1.0

        target_run = _longest_consecutive_run(target_z)
        overlap_run = _longest_consecutive_run(overlap_z)
        # 连通性分子只统计预测与目标交集中的最长连续段，
        # 避免错误高度上的连续预测段获得满分。
        conn_num += float(overlap_run)
        conn_den += float(target_run)

        target_bottom_z = z_min + float(target_bottom) * voxel_height
        target_height_m = float(target_span) * voxel_height
        if target_bottom_z <= float(trunk_base_max_z) and target_height_m >= float(trunk_min_height_m) and trunk_cap_vox > 0:
            trunk_top = min(target_bottom + trunk_cap_vox, pred.shape[0])
            target_trunk_region = target_mask[target_bottom:trunk_top, x_idx, y_idx]
            pred_trunk_region = pred_mask[target_bottom:trunk_top, x_idx, y_idx]
            trunk_den += float(np.count_nonzero(target_trunk_region))
            trunk_num += float(np.count_nonzero(target_trunk_region & pred_trunk_region))

    return {
        "height_coverage_recall": float(height_num / height_den) if height_den else 0.0,
        "height_coverage_numerator": float(height_num),
        "height_coverage_denominator": float(height_den),
        "top_height_recall": float(top_num / top_den) if top_den else 0.0,
        "top_height_numerator": float(top_num),
        "top_height_denominator": float(top_den),
        "vertical_connectivity_recall": float(conn_num / conn_den) if conn_den else 0.0,
        "vertical_connectivity_numerator": float(conn_num),
        "vertical_connectivity_denominator": float(conn_den),
        "trunk_region_recall": float(trunk_num / trunk_den) if trunk_den else 0.0,
        "trunk_region_numerator": float(trunk_num),
        "trunk_region_denominator": float(trunk_den),
    }


def threshold_label(value: float) -> str:
    text = f"{float(value):g}".replace("-", "m").replace(".", "p")
    return text


def voxel_to_points(
    voxel: np.ndarray,
    pc_range: Sequence[float] = DEFAULT_PC_RANGE,
    occ_threshold: float = 0.1,
) -> np.ndarray:
    arr = np.asarray(voxel, dtype=np.float32)
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D voxel, got {arr.shape}")
    if arr.shape[-1] <= 8:
        occ = arr[..., 0]
        layout = "xyzc"
    elif arr.shape[0] <= 8:
        occ = arr[0]
        layout = "czxy"
    else:
        occ = arr[..., 0]
        layout = "xyzc"

    idx = np.argwhere(occ > float(occ_threshold))
    if idx.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    x_min, y_min, z_min, x_max, y_max, z_max = [float(v) for v in pc_range]
    if layout == "xyzc":
        nx, ny, nz = occ.shape
        x = x_min + (idx[:, 0].astype(np.float32) + 0.5) * ((x_max - x_min) / max(nx, 1))
        y = y_min + (idx[:, 1].astype(np.float32) + 0.5) * ((y_max - y_min) / max(ny, 1))
        z = z_min + (idx[:, 2].astype(np.float32) + 0.5) * ((z_max - z_min) / max(nz, 1))
    else:
        nz, nx, ny = occ.shape
        z = z_min + (idx[:, 0].astype(np.float32) + 0.5) * ((z_max - z_min) / max(nz, 1))
        x = x_min + (idx[:, 1].astype(np.float32) + 0.5) * ((x_max - x_min) / max(nx, 1))
        y = y_min + (idx[:, 2].astype(np.float32) + 0.5) * ((y_max - y_min) / max(ny, 1))
    return np.stack([x, y, z], axis=1).astype(np.float32)


def filter_points_by_band(
    points: np.ndarray,
    pc_range: Sequence[float] = DEFAULT_PC_RANGE,
    x_min: float = None,
    x_max: float = None,
    z_min: float = None,
    z_max: float = None,
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] == 0:
        return pts.reshape(0, pts.shape[-1] if pts.ndim == 2 else 3)
    lo_x = float(pc_range[0] if x_min is None else x_min)
    hi_x = float(pc_range[3] if x_max is None else x_max)
    lo_z = float(pc_range[2] if z_min is None else z_min)
    hi_z = float(pc_range[5] if z_max is None else z_max)
    keep = (
        (pts[:, 0] >= lo_x)
        & (pts[:, 0] < hi_x)
        & (pts[:, 1] >= float(pc_range[1]))
        & (pts[:, 1] < float(pc_range[4]))
        & (pts[:, 2] >= lo_z)
        & (pts[:, 2] < hi_z)
    )
    return pts[keep]


def _bev_cells(points: np.ndarray, pc_range: Sequence[float], cell_size: float) -> set:
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] == 0:
        return set()
    keep = (
        (pts[:, 0] >= float(pc_range[0]))
        & (pts[:, 0] < float(pc_range[3]))
        & (pts[:, 1] >= float(pc_range[1]))
        & (pts[:, 1] < float(pc_range[4]))
    )
    xy = pts[keep, :2]
    if xy.shape[0] == 0:
        return set()
    ix = np.floor((xy[:, 0] - float(pc_range[0])) / float(cell_size)).astype(np.int32)
    iy = np.floor((xy[:, 1] - float(pc_range[1])) / float(cell_size)).astype(np.int32)
    return set(zip(ix.tolist(), iy.tolist()))


def bev_iou(
    pred_points: np.ndarray,
    target_points: np.ndarray,
    pc_range: Sequence[float] = DEFAULT_PC_RANGE,
    cell_size: float = 0.5,
) -> Dict[str, float]:
    pred = _bev_cells(pred_points, pc_range, cell_size)
    target = _bev_cells(target_points, pc_range, cell_size)
    inter = len(pred & target)
    union = len(pred | target)
    return {
        "bev_iou": float(inter / union) if union else float("nan"),
        "bev_intersection": float(inter),
        "bev_union": float(union),
    }


def occupancy_prf(
    pred_points: np.ndarray,
    target_points: np.ndarray,
    pc_range: Sequence[float] = DEFAULT_PC_RANGE,
    cell_size: float = 0.5,
) -> Dict[str, float]:
    pred = _bev_cells(pred_points, pc_range, cell_size)
    target = _bev_cells(target_points, pc_range, cell_size)
    tp = len(pred & target)
    fp = len(pred - target)
    fn = len(target - pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1), "tp": float(tp), "fp": float(fp), "fn": float(fn)}


def nearest_neighbor_metrics(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    thresholds: Iterable[float] = (0.5, 1.0, 2.0),
) -> Dict[str, float]:
    src = np.asarray(src_points, dtype=np.float32)
    dst = np.asarray(dst_points, dtype=np.float32)
    out = {"nn_mean": float("nan"), "nn_median": float("nan"), "nn_p90": float("nan")}
    for threshold in thresholds:
        out[f"match_ratio_{threshold_label(float(threshold))}"] = float("nan")
    if cKDTree is None or src.shape[0] == 0 or dst.shape[0] == 0:
        return out
    dists, _ = cKDTree(dst[:, :3]).query(src[:, :3], k=1)
    out["nn_mean"] = float(np.mean(dists))
    out["nn_median"] = float(np.median(dists))
    out["nn_p90"] = float(np.percentile(dists, 90))
    for threshold in thresholds:
        out[f"match_ratio_{threshold_label(float(threshold))}"] = float(np.mean(dists <= float(threshold)))
    return out


def parse_range_bins(text: str) -> List[Tuple[str, float, float]]:
    bins: List[Tuple[str, float, float]] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        lo, hi = [float(v) for v in chunk.split("-")]
        bins.append((f"x{threshold_label(lo)}_{threshold_label(hi)}", lo, hi))
    return bins
