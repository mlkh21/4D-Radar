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
