# -*- coding: utf-8 -*-
"""文件功能：定义 observed-domain 3D 占用、距离/高度分层与近地误报指标。"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Dict

import numpy as np


OCCUPANCY_3D_METRIC_PROTOCOL = "observed_3d_stratified_occupancy_metrics_v1"
OCCUPANCY_3D_AGGREGATION = "micro_confusion_counts_v1"


def _finite_range(values: Sequence[Any]) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or len(values) != 6:
        raise ValueError("pc_range 必须包含 6 个数值")
    resolved = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in resolved):
        raise ValueError("pc_range 必须全部为有限数")
    if any(resolved[index + 3] <= resolved[index] for index in range(3)):
        raise ValueError("pc_range 各轴上界必须大于下界")
    return resolved


def _format_edge(value: float) -> str:
    text = f"{float(value):g}".replace("-", "m").replace(".", "p")
    return text


def _clipped_bins(
    lower: float,
    upper: float,
    canonical_edges: Sequence[float],
    axis: str,
) -> list[Dict[str, Any]]:
    edges = sorted(
        {
            float(lower),
            float(upper),
            *(
                float(edge)
                for edge in canonical_edges
                if lower < float(edge) < upper
            ),
        }
    )
    return [
        {
            "label": f"{axis}{_format_edge(left)}_{_format_edge(right)}",
            "min_m": left,
            "max_m": right,
        }
        for left, right in zip(edges, edges[1:])
    ]


def build_occupancy_3d_stratification(
    pc_range: Sequence[Any],
) -> Dict[str, Any]:
    """按当前模型物理范围构造固定、可写入评价收据的分层。"""
    resolved = _finite_range(pc_range)
    x_min, _, z_min, x_max, _, z_max = resolved
    # 正式范围只统计 z>=-1m 的 target 域；legacy 诊断范围完全低于该高度时仍保持可用。
    task_z_min = max(z_min, -1.0) if z_max > -1.0 else z_min
    ground_max = min(z_max, 1.0)
    ground_band = (
        {
            "label": f"z{_format_edge(task_z_min)}_{_format_edge(ground_max)}",
            "min_m": task_z_min,
            "max_m": ground_max,
        }
        if ground_max > task_z_min
        else None
    )
    return {
        "protocol": OCCUPANCY_3D_METRIC_PROTOCOL,
        "coordinate_frame": "lidar",
        "tensor_axes": "ZXY",
        "pc_range": list(resolved),
        "distance_axis": "x",
        "distance_bins": _clipped_bins(
            x_min,
            x_max,
            (0.0, 20.0, 40.0, 80.0),
            "x",
        ),
        "height_axis": "z",
        "height_bins": _clipped_bins(
            task_z_min,
            z_max,
            (-1.0, 0.0, 2.0, 10.0),
            "z",
        ),
        "ground_band": ground_band,
    }


def _confusion_counts(
    prediction: np.ndarray,
    truth: np.ndarray,
    domain: np.ndarray,
) -> Dict[str, int]:
    return {
        "tp": int(np.count_nonzero(prediction & truth & domain)),
        "fp": int(np.count_nonzero(prediction & ~truth & domain)),
        "fn": int(np.count_nonzero(~prediction & truth & domain)),
        "tn": int(np.count_nonzero(~prediction & ~truth & domain)),
    }


def occupancy_3d_confusion_counts(
    prediction: np.ndarray,
    target: np.ndarray,
    observed_mask: np.ndarray,
    *,
    stratification: Mapping[str, Any],
) -> Dict[str, Any]:
    """计算单帧 observed-domain 3D confusion counts，保留可精确累加分母。"""
    pred = np.asarray(prediction, dtype=bool)
    truth = np.asarray(target, dtype=bool)
    observed = np.asarray(observed_mask, dtype=bool)
    if pred.shape != truth.shape or pred.shape != observed.shape or pred.ndim != 3:
        raise ValueError("3D occupancy prediction/target/observed 必须是同形 ZXY")
    if stratification.get("protocol") != OCCUPANCY_3D_METRIC_PROTOCOL:
        raise ValueError("3D occupancy stratification 协议不匹配")
    pc_range = _finite_range(stratification.get("pc_range", ()))
    z_count, x_count, _ = pred.shape
    z_centers = pc_range[2] + (np.arange(z_count) + 0.5) * (
        (pc_range[5] - pc_range[2]) / z_count
    )
    x_centers = pc_range[0] + (np.arange(x_count) + 0.5) * (
        (pc_range[3] - pc_range[0]) / x_count
    )

    def bin_counts(records, centers, axis):
        output = {}
        for record in records:
            mask_1d = (centers >= float(record["min_m"])) & (
                centers < float(record["max_m"])
            )
            region = (
                mask_1d[None, :, None]
                if axis == "x"
                else mask_1d[:, None, None]
            )
            output[str(record["label"])] = _confusion_counts(
                pred,
                truth,
                observed & region,
            )
        return output

    ground_record = stratification.get("ground_band")
    if ground_record is None:
        ground_counts = None
    else:
        ground_z = (z_centers >= float(ground_record["min_m"])) & (
            z_centers < float(ground_record["max_m"])
        )
        ground_counts = _confusion_counts(
            pred,
            truth,
            observed & ground_z[:, None, None],
        )
    return {
        "global": _confusion_counts(pred, truth, observed),
        "distance_bins": bin_counts(
            stratification.get("distance_bins", ()), x_centers, "x"
        ),
        "height_bins": bin_counts(
            stratification.get("height_bins", ()), z_centers, "z"
        ),
        "ground_band": ground_counts,
    }


def empty_occupancy_3d_counts(
    stratification: Mapping[str, Any],
) -> Dict[str, Any]:
    """构造与分层身份一致的零计数累计器。"""
    zero = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    return {
        "global": dict(zero),
        "distance_bins": {
            str(record["label"]): dict(zero)
            for record in stratification.get("distance_bins", ())
        },
        "height_bins": {
            str(record["label"]): dict(zero)
            for record in stratification.get("height_bins", ())
        },
        "ground_band": (
            dict(zero) if stratification.get("ground_band") is not None else None
        ),
    }


def accumulate_occupancy_3d_counts(
    total: Dict[str, Any],
    current: Mapping[str, Any],
) -> None:
    """原位累加同一 stratification 下的 confusion counts。"""
    for group in ("global", "distance_bins", "height_bins"):
        if group == "global":
            targets = [(total[group], current[group])]
        else:
            if set(total[group]) != set(current[group]):
                raise ValueError("3D occupancy 分层标签不一致")
            targets = [
                (total[group][label], current[group][label])
                for label in total[group]
            ]
        for target_counts, current_counts in targets:
            for name in ("tp", "fp", "fn", "tn"):
                target_counts[name] += int(current_counts[name])
    if (total["ground_band"] is None) != (current["ground_band"] is None):
        raise ValueError("3D occupancy ground band 身份不一致")
    if total["ground_band"] is not None:
        for name in ("tp", "fp", "fn", "tn"):
            total["ground_band"][name] += int(current["ground_band"][name])


def _metrics(counts: Mapping[str, Any]) -> Dict[str, Any]:
    tp, fp, fn, tn = (int(counts[name]) for name in ("tp", "fp", "fn", "tn"))
    occupied_denominator = tp + fn
    predicted_denominator = tp + fp
    union = tp + fp + fn
    free_denominator = tn + fp
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "occupied_precision": tp / predicted_denominator if predicted_denominator else None,
        "occupied_recall": tp / occupied_denominator if occupied_denominator else None,
        "occupied_iou": tp / union if union else None,
        "free_recall": tn / free_denominator if free_denominator else None,
        "false_positive_rate": fp / free_denominator if free_denominator else None,
        "observed_voxels": tp + fp + fn + tn,
    }


def finalize_occupancy_3d_metrics(
    counts: Mapping[str, Any],
    *,
    stratification: Mapping[str, Any],
) -> Dict[str, Any]:
    """把累计计数转换为正式 micro 指标并携带物理分层收据。"""
    ground = counts.get("ground_band")
    ground_metrics = _metrics(ground) if ground is not None else None
    if ground_metrics is not None:
        ground_metrics["ground_false_positive_rate"] = ground_metrics[
            "false_positive_rate"
        ]
    return {
        "protocol": OCCUPANCY_3D_METRIC_PROTOCOL,
        "aggregation": OCCUPANCY_3D_AGGREGATION,
        "stratification": dict(stratification),
        "global": _metrics(counts["global"]),
        "distance_bins": {
            label: _metrics(value)
            for label, value in counts["distance_bins"].items()
        },
        "height_bins": {
            label: _metrics(value)
            for label, value in counts["height_bins"].items()
        },
        "ground_band": ground_metrics,
    }


__all__ = [
    "OCCUPANCY_3D_AGGREGATION",
    "OCCUPANCY_3D_METRIC_PROTOCOL",
    "accumulate_occupancy_3d_counts",
    "build_occupancy_3d_stratification",
    "empty_occupancy_3d_counts",
    "finalize_occupancy_3d_metrics",
    "occupancy_3d_confusion_counts",
]
