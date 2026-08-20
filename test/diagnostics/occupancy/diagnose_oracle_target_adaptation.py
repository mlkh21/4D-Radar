#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""离线复现逐帧 target 数量匹配的 oracle 阈值、点云与审计报告。"""

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from diffusion_consistency_radar.diagnostics.occupancy_helpers import (
    load_target_occ_resized,
    voxel_to_pointcloud,
)


DEFAULT_SOURCE_PC_RANGE = (0.0, -20.0, -6.0, 120.0, 20.0, 10.0)
DEFAULT_MODEL_PC_RANGE = (0.0, -20.0, -6.0, 40.0, 20.0, 10.0)
DEFAULT_TARGET_SIZE = (32, 128, 128)
CSV_FIELDS = (
    "index",
    "frame_id",
    "pred_voxel_file",
    "target_voxel_file",
    "target_occ_count",
    "effective_match_count",
    "oracle_occ_threshold",
    "oracle_pred_point_count",
    "pred_to_target_count_ratio",
    "oracle_pointcloud_file",
)


def find_oracle_occ_threshold(
    pred_occ: np.ndarray,
    target_count: int,
) -> Tuple[float, int]:
    """沿用历史 top-k 规则计算 oracle 阈值，并返回实际匹配请求数。"""
    flat = np.asarray(pred_occ).reshape(-1)
    if not np.issubdtype(flat.dtype, np.floating):
        flat = flat.astype(np.float64)
    if flat.size == 0:
        raise ValueError("prediction occupancy 不能为空")
    if not np.all(np.isfinite(flat)):
        raise ValueError("prediction occupancy 必须全部为有限数")

    effective_count = int(min(max(int(target_count), 1), flat.size))
    negative_infinity = np.asarray(-np.inf, dtype=flat.dtype)
    if effective_count >= flat.size:
        threshold = float(np.nextafter(flat.min(), negative_infinity))
        return threshold, effective_count

    topk_indices = np.argpartition(flat, -effective_count)[-effective_count:]
    kth_value = flat[topk_indices].min()
    # 后续使用严格大于号，因此将阈值移动到第 k 大值的前一个浮点数。
    threshold = float(np.nextafter(kth_value, negative_infinity))
    return threshold, effective_count


def _validate_pc_range(values: Sequence[float], name: str) -> Tuple[float, ...]:
    if len(values) != 6:
        raise ValueError(f"{name} 必须包含 6 个数")
    resolved = tuple(float(value) for value in values)
    if not all(np.isfinite(value) for value in resolved):
        raise ValueError(f"{name} 必须全部为有限数")
    if any(resolved[index] >= resolved[index + 3] for index in range(3)):
        raise ValueError(f"{name} 的 XYZ 下界必须分别小于上界")
    return resolved


def _validate_target_size(values: Sequence[int]) -> Tuple[int, int, int]:
    if len(values) != 3:
        raise ValueError("target_size 必须按 Z,X,Y 提供三个正整数")
    resolved = tuple(int(value) for value in values)
    if any(value <= 0 for value in resolved):
        raise ValueError("target_size 必须按 Z,X,Y 提供三个正整数")
    return resolved


def _validate_voxel_size(values: Optional[Sequence[float]]) -> Optional[Tuple[float, ...]]:
    if values is None:
        return None
    if len(values) != 3:
        raise ValueError("voxel_size 必须包含 XYZ 三个正数")
    resolved = tuple(float(value) for value in values)
    if not all(np.isfinite(value) and value > 0.0 for value in resolved):
        raise ValueError("voxel_size 必须包含 XYZ 三个有限正数")
    return resolved


def _ensure_fresh_output_dir(output_dir: str) -> None:
    if os.path.exists(output_dir):
        if not os.path.isdir(output_dir):
            raise ValueError(f"output_dir 已存在但不是目录: {output_dir}")
        if os.listdir(output_dir):
            raise ValueError(f"output_dir 已存在且非空，拒绝覆盖: {output_dir}")


def _find_target_path(target_voxel_dir: str, frame_id: str) -> str:
    for extension in (".npz", ".npy"):
        path = os.path.join(target_voxel_dir, f"{frame_id}{extension}")
        if os.path.isfile(path):
            return path
    raise RuntimeError(f"prediction {frame_id} 缺少对应 target voxel")


def _collect_pairs(
    pred_voxel_dir: str,
    target_voxel_dir: str,
    target_size: Tuple[int, int, int],
    max_files: int,
) -> List[Tuple[str, str, str]]:
    if not os.path.isdir(pred_voxel_dir):
        raise ValueError(f"pred_voxel_dir 不存在: {pred_voxel_dir}")
    if not os.path.isdir(target_voxel_dir):
        raise ValueError(f"target_voxel_dir 不存在: {target_voxel_dir}")

    prediction_files = sorted(
        name for name in os.listdir(pred_voxel_dir) if name.endswith("_voxel.npy")
    )
    if not prediction_files:
        raise RuntimeError(f"pred_voxel_dir 中没有 *_voxel.npy: {pred_voxel_dir}")
    if max_files > 0:
        prediction_files = prediction_files[:max_files]

    pairs: List[Tuple[str, str, str]] = []
    suffix = "_voxel.npy"
    for prediction_name in prediction_files:
        frame_id = prediction_name[:-len(suffix)]
        prediction_path = os.path.join(pred_voxel_dir, prediction_name)
        target_path = _find_target_path(target_voxel_dir, frame_id)
        prediction = np.load(prediction_path, mmap_mode="r")
        if (
            prediction.ndim != 4
            or prediction.shape[0] < 2
            or tuple(int(value) for value in prediction.shape[1:]) != target_size
        ):
            raise ValueError(
                "prediction 必须为 (C,Z,X,Y)、C>=2 且空间尺寸等于 target_size；"
                f"文件={prediction_path}, shape={prediction.shape}, target_size={target_size}"
            )
        pairs.append((frame_id, prediction_path, target_path))
    return pairs


def _numeric_summary(values: Sequence[float]) -> Dict[str, Optional[float]]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"min": None, "median": None, "mean": None, "max": None}
    return {
        "min": float(array.min()),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "max": float(array.max()),
    }


def run_diagnostic(
    pred_voxel_dir: str,
    target_voxel_dir: str,
    output_dir: str,
    target_threshold: float = 0.1,
    source_pc_range: Sequence[float] = DEFAULT_SOURCE_PC_RANGE,
    model_pc_range: Sequence[float] = DEFAULT_MODEL_PC_RANGE,
    target_size: Sequence[int] = DEFAULT_TARGET_SIZE,
    voxel_size: Optional[Sequence[float]] = None,
    max_files: int = 0,
) -> Dict[str, object]:
    """执行离线 oracle 诊断并返回与 JSON 文件一致的审计报告。"""
    target_threshold = float(target_threshold)
    if not np.isfinite(target_threshold) or not 0.0 <= target_threshold <= 1.0:
        raise ValueError("target_threshold 必须是 [0,1] 内的有限数")
    max_files = int(max_files)
    if max_files < 0:
        raise ValueError("max_files 不能为负数")

    source_pc_range = _validate_pc_range(source_pc_range, "source_pc_range")
    model_pc_range = _validate_pc_range(model_pc_range, "model_pc_range")
    target_size = _validate_target_size(target_size)
    voxel_size = _validate_voxel_size(voxel_size)
    _ensure_fresh_output_dir(output_dir)
    pairs = _collect_pairs(
        pred_voxel_dir,
        target_voxel_dir,
        target_size,
        max_files,
    )

    os.makedirs(output_dir, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for index, (frame_id, prediction_path, target_path) in enumerate(pairs):
        prediction = np.load(prediction_path).astype(np.float32)
        target_occ = load_target_occ_resized(
            target_path,
            source_pc_range=source_pc_range,
            model_pc_range=model_pc_range,
            target_size=target_size,
        )
        target_occ_count = int(np.count_nonzero(target_occ > target_threshold))
        oracle_threshold, effective_match_count = find_oracle_occ_threshold(
            prediction[0],
            target_occ_count,
        )
        pointcloud, _ = voxel_to_pointcloud(
            prediction,
            voxel_size=voxel_size,
            pc_range=model_pc_range,
            occ_threshold=oracle_threshold,
            empty_fallback_topk=0,
        )
        pointcloud_name = f"{frame_id}_oracle_pcl.npy"
        np.save(os.path.join(output_dir, pointcloud_name), pointcloud.astype(np.float32))
        pred_point_count = int(pointcloud.shape[0])
        ratio = (
            float(pred_point_count / target_occ_count)
            if target_occ_count > 0
            else None
        )
        rows.append(
            {
                "index": index,
                "frame_id": frame_id,
                "pred_voxel_file": os.path.basename(prediction_path),
                "target_voxel_file": os.path.basename(target_path),
                "target_occ_count": target_occ_count,
                "effective_match_count": effective_match_count,
                "oracle_occ_threshold": oracle_threshold,
                "oracle_pred_point_count": pred_point_count,
                "pred_to_target_count_ratio": ratio,
                "oracle_pointcloud_file": pointcloud_name,
            }
        )

    csv_path = os.path.join(output_dir, "oracle_target_adaptation_frames.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            if csv_row["pred_to_target_count_ratio"] is None:
                csv_row["pred_to_target_count_ratio"] = ""
            writer.writerow(csv_row)

    report: Dict[str, object] = {
        "protocol": "oracle_target_count_matching",
        "deployable": False,
        "warning": "该结果使用测试 target 改变逐帧输出，不得作为正式推理性能。",
        "pred_voxel_dir": os.path.abspath(pred_voxel_dir),
        "target_voxel_dir": os.path.abspath(target_voxel_dir),
        "output_dir": os.path.abspath(output_dir),
        "target_threshold": target_threshold,
        "source_pc_range": list(source_pc_range),
        "model_pc_range": list(model_pc_range),
        "target_size": list(target_size),
        "voxel_size": list(voxel_size) if voxel_size is not None else None,
        "frame_count": len(rows),
        "oracle_occ_threshold": _numeric_summary(
            [float(row["oracle_occ_threshold"]) for row in rows]
        ),
        "target_occ_count": _numeric_summary(
            [float(row["target_occ_count"]) for row in rows]
        ),
        "oracle_pred_point_count": _numeric_summary(
            [float(row["oracle_pred_point_count"]) for row in rows]
        ),
        "pred_to_target_count_ratio": _numeric_summary(
            [
                float(row["pred_to_target_count_ratio"])
                for row in rows
                if row["pred_to_target_count_ratio"] is not None
            ]
        ),
    }
    json_path = os.path.join(output_dir, "oracle_target_adaptation_report.json")
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(report, json_file, ensure_ascii=False, indent=2, allow_nan=False)
    return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="离线诊断逐帧 target 数量匹配的 oracle 占用阈值与点云",
    )
    parser.add_argument("--pred_voxel_dir", required=True)
    parser.add_argument("--target_voxel_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_threshold", type=float, default=0.1)
    parser.add_argument(
        "--source_pc_range",
        type=float,
        nargs=6,
        default=DEFAULT_SOURCE_PC_RANGE,
    )
    parser.add_argument(
        "--model_pc_range",
        type=float,
        nargs=6,
        default=DEFAULT_MODEL_PC_RANGE,
    )
    parser.add_argument(
        "--target_size",
        type=int,
        nargs=3,
        default=DEFAULT_TARGET_SIZE,
        help="模型网格，顺序为 Z X Y",
    )
    parser.add_argument("--voxel_size", type=float, nargs=3, default=None)
    parser.add_argument("--max_files", type=int, default=0)
    args = parser.parse_args(argv)

    report = run_diagnostic(
        pred_voxel_dir=args.pred_voxel_dir,
        target_voxel_dir=args.target_voxel_dir,
        output_dir=args.output_dir,
        target_threshold=args.target_threshold,
        source_pc_range=args.source_pc_range,
        model_pc_range=args.model_pc_range,
        target_size=args.target_size,
        voxel_size=args.voxel_size,
        max_files=args.max_files,
    )
    print(f"Oracle diagnostic saved to: {report['output_dir']}")
    print("WARNING: 该结果不可作为正式推理性能。")


if __name__ == "__main__":
    main()
