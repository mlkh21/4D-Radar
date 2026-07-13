#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估已保存 LDM voxel 与 target voxel 的垂直结构保持情况。

脚本只读取已有推理结果和监督 target，不触发重新推理或训练。
"""

import argparse
import csv
import os
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from diffusion_consistency_radar.cm.evaluation_metrics import vertical_structure_metrics
from diffusion_consistency_radar.cm.dataset_loader import (
    crop_voxel_channels_to_pc_range,
    resize_voxel_channels,
)


DEFAULT_PC_RANGE = (0.0, -20.0, -6.0, 40.0, 20.0, 10.0)
DEFAULT_SOURCE_PC_RANGE = (0.0, -20.0, -6.0, 120.0, 20.0, 10.0)
DEFAULT_TARGET_SIZE = (32, 128, 128)
STRUCTURE_PREFIXES = (
    "height_coverage",
    "top_height",
    "vertical_connectivity",
    "trunk_region",
)
METRIC_FIELDS = [
    "frame_id",
    "pred_file",
    "target_file",
    "pred_occ_count",
    "target_occ_count",
    "height_coverage_recall",
    "height_coverage_numerator",
    "height_coverage_denominator",
    "top_height_recall",
    "top_height_numerator",
    "top_height_denominator",
    "vertical_connectivity_recall",
    "vertical_connectivity_numerator",
    "vertical_connectivity_denominator",
    "trunk_region_recall",
    "trunk_region_numerator",
    "trunk_region_denominator",
]
SUMMARY_FIELDS = [
    "frame_id",
    "frames",
    "pred_occ_total",
    "target_occ_total",
    "height_coverage_recall",
    "height_coverage_numerator",
    "height_coverage_denominator",
    "top_height_recall",
    "top_height_numerator",
    "top_height_denominator",
    "vertical_connectivity_recall",
    "vertical_connectivity_numerator",
    "vertical_connectivity_denominator",
    "trunk_region_recall",
    "trunk_region_numerator",
    "trunk_region_denominator",
]


def parse_pc_range(values: Sequence[float]) -> Tuple[float, float, float, float, float, float]:
    """校验并转换 pc_range 参数。"""
    if len(values) != 6:
        raise ValueError(f"pc_range 需要 6 个浮点数，实际为 {len(values)} 个")
    pc_range = tuple(float(v) for v in values)
    if pc_range[3] <= pc_range[0] or pc_range[4] <= pc_range[1] or pc_range[5] <= pc_range[2]:
        raise ValueError(f"pc_range 最大边界必须大于最小边界: {pc_range}")
    return pc_range


def parse_target_size(values: Sequence[int]) -> Tuple[int, int, int]:
    """校验并转换 [Z,X,Y] target_size 参数。"""
    if len(values) != 3:
        raise ValueError(f"target_size 需要 3 个整数，实际为 {len(values)} 个")
    target_size = tuple(int(v) for v in values)
    if any(v <= 0 for v in target_size):
        raise ValueError(f"target_size 必须为正整数: {target_size}")
    return target_size


def frame_id_from_prediction(filename: str) -> str:
    """从 LDM 输出文件名中提取帧号。"""
    stem, _ = os.path.splitext(os.path.basename(filename))
    if stem.endswith("_voxel"):
        return stem[: -len("_voxel")]
    return stem


def load_prediction_occupancy(path: str) -> np.ndarray:
    """读取 prediction dense npy，并统一成 ZXY occupancy score。"""
    arr = np.asarray(np.load(path), dtype=np.float32).squeeze()
    if arr.ndim == 3:
        return arr.astype(np.float32)
    if arr.ndim != 4:
        raise ValueError(f"prediction 需要 3D/4D voxel，实际 shape={arr.shape}: {path}")

    # 优先兼容 [Z,X,Y,C]；当 Z==4 时不能误判为 channel-first。
    if arr.shape[-1] in (1, 4) and arr.shape[1] not in (1, 4) and arr.shape[2] not in (1, 4):
        return arr[..., 0].astype(np.float32)
    # LDM 常见输出为 [C,Z,X,Y]，occupancy 在第 0 通道。
    if arr.shape[0] in (1, 4) and arr.shape[-1] not in (1, 4):
        return arr[0].astype(np.float32)
    if arr.shape[0] <= 8:
        return arr[0].astype(np.float32)
    if arr.shape[-1] <= 8:
        return arr[..., 0].astype(np.float32)
    raise ValueError(f"无法识别 prediction occupancy 通道布局: shape={arr.shape}, path={path}")


def _load_sparse_target_xyzc(path: str) -> np.ndarray:
    """按 sparse 预处理协议读取 target dense XY Z C。"""
    data = np.load(path)
    required = {"coords", "features", "shape"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"sparse target 缺少字段 {sorted(missing)}: {path}")
    shape = tuple(int(v) for v in data["shape"])
    voxel = np.zeros(shape, dtype=np.float32)
    coords = np.asarray(data["coords"], dtype=np.int64)
    features = np.asarray(data["features"], dtype=np.float32)
    if coords.size:
        voxel[coords[:, 0], coords[:, 1], coords[:, 2]] = features
    return voxel


def _target_xyzc_to_czxy(voxel: np.ndarray) -> np.ndarray:
    """将 target 的常见布局统一为 [C,Z,X,Y]。"""
    arr = np.asarray(voxel, dtype=np.float32).squeeze()
    if arr.ndim == 4:
        if arr.shape[-1] <= 8:
            return arr.transpose(3, 2, 0, 1)
        if arr.shape[0] <= 8:
            return arr
    if arr.ndim == 3:
        # 3D dense target 没有通道元数据，按脚本内部统一的 ZXY occupancy 兜底读取。
        return arr[None, ...]
    raise ValueError(f"target 需要 3D 或 4D voxel，实际 shape={arr.shape}")


def load_target_occupancy(
    path: str,
    target_threshold: float = 0.5,
    source_pc_range: Sequence[float] = DEFAULT_SOURCE_PC_RANGE,
    model_pc_range: Sequence[float] = DEFAULT_PC_RANGE,
    target_size: Sequence[int] = DEFAULT_TARGET_SIZE,
) -> np.ndarray:
    """读取 target，并按训练/推理网格协议输出二值 ZXY occupancy。"""
    if path.endswith(".npz"):
        voxel = _load_sparse_target_xyzc(path)
    else:
        voxel = np.load(path)
    tensor = torch.from_numpy(_target_xyzc_to_czxy(voxel)).float()
    tensor = crop_voxel_channels_to_pc_range(tensor, source_pc_range, model_pc_range)
    tensor = resize_voxel_channels(tensor, target_size, mask_channel=3 if tensor.shape[0] > 3 else None)
    return (tensor[0].cpu().numpy() > float(target_threshold)).astype(np.float32)


def find_target_file(target_voxel_dir: str, frame_id: str) -> str:
    """按帧号查找 target voxel 文件。"""
    candidates = (
        f"{frame_id}.npz",
        f"{frame_id}.npy",
        f"{frame_id}_voxel.npz",
        f"{frame_id}_voxel.npy",
    )
    for name in candidates:
        path = os.path.join(target_voxel_dir, name)
        if os.path.exists(path):
            return path
    return ""


def iter_prediction_files(pred_voxel_dir: str, max_files: int = 0) -> List[str]:
    """列出待评估 prediction voxel。"""
    files = [
        os.path.join(pred_voxel_dir, name)
        for name in sorted(os.listdir(pred_voxel_dir))
        if name.endswith("_voxel.npy")
    ]
    if max_files and int(max_files) > 0:
        files = files[: int(max_files)]
    return files


def compute_frame_metrics(
    frame_id: str,
    pred_path: str,
    target_path: str,
    pc_range: Sequence[float],
    occ_threshold: float,
    target_threshold: float,
    source_pc_range: Sequence[float] = DEFAULT_SOURCE_PC_RANGE,
    target_size: Sequence[int] = DEFAULT_TARGET_SIZE,
) -> Dict[str, float]:
    """计算单帧垂直结构指标。"""
    pred_occ = load_prediction_occupancy(pred_path)
    target_occ = load_target_occupancy(
        target_path,
        target_threshold=target_threshold,
        source_pc_range=source_pc_range,
        model_pc_range=pc_range,
        target_size=target_size,
    )
    if pred_occ.shape != target_occ.shape:
        raise ValueError(
            f"prediction/target shape 不一致 frame={frame_id}: "
            f"{pred_occ.shape} vs {target_occ.shape}"
        )
    metrics = vertical_structure_metrics(
        pred_occ,
        target_occ,
        pc_range=pc_range,
        occ_threshold=float(occ_threshold),
    )
    row = {
        "frame_id": frame_id,
        "pred_file": os.path.basename(pred_path),
        "target_file": os.path.basename(target_path),
        "pred_occ_count": float(np.count_nonzero(pred_occ > float(occ_threshold))),
        "target_occ_count": float(np.count_nonzero(target_occ > 0.5)),
    }
    row.update(metrics)
    return row


def summarize_rows(rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    """按 numerator/denominator 做 micro aggregate。"""
    rows = list(rows)
    summary: Dict[str, float] = {
        "frame_id": "__summary__",
        "frames": int(len(rows)),
        "pred_occ_total": float(sum(float(row["pred_occ_count"]) for row in rows)),
        "target_occ_total": float(sum(float(row["target_occ_count"]) for row in rows)),
    }
    for prefix in STRUCTURE_PREFIXES:
        numerator = float(sum(float(row[f"{prefix}_numerator"]) for row in rows))
        denominator = float(sum(float(row[f"{prefix}_denominator"]) for row in rows))
        summary[f"{prefix}_numerator"] = numerator
        summary[f"{prefix}_denominator"] = denominator
        summary[f"{prefix}_recall"] = numerator / denominator if denominator > 0.0 else 0.0
    return summary


def write_csv(path: str, rows: Sequence[Dict[str, float]], fields: Sequence[str]) -> None:
    """写入固定列顺序 CSV。"""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: str,
    summary: Dict[str, float],
    metrics_csv: str,
    summary_csv: str,
    occ_threshold: float,
    target_threshold: float,
    pc_range: Sequence[float],
    source_pc_range: Sequence[float],
    target_size: Sequence[int],
) -> None:
    """生成轻量 Markdown 报告。"""
    lines = [
        "# LDM Vertical Structure Evaluation",
        "",
        f"- frames: {int(summary['frames'])}",
        f"- occ_threshold: {float(occ_threshold):.4f}",
        f"- target_threshold: {float(target_threshold):.4f}",
        f"- source_pc_range: {' '.join(f'{float(v):g}' for v in source_pc_range)}",
        f"- pc_range: {' '.join(f'{float(v):g}' for v in pc_range)}",
        f"- target_size [Z X Y]: {' '.join(str(int(v)) for v in target_size)}",
        f"- per-frame CSV: `{metrics_csv}`",
        f"- summary CSV: `{summary_csv}`",
        "",
        "## Summary",
        "",
        f"- height coverage recall: {summary['height_coverage_recall']:.4f}",
        f"- top height recall: {summary['top_height_recall']:.4f}",
        f"- vertical connectivity recall: {summary['vertical_connectivity_recall']:.4f}",
        f"- trunk recall: {summary['trunk_region_recall']:.4f}",
        "",
        (
            "height coverage 衡量目标竖向占用体素被预测覆盖的比例，top height 衡量树冠/障碍物顶部是否到达，"
            "vertical connectivity 衡量竖向连续段是否断裂，trunk recall 衡量低处主干区域是否被保留。"
        ),
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def evaluate_directories(
    pred_voxel_dir: str,
    target_voxel_dir: str,
    output_dir: str,
    occ_threshold: float = 0.05,
    target_threshold: float = 0.5,
    pc_range: Sequence[float] = DEFAULT_PC_RANGE,
    source_pc_range: Sequence[float] = DEFAULT_SOURCE_PC_RANGE,
    target_size: Sequence[int] = DEFAULT_TARGET_SIZE,
    max_files: int = 0,
) -> Dict[str, float]:
    """评估目录中已保存的 LDM voxel，并写出 CSV/Markdown。"""
    pc_range = parse_pc_range(pc_range)
    source_pc_range = parse_pc_range(source_pc_range)
    target_size = parse_target_size(target_size)
    os.makedirs(output_dir, exist_ok=True)

    rows: List[Dict[str, float]] = []
    pred_files = iter_prediction_files(pred_voxel_dir, max_files=max_files)
    for pred_path in pred_files:
        frame_id = frame_id_from_prediction(os.path.basename(pred_path))
        target_path = find_target_file(target_voxel_dir, frame_id)
        if not target_path:
            raise FileNotFoundError(f"找不到 frame={frame_id} 对应 target voxel: {target_voxel_dir}")
        rows.append(
            compute_frame_metrics(
                frame_id=frame_id,
                pred_path=pred_path,
                target_path=target_path,
                pc_range=pc_range,
                occ_threshold=occ_threshold,
                target_threshold=target_threshold,
                source_pc_range=source_pc_range,
                target_size=target_size,
            )
        )

    if not rows:
        raise ValueError(f"没有找到可评估的 prediction voxel: {pred_voxel_dir}")

    summary = summarize_rows(rows)
    metrics_csv = os.path.join(output_dir, "vertical_structure_metrics.csv")
    summary_csv = os.path.join(output_dir, "vertical_structure_summary.csv")
    report_md = os.path.join(output_dir, "vertical_structure_report.md")
    write_csv(metrics_csv, rows, METRIC_FIELDS)
    write_csv(summary_csv, [summary], SUMMARY_FIELDS)
    write_report(
        report_md,
        summary=summary,
        metrics_csv=metrics_csv,
        summary_csv=summary_csv,
        occ_threshold=occ_threshold,
        target_threshold=target_threshold,
        pc_range=pc_range,
        source_pc_range=source_pc_range,
        target_size=target_size,
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate saved LDM voxels with vertical structure metrics.")
    parser.add_argument("--pred_voxel_dir", required=True, help="LDM 推理输出 voxel 目录")
    parser.add_argument("--target_voxel_dir", required=True, help="target voxel 目录")
    parser.add_argument("--output_dir", required=True, help="评估结果输出目录")
    parser.add_argument("--occ_threshold", type=float, default=0.05, help="prediction occupancy 阈值")
    parser.add_argument("--target_threshold", type=float, default=0.5, help="target occupancy 二值阈值")
    parser.add_argument(
        "--pc_range",
        type=float,
        nargs=6,
        default=DEFAULT_PC_RANGE,
        metavar=("X_MIN", "Y_MIN", "Z_MIN", "X_MAX", "Y_MAX", "Z_MAX"),
        help="模型物理范围，默认近场 0 -20 -6 40 20 10",
    )
    parser.add_argument(
        "--source_pc_range",
        type=float,
        nargs=6,
        default=DEFAULT_SOURCE_PC_RANGE,
        metavar=("X_MIN", "Y_MIN", "Z_MIN", "X_MAX", "Y_MAX", "Z_MAX"),
        help="target 原始预处理物理范围，默认 0 -20 -6 120 20 10",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        nargs=3,
        default=DEFAULT_TARGET_SIZE,
        metavar=("Z", "X", "Y"),
        help="模型输出体素尺寸 [Z X Y]，默认 32 128 128",
    )
    parser.add_argument("--max_files", type=int, default=0, help="最多评估帧数，0 表示全量")
    return parser


def main(argv: Sequence[str] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = evaluate_directories(
        pred_voxel_dir=args.pred_voxel_dir,
        target_voxel_dir=args.target_voxel_dir,
        output_dir=args.output_dir,
        occ_threshold=args.occ_threshold,
        target_threshold=args.target_threshold,
        pc_range=args.pc_range,
        source_pc_range=args.source_pc_range,
        target_size=args.target_size,
        max_files=args.max_files,
    )
    print(f"Saved vertical structure evaluation to: {args.output_dir}")
    print(
        "Summary: "
        f"height_coverage={summary['height_coverage_recall']:.4f}, "
        f"top_height={summary['top_height_recall']:.4f}, "
        f"vertical_connectivity={summary['vertical_connectivity_recall']:.4f}, "
        f"trunk_region={summary['trunk_region_recall']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
