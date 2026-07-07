#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VAE 重建上界诊断脚本。

该脚本只评估 target_voxel 经过 VAE 编码/解码后的占用保持能力，
用于判断树木/障碍物结构是在 VAE 阶段已经丢失，还是后续 LDM/CD
生成阶段才丢失。
"""

import argparse
import csv
import os
import sys
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from diffusion_consistency_radar.cm.dataset_loader import (
        crop_voxel_channels_to_pc_range,
        load_sparse_voxel,
        resize_voxel_channels,
    )
    from diffusion_consistency_radar.cm.evaluation_metrics import vertical_structure_metrics
    from diffusion_consistency_radar.cm.vae_3d import (
        VAE3D,
        build_vae_from_checkpoint,
        resolve_checkpoint_grid_config,
        create_lightweight_vae_config,
        create_standard_vae_config,
        create_ultra_lightweight_vae_config,
    )
except Exception:
    from cm.dataset_loader import crop_voxel_channels_to_pc_range, load_sparse_voxel, resize_voxel_channels
    from cm.evaluation_metrics import vertical_structure_metrics
    from cm.vae_3d import (
        VAE3D,
        build_vae_from_checkpoint,
        resolve_checkpoint_grid_config,
        create_lightweight_vae_config,
        create_standard_vae_config,
        create_ultra_lightweight_vae_config,
    )

STRUCTURE_METRIC_PREFIXES = [
    "height_coverage",
    "top_height",
    "vertical_connectivity",
    "trunk_region",
]


def parse_csv_floats(raw: str, expected: int) -> List[float]:
    values = [float(v.strip()) for v in str(raw).split(",") if v.strip()]
    if expected >= 0 and len(values) != expected:
        raise ValueError(f"Expected {expected} comma-separated values, got {raw}")
    return values


def safe_torch_load(path, map_location):
    """兼容不同 PyTorch 版本的 checkpoint 加载逻辑。"""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception as exc:
        if "Weights only load failed" in str(exc) or "Unsupported global" in str(exc):
            return torch.load(path, map_location=map_location)
        raise


def build_vae(config_type: str) -> VAE3D:
    if config_type == "lightweight":
        cfg = create_lightweight_vae_config()
    elif config_type == "standard":
        cfg = create_standard_vae_config()
    elif config_type == "ultra_lightweight":
        cfg = create_ultra_lightweight_vae_config()
    else:
        raise ValueError(f"Unknown VAE config_type: {config_type}")
    return VAE3D(**cfg)


def build_vae_from_diagnostic_checkpoint(
    checkpoint: Any,
    fallback_config_type: str = None,
):
    """复用共享协议构建诊断 VAE，fallback 仅服务历史权重。"""
    return build_vae_from_checkpoint(
        checkpoint,
        fallback_config_type=fallback_config_type,
    )


def resolve_diagnostic_grid_config(
    checkpoint_metadata,
    target_size,
    source_pc_range,
    model_pc_range,
):
    """解析诊断使用的有效网格，CLI 显式值拥有最高优先级。"""
    return resolve_checkpoint_grid_config(
        checkpoint_metadata,
        target_size=target_size,
        source_pc_range=source_pc_range,
        model_pc_range=model_pc_range,
    )


def resolve_occupancy_activation(checkpoint: Any) -> str:
    """读取 occupancy 激活元数据，旧 checkpoint 保持 raw 输出语义。"""
    if not isinstance(checkpoint, dict):
        return "raw"
    activation = checkpoint.get("occupancy_activation", "raw")
    if activation not in {"raw", "sigmoid"}:
        raise ValueError(f"Unsupported occupancy_activation: {activation}")
    return activation


def extract_reconstruction_occupancy(
    reconstruction: torch.Tensor,
    occupancy_activation: str,
) -> torch.Tensor:
    """仅提取第零个 occupancy 通道，并按 checkpoint 语义转换。"""
    occupancy = reconstruction[:, 0]
    if occupancy_activation == "sigmoid":
        return VAE3D.occupancy_probability(occupancy)
    if occupancy_activation == "raw":
        return occupancy
    raise ValueError(f"Unsupported occupancy_activation: {occupancy_activation}")


def compute_binary_occ_metrics(gt_occ: np.ndarray, pred_occ: np.ndarray) -> Dict[str, float]:
    """计算单帧二值 occupancy 的 IoU/Recall/Precision。"""
    gt = np.asarray(gt_occ).astype(bool)
    pred = np.asarray(pred_occ).astype(bool)
    inter = int(np.logical_and(gt, pred).sum())
    union = int(np.logical_or(gt, pred).sum())
    gt_count = int(gt.sum())
    pred_count = int(pred.sum())
    return {
        "intersection": inter,
        "union": union,
        "gt_occ": gt_count,
        "recon_occ": pred_count,
        "iou": inter / max(union, 1),
        "recall": inter / max(gt_count, 1),
        "precision": inter / max(pred_count, 1),
        "count_ratio": pred_count / max(gt_count, 1),
    }


def compute_vertical_structure_metrics(
    gt_occ: np.ndarray,
    pred_occ: np.ndarray,
    pc_range,
    top_height_tolerance_m: float,
    trunk_base_max_z: float,
    trunk_min_height_m: float,
    trunk_height_cap_m: float,
) -> Dict[str, float]:
    """计算单帧垂直结构保持指标。"""
    return vertical_structure_metrics(
        pred_occ,
        gt_occ,
        pc_range=pc_range,
        occ_threshold=0.5,
        top_height_tolerance_m=top_height_tolerance_m,
        trunk_base_max_z=trunk_base_max_z,
        trunk_min_height_m=trunk_min_height_m,
        trunk_height_cap_m=trunk_height_cap_m,
    )


def evaluate_reconstruction_threshold(
    recon_occ_score: np.ndarray,
    gt_occ: np.ndarray,
    threshold: float,
    pc_range,
    top_height_tolerance_m: float,
    trunk_base_max_z: float,
    trunk_min_height_m: float,
    trunk_height_cap_m: float,
) -> Dict[str, float]:
    """按单个阈值联合计算二值占用与垂直结构指标。"""
    pred_occ = np.asarray(recon_occ_score) > float(threshold)
    return {
        **compute_binary_occ_metrics(gt_occ, pred_occ),
        **compute_vertical_structure_metrics(
            gt_occ,
            pred_occ,
            pc_range=pc_range,
            top_height_tolerance_m=top_height_tolerance_m,
            trunk_base_max_z=trunk_base_max_z,
            trunk_min_height_m=trunk_min_height_m,
            trunk_height_cap_m=trunk_height_cap_m,
        ),
    }


def extend_metric_fields(base_fields: List[str]) -> List[str]:
    """按固定顺序附加结构指标字段。"""
    fields = list(base_fields)
    for metric_name in STRUCTURE_METRIC_PREFIXES:
        fields.extend(
            [
                f"{metric_name}_recall",
                f"{metric_name}_numerator",
                f"{metric_name}_denominator",
            ]
        )
    return fields


def summarize_threshold_rows(rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    rows = list(rows)
    structure_fields = {
        f"{prefix}_{suffix}"
        for prefix in STRUCTURE_METRIC_PREFIXES
        for suffix in ("numerator", "denominator")
    }
    structure_aware = any(
        any(key.startswith(f"{prefix}_") for prefix in STRUCTURE_METRIC_PREFIXES)
        for row in rows
        for key in row
    )
    if structure_aware:
        for row_index, row in enumerate(rows):
            missing_fields = sorted(structure_fields.difference(row))
            if missing_fields:
                raise ValueError(
                    f"Structure-aware row {row_index} missing fields: "
                    f"{', '.join(missing_fields)}"
                )

    inter = sum(int(r["intersection"]) for r in rows)
    union = sum(int(r["union"]) for r in rows)
    gt_count = sum(int(r["gt_occ"]) for r in rows)
    pred_count = sum(int(r["recon_occ"]) for r in rows)
    summary = {
        "frames": len(rows),
        "intersection": inter,
        "union": union,
        "gt_occ_total": gt_count,
        "recon_occ_total": pred_count,
        "iou": inter / max(union, 1),
        "recall": inter / max(gt_count, 1),
        "precision": inter / max(pred_count, 1),
        "count_ratio": pred_count / max(gt_count, 1),
    }
    for metric_name in STRUCTURE_METRIC_PREFIXES:
        numerator = sum(float(r[f"{metric_name}_numerator"]) for r in rows) if structure_aware else 0.0
        denominator = sum(float(r[f"{metric_name}_denominator"]) for r in rows) if structure_aware else 0.0
        summary[f"{metric_name}_numerator"] = numerator
        summary[f"{metric_name}_denominator"] = denominator
        summary[f"{metric_name}_recall"] = numerator / denominator if denominator > 0.0 else 0.0
    return summary


def build_best_threshold_report_lines(best_row: Dict[str, float]) -> List[str]:
    """生成最佳 IoU 阈值对应的关键结论。"""
    return [
        f"- best threshold by IoU: {best_row['threshold']:.3f}",
        (
            f"- best IoU / Recall / Precision: "
            f"{best_row['iou']:.4f} / {best_row['recall']:.4f} / {best_row['precision']:.4f}"
        ),
        (
            f"- best structure recall: "
            f"height_coverage={best_row.get('height_coverage_recall', 0.0):.4f}, "
            f"top_height={best_row.get('top_height_recall', 0.0):.4f}, "
            f"vertical_connectivity={best_row.get('vertical_connectivity_recall', 0.0):.4f}, "
            f"trunk_region={best_row.get('trunk_region_recall', 0.0):.4f}"
        ),
    ]


def load_target_tensor(path, target_size, source_pc_range, model_pc_range, device):
    if path.endswith(".npz"):
        voxel = load_sparse_voxel(path)
    else:
        voxel = np.load(path).astype(np.float32)
    tensor = torch.from_numpy(voxel).permute(3, 2, 0, 1)
    tensor = crop_voxel_channels_to_pc_range(tensor, source_pc_range, model_pc_range)
    tensor = resize_voxel_channels(tensor, target_size, mask_channel=3)
    return tensor.unsqueeze(0).to(device)


def main():
    parser = argparse.ArgumentParser(description="Diagnose VAE reconstruction upper bound")
    parser.add_argument("--vae_ckpt", required=True, help="VAE checkpoint path")
    parser.add_argument("--target_voxel_dir", required=True, help="target_voxel directory")
    parser.add_argument("--output_dir", default="test/result/vae_reconstruction_diagnostic")
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument(
        "--thresholds",
        default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,"
                "0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95",
    )
    parser.add_argument("--config_type", default=None,
                        choices=["ultra_lightweight", "lightweight", "standard"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target_size", default=None, help="[Z,X,Y] comma-separated")
    parser.add_argument("--source_pc_range", default=None)
    parser.add_argument("--model_pc_range", default=None)
    parser.add_argument("--top_height_tolerance_m", type=float, default=0.0)
    parser.add_argument("--trunk_base_max_z", type=float, default=1.0)
    parser.add_argument("--trunk_min_height_m", type=float, default=2.0)
    parser.add_argument("--trunk_height_cap_m", type=float, default=3.0)
    args = parser.parse_args()

    cli_target_size = (
        tuple(int(v) for v in parse_csv_floats(args.target_size, 3))
        if args.target_size is not None else None
    )
    cli_source_range = (
        tuple(parse_csv_floats(args.source_pc_range, 6))
        if args.source_pc_range is not None else None
    )
    cli_model_range = (
        tuple(parse_csv_floats(args.model_pc_range, 6))
        if args.model_pc_range is not None else None
    )
    thresholds = parse_csv_floats(args.thresholds, -1)
    if not thresholds:
        raise ValueError("--thresholds must contain at least one value")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    ckpt = safe_torch_load(args.vae_ckpt, map_location=device)
    model, checkpoint_metadata = build_vae_from_diagnostic_checkpoint(
        ckpt,
        fallback_config_type=args.config_type,
    )
    model = model.to(device)
    occupancy_activation = checkpoint_metadata["occupancy_activation"]
    target_size, source_pc_range, model_pc_range = resolve_diagnostic_grid_config(
        checkpoint_metadata,
        cli_target_size,
        cli_source_range,
        cli_model_range,
    )
    model.eval()

    files = sorted(f for f in os.listdir(args.target_voxel_dir) if f.endswith(".npz") or f.endswith(".npy"))
    if args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        raise RuntimeError(f"No target voxel files found in {args.target_voxel_dir}")

    rows = []
    os.makedirs(args.output_dir, exist_ok=True)

    with torch.no_grad():
        for name in tqdm(files, desc="VAE reconstruction", unit="frame"):
            path = os.path.join(args.target_voxel_dir, name)
            target = load_target_tensor(path, target_size, source_pc_range, model_pc_range, device)
            latent, _ = model.encode(target, deterministic=True)
            recon = model.decode(latent)
            gt_occ = (target[:, 0] > 0.5).detach().cpu().numpy()
            recon_occ_score = extract_reconstruction_occupancy(
                recon,
                occupancy_activation,
            ).detach().cpu().numpy()

            for threshold in thresholds:
                metrics = evaluate_reconstruction_threshold(
                    recon_occ_score,
                    gt_occ,
                    threshold=threshold,
                    pc_range=model_pc_range,
                    top_height_tolerance_m=args.top_height_tolerance_m,
                    trunk_base_max_z=args.trunk_base_max_z,
                    trunk_min_height_m=args.trunk_min_height_m,
                    trunk_height_cap_m=args.trunk_height_cap_m,
                )
                row = {
                    "frame": os.path.splitext(name)[0],
                    "threshold": threshold,
                    **metrics,
                }
                rows.append(row)

    per_frame_path = os.path.join(args.output_dir, "vae_reconstruction_metrics.csv")
    fieldnames = extend_metric_fields([
        "frame", "threshold", "intersection", "union", "gt_occ", "recon_occ",
        "iou", "recall", "precision", "count_ratio",
    ])
    with open(per_frame_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for threshold in thresholds:
        summary = summarize_threshold_rows(r for r in rows if float(r["threshold"]) == threshold)
        summary_rows.append({"threshold": threshold, **summary})

    summary_path = os.path.join(args.output_dir, "vae_reconstruction_summary.csv")
    summary_fields = extend_metric_fields([
        "threshold", "frames", "intersection", "union", "gt_occ_total",
        "recon_occ_total", "iou", "recall", "precision", "count_ratio",
    ])
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    report_path = os.path.join(args.output_dir, "vae_reconstruction_report.md")
    best = max(summary_rows, key=lambda r: r["iou"])
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# VAE Reconstruction Upper-Bound Diagnostic\n\n")
        f.write(f"- checkpoint: `{args.vae_ckpt}`\n")
        f.write(f"- target_voxel_dir: `{args.target_voxel_dir}`\n")
        f.write(f"- frames: {len(files)}\n")
        f.write(f"- target_size [Z,X,Y]: {list(target_size)}\n")
        f.write(f"- source_pc_range: {list(source_pc_range)}\n")
        f.write(f"- model_pc_range: {list(model_pc_range)}\n")
        f.write(f"- top_height_tolerance_m: {args.top_height_tolerance_m}\n")
        f.write(f"- trunk_base_max_z: {args.trunk_base_max_z}\n")
        f.write(f"- trunk_min_height_m: {args.trunk_min_height_m}\n")
        f.write(f"- trunk_height_cap_m: {args.trunk_height_cap_m}\n")
        for line in build_best_threshold_report_lines(best):
            f.write(f"{line}\n")

    print(f"Saved per-frame metrics to: {per_frame_path}")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()
