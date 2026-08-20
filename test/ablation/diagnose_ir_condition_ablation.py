#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断 LDM/CD 是否真正使用红外条件输入。

同一批帧分别使用原始 IR、置零 IR、mock IR 做推理，比较 decoded occupancy
差异及其相对 LiDAR target 的结构质量。如果差异长期接近 0，说明当前模型基本
忽略红外分支；如果只增加点数而 target 指标不提升，则说明红外主要造成增密。
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from typing import Dict, List, Sequence

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from diffusion_consistency_radar.cm.dataset_loader import NTU4DRadLM_VoxelDataset  # noqa: E402
from diffusion_consistency_radar.cm.evaluation_metrics import vertical_structure_metrics  # noqa: E402
from diffusion_consistency_radar.scripts.inference import RadarGenerator  # noqa: E402


VALID_IR_VARIANTS = ("real", "zero", "mock")


def _sha256_file(path: str) -> str:
    """流式计算模型文件哈希，绑定诊断结果与权重内容。"""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_manifest_sha256(root: str) -> str:
    """按相对路径、大小和修改时间生成轻量数据集 manifest 指纹。"""
    canonical_root = os.path.realpath(root)
    digest = hashlib.sha256()
    for current_root, dirs, files in os.walk(canonical_root, followlinks=False):
        dirs.sort()
        for name in sorted(files):
            path = os.path.join(current_root, name)
            stat = os.stat(path)
            relative = os.path.relpath(path, canonical_root)
            digest.update(f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8"))
    return digest.hexdigest()


def _source_fingerprints(model_ckpt: str, vae_ckpt: str, dataset_root: str) -> Dict[str, str]:
    """收集一次不可变输入快照，用于检测诊断期间的源文件变化。"""
    return {
        "checkpoint_sha256": _sha256_file(model_ckpt),
        "vae_sha256": _sha256_file(vae_ckpt),
        "dataset_manifest_sha256": _dataset_manifest_sha256(dataset_root),
    }


def parse_variants(raw: str) -> tuple:
    """解析需要执行的 IR 条件变体，并拒绝未知项或重复项。"""
    variants = tuple(token.strip() for token in str(raw).split(",") if token.strip())
    if not variants:
        raise ValueError("IR variants 不能为空")
    unknown = [variant for variant in variants if variant not in VALID_IR_VARIANTS]
    if unknown:
        raise ValueError(f"未知 IR variant: {unknown[0]}")
    if len(set(variants)) != len(variants):
        raise ValueError(f"IR variants 包含重复项: {variants}")
    if "real" not in variants:
        raise ValueError("IR variants 必须包含 real，作为差异比较基准")
    return variants


def clone_meta(meta: Dict[str, object], device: torch.device) -> Dict[str, object]:
    """把 dataset meta 复制成单 batch 推理格式。"""
    cloned = {}
    for key, value in meta.items():
        if torch.is_tensor(value):
            tensor = value.to(device)
            if key == "t_vec" and tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            elif key in {"ir_img", "r_mat", "k_mat"} and tensor.dim() in (2, 3):
                tensor = tensor.unsqueeze(0)
            if key in {"is_mock_ir", "is_mock_calib", "odom_cov_trace"} and tensor.dim() == 0:
                tensor = tensor.view(1)
            cloned[key] = tensor
        elif key in {"is_mock_ir", "is_mock_calib", "odom_cov_trace"} and isinstance(value, (bool, int, float)):
            cloned[key] = torch.as_tensor([value], device=device, dtype=torch.float32)
    for key in ("is_mock_ir", "is_mock_calib", "odom_cov_trace"):
        if key not in cloned:
            cloned[key] = torch.zeros(1, device=device)
    return cloned


def make_meta_variant(meta: Dict[str, object], variant: str, device: torch.device) -> Dict[str, object]:
    """生成原始、置零和 mock 三种 IR 条件。"""
    variant_meta = clone_meta(meta, device)
    if variant == "real":
        return variant_meta
    if variant == "zero":
        variant_meta["ir_img"] = torch.zeros_like(variant_meta["ir_img"])
        return variant_meta
    if variant == "mock":
        _, _, height, width = variant_meta["ir_img"].shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=device),
            torch.linspace(-1.0, 1.0, width, device=device),
            indexing="ij",
        )
        thermal = torch.exp(-((xx * 1.8) ** 2 + (yy * 1.2) ** 2))
        variant_meta["ir_img"] = torch.stack(
            [thermal, thermal * 0.85, thermal * 0.65],
            dim=0,
        ).unsqueeze(0)
        variant_meta["is_mock_ir"] = torch.ones(1, device=device)
        return variant_meta
    raise ValueError(f"未知 IR variant: {variant}")


def compare_outputs(reference: torch.Tensor, other: torch.Tensor) -> Dict[str, float]:
    """返回两个输出体素 occupancy 通道的差异摘要。"""
    ref_occ = reference[:, 0:1].float()
    other_occ = other[:, 0:1].float()
    diff = (ref_occ - other_occ).abs()
    return {
        "mean_abs_diff": float(diff.mean().item()),
        "max_abs_diff": float(diff.max().item()),
        "ref_occ_mean": float(ref_occ.mean().item()),
        "other_occ_mean": float(other_occ.mean().item()),
    }


def _occupancy_zxy(voxel: object) -> np.ndarray:
    """把常见 batch/channel 布局统一为 ZXY occupancy。"""
    if torch.is_tensor(voxel):
        arr = voxel.detach().float().cpu().numpy()
    else:
        arr = np.asarray(voxel, dtype=np.float32)
    if arr.ndim == 5:
        if arr.shape[0] != 1:
            raise ValueError(f"只支持单样本 batch，实际 shape={arr.shape}")
        arr = arr[0]
    if arr.ndim == 4:
        if arr.shape[0] <= 8:
            arr = arr[0]
        elif arr.shape[-1] <= 8:
            arr = arr[..., 0]
        else:
            raise ValueError(f"无法识别 occupancy 通道布局: shape={arr.shape}")
    if arr.ndim != 3:
        raise ValueError(f"occupancy 需要 3D/4D/5D voxel，实际 shape={arr.shape}")
    return np.asarray(arr, dtype=np.float32)


def _prf_from_counts(tp: float, fp: float, fn: float) -> Dict[str, float]:
    """由 TP/FP/FN 计算 precision、recall、F1 和 IoU。"""
    precision = tp / (tp + fp) if tp + fp > 0.0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0.0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0.0 else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn > 0.0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}


def compute_target_metrics(
    prediction: object,
    target: object,
    occ_threshold: float,
    target_threshold: float,
    pc_range: Sequence[float],
) -> Dict[str, float]:
    """计算一个 IR 变体相对 LiDAR target 的占用、BEV 与竖向结构指标。"""
    pred_occ = _occupancy_zxy(prediction)
    target_occ = _occupancy_zxy(target)
    if pred_occ.shape != target_occ.shape:
        raise ValueError(f"prediction/target shape 不一致: {pred_occ.shape} vs {target_occ.shape}")

    pred_mask = pred_occ > float(occ_threshold)
    target_mask = target_occ > float(target_threshold)
    voxel_tp = float(np.count_nonzero(pred_mask & target_mask))
    voxel_fp = float(np.count_nonzero(pred_mask & ~target_mask))
    voxel_fn = float(np.count_nonzero(~pred_mask & target_mask))
    voxel_prf = _prf_from_counts(voxel_tp, voxel_fp, voxel_fn)

    pred_bev = np.any(pred_mask, axis=0)
    target_bev = np.any(target_mask, axis=0)
    bev_tp = float(np.count_nonzero(pred_bev & target_bev))
    bev_fp = float(np.count_nonzero(pred_bev & ~target_bev))
    bev_fn = float(np.count_nonzero(~pred_bev & target_bev))
    bev_prf = _prf_from_counts(bev_tp, bev_fp, bev_fn)

    # 结构指标内部会 squeeze；仅在合成测试存在单元素轴时补零，避免维度被压扁。
    padding = tuple((0, 1 if size == 1 else 0) for size in pred_occ.shape)
    structure_pred = np.pad(pred_occ, padding, mode="constant") if any(size == 1 for size in pred_occ.shape) else pred_occ
    structure_target = (
        np.pad(target_mask.astype(np.float32), padding, mode="constant")
        if any(size == 1 for size in target_occ.shape)
        else target_mask.astype(np.float32)
    )
    structure = vertical_structure_metrics(
        structure_pred,
        structure_target,
        pc_range=pc_range,
        occ_threshold=float(occ_threshold),
    )
    pred_count = float(np.count_nonzero(pred_mask))
    target_count = float(np.count_nonzero(target_mask))
    metrics = {
        "pred_occ_count": pred_count,
        "target_occ_count": target_count,
        "pred_to_target_ratio": pred_count / target_count if target_count > 0.0 else 0.0,
        "voxel_tp": voxel_tp,
        "voxel_fp": voxel_fp,
        "voxel_fn": voxel_fn,
        "voxel_precision": voxel_prf["precision"],
        "voxel_recall": voxel_prf["recall"],
        "voxel_f1": voxel_prf["f1"],
        "voxel_iou": voxel_prf["iou"],
        "bev_tp": bev_tp,
        "bev_fp": bev_fp,
        "bev_fn": bev_fn,
        "bev_precision": bev_prf["precision"],
        "bev_recall": bev_prf["recall"],
        "bev_f1": bev_prf["f1"],
        "bev_iou": bev_prf["iou"],
    }
    metrics.update(structure)
    return metrics


def summarize_target_rows(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """对同一 IR 变体的逐帧指标做 micro aggregate。"""
    rows = list(rows)
    if not rows:
        raise ValueError("target-aware IR 消融没有可汇总的帧")
    variant = str(rows[0]["variant"])
    summary: Dict[str, float] = {"variant": variant, "frames": int(len(rows))}
    pred_total = float(sum(float(row["pred_occ_count"]) for row in rows))
    target_total = float(sum(float(row["target_occ_count"]) for row in rows))
    summary["pred_occ_total"] = pred_total
    summary["target_occ_total"] = target_total
    summary["pred_to_target_ratio"] = pred_total / target_total if target_total > 0.0 else 0.0

    for prefix in ("voxel", "bev"):
        tp = float(sum(float(row[f"{prefix}_tp"]) for row in rows))
        fp = float(sum(float(row[f"{prefix}_fp"]) for row in rows))
        fn = float(sum(float(row[f"{prefix}_fn"]) for row in rows))
        summary[f"{prefix}_tp"] = tp
        summary[f"{prefix}_fp"] = fp
        summary[f"{prefix}_fn"] = fn
        prf = _prf_from_counts(tp, fp, fn)
        for name, value in prf.items():
            summary[f"{prefix}_{name}"] = value

    for prefix in ("height_coverage", "top_height", "vertical_connectivity", "trunk_region"):
        numerator = float(sum(float(row[f"{prefix}_numerator"]) for row in rows))
        denominator = float(sum(float(row[f"{prefix}_denominator"]) for row in rows))
        summary[f"{prefix}_numerator"] = numerator
        summary[f"{prefix}_denominator"] = denominator
        summary[f"{prefix}_recall"] = numerator / denominator if denominator > 0.0 else 0.0

    summary["mean_abs_diff_vs_real"] = float(
        np.mean([float(row.get("mean_abs_diff_vs_real", 0.0)) for row in rows])
    )
    return summary


def select_sample_indices(dataset_size: int, sample_index: int, max_samples: int) -> List[int]:
    """单帧保持旧行为，多帧时在整个 split 中做确定性等间隔抽样。"""
    if int(dataset_size) <= 0:
        raise ValueError("dataset_size 必须大于 0")
    if int(sample_index) < 0 or int(sample_index) >= int(dataset_size):
        raise IndexError(f"sample_index={sample_index} 超出数据集长度 {dataset_size}")
    count = max(1, min(int(max_samples), int(dataset_size)))
    if count == 1:
        return [int(sample_index)]
    return np.linspace(0, int(dataset_size) - 1, num=count, dtype=np.int64).tolist()


def _write_csv(
    path: str,
    rows: Sequence[Dict[str, object]],
    fieldnames: Sequence[str] = (),
) -> None:
    """按首行字段顺序写 CSV。"""
    resolved_fields = list(rows[0].keys()) if rows else list(fieldnames)
    if not resolved_fields:
        raise ValueError(f"空 CSV 必须显式提供 fieldnames: {path}")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=resolved_fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: str, summaries: Sequence[Dict[str, float]], args: argparse.Namespace) -> None:
    """输出便于比较 real/zero/mock 的 Markdown 摘要。"""
    lines = [
        "# Target-aware IR Condition Ablation",
        "",
        f"- split: `{args.split}`",
        f"- frames: {summaries[0]['frames']}",
        f"- occupancy threshold: {args.occ_threshold:.4f}",
        f"- target threshold: {args.target_threshold:.4f}",
        f"- checkpoint: `{os.path.abspath(args.model_ckpt)}`",
        "",
        "| IR variant | Voxel P | Voxel R | Voxel IoU | BEV P | BEV R | BEV IoU | Count ratio | Height | Top | Connectivity | Trunk |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['variant']} | {row['voxel_precision']:.4f} | {row['voxel_recall']:.4f} | "
            f"{row['voxel_iou']:.4f} | {row['bev_precision']:.4f} | {row['bev_recall']:.4f} | "
            f"{row['bev_iou']:.4f} | {row['pred_to_target_ratio']:.4f} | "
            f"{row['height_coverage_recall']:.4f} | {row['top_height_recall']:.4f} | "
            f"{row['vertical_connectivity_recall']:.4f} | {row['trunk_region_recall']:.4f} |"
        )
    lines.extend(
        [
            "",
            "real 优于 zero/mock 才说明红外带来了 LiDAR 对齐的结构收益；若主要只有 count ratio 上升，",
            "则应先控制红外增密或修正融合监督，再进入下一轮训练。",
        ]
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="IR condition ablation diagnostic for multimodal LDM/CD")
    parser.add_argument("--dataset_root", required=True, help="例如 Data/NTU4DRadLM_Pre_sensor_aware")
    parser.add_argument("--vae_ckpt", required=True)
    parser.add_argument("--model_ckpt", required=True)
    parser.add_argument(
        "--output_dir", default="test/result/ldm/ablation/ir_condition_ablation"
    )
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=1, help="大于 1 时在当前 split 等间隔抽样")
    parser.add_argument(
        "--require_sample_count",
        type=int,
        default=0,
        help="大于 0 时要求实际抽样数严格相等，避免固定验证协议静默退化",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--model_type", choices=("ldm", "cd"), default="ldm")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--sampler", choices=("euler", "heun"), default="euler")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--target_size", type=int, nargs=3, default=None)
    parser.add_argument("--source_pc_range", type=float, nargs=6, default=None)
    parser.add_argument("--model_pc_range", type=float, nargs=6, default=None)
    parser.add_argument("--occ_threshold", type=float, default=0.5)
    parser.add_argument("--target_threshold", type=float, default=0.5)
    parser.add_argument(
        "--allow_legacy_radar_units",
        action="store_true",
        help="仅用于缺少 normalization 协议的历史 checkpoint 诊断",
    )
    parser.add_argument(
        "--variants",
        default="real,zero,mock",
        help="逗号分隔的 IR 条件变体；checkpoint 选择可使用 real 以减少推理量",
    )
    args = parser.parse_args()
    variants = parse_variants(args.variants)

    device = torch.device(args.device)
    initial_fingerprints = _source_fingerprints(
        args.model_ckpt, args.vae_ckpt, args.dataset_root
    )
    generator = RadarGenerator(
        vae_path=args.vae_ckpt,
        model_path=args.model_ckpt,
        model_type=args.model_type,
        device=str(device),
        target_size=tuple(args.target_size) if args.target_size else None,
        source_pc_range=tuple(args.source_pc_range) if args.source_pc_range else None,
        pc_range=tuple(args.model_pc_range) if args.model_pc_range else None,
        allow_legacy_radar_units=args.allow_legacy_radar_units,
    )
    dataset = NTU4DRadLM_VoxelDataset(
        args.dataset_root,
        split=args.split,
        use_augmentation=False,
        target_size=generator.target_size,
        source_pc_range=generator.source_pc_range,
        model_pc_range=generator.pc_range,
        radar_normalization=generator.radar_normalization,
        radar_normalization_sha256=generator.radar_normalization_sha256,
        allow_legacy_radar_units=generator.allow_legacy_radar_units,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"dataset 为空: {args.dataset_root}")
    sample_indices = select_sample_indices(len(dataset), args.sample_index, args.max_samples)
    if args.require_sample_count > 0 and len(sample_indices) != args.require_sample_count:
        raise RuntimeError(
            f"固定验证协议要求 {args.require_sample_count} 帧，实际只能抽取 {len(sample_indices)} 帧"
        )
    os.makedirs(args.output_dir, exist_ok=True)

    pc_range = generator.pc_range
    detailed_rows: List[Dict[str, object]] = []
    legacy_rows: List[Dict[str, object]] = []
    for position, sample_index in enumerate(sample_indices, start=1):
        target, radar, meta = dataset[sample_index]
        radar = radar.unsqueeze(0).to(device)
        outputs = {}
        for variant in variants:
            torch.manual_seed(args.seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(args.seed)
            outputs[variant] = generator.generate(
                radar,
                steps=args.steps,
                sampler=args.sampler,
                meta_dict=make_meta_variant(meta, variant, device),
            ).detach().cpu()

        for variant in variants:
            comparison = compare_outputs(outputs["real"], outputs[variant])
            row: Dict[str, object] = {
                "sample_index": int(sample_index),
                "variant": variant,
                "mean_abs_diff_vs_real": comparison["mean_abs_diff"],
                "max_abs_diff_vs_real": comparison["max_abs_diff"],
                "occ_mean": comparison["other_occ_mean"],
            }
            row.update(
                compute_target_metrics(
                    outputs[variant],
                    target,
                    occ_threshold=args.occ_threshold,
                    target_threshold=args.target_threshold,
                    pc_range=pc_range,
                )
            )
            detailed_rows.append(row)

        if position == 1:
            for variant in variants:
                if variant == "real":
                    continue
                row = {"variant": variant}
                row.update(compare_outputs(outputs["real"], outputs[variant]))
                legacy_rows.append(row)
        print(f"sample[{position}/{len(sample_indices)}] index={sample_index}")

    summaries = [
        summarize_target_rows([row for row in detailed_rows if row["variant"] == variant])
        for variant in variants
    ]

    csv_path = os.path.join(args.output_dir, "ir_condition_ablation.csv")
    json_path = os.path.join(args.output_dir, "ir_condition_ablation.json")
    metrics_csv_path = os.path.join(args.output_dir, "ir_condition_ablation_metrics.csv")
    summary_csv_path = os.path.join(args.output_dir, "ir_condition_ablation_summary.csv")
    report_path = os.path.join(args.output_dir, "ir_condition_ablation_report.md")
    final_fingerprints = _source_fingerprints(
        args.model_ckpt, args.vae_ckpt, args.dataset_root
    )
    if final_fingerprints != initial_fingerprints:
        raise RuntimeError("诊断期间 checkpoint、VAE 或 dataset 内容发生变化，结果已拒绝写入")
    _write_csv(
        csv_path,
        legacy_rows,
        fieldnames=(
            "variant", "mean_abs_diff", "max_abs_diff", "ref_occ_mean", "other_occ_mean",
        ),
    )
    _write_csv(metrics_csv_path, detailed_rows)
    _write_csv(summary_csv_path, summaries)
    _write_report(report_path, summaries, args)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_root": args.dataset_root,
                "sample_index": args.sample_index,
                "sample_indices": sample_indices,
                "model_ckpt": args.model_ckpt,
                "vae_ckpt": args.vae_ckpt,
                "protocol": {
                    "dataset_root": os.path.realpath(args.dataset_root),
                    "vae_ckpt": os.path.realpath(args.vae_ckpt),
                    **initial_fingerprints,
                    "split": args.split,
                    "max_samples": int(args.max_samples),
                    "required_sample_count": int(args.require_sample_count),
                    "steps": int(args.steps),
                    "sampler": args.sampler,
                    "seed": int(args.seed),
                    "occ_threshold": float(args.occ_threshold),
                    "target_threshold": float(args.target_threshold),
                    "variants": list(variants),
                    "target_size": list(generator.target_size),
                    "source_pc_range": list(generator.source_pc_range),
                    "model_pc_range": list(generator.pc_range),
                    "radar_normalization": generator.radar_normalization,
                    "radar_normalization_sha256": generator.radar_normalization_sha256,
                    "allow_legacy_radar_units": generator.allow_legacy_radar_units,
                    "formal_protocol": generator.radar_normalization is not None,
                    "sample_indices": list(sample_indices),
                },
                "rows": legacy_rows,
                "summaries": summaries,
            },
            f,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    print(f"Saved IR ablation CSV to: {csv_path}")
    print(f"Saved IR ablation JSON to: {json_path}")
    print(f"Saved target-aware metrics to: {metrics_csv_path}")
    print(f"Saved target-aware summary to: {summary_csv_path}")
    print(f"Saved target-aware report to: {report_path}")


if __name__ == "__main__":
    main()
