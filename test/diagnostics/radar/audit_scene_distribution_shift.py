#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""审计训练场景与独立验证场景的 Radar、target 和 IR 数据分布差异。"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_SOURCE_PC_RANGE = (0.0, -20.0, -6.0, 120.0, 20.0, 10.0)
DEFAULT_MODEL_PC_RANGE = (0.0, -20.0, -6.0, 40.0, 20.0, 10.0)
DEFAULT_TARGET_SIZE = (64, 128, 128)
DEFAULT_X_EDGES = (0.0, 10.0, 20.0, 30.0, 40.0)
DEFAULT_Z_EDGES = (-6.0, -1.0, 0.0, 2.0, 5.0, 10.0)


def evenly_spaced_indices(total: int, max_frames: int) -> List[int]:
    """确定性均匀抽帧；max_frames<=0 时保留全部帧。"""
    if total <= 0:
        return []
    if max_frames <= 0 or max_frames >= total:
        return list(range(total))
    return np.linspace(0, total - 1, max_frames, dtype=np.int64).tolist()


def _voxel_files(folder: str) -> Dict[str, str]:
    if not os.path.isdir(folder):
        return {}
    paths: Dict[str, str] = {}
    for name in sorted(os.listdir(folder)):
        if not name.endswith((".npz", ".npy")) or name.endswith("_pcl.npy"):
            continue
        paths[os.path.splitext(name)[0]] = os.path.join(folder, name)
    return paths


def paired_frame_paths(radar_dir: str, target_dir: str) -> List[Tuple[str, str, str]]:
    """按帧名配对 Radar 和 target，避免错位统计。"""
    radar = _voxel_files(radar_dir)
    target = _voxel_files(target_dir)
    stems = sorted(set(radar).intersection(target))
    return [(stem, radar[stem], target[stem]) for stem in stems]


def _physical_centers(coords: np.ndarray, shape: Sequence[int], pc_range: Sequence[float]) -> np.ndarray:
    if len(shape) < 3 or coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"无效稀疏体素协议: coords={coords.shape}, shape={tuple(shape)}")
    mins = np.asarray(pc_range[:3], dtype=np.float64)
    maxs = np.asarray(pc_range[3:], dtype=np.float64)
    steps = (maxs - mins) / np.asarray(shape[:3], dtype=np.float64)
    return mins + (coords.astype(np.float64) + 0.5) * steps


def _band_counts(values: np.ndarray, edges: Sequence[float]) -> List[int]:
    return [
        int(np.count_nonzero((values >= low) & (values < high)))
        for low, high in zip(edges[:-1], edges[1:])
    ]


def sparse_voxel_stats(
    path: str,
    source_pc_range: Sequence[float],
    model_pc_range: Sequence[float],
    x_edges: Sequence[float],
    z_edges: Sequence[float],
    include_radar_channels: bool,
) -> Dict[str, object]:
    """在物理裁剪范围内统计单帧稀疏体素，不构造稠密网格。"""
    if not path.endswith(".npz"):
        raise ValueError(f"分布审计要求稀疏 NPZ 输入，收到: {path}")
    with np.load(path) as data:
        required = {"coords", "features", "shape"}
        if not required.issubset(data.files):
            raise ValueError(f"{path} 缺少稀疏体素字段: {sorted(required - set(data.files))}")
        coords = np.asarray(data["coords"])
        features = np.asarray(data["features"], dtype=np.float64)
        shape = np.asarray(data["shape"]).reshape(-1)

    if coords.shape[0] != features.shape[0]:
        raise ValueError(f"coords/features 数量不一致: {coords.shape[0]} vs {features.shape[0]}")
    centers = _physical_centers(coords, shape, source_pc_range)
    model_min = np.asarray(model_pc_range[:3], dtype=np.float64)
    model_max = np.asarray(model_pc_range[3:], dtype=np.float64)
    inside = np.all((centers >= model_min) & (centers < model_max), axis=1)
    if features.ndim == 2 and features.shape[1] > 0:
        inside &= features[:, 0] > 0.0
    centers = centers[inside]
    features = features[inside]

    count = int(centers.shape[0])
    stats: Dict[str, object] = {
        "occupied_count": count,
        "x_band_counts": _band_counts(centers[:, 0], x_edges) if count else [0] * (len(x_edges) - 1),
        "z_band_counts": _band_counts(centers[:, 2], z_edges) if count else [0] * (len(z_edges) - 1),
    }
    if include_radar_channels:
        if features.ndim != 2 or features.shape[1] < 4:
            raise ValueError(f"Radar features 至少需要 4 通道，收到 {features.shape}")
        doppler = features[:, 2] if count else np.empty(0, dtype=np.float64)
        variance = features[:, 3] if count else np.empty(0, dtype=np.float64)
        stats.update(
            {
                "doppler_mean": float(doppler.mean()) if count else 0.0,
                "doppler_abs_mean": float(np.abs(doppler).mean()) if count else 0.0,
                "doppler_std": float(doppler.std()) if count else 0.0,
                "doppler_variance_mean": float(variance.mean()) if count else 0.0,
                "doppler_variance_p90": float(np.percentile(variance, 90)) if count else 0.0,
            }
        )
    return stats


def _ir_path(scene_dir: str, stem: str) -> str:
    for suffix in ("_ir.npy", ".npy", ".npz"):
        path = os.path.join(scene_dir, "ir_image", f"{stem}{suffix}")
        if os.path.exists(path):
            return path
    return ""


def _ir_stats(path: str) -> Dict[str, float]:
    if not path:
        return {"ir_available": 0.0, "ir_mean": 0.0, "ir_std": 0.0, "ir_p10": 0.0, "ir_p90": 0.0}
    data = np.load(path)
    if isinstance(data, np.lib.npyio.NpzFile):
        try:
            keys = data.files
            if not keys:
                raise ValueError(f"空 IR NPZ: {path}")
            image = np.asarray(data[keys[0]], dtype=np.float64)
        finally:
            data.close()
    else:
        image = np.asarray(data, dtype=np.float64)
    sample = image.reshape(-1)[::16]
    return {
        "ir_available": 1.0,
        "ir_mean": float(sample.mean()) if sample.size else 0.0,
        "ir_std": float(sample.std()) if sample.size else 0.0,
        "ir_p10": float(np.percentile(sample, 10)) if sample.size else 0.0,
        "ir_p90": float(np.percentile(sample, 90)) if sample.size else 0.0,
    }


def _summary(values: Iterable[float], prefix: str) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return {f"{prefix}_{name}": 0.0 for name in ("mean", "median", "p10", "p90")}
    return {
        f"{prefix}_mean": float(array.mean()),
        f"{prefix}_median": float(np.median(array)),
        f"{prefix}_p10": float(np.percentile(array, 10)),
        f"{prefix}_p90": float(np.percentile(array, 90)),
    }


def audit_scene(
    dataset_root: str,
    scene: str,
    max_frames: int,
    source_pc_range: Sequence[float],
    model_pc_range: Sequence[float],
    target_size: Sequence[int],
    x_edges: Sequence[float],
    z_edges: Sequence[float],
) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]]]:
    # 标定链路依赖 PyTorch，仅在真实数据审计时加载，避免纯函数测试初始化重依赖。
    from diffusion_consistency_radar.cm.dataset_loader import CalibrationProvider
    from diffusion_consistency_radar.scripts.audit_dataset_protocol import _frustum_voxel_ratio

    scene_dir = os.path.join(dataset_root, scene)
    pairs = paired_frame_paths(
        os.path.join(scene_dir, "radar_voxel"),
        os.path.join(scene_dir, "target_voxel"),
    )
    selected = [pairs[index] for index in evenly_spaced_indices(len(pairs), max_frames)]
    frame_rows: List[Dict[str, object]] = []
    radar_x_total = np.zeros(len(x_edges) - 1, dtype=np.int64)
    target_x_total = np.zeros(len(x_edges) - 1, dtype=np.int64)
    radar_z_total = np.zeros(len(z_edges) - 1, dtype=np.int64)
    target_z_total = np.zeros(len(z_edges) - 1, dtype=np.int64)

    for stem, radar_path, target_path in selected:
        radar = sparse_voxel_stats(
            radar_path, source_pc_range, model_pc_range, x_edges, z_edges, True
        )
        target = sparse_voxel_stats(
            target_path, source_pc_range, model_pc_range, x_edges, z_edges, False
        )
        ir = _ir_stats(_ir_path(scene_dir, stem))
        target_count = int(target["occupied_count"])
        row = {
            "scene": scene,
            "frame": stem,
            "radar_occupied": int(radar["occupied_count"]),
            "target_occupied": target_count,
            "radar_target_ratio": float(radar["occupied_count"]) / max(target_count, 1),
            **{key: radar[key] for key in (
                "doppler_mean", "doppler_abs_mean", "doppler_std",
                "doppler_variance_mean", "doppler_variance_p90",
            )},
            **ir,
        }
        frame_rows.append(row)
        radar_x_total += np.asarray(radar["x_band_counts"], dtype=np.int64)
        target_x_total += np.asarray(target["x_band_counts"], dtype=np.int64)
        radar_z_total += np.asarray(radar["z_band_counts"], dtype=np.int64)
        target_z_total += np.asarray(target["z_band_counts"], dtype=np.int64)

    r_mat, t_vec, k_mat, calib_meta = CalibrationProvider(dataset_root).load_with_metadata()
    frustum_ratio = _frustum_voxel_ratio(
        r_mat.numpy(), t_vec.numpy(), k_mat.numpy(), target_size, model_pc_range
    )
    summary: Dict[str, object] = {
        "scene": scene,
        "paired_frames": len(pairs),
        "sampled_frames": len(frame_rows),
        "calib_source": calib_meta["calib_source"],
        "is_mock_calib": bool(calib_meta["is_mock_calib"]),
        "ir_frustum_voxel_ratio": frustum_ratio,
        "ir_coverage": float(np.mean([row["ir_available"] for row in frame_rows])) if frame_rows else 0.0,
    }
    for key in (
        "radar_occupied", "target_occupied", "radar_target_ratio", "doppler_mean",
        "doppler_abs_mean", "doppler_std", "doppler_variance_mean",
        "doppler_variance_p90", "ir_mean", "ir_std", "ir_p10", "ir_p90",
    ):
        summary.update(_summary((float(row[key]) for row in frame_rows), key))

    band_rows: List[Dict[str, object]] = []
    for axis, edges, radar_counts, target_counts in (
        ("x", x_edges, radar_x_total, target_x_total),
        ("z", z_edges, radar_z_total, target_z_total),
    ):
        radar_sum = max(int(radar_counts.sum()), 1)
        target_sum = max(int(target_counts.sum()), 1)
        for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
            band_rows.append(
                {
                    "scene": scene,
                    "axis": axis,
                    "band_min": float(low),
                    "band_max": float(high),
                    "radar_count": int(radar_counts[index]),
                    "target_count": int(target_counts[index]),
                    "radar_fraction": float(radar_counts[index]) / radar_sum,
                    "target_fraction": float(target_counts[index]) / target_sum,
                }
            )
    return summary, frame_rows, band_rows


def _write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _relative_delta(reference: float, candidate: float) -> float:
    return (candidate - reference) / max(abs(reference), 1e-12)


def write_outputs(
    output_dir: str,
    summaries: List[Dict[str, object]],
    frame_rows: List[Dict[str, object]],
    band_rows: List[Dict[str, object]],
    protocol: Dict[str, object],
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    _write_csv(os.path.join(output_dir, "scene_distribution_summary.csv"), summaries)
    _write_csv(os.path.join(output_dir, "scene_distribution_frames.csv"), frame_rows)
    _write_csv(os.path.join(output_dir, "scene_distribution_bands.csv"), band_rows)
    with open(os.path.join(output_dir, "scene_distribution_protocol.json"), "w", encoding="utf-8") as handle:
        json.dump(protocol, handle, ensure_ascii=False, indent=2)

    report_path = os.path.join(output_dir, "scene_distribution_report.md")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("# 场景数据分布差异审计\n\n")
        handle.write("本报告在原始稀疏体素的物理坐标中统计，不改变训练数据或模型。\n\n")
        handle.write("| scene | frames | radar occ | target occ | R/T ratio | |Doppler| | variance | IR coverage | frustum |\n")
        handle.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in summaries:
            handle.write(
                f"| {row['scene']} | {row['sampled_frames']} | {row['radar_occupied_mean']:.2f} | "
                f"{row['target_occupied_mean']:.2f} | {row['radar_target_ratio_mean']:.3f} | "
                f"{row['doppler_abs_mean_mean']:.3f} | {row['doppler_variance_mean_mean']:.3f} | "
                f"{row['ir_coverage']:.3f} | {row['ir_frustum_voxel_ratio']:.3f} |\n"
            )
        if len(summaries) == 2:
            reference, candidate = summaries
            handle.write("\n## 相对变化\n\n")
            handle.write(f"以 `{reference['scene']}` 为训练域参考，`{candidate['scene']}` 的变化为：\n\n")
            for key, label in (
                ("radar_occupied_mean", "Radar 占用数"),
                ("target_occupied_mean", "Target 占用数"),
                ("radar_target_ratio_mean", "Radar/Target 比例"),
                ("doppler_abs_mean_mean", "绝对 Doppler"),
                ("doppler_variance_mean_mean", "Doppler 方差"),
                ("ir_mean_mean", "IR 均值"),
                ("ir_std_mean", "IR 标准差"),
            ):
                delta = _relative_delta(float(reference[key]), float(candidate[key]))
                handle.write(f"- {label}: {delta:+.1%}\n")
        handle.write("\n## 分档分布\n\n")
        handle.write("| scene | axis | band | radar fraction | target fraction |\n")
        handle.write("| --- | --- | --- | ---: | ---: |\n")
        for row in band_rows:
            handle.write(
                f"| {row['scene']} | {row['axis']} | [{row['band_min']:.1f}, {row['band_max']:.1f}) | "
                f"{row['radar_fraction']:.4f} | {row['target_fraction']:.4f} |\n"
            )
    print(f"Saved scene distribution report to: {report_path}")


def _parse_edges(raw: str, name: str) -> Tuple[float, ...]:
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if len(values) < 2 or any(a >= b for a, b in zip(values[:-1], values[1:])):
        raise argparse.ArgumentTypeError(f"{name} 必须是严格递增且至少包含两个值")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="审计训练/验证场景的 Radar、target 与 IR 分布差异")
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--scenes", default="garden,loop3")
    parser.add_argument("--output_dir", default="test/result/comparison/scene_distribution_audit_v11")
    parser.add_argument("--max_frames", type=int, default=500, help="每个场景均匀抽帧数；0 表示全部")
    parser.add_argument("--source_pc_range", type=float, nargs=6, default=DEFAULT_SOURCE_PC_RANGE)
    parser.add_argument("--model_pc_range", type=float, nargs=6, default=DEFAULT_MODEL_PC_RANGE)
    parser.add_argument("--target_size", type=int, nargs=3, default=DEFAULT_TARGET_SIZE)
    parser.add_argument("--x_edges", default=",".join(map(str, DEFAULT_X_EDGES)))
    parser.add_argument("--z_edges", default=",".join(map(str, DEFAULT_Z_EDGES)))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.max_frames < 0:
        parser.error("max_frames 不能为负数")
    scenes = [scene.strip() for scene in args.scenes.split(",") if scene.strip()]
    if len(scenes) < 2:
        parser.error("至少需要两个场景才能审计分布差异")
    x_edges = _parse_edges(args.x_edges, "x_edges")
    z_edges = _parse_edges(args.z_edges, "z_edges")
    if os.path.isdir(args.output_dir) and os.listdir(args.output_dir) and not args.overwrite:
        parser.error(f"输出目录已存在且非空: {args.output_dir}; 如需覆盖请传 --overwrite")

    summaries: List[Dict[str, object]] = []
    frame_rows: List[Dict[str, object]] = []
    band_rows: List[Dict[str, object]] = []
    for scene in scenes:
        summary, scene_frames, scene_bands = audit_scene(
            args.dataset_root,
            scene,
            args.max_frames,
            args.source_pc_range,
            args.model_pc_range,
            args.target_size,
            x_edges,
            z_edges,
        )
        if not scene_frames:
            parser.error(f"场景 {scene} 没有可配对的 Radar/target 稀疏体素")
        summaries.append(summary)
        frame_rows.extend(scene_frames)
        band_rows.extend(scene_bands)

    protocol = {
        "dataset_root": os.path.abspath(args.dataset_root),
        "scenes": scenes,
        "max_frames_per_scene": args.max_frames,
        "source_pc_range": list(args.source_pc_range),
        "model_pc_range": list(args.model_pc_range),
        "target_size_zxy": list(args.target_size),
        "x_edges": list(x_edges),
        "z_edges": list(z_edges),
        "sampling": "deterministic_even_spacing_in_common_frame_stems",
    }
    write_outputs(args.output_dir, summaries, frame_rows, band_rows, protocol)


if __name__ == "__main__":
    main()
