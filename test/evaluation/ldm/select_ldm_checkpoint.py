#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""根据固定验证集任务与竖向结构指标选择 LDM checkpoint。

该工具只生成推荐报告，不复制或覆盖训练产生的 checkpoint。所有候选必须由同一
验证协议生成，以避免把阈值、采样步数或数据切分差异误认为模型差异。
"""

import argparse
import csv
import hashlib
import json
import math
import os
from typing import Dict, List, Sequence


DEFAULT_GATES = {
    "bev_iou": 0.2548,
    "bev_recall": 0.80,
    "top_height_recall": 0.10,
    "trunk_region_recall": 0.65,
    "max_pred_to_target_ratio": 6.0,
}

EXPECTED_FIXED_PROTOCOL = {
    "split": "validation",
    "max_samples": 32,
    "required_sample_count": 32,
    "steps": 20,
    "sampler": "euler",
    "seed": 42,
    "occ_threshold": 0.99,
    "target_threshold": 0.5,
    "variants": ["real"],
    "target_size": [64, 128, 128],
    "source_pc_range": [0.0, -20.0, -6.0, 120.0, 20.0, 10.0],
    "model_pc_range": [0.0, -20.0, -6.0, 40.0, 20.0, 10.0],
}


def sha256_file(path: str) -> str:
    """流式计算文件 SHA256。"""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_manifest_sha256(root: str) -> str:
    """按相对路径、大小和 mtime 计算数据集 manifest 指纹。"""
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


def validate_source_fingerprints(
    name: str,
    checkpoint: str,
    protocol: Dict[str, object],
) -> None:
    """拒绝权重或数据内容已变化的陈旧诊断结果。"""
    dataset_root = str(protocol.get("dataset_root", ""))
    vae_ckpt = str(protocol.get("vae_ckpt", ""))
    if not os.path.isfile(checkpoint):
        raise ValueError(f"候选 {name} checkpoint 不存在: {checkpoint}")
    if not os.path.isfile(vae_ckpt):
        raise ValueError(f"候选 {name} VAE 不存在: {vae_ckpt}")
    if not os.path.isdir(dataset_root):
        raise ValueError(f"候选 {name} dataset 不存在: {dataset_root}")
    actual = {
        "checkpoint_sha256": sha256_file(checkpoint),
        "vae_sha256": sha256_file(vae_ckpt),
        "dataset_manifest_sha256": dataset_manifest_sha256(dataset_root),
    }
    for key, value in actual.items():
        if protocol.get(key) != value:
            raise ValueError(f"候选 {name} source hash 不匹配: {key}")


def _finite_float(row: Dict[str, str], key: str, path: str) -> float:
    """读取有限浮点指标，错误时保留来源上下文。"""
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path} 缺少有效指标 {key}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{path} 指标 {key} 不是有限值: {value}")
    return value


def load_candidate(name: str, checkpoint: str, summary_csv: str) -> Dict[str, object]:
    """从 real-only 或完整 IR 消融 summary 中加载一个候选。"""
    with open(summary_csv, newline="", encoding="utf-8") as handle:
        real_rows = [row for row in csv.DictReader(handle) if row.get("variant") == "real"]
    if len(real_rows) != 1:
        raise ValueError(f"{summary_csv} 必须且只能包含一行 real summary")
    metadata_path = os.path.join(os.path.dirname(summary_csv), "ir_condition_ablation.json")
    with open(metadata_path, encoding="utf-8") as handle:
        metadata = json.load(handle)
    expected_checkpoint = os.path.realpath(checkpoint)
    recorded_checkpoint = os.path.realpath(str(metadata.get("model_ckpt", "")))
    if recorded_checkpoint != expected_checkpoint:
        raise ValueError(
            f"checkpoint 来源不匹配: expected={expected_checkpoint}, recorded={recorded_checkpoint}"
        )
    protocol = metadata.get("protocol")
    if not isinstance(protocol, dict) or not protocol:
        raise ValueError(f"{metadata_path} 缺少固定 validation protocol")
    validate_source_fingerprints(name, expected_checkpoint, protocol)
    row = real_rows[0]
    return {
        "name": str(name),
        "checkpoint": os.path.abspath(checkpoint),
        "summary_csv": os.path.abspath(summary_csv),
        "protocol": protocol,
        "bev_iou": _finite_float(row, "bev_iou", summary_csv),
        "bev_recall": _finite_float(row, "bev_recall", summary_csv),
        "top_height_recall": _finite_float(row, "top_height_recall", summary_csv),
        "trunk_region_recall": _finite_float(row, "trunk_region_recall", summary_csv),
        "vertical_connectivity_recall": _finite_float(
            row, "vertical_connectivity_recall", summary_csv
        ),
        "voxel_iou": _finite_float(row, "voxel_iou", summary_csv),
        "pred_to_target_ratio": _finite_float(row, "pred_to_target_ratio", summary_csv),
    }


def validate_common_protocol(candidates: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """确保全部候选来自完全相同的验证协议。"""
    if not candidates:
        raise ValueError("checkpoint candidates 不能为空")
    reference = candidates[0].get("protocol")
    if not isinstance(reference, dict) or not reference:
        raise ValueError(f"候选 {candidates[0].get('name')} 缺少 protocol")
    for key, expected in EXPECTED_FIXED_PROTOCOL.items():
        if reference.get(key) != expected:
            raise ValueError(
                f"candidate does not use fixed protocol: {key}={reference.get(key)!r}, "
                f"expected={expected!r}"
            )
    for source_key in ("dataset_root", "vae_ckpt"):
        source = reference.get(source_key)
        if not isinstance(source, str) or not os.path.isabs(source):
            raise ValueError(f"fixed protocol 缺少规范化来源 {source_key}: {source!r}")
    sample_indices = reference.get("sample_indices")
    if not isinstance(sample_indices, list) or len(sample_indices) != 32:
        raise ValueError("fixed protocol 必须记录恰好 32 个 sample_indices")
    if len(set(sample_indices)) != 32:
        raise ValueError("fixed protocol 的 sample_indices 必须互不重复")
    common_reference = {
        key: value for key, value in reference.items() if key != "checkpoint_sha256"
    }
    for candidate in candidates[1:]:
        candidate_protocol = candidate.get("protocol")
        common_candidate = (
            {key: value for key, value in candidate_protocol.items() if key != "checkpoint_sha256"}
            if isinstance(candidate_protocol, dict)
            else candidate_protocol
        )
        if common_candidate != common_reference:
            raise ValueError(
                f"candidate protocol 不一致: {candidates[0].get('name')} vs {candidate.get('name')}"
            )
    return common_reference


def _gate_status(candidate: Dict[str, object], gates: Dict[str, float]) -> Dict[str, object]:
    """计算各门槛通过状态与最差归一化满足度。"""
    ratio = float(candidate["pred_to_target_ratio"])
    if ratio <= 0.0:
        raise ValueError(f"候选 {candidate['name']} 的 pred_to_target_ratio 必须大于 0")
    satisfaction = {
        "bev_iou": float(candidate["bev_iou"]) / gates["bev_iou"],
        "bev_recall": float(candidate["bev_recall"]) / gates["bev_recall"],
        "top_height_recall": float(candidate["top_height_recall"]) / gates["top_height_recall"],
        "trunk_region_recall": float(candidate["trunk_region_recall"]) / gates["trunk_region_recall"],
        "pred_to_target_ratio": gates["max_pred_to_target_ratio"] / ratio,
    }
    passed = {key: value >= 1.0 for key, value in satisfaction.items()}
    return {
        "passed": passed,
        "pass_count": sum(passed.values()),
        "minimum_satisfaction": min(satisfaction.values()),
        "satisfaction": satisfaction,
        "gate_satisfied": all(passed.values()),
    }


def select_checkpoint(
    candidates: Sequence[Dict[str, object]],
    gates: Dict[str, float] = None,
) -> Dict[str, object]:
    """优先选择过门槛候选；无可行项时选择最平衡的候选。"""
    if not candidates:
        raise ValueError("checkpoint candidates 不能为空")
    effective_gates = dict(DEFAULT_GATES if gates is None else gates)
    evaluated: List[Dict[str, object]] = []
    for raw in candidates:
        item = dict(raw)
        item["gate"] = _gate_status(item, effective_gates)
        evaluated.append(item)

    feasible = [item for item in evaluated if item["gate"]["gate_satisfied"]]
    if feasible:
        selected = max(
            feasible,
            key=lambda item: (
                item["bev_iou"], item["top_height_recall"],
                item["trunk_region_recall"], item["bev_recall"],
                -abs(item["pred_to_target_ratio"] - 1.0),
            ),
        )
        reason = "selected highest-BEV-IoU candidate among candidates passing every gate"
    else:
        selected = max(
            evaluated,
            key=lambda item: (
                item["gate"]["minimum_satisfaction"],
                item["gate"]["pass_count"], item["bev_iou"],
                item["top_height_recall"], item["trunk_region_recall"],
            ),
        )
        reason = "no candidate passed every gate; selected maximum worst normalized gate satisfaction"
    return {
        "selected": selected,
        "gate_satisfied": bool(selected["gate"]["gate_satisfied"]),
        "reason": reason,
        "gates": effective_gates,
        "candidates": evaluated,
    }


def validate_gates(gates: Dict[str, float]) -> Dict[str, float]:
    """验证门槛均为有限正数，并返回普通 float 副本。"""
    validated = {}
    for key, raw in gates.items():
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f"checkpoint gate 必须为 finite 数值: {key}={value}")
        if value <= 0.0:
            raise ValueError(f"checkpoint gate 必须大于 0: {key}={value}")
        validated[key] = value
    return validated


def _write_report(path: str, result: Dict[str, object]) -> None:
    """写出可审计的 checkpoint 排名与门槛报告。"""
    lines = [
        "# LDM Validation Checkpoint Selection",
        "",
        f"- selected: `{result['selected']['name']}`",
        f"- checkpoint: `{result['selected']['checkpoint']}`",
        f"- all gates passed: `{result['gate_satisfied']}`",
        f"- reason: {result['reason']}",
        "",
        "| Candidate | BEV IoU | BEV Recall | Top | Trunk | Connectivity | Ratio | Worst gate | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in result["candidates"]:
        lines.append(
            f"| {item['name']} | {item['bev_iou']:.4f} | {item['bev_recall']:.4f} | "
            f"{item['top_height_recall']:.4f} | {item['trunk_region_recall']:.4f} | "
            f"{item['vertical_connectivity_recall']:.4f} | {item['pred_to_target_ratio']:.4f} | "
            f"{item['gate']['minimum_satisfaction']:.4f} | {item['gate']['pass_count']}/5 |"
        )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Select LDM checkpoint by validation task/structure metrics")
    parser.add_argument(
        "--candidate", action="append", nargs=3, metavar=("NAME", "CHECKPOINT", "SUMMARY_CSV"),
        required=True,
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_bev_iou", type=float, default=DEFAULT_GATES["bev_iou"])
    parser.add_argument("--min_bev_recall", type=float, default=DEFAULT_GATES["bev_recall"])
    parser.add_argument("--min_top_recall", type=float, default=DEFAULT_GATES["top_height_recall"])
    parser.add_argument("--min_trunk_recall", type=float, default=DEFAULT_GATES["trunk_region_recall"])
    parser.add_argument("--max_count_ratio", type=float, default=DEFAULT_GATES["max_pred_to_target_ratio"])
    args = parser.parse_args()

    candidates = [load_candidate(*values) for values in args.candidate]
    protocol = validate_common_protocol(candidates)
    gates = {
        "bev_iou": args.min_bev_iou,
        "bev_recall": args.min_bev_recall,
        "top_height_recall": args.min_top_recall,
        "trunk_region_recall": args.min_trunk_recall,
        "max_pred_to_target_ratio": args.max_count_ratio,
    }
    gates = validate_gates(gates)
    result = select_checkpoint(candidates, gates)
    result["protocol"] = protocol
    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "checkpoint_selection.json")
    report_path = os.path.join(args.output_dir, "checkpoint_selection_report.md")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
    _write_report(report_path, result)
    print(f"Selected checkpoint: {result['selected']['checkpoint']}")
    print(f"All gates passed: {result['gate_satisfied']}")
    print(f"Saved checkpoint selection to: {json_path}")
    print(f"Saved checkpoint report to: {report_path}")


if __name__ == "__main__":
    main()
