# -*- coding: utf-8 -*-
"""训练期 occupancy 阈值扫描、选择和 checkpoint 绑定 artifact 合同。"""

import hashlib
import json
import math
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch


THRESHOLD_SWEEP_PROTOCOL = "observed_micro_occupancy_threshold_sweep_v1"
THRESHOLD_ARTIFACT_PROTOCOL = "occupancy_threshold_validation_artifact_v1"
THRESHOLD_SELECTION_RULE = "max_iou_then_max_recall_then_lower_threshold_v1"
DEFAULT_THRESHOLD_CANDIDATES = (
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
)


def sha256_file(path: str) -> str:
    """流式计算 artifact/checkpoint 文件 SHA-256。"""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_threshold_candidates(values: Iterable[Any]) -> tuple:
    """返回严格递增、位于 (0,1) 的候选阈值。"""
    if isinstance(values, (str, bytes)):
        raise ValueError("threshold candidates 必须是数值序列")
    try:
        candidates = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("threshold candidates 必须是数值序列") from exc
    if not candidates:
        raise ValueError("threshold candidates 不能为空")
    if any(not math.isfinite(value) or not 0.0 < value < 1.0 for value in candidates):
        raise ValueError("threshold candidates 必须是 (0,1) 内有限数")
    if any(right <= left for left, right in zip(candidates, candidates[1:])):
        raise ValueError("threshold candidates 必须严格递增且不能重复")
    return candidates


def threshold_sweep_batch_counts(
    probability: torch.Tensor,
    target: torch.Tensor,
    observed_mask: Optional[torch.Tensor],
    candidates: Sequence[float],
) -> Dict[str, int]:
    """计算一个 batch 在各候选阈值上的 observed-domain TP/FP/FN。"""
    candidates = validate_threshold_candidates(candidates)
    if probability.shape != target.shape:
        raise ValueError("threshold sweep probability/target shape 不一致")
    if not torch.isfinite(probability).all():
        raise ValueError("threshold sweep probability 含非有限数")
    truth = target >= 0.5
    observed = None
    if observed_mask is not None:
        observed = observed_mask.to(device=target.device, dtype=torch.bool)
        if observed.shape != target.shape:
            observed = observed.expand_as(target)
        truth = truth & observed
    counts: Dict[str, int] = {}
    for index, threshold in enumerate(candidates):
        prediction = probability >= threshold
        if observed is not None:
            prediction = prediction & observed
        counts[f"threshold_{index:03d}_tp"] = int((prediction & truth).sum().item())
        counts[f"threshold_{index:03d}_fp"] = int((prediction & ~truth).sum().item())
        counts[f"threshold_{index:03d}_fn"] = int((~prediction & truth).sum().item())
    return counts


def threshold_sweep_metrics(
    counts: Mapping[str, Any],
    candidates: Sequence[float],
) -> List[Dict[str, Any]]:
    """把可跨 rank 累加的 TP/FP/FN 转成稳定指标记录。"""
    records = []
    for index, threshold in enumerate(validate_threshold_candidates(candidates)):
        values = {}
        for name in ("tp", "fp", "fn"):
            raw = counts.get(f"threshold_{index:03d}_{name}")
            value = int(raw) if raw is not None else -1
            if value < 0 or float(value) != float(raw):
                raise ValueError("threshold sweep counts 必须是非负整数")
            values[name] = value
        tp, fp, fn = values["tp"], values["fp"], values["fn"]
        union = tp + fp + fn
        precision_denominator = tp + fp
        recall_denominator = tp + fn
        precision = tp / precision_denominator if precision_denominator else 0.0
        recall = tp / recall_denominator if recall_denominator else 0.0
        f1_denominator = 2 * tp + fp + fn
        records.append({
            "threshold": float(threshold),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "iou": tp / union if union else 0.0,
            "precision": precision,
            "recall": recall,
            "f1": (2 * tp / f1_denominator if f1_denominator else 0.0),
        })
    return records


def select_threshold_record(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """按 IoU、recall、较低阈值的固定顺序选择 validation 阈值。"""
    if not isinstance(records, Sequence) or not records:
        raise ValueError("threshold metrics 不能为空")
    validated = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("threshold metric record 必须是映射")
        threshold = float(record.get("threshold"))
        iou = float(record.get("iou"))
        recall = float(record.get("recall"))
        if (
            not math.isfinite(threshold)
            or not 0.0 < threshold < 1.0
            or not math.isfinite(iou)
            or not 0.0 <= iou <= 1.0
            or not math.isfinite(recall)
            or not 0.0 <= recall <= 1.0
        ):
            raise ValueError("threshold metric record 含非法 threshold/IoU/recall")
        validated.append(dict(record))
    return dict(max(validated, key=lambda item: (item["iou"], item["recall"], -item["threshold"])))


def validate_checkpoint_threshold_sweep(
    checkpoint: Mapping[str, Any],
    *,
    expected_stage: str,
    expected_weight_source: str,
    expected_candidates: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """严格验证 checkpoint 内嵌的 validation-only threshold sweep。"""
    sweep = checkpoint.get("occupancy_threshold_validation")
    if not isinstance(sweep, Mapping) or sweep.get("protocol") != THRESHOLD_SWEEP_PROTOCOL:
        raise ValueError("checkpoint 缺少正式 occupancy threshold validation sweep")
    if checkpoint.get("stage") != expected_stage:
        raise ValueError("checkpoint threshold validation stage 不匹配")
    if sweep.get("deployment_weight_source") != expected_weight_source:
        raise ValueError("checkpoint threshold validation 权重来源不匹配")
    candidates = validate_threshold_candidates(sweep.get("candidate_thresholds", ()))
    if expected_candidates is not None and candidates != validate_threshold_candidates(expected_candidates):
        raise ValueError("checkpoint threshold candidates 与当前配置不一致")
    records = sweep.get("metrics_by_threshold")
    if not isinstance(records, list) or len(records) != len(candidates):
        raise ValueError("checkpoint threshold metrics 数量不匹配")
    if [float(item.get("threshold")) for item in records] != list(candidates):
        raise ValueError("checkpoint threshold metrics 与 candidates 顺序不一致")
    # 选择函数同时验证 threshold/IoU/recall 的数值域。
    select_threshold_record(records)
    if sweep.get("split") != "temporal_block_validation_suffix":
        raise ValueError("checkpoint threshold validation split 不匹配")
    if sweep.get("observation_domain") != "persisted_observed_mask_v1":
        raise ValueError("checkpoint threshold observation domain 不匹配")
    return dict(sweep)


def build_threshold_artifact(
    checkpoint: Mapping[str, Any],
    *,
    checkpoint_path: str,
) -> Dict[str, Any]:
    """从 checkpoint 内嵌的 validation sweep 构造独立绑定 artifact。"""
    stage = checkpoint.get("stage")
    expected_weight_source = (
        checkpoint.get("deployment_weight_source")
        if stage == "cd"
        else "model_state_dict"
    )
    sweep = validate_checkpoint_threshold_sweep(
        checkpoint,
        expected_stage=stage,
        expected_weight_source=expected_weight_source,
    )
    records = sweep.get("metrics_by_threshold")
    selected = select_threshold_record(records)
    candidates = validate_threshold_candidates(sweep.get("candidate_thresholds", ()))
    if [float(item["threshold"]) for item in records] != list(candidates):
        raise ValueError("threshold sweep metrics 与 candidate_thresholds 不一致")
    if stage not in {"ldm", "cd"}:
        raise ValueError("threshold artifact 只支持 LDM/CD checkpoint")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    return {
        "schema_version": 1,
        "protocol": THRESHOLD_ARTIFACT_PROTOCOL,
        "formal": True,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_protocol": checkpoint.get("checkpoint_protocol"),
        "stage": stage,
        "deployment_weight_source": sweep.get("deployment_weight_source"),
        "validation_split": sweep.get("split"),
        "validation_data_protocol": checkpoint.get("data_protocol"),
        "stage_training_selection": checkpoint.get("stage_training_selection"),
        "observation_domain": sweep.get("observation_domain"),
        "candidate_thresholds": list(candidates),
        "selection_rule": THRESHOLD_SELECTION_RULE,
        "selected_threshold": float(selected["threshold"]),
        "selected_metrics": selected,
        "metrics_by_threshold": [dict(record) for record in records],
    }


def validate_threshold_artifact(
    artifact: Mapping[str, Any],
    *,
    checkpoint_path: str,
    expected_stage: str,
    expected_weight_source: str,
) -> Dict[str, Any]:
    """验证 artifact 的协议、checkpoint hash、stage 和部署权重来源。"""
    if not isinstance(artifact, Mapping):
        raise ValueError("occupancy threshold artifact 必须是 JSON object")
    if artifact.get("protocol") != THRESHOLD_ARTIFACT_PROTOCOL or artifact.get("formal") is not True:
        raise ValueError("occupancy threshold artifact 协议不匹配")
    if artifact.get("checkpoint_sha256") != sha256_file(checkpoint_path):
        raise ValueError("occupancy threshold artifact checkpoint SHA-256 不匹配")
    if artifact.get("stage") != expected_stage:
        raise ValueError("occupancy threshold artifact stage 不匹配")
    if artifact.get("deployment_weight_source") != expected_weight_source:
        raise ValueError("occupancy threshold artifact 部署权重来源不匹配")
    records = artifact.get("metrics_by_threshold")
    selected = select_threshold_record(records)
    if float(artifact.get("selected_threshold")) != float(selected["threshold"]):
        raise ValueError("occupancy threshold artifact 选择结果不一致")
    if artifact.get("selection_rule") != THRESHOLD_SELECTION_RULE:
        raise ValueError("occupancy threshold artifact selection rule 不匹配")
    return dict(artifact)


def load_threshold_artifact(
    path: str,
    **validation_kwargs,
) -> tuple:
    """加载并验证 threshold artifact，同时返回原文件 SHA-256。"""
    with open(path, "r", encoding="utf-8") as handle:
        artifact = json.load(handle)
    return (
        validate_threshold_artifact(artifact, **validation_kwargs),
        sha256_file(path),
    )


def write_threshold_artifact(path: str, artifact: Mapping[str, Any]) -> None:
    """同目录原子发布 threshold artifact。"""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    temporary = f"{path}.tmp-{os.getpid()}"
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(dict(artifact), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
