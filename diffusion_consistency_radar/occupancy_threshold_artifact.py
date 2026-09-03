# -*- coding: utf-8 -*-
"""训练期 occupancy 阈值扫描、选择和 checkpoint 绑定 artifact 合同。"""

import hashlib
import json
import math
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch


THRESHOLD_SWEEP_PROTOCOL = "observed_micro_occupancy_threshold_sweep_v2"
THRESHOLD_SAMPLING_PROTOCOL = "full_deployment_sampling_validation_v1"
THRESHOLD_ARTIFACT_PROTOCOL = "occupancy_threshold_validation_artifact_v2"
THRESHOLD_SELECTION_RULE = (
    "declared_min_recall_then_max_iou_then_max_recall_then_lower_threshold_v2"
)
THRESHOLD_RECALL_CONSTRAINT_PROTOCOL = (
    "declared_validation_occupied_recall_constraint_v1"
)
THRESHOLD_RECALL_CONSTRAINT_SCOPE = (
    "validation_occupied_recall_only_not_flight_safety_guarantee"
)
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


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
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


def resolve_threshold_recall_constraint(
    min_occupied_recall: Any = None,
    authority: Any = None,
) -> Dict[str, Any]:
    """解析可选 recall 门槛；无权威来源时显式保持未启用。"""
    authority_text = str(authority or "").strip()
    if min_occupied_recall is None:
        if authority_text:
            raise ValueError("未设置 min_occupied_recall 时不得声明 authority")
        return {
            "protocol": THRESHOLD_RECALL_CONSTRAINT_PROTOCOL,
            "enabled": False,
            "min_occupied_recall": None,
            "authority": None,
            "scope": THRESHOLD_RECALL_CONSTRAINT_SCOPE,
        }
    if isinstance(min_occupied_recall, bool):
        raise ValueError("min_occupied_recall 必须是 [0,1] 内有限数")
    minimum = float(min_occupied_recall)
    if not math.isfinite(minimum) or not 0.0 <= minimum <= 1.0:
        raise ValueError("min_occupied_recall 必须是 [0,1] 内有限数")
    if not authority_text:
        raise ValueError("启用 min_occupied_recall 必须声明 authority")
    return {
        "protocol": THRESHOLD_RECALL_CONSTRAINT_PROTOCOL,
        "enabled": True,
        "min_occupied_recall": minimum,
        "authority": authority_text,
        "scope": THRESHOLD_RECALL_CONSTRAINT_SCOPE,
    }


def validate_threshold_recall_constraint(value: Any) -> Dict[str, Any]:
    """严格校验 checkpoint/artifact 中的 recall 约束收据。"""
    if not isinstance(value, Mapping) or set(value) != {
        "protocol",
        "enabled",
        "min_occupied_recall",
        "authority",
        "scope",
    }:
        raise ValueError("threshold recall constraint 字段不完整")
    if value.get("protocol") != THRESHOLD_RECALL_CONSTRAINT_PROTOCOL:
        raise ValueError("threshold recall constraint 协议不匹配")
    if value.get("scope") != THRESHOLD_RECALL_CONSTRAINT_SCOPE:
        raise ValueError("threshold recall constraint scope 不匹配")
    expected = resolve_threshold_recall_constraint(
        value.get("min_occupied_recall"),
        value.get("authority"),
    )
    if bool(value.get("enabled")) != expected["enabled"] or dict(value) != expected:
        raise ValueError("threshold recall constraint 内容不一致")
    return expected


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


def _validate_threshold_metric_records(
    records: Sequence[Mapping[str, Any]],
) -> None:
    """复算正式 sweep 的计数派生指标，拒绝内部不一致的收据。"""
    for record in records:
        try:
            tp = int(record["tp"])
            fp = int(record["fp"])
            fn = int(record["fn"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("threshold metric 缺少有效 TP/FP/FN") from exc
        count_values = (("tp", tp), ("fp", fp), ("fn", fn))
        if any(value < 0 for _, value in count_values):
            raise ValueError("threshold metric TP/FP/FN 必须是非负整数")
        if any(float(record[name]) != value for name, value in count_values):
            raise ValueError("threshold metric TP/FP/FN 必须是非负整数")
        union = tp + fp + fn
        precision_denominator = tp + fp
        recall_denominator = tp + fn
        f1_denominator = 2 * tp + fp + fn
        expected = {
            "iou": tp / union if union else 0.0,
            "precision": (
                tp / precision_denominator if precision_denominator else 0.0
            ),
            "recall": tp / recall_denominator if recall_denominator else 0.0,
            "f1": 2 * tp / f1_denominator if f1_denominator else 0.0,
        }
        for name, expected_value in expected.items():
            try:
                actual = float(record[name])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"threshold metric 缺少有效 {name}") from exc
            if not math.isfinite(actual) or not math.isclose(
                actual,
                expected_value,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"threshold metric {name} 与 TP/FP/FN 不一致")


def select_threshold_record(
    records: Sequence[Mapping[str, Any]],
    *,
    min_occupied_recall: Optional[float] = None,
) -> Dict[str, Any]:
    """先满足显式 occupied recall 门槛，再按 IoU/recall/低阈值选优。"""
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
    if min_occupied_recall is not None:
        if isinstance(min_occupied_recall, bool):
            raise ValueError("min_occupied_recall 必须是 [0,1] 内有限数")
        minimum = float(min_occupied_recall)
        if not math.isfinite(minimum) or not 0.0 <= minimum <= 1.0:
            raise ValueError("min_occupied_recall 必须是 [0,1] 内有限数")
        validated = [
            record for record in validated if float(record["recall"]) >= minimum
        ]
        if not validated:
            raise ValueError("没有 threshold candidate 满足 min_occupied_recall")
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
    _validate_threshold_metric_records(records)
    constraints = validate_threshold_recall_constraint(
        sweep.get("selection_constraints")
    )
    # 选择函数同时验证 threshold/IoU/recall 的数值域，并确保声明的召回门槛可满足。
    select_threshold_record(
        records,
        min_occupied_recall=constraints["min_occupied_recall"],
    )
    if sweep.get("split") != "temporal_block_validation_suffix":
        raise ValueError("checkpoint threshold validation split 不匹配")
    if sweep.get("observation_domain") != "persisted_observed_mask_v1":
        raise ValueError("checkpoint threshold observation domain 不匹配")
    if sweep.get("sampling_protocol") != THRESHOLD_SAMPLING_PROTOCOL:
        raise ValueError("checkpoint threshold 必须来自完整 deployment sampling")
    if not _is_sha256(sweep.get("deployment_validation_selection_sha256")):
        raise ValueError("checkpoint threshold 缺少 deployment validation 子集身份")
    return dict(sweep)


def build_threshold_artifact(
    checkpoint: Mapping[str, Any],
    *,
    checkpoint_path: str,
) -> Dict[str, Any]:
    """从 checkpoint 内嵌的 validation sweep 构造独立绑定 artifact。"""
    stage = checkpoint.get("stage")
    if stage not in {"ldm", "cd"}:
        raise ValueError("threshold artifact 只支持 LDM/CD checkpoint")
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
    constraints = validate_threshold_recall_constraint(
        sweep.get("selection_constraints")
    )
    selected = select_threshold_record(
        records,
        min_occupied_recall=constraints["min_occupied_recall"],
    )
    candidates = validate_threshold_candidates(sweep.get("candidate_thresholds", ()))
    if [float(item["threshold"]) for item in records] != list(candidates):
        raise ValueError("threshold sweep metrics 与 candidate_thresholds 不一致")
    stage_validation = checkpoint.get(f"{stage}_validation")
    selection = (
        stage_validation.get("selection")
        if isinstance(stage_validation, Mapping)
        else None
    )
    selection_sha256 = (
        selection.get("selection_sha256")
        if isinstance(selection, Mapping)
        else None
    )
    if selection_sha256 != sweep["deployment_validation_selection_sha256"]:
        raise ValueError(
            "threshold sweep 与 deployment validation 子集身份不一致"
        )
    checkpoint_sha256 = sha256_file(checkpoint_path)
    return {
        "schema_version": 2,
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
        "sampling_protocol": sweep.get("sampling_protocol"),
        "deployment_validation_selection_sha256": selection_sha256,
        "candidate_thresholds": list(candidates),
        "selection_rule": THRESHOLD_SELECTION_RULE,
        "selection_constraints": constraints,
        "safety_recall_qualification": {
            "qualified": bool(constraints["enabled"]),
            "status": (
                "validation_recall_constraint_satisfied"
                if constraints["enabled"]
                else "not_qualified_no_authoritative_minimum"
            ),
            "scope": THRESHOLD_RECALL_CONSTRAINT_SCOPE,
        },
        "selected_threshold": float(selected["threshold"]),
        "selected_metrics": selected,
        "metrics_by_threshold": [dict(record) for record in records],
    }


def validate_threshold_artifact_metadata(
    artifact: Mapping[str, Any],
) -> Dict[str, Any]:
    """验证可嵌入推理收据的 threshold artifact 自包含合同。"""
    if not isinstance(artifact, Mapping):
        raise ValueError("occupancy threshold artifact 必须是 JSON object")
    if (
        artifact.get("schema_version") != 2
        or artifact.get("protocol") != THRESHOLD_ARTIFACT_PROTOCOL
        or artifact.get("formal") is not True
    ):
        raise ValueError("occupancy threshold artifact 协议不匹配")
    if not _is_sha256(artifact.get("checkpoint_sha256")):
        raise ValueError("occupancy threshold artifact checkpoint SHA-256 非法")
    stage = artifact.get("stage")
    weight_source = artifact.get("deployment_weight_source")
    if stage not in {"ldm", "cd"}:
        raise ValueError("occupancy threshold artifact stage 不支持")
    if weight_source not in {"model_state_dict", "ema_model_state_dict"}:
        raise ValueError("occupancy threshold artifact 部署权重来源不支持")
    if stage == "ldm" and weight_source != "model_state_dict":
        raise ValueError("LDM threshold artifact 部署权重来源不匹配")
    candidates = validate_threshold_candidates(
        artifact.get("candidate_thresholds", ())
    )
    if artifact.get("checkpoint_protocol") != "formal_chain_v2":
        raise ValueError("occupancy threshold artifact checkpoint protocol 不匹配")
    if artifact.get("validation_split") != "temporal_block_validation_suffix":
        raise ValueError("occupancy threshold artifact validation split 不匹配")
    if artifact.get("observation_domain") != "persisted_observed_mask_v1":
        raise ValueError("occupancy threshold artifact observation domain 不匹配")
    records = artifact.get("metrics_by_threshold")
    if not isinstance(records, list) or len(records) != len(candidates):
        raise ValueError("occupancy threshold artifact metrics 数量不匹配")
    if [float(item.get("threshold")) for item in records] != list(candidates):
        raise ValueError("occupancy threshold artifact metrics 顺序不匹配")
    _validate_threshold_metric_records(records)
    constraints = validate_threshold_recall_constraint(
        artifact.get("selection_constraints")
    )
    selected = select_threshold_record(
        records,
        min_occupied_recall=constraints["min_occupied_recall"],
    )
    if float(artifact.get("selected_threshold")) != float(selected["threshold"]):
        raise ValueError("occupancy threshold artifact 选择结果不一致")
    if artifact.get("selected_metrics") != selected:
        raise ValueError("occupancy threshold artifact selected metrics 不一致")
    if artifact.get("selection_rule") != THRESHOLD_SELECTION_RULE:
        raise ValueError("occupancy threshold artifact selection rule 不匹配")
    expected_qualification = {
        "qualified": bool(constraints["enabled"]),
        "status": (
            "validation_recall_constraint_satisfied"
            if constraints["enabled"]
            else "not_qualified_no_authoritative_minimum"
        ),
        "scope": THRESHOLD_RECALL_CONSTRAINT_SCOPE,
    }
    if artifact.get("safety_recall_qualification") != expected_qualification:
        raise ValueError("occupancy threshold artifact recall qualification 不一致")
    if artifact.get("sampling_protocol") != THRESHOLD_SAMPLING_PROTOCOL:
        raise ValueError("occupancy threshold artifact 不是完整 deployment sampling")
    if not _is_sha256(
        artifact.get("deployment_validation_selection_sha256")
    ):
        raise ValueError(
            "occupancy threshold artifact 缺少 deployment validation 子集身份"
        )
    return dict(artifact)


def validate_threshold_artifact(
    artifact: Mapping[str, Any],
    *,
    checkpoint_path: str,
    expected_stage: str,
    expected_weight_source: str,
) -> Dict[str, Any]:
    """验证 artifact 的自包含合同、checkpoint hash、stage 和权重来源。"""
    validated = validate_threshold_artifact_metadata(artifact)
    if validated.get("checkpoint_sha256") != sha256_file(checkpoint_path):
        raise ValueError("occupancy threshold artifact checkpoint SHA-256 不匹配")
    if validated.get("stage") != expected_stage:
        raise ValueError("occupancy threshold artifact stage 不匹配")
    if validated.get("deployment_weight_source") != expected_weight_source:
        raise ValueError("occupancy threshold artifact 部署权重来源不匹配")
    return validated


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
