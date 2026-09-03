#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""文件功能：定义 CD 的真实 EMA consistency 训练与部署采样配置协议。"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Dict

import torch


CD_CONSISTENCY_CONFIG_PROTOCOL = (
    "boundary_anchored_ema_consistency_training_config_v2"
)
CD_TRAINING_SEMANTICS = (
    "ldm_initialized_boundary_anchored_ema_consistency_v2"
)
CD_DENOISING_PARAMETERIZATION = "direct_x0_sigma_min_skip_boundary_v2"
CD_CONSISTENCY_TARGET_SOURCE = "cd_model_ema"
CD_CONSISTENCY_LOSS = "weighted_consistency_plus_observed_reconstruction_v2"
CD_BOUNDARY_PARAMETERIZATION = "sigma_min_squared_ratio_skip_v1"
CD_RECONSTRUCTION_ANCHOR = "persisted_observed_latent_target_mse_v1"
CD_COLLAPSE_DIAGNOSTIC_PROTOCOL = (
    "cd_condition_and_output_collapse_diagnostic_v1"
)
CD_COLLAPSE_DIAGNOSTIC_ROLE = "diagnostic_guard_not_checkpoint_selector"
CD_COLLAPSE_DIAGNOSTICS_PROTOCOL = "cd_weight_source_collapse_diagnostics_v1"

_RECEIPT_KEYS = frozenset(
    {
        "protocol",
        "training_semantics",
        "denoising_parameterization",
        "consistency_target_source",
        "loss",
        "num_scales",
        "ema_rate",
        "sigma_min",
        "sigma_max",
        "rho",
        "boundary_parameterization",
        "reconstruction_anchor",
        "consistency_loss_weight",
        "reconstruction_anchor_weight",
        "collapse_guard_epsilon",
    }
)

_COLLAPSE_RECEIPT_KEYS = frozenset(
    {
        "protocol",
        "role",
        "guard_epsilon",
        "output_variance",
        "inter_sample_mse",
        "inter_sample_pair_count",
        "condition_ablation_mse",
        "constant_output_status",
        "condition_sensitivity_status",
        "status",
    }
)


def resolve_cd_consistency_config(config: Mapping[str, Any] | None) -> Dict[str, Any]:
    """从 YAML/CD 配置解析当前 trainer 真正消费的超参数。"""
    if config is not None and not isinstance(config, Mapping):
        raise ValueError("CD consistency 配置必须是对象")
    source = dict(config or {})
    raw_num_scales = source.get("num_scales", 40)
    if type(raw_num_scales) is bool:
        raise ValueError("cd.num_scales 必须是整数")
    try:
        numeric_num_scales = float(raw_num_scales)
        if not math.isfinite(numeric_num_scales) or not numeric_num_scales.is_integer():
            raise ValueError("cd.num_scales 必须是整数")
        num_scales = int(numeric_num_scales)
        ema_rate = float(source.get("ema_rate", 0.999))
        sigma_min = float(source.get("sigma_min", 0.002))
        sigma_max = float(source.get("sigma_max", 80.0))
        rho = float(source.get("rho", 7.0))
        consistency_loss_weight = float(
            source.get("consistency_loss_weight", 1.0)
        )
        reconstruction_anchor_weight = float(
            source.get("reconstruction_anchor_weight", 0.1)
        )
        collapse_guard_epsilon = float(
            source.get("collapse_guard_epsilon", 0.0)
        )
    except (TypeError, ValueError) as exc:
        if str(exc) == "cd.num_scales 必须是整数":
            raise
        raise ValueError("CD consistency 超参数必须是数值") from exc
    if num_scales < 2:
        raise ValueError("cd.num_scales 必须至少为 2")
    if not math.isfinite(ema_rate) or not 0.0 <= ema_rate < 1.0:
        raise ValueError("cd.ema_rate 必须位于 [0,1)")
    if (
        not math.isfinite(sigma_min)
        or not math.isfinite(sigma_max)
        or sigma_min <= 0.0
        or sigma_max <= sigma_min
    ):
        raise ValueError("cd.sigma_min/max 必须是递增正有限数")
    if not math.isfinite(rho) or rho <= 0.0:
        raise ValueError("cd.rho 必须是正有限数")
    if not math.isfinite(consistency_loss_weight) or consistency_loss_weight <= 0.0:
        raise ValueError("cd.consistency_loss_weight 必须是正有限数")
    if (
        not math.isfinite(reconstruction_anchor_weight)
        or reconstruction_anchor_weight <= 0.0
    ):
        raise ValueError("cd.reconstruction_anchor_weight 必须是正有限数")
    if not math.isfinite(collapse_guard_epsilon) or collapse_guard_epsilon < 0.0:
        raise ValueError("cd.collapse_guard_epsilon 必须是非负有限数")
    semantics = source.get("training_semantics", CD_TRAINING_SEMANTICS)
    if semantics != CD_TRAINING_SEMANTICS:
        raise ValueError(
            f"cd.training_semantics 必须为 {CD_TRAINING_SEMANTICS!r}"
        )
    return {
        "protocol": CD_CONSISTENCY_CONFIG_PROTOCOL,
        "training_semantics": CD_TRAINING_SEMANTICS,
        "denoising_parameterization": CD_DENOISING_PARAMETERIZATION,
        "consistency_target_source": CD_CONSISTENCY_TARGET_SOURCE,
        "loss": CD_CONSISTENCY_LOSS,
        "num_scales": num_scales,
        "ema_rate": ema_rate,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "rho": rho,
        "boundary_parameterization": CD_BOUNDARY_PARAMETERIZATION,
        "reconstruction_anchor": CD_RECONSTRUCTION_ANCHOR,
        "consistency_loss_weight": consistency_loss_weight,
        "reconstruction_anchor_weight": reconstruction_anchor_weight,
        "collapse_guard_epsilon": collapse_guard_epsilon,
    }


def validate_cd_consistency_receipt(value: Any) -> Dict[str, Any]:
    """严格校验 checkpoint 中的训练/采样收据，不接受缺字段或额外字段。"""
    if not isinstance(value, Mapping):
        raise ValueError("consistency_training_config 必须是对象")
    if set(value) != _RECEIPT_KEYS:
        raise ValueError(
            "consistency_training_config 字段必须精确为 "
            f"{sorted(_RECEIPT_KEYS)}"
        )
    resolved = resolve_cd_consistency_config(value)
    if dict(value) != resolved:
        raise ValueError("consistency_training_config 值与 EMA consistency 协议不一致")
    return resolved


def apply_cd_boundary_parameterization(
    raw_output: torch.Tensor,
    x_t: torch.Tensor,
    sigmas: torch.Tensor,
    config: Mapping[str, Any],
) -> torch.Tensor:
    """施加 sigma-min 硬边界，同时在高噪声端保持 LDM 初始化输出。"""
    if raw_output.shape != x_t.shape:
        raise ValueError("CD boundary raw_output/x_t shape 不一致")
    if sigmas.ndim != 1 or sigmas.shape[0] != x_t.shape[0]:
        raise ValueError("CD boundary sigma 数量与 batch 不一致")
    if config.get("boundary_parameterization") != CD_BOUNDARY_PARAMETERIZATION:
        raise ValueError("CD boundary parameterization 协议不匹配")
    sigma_min = float(config.get("sigma_min"))
    sigma_values = sigmas.to(device=x_t.device, dtype=x_t.dtype)
    if not torch.isfinite(sigma_values).all() or bool(
        (sigma_values < sigma_min).any().item()
    ):
        raise ValueError("CD boundary sigma 必须为不小于 sigma_min 的有限数")
    skip = torch.clamp(
        (sigma_min / sigma_values) ** 2,
        min=0.0,
        max=1.0,
    ).view(-1, 1, 1, 1, 1)
    return skip * x_t + (1.0 - skip) * raw_output


def build_cd_collapse_diagnostic(
    *,
    output_variance: Any,
    inter_sample_mse: Any,
    inter_sample_pair_count: Any,
    condition_ablation_mse: Any,
    guard_epsilon: Any,
) -> Dict[str, Any]:
    """构造常数输出与条件忽略的审计收据；不参与质量选优。"""
    try:
        output_variance = float(output_variance)
        condition_ablation_mse = float(condition_ablation_mse)
        guard_epsilon = float(guard_epsilon)
    except (TypeError, ValueError) as exc:
        raise ValueError("CD collapse diagnostic 必须是数值") from exc
    if type(inter_sample_pair_count) is not int or inter_sample_pair_count < 0:
        raise ValueError("CD collapse diagnostic pair count 必须是非负整数")
    if inter_sample_pair_count == 0:
        if inter_sample_mse is not None:
            raise ValueError("无跨样本 pair 时 inter_sample_mse 必须为 null")
        resolved_inter_sample_mse = None
    else:
        try:
            resolved_inter_sample_mse = float(inter_sample_mse)
        except (TypeError, ValueError) as exc:
            raise ValueError("CD collapse diagnostic inter_sample_mse 无效") from exc
    numeric_values = [output_variance, condition_ablation_mse, guard_epsilon]
    if resolved_inter_sample_mse is not None:
        numeric_values.append(resolved_inter_sample_mse)
    if any(not math.isfinite(value) or value < 0.0 for value in numeric_values):
        raise ValueError("CD collapse diagnostic 指标必须是非负有限数")

    condition_pass = condition_ablation_mse > guard_epsilon
    sample_difference_pass = (
        resolved_inter_sample_mse is not None
        and resolved_inter_sample_mse > guard_epsilon
    )
    constant_pass = condition_pass or sample_difference_pass
    overall_pass = constant_pass and condition_pass
    return {
        "protocol": CD_COLLAPSE_DIAGNOSTIC_PROTOCOL,
        "role": CD_COLLAPSE_DIAGNOSTIC_ROLE,
        "guard_epsilon": guard_epsilon,
        "output_variance": output_variance,
        "inter_sample_mse": resolved_inter_sample_mse,
        "inter_sample_pair_count": inter_sample_pair_count,
        "condition_ablation_mse": condition_ablation_mse,
        "constant_output_status": "pass" if constant_pass else "fail",
        "condition_sensitivity_status": "pass" if condition_pass else "fail",
        "status": "pass" if overall_pass else "fail",
    }


def validate_cd_collapse_diagnostic(value: Any) -> Dict[str, Any]:
    """严格复算 collapse diagnostic 状态，拒绝手工篡改状态字段。"""
    if not isinstance(value, Mapping) or set(value) != _COLLAPSE_RECEIPT_KEYS:
        raise ValueError("cd_collapse_diagnostic 字段不完整")
    if value.get("protocol") != CD_COLLAPSE_DIAGNOSTIC_PROTOCOL:
        raise ValueError("cd_collapse_diagnostic 协议不匹配")
    if value.get("role") != CD_COLLAPSE_DIAGNOSTIC_ROLE:
        raise ValueError("cd_collapse_diagnostic role 不匹配")
    expected = build_cd_collapse_diagnostic(
        output_variance=value.get("output_variance"),
        inter_sample_mse=value.get("inter_sample_mse"),
        inter_sample_pair_count=value.get("inter_sample_pair_count"),
        condition_ablation_mse=value.get("condition_ablation_mse"),
        guard_epsilon=value.get("guard_epsilon"),
    )
    if dict(value) != expected:
        raise ValueError("cd_collapse_diagnostic 状态与指标不一致")
    return expected


def build_cd_collapse_diagnostics_receipt(
    *,
    selected_source: str,
    diagnostics: Mapping[str, Any],
) -> Dict[str, Any]:
    """绑定 online/EMA 防塌缩诊断，并要求实际部署权重通过门禁。"""
    expected_sources = {"model_state_dict", "ema_model_state_dict"}
    if selected_source not in expected_sources:
        raise ValueError("CD collapse diagnostics selected_source 无效")
    if not isinstance(diagnostics, Mapping) or set(diagnostics) != expected_sources:
        raise ValueError("CD collapse diagnostics 必须同时包含 online/EMA")
    resolved = {
        source: validate_cd_collapse_diagnostic(diagnostics[source])
        for source in sorted(expected_sources)
    }
    if resolved[selected_source]["status"] != "pass":
        raise ValueError("CD 实际部署权重未通过防塌缩诊断")
    return {
        "protocol": CD_COLLAPSE_DIAGNOSTICS_PROTOCOL,
        "selected_source": selected_source,
        "diagnostics": resolved,
    }


def validate_cd_collapse_diagnostics_receipt(value: Any) -> Dict[str, Any]:
    """严格校验 checkpoint 的权重源防塌缩诊断收据。"""
    if not isinstance(value, Mapping) or set(value) != {
        "protocol",
        "selected_source",
        "diagnostics",
    }:
        raise ValueError("cd_collapse_diagnostics 字段不完整")
    if value.get("protocol") != CD_COLLAPSE_DIAGNOSTICS_PROTOCOL:
        raise ValueError("cd_collapse_diagnostics 协议不匹配")
    expected = build_cd_collapse_diagnostics_receipt(
        selected_source=value.get("selected_source"),
        diagnostics=value.get("diagnostics"),
    )
    if dict(value) != expected:
        raise ValueError("cd_collapse_diagnostics 内容与协议不一致")
    return expected


__all__ = [
    "CD_CONSISTENCY_CONFIG_PROTOCOL",
    "CD_CONSISTENCY_LOSS",
    "CD_CONSISTENCY_TARGET_SOURCE",
    "CD_DENOISING_PARAMETERIZATION",
    "CD_TRAINING_SEMANTICS",
    "CD_BOUNDARY_PARAMETERIZATION",
    "CD_RECONSTRUCTION_ANCHOR",
    "CD_COLLAPSE_DIAGNOSTIC_PROTOCOL",
    "CD_COLLAPSE_DIAGNOSTIC_ROLE",
    "CD_COLLAPSE_DIAGNOSTICS_PROTOCOL",
    "apply_cd_boundary_parameterization",
    "build_cd_collapse_diagnostic",
    "build_cd_collapse_diagnostics_receipt",
    "resolve_cd_consistency_config",
    "validate_cd_collapse_diagnostic",
    "validate_cd_collapse_diagnostics_receipt",
    "validate_cd_consistency_receipt",
]
