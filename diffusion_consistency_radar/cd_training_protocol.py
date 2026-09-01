#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""文件功能：定义 CD 的真实 EMA consistency 训练与部署采样配置协议。"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Dict


CD_CONSISTENCY_CONFIG_PROTOCOL = "ema_consistency_training_config_v1"
CD_TRAINING_SEMANTICS = "ldm_initialized_ema_consistency_v1"
CD_DENOISING_PARAMETERIZATION = "direct_x0_sigma_conditioned_v1"
CD_CONSISTENCY_TARGET_SOURCE = "cd_model_ema"
CD_CONSISTENCY_LOSS = "mse"

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


__all__ = [
    "CD_CONSISTENCY_CONFIG_PROTOCOL",
    "CD_CONSISTENCY_LOSS",
    "CD_CONSISTENCY_TARGET_SOURCE",
    "CD_DENOISING_PARAMETERIZATION",
    "CD_TRAINING_SEMANTICS",
    "resolve_cd_consistency_config",
    "validate_cd_consistency_receipt",
]
