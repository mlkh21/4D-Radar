# -*- coding: utf-8 -*-
"""固定 validation 子集与部署一致的纯噪声 Karras 采样合同。"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from diffusion_consistency_radar.occupancy_threshold_artifact import (
    DEFAULT_THRESHOLD_CANDIDATES,
    THRESHOLD_SAMPLING_PROTOCOL,
    resolve_threshold_recall_constraint,
    validate_threshold_candidates,
)


DEPLOYMENT_VALIDATION_PROTOCOL = THRESHOLD_SAMPLING_PROTOCOL
DEPLOYMENT_VALIDATION_SPLIT = "temporal_block_validation_suffix"
DEPLOYMENT_VALIDATION_NOISE_IDENTITY = "sha256_sample_id_seed_v1"
DEPLOYMENT_VALIDATION_SELECTION_STRATEGY = "ordered_prefix_per_scene"
DEPLOYMENT_INITIAL_LATENT = "sigma_max_gaussian_noise"
LDM_DEPLOYMENT_VALIDATION_SELECTOR = (
    "max_deployment_observed_iou_then_min_deployment_latent_loss_v1"
)
CD_DEPLOYMENT_VALIDATION_SELECTOR = (
    "max_deployment_observed_iou_then_min_deployment_latent_loss_prefer_ema_v1"
)
DEPLOYMENT_METRIC_NAMES = (
    "deployment_latent_loss",
    "deployment_occupancy_iou",
)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _positive_int(value: Any, *, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} 必须是正整数")
    return value


def _positive_finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是正有限数")
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是正有限数") from exc
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} 必须是正有限数")
    return resolved


def resolve_deployment_validation_config(
    stage_config: Mapping[str, Any],
    *,
    stage: str,
) -> Dict[str, Any]:
    """解析正式 LDM/CD 完整采样 validation 参数并冻结采样语义。"""
    if not isinstance(stage_config, Mapping):
        raise ValueError("deployment validation stage config 必须是映射")
    if stage not in {"ldm", "cd"}:
        raise ValueError("deployment validation stage 只支持 ldm/cd")

    frames_per_scene = _positive_int(
        stage_config.get("deployment_validation_frames_per_scene", 16),
        name=f"{stage}.deployment_validation_frames_per_scene",
    )
    seed = stage_config.get(
        "deployment_validation_seed",
        stage_config.get("validation_seed", 42),
    )
    if type(seed) is not int or seed < 0:
        raise ValueError(f"{stage}.deployment_validation_seed 必须是非负整数")

    sigma_min = _positive_finite(
        stage_config.get("sigma_min", 0.002),
        name=f"{stage}.sigma_min",
    )
    sigma_max = _positive_finite(
        stage_config.get("sigma_max", 80.0),
        name=f"{stage}.sigma_max",
    )
    rho = _positive_finite(
        stage_config.get("rho", 7.0),
        name=f"{stage}.rho",
    )
    if sigma_min >= sigma_max:
        raise ValueError(f"{stage}.sigma_min 必须小于 sigma_max")

    threshold = _positive_finite(
        stage_config.get(
            "deployment_validation_occupancy_threshold",
            stage_config.get("validation_occupancy_threshold", 0.5),
        ),
        name=f"{stage}.deployment_validation_occupancy_threshold",
    )
    if threshold >= 1.0:
        raise ValueError(
            f"{stage}.deployment_validation_occupancy_threshold 必须严格位于 (0,1)"
        )

    if stage == "ldm":
        steps = _positive_int(
            stage_config.get("deployment_validation_steps", 40),
            name="ldm.deployment_validation_steps",
        )
        sampler = stage_config.get("deployment_validation_sampler", "heun")
        if sampler not in {"heun", "euler"}:
            raise ValueError(
                "ldm.deployment_validation_sampler 必须是 heun/euler"
            )
        selector = LDM_DEPLOYMENT_VALIDATION_SELECTOR
    else:
        steps = _positive_int(
            stage_config.get("deployment_validation_steps", 1),
            name="cd.deployment_validation_steps",
        )
        if steps != 1:
            raise ValueError("正式 CD deployment validation 必须使用一步采样")
        sampler = stage_config.get("deployment_validation_sampler", "one_step")
        if sampler != "one_step":
            raise ValueError(
                "cd.deployment_validation_sampler 必须是 one_step"
            )
        selector = CD_DEPLOYMENT_VALIDATION_SELECTOR

    return {
        "protocol": DEPLOYMENT_VALIDATION_PROTOCOL,
        "stage": stage,
        "split": DEPLOYMENT_VALIDATION_SPLIT,
        "selection_strategy": DEPLOYMENT_VALIDATION_SELECTION_STRATEGY,
        "frames_per_scene": frames_per_scene,
        "noise_identity": DEPLOYMENT_VALIDATION_NOISE_IDENTITY,
        "seed": seed,
        "initial_latent": DEPLOYMENT_INITIAL_LATENT,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "rho": rho,
        "steps": steps,
        "sampler": sampler,
        "occupancy_threshold": threshold,
        "threshold_candidates": list(
            validate_threshold_candidates(
                stage_config.get(
                    "deployment_validation_threshold_candidates",
                    stage_config.get(
                        "validation_threshold_candidates",
                        DEFAULT_THRESHOLD_CANDIDATES,
                    ),
                )
            )
        ),
        "threshold_selection_constraints": resolve_threshold_recall_constraint(
            stage_config.get("deployment_validation_min_occupied_recall"),
            stage_config.get(
                "deployment_validation_recall_constraint_authority"
            ),
        ),
        "selector": selector,
    }


def build_deployment_validation_selection(
    available_frame_ids_by_scene: Mapping[str, Sequence[str]],
    *,
    frames_per_scene: int,
) -> Tuple[Dict[str, list[str]], Dict[str, Any]]:
    """在 DDP sampler 之前选择固定有序前缀并生成内容身份。"""
    limit = _positive_int(
        frames_per_scene,
        name="deployment validation frames_per_scene",
    )
    if not isinstance(available_frame_ids_by_scene, Mapping) or not available_frame_ids_by_scene:
        raise ValueError("deployment validation frame mapping 不能为空")

    selected: Dict[str, list[str]] = {}
    for scene in sorted(available_frame_ids_by_scene):
        frame_ids = available_frame_ids_by_scene[scene]
        if (
            not isinstance(scene, str)
            or not scene
            or isinstance(frame_ids, (str, bytes))
            or not isinstance(frame_ids, Sequence)
            or not frame_ids
        ):
            raise ValueError("deployment validation scene/frame IDs 无效")
        resolved_ids = list(frame_ids)
        if any(not isinstance(frame_id, str) or not frame_id for frame_id in resolved_ids):
            raise ValueError("deployment validation frame_id 必须是非空字符串")
        if len(set(resolved_ids)) != len(resolved_ids):
            raise ValueError("deployment validation frame_id 不得重复")
        selected[scene] = resolved_ids[: min(limit, len(resolved_ids))]

    identity_payload = {
        "protocol": DEPLOYMENT_VALIDATION_PROTOCOL,
        "split": DEPLOYMENT_VALIDATION_SPLIT,
        "strategy": DEPLOYMENT_VALIDATION_SELECTION_STRATEGY,
        "frame_ids_by_scene": selected,
    }
    receipt = {
        **identity_payload,
        "configured_frames_per_scene": limit,
        "selected_frames": sum(len(values) for values in selected.values()),
        "selection_sha256": _canonical_sha256(identity_payload),
    }
    return selected, receipt


def validate_deployment_validation_selection(
    value: Mapping[str, Any],
) -> Dict[str, Any]:
    """校验固定 validation frame-ID 集合及其可复算内容身份。"""
    if not isinstance(value, Mapping):
        raise ValueError("deployment validation selection 必须是映射")
    expected_fields = {
        "protocol": DEPLOYMENT_VALIDATION_PROTOCOL,
        "split": DEPLOYMENT_VALIDATION_SPLIT,
        "strategy": DEPLOYMENT_VALIDATION_SELECTION_STRATEGY,
    }
    for field, expected in expected_fields.items():
        if value.get(field) != expected:
            raise ValueError(
                f"deployment validation selection.{field} 不匹配"
            )

    frame_ids_by_scene = value.get("frame_ids_by_scene")
    if not isinstance(frame_ids_by_scene, Mapping) or not frame_ids_by_scene:
        raise ValueError("deployment validation selection 缺少 frame IDs")
    normalized: Dict[str, list[str]] = {}
    for scene in sorted(frame_ids_by_scene):
        frame_ids = frame_ids_by_scene[scene]
        if (
            not isinstance(scene, str)
            or not scene
            or isinstance(frame_ids, (str, bytes))
            or not isinstance(frame_ids, Sequence)
            or not frame_ids
        ):
            raise ValueError("deployment validation selection frame IDs 无效")
        normalized_ids = list(frame_ids)
        if any(
            not isinstance(frame_id, str) or not frame_id
            for frame_id in normalized_ids
        ) or len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError("deployment validation selection frame IDs 无效")
        normalized[scene] = normalized_ids

    configured_frames = _positive_int(
        value.get("configured_frames_per_scene"),
        name="deployment validation configured_frames_per_scene",
    )
    if any(len(frame_ids) > configured_frames for frame_ids in normalized.values()):
        raise ValueError("deployment validation selection 超过配置帧数")
    selected_frames = sum(len(frame_ids) for frame_ids in normalized.values())
    if value.get("selected_frames") != selected_frames:
        raise ValueError("deployment validation selection 帧数统计不一致")

    identity_payload = {
        "protocol": DEPLOYMENT_VALIDATION_PROTOCOL,
        "split": DEPLOYMENT_VALIDATION_SPLIT,
        "strategy": DEPLOYMENT_VALIDATION_SELECTION_STRATEGY,
        "frame_ids_by_scene": normalized,
    }
    if value.get("selection_sha256") != _canonical_sha256(identity_payload):
        raise ValueError("deployment validation selection 内容 hash 不一致")
    return dict(value)


def validate_deployment_metrics(
    metrics: Mapping[str, Any],
    *,
    source: str,
) -> Dict[str, float]:
    """验证完整采样 selector 使用的 latent/occupancy 指标。"""
    if not isinstance(metrics, Mapping) or set(metrics) != set(
        DEPLOYMENT_METRIC_NAMES
    ):
        raise ValueError(
            f"{source} deployment metrics 必须精确包含 "
            "deployment_latent_loss/deployment_occupancy_iou"
        )
    resolved: Dict[str, float] = {}
    for name in DEPLOYMENT_METRIC_NAMES:
        raw = metrics[name]
        if isinstance(raw, bool):
            raise ValueError(f"{source} {name} 必须是有限数")
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{source} {name} 必须是有限数") from exc
        if not math.isfinite(value):
            raise ValueError(f"{source} {name} 必须是有限数")
        resolved[name] = value
    if resolved["deployment_latent_loss"] < 0.0:
        raise ValueError(f"{source} deployment_latent_loss 必须非负")
    if not 0.0 <= resolved["deployment_occupancy_iou"] <= 1.0:
        raise ValueError(f"{source} deployment_occupancy_iou 必须位于 [0,1]")
    return resolved


def deployment_metrics_are_improved(
    current: Mapping[str, Any],
    best: Mapping[str, Any],
) -> bool:
    """完整采样 observed IoU 优先、latent loss 次优。"""
    current_values = validate_deployment_metrics(current, source="current")
    best_values = validate_deployment_metrics(best, source="best")
    current_iou = current_values["deployment_occupancy_iou"]
    best_iou = best_values["deployment_occupancy_iou"]
    if current_iou != best_iou:
        return current_iou > best_iou
    return (
        current_values["deployment_latent_loss"]
        < best_values["deployment_latent_loss"]
    )


def karras_sigma_schedule(
    *,
    steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """构造训练 validation 与推理共用的 Karras 递减 sigma 序列。"""
    resolved_steps = _positive_int(steps, name="Karras steps")
    resolved_min = _positive_finite(sigma_min, name="Karras sigma_min")
    resolved_max = _positive_finite(sigma_max, name="Karras sigma_max")
    resolved_rho = _positive_finite(rho, name="Karras rho")
    if resolved_min >= resolved_max:
        raise ValueError("Karras sigma_min 必须小于 sigma_max")
    step_indices = torch.arange(
        resolved_steps + 1,
        device=device,
        dtype=dtype,
    )
    t = step_indices / resolved_steps
    sigmas = (
        resolved_max ** (1.0 / resolved_rho)
        + t
        * (
            resolved_min ** (1.0 / resolved_rho)
            - resolved_max ** (1.0 / resolved_rho)
        )
    ) ** resolved_rho
    # 浮点幂运算会让理论首尾端点略微越界；显式钉住合同端点。
    represented_min = torch.as_tensor(
        resolved_min,
        device=device,
        dtype=sigmas.dtype,
    )
    represented_max = torch.as_tensor(
        resolved_max,
        device=device,
        dtype=sigmas.dtype,
    )
    sigmas = torch.minimum(
        torch.maximum(sigmas, represented_min),
        represented_max,
    )
    sigmas[0] = represented_max
    sigmas[-1] = represented_min
    return sigmas


def sample_karras_ode(
    initial_latent: torch.Tensor,
    sigmas: torch.Tensor,
    denoise_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    sampler: str,
    step_callback: Optional[Callable[[int, int], None]] = None,
) -> torch.Tensor:
    """从 sigma_max latent 执行 Euler/Heun ODE；模型调用由入口注入。"""
    if sampler not in {"heun", "euler"}:
        raise ValueError("Karras sampler 必须是 heun/euler")
    if not torch.is_tensor(initial_latent) or initial_latent.ndim < 2:
        raise ValueError("initial_latent 必须是带 batch 维的 tensor")
    if not torch.is_tensor(sigmas) or sigmas.ndim != 1 or sigmas.numel() < 2:
        raise ValueError("sigmas 必须是一维且至少含两个值")
    if not torch.isfinite(sigmas).all() or bool((sigmas <= 0).any().item()):
        raise ValueError("sigmas 必须是正有限数")
    if bool((sigmas[1:] >= sigmas[:-1]).any().item()):
        raise ValueError("sigmas 必须严格递减")

    latent = initial_latent
    batch_size = initial_latent.shape[0]
    for index in range(sigmas.numel() - 1):
        sigma_t = sigmas[index]
        sigma_next = sigmas[index + 1]
        sigma_batch = torch.full(
            (batch_size,),
            float(sigma_t.item()),
            device=latent.device,
            dtype=latent.dtype,
        )
        denoised = denoise_fn(latent, sigma_batch)
        if not torch.is_tensor(denoised) or denoised.shape != latent.shape:
            raise ValueError("denoise_fn 输出 shape 必须与 latent 一致")
        derivative = (latent - denoised) / sigma_t
        delta = sigma_next - sigma_t
        if sampler == "heun" and index < sigmas.numel() - 2:
            predicted = latent + derivative * delta
            sigma_batch_next = torch.full(
                (batch_size,),
                float(sigma_next.item()),
                device=latent.device,
                dtype=latent.dtype,
            )
            denoised_next = denoise_fn(predicted, sigma_batch_next)
            if (
                not torch.is_tensor(denoised_next)
                or denoised_next.shape != latent.shape
            ):
                raise ValueError("denoise_fn Heun 校正输出 shape 不一致")
            derivative_next = (predicted - denoised_next) / sigma_next
            latent = latent + 0.5 * (derivative + derivative_next) * delta
        else:
            latent = latent + derivative * delta
        if step_callback is not None:
            step_callback(index + 1, sigmas.numel() - 1)
    return latent
