# -*- coding: utf-8 -*-
"""占用监督域协议常量与校验。

该模块只描述监督域的语义，不计算哈希，也不改变既有正式数据文件。
"""

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F


LEGACY_OBSERVATION_SUPERVISION_PROTOCOL = (
    "persisted_voxel_and_adaptive_latent_any_v1"
)
RELIABLE_FREE_OBSERVATION_SUPERVISION_PROTOCOL = (
    "occupied_reliable_free_unknown_v1"
)
SUPPORTED_OBSERVATION_SUPERVISION_PROTOCOLS = (
    LEGACY_OBSERVATION_SUPERVISION_PROTOCOL,
    RELIABLE_FREE_OBSERVATION_SUPERVISION_PROTOCOL,
)


def validate_observation_supervision_protocol(
    protocol: Optional[str],
) -> str:
    """验证监督域协议；缺省值保持历史 any-observed 语义。"""
    resolved = (
        LEGACY_OBSERVATION_SUPERVISION_PROTOCOL
        if protocol in (None, "")
        else str(protocol).strip()
    )
    if resolved not in SUPPORTED_OBSERVATION_SUPERVISION_PROTOCOLS:
        raise ValueError(
            "不支持的 observation_supervision_protocol: "
            f"{resolved!r}，可选值为 {list(SUPPORTED_OBSERVATION_SUPERVISION_PROTOCOLS)!r}"
        )
    return resolved


def uses_reliable_free_supervision(protocol: Optional[str]) -> bool:
    """判断是否启用 occupied/reliable-free/unknown 三态监督。"""
    return (
        validate_observation_supervision_protocol(protocol)
        == RELIABLE_FREE_OBSERVATION_SUPERVISION_PROTOCOL
    )


def threshold_observation_domain(protocol: Optional[str]) -> str:
    """为验证阈值收据给出与实际标签域一致的名称。"""
    return (
        RELIABLE_FREE_OBSERVATION_SUPERVISION_PROTOCOL
        if uses_reliable_free_supervision(protocol)
        else "persisted_observed_mask_v1"
    )


def select_training_supervision_mask(
    metadata: Dict[str, Any], protocol: Optional[str]
) -> Any:
    """按协议选择训练张量；三态缺失时禁止落回旧 any-observed mask。"""
    resolved = validate_observation_supervision_protocol(protocol)
    if not uses_reliable_free_supervision(resolved):
        return metadata.get("occupancy_observed_mask")
    batch_protocol = metadata.get("observation_supervision_protocol")
    if isinstance(batch_protocol, (list, tuple)):
        protocol_matches = bool(batch_protocol) and all(
            item == resolved for item in batch_protocol
        )
    else:
        protocol_matches = batch_protocol == resolved
    if not protocol_matches:
        raise ValueError(
            "三态监督 batch 的 observation_supervision_protocol 不匹配"
        )
    mask = metadata.get("occupancy_supervision_mask")
    if mask is None:
        raise RuntimeError(
            "三态监督 batch 缺少 occupancy_supervision_mask；"
            "请使用支持 observation_supervision_protocol 的 Dataset"
        )
    return mask


def pool_latent_supervision_mask(
    voxel_mask: torch.Tensor,
    target: Optional[torch.Tensor],
    output_size,
    protocol: Optional[str],
) -> torch.Tensor:
    """规约至潜网格：正例优先，其余仅全域受监督的块可作负例。

    旧协议维持 any-pooling；新协议拒绝部分已知的空块，防止在 latent
    损失中把模型网格上的 unknown 再次扩张成负监督。
    """
    if not uses_reliable_free_supervision(protocol):
        return F.adaptive_max_pool3d(voxel_mask.float(), output_size).bool()
    if (
        target.ndim != 5
        or voxel_mask.ndim != 5
        or target.shape[0] != voxel_mask.shape[0]
        or tuple(target.shape[-3:]) != tuple(voxel_mask.shape[-3:])
    ):
        raise ValueError("三态 latent 监督要求 target 与 voxel_mask 网格一致")
    occupied = F.adaptive_max_pool3d(
        (target[:, 0:1] > 0.5).float(), output_size
    ).bool()
    has_unknown = F.adaptive_max_pool3d(
        (~voxel_mask.bool()).float(), output_size
    ).bool()
    return occupied | ~has_unknown


def assert_checkpoint_supervision_protocol(
    checkpoint: Dict[str, Any], protocol: Optional[str], stage: str
) -> None:
    """三态父链必须精确一致；历史缺字段的 checkpoint 只能当旧协议。"""
    expected = validate_observation_supervision_protocol(protocol)
    recorded = checkpoint.get("observation_supervision_protocol")
    if recorded is None:
        recorded = LEGACY_OBSERVATION_SUPERVISION_PROTOCOL
    if recorded != expected:
        raise ValueError(
            f"{stage} checkpoint observation_supervision_protocol 不匹配："
            f"checkpoint={recorded!r}, current={expected!r}"
        )
