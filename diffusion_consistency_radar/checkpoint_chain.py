#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""正式 VAE/LDM/CD checkpoint 链的安全读取、哈希和协议校验。"""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Mapping
from typing import Any, Dict, Iterable, Sequence

import torch

from diffusion_consistency_radar.radar_normalization import (
    RadarNormalizationError,
    assert_same_radar_normalization,
    radar_normalization_from_checkpoint,
)


FORMAL_CHECKPOINT_PROTOCOL = "formal_chain_v1"
FORMAL_MINI_CHECKPOINT_PROTOCOL = "formal_mini_chain_v1"
_TRAINING_CHECKPOINT_PROTOCOLS = {
    FORMAL_CHECKPOINT_PROTOCOL,
    FORMAL_MINI_CHECKPOINT_PROTOCOL,
}
_MULTIMODAL_PREFIXES = (
    "radar_encoder.",
    "model_uncertainty_head.",
    "ir_extractor.",
    "fusion_conv.",
)


class CheckpointChainError(ValueError):
    """正式 checkpoint 链不满足协议时返回的聚合错误。"""

    def __init__(self, errors: Iterable[str]):
        self.errors = tuple(str(error) for error in errors if str(error))
        message = "正式 checkpoint 链校验失败:\n- " + "\n- ".join(self.errors)
        super().__init__(message)


def resolve_training_checkpoint_protocol(value: Any) -> str:
    """解析训练输出协议；mini 权重必须与正式全量链物理隔离。"""
    protocol = FORMAL_CHECKPOINT_PROTOCOL if value in (None, "") else value
    if not isinstance(protocol, str) or protocol not in _TRAINING_CHECKPOINT_PROTOCOLS:
        raise ValueError(
            "data.checkpoint_protocol 必须为 "
            f"{sorted(_TRAINING_CHECKPOINT_PROTOCOLS)}，实际为 {protocol!r}"
        )
    return protocol


def sha256_file(path: str) -> str:
    """以固定分块计算普通 checkpoint 文件的 SHA-256。"""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_torch_load(path: str) -> Any:
    """优先使用 weights_only，兼容旧版 PyTorch 的安全加载入口。"""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception as exc:
        # NOTE: 历史 PyTorch 对 weights_only 支持不一致，仅按明确兼容错误回退。
        message = str(exc)
        if "Weights only load failed" in message or "Unsupported global" in message:
            return torch.load(path, map_location="cpu")
        raise


def checkpoint_state_dict(checkpoint: Any) -> Mapping:
    """返回包裹 checkpoint 中的模型参数，拒绝空或非字典状态。"""
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint 必须是字典")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("checkpoint 缺少非空 model_state_dict")
    return state


def _finite_sequence(value: Any, name: str, length: int, errors: list[str]):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        errors.append(f"{name} 必须是长度为 {length} 的数组")
        return None
    try:
        values = [float(item) for item in value]
    except (TypeError, ValueError):
        errors.append(f"{name} 含有不可转换的数值")
        return None
    if not all(math.isfinite(item) for item in values):
        errors.append(f"{name} 含有非有限数值")
        return None
    return values


def _grid_from_checkpoint(checkpoint: Mapping, stage: str, errors: list[str]):
    grid = checkpoint.get("data_grid_config")
    if not isinstance(grid, Mapping):
        errors.append(f"{stage} 缺少 data_grid_config")
        return None
    target = _finite_sequence(grid.get("target_size"), f"{stage}.target_size", 3, errors)
    source = _finite_sequence(
        grid.get("source_pc_range"), f"{stage}.source_pc_range", 6, errors
    )
    model = _finite_sequence(
        grid.get("model_pc_range"), f"{stage}.model_pc_range", 6, errors
    )
    if target is not None:
        if any(int(item) != item or int(item) <= 0 for item in target):
            errors.append(f"{stage}.target_size 必须是正整数")
        target = [int(item) for item in target]
    return {
        "target_size": target,
        "source_pc_range": source,
        "model_pc_range": model,
    }


def _model_grid(checkpoint: Mapping, stage: str, grid: Mapping, errors: list[str]):
    grid = grid or {}
    config = checkpoint.get("model_config")
    if not isinstance(config, Mapping):
        errors.append(f"{stage} 缺少 model_config")
        return None
    required = ("latent_dim", "in_channels", "out_channels")
    for key in required:
        if key not in config:
            errors.append(f"{stage}.model_config 缺少 {key}")
    fusion_shape = config.get("fusion_voxel_shape")
    fusion_range = config.get("fusion_pc_range")
    if config.get("fusion_latent_shape") is None:
        errors.append(f"{stage}.model_config 缺少 fusion_latent_shape")
    if fusion_shape is None:
        errors.append(f"{stage}.model_config 缺少 fusion_voxel_shape")
    else:
        try:
            fusion_shape = [int(item) for item in fusion_shape]
        except (TypeError, ValueError):
            errors.append(f"{stage}.fusion_voxel_shape 非法")
            fusion_shape = None
    fusion_range = _finite_sequence(
        fusion_range, f"{stage}.fusion_pc_range", 6, errors
    )
    if fusion_shape is not None and grid.get("target_size") is not None:
        if fusion_shape != grid["target_size"]:
            errors.append(
                f"{stage} fusion_voxel_shape 与 data_grid_config.target_size 不一致"
            )
    if fusion_range is not None and grid.get("model_pc_range") is not None:
        if fusion_range != grid["model_pc_range"]:
            errors.append(
                f"{stage} fusion_pc_range 与 data_grid_config.model_pc_range 不一致"
            )
    return {
        "latent_dim": config.get("latent_dim"),
        "in_channels": config.get("in_channels"),
        "out_channels": config.get("out_channels"),
        "fusion_voxel_shape": fusion_shape,
        "fusion_pc_range": fusion_range,
    }


def _load_stage(path: str, expected_stage: str, errors: list[str], report: Dict[str, Any]):
    if not path:
        errors.append(f"{expected_stage} checkpoint 路径为空")
        return None
    if os.path.islink(path):
        errors.append(f"{expected_stage} checkpoint 拒绝符号链接: {path}")
        return None
    if not os.path.isfile(path):
        errors.append(f"{expected_stage} checkpoint 不存在或不是普通文件: {path}")
        return None
    try:
        report["sha256"][expected_stage] = sha256_file(path)
        checkpoint = safe_torch_load(path)
    except Exception as exc:
        errors.append(f"{expected_stage} checkpoint 无法读取: {exc}")
        return None
    if not isinstance(checkpoint, Mapping):
        errors.append(f"{expected_stage} checkpoint 必须是字典")
        return None
    protocol = checkpoint.get("checkpoint_protocol")
    if protocol != FORMAL_CHECKPOINT_PROTOCOL:
        errors.append(
            f"{expected_stage} checkpoint_protocol 必须为 {FORMAL_CHECKPOINT_PROTOCOL!r}，实际为 {protocol!r}"
        )
    stage = checkpoint.get("stage")
    if stage != expected_stage:
        errors.append(f"checkpoint stage={stage!r} 与预期 {expected_stage!r} 不一致")
    try:
        state = checkpoint_state_dict(checkpoint)
        report["state_key_counts"][expected_stage] = len(state)
    except (TypeError, ValueError) as exc:
        errors.append(f"{expected_stage}: {exc}")
    report["stages"].append(expected_stage)
    return checkpoint


def _check_multimodal_state(checkpoint: Mapping, stage: str, errors: list[str]):
    try:
        state = checkpoint_state_dict(checkpoint)
    except (TypeError, ValueError):
        return
    missing = [
        prefix
        for prefix in _MULTIMODAL_PREFIXES
        if not any(str(key).startswith(prefix) for key in state)
    ]
    if missing:
        errors.append(
            f"{stage} 缺少多模态关键权重前缀: {', '.join(missing)}；拒绝 legacy checkpoint"
        )


def _same_grid(left: Mapping, right: Mapping) -> bool:
    return left == right


def validate_formal_checkpoint_chain(
    vae_path: str,
    ldm_path: str,
    cd_path: str,
    require_multimodal: bool = True,
) -> Dict[str, Any]:
    """校验三阶段正式链并返回可写入报告的摘要；失败时聚合后抛错。"""
    errors: list[str] = []
    report: Dict[str, Any] = {
        "chain_valid": False,
        "protocol": FORMAL_CHECKPOINT_PROTOCOL,
        "stages": [],
        "sha256": {},
        "state_key_counts": {},
    }
    vae = _load_stage(vae_path, "vae", errors, report)
    ldm = _load_stage(ldm_path, "ldm", errors, report)
    cd = _load_stage(cd_path, "cd", errors, report)
    grids: Dict[str, Mapping] = {}
    radar_normalizations: Dict[str, tuple[dict, str]] = {}

    for stage, checkpoint in (("vae", vae), ("ldm", ldm), ("cd", cd)):
        if checkpoint is None:
            continue
        grids[stage] = _grid_from_checkpoint(checkpoint, stage, errors)
        if stage == "vae":
            if (
                "radar_normalization" in checkpoint
                or "radar_normalization_sha256" in checkpoint
            ):
                errors.append(
                    "VAE checkpoint 不得绑定 Radar normalization；该协议只属于 LDM/CD 条件输入"
                )
            config = checkpoint.get("vae_config")
            if not isinstance(config, Mapping):
                errors.append("vae 缺少 vae_config")
            else:
                report["vae_latent_dim"] = config.get("latent_dim")
        else:
            _model_grid(checkpoint, stage, grids[stage], errors)
            try:
                radar_normalizations[stage] = radar_normalization_from_checkpoint(
                    checkpoint,
                    target_size=grids[stage]["target_size"],
                    source_pc_range=grids[stage]["source_pc_range"],
                    model_pc_range=grids[stage]["model_pc_range"],
                    context=f"{stage} checkpoint",
                )
            except RadarNormalizationError as exc:
                errors.append(str(exc))
            if require_multimodal:
                _check_multimodal_state(checkpoint, stage, errors)
            parent_hash = checkpoint.get("vae_checkpoint_sha256")
            if parent_hash != report["sha256"].get("vae"):
                errors.append(f"{stage}.vae_checkpoint_sha256 与 VAE 文件 hash 不匹配")
            if stage == "cd":
                ldm_hash = checkpoint.get("ldm_checkpoint_sha256")
                if ldm_hash != report["sha256"].get("ldm"):
                    errors.append("cd.ldm_checkpoint_sha256 与 LDM 文件 hash 不匹配")

    if "ldm" in radar_normalizations and "cd" in radar_normalizations:
        ldm_spec, ldm_normalization_hash = radar_normalizations["ldm"]
        cd_spec, cd_normalization_hash = radar_normalizations["cd"]
        try:
            assert_same_radar_normalization(
                ldm_spec,
                ldm_normalization_hash,
                cd_spec,
                cd_normalization_hash,
                context="LDM/CD checkpoint 链",
            )
        except RadarNormalizationError as exc:
            errors.append(str(exc))
        else:
            report["radar_normalization_protocol"] = ldm_spec["protocol"]
            report["radar_normalization_sha256"] = ldm_normalization_hash

    if "vae" in grids:
        for stage in ("ldm", "cd"):
            if stage in grids and not _same_grid(grids["vae"], grids[stage]):
                errors.append(f"{stage} data_grid_config 与 VAE 网格不一致")
    latent_dim = report.get("vae_latent_dim")
    if latent_dim is not None:
        for stage, checkpoint in (("ldm", ldm), ("cd", cd)):
            if checkpoint is None:
                continue
            model_config = checkpoint.get("model_config") or {}
            if model_config.get("latent_dim") != latent_dim:
                errors.append(f"{stage}.model_config.latent_dim 与 VAE 不一致")

    report["grid"] = grids.get("vae")
    report["chain_valid"] = not errors
    if errors:
        raise CheckpointChainError(errors)
    return report


__all__ = [
    "CheckpointChainError",
    "FORMAL_CHECKPOINT_PROTOCOL",
    "FORMAL_MINI_CHECKPOINT_PROTOCOL",
    "checkpoint_state_dict",
    "resolve_training_checkpoint_protocol",
    "safe_torch_load",
    "sha256_file",
    "validate_formal_checkpoint_chain",
]
