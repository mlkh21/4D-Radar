#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""正式 VAE/LDM/CD checkpoint 链的安全读取、哈希和协议校验。"""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections.abc import Mapping
from typing import Any, Dict, Iterable, Sequence

import torch

from diffusion_consistency_radar.radar_normalization import (
    RadarNormalizationError,
    assert_same_radar_normalization,
    radar_normalization_from_checkpoint,
)


FORMAL_CHECKPOINT_PROTOCOL = "formal_chain_v2"
FORMAL_MINI_CHECKPOINT_PROTOCOL = "formal_mini_chain_v2"
LEGACY_FORMAL_CHECKPOINT_PROTOCOL = "formal_chain_v1"
LEGACY_FORMAL_MINI_CHECKPOINT_PROTOCOL = "formal_mini_chain_v1"
FORMAL_DATA_PROTOCOL = "formal_data_v2"
FORMAL_MINI_SELECTION_PROTOCOL = "formal_mini_selection_v1"
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
_STAGE_ORDER = ("vae", "ldm", "cd")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


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


def build_formal_mini_selection(
    train_frames_per_scene: Any,
    validation_frames_per_scene: Any,
) -> Dict[str, Any]:
    """构造由正式 split hash 唯一约束的确定性 mini 子集身份。"""
    values = {
        "train_frames_per_scene": train_frames_per_scene,
        "validation_frames_per_scene": validation_frames_per_scene,
    }
    for name, value in values.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} 必须是正整数")
    return {
        "protocol": FORMAL_MINI_SELECTION_PROTOCOL,
        "strategy": "ordered_prefix_per_scene",
        "train_frames_per_scene": train_frames_per_scene,
        "validation_frames_per_scene": validation_frames_per_scene,
    }


def _validate_formal_mini_selection(
    value: Any,
    name: str,
    errors: list[str],
) -> Dict[str, Any] | None:
    if not isinstance(value, Mapping):
        errors.append(f"{name} 必须是对象")
        return None
    expected_keys = {
        "protocol",
        "strategy",
        "train_frames_per_scene",
        "validation_frames_per_scene",
    }
    if set(value) != expected_keys:
        errors.append(f"{name} 字段必须精确为 {sorted(expected_keys)}")
        return None
    try:
        expected = build_formal_mini_selection(
            value.get("train_frames_per_scene"),
            value.get("validation_frames_per_scene"),
        )
    except ValueError as exc:
        errors.append(f"{name}: {exc}")
        return None
    if dict(value) != expected:
        errors.append(
            f"{name} protocol/strategy 必须为 "
            f"{FORMAL_MINI_SELECTION_PROTOCOL}/ordered_prefix_per_scene"
        )
        return None
    return expected


def sha256_file(path: str) -> str:
    """以固定分块计算普通 checkpoint 文件的 SHA-256。"""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_torch_load(
    path: str,
    *,
    map_location: Any = "cpu",
    allow_legacy_pickle: bool = False,
) -> Any:
    """默认只允许 weights-only；历史可信文件需显式允许 pickle 回退。"""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError as exc:
        if allow_legacy_pickle:
            return torch.load(path, map_location=map_location)
        raise RuntimeError(
            "当前 PyTorch 不支持 weights_only=True；正式 checkpoint 拒绝回退 pickle，"
            "历史可信文件只能在独立诊断中显式启用 legacy 开关"
        ) from exc
    except Exception as exc:
        message = str(exc)
        if allow_legacy_pickle and (
            "Weights only load failed" in message or "Unsupported global" in message
        ):
            return torch.load(path, map_location=map_location)
        raise


def _validate_sha256(value: Any, name: str, errors: list[str]) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        errors.append(f"{name} 必须是 64 位小写 SHA-256")


def _validate_hash_mapping(value: Any, name: str, errors: list[str]) -> None:
    if not isinstance(value, Mapping) or not value:
        errors.append(f"{name} 必须是非空场景 SHA-256 映射")
        return
    for key, digest in value.items():
        if not isinstance(key, str) or not key or os.path.basename(key) != key:
            errors.append(f"{name} 包含非法场景名 {key!r}")
        _validate_sha256(digest, f"{name}.{key}", errors)


def validate_checkpoint_data_protocol(
    value: Any,
    *,
    stage: str,
    errors: list[str] | None = None,
) -> Dict[str, Any] | None:
    """校验 checkpoint 绑定的数据、监督、划分和标定身份。"""
    own_errors: list[str] = [] if errors is None else errors
    start_count = len(own_errors)
    if not isinstance(value, Mapping):
        own_errors.append(f"{stage}.data_protocol 必须是对象")
        if errors is None:
            raise CheckpointChainError(own_errors)
        return None
    if value.get("protocol") != FORMAL_DATA_PROTOCOL:
        own_errors.append(
            f"{stage}.data_protocol.protocol 必须为 {FORMAL_DATA_PROTOCOL!r}"
        )
    _validate_hash_mapping(
        value.get("dataset_manifest_sha256"),
        f"{stage}.data_protocol.dataset_manifest_sha256",
        own_errors,
    )
    _validate_sha256(
        value.get("split_artifact_sha256"),
        f"{stage}.data_protocol.split_artifact_sha256",
        own_errors,
    )
    _validate_hash_mapping(
        value.get("target_policy_sha256"),
        f"{stage}.data_protocol.target_policy_sha256",
        own_errors,
    )
    _validate_hash_mapping(
        value.get("observed_mask_sha256"),
        f"{stage}.data_protocol.observed_mask_sha256",
        own_errors,
    )
    observed_protocol = value.get("observed_mask_protocol")
    if not isinstance(observed_protocol, str) or not observed_protocol:
        own_errors.append(f"{stage}.data_protocol.observed_mask_protocol 必须是非空字符串")
    if stage in ("ldm", "cd"):
        calibration = value.get("calibration_sha256")
        expected_calibration = {"lidar_to_thermal", "thermal_intrinsics"}
        if not isinstance(calibration, Mapping) or set(calibration) != expected_calibration:
            own_errors.append(
                f"{stage}.data_protocol.calibration_sha256 必须精确包含 "
                f"{sorted(expected_calibration)}"
            )
        else:
            for key in sorted(expected_calibration):
                _validate_sha256(
                    calibration.get(key),
                    f"{stage}.data_protocol.calibration_sha256.{key}",
                    own_errors,
                )
        _validate_hash_mapping(
            value.get("radar_ir_sync_sha256"),
            f"{stage}.data_protocol.radar_ir_sync_sha256",
            own_errors,
        )
    if "mini_selection" in value:
        _validate_formal_mini_selection(
            value.get("mini_selection"),
            f"{stage}.data_protocol.mini_selection",
            own_errors,
        )
    if errors is None and own_errors:
        raise CheckpointChainError(own_errors)
    if len(own_errors) != start_count:
        return None
    return dict(value)


def checkpoint_state_dict(checkpoint: Any) -> Mapping:
    """返回包裹 checkpoint 中的模型参数，拒绝空或非字典状态。"""
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint 必须是字典")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("checkpoint 缺少非空 model_state_dict")
    return state


def assert_checkpoint_training_identity(
    checkpoint: Any,
    *,
    expected_stage: str,
    checkpoint_protocol: str,
    data_protocol: Mapping[str, Any],
) -> None:
    """在恢复优化器状态前验证 stage、链协议和完整数据身份。"""
    errors: list[str] = []
    if not isinstance(checkpoint, Mapping):
        raise CheckpointChainError(["resume checkpoint 必须是对象"])
    if checkpoint.get("stage") != expected_stage:
        errors.append(
            f"resume checkpoint stage={checkpoint.get('stage')!r} 与预期 "
            f"{expected_stage!r} 不一致"
        )
    if checkpoint.get("checkpoint_protocol") != checkpoint_protocol:
        errors.append("resume checkpoint_protocol 与当前训练协议不一致")
    if checkpoint_protocol == FORMAL_MINI_CHECKPOINT_PROTOCOL:
        current_selection = (
            data_protocol.get("mini_selection")
            if isinstance(data_protocol, Mapping)
            else None
        )
        checkpoint_data = checkpoint.get("data_protocol")
        checkpoint_selection = (
            checkpoint_data.get("mini_selection")
            if isinstance(checkpoint_data, Mapping)
            else None
        )
        _validate_formal_mini_selection(
            current_selection,
            "current.data_protocol.mini_selection",
            errors,
        )
        _validate_formal_mini_selection(
            checkpoint_selection,
            "resume.data_protocol.mini_selection",
            errors,
        )
    validated = validate_checkpoint_data_protocol(
        checkpoint.get("data_protocol"),
        stage=expected_stage,
        errors=errors,
    )
    if validated is not None and dict(validated) != dict(data_protocol):
        errors.append("resume data_protocol 与当前训练数据协议不一致")
    if errors:
        raise CheckpointChainError(errors)


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


def _load_stage(
    path: str,
    expected_stage: str,
    errors: list[str],
    report: Dict[str, Any],
    *,
    allow_legacy_protocol: bool,
):
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
        checkpoint = safe_torch_load(
            path,
            allow_legacy_pickle=allow_legacy_protocol,
        )
    except Exception as exc:
        errors.append(f"{expected_stage} checkpoint 无法读取: {exc}")
        return None
    if not isinstance(checkpoint, Mapping):
        errors.append(f"{expected_stage} checkpoint 必须是字典")
        return None
    protocol = checkpoint.get("checkpoint_protocol")
    allowed_protocols = {FORMAL_CHECKPOINT_PROTOCOL}
    if allow_legacy_protocol:
        allowed_protocols.add(LEGACY_FORMAL_CHECKPOINT_PROTOCOL)
    if protocol not in allowed_protocols:
        errors.append(
            f"{expected_stage} checkpoint_protocol 必须为 {sorted(allowed_protocols)}，实际为 {protocol!r}"
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
    ldm_path: str = "",
    cd_path: str = "",
    require_multimodal: bool = True,
    *,
    target_stage: str = "cd",
    allow_legacy_protocol: bool = False,
) -> Dict[str, Any]:
    """只校验目标 stage 及其父链，避免 LDM 被尚不存在的 CD 阻塞。"""
    if target_stage not in _STAGE_ORDER:
        raise ValueError(f"target_stage 必须为 {_STAGE_ORDER} 之一")
    errors: list[str] = []
    report: Dict[str, Any] = {
        "chain_valid": False,
        "protocol": FORMAL_CHECKPOINT_PROTOCOL,
        "target_stage": target_stage,
        "stages": [],
        "sha256": {},
        "state_key_counts": {},
    }
    paths = {"vae": vae_path, "ldm": ldm_path, "cd": cd_path}
    required_stages = _STAGE_ORDER[: _STAGE_ORDER.index(target_stage) + 1]
    loaded = {
        stage: _load_stage(
            paths[stage],
            stage,
            errors,
            report,
            allow_legacy_protocol=allow_legacy_protocol,
        )
        for stage in required_stages
    }
    vae = loaded.get("vae")
    ldm = loaded.get("ldm")
    cd = loaded.get("cd")
    grids: Dict[str, Mapping] = {}
    radar_normalizations: Dict[str, tuple[dict, str]] = {}

    data_protocols: Dict[str, Dict[str, Any]] = {}
    for stage in required_stages:
        checkpoint = loaded.get(stage)
        if checkpoint is None:
            continue
        protocol = checkpoint.get("checkpoint_protocol")
        if protocol == FORMAL_CHECKPOINT_PROTOCOL:
            validated_data_protocol = validate_checkpoint_data_protocol(
                checkpoint.get("data_protocol"),
                stage=stage,
                errors=errors,
            )
            if validated_data_protocol is not None:
                data_protocols[stage] = validated_data_protocol
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
        for stage in required_stages[1:]:
            if stage in grids and not _same_grid(grids["vae"], grids[stage]):
                errors.append(f"{stage} data_grid_config 与 VAE 网格不一致")
    latent_dim = report.get("vae_latent_dim")
    if latent_dim is not None:
        for stage in required_stages[1:]:
            checkpoint = loaded.get(stage)
            if checkpoint is None:
                continue
            model_config = checkpoint.get("model_config") or {}
            if model_config.get("latent_dim") != latent_dim:
                errors.append(f"{stage}.model_config.latent_dim 与 VAE 不一致")

    if "vae" in data_protocols:
        for stage in required_stages[1:]:
            if stage in data_protocols and data_protocols[stage] != data_protocols["vae"]:
                errors.append(f"{stage}.data_protocol 与 VAE 数据协议不一致")
        report["data_protocol"] = data_protocols["vae"]
    if allow_legacy_protocol and loaded.get("vae") is not None:
        report["protocol"] = loaded["vae"].get("checkpoint_protocol")
        report["legacy_diagnostic"] = report["protocol"] != FORMAL_CHECKPOINT_PROTOCOL
    report["grid"] = grids.get("vae")
    report["chain_valid"] = not errors
    if errors:
        raise CheckpointChainError(errors)
    return report


__all__ = [
    "CheckpointChainError",
    "FORMAL_CHECKPOINT_PROTOCOL",
    "FORMAL_DATA_PROTOCOL",
    "FORMAL_MINI_SELECTION_PROTOCOL",
    "FORMAL_MINI_CHECKPOINT_PROTOCOL",
    "LEGACY_FORMAL_CHECKPOINT_PROTOCOL",
    "LEGACY_FORMAL_MINI_CHECKPOINT_PROTOCOL",
    "checkpoint_state_dict",
    "assert_checkpoint_training_identity",
    "build_formal_mini_selection",
    "resolve_training_checkpoint_protocol",
    "safe_torch_load",
    "sha256_file",
    "validate_formal_checkpoint_chain",
    "validate_checkpoint_data_protocol",
]
