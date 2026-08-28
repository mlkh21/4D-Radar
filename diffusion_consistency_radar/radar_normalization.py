#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""定义 Radar 四通道归一化 artifact 的校验、加载与运行时变换。"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from typing import Any, Optional, Tuple

import torch


RADAR_NORMALIZATION_PROTOCOL = "radar_normalization_v1"
LEGACY_RADAR_NORMALIZATION_PROTOCOL = "legacy_identity"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_TOP_LEVEL_KEYS = {
    "protocol",
    "formal",
    "training_scenes",
    "frame_count",
    "target_size",
    "source_pc_range",
    "model_pc_range",
    "intensity",
    "doppler",
    "variance",
    "input_provenance",
}


class RadarNormalizationError(ValueError):
    """Radar 归一化 artifact 或绑定关系不满足严格协议。"""


def _require_mapping(value: Any, name: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise RadarNormalizationError(f"{name} 必须是 JSON 对象")
    return value


def _require_exact_keys(value: Any, expected: set[str], name: str) -> Mapping:
    mapping = _require_mapping(value, name)
    actual = set(mapping)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RadarNormalizationError(
            f"{name} 字段不符合协议: missing={missing}, extra={extra}"
        )
    return mapping


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RadarNormalizationError(f"{name} 必须是有限数")
    result = float(value)
    if not math.isfinite(result):
        raise RadarNormalizationError(f"{name} 必须是有限数")
    return result


def _finite_sequence(value: Any, name: str, length: int) -> list[float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != length
    ):
        raise RadarNormalizationError(f"{name} 必须是长度为 {length} 的数组")
    return [_finite_number(item, f"{name}[{index}]") for index, item in enumerate(value)]


def _positive_integer_sequence(value: Any, name: str, length: int) -> list[int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != length
    ):
        raise RadarNormalizationError(f"{name} 必须是长度为 {length} 的正整数数组")
    result = []
    for index, item in enumerate(value):
        if type(item) is not int or item <= 0:
            raise RadarNormalizationError(f"{name}[{index}] 必须是正整数")
        result.append(item)
    return result


def _same_numeric_sequence(left: Sequence[float], right: Sequence[float]) -> bool:
    return len(left) == len(right) and all(
        math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-12)
        for a, b in zip(left, right)
    )


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_radar_normalization_sha256(
    digest: Any,
    *,
    context: str = "Radar normalization",
) -> str:
    """验证 artifact 文件身份字段并返回原字符串。"""
    if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
        raise RadarNormalizationError(f"{context} SHA-256 无效")
    return digest


def validate_radar_normalization_spec(
    spec: Any,
    *,
    target_size: Sequence[int],
    source_pc_range: Sequence[float],
    model_pc_range: Sequence[float],
    doppler_scale_mps: Optional[float] = None,
    require_formal: bool = True,
    expected_split_artifact_sha256: Optional[str] = None,
) -> dict:
    """严格验证 normalization v1，并返回不共享可变对象的副本。"""
    mapping = _require_exact_keys(spec, _TOP_LEVEL_KEYS, "radar_normalization")
    if mapping.get("protocol") != RADAR_NORMALIZATION_PROTOCOL:
        raise RadarNormalizationError(
            "radar_normalization protocol 必须为 "
            f"{RADAR_NORMALIZATION_PROTOCOL!r}"
        )
    formal = mapping.get("formal")
    if type(formal) is not bool:
        raise RadarNormalizationError("radar_normalization formal 必须是 bool")
    if require_formal and not formal:
        raise RadarNormalizationError("正式入口拒绝 formal=false 的 normalization artifact")

    scenes = mapping.get("training_scenes")
    if (
        not isinstance(scenes, list)
        or not scenes
        or any(not isinstance(scene, str) or not scene for scene in scenes)
        or len(set(scenes)) != len(scenes)
    ):
        raise RadarNormalizationError(
            "training_scenes 必须是非空、无重复的场景字符串数组"
        )
    frame_count = mapping.get("frame_count")
    if type(frame_count) is not int or frame_count <= 0:
        raise RadarNormalizationError("frame_count 必须是严格正整数")

    artifact_target = _positive_integer_sequence(
        mapping.get("target_size"), "target_size", 3
    )
    artifact_source = _finite_sequence(
        mapping.get("source_pc_range"), "source_pc_range", 6
    )
    artifact_model = _finite_sequence(
        mapping.get("model_pc_range"), "model_pc_range", 6
    )
    for axis in range(3):
        if artifact_source[axis] >= artifact_source[axis + 3]:
            raise RadarNormalizationError("source_pc_range 上下界无效")
        if artifact_model[axis] >= artifact_model[axis + 3]:
            raise RadarNormalizationError("model_pc_range 上下界无效")
        if (
            artifact_model[axis] < artifact_source[axis]
            or artifact_model[axis + 3] > artifact_source[axis + 3]
        ):
            raise RadarNormalizationError(
                "model_pc_range 必须完全位于 source_pc_range 内"
            )

    expected_target = _positive_integer_sequence(target_size, "expected target_size", 3)
    expected_source = _finite_sequence(
        source_pc_range, "expected source_pc_range", 6
    )
    expected_model = _finite_sequence(model_pc_range, "expected model_pc_range", 6)
    if artifact_target != expected_target:
        raise RadarNormalizationError(
            "normalization target_size 与运行网格不一致: "
            f"artifact={artifact_target}, runtime={expected_target}"
        )
    if not _same_numeric_sequence(artifact_source, expected_source):
        raise RadarNormalizationError("normalization source_pc_range 与运行网格不一致")
    if not _same_numeric_sequence(artifact_model, expected_model):
        raise RadarNormalizationError("normalization model_pc_range 与运行网格不一致")

    intensity = _require_exact_keys(
        mapping.get("intensity"),
        {"transform", "log_median", "log_iqr", "clip"},
        "intensity",
    )
    if intensity.get("transform") != "log1p_robust_zscore":
        raise RadarNormalizationError("intensity transform 不支持")
    _finite_number(intensity.get("log_median"), "intensity.log_median")
    log_iqr = _finite_number(intensity.get("log_iqr"), "intensity.log_iqr")
    if log_iqr <= 0.0:
        raise RadarNormalizationError("intensity.log_iqr/IQR 必须是正有限数")
    intensity_clip = _finite_sequence(intensity.get("clip"), "intensity.clip", 2)
    if intensity_clip[0] >= intensity_clip[1]:
        raise RadarNormalizationError("intensity.clip 上下界无效")

    doppler = _require_exact_keys(
        mapping.get("doppler"),
        {"transform", "scale_mps", "clip"},
        "doppler",
    )
    if doppler.get("transform") != "symmetric_physical_scale":
        raise RadarNormalizationError("doppler transform 不支持")
    artifact_scale = _finite_number(doppler.get("scale_mps"), "doppler.scale_mps")
    if artifact_scale <= 0.0:
        raise RadarNormalizationError("doppler.scale_mps 必须是正有限数")
    doppler_clip = _finite_sequence(doppler.get("clip"), "doppler.clip", 2)
    if not _same_numeric_sequence(doppler_clip, [-1.0, 1.0]):
        raise RadarNormalizationError("doppler.clip 必须严格为 [-1, 1]")
    if doppler_scale_mps is not None:
        expected_scale = _finite_number(doppler_scale_mps, "doppler_scale_mps")
        if expected_scale <= 0.0 or not math.isclose(
            artifact_scale,
            expected_scale,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RadarNormalizationError(
                "normalization Doppler scale/量程与配置不一致: "
                f"artifact={artifact_scale}, config={expected_scale}"
            )

    variance = _require_exact_keys(
        mapping.get("variance"),
        {"transform", "unit", "aggregation"},
        "variance",
    )
    if variance.get("transform") != "identity":
        raise RadarNormalizationError("variance transform 必须为 identity")
    if variance.get("unit") != "m2_s2":
        raise RadarNormalizationError("variance unit 必须为 m2_s2")
    if variance.get("aggregation") != "occupied_voxel_equal_weight_total_variance":
        raise RadarNormalizationError("variance aggregation 不符合协议")

    provenance = _require_mapping(
        mapping.get("input_provenance"),
        "input_provenance",
    )
    allowed_provenance_keys = {
        frozenset({"dataset_manifest_sha256"}),
        frozenset({"dataset_manifest_sha256", "split_artifact_sha256"}),
    }
    if frozenset(provenance) not in allowed_provenance_keys:
        raise RadarNormalizationError(
            "input_provenance 字段必须包含 dataset manifest，且只可选配 split artifact"
        )
    manifests = _require_mapping(
        provenance.get("dataset_manifest_sha256"),
        "input_provenance.dataset_manifest_sha256",
    )
    if set(manifests) != set(scenes):
        raise RadarNormalizationError(
            "dataset manifest provenance 场景集合与 training_scenes 不一致"
        )
    for scene, digest in manifests.items():
        if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
            raise RadarNormalizationError(
                f"场景 {scene!r} 的 dataset manifest SHA-256 无效"
            )
    split_digest = provenance.get("split_artifact_sha256")
    if split_digest is not None:
        validate_radar_normalization_sha256(
            split_digest,
            context="normalization split artifact",
        )
    if expected_split_artifact_sha256 is not None:
        expected_split = validate_radar_normalization_sha256(
            expected_split_artifact_sha256,
            context="expected split artifact",
        )
        if split_digest != expected_split:
            raise RadarNormalizationError(
                "normalization 未绑定当前 temporal split artifact SHA-256"
            )

    return copy.deepcopy(dict(mapping))


def load_radar_normalization_artifact(
    path: str,
    *,
    target_size: Sequence[int],
    source_pc_range: Sequence[float],
    model_pc_range: Sequence[float],
    doppler_scale_mps: float,
    require_formal: bool = True,
    expected_split_artifact_sha256: Optional[str] = None,
) -> Tuple[dict, str]:
    """从非 symlink 普通文件读取、验证 spec，并返回真实文件 SHA-256。"""
    artifact_path = os.path.abspath(os.fspath(path))
    if not os.path.lexists(artifact_path):
        raise RadarNormalizationError(f"normalization artifact 不存在: {artifact_path}")
    if os.path.islink(artifact_path):
        raise RadarNormalizationError(
            f"normalization artifact 拒绝符号链接/symlink: {artifact_path}"
        )
    if not os.path.isfile(artifact_path):
        raise RadarNormalizationError(
            f"normalization artifact 不是普通文件: {artifact_path}"
        )
    try:
        with open(artifact_path, "r", encoding="utf-8") as handle:
            spec = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RadarNormalizationError(
            f"normalization artifact 无法解析: {artifact_path}: {exc}"
        ) from exc
    validated = validate_radar_normalization_spec(
        spec,
        target_size=target_size,
        source_pc_range=source_pc_range,
        model_pc_range=model_pc_range,
        doppler_scale_mps=doppler_scale_mps,
        require_formal=require_formal,
        expected_split_artifact_sha256=expected_split_artifact_sha256,
    )
    return validated, _sha256_file(artifact_path)


def apply_radar_normalization(radar_tensor: torch.Tensor, spec: Mapping) -> torch.Tensor:
    """把模型网格上的物理 Radar 四通道转换为冻结的网络输入量纲。"""
    if not isinstance(spec, Mapping):
        raise RadarNormalizationError("radar_normalization 必须是 JSON 对象")
    validated = validate_radar_normalization_spec(
        spec,
        target_size=spec.get("target_size"),
        source_pc_range=spec.get("source_pc_range"),
        model_pc_range=spec.get("model_pc_range"),
        doppler_scale_mps=spec.get("doppler", {}).get("scale_mps")
        if isinstance(spec.get("doppler"), Mapping)
        else None,
        require_formal=False,
    )
    if radar_tensor.ndim != 4 or radar_tensor.shape[0] != 4:
        raise RadarNormalizationError(
            f"Radar tensor 必须是 (4,Z,X,Y)，当前为 {tuple(radar_tensor.shape)}"
        )
    if tuple(radar_tensor.shape[1:]) != tuple(validated["target_size"]):
        raise RadarNormalizationError(
            "Radar tensor shape 与 normalization target_size 不一致: "
            f"tensor={tuple(radar_tensor.shape[1:])}, "
            f"artifact={tuple(validated['target_size'])}"
        )
    if not torch.isfinite(radar_tensor).all():
        raise RadarNormalizationError("Radar tensor 必须全部为有限数")

    result = radar_tensor.clone().float()
    occupied = result[0:1] > 0
    if torch.any((result[3:4] < 0) & occupied):
        raise RadarNormalizationError("occupied Radar variance 不得为负")
    intensity = validated["intensity"]
    result[1:2] = torch.clamp(
        (
            torch.log1p(result[1:2].clamp_min(0.0))
            - float(intensity["log_median"])
        )
        / float(intensity["log_iqr"]),
        min=float(intensity["clip"][0]),
        max=float(intensity["clip"][1]),
    )
    doppler = validated["doppler"]
    result[2:3] = torch.clamp(
        result[2:3] / float(doppler["scale_mps"]),
        min=float(doppler["clip"][0]),
        max=float(doppler["clip"][1]),
    )
    return torch.where(occupied.expand_as(result), result, torch.zeros_like(result))


def radar_normalization_from_checkpoint(
    checkpoint: Any,
    *,
    target_size: Sequence[int],
    source_pc_range: Sequence[float],
    model_pc_range: Sequence[float],
    context: str,
) -> Tuple[dict, str]:
    """从 LDM/CD checkpoint 提取并验证内嵌的正式 normalization。"""
    mapping = _require_mapping(checkpoint, context)
    if "radar_normalization" not in mapping:
        raise RadarNormalizationError(f"{context} 缺少 radar_normalization")
    if "radar_normalization_sha256" not in mapping:
        raise RadarNormalizationError(f"{context} 缺少 radar_normalization_sha256")
    spec = mapping["radar_normalization"]
    doppler = spec.get("doppler") if isinstance(spec, Mapping) else None
    scale_mps = doppler.get("scale_mps") if isinstance(doppler, Mapping) else None
    validated = validate_radar_normalization_spec(
        spec,
        target_size=target_size,
        source_pc_range=source_pc_range,
        model_pc_range=model_pc_range,
        doppler_scale_mps=scale_mps,
        require_formal=True,
    )
    digest = validate_radar_normalization_sha256(
        mapping["radar_normalization_sha256"],
        context=context,
    )
    return validated, digest


def assert_checkpoint_radar_normalization(
    checkpoint: Any,
    expected_spec: Optional[Mapping],
    expected_sha256: str,
    *,
    target_size: Sequence[int],
    source_pc_range: Sequence[float],
    model_pc_range: Sequence[float],
    allow_legacy_radar_units: bool,
    context: str,
) -> None:
    """在加载模型/优化器和创建输出前校验 resume/父 checkpoint。"""
    mapping = _require_mapping(checkpoint, context)
    has_embedded = (
        "radar_normalization" in mapping
        or "radar_normalization_sha256" in mapping
    )
    if expected_spec is None:
        if not allow_legacy_radar_units:
            raise RadarNormalizationError(f"{context} 缺少正式 Radar normalization")
        if expected_sha256 or has_embedded:
            raise RadarNormalizationError(
                f"{context}: legacy 与正式 Radar normalization 不能混用"
            )
        return
    if allow_legacy_radar_units:
        raise RadarNormalizationError(
            f"{context}: 正式 Radar normalization 与 legacy 开关不能同时启用"
        )
    saved_spec, saved_sha256 = radar_normalization_from_checkpoint(
        mapping,
        target_size=target_size,
        source_pc_range=source_pc_range,
        model_pc_range=model_pc_range,
        context=context,
    )
    assert_same_radar_normalization(
        expected_spec,
        expected_sha256,
        saved_spec,
        saved_sha256,
        context=context,
    )


def assert_same_radar_normalization(
    left_spec: Mapping,
    left_sha256: str,
    right_spec: Mapping,
    right_sha256: str,
    *,
    context: str,
) -> None:
    """比较两阶段的完整 spec 与 artifact 文件 hash，拒绝部分匹配。"""
    label = str(context) if str(context) else "Radar normalization"
    for side, digest in (("left", left_sha256), ("right", right_sha256)):
        validate_radar_normalization_sha256(
            digest,
            context=f"{label}: {side}",
        )
    try:
        left_bytes = json.dumps(
            left_spec,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        right_bytes = json.dumps(
            right_spec,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RadarNormalizationError(f"{label}: spec 不是合法 JSON: {exc}") from exc
    if left_bytes != right_bytes:
        raise RadarNormalizationError(f"{label}: normalization spec 内容不一致")
    if left_sha256 != right_sha256:
        raise RadarNormalizationError(f"{label}: normalization artifact SHA-256 不一致")


__all__ = [
    "LEGACY_RADAR_NORMALIZATION_PROTOCOL",
    "RADAR_NORMALIZATION_PROTOCOL",
    "RadarNormalizationError",
    "apply_radar_normalization",
    "assert_checkpoint_radar_normalization",
    "assert_same_radar_normalization",
    "load_radar_normalization_artifact",
    "radar_normalization_from_checkpoint",
    "validate_radar_normalization_sha256",
    "validate_radar_normalization_spec",
]
