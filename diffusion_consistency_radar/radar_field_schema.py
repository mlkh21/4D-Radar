# -*- coding: utf-8 -*-
"""文件功能：验证 Radar 原始点云字段、单位、符号和权威来源合同。"""

import hashlib
import json
import os
from typing import Dict, Mapping, Tuple


RADAR_FIELD_SCHEMA_PROTOCOL = "radar_raw_field_semantics_v1"
RADAR_DOPPLER_POSITIVE_DIRECTIONS = frozenset(
    {"toward_sensor", "away_from_sensor"}
)
_VERIFIED_AUTHORITY_TYPES = frozenset(
    {
        "sensor_manual",
        "official_message_definition",
        "dataset_provider_documentation",
    }
)
_RETURN_QUANTITIES = frozenset(
    {
        "intensity",
        "reflectivity",
        "power",
        "radar_cross_section",
        "signal_to_noise_ratio",
    }
)
_SUPPORTED_RETURN_UNITS = frozenset(
    {
        "sensor_native_linear_nonnegative",
        "linear_power",
        "m2",
        "linear_ratio",
    }
)


def _exact_mapping(value, expected_keys, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != set(expected_keys):
        raise ValueError(f"{name} 字段必须精确为 {sorted(expected_keys)}")
    return value


def _nonempty_string(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须是非空字符串")
    return value


def _is_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_radar_field_schema(
    value: Mapping[str, object],
    *,
    require_verified: bool = False,
) -> Dict[str, object]:
    """严格验证字段合同；正式路径只接受可追溯的 verified artifact。"""
    schema = _exact_mapping(
        value,
        {"protocol", "storage", "fields", "verification"},
        "Radar field schema",
    )
    if schema.get("protocol") != RADAR_FIELD_SCHEMA_PROTOCOL:
        raise ValueError("Radar field schema protocol 不匹配")

    storage = _exact_mapping(
        schema.get("storage"),
        {"format", "dtype", "column_count"},
        "Radar field schema.storage",
    )
    if storage != {"format": "npy", "dtype": "float32", "column_count": 5}:
        raise ValueError("Radar field schema storage 必须是五列 float32 NPY")

    fields = _exact_mapping(
        schema.get("fields"),
        {"xyz", "return_strength", "doppler"},
        "Radar field schema.fields",
    )
    xyz = _exact_mapping(
        fields.get("xyz"),
        {"column_indices", "unit", "coordinate_frame"},
        "Radar field schema.fields.xyz",
    )
    if xyz.get("column_indices") != [0, 1, 2] or xyz.get("unit") != "m":
        raise ValueError("Radar XYZ 必须固定为 0/1/2 列且单位为 m")
    if xyz.get("coordinate_frame") != "radar":
        raise ValueError("Radar raw XYZ coordinate_frame 必须为 radar")

    return_strength = _exact_mapping(
        fields.get("return_strength"),
        {"column_index", "source_field", "quantity", "unit", "missing_value"},
        "Radar field schema.fields.return_strength",
    )
    if return_strength.get("column_index") != 3:
        raise ValueError("Radar return strength 必须位于第 3 列")
    if return_strength.get("missing_value") != "nan":
        raise ValueError("Radar return strength 缺失值必须编码为 nan")
    if return_strength.get("quantity") not in _RETURN_QUANTITIES:
        raise ValueError("Radar return strength quantity 不支持")

    doppler = _exact_mapping(
        fields.get("doppler"),
        {
            "column_index",
            "source_field",
            "quantity",
            "unit",
            "reference",
            "positive_direction",
            "missing_value",
        },
        "Radar field schema.fields.doppler",
    )
    if doppler.get("column_index") != 4:
        raise ValueError("Radar Doppler 必须位于第 4 列")
    if doppler.get("quantity") != "radial_velocity":
        raise ValueError("Radar Doppler quantity 必须为 radial_velocity")
    if doppler.get("reference") != "sensor_relative":
        raise ValueError("Radar Doppler reference 必须为 sensor_relative")
    if doppler.get("missing_value") != "nan":
        raise ValueError("Radar Doppler 缺失值必须编码为 nan")

    verification = _exact_mapping(
        schema.get("verification"),
        {"status", "authority_type", "reference", "evidence_file", "sha256"},
        "Radar field schema.verification",
    )
    status = verification.get("status")
    if status not in {"verified", "unverified"}:
        raise ValueError("Radar field schema verification.status 不支持")
    if require_verified and status != "verified":
        raise ValueError("Radar field schema 未通过权威验证，禁止用于正式链")

    if status == "verified":
        _nonempty_string(return_strength.get("source_field"), "return source_field")
        _nonempty_string(doppler.get("source_field"), "Doppler source_field")
        if return_strength.get("unit") not in _SUPPORTED_RETURN_UNITS:
            raise ValueError("verified Radar return strength unit 不支持")
        if doppler.get("unit") != "m/s":
            raise ValueError("verified Radar Doppler unit 必须为 m/s")
        if doppler.get("positive_direction") not in RADAR_DOPPLER_POSITIVE_DIRECTIONS:
            raise ValueError("verified Radar Doppler 正方向不支持")
        if verification.get("authority_type") not in _VERIFIED_AUTHORITY_TYPES:
            raise ValueError("verified Radar schema authority_type 不支持")
        _nonempty_string(verification.get("reference"), "verification.reference")
        evidence_file = _nonempty_string(
            verification.get("evidence_file"),
            "verification.evidence_file",
        )
        if os.path.isabs(evidence_file) or ".." in evidence_file.split(os.sep):
            raise ValueError("verification.evidence_file 必须是安全相对路径")
        if not _is_sha256(verification.get("sha256")):
            raise ValueError("verification.sha256 必须是小写 SHA-256")

    # JSON roundtrip 生成普通 dict/list，避免调用方持有可变 Mapping 子类。
    return json.loads(json.dumps(schema, ensure_ascii=False))


def load_radar_field_schema_artifact(
    path: str,
    *,
    require_verified: bool = False,
) -> Tuple[Dict[str, object], str]:
    """加载普通 JSON 文件，返回严格 schema 与文件内容 SHA-256。"""
    artifact_path = os.path.abspath(os.fspath(path))
    if os.path.islink(artifact_path) or not os.path.isfile(artifact_path):
        raise ValueError(f"Radar field schema 必须是普通文件: {artifact_path}")
    with open(artifact_path, "rb") as handle:
        payload = handle.read()
    digest = hashlib.sha256(payload).hexdigest()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Radar field schema 不是合法 UTF-8 JSON: {artifact_path}") from exc
    schema = validate_radar_field_schema(
        value,
        require_verified=require_verified,
    )
    if schema["verification"]["status"] == "verified":
        artifact_dir = os.path.dirname(artifact_path)
        evidence_path = os.path.abspath(
            os.path.join(
                artifact_dir,
                schema["verification"]["evidence_file"],
            )
        )
        if (
            os.path.commonpath([artifact_dir, evidence_path]) != artifact_dir
            or os.path.islink(evidence_path)
            or not os.path.isfile(evidence_path)
        ):
            raise ValueError("Radar field schema 权威证据文件不存在或路径不安全")
        with open(evidence_path, "rb") as handle:
            evidence_digest = hashlib.sha256(handle.read()).hexdigest()
        if evidence_digest != schema["verification"]["sha256"]:
            raise ValueError("Radar field schema 权威证据 SHA-256 不匹配")
    return schema, digest


def validate_radar_layout_schema(
    value: Mapping[str, object],
    radar_field_schema: Mapping[str, object],
) -> Dict[str, object]:
    """交叉核对解包列布局与经验证的物理字段身份。"""
    layout = _exact_mapping(
        value,
        {
            "schema_version",
            "storage_format",
            "columns",
            "column_indices",
            "source_fields",
            "selected_fields",
            "field_mapping",
            "missing_fields",
            "physical_semantics_status",
            "missing_value_encoding",
            "shape",
            "dtype",
        },
        "Radar pointcloud layout schema",
    )
    if layout.get("schema_version") != 1:
        raise ValueError("Radar pointcloud layout schema_version 不支持")
    if layout.get("storage_format") != "npy" or layout.get("dtype") != "float32":
        raise ValueError("Radar pointcloud layout 必须是 float32 NPY")
    expected_columns = ["x", "y", "z", "intensity", "doppler"]
    if layout.get("columns") != expected_columns:
        raise ValueError("Radar pointcloud layout columns 不匹配")
    if layout.get("column_indices") != {
        name: index for index, name in enumerate(expected_columns)
    }:
        raise ValueError("Radar pointcloud layout column_indices 不匹配")
    if layout.get("physical_semantics_status") != "unverified_layout_only":
        raise ValueError("解包 layout 不得自行声称已验证物理语义")
    if layout.get("missing_value_encoding") != "nan":
        raise ValueError("Radar pointcloud layout 缺失值必须编码为 nan")
    if layout.get("missing_fields") != []:
        raise ValueError("正式 Radar layout 不允许缺少 return strength 或 Doppler")
    shape = layout.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or not isinstance(shape[0], int)
        or shape[0] < 0
        or shape[1] != 5
    ):
        raise ValueError("Radar pointcloud layout shape 必须为 [N,5]")

    validated_fields = radar_field_schema["fields"]
    expected_mapping = {
        "x": "x",
        "y": "y",
        "z": "z",
        "intensity": validated_fields["return_strength"]["source_field"],
        "doppler": validated_fields["doppler"]["source_field"],
    }
    if layout.get("field_mapping") != expected_mapping:
        raise ValueError("Radar layout source field 与 field semantics artifact 不一致")
    selected_fields = layout.get("selected_fields")
    if selected_fields != list(expected_mapping.values()):
        raise ValueError("Radar layout selected_fields 顺序与物理字段合同不一致")
    source_fields = layout.get("source_fields")
    if not isinstance(source_fields, list) or not set(expected_mapping.values()).issubset(
        set(source_fields)
    ):
        raise ValueError("Radar layout source_fields 缺少已绑定字段")
    return json.loads(json.dumps(layout, ensure_ascii=False))


def load_radar_layout_schema(
    path: str,
    radar_field_schema: Mapping[str, object],
) -> Tuple[Dict[str, object], str]:
    """加载并交叉验证解包 layout sidecar。"""
    layout_path = os.path.abspath(os.fspath(path))
    if os.path.islink(layout_path) or not os.path.isfile(layout_path):
        raise ValueError(f"Radar pointcloud layout 必须是普通文件: {layout_path}")
    with open(layout_path, "rb") as handle:
        payload = handle.read()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Radar pointcloud layout 不是合法 UTF-8 JSON") from exc
    return (
        validate_radar_layout_schema(value, radar_field_schema),
        hashlib.sha256(payload).hexdigest(),
    )


__all__ = [
    "RADAR_DOPPLER_POSITIVE_DIRECTIONS",
    "RADAR_FIELD_SCHEMA_PROTOCOL",
    "load_radar_field_schema_artifact",
    "load_radar_layout_schema",
    "validate_radar_layout_schema",
    "validate_radar_field_schema",
]
