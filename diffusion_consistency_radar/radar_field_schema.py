# -*- coding: utf-8 -*-
"""文件功能：验证 Radar 原始点云字段、单位、符号和权威来源合同。"""

import hashlib
import json
import os
from typing import Dict, Mapping, Tuple


LEGACY_RADAR_FIELD_SCHEMA_PROTOCOL = "radar_raw_field_semantics_v1"
RADAR_FIELD_SCHEMA_PROTOCOL = "radar_raw_field_semantics_v2"
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
        "dB",
    }
)
_SUPPORTED_POINT_COORDINATE_FRAMES = frozenset({"radar", "lidar", "base_link"})
_HEADER_FRAME_RELATIONSHIPS = frozenset(
    {
        "matches_physical_coordinate_frame",
        "header_label_only_points_in_sensor_frame",
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
    expected_protocol: str = "",
) -> Dict[str, object]:
    """严格验证字段合同；正式路径只接受可追溯的 verified artifact。"""
    if not isinstance(value, Mapping):
        raise ValueError("Radar field schema 必须是 JSON 对象")
    protocol = value.get("protocol")
    if protocol not in {
        LEGACY_RADAR_FIELD_SCHEMA_PROTOCOL,
        RADAR_FIELD_SCHEMA_PROTOCOL,
    }:
        raise ValueError("Radar field schema protocol 不匹配")
    if expected_protocol and protocol != expected_protocol:
        raise ValueError(
            "Radar field schema protocol 与调用方要求不匹配: "
            f"{protocol!r} != {expected_protocol!r}"
        )
    is_v2 = protocol == RADAR_FIELD_SCHEMA_PROTOCOL
    schema = _exact_mapping(
        value,
        (
            {"protocol", "storage", "ros_transport", "fields", "verification"}
            if is_v2
            else {"protocol", "storage", "fields", "verification"}
        ),
        "Radar field schema",
    )

    storage = _exact_mapping(
        schema.get("storage"),
        {"format", "dtype", "column_count"},
        "Radar field schema.storage",
    )
    if storage != {"format": "npy", "dtype": "float32", "column_count": 5}:
        raise ValueError("Radar field schema storage 必须是五列 float32 NPY")

    ros_transport = None
    if is_v2:
        ros_transport = _exact_mapping(
            schema.get("ros_transport"),
            {"topic", "message_type", "header_frame_id"},
            "Radar field schema.ros_transport",
        )
        topic = _nonempty_string(ros_transport.get("topic"), "ros_transport.topic")
        if not topic.startswith("/"):
            raise ValueError("ros_transport.topic 必须是绝对 ROS topic")
        _nonempty_string(
            ros_transport.get("message_type"),
            "ros_transport.message_type",
        )
        _nonempty_string(
            ros_transport.get("header_frame_id"),
            "ros_transport.header_frame_id",
        )

    fields = _exact_mapping(
        schema.get("fields"),
        {"xyz", "return_strength", "doppler"},
        "Radar field schema.fields",
    )
    xyz = _exact_mapping(
        fields.get("xyz"),
        (
            {
                "column_indices",
                "unit",
                "physical_coordinate_frame",
                "header_frame_relationship",
            }
            if is_v2
            else {"column_indices", "unit", "coordinate_frame"}
        ),
        "Radar field schema.fields.xyz",
    )
    if xyz.get("column_indices") != [0, 1, 2] or xyz.get("unit") != "m":
        raise ValueError("Radar XYZ 必须固定为 0/1/2 列且单位为 m")
    if is_v2:
        physical_frame = xyz.get("physical_coordinate_frame")
        relationship = xyz.get("header_frame_relationship")
        if physical_frame not in _SUPPORTED_POINT_COORDINATE_FRAMES:
            raise ValueError("Radar raw XYZ physical_coordinate_frame 不支持")
        if relationship not in _HEADER_FRAME_RELATIONSHIPS:
            raise ValueError("Radar raw XYZ header_frame_relationship 不支持")
        if (
            relationship == "matches_physical_coordinate_frame"
            and ros_transport.get("header_frame_id") != physical_frame
        ):
            raise ValueError("ROS header frame 与物理点坐标系声明矛盾")
        if (
            relationship == "header_label_only_points_in_sensor_frame"
            and physical_frame != "radar"
        ):
            raise ValueError("header label-only 合同只允许点仍位于 radar sensor frame")
    elif xyz.get("coordinate_frame") != "radar":
        raise ValueError("legacy Radar raw XYZ coordinate_frame 必须为 radar")

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
    if (
        require_verified
        and not is_v2
        and expected_protocol != LEGACY_RADAR_FIELD_SCHEMA_PROTOCOL
    ):
        raise ValueError("正式 verified Radar field schema 必须使用 v2 坐标合同")

    if status == "verified":
        _nonempty_string(return_strength.get("source_field"), "return source_field")
        _nonempty_string(doppler.get("source_field"), "Doppler source_field")
        if return_strength.get("unit") not in _SUPPORTED_RETURN_UNITS:
            raise ValueError("verified Radar return strength unit 不支持")
        if is_v2 and (
            (return_strength.get("quantity") == "signal_to_noise_ratio")
            != (return_strength.get("unit") == "dB")
        ):
            raise ValueError("verified dB return strength 必须明确为 signal_to_noise_ratio")
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
    expected_protocol: str = "",
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
        expected_protocol=expected_protocol,
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
    field_protocol = radar_field_schema.get("protocol")
    is_v2 = field_protocol == RADAR_FIELD_SCHEMA_PROTOCOL
    layout = _exact_mapping(
        value,
        ({
            "schema_version",
            "storage_format",
            "ros_transport",
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
        } if is_v2 else {
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
        }),
        "Radar pointcloud layout schema",
    )
    expected_layout_version = 2 if is_v2 else 1
    if layout.get("schema_version") != expected_layout_version:
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
    if is_v2 and layout.get("ros_transport") != radar_field_schema.get("ros_transport"):
        raise ValueError("Radar layout ROS transport/topic/type/frame 与字段合同不一致")
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
    "LEGACY_RADAR_FIELD_SCHEMA_PROTOCOL",
    "RADAR_DOPPLER_POSITIVE_DIRECTIONS",
    "RADAR_FIELD_SCHEMA_PROTOCOL",
    "load_radar_field_schema_artifact",
    "load_radar_layout_schema",
    "validate_radar_layout_schema",
    "validate_radar_field_schema",
]
