# -*- coding: utf-8 -*-
"""文件功能：统一正式 inference prediction voxel 的通道、记录与地图消费合同。"""

import copy
import hashlib
import os
from typing import Dict, Iterable, List


PREDICTION_VOXEL_PROTOCOL = "generated_voxel_artifact_v2"
PREDICTION_MAPPING_PROTOCOL = "generated_occupancy_mapping_input_v1"
GENERATED_OCCUPANCY_EVIDENCE_SEMANTICS = "generated_occupancy_probability_v1"

PREDICTION_CHANNEL_SCHEMA = [
    {
        "index": 0,
        "name": "occupancy_probability",
        "value_semantics": "probability",
        "range": [0.0, 1.0],
        "mapping_role": "occupancy_evidence",
    },
    {
        "index": 1,
        "name": "auxiliary_lidar_intensity_reconstruction",
        "value_semantics": "unbounded_decoder_output",
        "mapping_role": "not_consumed",
    },
    {
        "index": 2,
        "name": "auxiliary_radar_doppler_reconstruction",
        "value_semantics": "unbounded_decoder_output",
        "mapping_role": "not_consumed",
    },
    {
        "index": 3,
        "name": "auxiliary_doppler_validity_reconstruction",
        "value_semantics": "unbounded_decoder_output",
        "mapping_role": "not_consumed",
    },
]

PREDICTION_MAPPING_CONTRACT = {
    "protocol": PREDICTION_MAPPING_PROTOCOL,
    "evidence_semantics": GENERATED_OCCUPANCY_EVIDENCE_SEMANTICS,
    "occupancy_channel": 0,
    "occupancy_semantics": "probability",
    "occupancy_range": [0.0, 1.0],
    "observed_domain": "external_authoritative_mask",
    "auxiliary_channels_consumed": False,
    "dem_height_source": "observed_occupancy_z_distribution",
    "dem_variance_unit": "m^2",
}


def normalize_prediction_voxel_records(
    records: Iterable[Dict[str, object]],
) -> List[Dict[str, object]]:
    """规范化逐帧 prediction 记录，并拒绝重复帧和隐式布局。"""
    normalized = []
    seen = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("prediction voxel record 必须是对象")
        frame_id = str(record.get("frame_id", ""))
        file_name = str(record.get("file", ""))
        file_sha256 = str(record.get("sha256", ""))
        shape = record.get("shape_czxy")
        dtype = str(record.get("dtype", ""))
        valid_shape = (
            isinstance(shape, (list, tuple))
            and len(shape) == 4
            and all(type(value) is int and value > 0 for value in shape)
        )
        if (
            not frame_id
            or frame_id in seen
            or file_name != f"{frame_id}_voxel.npy"
            or os.path.basename(file_name) != file_name
            or len(file_sha256) != 64
            or any(character not in "0123456789abcdef" for character in file_sha256)
            or not valid_shape
            or dtype != "float32"
        ):
            raise ValueError("prediction voxel 记录格式无效或 frame 重复")
        seen.add(frame_id)
        normalized.append(
            {
                "frame_id": frame_id,
                "file": file_name,
                "sha256": file_sha256,
                "shape_czxy": [int(value) for value in shape],
                "dtype": dtype,
            }
        )
    return normalized


def prediction_voxel_records_digest(records: Iterable[Dict[str, object]]) -> str:
    """按帧顺序绑定 prediction voxel 的内容、形状和 dtype。"""
    digest = hashlib.sha256()
    for record in records:
        digest.update(str(record["frame_id"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["file"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["sha256"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(
            ",".join(
                str(int(value)) for value in record["shape_czxy"]
            ).encode("ascii")
        )
        digest.update(b"\0")
        digest.update(str(record["dtype"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_prediction_voxel_metadata(records) -> Dict[str, object]:
    """构造地图入口可逐帧重算的正式 prediction voxel 合同。"""
    normalized = normalize_prediction_voxel_records(records)
    metadata = {
        "protocol": PREDICTION_VOXEL_PROTOCOL,
        "coordinate_frame": "lidar",
        "layout": "czxy",
        "channel_axis": 0,
        "channels": copy.deepcopy(PREDICTION_CHANNEL_SCHEMA),
        "mapping_contract": copy.deepcopy(PREDICTION_MAPPING_CONTRACT),
        "frame_count": len(normalized),
        "records_sha256": prediction_voxel_records_digest(normalized),
        "records": normalized,
    }
    validate_prediction_voxel_metadata(metadata)
    return metadata


def validate_prediction_voxel_metadata(metadata: Dict[str, object]) -> None:
    """严格验证正式 prediction 通道和地图消费身份。"""
    if not isinstance(metadata, dict):
        raise ValueError("prediction voxel metadata 必须是对象")
    records = metadata.get("records")
    if (
        metadata.get("protocol") != PREDICTION_VOXEL_PROTOCOL
        or metadata.get("coordinate_frame") != "lidar"
        or metadata.get("layout") != "czxy"
        or metadata.get("channel_axis") != 0
        or metadata.get("channels") != PREDICTION_CHANNEL_SCHEMA
        or metadata.get("mapping_contract") != PREDICTION_MAPPING_CONTRACT
        or type(metadata.get("frame_count")) is not int
        or metadata.get("frame_count") < 0
        or not isinstance(records, list)
        or len(records) != metadata.get("frame_count")
    ):
        raise ValueError("prediction voxel metadata 通道或地图消费合同不完整")
    normalized = normalize_prediction_voxel_records(records)
    if any(record["shape_czxy"][0] != len(PREDICTION_CHANNEL_SCHEMA) for record in normalized):
        raise ValueError("正式 prediction voxel 必须恰好包含四个声明通道")
    if prediction_voxel_records_digest(normalized) != metadata.get("records_sha256"):
        raise ValueError("prediction voxel records SHA-256 不匹配")
