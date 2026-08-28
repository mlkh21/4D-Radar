# -*- coding: utf-8 -*-
"""文件功能：统一正式 inference prediction voxel 的记录格式与摘要算法。"""

import hashlib
import os
from typing import Dict, Iterable, List


PREDICTION_VOXEL_PROTOCOL = "generated_voxel_artifact_v1"


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
    return {
        "protocol": PREDICTION_VOXEL_PROTOCOL,
        "coordinate_frame": "lidar",
        "layout": "czxy",
        "frame_count": len(normalized),
        "records_sha256": prediction_voxel_records_digest(normalized),
        "records": normalized,
    }
