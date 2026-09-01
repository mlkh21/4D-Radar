# -*- coding: utf-8 -*-
"""文件功能：统一 deployment observed-mask 的逐帧记录与摘要协议。"""

import hashlib
import os

import numpy as np


RADAR_ENDPOINT_RAY_OBSERVED_PROTOCOL = "radar_endpoint_ray_visibility_v1"


def observed_mask_records_digest(records) -> str:
    """按帧顺序绑定 observed-mask 文件内容和有效体素计数。"""
    digest = hashlib.sha256()
    for record in records:
        digest.update(str(record["frame_id"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["file"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["sha256"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(int(record["observed_voxels"])).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_observed_mask_metadata(records, identity=None):
    """构造可由 evaluator/map 逐帧复算的 Radar 射线 observed 合同。"""
    normalized = []
    seen = set()
    for record in records:
        frame_id = str(record.get("frame_id", ""))
        file_name = str(record.get("file", ""))
        file_sha256 = str(record.get("sha256", ""))
        observed_voxels = int(record.get("observed_voxels", -1))
        if (
            not frame_id
            or frame_id in seen
            or os.path.basename(file_name) != file_name
            or file_name != f"{frame_id}_observed_mask.npy"
            or len(file_sha256) != 64
            or any(character not in "0123456789abcdef" for character in file_sha256)
            or observed_voxels < 0
        ):
            raise ValueError("observed mask 记录格式无效或 frame 重复")
        seen.add(frame_id)
        normalized.append(
            {
                "frame_id": frame_id,
                "file": file_name,
                "sha256": file_sha256,
                "observed_voxels": observed_voxels,
            }
        )
    metadata = {
        "protocol": RADAR_ENDPOINT_RAY_OBSERVED_PROTOCOL,
        "coordinate_frame": "lidar",
        "source": "radar_endpoint_rays",
        "ir_frustum_marks_free_space": False,
        "frame_count": len(normalized),
        "observed_voxels": sum(record["observed_voxels"] for record in normalized),
        "files_sha256": observed_mask_records_digest(normalized),
        "records": normalized,
    }
    if identity is not None:
        origin = np.asarray(identity.get("radar_origin_lidar_m"), dtype=np.float64)
        calibration_sha256 = str(identity.get("radar_to_lidar_sha256", ""))
        if (
            origin.shape != (3,)
            or not np.all(np.isfinite(origin))
            or len(calibration_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in calibration_sha256
            )
        ):
            raise ValueError("Radar observed identity 的原点或标定 SHA-256 无效")
        metadata["radar_origin_lidar_m"] = origin.astype(float).tolist()
        metadata["radar_to_lidar_sha256"] = calibration_sha256
    return metadata
