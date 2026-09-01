#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""文件功能：定义并校验离线建图消费的逐帧 local-frame 轨迹合同。"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Sequence

import numpy as np


LOCAL_TRAJECTORY_ARTIFACT_PROTOCOL = "local_trajectory_frames_v1"


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _records_digest(records: List[Dict[str, object]]) -> str:
    payload = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_local_trajectory_artifact(
    path: str,
    expected_frame_ids: Sequence[str],
) -> Dict[str, object]:
    """加载轨迹 artifact，并严格绑定当次建图实际消费的帧集。"""
    normalized_path = os.path.abspath(os.path.expanduser(path))
    if os.path.islink(normalized_path) or not os.path.isfile(normalized_path):
        raise ValueError(f"trajectory artifact 必须是普通文件: {normalized_path}")
    with open(normalized_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("trajectory artifact 顶层必须是 JSON object")
    expected_top_level = {
        "protocol",
        "coordinate_frame",
        "frame_count",
        "records",
    }
    if set(payload) != expected_top_level:
        raise ValueError(
            "trajectory artifact 顶层字段必须精确为 "
            f"{sorted(expected_top_level)}"
        )
    if payload["protocol"] != LOCAL_TRAJECTORY_ARTIFACT_PROTOCOL:
        raise ValueError(f"trajectory protocol 不支持: {payload['protocol']}")
    if payload["coordinate_frame"] != "local":
        raise ValueError("trajectory coordinate_frame 必须为 local")
    frame_count = payload["frame_count"]
    if type(frame_count) is not int or frame_count < 0:
        raise ValueError("trajectory frame_count 必须是非负整数")
    records = payload["records"]
    if not isinstance(records, list) or frame_count != len(records):
        raise ValueError("trajectory frame_count 必须与 records 长度一致")

    expected = [str(frame_id) for frame_id in expected_frame_ids]
    if len(expected) != len(set(expected)):
        raise ValueError("expected trajectory frame 集合包含重复值")
    normalized_records: List[Dict[str, object]] = []
    trajectory_table: Dict[str, np.ndarray] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != {
            "frame_id",
            "waypoints_local_m",
        }:
            raise ValueError(
                f"trajectory record[{index}] 必须仅包含 frame_id/waypoints_local_m"
            )
        frame_id = record["frame_id"]
        if not isinstance(frame_id, str) or not frame_id:
            raise ValueError(f"trajectory record[{index}].frame_id 必须是非空字符串")
        if frame_id in trajectory_table:
            raise ValueError(f"trajectory frame 重复: {frame_id}")
        try:
            waypoints = np.asarray(record["waypoints_local_m"], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"trajectory frame {frame_id} waypoints 必须是数值数组"
            ) from exc
        if (
            waypoints.ndim != 2
            or waypoints.shape[0] < 2
            or waypoints.shape[1] != 3
            or not np.all(np.isfinite(waypoints))
        ):
            raise ValueError(
                f"trajectory frame {frame_id} 必须含至少两个有限 local XYZ 点"
            )
        if not np.any(np.linalg.norm(np.diff(waypoints, axis=0), axis=1) > 1e-6):
            raise ValueError(f"trajectory frame {frame_id} 轨迹长度必须大于 0")
        trajectory_table[frame_id] = waypoints
        normalized_records.append(
            {
                "frame_id": frame_id,
                "waypoints_local_m": waypoints.astype(float).tolist(),
            }
        )

    actual = [str(record["frame_id"]) for record in normalized_records]
    if actual != expected:
        raise ValueError(
            "trajectory frame 覆盖/顺序必须与建图消费帧精确一致: "
            f"expected={expected[:5]}, actual={actual[:5]}"
        )
    return {
        "protocol": LOCAL_TRAJECTORY_ARTIFACT_PROTOCOL,
        "coordinate_frame": "local",
        "frame_count": frame_count,
        "records_sha256": _records_digest(normalized_records),
        "artifact_path": normalized_path,
        "artifact_sha256": _sha256_file(normalized_path),
        "trajectory_table": trajectory_table,
    }
