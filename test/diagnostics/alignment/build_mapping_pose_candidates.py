#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""文件功能：组合 LiDAR→body 外参候选，并按显式参考时间生成双 frame 位姿诊断。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diffusion_consistency_radar.geometry_protocol import (  # noqa: E402
    load_extrinsic_transform,
)


PROTOCOL = "mapping_pose_candidate_diagnostic_v1"
PROTOCOL_LIDAR_TIME = "mapping_pose_candidate_diagnostic_v2"
POSE_FIELDNAMES = (
    "frame",
    "timestamp",
    "tx",
    "ty",
    "tz",
    "qx",
    "qy",
    "qz",
    "qw",
    "diagnostic_formal",
    "gt_pose_hypothesis",
)


def _regular_file(path: str, label: str) -> str:
    """返回绝对路径，拒绝符号链接和非普通文件。"""
    normalized = os.path.abspath(os.fspath(path))
    if os.path.islink(normalized) or not os.path.isfile(normalized):
        raise ValueError(f"{label}必须是普通文件: {normalized}")
    return normalized


def _sha256_file(path: str) -> str:
    """流式计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_transform(matrix: np.ndarray, label: str) -> np.ndarray:
    """校验齐次刚体变换，不对非正交矩阵做静默修复。"""
    transform = np.asarray(matrix, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"{label}必须是有限 4x4 矩阵")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8, rtol=0.0):
        raise ValueError(f"{label}齐次底行必须为 0 0 0 1")
    rotation = transform[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=5e-3, rtol=0.0):
        raise ValueError(f"{label}旋转矩阵不正交")
    determinant = float(np.linalg.det(rotation))
    if abs(determinant - 1.0) > 5e-3:
        raise ValueError(f"{label}旋转 determinant 必须接近 1: {determinant}")
    return transform


def load_matrix4_transform(path: str) -> np.ndarray:
    """严格加载旧式 4×4 外参文本。"""
    normalized = _regular_file(path, "Radar→IMU 候选外参")
    try:
        matrix = np.loadtxt(normalized, dtype=np.float64, comments="#")
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"4x4 外参无法解析: {normalized}") from exc
    return _validate_transform(matrix, "Radar→IMU 候选外参")


def _load_ground_truth(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取 `timestamp tx ty tz qx qy qz qw` 并严格校验。"""
    normalized = _regular_file(path, "ground truth odometry")
    try:
        values = np.loadtxt(
            normalized,
            dtype=np.float64,
            comments="#",
            ndmin=2,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"ground truth odometry 无法解析: {normalized}") from exc
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != 8:
        raise ValueError("ground truth 必须至少两行且每行恰好 8 列")
    if not np.all(np.isfinite(values)):
        raise ValueError("ground truth 含非有限数")
    timestamps = values[:, 0]
    if np.any(np.diff(timestamps) <= 0.0) or np.any(timestamps < 0.0):
        raise ValueError("ground truth timestamp 必须有限、非负且严格递增")
    translations = values[:, 1:4]
    quaternions = values[:, 4:8]
    norms = np.linalg.norm(quaternions, axis=1)
    if np.any(np.abs(norms - 1.0) > 1e-3):
        raise ValueError("ground truth 四元数必须归一化")
    quaternions = quaternions / norms[:, np.newaxis]
    return timestamps, translations, quaternions


def _load_radar_timestamps(path: str, frame_width: int) -> List[Dict[str, object]]:
    """读取严格连续的 frame_index 和 Radar timestamp。"""
    if type(frame_width) is not int or frame_width <= 0:
        raise ValueError("frame_width 必须为正整数")
    normalized = _regular_file(path, "Radar sync CSV")
    records: List[Dict[str, object]] = []
    with open(normalized, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"frame_index", "radar_timestamp"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError("Radar sync CSV 缺少 frame_index/radar_timestamp")
        for expected_index, row in enumerate(reader):
            try:
                frame_index = int(row["frame_index"])
                timestamp = float(row["radar_timestamp"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Radar sync CSV 第 {expected_index + 2} 行格式错误") from exc
            if frame_index != expected_index:
                raise ValueError("Radar sync frame_index 必须从 0 严格连续")
            if not math.isfinite(timestamp) or timestamp < 0.0:
                raise ValueError("Radar timestamp 必须是有限非负数")
            records.append(
                {
                    "frame_id": f"{frame_index:0{frame_width}d}",
                    "timestamp": timestamp,
                }
            )
    if not records:
        raise ValueError("Radar sync CSV 不得为空")
    timestamps = [float(record["timestamp"]) for record in records]
    if any(after <= before for before, after in zip(timestamps, timestamps[1:])):
        raise ValueError("Radar timestamp 必须严格递增")
    return records


def _load_lidar_reference_timestamps(
    path: str,
    radar_records: Sequence[Dict[str, object]],
    frame_width: int,
) -> List[Dict[str, object]]:
    """读取 Radar--LiDAR 收据，并与 frame/Radar 时间逐行交叉验证。"""
    normalized = _regular_file(path, "Radar--LiDAR sync CSV")
    records: List[Dict[str, object]] = []
    with open(normalized, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "pair_index",
            "radar_timestamp",
            "lidar_timestamp",
            "delta_seconds",
            "signed_delta_seconds",
        }
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError("Radar--LiDAR sync CSV 字段不完整")
        for expected_index, row in enumerate(reader):
            if expected_index >= len(radar_records):
                raise ValueError("Radar--LiDAR sync 行数多于 Radar frame")
            try:
                pair_index = int(row["pair_index"])
                radar_timestamp = float(row["radar_timestamp"])
                lidar_timestamp = float(row["lidar_timestamp"])
                delta_seconds = float(row["delta_seconds"])
                signed_delta_seconds = float(row["signed_delta_seconds"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Radar--LiDAR sync 第 {expected_index + 2} 行格式错误"
                ) from exc
            if pair_index != expected_index:
                raise ValueError("Radar--LiDAR pair_index 必须从 0 严格连续")
            values = (
                radar_timestamp,
                lidar_timestamp,
                delta_seconds,
                signed_delta_seconds,
            )
            if not all(math.isfinite(value) for value in values):
                raise ValueError("Radar--LiDAR sync 含非有限数")
            expected_radar = float(radar_records[expected_index]["timestamp"])
            if not math.isclose(
                radar_timestamp,
                expected_radar,
                abs_tol=5e-7,
                rel_tol=0.0,
            ):
                raise ValueError(
                    f"Radar--LiDAR sync 与 Radar frame 时间不一致: {expected_index}"
                )
            measured_signed = lidar_timestamp - radar_timestamp
            if not math.isclose(
                signed_delta_seconds,
                measured_signed,
                abs_tol=1e-6,
                rel_tol=0.0,
            ) or not math.isclose(
                delta_seconds,
                abs(measured_signed),
                abs_tol=1e-6,
                rel_tol=0.0,
            ):
                raise ValueError(
                    f"Radar--LiDAR sync delta 与 timestamp 不一致: {expected_index}"
                )
            records.append(
                {
                    "frame_id": f"{pair_index:0{frame_width}d}",
                    "timestamp": lidar_timestamp,
                }
            )
    if len(records) != len(radar_records):
        raise ValueError(
            "Radar--LiDAR sync 行数必须与 Radar frame 一致: "
            f"sync={len(records)}, radar={len(radar_records)}"
        )
    timestamps = [float(record["timestamp"]) for record in records]
    if any(after <= before for before, after in zip(timestamps, timestamps[1:])):
        raise ValueError("LiDAR reference timestamp 必须严格递增")
    return records


def slerp_xyzw(first: Sequence[float], second: Sequence[float], alpha: float) -> np.ndarray:
    """在 xyzw 四元数之间执行最短弧 SLERP。"""
    q0 = np.asarray(first, dtype=np.float64)
    q1 = np.asarray(second, dtype=np.float64)
    if q0.shape != (4,) or q1.shape != (4,) or not math.isfinite(float(alpha)):
        raise ValueError("SLERP 要求两个 xyzw 四元数和有限 alpha")
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("SLERP alpha 必须位于 [0,1]")
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = q0 + float(alpha) * (q1 - q0)
        return result / np.linalg.norm(result)
    theta = math.acos(dot)
    denominator = math.sin(theta)
    result = (
        math.sin((1.0 - float(alpha)) * theta) / denominator * q0
        + math.sin(float(alpha) * theta) / denominator * q1
    )
    return result / np.linalg.norm(result)


def _quaternion_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    """把 xyzw 单位四元数转换为主动旋转矩阵。"""
    qx, qy, qz, qw = np.asarray(quaternion, dtype=np.float64)
    return np.asarray(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def _matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """把正交旋转矩阵转换为确定符号的 xyzw 四元数。"""
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (matrix[2, 1] - matrix[1, 2]) / scale
        qy = (matrix[0, 2] - matrix[2, 0]) / scale
        qz = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            qw = (matrix[2, 1] - matrix[1, 2]) / scale
            qx = 0.25 * scale
            qy = (matrix[0, 1] + matrix[1, 0]) / scale
            qz = (matrix[0, 2] + matrix[2, 0]) / scale
        elif axis == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            qw = (matrix[0, 2] - matrix[2, 0]) / scale
            qx = (matrix[0, 1] + matrix[1, 0]) / scale
            qy = 0.25 * scale
            qz = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            qw = (matrix[1, 0] - matrix[0, 1]) / scale
            qx = (matrix[0, 2] + matrix[2, 0]) / scale
            qy = (matrix[1, 2] + matrix[2, 1]) / scale
            qz = 0.25 * scale
    quaternion = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    return quaternion


def _pose_matrix(translation: Sequence[float], quaternion: Sequence[float]) -> np.ndarray:
    """由 local-frame 平移和 body→local 四元数构造齐次变换。"""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _quaternion_to_matrix(quaternion)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return _validate_transform(transform, "插值 pose")


def _interpolate_pose(
    timestamp: float,
    gt_times: np.ndarray,
    gt_translations: np.ndarray,
    gt_quaternions: np.ndarray,
    max_gap_s: float,
) -> Tuple[Optional[np.ndarray], Dict[str, object]]:
    """只在 GT 包围且 gap 不超限时插值，不做任何外推。"""
    if timestamp < float(gt_times[0]):
        return None, {"reason": "before_ground_truth"}
    if timestamp > float(gt_times[-1]):
        return None, {"reason": "after_ground_truth"}
    right = int(np.searchsorted(gt_times, timestamp, side="left"))
    if right < len(gt_times) and math.isclose(
        float(gt_times[right]), timestamp, abs_tol=1e-9, rel_tol=0.0
    ):
        return _pose_matrix(gt_translations[right], gt_quaternions[right]), {
            "reason": "exact",
            "gap_s": 0.0,
            "alpha": 0.0,
        }
    if right <= 0 or right >= len(gt_times):
        raise RuntimeError("插值搜索边界与时间范围检查不一致")
    left = right - 1
    gap_s = float(gt_times[right] - gt_times[left])
    if gap_s > max_gap_s:
        return None, {
            "reason": "interpolation_gap_exceeded",
            "gap_s": gap_s,
            "left_timestamp": float(gt_times[left]),
            "right_timestamp": float(gt_times[right]),
        }
    alpha = float((timestamp - float(gt_times[left])) / gap_s)
    translation = (1.0 - alpha) * gt_translations[left] + alpha * gt_translations[right]
    quaternion = slerp_xyzw(gt_quaternions[left], gt_quaternions[right], alpha)
    return _pose_matrix(translation, quaternion), {
        "reason": "interpolated",
        "gap_s": gap_s,
        "alpha": alpha,
    }


def _check_fresh_output_target(path: str) -> str:
    """在计算前检查输出目标，但不创建目录。"""
    normalized = os.path.abspath(os.fspath(path))
    if os.path.islink(normalized):
        raise ValueError(f"输出目录不得为符号链接: {normalized}")
    if os.path.exists(normalized):
        if not os.path.isdir(normalized):
            raise ValueError(f"输出路径必须是目录: {normalized}")
        if os.listdir(normalized):
            raise ValueError(f"拒绝覆盖非空输出目录: {normalized}")
    return normalized


def _write_text_atomic(path: str, content: str) -> None:
    """在目标目录内原子发布 UTF-8 文本。"""
    directory = os.path.dirname(path)
    fd, temporary = tempfile.mkstemp(prefix=".tmp_mapping_pose_", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_json_atomic(path: str, payload: Dict[str, object]) -> None:
    """原子发布排序诊断 JSON。"""
    _write_text_atomic(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _format_extrinsic_candidate(transform: np.ndarray) -> str:
    """输出可审计的 R/T 候选，文件头明确禁止 formal 使用。"""
    rotation = " ".join(f"{value:.12f}" for value in transform[:3, :3].reshape(-1))
    translation = " ".join(f"{value:.12f}" for value in transform[:3, 3])
    return (
        "# DIAGNOSTIC CANDIDATE ONLY; formal=false\n"
        "# 未验证假设: body=IMU, 4x4 文件表示 T_imu_radar。\n"
        "# p_body = R * p_lidar + T\n"
        f"R: {rotation}\n"
        f"T: {translation}\n"
    )


def _pose_row(
    frame_id: str,
    timestamp: float,
    transform: np.ndarray,
    hypothesis: str,
) -> Dict[str, object]:
    """把 `T_local_body` 转换成地图 loader 可读的候选行。"""
    quaternion = _matrix_to_quaternion(transform[:3, :3])
    translation = transform[:3, 3]
    return {
        "frame": frame_id,
        "timestamp": f"{timestamp:.9f}",
        "tx": f"{translation[0]:.12f}",
        "ty": f"{translation[1]:.12f}",
        "tz": f"{translation[2]:.12f}",
        "qx": f"{quaternion[0]:.12f}",
        "qy": f"{quaternion[1]:.12f}",
        "qz": f"{quaternion[2]:.12f}",
        "qw": f"{quaternion[3]:.12f}",
        "diagnostic_formal": "false",
        "gt_pose_hypothesis": hypothesis,
    }


def _write_pose_csv_atomic(path: str, rows: Sequence[Dict[str, object]]) -> None:
    """原子写入候选 pose CSV，即使零覆盖也保留表头。"""
    directory = os.path.dirname(path)
    fd, temporary = tempfile.mkstemp(prefix=".tmp_mapping_pose_", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=POSE_FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _finite_summary(values: Sequence[float]) -> Dict[str, Optional[float]]:
    """生成紧凑有限数分位统计。"""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"min": None, "median": None, "p95": None, "max": None}
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def build_mapping_pose_candidates(
    *,
    radar_to_imu_matrix_path: str,
    radar_to_lidar_path: str,
    ground_truth_path: str,
    radar_sync_csv_path: str,
    output_dir: str,
    radar_lidar_sync_csv_path: Optional[str] = None,
    pose_reference_sensor: str = "radar",
    max_interpolation_gap_s: float = 0.2,
    frame_width: int = 6,
    command_line: Optional[str] = None,
) -> Dict[str, object]:
    """构建不可冒充 formal 的外参和两套 pose-frame 候选。"""
    if not math.isfinite(float(max_interpolation_gap_s)) or max_interpolation_gap_s <= 0.0:
        raise ValueError("max_interpolation_gap_s 必须为正有限数")
    normalized_output = _check_fresh_output_target(output_dir)

    radar_to_imu_path = _regular_file(
        radar_to_imu_matrix_path,
        "Radar→IMU 候选外参",
    )
    radar_to_lidar_normalized = _regular_file(
        radar_to_lidar_path,
        "Radar→LiDAR 外参",
    )
    ground_truth_normalized = _regular_file(ground_truth_path, "ground truth odometry")
    radar_sync_normalized = _regular_file(radar_sync_csv_path, "Radar sync CSV")

    T_imu_radar = load_matrix4_transform(radar_to_imu_path)
    T_lidar_radar = _validate_transform(
        load_extrinsic_transform(radar_to_lidar_normalized),
        "Radar→LiDAR 外参",
    )
    T_body_lidar = _validate_transform(
        T_imu_radar @ np.linalg.inv(T_lidar_radar),
        "LiDAR→body 候选",
    )
    T_lidar_body = np.linalg.inv(T_body_lidar)

    gt_times, gt_translations, gt_quaternions = _load_ground_truth(
        ground_truth_normalized
    )
    radar_records = _load_radar_timestamps(radar_sync_normalized, frame_width)
    reference_sensor = str(pose_reference_sensor).strip().lower()
    if reference_sensor not in {"radar", "lidar"}:
        raise ValueError("pose_reference_sensor 必须是 radar 或 lidar")
    radar_lidar_sync_normalized: Optional[str] = None
    protocol = PROTOCOL
    if reference_sensor == "lidar":
        if radar_lidar_sync_csv_path is None:
            raise ValueError(
                "pose_reference_sensor=lidar 时必须提供 radar_lidar_sync_csv_path"
            )
        radar_lidar_sync_normalized = _regular_file(
            radar_lidar_sync_csv_path,
            "Radar--LiDAR sync CSV",
        )
        radar_records = _load_lidar_reference_timestamps(
            radar_lidar_sync_normalized,
            radar_records,
            frame_width,
        )
        protocol = PROTOCOL_LIDAR_TIME
    elif radar_lidar_sync_csv_path is not None:
        raise ValueError(
            "pose_reference_sensor=radar 时不得传入未使用的 radar_lidar_sync_csv_path"
        )

    imu_rows: List[Dict[str, object]] = []
    lidar_rows: List[Dict[str, object]] = []
    uncovered_records: List[Dict[str, object]] = []
    interpolation_gaps: List[float] = []
    for record in radar_records:
        frame_id = str(record["frame_id"])
        timestamp = float(record["timestamp"])
        T_local_gt, interpolation = _interpolate_pose(
            timestamp,
            gt_times,
            gt_translations,
            gt_quaternions,
            float(max_interpolation_gap_s),
        )
        if T_local_gt is None:
            uncovered_records.append(
                {
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    **interpolation,
                }
            )
            continue
        gap_s = float(interpolation.get("gap_s", 0.0))
        interpolation_gaps.append(gap_s)
        # 假设 1：GT 已是 IMU/body→local，直接作为候选。
        imu_rows.append(
            _pose_row(frame_id, timestamp, T_local_gt, "gt_pose_is_imu")
        )
        # 假设 2：GT 是 LiDAR→local，则
        # T_local_lidar = T_local_body @ T_body_lidar。
        T_local_body_from_lidar = _validate_transform(
            T_local_gt @ T_lidar_body,
            "GT-as-LiDAR 的 body→local 候选",
        )
        lidar_rows.append(
            _pose_row(
                frame_id,
                timestamp,
                T_local_body_from_lidar,
                "gt_pose_is_lidar_then_convert_to_imu_body",
            )
        )

    os.makedirs(normalized_output, exist_ok=True)
    extrinsic_file = "candidate_lidar_to_imu_body.diagnostic.txt"
    imu_pose_file = "candidate_body_to_local_gt_as_imu.diagnostic.csv"
    lidar_pose_file = "candidate_body_to_local_gt_as_lidar.diagnostic.csv"
    extrinsic_path = os.path.join(normalized_output, extrinsic_file)
    imu_pose_path = os.path.join(normalized_output, imu_pose_file)
    lidar_pose_path = os.path.join(normalized_output, lidar_pose_file)
    _write_text_atomic(extrinsic_path, _format_extrinsic_candidate(T_body_lidar))
    _write_pose_csv_atomic(imu_pose_path, imu_rows)
    _write_pose_csv_atomic(lidar_pose_path, lidar_rows)
    radar_lidar_snapshot: Optional[Dict[str, object]] = None
    if radar_lidar_sync_normalized is not None:
        snapshot_file = "radar_lidar_sync.snapshot.csv"
        snapshot_path = os.path.join(normalized_output, snapshot_file)
        with open(
            radar_lidar_sync_normalized,
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            _write_text_atomic(snapshot_path, handle.read())
        source_sha256 = _sha256_file(radar_lidar_sync_normalized)
        snapshot_sha256 = _sha256_file(snapshot_path)
        if snapshot_sha256 != source_sha256:
            raise RuntimeError("Radar--LiDAR sync snapshot 内容漂移")
        radar_lidar_snapshot = {
            "source_path": radar_lidar_sync_normalized,
            "source_sha256": source_sha256,
            "file": snapshot_file,
            "sha256": snapshot_sha256,
        }

    report: Dict[str, object] = {
        "protocol": protocol,
        "formal": False,
        "candidate_only": True,
        "assumptions_resolved": False,
        "assumptions": {
            "body_frame_candidate": "imu",
            "radar_to_imu_matrix_direction": "T_imu_radar_unverified",
            "radar_to_lidar_direction": "T_lidar_radar_declared_by_file",
            "ground_truth_pose_frame": "unresolved_imu_or_lidar",
        },
        "formal_blockers": [
            "radar_to_imu_direction_not_authoritatively_verified",
            "ground_truth_pose_frame_not_authoritatively_verified",
            "radar_frames_without_ground_truth_must_not_be_extrapolated",
        ],
        "inputs": {
            "radar_to_imu_matrix": {
                "path": radar_to_imu_path,
                "sha256": _sha256_file(radar_to_imu_path),
            },
            "radar_to_lidar": {
                "path": radar_to_lidar_normalized,
                "sha256": _sha256_file(radar_to_lidar_normalized),
            },
            "ground_truth": {
                "path": ground_truth_normalized,
                "sha256": _sha256_file(ground_truth_normalized),
            },
            "radar_sync_csv": {
                "path": radar_sync_normalized,
                "sha256": _sha256_file(radar_sync_normalized),
            },
        },
        "candidate_lidar_to_body": {
            "body_frame_candidate": "imu",
            "direction": "lidar_to_imu_body",
            "file": extrinsic_file,
            "sha256": _sha256_file(extrinsic_path),
            "matrix_4x4": T_body_lidar.tolist(),
        },
        "pose_candidates": {
            "gt_as_imu": {
                "file": imu_pose_file,
                "sha256": _sha256_file(imu_pose_path),
                "frame_count": len(imu_rows),
                "direction": "imu_body_to_local",
                "gt_pose_hypothesis": "imu",
            },
            "gt_as_lidar": {
                "file": lidar_pose_file,
                "sha256": _sha256_file(lidar_pose_path),
                "frame_count": len(lidar_rows),
                "direction": "imu_body_to_local",
                "gt_pose_hypothesis": "lidar",
            },
        },
        "coverage": {
            "radar_frame_count": len(radar_records),
            "covered_frame_count": len(imu_rows),
            "uncovered_frame_count": len(uncovered_records),
            "uncovered_frame_ids": [
                str(record["frame_id"]) for record in uncovered_records
            ],
            "uncovered_records": uncovered_records,
            "no_extrapolation": True,
            "max_interpolation_gap_s": float(max_interpolation_gap_s),
            "interpolation_gap_s": _finite_summary(interpolation_gaps),
        },
        "timing": {
            "pose_reference_sensor": reference_sensor,
            "ground_truth_record_count": int(len(gt_times)),
            "ground_truth_start": float(gt_times[0]),
            "ground_truth_end": float(gt_times[-1]),
            "radar_start": float(radar_records[0]["timestamp"]),
            "radar_end": float(radar_records[-1]["timestamp"]),
        },
    }
    if radar_lidar_snapshot is not None:
        report["inputs"]["radar_lidar_sync_snapshot"] = radar_lidar_snapshot

    readme = (
        "# Mapping pose diagnostic candidates\n\n"
        "本目录只包含诊断候选，`formal=false`，不得直接作为飞行/避障真值。\n\n"
        "- `audit.json`：输入哈希、假设、覆盖范围与 formal blockers。\n"
        "- `candidate_lidar_to_imu_body.diagnostic.txt`：在 body=IMU 假设下的 LiDAR→body 候选。\n"
        "- 两份 CSV：分别假设 GT pose 属于 IMU 或 LiDAR，均只含可插值帧。\n"
        "- v2 额外封存 Radar--LiDAR sync，并按 LiDAR reference timestamp 插值。\n\n"
        "确认 Radar→IMU 方向、GT pose frame 并补齐/剥离 uncovered 帧后，"
        "必须通过新的 formal receipt 另行发布。\n"
    )
    _write_text_atomic(os.path.join(normalized_output, "README.md"), readme)
    _write_text_atomic(
        os.path.join(normalized_output, "command.txt"),
        (command_line or "Python API invocation; see audit.json inputs") + "\n",
    )
    _write_json_atomic(os.path.join(normalized_output, "audit.json"), report)
    return report


def build_parser() -> argparse.ArgumentParser:
    """构造只允许诊断候选的 CLI。"""
    parser = argparse.ArgumentParser(
        description="组合 LiDAR→IMU-body 候选并生成双 GT-frame 位姿诊断"
    )
    parser.add_argument("--radar_to_imu_matrix", required=True)
    parser.add_argument("--radar_to_lidar_calib", required=True)
    parser.add_argument("--ground_truth", required=True)
    parser.add_argument("--radar_sync_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--radar_lidar_sync_csv")
    parser.add_argument(
        "--pose_reference_sensor",
        choices=("radar", "lidar"),
        default="radar",
    )
    parser.add_argument("--max_interpolation_gap_s", type=float, default=0.2)
    parser.add_argument("--frame_width", type=int, default=6)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = build_mapping_pose_candidates(
        radar_to_imu_matrix_path=args.radar_to_imu_matrix,
        radar_to_lidar_path=args.radar_to_lidar_calib,
        ground_truth_path=args.ground_truth,
        radar_sync_csv_path=args.radar_sync_csv,
        output_dir=args.output_dir,
        radar_lidar_sync_csv_path=args.radar_lidar_sync_csv,
        pose_reference_sensor=args.pose_reference_sensor,
        max_interpolation_gap_s=args.max_interpolation_gap_s,
        frame_width=args.frame_width,
        command_line=shlex.join(sys.argv),
    )
    print(
        json.dumps(
            {
                "audit_path": os.path.join(os.path.abspath(args.output_dir), "audit.json"),
                "formal": report["formal"],
                "covered_frame_count": report["coverage"]["covered_frame_count"],
                "uncovered_frame_count": report["coverage"]["uncovered_frame_count"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
