#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""预处理运动补偿协议：显式区分无补偿、固定速度和逐帧记录速度。"""

from __future__ import annotations

import hashlib
import os
from typing import Optional, Sequence

import numpy as np


VELOCITY_MODES = ("none", "fixed", "recorded")
VELOCITY_FRAMES = ("radar", "lidar")


def sha256_file(path: str) -> str:
    """计算速度源文件内容 hash，供 preprocess_policy 固化实际输入。"""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_vector(value: Sequence[float], name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} 必须是三个有限数")
    return vector


def load_recorded_velocity_table(path: str) -> np.ndarray:
    """读取 `timestamp,vx,vy,vz` 表，并严格要求时间戳递增。"""
    if not path:
        raise ValueError("recorded 模式必须提供 velocity_file")
    if os.path.islink(path) or not os.path.isfile(path):
        raise ValueError(f"velocity_file 必须是普通文件: {path}")

    rows = []
    header_checked = False
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.replace(",", " ").split()
            if not header_checked:
                header_checked = True
                normalized = tuple(token.strip().lower() for token in tokens)
                if normalized == ("timestamp", "vx", "vy", "vz"):
                    continue
            if len(tokens) != 4:
                raise ValueError(
                    f"velocity_file 第 {line_number} 行必须包含 timestamp,vx,vy,vz 四列"
                )
            try:
                row = [float(token) for token in tokens]
            except ValueError as exc:
                raise ValueError(
                    f"velocity_file 第 {line_number} 行包含非数值字段"
                ) from exc
            if not np.all(np.isfinite(row)):
                raise ValueError(f"velocity_file 第 {line_number} 行必须全部为有限数")
            rows.append(row)

    if not rows:
        raise ValueError(f"velocity_file 不包含速度记录: {path}")
    table = np.asarray(rows, dtype=np.float64)
    if np.any(np.diff(table[:, 0]) <= 0.0):
        raise ValueError("velocity_file 的 timestamp 必须严格递增且不可重复")
    return table


def resolve_frame_velocity(
    mode: str,
    fixed_velocity: Optional[Sequence[float]],
    frame_timestamp: float,
    recorded_table: Optional[np.ndarray],
    max_delta: float,
) -> Optional[np.ndarray]:
    """为单帧解析速度；recorded 模式只允许在时间容差内最近邻匹配。"""
    mode = str(mode).strip().lower()
    if mode not in VELOCITY_MODES:
        raise ValueError(f"velocity_mode 必须是 {VELOCITY_MODES} 之一，当前为 {mode!r}")
    timestamp = float(frame_timestamp)
    if not np.isfinite(timestamp):
        raise ValueError("frame_timestamp 必须是有限数")
    max_delta = float(max_delta)
    if not np.isfinite(max_delta) or max_delta < 0.0:
        raise ValueError("velocity_max_delta 必须是有限非负数")

    if mode == "none":
        return None
    if mode == "fixed":
        return _finite_vector(fixed_velocity, "fixed_velocity")
    if recorded_table is None:
        raise ValueError("recorded 模式必须先加载 velocity_file")

    table = np.asarray(recorded_table, dtype=np.float64)
    if table.ndim != 2 or table.shape[1] != 4 or table.shape[0] == 0:
        raise ValueError("recorded_table 必须是形状为 (N,4) 的非空数组")
    if not np.all(np.isfinite(table)) or np.any(np.diff(table[:, 0]) <= 0.0):
        raise ValueError("recorded_table 必须包含有限且严格递增的 timestamp")
    position = int(np.searchsorted(table[:, 0], timestamp, side="left"))
    candidates = [min(max(position, 0), table.shape[0] - 1)]
    if position > 0:
        candidates.append(position - 1)
    best = min(candidates, key=lambda index: abs(float(table[index, 0]) - timestamp))
    delta = abs(float(table[best, 0]) - timestamp)
    if delta > max_delta:
        raise ValueError(
            f"frame timestamp 与 recorded velocity 时间差 {delta:.6f}s 超过容差 "
            f"{max_delta:.6f}s"
        )
    return table[best, 1:4].copy()


def transform_velocity(
    velocity: Sequence[float],
    source_frame: str,
    target_frame: str,
    radar_to_lidar_rotation: np.ndarray,
) -> np.ndarray:
    """仅用旋转矩阵转换速度，禁止把平移量错误地加入速度。"""
    source_frame = str(source_frame).strip().lower()
    target_frame = str(target_frame).strip().lower()
    if source_frame not in VELOCITY_FRAMES or target_frame not in VELOCITY_FRAMES:
        raise ValueError(
            f"velocity_frame 必须是 {VELOCITY_FRAMES}，当前为 "
            f"{source_frame!r}->{target_frame!r}"
        )
    vector = _finite_vector(velocity, "velocity")
    if source_frame == target_frame:
        return vector
    rotation = np.asarray(radar_to_lidar_rotation, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("radar_to_lidar_rotation 必须是有限的 3x3 矩阵")
    if source_frame == "radar" and target_frame == "lidar":
        return rotation @ vector
    return rotation.T @ vector
