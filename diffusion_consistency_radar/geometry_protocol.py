# -*- coding: utf-8 -*-
"""文件功能：提供部署推理与地图入口共用的轻量刚体外参协议。"""

import os
from typing import Dict

import numpy as np


def load_extrinsic_transform(path: str) -> np.ndarray:
    """严格加载 ``R:/T:`` 外参文件并返回 source→target 的 4×4 变换。"""
    normalized = os.path.abspath(os.fspath(path))
    if os.path.islink(normalized) or not os.path.isfile(normalized):
        raise ValueError(f"外参必须是普通文件: {normalized}")
    values: Dict[str, list] = {}
    with open(normalized, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            compact_metadata = line.lower().replace(" ", "")
            if line.startswith("#") and (
                "formal=false" in compact_metadata
                or "formal:false" in compact_metadata
            ):
                raise ValueError(
                    f"正式入口拒绝 formal=false 的诊断候选外参: {normalized}"
                )
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, raw = line.split(":", 1)
            try:
                values[key.strip()] = [float(value) for value in raw.split()]
            except ValueError as exc:
                raise ValueError(f"外参含非数值字段: {normalized}") from exc
    r_values = values.get("R")
    t_values = values.get("T")
    if r_values is None or len(r_values) != 9 or t_values is None or len(t_values) != 3:
        raise ValueError(f"外参文件必须精确提供 3x3 R 和 3 元 T: {normalized}")
    rotation = np.asarray(r_values, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(t_values, dtype=np.float64)
    if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)):
        raise ValueError(f"外参 R/T 含非有限数: {normalized}")
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=5e-3, rtol=0.0):
        raise ValueError(f"外参 R 不是正交旋转矩阵: {normalized}")
    determinant = float(np.linalg.det(rotation))
    if abs(determinant - 1.0) > 5e-3:
        raise ValueError(
            f"外参 R determinant 必须接近 1，实际为 {determinant}: {normalized}"
        )
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation.astype(np.float32)
    transform[:3, 3] = translation.astype(np.float32)
    return transform
