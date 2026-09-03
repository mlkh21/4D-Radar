# -*- coding: utf-8 -*-
"""文件功能：构建、持久化并严格加载 LiDAR 射线可观测体素掩码。"""

from typing import Optional, Sequence, Tuple

import numpy as np


OBSERVED_MASK_PROTOCOL = "lidar_ray_observed_target_domain_v2"
PERSISTED_OBSERVED_MASK_SOURCE = "persisted_lidar_ray_target_domain_v2"
OBSERVED_MASK_SPATIAL_DOMAIN = "target_z_min_x_max_voxel_centers_v1"


def _validate_pc_range(pc_range: Sequence[float]) -> Tuple[float, ...]:
    bounds = tuple(float(value) for value in pc_range)
    if len(bounds) != 6 or not np.all(np.isfinite(bounds)):
        raise ValueError(f"pc_range 必须包含 6 个有限数，当前为 {pc_range}")
    if any(bounds[axis] >= bounds[axis + 3] for axis in range(3)):
        raise ValueError(f"pc_range 上下界无效: {bounds}")
    return bounds


def build_spatial_valid_domain(
    voxel_shape: Sequence[int],
    pc_range: Sequence[float],
    *,
    z_min: Optional[float] = None,
    x_max: Optional[float] = None,
) -> np.ndarray:
    """按体素中心构建 target/observed 共用的 ``(X,Y,Z)`` 任务域。"""
    shape = tuple(int(value) for value in voxel_shape)
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError(f"voxel_shape 必须是三个正整数，当前为 {voxel_shape}")
    bounds = _validate_pc_range(pc_range)
    for name, value in (("z_min", z_min), ("x_max", x_max)):
        if value is not None and not np.isfinite(float(value)):
            raise ValueError(f"{name} 必须是有限数或 None")

    x_size = (bounds[3] - bounds[0]) / float(shape[0])
    z_size = (bounds[5] - bounds[2]) / float(shape[2])
    x_centers = bounds[0] + (np.arange(shape[0], dtype=np.float64) + 0.5) * x_size
    z_centers = bounds[2] + (np.arange(shape[2], dtype=np.float64) + 0.5) * z_size
    valid_x = np.ones(shape[0], dtype=bool)
    valid_z = np.ones(shape[2], dtype=bool)
    if x_max is not None:
        valid_x &= x_centers <= float(x_max)
    if z_min is not None:
        valid_z &= z_centers >= float(z_min)
    return np.broadcast_to(
        valid_x[:, np.newaxis, np.newaxis]
        & valid_z[np.newaxis, np.newaxis, :],
        shape,
    ).copy()


def _validate_spatial_valid_domain(
    valid_domain: np.ndarray,
    expected_shape: Sequence[int],
) -> np.ndarray:
    """严格验证调用方提供的共享任务域，避免静默广播或非二值输入。"""
    domain = np.asarray(valid_domain)
    shape = tuple(int(value) for value in expected_shape)
    if domain.shape != shape:
        raise ValueError(
            f"valid_domain shape 不匹配: {domain.shape} != {shape}"
        )
    if domain.dtype != np.bool_:
        if not np.all(np.isin(domain, (0, 1))):
            raise ValueError("valid_domain 只能包含 bool 或 0/1")
        domain = domain.astype(bool)
    return domain


def build_lidar_observed_mask(
    lidar_voxel: np.ndarray,
    pc_range: Sequence[float],
    ray_step_fraction: float = 0.5,
    *,
    valid_domain: Optional[np.ndarray] = None,
) -> np.ndarray:
    """从 LiDAR occupied 端点向传感器原点投射 observed mask。

    返回轴序为 ``(X,Y,Z)`` 的 bool 数组。所有 occupied 端点均保留为
    observed；同一离散方向只对最近端点投射 free-space 射线，避免把遮挡后
    空间误当作已观测 free。没有 occupied 端点时返回全 False。
    """
    voxel = np.asarray(lidar_voxel, dtype=np.float32)
    if voxel.ndim != 4 or voxel.shape[-1] < 1:
        raise ValueError(f"lidar_voxel 必须是 (X,Y,Z,C)，当前为 {voxel.shape}")
    if not np.all(np.isfinite(voxel)):
        raise ValueError("lidar_voxel 必须全部为有限数")
    bounds = _validate_pc_range(pc_range)
    ray_step_fraction = float(ray_step_fraction)
    if not np.isfinite(ray_step_fraction) or ray_step_fraction <= 0.0:
        raise ValueError("ray_step_fraction 必须是正有限数")
    domain = (
        np.ones(voxel.shape[:3], dtype=bool)
        if valid_domain is None
        else _validate_spatial_valid_domain(valid_domain, voxel.shape[:3])
    )

    observed = np.zeros(voxel.shape[:3], dtype=bool)
    occupied_coords = np.argwhere(voxel[..., 0] > 0.5)
    if occupied_coords.size == 0:
        return observed

    observed[tuple(occupied_coords.T)] = True
    voxel_size = np.asarray(
        [
            (bounds[axis + 3] - bounds[axis]) / voxel.shape[axis]
            for axis in range(3)
        ],
        dtype=np.float32,
    )
    bounds_min = np.asarray(bounds[:3], dtype=np.float32)
    origin = np.zeros(3, dtype=np.float32)
    centers = bounds_min + (occupied_coords.astype(np.float32) + 0.5) * voxel_size
    origin_index = np.floor((origin - bounds_min) / voxel_size).astype(np.int64)
    directions = occupied_coords.astype(np.int64) - origin_index
    gcd = np.gcd.reduce(np.abs(directions), axis=1)
    gcd[gcd == 0] = 1
    normalized_directions = directions // gcd[:, np.newaxis]
    distances = np.linalg.norm(centers - origin[np.newaxis, :], axis=1)
    order = np.argsort(distances, kind="stable")
    _, first_indices = np.unique(
        normalized_directions[order], axis=0, return_index=True
    )
    selected = order[first_indices]
    min_step = max(float(np.min(voxel_size)) * ray_step_fraction, 1e-6)

    for endpoint in centers[selected]:
        vector = endpoint - origin
        distance = float(np.linalg.norm(vector))
        step_count = max(1, int(np.ceil(distance / min_step)))
        samples = origin + vector * (
            np.arange(1, step_count + 1, dtype=np.float32)[:, np.newaxis]
            / float(step_count)
        )
        indices = np.floor((samples - bounds_min) / voxel_size).astype(np.int64)
        valid = np.all(
            (indices >= 0)
            & (indices < np.asarray(voxel.shape[:3], dtype=np.int64)),
            axis=1,
        )
        indices = indices[valid]
        if indices.size:
            observed[indices[:, 0], indices[:, 1], indices[:, 2]] = True
    # NOTE: 域外 LiDAR 端点仍可贡献穿过任务域的 free 射线，但域外体素本身
    # 必须保持 unknown，不能因 target 的空间裁剪被解释为明确 free。
    observed &= domain
    return observed


def save_observed_mask(
    path: str,
    observed_mask: np.ndarray,
    pc_range: Sequence[float],
) -> None:
    """以稀疏、无 pickle 的 NPZ 格式保存 observed mask 及几何身份。"""
    mask = np.asarray(observed_mask)
    if mask.ndim != 3:
        raise ValueError(f"observed_mask 必须是三维数组，当前为 {mask.shape}")
    if mask.dtype != np.bool_:
        if not np.all(np.isin(mask, (0, 1))):
            raise ValueError("observed_mask 只能包含 0/1")
        mask = mask.astype(bool)
    if any(int(size) <= 0 for size in mask.shape):
        raise ValueError(f"observed_mask shape 无效: {mask.shape}")
    bounds = _validate_pc_range(pc_range)
    coords = np.argwhere(mask).astype(np.int32, copy=False)
    np.savez(
        path,
        protocol=np.asarray(OBSERVED_MASK_PROTOCOL),
        coords=coords,
        shape=np.asarray(mask.shape, dtype=np.int32),
        pc_range=np.asarray(bounds, dtype=np.float64),
    )


def load_observed_mask(
    path: str,
    *,
    expected_shape: Optional[Sequence[int]] = None,
    expected_pc_range: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """严格加载 persisted observed mask，并拒绝协议或几何不匹配。"""
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {"protocol", "coords", "shape", "pc_range"}
            if set(data.files) != required:
                raise ValueError(
                    f"observed mask 字段必须精确为 {sorted(required)}，"
                    f"当前为 {sorted(data.files)}"
                )
            protocol = str(np.asarray(data["protocol"]).item())
            coords = np.asarray(data["coords"])
            shape_values = np.asarray(data["shape"])
            stored_pc_range = np.asarray(data["pc_range"], dtype=np.float64)
    except (OSError, KeyError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("observed mask"):
            raise
        raise ValueError(f"无法加载 observed mask {path}: {exc}") from exc

    if protocol != OBSERVED_MASK_PROTOCOL:
        raise ValueError(
            f"observed mask protocol 不匹配: {protocol!r} != "
            f"{OBSERVED_MASK_PROTOCOL!r}"
        )
    if shape_values.shape != (3,) or not np.issubdtype(shape_values.dtype, np.integer):
        raise ValueError("observed mask shape 必须是三个整数")
    shape = tuple(int(value) for value in shape_values)
    if any(value <= 0 for value in shape):
        raise ValueError(f"observed mask shape 无效: {shape}")
    if expected_shape is not None and shape != tuple(int(value) for value in expected_shape):
        raise ValueError(
            f"observed mask shape 不匹配: stored={shape}, expected={tuple(expected_shape)}"
        )
    if stored_pc_range.shape != (6,) or not np.all(np.isfinite(stored_pc_range)):
        raise ValueError("observed mask pc_range 必须是六个有限数")
    stored_bounds = _validate_pc_range(stored_pc_range.tolist())
    if expected_pc_range is not None:
        expected_bounds = _validate_pc_range(expected_pc_range)
        if not np.allclose(stored_bounds, expected_bounds, rtol=0.0, atol=1e-9):
            raise ValueError(
                "observed mask pc_range 不匹配: "
                f"stored={stored_bounds}, expected={expected_bounds}"
            )
    if coords.ndim != 2 or coords.shape[1:] != (3,) or not np.issubdtype(coords.dtype, np.integer):
        raise ValueError("observed mask coords 必须是 (N,3) 整数数组")
    coords = coords.astype(np.int64, copy=False)
    if coords.size:
        limits = np.asarray(shape, dtype=np.int64)
        if np.any(coords < 0) or np.any(coords >= limits):
            raise ValueError("observed mask coords 越界")
        if np.unique(coords, axis=0).shape[0] != coords.shape[0]:
            raise ValueError("observed mask coords 含重复体素")
    mask = np.zeros(shape, dtype=bool)
    if coords.size:
        mask[tuple(coords.T)] = True
    return mask
