# -*- coding: utf-8 -*-
"""文件功能：定义 Radar 分字段 finite-count 稀疏统计存储合同。"""

import os
from typing import Dict, Mapping, Optional, Tuple

import numpy as np


RADAR_STATISTICS_PROTOCOL_V1 = "radar_point_count_doppler_validity_v1"
RADAR_STATISTICS_PROTOCOL = "radar_point_count_field_validity_v2"
RADAR_RESIZE_AGGREGATION_V1 = "occupied_voxel_equal_weight_total_variance"
RADAR_RESIZE_AGGREGATION = "field_valid_count_weighted_total_variance_v2"
SUPPORTED_RADAR_STATISTICS_PROTOCOLS = frozenset(
    {RADAR_STATISTICS_PROTOCOL_V1, RADAR_STATISTICS_PROTOCOL}
)
_BASE_KEYS = {"coords", "features", "shape"}
_STATISTICS_KEYS_V1 = {
    "radar_statistics_protocol",
    "point_count",
    "doppler_valid_count",
}
_STATISTICS_KEYS_V2 = _STATISTICS_KEYS_V1 | {"intensity_valid_count"}


def _validate_sparse_base(
    coords: np.ndarray,
    features: np.ndarray,
    shape: np.ndarray,
    *,
    path: str,
) -> Tuple[Tuple[int, int, int, int], np.ndarray]:
    """验证 Radar 稀疏体素基础结构，并返回规范 shape/coords。"""
    shape_array = np.asarray(shape)
    if (
        shape_array.shape != (4,)
        or not np.issubdtype(shape_array.dtype, np.integer)
    ):
        raise ValueError(f"Radar sparse shape 必须是四维整数数组: {path}")
    shape_tuple = tuple(int(value) for value in shape_array.tolist())
    if any(value <= 0 for value in shape_tuple) or shape_tuple[-1] != 4:
        raise ValueError(f"Radar sparse shape 必须为正数且通道数为 4: {path}")

    coords_array = np.asarray(coords)
    features_array = np.asarray(features)
    if (
        coords_array.ndim != 2
        or coords_array.shape[1:] != (3,)
        or not np.issubdtype(coords_array.dtype, np.integer)
    ):
        raise ValueError(f"Radar sparse coords 必须是 (N,3) 整数数组: {path}")
    if features_array.shape != (coords_array.shape[0], 4):
        raise ValueError(f"Radar sparse features 必须是 (N,4): {path}")
    if not np.all(np.isfinite(features_array)):
        raise ValueError(f"Radar sparse features 含非有限数: {path}")
    coords_array = coords_array.astype(np.int64, copy=False)
    if coords_array.shape[0] > 0:
        spatial_shape = np.asarray(shape_tuple[:3], dtype=np.int64)
        if np.any(coords_array < 0) or np.any(coords_array >= spatial_shape):
            raise ValueError(f"Radar sparse coords 超出 shape: {path}")
        flat = np.ravel_multi_index(coords_array.T, shape_tuple[:3])
        if len(np.unique(flat)) != len(flat):
            raise ValueError(f"Radar sparse coords 含重复项: {path}")
        if np.any(features_array[:, 0] <= 0.0):
            raise ValueError(f"Radar sparse occupied features 的通道 0 必须为正: {path}")
    return shape_tuple, coords_array


def _validate_statistics(
    *,
    coords: np.ndarray,
    point_count: np.ndarray,
    intensity_valid_count: Optional[np.ndarray],
    doppler_valid_count: np.ndarray,
    protocol: object,
    path: str,
) -> Dict[str, object]:
    """验证与 coords 一一对齐的无符号计数，并生成审计摘要。"""
    protocol_array = np.asarray(protocol)
    if protocol_array.shape != ():
        raise ValueError(f"Radar statistics protocol 不匹配: {path}")
    protocol_name = str(protocol_array.item())
    if protocol_name not in SUPPORTED_RADAR_STATISTICS_PROTOCOLS:
        raise ValueError(f"Radar statistics protocol 不匹配: {path}")
    point_array = np.asarray(point_count)
    intensity_array = (
        np.asarray(intensity_valid_count)
        if intensity_valid_count is not None
        else None
    )
    valid_array = np.asarray(doppler_valid_count)
    expected_shape = (coords.shape[0],)
    if point_array.shape != expected_shape or point_array.dtype != np.dtype(np.uint32):
        raise ValueError(f"Radar statistics point_count 必须是与 coords 对齐的 uint32: {path}")
    if valid_array.shape != expected_shape or valid_array.dtype != np.dtype(np.uint32):
        raise ValueError(
            f"Radar statistics doppler_valid_count 必须是与 coords 对齐的 uint32: {path}"
        )
    if protocol_name == RADAR_STATISTICS_PROTOCOL:
        if (
            intensity_array is None
            or intensity_array.shape != expected_shape
            or intensity_array.dtype != np.dtype(np.uint32)
        ):
            raise ValueError(
                "Radar statistics intensity_valid_count 必须是与 coords "
                f"对齐的 uint32: {path}"
            )
    elif intensity_array is not None:
        raise ValueError(f"Radar statistics v1 不允许 intensity_valid_count: {path}")
    if np.any(point_array == 0):
        raise ValueError(f"Radar statistics occupied point count 必须大于 0: {path}")
    if np.any(valid_array > point_array):
        raise ValueError(f"Radar statistics Doppler valid count 不得超过 point count: {path}")
    if intensity_array is not None and np.any(intensity_array > point_array):
        raise ValueError(
            f"Radar statistics intensity valid count 不得超过 point count: {path}"
        )
    summary = {
        "protocol": protocol_name,
        "occupied_voxels": int(coords.shape[0]),
        "total_point_count": int(point_array.astype(np.uint64).sum()),
        "total_doppler_valid_count": int(valid_array.astype(np.uint64).sum()),
        "multi_point_voxels": int(np.count_nonzero(point_array >= 2)),
        "doppler_multi_sample_voxels": int(np.count_nonzero(valid_array >= 2)),
        "doppler_missing_voxels": int(np.count_nonzero(valid_array == 0)),
    }
    if intensity_array is not None:
        summary.update(
            {
                "total_intensity_valid_count": int(
                    intensity_array.astype(np.uint64).sum()
                ),
                "intensity_multi_sample_voxels": int(
                    np.count_nonzero(intensity_array >= 2)
                ),
                "intensity_missing_voxels": int(
                    np.count_nonzero(intensity_array == 0)
                ),
            }
        )
    return summary


def save_sparse_radar_voxel(
    path: str,
    voxel_grid: np.ndarray,
    statistics: Mapping[str, object],
) -> None:
    """把四通道体素与坐标对齐的统计写入同一个压缩 NPZ。"""
    output_path = os.path.abspath(os.fspath(path))
    if not output_path.endswith(".npz"):
        raise ValueError("Radar statistics 只支持 .npz 稀疏存储")
    voxel = np.asarray(voxel_grid)
    if voxel.ndim != 4 or voxel.shape[-1] != 4 or not np.all(np.isfinite(voxel)):
        raise ValueError("Radar voxel 必须是有限的 (X,Y,Z,4) 数组")
    occupied = voxel[..., 0] > 0.0
    coords = np.column_stack(np.where(occupied)).astype(np.int32, copy=False)
    features = voxel[occupied].astype(np.float32, copy=False)
    protocol = statistics.get("protocol") if isinstance(statistics, Mapping) else None
    expected_keys = (
        {"protocol", "coords", "point_count", "intensity_valid_count", "doppler_valid_count"}
        if protocol == RADAR_STATISTICS_PROTOCOL
        else {"protocol", "coords", "point_count", "doppler_valid_count"}
    )
    if not isinstance(statistics, Mapping) or set(statistics) != expected_keys:
        raise ValueError(f"Radar statistics 字段必须精确为 {sorted(expected_keys)}")
    statistics_coords = np.asarray(statistics["coords"])
    if statistics_coords.shape != coords.shape or not np.array_equal(
        statistics_coords.astype(np.int64, copy=False),
        coords.astype(np.int64, copy=False),
    ):
        raise ValueError("Radar statistics coords 与 occupied voxel 不一致")
    summary = _validate_statistics(
        coords=coords,
        point_count=np.asarray(statistics["point_count"]),
        intensity_valid_count=(
            np.asarray(statistics["intensity_valid_count"])
            if "intensity_valid_count" in statistics
            else None
        ),
        doppler_valid_count=np.asarray(statistics["doppler_valid_count"]),
        protocol=statistics["protocol"],
        path=output_path,
    )
    if summary["occupied_voxels"] != int(np.count_nonzero(occupied)):
        raise ValueError("Radar statistics occupied voxel 数量不一致")
    output = {
        "coords": coords,
        "features": features,
        "shape": np.asarray(voxel.shape, dtype=np.int64),
        "radar_statistics_protocol": np.asarray(str(statistics["protocol"])),
        "point_count": np.asarray(statistics["point_count"], dtype=np.uint32),
        "doppler_valid_count": np.asarray(
            statistics["doppler_valid_count"], dtype=np.uint32
        ),
    }
    if "intensity_valid_count" in statistics:
        output["intensity_valid_count"] = np.asarray(
            statistics["intensity_valid_count"], dtype=np.uint32
        )
    np.savez_compressed(
        output_path,
        **output,
    )


def _read_sparse_radar_payload(
    path: str,
    *,
    require_statistics: bool = False,
) -> Tuple[
    Tuple[int, int, int, int],
    np.ndarray,
    np.ndarray,
    Optional[Dict[str, object]],
    Optional[Dict[str, object]],
]:
    """读取并验证 NPZ，不分配全尺寸稠密体素。"""
    input_path = os.path.abspath(os.fspath(path))
    if not input_path.endswith(".npz") or not os.path.isfile(input_path):
        raise ValueError(f"Radar sparse voxel 必须是普通 .npz 文件: {input_path}")
    try:
        with np.load(input_path, allow_pickle=False) as payload:
            keys = set(payload.files)
            if not _BASE_KEYS.issubset(keys):
                raise ValueError(f"Radar sparse voxel 缺少基础字段: {input_path}")
            coords = np.asarray(payload["coords"])
            features = np.asarray(payload["features"])
            shape = np.asarray(payload["shape"])
            has_protocol = "radar_statistics_protocol" in keys
            protocol_name = (
                str(np.asarray(payload["radar_statistics_protocol"]).item())
                if has_protocol
                else None
            )
            expected_statistics_keys = (
                _STATISTICS_KEYS_V2
                if protocol_name == RADAR_STATISTICS_PROTOCOL
                else _STATISTICS_KEYS_V1
            )
            has_statistics = has_protocol and expected_statistics_keys.issubset(keys)
            if require_statistics and not has_statistics:
                raise ValueError(f"Radar statistics 缺失: {input_path}")
            if has_statistics and keys != _BASE_KEYS | expected_statistics_keys:
                raise ValueError(f"Radar statistics NPZ 含未绑定字段: {input_path}")
            if not has_statistics and keys != _BASE_KEYS:
                raise ValueError(f"Radar legacy NPZ 含不完整 statistics 字段: {input_path}")
            shape_tuple, normalized_coords = _validate_sparse_base(
                coords,
                features,
                shape,
                path=input_path,
            )
            statistics_payload = None
            summary = None
            if has_statistics:
                statistics_payload = {
                    "protocol": protocol_name,
                    "point_count": np.asarray(payload["point_count"]).copy(),
                    "doppler_valid_count": np.asarray(
                        payload["doppler_valid_count"]
                    ).copy(),
                }
                if "intensity_valid_count" in payload:
                    statistics_payload["intensity_valid_count"] = np.asarray(
                        payload["intensity_valid_count"]
                    ).copy()
                summary = _validate_statistics(
                    coords=normalized_coords,
                    point_count=np.asarray(payload["point_count"]),
                    intensity_valid_count=(
                        np.asarray(payload["intensity_valid_count"])
                        if "intensity_valid_count" in payload
                        else None
                    ),
                    doppler_valid_count=np.asarray(payload["doppler_valid_count"]),
                    protocol=payload["radar_statistics_protocol"],
                    path=input_path,
                )
    except (OSError, ValueError, KeyError) as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"Radar sparse voxel 无法解析: {input_path}") from exc
    return (
        shape_tuple,
        normalized_coords,
        features,
        statistics_payload,
        summary,
    )


def validate_sparse_radar_statistics(path: str) -> Dict[str, object]:
    """只验证统计合同并返回摘要，供 Dataset 全帧预检使用。"""
    _shape, _coords, _features, _statistics, summary = _read_sparse_radar_payload(
        path,
        require_statistics=True,
    )
    if summary is None:  # pragma: no cover - require_statistics 已保证不可达
        raise ValueError(f"Radar statistics 缺失: {path}")
    return summary


def load_sparse_radar_voxel(
    path: str,
    *,
    require_statistics: bool = False,
) -> Tuple[np.ndarray, Optional[Dict[str, object]]]:
    """恢复 Radar 体素；存在或要求统计时执行完整合同校验。"""
    shape_tuple, normalized_coords, features, _statistics, summary = (
        _read_sparse_radar_payload(
            path,
            require_statistics=require_statistics,
        )
    )

    voxel_grid = np.zeros(shape_tuple, dtype=np.float32)
    if normalized_coords.shape[0] > 0:
        voxel_grid[
            normalized_coords[:, 0],
            normalized_coords[:, 1],
            normalized_coords[:, 2],
        ] = features.astype(np.float32, copy=False)
    return voxel_grid, summary


def load_sparse_radar_voxel_with_statistics(
    path: str,
) -> Tuple[np.ndarray, Dict[str, object], Dict[str, object]]:
    """恢复四通道体素及逐体素字段计数，供物理一致的 resize 使用。"""
    (
        shape_tuple,
        normalized_coords,
        features,
        sparse_statistics,
        summary,
    ) = _read_sparse_radar_payload(path, require_statistics=True)
    if sparse_statistics is None or summary is None:  # pragma: no cover
        raise ValueError(f"Radar statistics 缺失: {path}")

    voxel_grid = np.zeros(shape_tuple, dtype=np.float32)
    if normalized_coords.shape[0] > 0:
        voxel_grid[
            normalized_coords[:, 0],
            normalized_coords[:, 1],
            normalized_coords[:, 2],
        ] = features.astype(np.float32, copy=False)

    fields: Dict[str, object] = {"protocol": sparse_statistics["protocol"]}
    for name in (
        "point_count",
        "intensity_valid_count",
        "doppler_valid_count",
    ):
        if name not in sparse_statistics:
            continue
        dense = np.zeros(shape_tuple[:3], dtype=np.uint32)
        if normalized_coords.shape[0] > 0:
            dense[
                normalized_coords[:, 0],
                normalized_coords[:, 1],
                normalized_coords[:, 2],
            ] = np.asarray(sparse_statistics[name], dtype=np.uint32)
        fields[name] = dense
    return voxel_grid, fields, summary
