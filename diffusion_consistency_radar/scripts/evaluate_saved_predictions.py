#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""正式离线评价已保存预测；指标在权威 observed 域内计算。"""

import argparse
import csv
import json
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from diffusion_consistency_radar.cm.dataset_loader import (  # noqa: E402
    crop_voxel_channels_to_pc_range,
    load_sparse_voxel,
    resize_voxel_channels,
)
from diffusion_consistency_radar.cm.evaluation_metrics import (  # noqa: E402
    bev_iou,
    filter_points_by_band,
    nearest_neighbor_metrics,
    occupancy_prf,
    uncertainty_calibration_metrics,
)
from diffusion_consistency_radar.dataset_manifest import sha256_file  # noqa: E402
from diffusion_consistency_radar.observed_artifact_protocol import (  # noqa: E402
    RADAR_ENDPOINT_RAY_OBSERVED_PROTOCOL,
    observed_mask_records_digest as _observed_mask_records_digest,
)

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None


PREDICTION_PATTERN = re.compile(r"^(\d+)_voxel\.npy$")
UNCERTAINTY_PATTERN = re.compile(r"^(\d+)_uncertainty\.npy$")
POINTCLOUD_PATTERN = re.compile(r"^(\d+)_pcl\.npy$")
OBSERVED_MASK_PATTERN = re.compile(r"^(\d+)_observed_mask\.npy$")
SOURCE_PATTERN = re.compile(r"^(\d+)\.(npy|npz)$")
ALLOWED_AUXILIARY_FILES = {
    "inference_run.json",
    "inference_runtime.csv",
    "inference_metrics.csv",
    "inference_runtime.log",
}

FORMAL_SAVED_EVALUATION_PROTOCOL = (
    "formal_saved_prediction_observed_domain_evaluation_v1"
)
DIAGNOSTIC_SAVED_EVALUATION_PROTOCOL = "saved_prediction_diagnostic_evaluation_v1"
FORMAL_METRIC_FIELDS = (
    "pred_point_count",
    "target_point_count",
    "pred_target_chamfer",
    "pred_target_count_ratio",
    "pred_target_dx",
    "pred_target_dy",
    "pred_target_dz",
    "near_precision",
    "near_recall",
    "near_bev_iou",
    "near_nn_mean",
    "near_match_ratio_2",
    "uncertainty_ece",
    "uncertainty_brier",
    "uncertainty_nll",
    "uncertainty_error_corr",
)

FRAME_FIELDS = [
    "frame_id",
    "pred_file",
    "radar_file",
    "target_file",
    "lidar_file",
    "observed_mask_file",
    "observed_voxels",
    "effective_occ_threshold",
    "target_threshold",
    "pred_point_count",
    "radar_point_count",
    "target_point_count",
    "lidar_point_count",
    "pred_target_chamfer",
    "radar_target_chamfer",
    "raw_lidar_chamfer",
    "pred_target_count_ratio",
    "pred_target_dx",
    "pred_target_dy",
    "pred_target_dz",
    "radar_target_dx",
    "radar_target_dy",
    "radar_target_dz",
    "near_precision",
    "near_recall",
    "near_bev_iou",
    "near_nn_mean",
    "near_match_ratio_2",
    "uncertainty_ece",
    "uncertainty_brier",
    "uncertainty_nll",
    "uncertainty_error_corr",
]


def _require_directory(path: str, label: str) -> None:
    if not path or not os.path.isdir(path):
        raise ValueError(f"{label} directory does not exist: {path}")


def _validate_output_directory(path: str) -> None:
    if not path:
        raise ValueError("output_dir must not be empty")
    if os.path.exists(path):
        if not os.path.isdir(path):
            raise ValueError(f"output_dir is not a directory: {path}")
        if os.listdir(path):
            raise ValueError(f"评价输出目录必须不存在或为空（non-empty）: {path}")


def _require_regular_file(path: str, label: str) -> None:
    if os.path.islink(path) or not os.path.isfile(path):
        raise ValueError(f"{label} 必须是普通文件: {path}")


def _load_json(path: str) -> dict:
    _require_regular_file(path, "run metadata")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 run metadata: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"run metadata 必须是 JSON object: {path}")
    return payload


def _resolve_sequence(
    name: str,
    override: Optional[Sequence[float]],
    metadata: dict,
    expected_length: int,
    integer: bool = False,
) -> Tuple[Tuple, str]:
    value = override if override is not None else metadata.get(name)
    source = "cli_override" if override is not None else "run_metadata"
    if value is None:
        raise ValueError(f"metadata 缺少 {name}，且未提供显式 override")
    if len(value) != expected_length:
        raise ValueError(f"{name} 必须含 {expected_length} 个值: {value}")
    cast = int if integer else float
    resolved = tuple(cast(item) for item in value)
    if not all(np.isfinite(float(item)) for item in resolved):
        raise ValueError(f"{name} 含非有限值: {value}")
    if integer and any(item <= 0 for item in resolved):
        raise ValueError(f"{name} 必须为正整数: {value}")
    return resolved, source


def _resolve_parameters(
    pred_voxel_dir: str,
    run_metadata_path: str,
    occ_threshold: Optional[float],
    source_pc_range: Optional[Sequence[float]],
    model_pc_range: Optional[Sequence[float]],
    target_size: Optional[Sequence[int]],
) -> dict:
    metadata_path = run_metadata_path or os.path.join(
        pred_voxel_dir,
        "inference_run.json",
    )
    metadata = _load_json(metadata_path) if os.path.isfile(metadata_path) else {}
    if not metadata and (
        occ_threshold is None
        or source_pc_range is None
        or model_pc_range is None
        or target_size is None
    ):
        raise ValueError(
            "缺少 inference_run.json metadata；诊断模式必须显式提供阈值和全部网格 override"
        )

    resolved_target_size, target_source = _resolve_sequence(
        "target_size", target_size, metadata, 3, integer=True
    )
    resolved_source_range, source_range_source = _resolve_sequence(
        "source_pc_range", source_pc_range, metadata, 6
    )
    resolved_model_range, model_range_source = _resolve_sequence(
        "model_pc_range", model_pc_range, metadata, 6
    )
    metadata_threshold = metadata.get("occ_threshold")
    if occ_threshold is None and metadata_threshold is None:
        raise ValueError("metadata 缺少 occ_threshold，且未提供显式 override")
    formal_run = metadata.get("formal_protocol") is True
    if formal_run and occ_threshold is not None:
        raise ValueError("正式评价禁止 threshold CLI override")
    if formal_run:
        threshold_artifact = metadata.get("occupancy_threshold_artifact")
        threshold_artifact_sha256 = metadata.get(
            "occupancy_threshold_artifact_sha256"
        )
        if (
            not isinstance(threshold_artifact, dict)
            or threshold_artifact.get("protocol")
            != "occupancy_threshold_validation_artifact_v1"
            or metadata.get("occ_threshold_source") != "validation_artifact"
            or not isinstance(threshold_artifact_sha256, str)
            or len(threshold_artifact_sha256) != 64
        ):
            raise ValueError("正式评价缺少完整 validation threshold artifact 合同")
        if float(threshold_artifact.get("selected_threshold")) != float(
            metadata_threshold
        ):
            raise ValueError("正式评价阈值与 validation artifact 不一致")
    resolved_threshold = (
        float(occ_threshold)
        if occ_threshold is not None
        else float(metadata_threshold)
    )
    if not np.isfinite(resolved_threshold):
        raise ValueError(f"occ_threshold 含非有限值: {resolved_threshold}")
    threshold_source = (
        "validation_artifact"
        if formal_run
        else ("cli_override" if occ_threshold is not None else "run_metadata")
    )

    for axis in range(3):
        if resolved_source_range[axis] >= resolved_source_range[axis + 3]:
            raise ValueError(f"source_pc_range 非法: {resolved_source_range}")
        if resolved_model_range[axis] >= resolved_model_range[axis + 3]:
            raise ValueError(f"model_pc_range 非法: {resolved_model_range}")
        if (
            resolved_model_range[axis] < resolved_source_range[axis]
            or resolved_model_range[axis + 3]
            > resolved_source_range[axis + 3]
        ):
            raise ValueError("model_pc_range 必须位于 source_pc_range 内")

    voxel_size = metadata.get("voxel_size")
    if voxel_size is not None:
        if len(voxel_size) != 3 or not all(
            np.isfinite(float(value)) and float(value) > 0.0
            for value in voxel_size
        ):
            raise ValueError(f"voxel_size 非法: {voxel_size}")
        resolved_voxel_size = tuple(float(value) for value in voxel_size)
        voxel_source = "run_metadata"
    else:
        z_size, x_size, y_size = resolved_target_size
        resolved_voxel_size = (
            (resolved_model_range[3] - resolved_model_range[0]) / x_size,
            (resolved_model_range[4] - resolved_model_range[1]) / y_size,
            (resolved_model_range[5] - resolved_model_range[2]) / z_size,
        )
        voxel_source = "derived_from_resolved_grid"

    return {
        "metadata": metadata,
        "metadata_path": metadata_path if metadata else "",
        "formal_run": formal_run,
        "target_size": resolved_target_size,
        "source_pc_range": resolved_source_range,
        "model_pc_range": resolved_model_range,
        "voxel_size": resolved_voxel_size,
        "occ_threshold": resolved_threshold,
        "parameter_sources": {
            "target_size": target_source,
            "source_pc_range": source_range_source,
            "model_pc_range": model_range_source,
            "voxel_size": voxel_source,
            "occ_threshold": threshold_source,
        },
    }


def _discover_prediction_frames(
    folder: str,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    predictions: Dict[str, str] = {}
    uncertainties: Dict[str, str] = {}
    observed_masks: Dict[str, str] = {}
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        pred_match = PREDICTION_PATTERN.fullmatch(name)
        uncertainty_match = UNCERTAINTY_PATTERN.fullmatch(name)
        observed_match = OBSERVED_MASK_PATTERN.fullmatch(name)
        if pred_match:
            frame_id = pred_match.group(1)
            _require_regular_file(path, "prediction voxel")
            predictions[frame_id] = path
        elif uncertainty_match:
            frame_id = uncertainty_match.group(1)
            _require_regular_file(path, "uncertainty")
            uncertainties[frame_id] = path
        elif observed_match:
            frame_id = observed_match.group(1)
            _require_regular_file(path, "observed mask")
            observed_masks[frame_id] = path
        elif POINTCLOUD_PATTERN.fullmatch(name) or name in ALLOWED_AUXILIARY_FILES:
            continue
        else:
            raise ValueError(f"prediction 目录含未知文件（unknown）: {path}")
    if not predictions:
        raise ValueError(f"prediction 目录没有 *_voxel.npy: {folder}")
    unknown_uncertainty = sorted(set(uncertainties) - set(predictions))
    if unknown_uncertainty:
        raise ValueError(f"uncertainty 存在未知帧: {unknown_uncertainty}")
    unknown_observed = sorted(set(observed_masks) - set(predictions))
    if unknown_observed:
        raise ValueError(f"observed mask 存在未知 frame: {unknown_observed}")
    return predictions, uncertainties, observed_masks


def _validate_observed_mask_contract(
    observed_masks: Dict[str, str],
    metadata: dict,
    predictions: Dict[str, str],
    target_size: Sequence[int],
) -> Optional[dict]:
    """校验 inference_run 声明与逐帧 Radar observed mask 内容一致。"""
    contract = metadata.get("observed_mask")
    require_formal = bool(
        metadata.get("formal_protocol") is True or metadata.get("require_real_ir")
    )
    if contract is None:
        if observed_masks:
            raise ValueError("observed mask 文件缺少 inference_run metadata 绑定")
        if require_formal:
            raise ValueError("formal inference metadata 缺少 observed mask 合同")
        return None
    if not isinstance(contract, dict):
        raise ValueError("observed mask metadata 必须是 object")
    if contract.get("protocol") != RADAR_ENDPOINT_RAY_OBSERVED_PROTOCOL:
        raise ValueError("observed mask protocol 不匹配")
    records = contract.get("records")
    if not isinstance(records, list):
        raise ValueError("observed mask metadata 缺少 records")
    if int(contract.get("frame_count", -1)) != len(records):
        raise ValueError("observed mask frame_count 与 records 不一致")

    normalized_records = []
    record_ids = set()
    total_observed = 0
    expected_shape = tuple(int(value) for value in target_size)
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("observed mask record 必须是 object")
        frame_id = str(record.get("frame_id", ""))
        file_name = str(record.get("file", ""))
        if frame_id in record_ids or file_name != f"{frame_id}_observed_mask.npy":
            raise ValueError("observed mask record frame/file 无效或重复")
        record_ids.add(frame_id)
        path = observed_masks.get(frame_id)
        if path is None or os.path.basename(path) != file_name:
            raise ValueError(f"observed mask frame 缺失: {frame_id}")
        array = _load_array(path)
        if array.ndim != 3 or tuple(array.shape) != expected_shape:
            raise ValueError(
                f"observed mask shape 必须为 {expected_shape}: {path} -> {array.shape}"
            )
        if not np.logical_or(array == 0, array == 1).all():
            raise ValueError(f"observed mask 必须严格为 0/1: {path}")
        actual_sha256 = sha256_file(path)
        actual_count = int(np.count_nonzero(array))
        if require_formal and actual_count <= 0:
            raise ValueError(f"正式 observed mask 不得为空域: {path}")
        if str(record.get("sha256", "")) != actual_sha256:
            raise ValueError(f"observed mask SHA-256 不匹配: {path}")
        if int(record.get("observed_voxels", -1)) != actual_count:
            raise ValueError(f"observed mask voxel count 不匹配: {path}")
        normalized_records.append(
            {
                "frame_id": frame_id,
                "file": file_name,
                "sha256": actual_sha256,
                "observed_voxels": actual_count,
            }
        )
        total_observed += actual_count

    if record_ids != set(predictions) or set(observed_masks) != set(predictions):
        raise ValueError("observed mask frame 集合与 prediction 不一致")
    if int(contract.get("observed_voxels", -1)) != total_observed:
        raise ValueError("observed mask 总 voxel count 不一致")
    if contract.get("files_sha256") != _observed_mask_records_digest(
        normalized_records
    ):
        raise ValueError("observed mask records digest 不匹配")
    return contract


def _discover_source_frames(folder: str, label: str) -> Dict[str, str]:
    frames: Dict[str, str] = {}
    for name in sorted(os.listdir(folder)):
        match = SOURCE_PATTERN.fullmatch(name)
        if not match:
            raise ValueError(f"{label} 目录含未知文件: {os.path.join(folder, name)}")
        path = os.path.join(folder, name)
        _require_regular_file(path, label)
        frame_id = match.group(1)
        if frame_id in frames:
            raise ValueError(f"{label} 帧同时存在 npy/npz: {frame_id}")
        frames[frame_id] = path
    return frames


def _validate_continuous_frame_ids(frame_ids: Sequence[str]) -> None:
    numeric = [int(frame_id) for frame_id in frame_ids]
    expected = list(range(numeric[0], numeric[0] + len(numeric)))
    if numeric != expected:
        raise ValueError(f"prediction 帧不连续: {list(frame_ids)}")


def _load_array(path: str) -> np.ndarray:
    try:
        array = load_sparse_voxel(path) if path.endswith(".npz") else np.load(
            path,
            allow_pickle=False,
        )
    except Exception as exc:
        raise ValueError(f"无法读取数组 {path}: {exc}") from exc
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"数组含非有限（finite）值: {path}")
    return array


def _load_prediction(path: str, target_size: Sequence[int]) -> np.ndarray:
    array = _load_array(path)
    expected_spatial = tuple(int(value) for value in target_size)
    if array.ndim != 4 or array.shape[0] < 1 or tuple(array.shape[1:]) != expected_spatial:
        raise ValueError(
            f"prediction shape 必须为 (C,Z,X,Y) 且空间为 {expected_spatial}: "
            f"{path} -> {array.shape}"
        )
    return array


def _load_source_as_czxy(
    path: str,
    target_size: Sequence[int],
    source_pc_range: Sequence[float],
    model_pc_range: Sequence[float],
    is_target: bool,
) -> np.ndarray:
    array = _load_array(path)
    if array.ndim != 4 or array.shape[-1] < 1:
        raise ValueError(f"source shape 必须为 (X,Y,Z,C): {path} -> {array.shape}")
    tensor = torch.from_numpy(array).permute(3, 2, 0, 1)
    tensor = crop_voxel_channels_to_pc_range(
        tensor,
        source_pc_range,
        model_pc_range,
    )
    mask_channel = 3 if is_target and tensor.shape[0] > 3 else None
    tensor = resize_voxel_channels(
        tensor,
        target_size,
        mask_channel=mask_channel,
    )
    return tensor.cpu().numpy().astype(np.float32, copy=False)


def _voxel_czxy_to_points(
    voxel: np.ndarray,
    pc_range: Sequence[float],
    threshold: float,
    voxel_size: Optional[Sequence[float]] = None,
) -> np.ndarray:
    occ = np.asarray(voxel, dtype=np.float32)[0]
    indices = np.argwhere(occ > float(threshold))
    if indices.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    z_count, x_count, y_count = occ.shape
    x_min, y_min, z_min, x_max, y_max, z_max = [float(v) for v in pc_range]
    z_idx, x_idx, y_idx = indices[:, 0], indices[:, 1], indices[:, 2]
    if voxel_size is None:
        step_x = (x_max - x_min) / max(x_count, 1)
        step_y = (y_max - y_min) / max(y_count, 1)
        step_z = (z_max - z_min) / max(z_count, 1)
    else:
        if len(voxel_size) != 3 or not all(float(value) > 0.0 for value in voxel_size):
            raise ValueError(f"voxel_size 非法: {voxel_size}")
        step_x, step_y, step_z = [float(value) for value in voxel_size]
    x = x_min + (x_idx.astype(np.float32) + 0.5) * step_x
    y = y_min + (y_idx.astype(np.float32) + 0.5) * step_y
    z = z_min + (z_idx.astype(np.float32) + 0.5) * step_z
    return np.stack([x, y, z], axis=1).astype(np.float32)


def _chamfer(points_a: np.ndarray, points_b: np.ndarray) -> float:
    if cKDTree is None:
        raise RuntimeError("scipy is required for Chamfer distance")
    if points_a.shape[0] == 0 or points_b.shape[0] == 0:
        return float("nan")
    tree_a = cKDTree(points_a[:, :3])
    tree_b = cKDTree(points_b[:, :3])
    dists_a, _ = tree_b.query(points_a[:, :3], k=1)
    dists_b, _ = tree_a.query(points_b[:, :3], k=1)
    return float(np.mean(dists_a) + np.mean(dists_b))


def _centroid_delta(points_a: np.ndarray, points_b: np.ndarray) -> Tuple[float, float, float]:
    if points_a.shape[0] == 0 or points_b.shape[0] == 0:
        return float("nan"), float("nan"), float("nan")
    delta = np.mean(points_a[:, :3], axis=0) - np.mean(points_b[:, :3], axis=0)
    return float(delta[0]), float(delta[1]), float(delta[2])


def _read_lidar_mapping(
    raw_livox_dir: str,
    lidar_index_file: str,
    frame_ids: Sequence[str],
) -> Dict[str, str]:
    if bool(raw_livox_dir) != bool(lidar_index_file):
        raise ValueError("raw_livox_dir 与 lidar_index_file 必须同时（together）提供")
    if not raw_livox_dir:
        return {}
    _require_directory(raw_livox_dir, "raw LiDAR")
    _require_regular_file(lidar_index_file, "LiDAR index")
    lidar_files = sorted(
        name for name in os.listdir(raw_livox_dir) if name.endswith(".npy")
    )
    if len(lidar_files) != len(os.listdir(raw_livox_dir)):
        raise ValueError(f"raw LiDAR 目录含未知文件: {raw_livox_dir}")
    with open(lidar_index_file, "r", encoding="utf-8") as handle:
        try:
            indices = [int(line.strip()) for line in handle if line.strip()]
        except ValueError as exc:
            raise ValueError(f"LiDAR index 含非整数: {lidar_index_file}") from exc

    mapping = {}
    for frame_id in frame_ids:
        frame_position = int(frame_id)
        if frame_position >= len(indices):
            raise ValueError(f"LiDAR index 帧位置越界（bounds）: {frame_id}")
        raw_index = indices[frame_position]
        if raw_index < 0 or raw_index >= len(lidar_files):
            raise ValueError(
                f"LiDAR index 值越界（bounds）: frame={frame_id}, index={raw_index}"
            )
        path = os.path.join(raw_livox_dir, lidar_files[raw_index])
        _require_regular_file(path, "raw LiDAR")
        points = _load_array(path)
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(f"raw LiDAR shape 必须为 (N,>=3): {path} -> {points.shape}")
        mapping[frame_id] = path
    return mapping


def _csv_value(value):
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.8f}" if np.isfinite(float(value)) else ""
    return value


def _finite_mean(rows: Sequence[dict], field: str):
    values = []
    for row in rows:
        value = row.get(field)
        if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(
            float(value)
        ):
            values.append(float(value))
    return float(np.mean(values)) if values else None


def _write_json_atomic(path: str, payload: dict) -> None:
    temp_path = f"{path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temp_path, path)


def evaluate_saved_predictions(
    pred_voxel_dir: str,
    radar_voxel_dir: str,
    target_voxel_dir: str,
    output_dir: str,
    run_metadata_path: str = "",
    raw_livox_dir: str = "",
    lidar_index_file: str = "",
    occ_threshold: Optional[float] = None,
    target_threshold: float = 0.5,
    source_pc_range: Optional[Sequence[float]] = None,
    model_pc_range: Optional[Sequence[float]] = None,
    target_size: Optional[Sequence[int]] = None,
    max_files: int = 0,
) -> dict:
    """严格配对并评价已保存 voxel；全部输入通过 preflight 后才写输出。"""
    _require_directory(pred_voxel_dir, "prediction")
    _require_directory(radar_voxel_dir, "radar")
    _require_directory(target_voxel_dir, "target")
    _validate_output_directory(output_dir)
    if not np.isfinite(float(target_threshold)):
        raise ValueError(f"target_threshold 含非有限值: {target_threshold}")
    if int(max_files) < 0:
        raise ValueError("max_files must be >= 0")

    resolved = _resolve_parameters(
        pred_voxel_dir,
        run_metadata_path,
        occ_threshold,
        source_pc_range,
        model_pc_range,
        target_size,
    )
    predictions, uncertainties, observed_masks = _discover_prediction_frames(
        pred_voxel_dir
    )
    observed_contract = _validate_observed_mask_contract(
        observed_masks,
        resolved["metadata"],
        predictions,
        resolved["target_size"],
    )
    radar_frames = _discover_source_frames(radar_voxel_dir, "radar")
    target_frames = _discover_source_frames(target_voxel_dir, "target")
    all_frame_ids = sorted(predictions)
    _validate_continuous_frame_ids(all_frame_ids)

    metadata_frame_count = resolved["metadata"].get("frame_count")
    if metadata_frame_count is not None and int(metadata_frame_count) != len(all_frame_ids):
        raise ValueError(
            "run metadata frame_count 与 prediction frame 数不一致: "
            f"{metadata_frame_count} != {len(all_frame_ids)}"
        )
    missing_radar = sorted(set(all_frame_ids) - set(radar_frames))
    missing_target = sorted(set(all_frame_ids) - set(target_frames))
    if missing_radar or missing_target:
        raise ValueError(
            f"frame 帧配对失败: missing_radar={missing_radar}, "
            f"missing_target={missing_target}"
        )

    frame_ids = (
        all_frame_ids[: int(max_files)] if int(max_files) > 0 else all_frame_ids
    )
    lidar_mapping = _read_lidar_mapping(
        raw_livox_dir,
        lidar_index_file,
        frame_ids,
    )

    # NOTE: preflight 逐帧读取但不常驻缓存，避免正式大场景验证占用过量内存。
    for frame_id in frame_ids:
        _load_prediction(predictions[frame_id], resolved["target_size"])
        _load_source_as_czxy(
            radar_frames[frame_id],
            resolved["target_size"],
            resolved["source_pc_range"],
            resolved["model_pc_range"],
            is_target=False,
        )
        _load_source_as_czxy(
            target_frames[frame_id],
            resolved["target_size"],
            resolved["source_pc_range"],
            resolved["model_pc_range"],
            is_target=True,
        )
        if frame_id in uncertainties:
            uncertainty = _load_array(uncertainties[frame_id])
            if uncertainty.squeeze().ndim != 3:
                raise ValueError(
                    f"uncertainty shape 必须可压缩为 3D: "
                    f"{uncertainties[frame_id]} -> {uncertainty.shape}"
                )
        if frame_id in observed_masks:
            _load_array(observed_masks[frame_id])

    os.makedirs(output_dir, exist_ok=True)
    rows: List[dict] = []
    for frame_id in frame_ids:
        pred = _load_prediction(predictions[frame_id], resolved["target_size"])
        radar = _load_source_as_czxy(
            radar_frames[frame_id],
            resolved["target_size"],
            resolved["source_pc_range"],
            resolved["model_pc_range"],
            is_target=False,
        )
        target = _load_source_as_czxy(
            target_frames[frame_id],
            resolved["target_size"],
            resolved["source_pc_range"],
            resolved["model_pc_range"],
            is_target=True,
        )
        if frame_id in observed_masks:
            observed_domain = _load_array(observed_masks[frame_id]).squeeze() > 0.5
            if observed_domain.shape != tuple(resolved["target_size"]):
                raise ValueError(
                    f"observed mask shape 与 target_size 不一致: {frame_id} -> "
                    f"{observed_domain.shape}"
                )
        else:
            observed_domain = np.ones(tuple(resolved["target_size"]), dtype=bool)

        # NOTE: ch0 是唯一 occupancy 通道；正式点云/占用指标只消费权威 observed 域。
        pred_for_metrics = pred.copy()
        radar_for_metrics = radar.copy()
        target_for_metrics = target.copy()
        pred_for_metrics[0] = np.where(observed_domain, pred[0], 0.0)
        radar_for_metrics[0] = np.where(observed_domain, radar[0], 0.0)
        target_for_metrics[0] = np.where(observed_domain, target[0], 0.0)
        pred_points = _voxel_czxy_to_points(
            pred_for_metrics,
            resolved["model_pc_range"],
            resolved["occ_threshold"],
            resolved["voxel_size"],
        )
        radar_points = _voxel_czxy_to_points(
            radar_for_metrics,
            resolved["model_pc_range"],
            resolved["occ_threshold"],
            resolved["voxel_size"],
        )
        target_points = _voxel_czxy_to_points(
            target_for_metrics,
            resolved["model_pc_range"],
            target_threshold,
            resolved["voxel_size"],
        )
        pred_delta = _centroid_delta(pred_points, target_points)
        radar_delta = _centroid_delta(radar_points, target_points)
        near_pred = filter_points_by_band(
            pred_points,
            resolved["model_pc_range"],
            x_min=0.0,
            x_max=20.0,
            z_min=-1.0,
        )
        near_target = filter_points_by_band(
            target_points,
            resolved["model_pc_range"],
            x_min=0.0,
            x_max=20.0,
            z_min=-1.0,
        )
        prf = occupancy_prf(
            near_pred, near_target, resolved["model_pc_range"], cell_size=0.5
        )
        iou = bev_iou(
            near_pred, near_target, resolved["model_pc_range"], cell_size=0.5
        )
        nn = nearest_neighbor_metrics(near_pred, near_target, thresholds=(2.0,))

        lidar_path = lidar_mapping.get(frame_id, "")
        lidar_points = _load_array(lidar_path) if lidar_path else np.zeros((0, 3))
        row = {
            "frame_id": frame_id,
            "pred_file": os.path.basename(predictions[frame_id]),
            "radar_file": os.path.basename(radar_frames[frame_id]),
            "target_file": os.path.basename(target_frames[frame_id]),
            "lidar_file": os.path.basename(lidar_path) if lidar_path else "",
            "observed_mask_file": (
                os.path.basename(observed_masks[frame_id])
                if frame_id in observed_masks
                else ""
            ),
            "observed_voxels": (
                int(np.count_nonzero(_load_array(observed_masks[frame_id])))
                if frame_id in observed_masks
                else ""
            ),
            "effective_occ_threshold": float(resolved["occ_threshold"]),
            "target_threshold": float(target_threshold),
            "pred_point_count": int(pred_points.shape[0]),
            "radar_point_count": int(radar_points.shape[0]),
            "target_point_count": int(target_points.shape[0]),
            "lidar_point_count": int(lidar_points.shape[0]) if lidar_path else "",
            "pred_target_chamfer": _chamfer(pred_points, target_points),
            "radar_target_chamfer": _chamfer(radar_points, target_points),
            "raw_lidar_chamfer": (
                _chamfer(pred_points, lidar_points) if lidar_path else float("nan")
            ),
            "pred_target_count_ratio": (
                float(pred_points.shape[0] / target_points.shape[0])
                if target_points.shape[0] > 0
                else float("nan")
            ),
            "pred_target_dx": pred_delta[0],
            "pred_target_dy": pred_delta[1],
            "pred_target_dz": pred_delta[2],
            "radar_target_dx": radar_delta[0],
            "radar_target_dy": radar_delta[1],
            "radar_target_dz": radar_delta[2],
            "near_precision": prf["precision"],
            "near_recall": prf["recall"],
            "near_bev_iou": iou["bev_iou"],
            "near_nn_mean": nn["nn_mean"],
            "near_match_ratio_2": nn["match_ratio_2"],
            "uncertainty_ece": float("nan"),
            "uncertainty_brier": float("nan"),
            "uncertainty_nll": float("nan"),
            "uncertainty_error_corr": float("nan"),
        }
        if frame_id in uncertainties:
            uncertainty = _load_array(uncertainties[frame_id])
            calibration = uncertainty_calibration_metrics(
                pred[0],
                (target[0] > float(target_threshold)).astype(np.float32),
                uncertainty,
                occ_threshold=float(resolved["occ_threshold"]),
                observed_mask=observed_domain,
            )
            for field in (
                "uncertainty_ece",
                "uncertainty_brier",
                "uncertainty_nll",
                "uncertainty_error_corr",
            ):
                row[field] = calibration[field]
        rows.append(row)

    csv_path = os.path.join(output_dir, "evaluation_frames.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FRAME_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in FRAME_FIELDS})

    aggregate_fields = [
        field
        for field in FRAME_FIELDS
        if field
        not in {
            "frame_id",
            "pred_file",
            "radar_file",
            "target_file",
            "lidar_file",
            "observed_mask_file",
        }
    ]
    aggregate_metrics = {
        f"mean_{field}": _finite_mean(rows, field)
        for field in aggregate_fields
    }
    summary = {
        "protocol": (
            FORMAL_SAVED_EVALUATION_PROTOCOL
            if resolved["formal_run"]
            else DIAGNOSTIC_SAVED_EVALUATION_PROTOCOL
        ),
        "formal_protocol": bool(resolved["formal_run"]),
        "stage": "offline_evaluation",
        "prediction_unchanged": True,
        "occupancy_metric_domain": (
            "external_authoritative_observed_mask"
            if observed_contract
            else "full_grid_diagnostic"
        ),
        "metric_aggregation": "finite_per_frame_macro_mean_v1",
        "auxiliary_metric_domain": {
            "radar_baseline": "same_authoritative_observed_mask",
            "raw_lidar_chamfer": "unmasked_raw_lidar_diagnostic_reference",
        },
        "frame_count": len(frame_ids),
        "prediction_frame_count": len(all_frame_ids),
        "observed_mask_protocol": (
            observed_contract.get("protocol") if observed_contract else None
        ),
        "observed_mask_frame_count": (
            int(observed_contract["frame_count"]) if observed_contract else 0
        ),
        "selected_frame_ids": list(frame_ids),
        "occ_threshold": float(resolved["occ_threshold"]),
        "occ_threshold_source": resolved["parameter_sources"]["occ_threshold"],
        "occupancy_threshold_artifact_sha256": resolved["metadata"].get(
            "occupancy_threshold_artifact_sha256"
        ),
        "target_threshold": float(target_threshold),
        "target_size": [int(value) for value in resolved["target_size"]],
        "source_pc_range": [float(value) for value in resolved["source_pc_range"]],
        "model_pc_range": [float(value) for value in resolved["model_pc_range"]],
        "voxel_size": [float(value) for value in resolved["voxel_size"]],
        "parameter_sources": resolved["parameter_sources"],
        "run_metadata_path": (
            os.path.abspath(resolved["metadata_path"])
            if resolved["metadata_path"]
            else ""
        ),
        "pred_voxel_dir": os.path.abspath(pred_voxel_dir),
        "radar_voxel_dir": os.path.abspath(radar_voxel_dir),
        "target_voxel_dir": os.path.abspath(target_voxel_dir),
        "raw_livox_dir": os.path.abspath(raw_livox_dir) if raw_livox_dir else "",
        "lidar_index_file": (
            os.path.abspath(lidar_index_file) if lidar_index_file else ""
        ),
        "formal_metrics": (
            {
                f"mean_{field}": aggregate_metrics[f"mean_{field}"]
                for field in FORMAL_METRIC_FIELDS
            }
            if resolved["formal_run"]
            else {}
        ),
        "metrics": aggregate_metrics,
    }
    _write_json_atomic(
        os.path.join(output_dir, "evaluation_summary.json"),
        summary,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="离线评价部署阶段已保存的 prediction voxel"
    )
    parser.add_argument("--pred_voxel_dir", required=True)
    parser.add_argument("--radar_voxel_dir", required=True)
    parser.add_argument("--target_voxel_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_metadata_path", default="")
    parser.add_argument("--raw_livox_dir", default="")
    parser.add_argument("--lidar_index_file", default="")
    parser.add_argument("--occ_threshold", type=float, default=None)
    parser.add_argument("--target_threshold", type=float, default=0.5)
    parser.add_argument("--source_pc_range", type=float, nargs=6, default=None)
    parser.add_argument("--model_pc_range", type=float, nargs=6, default=None)
    parser.add_argument("--target_size", type=int, nargs=3, default=None)
    parser.add_argument("--max_files", type=int, default=0)
    args = parser.parse_args()
    try:
        summary = evaluate_saved_predictions(**vars(args))
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
