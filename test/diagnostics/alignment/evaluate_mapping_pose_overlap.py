#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""文件功能：用多窗口静态 LiDAR 重合度反证 mapping pose-frame 候选。"""

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
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - 目标环境有 SciPy，保留小数据回退。
    cKDTree = None


PROTOCOL = "mapping_pose_overlap_diagnostic_v1"
CANDIDATE_PROTOCOLS = {
    "mapping_pose_candidate_diagnostic_v1",
    "mapping_pose_candidate_diagnostic_v2",
}
HYPOTHESES = ("gt_as_imu", "gt_as_lidar")
POSE_HYPOTHESIS_MARKERS = {
    "gt_as_imu": "gt_pose_is_imu",
    "gt_as_lidar": "gt_pose_is_lidar_then_convert_to_imu_body",
}
CSV_FIELDS = (
    "hypothesis",
    "frame_a",
    "frame_b",
    "timestamp_a",
    "timestamp_b",
    "delta_s",
    "relative_rotation_deg",
    "relative_translation_m",
    "point_count_a",
    "point_count_b",
    "symmetric_nn_mean_m",
    "symmetric_nn_median_m",
    "symmetric_nn_p90_m",
    "match_ratio_0_5m",
    "match_ratio_1_0m",
)


def _regular_file(path: str, label: str) -> str:
    """只接受非符号链接普通文件。"""
    normalized = os.path.abspath(os.path.expanduser(path))
    if os.path.islink(normalized):
        raise ValueError(f"{label} 不得是符号链接: {normalized}")
    if not os.path.isfile(normalized):
        raise FileNotFoundError(f"缺少 {label}: {normalized}")
    return normalized


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json_value(value: object) -> str:
    """按 dataset manifest 的 canonical JSON 规则计算内容身份。"""
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _verify_sha256(path: str, expected: object, label: str) -> str:
    expected_text = str(expected or "").strip().lower()
    if len(expected_text) != 64:
        raise ValueError(f"{label} 缺少有效 SHA-256 收据")
    actual = _sha256_file(path)
    if actual != expected_text:
        raise ValueError(
            f"{label} SHA-256 不匹配: expected={expected_text}, actual={actual}"
        )
    return actual


def _candidate_member(root: str, relative_path: object, label: str) -> str:
    """解析候选目录成员并拒绝绝对路径、越界和链接。"""
    text = str(relative_path or "").strip()
    if not text or os.path.isabs(text):
        raise ValueError(f"{label} 必须是候选目录内相对路径")
    root_real = os.path.realpath(root)
    path = os.path.abspath(os.path.join(root, text))
    if os.path.commonpath((root_real, os.path.realpath(path))) != root_real:
        raise ValueError(f"{label} 越出候选目录: {text}")
    return _regular_file(path, label)


def _scene_member(root: str, relative_path: object, label: str) -> str:
    """解析场景 manifest 成员并拒绝目录逃逸。"""
    text = str(relative_path or "").strip()
    if not text or os.path.isabs(text):
        raise ValueError(f"{label} 必须是场景内相对路径")
    root_real = os.path.realpath(root)
    path = os.path.abspath(os.path.join(root, text))
    if os.path.commonpath((root_real, os.path.realpath(path))) != root_real:
        raise ValueError(f"{label} 越出场景目录: {text}")
    return _regular_file(path, label)


def _validate_transform(matrix: object, label: str) -> np.ndarray:
    transform = np.asarray(matrix, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"{label} 必须是有限 4x4 矩阵")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{label} 齐次矩阵末行非法")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{label} 旋转矩阵不正交")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-5):
        raise ValueError(f"{label} 旋转矩阵 determinant 必须约为 1")
    return transform


def _quaternion_to_matrix(values: Sequence[float]) -> np.ndarray:
    quaternion = np.asarray(values, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("pose 四元数必须包含四个有限数")
    norm = float(np.linalg.norm(quaternion))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-5):
        raise ValueError(f"pose 四元数未归一化: norm={norm:.9f}")
    x, y, z, w = quaternion / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _load_diagnostic_extrinsic(path: str) -> np.ndarray:
    """读取带 formal=false 文件头的诊断 R/T，禁止把普通 formal 文件混入。"""
    with open(path, "r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    if not any("formal=false" in line.lower() for line in lines if line.startswith("#")):
        raise ValueError("LiDAR→body 候选缺少 formal=false 标记")
    rotation_values: Optional[List[float]] = None
    translation_values: Optional[List[float]] = None
    for line in lines:
        if line.startswith("R:"):
            rotation_values = [float(value) for value in line[2:].split()]
        elif line.startswith("T:"):
            translation_values = [float(value) for value in line[2:].split()]
    if rotation_values is None or len(rotation_values) != 9:
        raise ValueError("LiDAR→body 候选 R 必须包含 9 个数")
    if translation_values is None or len(translation_values) != 3:
        raise ValueError("LiDAR→body 候选 T 必须包含 3 个数")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation_values).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation_values)
    return _validate_transform(transform, "LiDAR→body 候选")


def _load_pose_csv(path: str, hypothesis: str) -> Dict[str, Dict[str, object]]:
    """加载诊断 pose，并验证 frame 唯一、时间有限和内容标记。"""
    expected_marker = POSE_HYPOTHESIS_MARKERS[hypothesis]
    records: Dict[str, Dict[str, object]] = {}
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
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
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{hypothesis} pose CSV 字段不完整")
        for row in reader:
            frame_id = str(row["frame"]).strip()
            if not frame_id or frame_id in records:
                raise ValueError(f"{hypothesis} pose frame 为空或重复: {frame_id}")
            if str(row["diagnostic_formal"]).strip().lower() != "false":
                raise ValueError(f"{hypothesis} pose 缺少 diagnostic_formal=false")
            if str(row["gt_pose_hypothesis"]).strip() != expected_marker:
                raise ValueError(f"{hypothesis} pose hypothesis 标记不匹配")
            timestamp = float(row["timestamp"])
            translation = np.asarray(
                [float(row["tx"]), float(row["ty"]), float(row["tz"])],
                dtype=np.float64,
            )
            if not math.isfinite(timestamp) or not np.all(np.isfinite(translation)):
                raise ValueError(f"{hypothesis} pose 含非有限数: {frame_id}")
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = _quaternion_to_matrix(
                [float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])]
            )
            transform[:3, 3] = translation
            records[frame_id] = {
                "timestamp": timestamp,
                "T_local_body": _validate_transform(
                    transform,
                    f"{hypothesis} frame {frame_id}",
                ),
            }
    if not records:
        raise ValueError(f"{hypothesis} pose CSV 没有可用帧")
    return records


def _load_candidate_contract(candidate_dir: str) -> Dict[str, object]:
    root = os.path.abspath(os.path.expanduser(candidate_dir))
    if os.path.islink(root) or not os.path.isdir(root):
        raise ValueError(f"candidate_dir 必须是非链接目录: {root}")
    audit_path = _regular_file(os.path.join(root, "audit.json"), "候选 audit")
    with open(audit_path, "r", encoding="utf-8") as handle:
        audit = json.load(handle)
    if audit.get("protocol") not in CANDIDATE_PROTOCOLS or audit.get("formal") is not False:
        raise ValueError("candidate audit 不是受支持的 formal=false 诊断协议")

    extrinsic_receipt = audit.get("candidate_lidar_to_body", {})
    extrinsic_path = _candidate_member(
        root,
        extrinsic_receipt.get("file"),
        "LiDAR→body 候选",
    )
    _verify_sha256(
        extrinsic_path,
        extrinsic_receipt.get("sha256"),
        "LiDAR→body 候选",
    )
    body_from_lidar = _load_diagnostic_extrinsic(extrinsic_path)
    receipt_matrix = _validate_transform(
        extrinsic_receipt.get("matrix_4x4"),
        "candidate audit LiDAR→body",
    )
    if not np.allclose(body_from_lidar, receipt_matrix, atol=1e-8):
        raise ValueError("LiDAR→body 候选文件与 audit matrix 不一致")

    pose_tables: Dict[str, Dict[str, Dict[str, object]]] = {}
    pose_paths: Dict[str, str] = {}
    pose_receipts = audit.get("pose_candidates", {})
    for hypothesis in HYPOTHESES:
        receipt = pose_receipts.get(hypothesis, {})
        path = _candidate_member(root, receipt.get("file"), f"{hypothesis} pose")
        _verify_sha256(path, receipt.get("sha256"), f"{hypothesis} pose")
        pose_paths[hypothesis] = path
        pose_tables[hypothesis] = _load_pose_csv(path, hypothesis)

    common = set(pose_tables[HYPOTHESES[0]]) & set(pose_tables[HYPOTHESES[1]])
    if not common:
        raise ValueError("两个 pose 假设没有共同 frame")
    for frame_id in common:
        first_time = float(pose_tables[HYPOTHESES[0]][frame_id]["timestamp"])
        second_time = float(pose_tables[HYPOTHESES[1]][frame_id]["timestamp"])
        if not math.isclose(first_time, second_time, abs_tol=1e-6):
            raise ValueError(f"两个 pose 假设 timestamp 不一致: {frame_id}")

    inputs = audit.get("inputs", {})
    sync_receipt = inputs.get("radar_sync_csv", {})
    radar_lidar_snapshot: Optional[Dict[str, object]] = None
    if audit.get("protocol") == "mapping_pose_candidate_diagnostic_v2":
        snapshot_receipt = inputs.get("radar_lidar_sync_snapshot", {})
        snapshot_path = _candidate_member(
            root,
            snapshot_receipt.get("file"),
            "Radar--LiDAR sync snapshot",
        )
        snapshot_sha256 = _verify_sha256(
            snapshot_path,
            snapshot_receipt.get("sha256"),
            "Radar--LiDAR sync snapshot",
        )
        if snapshot_sha256 != str(snapshot_receipt.get("source_sha256") or ""):
            raise ValueError("Radar--LiDAR sync snapshot 与 source SHA-256 不一致")
        radar_lidar_snapshot = {
            "path": snapshot_path,
            "sha256": snapshot_sha256,
        }
    return {
        "audit": audit,
        "audit_path": audit_path,
        "audit_sha256": _sha256_file(audit_path),
        "body_from_lidar": body_from_lidar,
        "extrinsic_path": extrinsic_path,
        "pose_paths": pose_paths,
        "pose_tables": pose_tables,
        "common_frames": common,
        "sync_sha256": str(sync_receipt.get("sha256") or ""),
        "radar_lidar_snapshot": radar_lidar_snapshot,
    }


def _load_scene_contract(scene_dir: str) -> Dict[str, object]:
    root = os.path.abspath(os.path.expanduser(scene_dir))
    if os.path.islink(root) or not os.path.isdir(root):
        raise ValueError(f"processed_scene_dir 必须是非链接目录: {root}")
    policy_path = _regular_file(os.path.join(root, "preprocess_policy.json"), "预处理 policy")
    manifest_path = _regular_file(os.path.join(root, "dataset_manifest.json"), "数据 manifest")
    sync_path = _regular_file(os.path.join(root, "radar_ir_sync.csv"), "Radar--IR sync")
    with open(policy_path, "r", encoding="utf-8") as handle:
        policy = json.load(handle)
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    recorded_content_sha256 = manifest.get("content_sha256")
    manifest_payload = {
        key: value for key, value in manifest.items() if key != "content_sha256"
    }
    if (
        not isinstance(recorded_content_sha256, str)
        or _sha256_json_value(manifest_payload) != recorded_content_sha256
    ):
        raise ValueError("dataset manifest content_sha256 不一致")
    if str(policy.get("align_to", "")).lower() != "lidar":
        raise ValueError("多窗口诊断只接受 align_to=lidar 的预处理数据")
    pc_range = np.asarray(policy.get("pc_range"), dtype=np.float64)
    voxel_size = np.asarray(policy.get("voxel_size"), dtype=np.float64)
    if pc_range.shape != (6,) or not np.all(np.isfinite(pc_range)):
        raise ValueError("preprocess policy pc_range 必须是 6 个有限数")
    if voxel_size.shape != (3,) or not np.all(np.isfinite(voxel_size)) or np.any(voxel_size <= 0):
        raise ValueError("preprocess policy voxel_size 必须是 3 个正有限数")
    expected_shape = np.rint((pc_range[3:] - pc_range[:3]) / voxel_size).astype(np.int64)
    if np.any(expected_shape <= 0):
        raise ValueError("preprocess policy 体素网格范围非法")

    lidar_records = manifest.get("modalities", {}).get("lidar_voxel")
    if not isinstance(lidar_records, list) or not lidar_records:
        raise ValueError("dataset manifest 缺少 lidar_voxel records")
    record_map: Dict[str, Mapping[str, object]] = {}
    for record in lidar_records:
        frame_id = str(record.get("frame_id", "")).strip()
        if not frame_id or frame_id in record_map:
            raise ValueError(f"dataset manifest LiDAR frame 为空或重复: {frame_id}")
        record_map[frame_id] = record
    if int(manifest.get("frame_count", -1)) != len(record_map):
        raise ValueError("dataset manifest frame_count 与 LiDAR records 不一致")
    preprocessing = manifest.get("preprocessing", {})
    recorded_policy_sha256 = preprocessing.get("policy_sha256")
    if recorded_policy_sha256 is not None and recorded_policy_sha256 != _sha256_file(policy_path):
        raise ValueError("preprocess policy SHA-256 与 dataset manifest 不一致")
    return {
        "root": root,
        "policy": policy,
        "policy_path": policy_path,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "sync_path": sync_path,
        "pc_range": pc_range,
        "voxel_size": voxel_size,
        "expected_shape": expected_shape,
        "lidar_records": record_map,
    }


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = (float(np.trace(rotation)) - 1.0) * 0.5
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _select_pairs(
    lidar_poses: Mapping[str, Mapping[str, object]],
    available_frames: Iterable[str],
    *,
    pair_delta_s: float,
    pair_delta_tolerance_s: float,
    min_rotation_deg: float,
    max_translation_m: float,
    max_pairs: int,
) -> Tuple[List[Dict[str, object]], int]:
    """从共同帧均匀抽取有旋转且仍有视野重合的 pair。"""
    frames = sorted(
        set(available_frames) & set(lidar_poses),
        key=lambda frame: float(lidar_poses[frame]["timestamp"]),
    )
    if len(frames) < 2:
        raise ValueError("共同 covered LiDAR frame 少于 2")
    timestamps = np.asarray(
        [float(lidar_poses[frame]["timestamp"]) for frame in frames],
        dtype=np.float64,
    )
    candidates: List[Dict[str, object]] = []
    for index, frame_a in enumerate(frames[:-1]):
        target = timestamps[index] + pair_delta_s
        right = int(np.searchsorted(timestamps, target, side="left"))
        choices = [candidate for candidate in (right - 1, right) if candidate > index and candidate < len(frames)]
        if not choices:
            continue
        other = min(choices, key=lambda candidate: abs(timestamps[candidate] - target))
        delta_s = float(timestamps[other] - timestamps[index])
        if abs(delta_s - pair_delta_s) > pair_delta_tolerance_s:
            continue
        frame_b = frames[other]
        first = np.asarray(lidar_poses[frame_a]["T_local_lidar"], dtype=np.float64)
        second = np.asarray(lidar_poses[frame_b]["T_local_lidar"], dtype=np.float64)
        relative = np.linalg.inv(first) @ second
        rotation_deg = _rotation_angle_deg(relative[:3, :3])
        translation_m = float(np.linalg.norm(relative[:3, 3]))
        if rotation_deg < min_rotation_deg or translation_m > max_translation_m:
            continue
        candidates.append(
            {
                "frame_a": frame_a,
                "frame_b": frame_b,
                "timestamp_a": float(timestamps[index]),
                "timestamp_b": float(timestamps[other]),
                "delta_s": delta_s,
                "relative_rotation_deg": rotation_deg,
                "relative_translation_m": translation_m,
            }
        )
    if not candidates:
        raise ValueError("当前时间/旋转/位移门限没有可诊断 frame pair")
    if len(candidates) <= max_pairs:
        return candidates, len(candidates)
    indices = np.rint(np.linspace(0, len(candidates) - 1, max_pairs)).astype(np.int64)
    selected = [candidates[int(index)] for index in np.unique(indices)]
    return selected, len(candidates)


def _load_lidar_points(
    scene: Mapping[str, object],
    frame_id: str,
    *,
    min_sensor_range_m: float,
    max_sensor_range_m: float,
    max_points_per_frame: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    record = scene["lidar_records"].get(frame_id)
    if record is None:
        raise ValueError(f"dataset manifest 缺少所选 LiDAR frame: {frame_id}")
    path = _scene_member(
        str(scene["root"]),
        record.get("path"),
        f"LiDAR voxel {frame_id}",
    )
    actual_sha256 = _verify_sha256(path, record.get("sha256"), f"LiDAR voxel {frame_id}")
    recorded_size = record.get("size")
    if type(recorded_size) is not int or os.path.getsize(path) != recorded_size:
        raise ValueError(f"LiDAR voxel {frame_id} size 与 manifest 不一致")
    if not path.endswith(".npz"):
        raise ValueError("多窗口诊断当前只接受稀疏 .npz LiDAR voxel")
    with np.load(path, allow_pickle=False) as payload:
        required = {"coords", "features", "shape"}
        if not required.issubset(payload.files):
            raise ValueError(f"LiDAR voxel {frame_id} 缺少稀疏字段")
        coords = np.asarray(payload["coords"])
        features = np.asarray(payload["features"])
        shape = np.asarray(payload["shape"], dtype=np.int64).reshape(-1)
    if coords.ndim != 2 or coords.shape[1] != 3 or features.ndim != 2:
        raise ValueError(f"LiDAR voxel {frame_id} 稀疏数组维度非法")
    if coords.shape[0] != features.shape[0] or features.shape[1] < 1:
        raise ValueError(f"LiDAR voxel {frame_id} coords/features 数量不一致")
    expected_shape = np.asarray(scene["expected_shape"], dtype=np.int64)
    if shape.size < 3 or not np.array_equal(shape[:3], expected_shape):
        raise ValueError(f"LiDAR voxel {frame_id} shape 与 preprocess policy 不一致")
    if not np.issubdtype(coords.dtype, np.integer):
        raise ValueError(f"LiDAR voxel {frame_id} coords 必须为整数")
    occupied = np.asarray(features[:, 0] > 0.0)
    coords = coords[occupied].astype(np.float64, copy=False)
    if coords.size == 0:
        raise ValueError(f"LiDAR voxel {frame_id} 没有 occupied 体素")
    if np.any(coords < 0) or np.any(coords >= expected_shape[None, :]):
        raise ValueError(f"LiDAR voxel {frame_id} coords 越界")
    pc_min = np.asarray(scene["pc_range"], dtype=np.float64)[:3]
    voxel_size = np.asarray(scene["voxel_size"], dtype=np.float64)
    points = pc_min + (coords + 0.5) * voxel_size
    distance = np.linalg.norm(points, axis=1)
    points = points[
        (distance >= min_sensor_range_m) & (distance <= max_sensor_range_m)
    ]
    if points.shape[0] == 0:
        raise ValueError(f"LiDAR voxel {frame_id} 在诊断距离范围内没有点")
    if points.shape[0] > max_points_per_frame:
        indices = np.rint(
            np.linspace(0, points.shape[0] - 1, max_points_per_frame)
        ).astype(np.int64)
        points = points[indices]
    return points.astype(np.float64, copy=False), {
        "frame_id": frame_id,
        "path": str(record.get("path")),
        "sha256": actual_sha256,
        "selected_point_count": int(points.shape[0]),
    }


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return (transform[:3, :3] @ points.T).T + transform[:3, 3]


def _nearest_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if cKDTree is not None:
        distances, _ = cKDTree(target).query(source, k=1)
        return np.asarray(distances, dtype=np.float64)
    distances: List[np.ndarray] = []
    for start in range(0, source.shape[0], 256):
        chunk = source[start : start + 256]
        squared = np.sum((chunk[:, None, :] - target[None, :, :]) ** 2, axis=2)
        distances.append(np.sqrt(np.min(squared, axis=1)))
    return np.concatenate(distances)


def _pair_metrics(first: np.ndarray, second: np.ndarray) -> Dict[str, float]:
    distances = np.concatenate(
        [_nearest_distances(first, second), _nearest_distances(second, first)]
    )
    return {
        "symmetric_nn_mean_m": float(np.mean(distances)),
        "symmetric_nn_median_m": float(np.median(distances)),
        "symmetric_nn_p90_m": float(np.quantile(distances, 0.9)),
        "match_ratio_0_5m": float(np.mean(distances <= 0.5)),
        "match_ratio_1_0m": float(np.mean(distances <= 1.0)),
    }


def _finite_summary(values: Sequence[float]) -> Dict[str, Optional[float]]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"min": None, "median": None, "p90": None, "max": None}
    if not np.all(np.isfinite(array)):
        raise ValueError("指标汇总出现非有限数")
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(np.max(array)),
    }


def _fresh_output(path: str) -> str:
    normalized = os.path.abspath(os.path.expanduser(path))
    if os.path.islink(normalized):
        raise ValueError(f"output_dir 不得是符号链接: {normalized}")
    if os.path.exists(normalized):
        if not os.path.isdir(normalized):
            raise ValueError(f"output_dir 已存在且不是目录: {normalized}")
        if os.listdir(normalized):
            raise ValueError(f"output_dir 已存在且非空，拒绝覆盖: {normalized}")
    parent = os.path.dirname(normalized)
    if not os.path.isdir(parent):
        raise FileNotFoundError(f"output_dir 父目录不存在: {parent}")
    return normalized


def _write_text_atomic(path: str, text: str) -> None:
    fd, temporary = tempfile.mkstemp(prefix=".tmp_pose_overlap_", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_csv_atomic(path: str, rows: Sequence[Mapping[str, object]]) -> None:
    fd, temporary = tempfile.mkstemp(prefix=".tmp_pose_overlap_", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def evaluate_mapping_pose_overlap(
    *,
    processed_scene_dir: str,
    candidate_dir: str,
    output_dir: str,
    pair_delta_s: float = 1.0,
    pair_delta_tolerance_s: float = 0.15,
    min_rotation_deg: float = 3.0,
    max_translation_m: float = 12.0,
    min_sensor_range_m: float = 2.0,
    max_sensor_range_m: float = 50.0,
    max_pairs: int = 48,
    max_points_per_frame: int = 20000,
    command_line: Optional[str] = None,
) -> Dict[str, object]:
    """比较两种 GT frame 假设；结果只能用于反证，不能发布 formal pose。"""
    numeric_positive = {
        "pair_delta_s": pair_delta_s,
        "pair_delta_tolerance_s": pair_delta_tolerance_s,
        "max_translation_m": max_translation_m,
        "max_sensor_range_m": max_sensor_range_m,
    }
    for label, value in numeric_positive.items():
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{label} 必须为正有限数")
    if not math.isfinite(float(min_rotation_deg)) or min_rotation_deg < 0.0:
        raise ValueError("min_rotation_deg 必须为非负有限数")
    if not math.isfinite(float(min_sensor_range_m)) or min_sensor_range_m < 0.0:
        raise ValueError("min_sensor_range_m 必须为非负有限数")
    if max_sensor_range_m <= min_sensor_range_m:
        raise ValueError("max_sensor_range_m 必须大于 min_sensor_range_m")
    if int(max_pairs) <= 0 or int(max_points_per_frame) <= 0:
        raise ValueError("max_pairs 和 max_points_per_frame 必须为正整数")
    normalized_output = _fresh_output(output_dir)

    candidate = _load_candidate_contract(candidate_dir)
    scene = _load_scene_contract(processed_scene_dir)
    _verify_sha256(
        str(scene["sync_path"]),
        candidate["sync_sha256"],
        "当前场景 Radar--IR sync 与候选输入",
    )

    body_from_lidar = np.asarray(candidate["body_from_lidar"], dtype=np.float64)
    local_lidar_poses: Dict[str, Dict[str, Dict[str, object]]] = {}
    for hypothesis in HYPOTHESES:
        local_lidar_poses[hypothesis] = {}
        for frame_id, record in candidate["pose_tables"][hypothesis].items():
            local_lidar_poses[hypothesis][frame_id] = {
                "timestamp": record["timestamp"],
                "T_local_lidar": _validate_transform(
                    np.asarray(record["T_local_body"]) @ body_from_lidar,
                    f"{hypothesis} frame {frame_id} LiDAR→local",
                ),
            }

    common_frames = (
        set(candidate["common_frames"])
        & set(scene["lidar_records"])
        & set(local_lidar_poses["gt_as_lidar"])
    )
    pairs, eligible_pair_count = _select_pairs(
        local_lidar_poses["gt_as_lidar"],
        common_frames,
        pair_delta_s=float(pair_delta_s),
        pair_delta_tolerance_s=float(pair_delta_tolerance_s),
        min_rotation_deg=float(min_rotation_deg),
        max_translation_m=float(max_translation_m),
        max_pairs=int(max_pairs),
    )

    selected_frames = sorted(
        {str(pair["frame_a"]) for pair in pairs}
        | {str(pair["frame_b"]) for pair in pairs}
    )
    points_by_frame: Dict[str, np.ndarray] = {}
    voxel_receipts: List[Dict[str, object]] = []
    for frame_id in selected_frames:
        points, receipt = _load_lidar_points(
            scene,
            frame_id,
            min_sensor_range_m=float(min_sensor_range_m),
            max_sensor_range_m=float(max_sensor_range_m),
            max_points_per_frame=int(max_points_per_frame),
        )
        points_by_frame[frame_id] = points
        voxel_receipts.append(receipt)

    rows: List[Dict[str, object]] = []
    summaries: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    for hypothesis in HYPOTHESES:
        hypothesis_rows: List[Dict[str, object]] = []
        for pair in pairs:
            frame_a = str(pair["frame_a"])
            frame_b = str(pair["frame_b"])
            first = _transform_points(
                np.asarray(local_lidar_poses[hypothesis][frame_a]["T_local_lidar"]),
                points_by_frame[frame_a],
            )
            second = _transform_points(
                np.asarray(local_lidar_poses[hypothesis][frame_b]["T_local_lidar"]),
                points_by_frame[frame_b],
            )
            metrics = _pair_metrics(first, second)
            row = {
                "hypothesis": hypothesis,
                **pair,
                "point_count_a": int(first.shape[0]),
                "point_count_b": int(second.shape[0]),
                **metrics,
            }
            hypothesis_rows.append(row)
            rows.append(row)
        summaries[hypothesis] = {
            "pair_mean_nn_m": _finite_summary(
                [float(row["symmetric_nn_mean_m"]) for row in hypothesis_rows]
            ),
            "pair_median_nn_m": _finite_summary(
                [float(row["symmetric_nn_median_m"]) for row in hypothesis_rows]
            ),
            "pair_p90_nn_m": _finite_summary(
                [float(row["symmetric_nn_p90_m"]) for row in hypothesis_rows]
            ),
            "pair_match_ratio_0_5m": _finite_summary(
                [float(row["match_ratio_0_5m"]) for row in hypothesis_rows]
            ),
            "pair_match_ratio_1_0m": _finite_summary(
                [float(row["match_ratio_1_0m"]) for row in hypothesis_rows]
            ),
        }

    ranking = sorted(
        HYPOTHESES,
        key=lambda hypothesis: (
            float(summaries[hypothesis]["pair_median_nn_m"]["median"]),
            hypothesis,
        ),
    )
    report: Dict[str, object] = {
        "protocol": PROTOCOL,
        "formal": False,
        "diagnostic_only": True,
        "inputs": {
            "processed_scene_dir": os.path.abspath(processed_scene_dir),
            "candidate_dir": os.path.abspath(candidate_dir),
            "candidate_audit_sha256": candidate["audit_sha256"],
            "preprocess_policy_sha256": _sha256_file(str(scene["policy_path"])),
            "dataset_manifest_sha256": _sha256_file(str(scene["manifest_path"])),
            "radar_ir_sync_sha256": _sha256_file(str(scene["sync_path"])),
            "selected_lidar_voxels": voxel_receipts,
        },
        "coordinate_contract": {
            "voxel_coordinate_frame": "lidar",
            "voxel_coords_axis_order": "xyz",
            "voxel_center_formula": "pc_min + (coords + 0.5) * voxel_size",
            "pose_composition": "T_local_lidar = T_local_body @ T_body_lidar",
            "gt_as_lidar_external_cancels": True,
        },
        "pair_selection": {
            "common_frame_count": len(common_frames),
            "eligible_pair_count": eligible_pair_count,
            "selected_pair_count": len(pairs),
            "pair_delta_s": float(pair_delta_s),
            "pair_delta_tolerance_s": float(pair_delta_tolerance_s),
            "min_rotation_deg": float(min_rotation_deg),
            "max_translation_m": float(max_translation_m),
            "min_sensor_range_m": float(min_sensor_range_m),
            "max_sensor_range_m": float(max_sensor_range_m),
            "max_points_per_frame": int(max_points_per_frame),
            "selection": "time_order_evenly_spaced_after_thresholds",
        },
        "hypothesis_summary": summaries,
        "empirical_ranking": {
            "metric": "median_of_pair_symmetric_nn_median_m_lower_is_better",
            "lower_median_residual_first": ranking,
            "preferred_hypothesis_diagnostic_only": ranking[0],
        },
        "identifiability": {
            "can_compare_gt_pose_frame_hypotheses": True,
            "can_confirm_radar_to_imu_direction": False,
            "reason_radar_to_imu_direction": "GT-as-LiDAR 分支中外参代数消去，LiDAR 自重合无法独立验证该方向",
            "can_publish_formal_pose": False,
            "required_authoritative_evidence": [
                "原始 gt_odom.bag 的 header.frame_id 与 child_frame_id，或官方 exporter 定义",
                "VectorNav IMU 到 airborne body 的轴约定/CAD 实测",
            ],
        },
    }
    if candidate["radar_lidar_snapshot"] is not None:
        report["inputs"]["radar_lidar_sync_snapshot_sha256"] = candidate[
            "radar_lidar_snapshot"
        ]["sha256"]

    os.makedirs(normalized_output, exist_ok=True)
    _write_csv_atomic(os.path.join(normalized_output, "pair_metrics.csv"), rows)
    _write_text_atomic(
        os.path.join(normalized_output, "audit.json"),
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _write_text_atomic(
        os.path.join(normalized_output, "README.md"),
        "# Mapping pose overlap diagnostic\n\n"
        "本结果仅用于比较 GT-as-IMU/body 与 GT-as-LiDAR 两种 frame 假设，"
        "始终为 `formal=false`。经验排序不能确认 Radar→IMU 方向，"
        "也不能替代原始 `gt_odom.bag` frame 或 CAD/轴约定证据。\n",
    )
    _write_text_atomic(
        os.path.join(normalized_output, "command.txt"),
        (command_line or "Python API invocation; see audit.json inputs") + "\n",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed_scene_dir", required=True)
    parser.add_argument("--candidate_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pair_delta_s", type=float, default=1.0)
    parser.add_argument("--pair_delta_tolerance_s", type=float, default=0.15)
    parser.add_argument("--min_rotation_deg", type=float, default=3.0)
    parser.add_argument("--max_translation_m", type=float, default=12.0)
    parser.add_argument("--min_sensor_range_m", type=float, default=2.0)
    parser.add_argument("--max_sensor_range_m", type=float, default=50.0)
    parser.add_argument("--max_pairs", type=int, default=48)
    parser.add_argument("--max_points_per_frame", type=int, default=20000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = evaluate_mapping_pose_overlap(
        processed_scene_dir=args.processed_scene_dir,
        candidate_dir=args.candidate_dir,
        output_dir=args.output_dir,
        pair_delta_s=args.pair_delta_s,
        pair_delta_tolerance_s=args.pair_delta_tolerance_s,
        min_rotation_deg=args.min_rotation_deg,
        max_translation_m=args.max_translation_m,
        min_sensor_range_m=args.min_sensor_range_m,
        max_sensor_range_m=args.max_sensor_range_m,
        max_pairs=args.max_pairs,
        max_points_per_frame=args.max_points_per_frame,
        command_line=shlex.join(sys.argv),
    )
    print(
        json.dumps(
            {
                "audit_path": os.path.join(os.path.abspath(args.output_dir), "audit.json"),
                "formal": report["formal"],
                "selected_pair_count": report["pair_selection"]["selected_pair_count"],
                "empirical_preference": report["empirical_ranking"]["preferred_hypothesis_diagnostic_only"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
