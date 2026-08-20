#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Streaming map update entrypoint (v1)."""

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import tempfile
import time
from typing import Dict, List, Tuple

import numpy as np

# 该离线入口只需要两个轻量模块；直接执行时不得加载 `cm/__init__.py`
# 中完整的 Torch/分布式训练栈，否则 `--help` 也可能触发 OpenMPI 初始化。
CM_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cm")
if CM_DIR not in sys.path:
    sys.path.insert(0, CM_DIR)

from probabilistic_mapping import (  # noqa: E402
    GridMapConfig,
    LazyLocalMapQuery,
    SlidingProbabilisticGridMap,
    load_sparse_voxel_npz,
)
from evaluation_metrics import occupancy_prf, voxel_to_points  # noqa: E402


DYNAMIC_EVIDENCE_PROTOCOL = "dynamic_occupancy_evidence_v1"
DYNAMIC_EVIDENCE_METADATA = "dynamic_evidence.json"
DYNAMIC_EVIDENCE_KEYS = {
    "protocol",
    "coordinate_frame",
    "value_semantics",
    "observed_semantics",
    "source",
    "source_artifact_sha256",
    "frame_count",
    "pc_range",
    "shape_xyz",
}


def list_voxel_files(folder: str) -> List[str]:
    files = [
        f for f in os.listdir(folder)
        if not f.endswith("_observed_mask.npy")
        and not f.endswith("_observed_mask.npz")
        and not f.endswith("_dynamic_evidence.npz")
        and (
            f.endswith(".npz")
            or (
                f.endswith(".npy")
                and not f.endswith("_pcl.npy")
                and not f.endswith("_uncertainty.npy")
                and (f.endswith("_voxel.npy") or not f.endswith("_bev.npy"))
            )
        )
    ]
    files.sort()
    return files


def load_voxel(path: str, layout: str = "auto") -> np.ndarray:
    if os.path.islink(path) or not os.path.isfile(path):
        raise ValueError(f"voxel 必须是普通文件: {path}")
    if path.endswith(".npz"):
        arr = load_sparse_voxel_npz(path)
    else:
        arr = np.load(path, allow_pickle=False).astype(np.float32)

    # NOTE: 部分推理输出带批次维度：(N, C, Z, X, Y)。
    # NOTE: 流式冒烟测试仅取批次中的第一个样本。
    # TODO: 支持批量样本并行更新与异步队列，面向Orin级实时部署。
    if arr.ndim == 5:
        if arr.shape[0] != 1:
            raise ValueError(
                f"单帧 streaming 文件必须恰好一个样本，当前 batch={arr.shape[0]}"
            )
        arr = arr[0]

    return to_xyzc(arr, layout=layout)


def to_xyzc(arr: np.ndarray, layout: str = "auto") -> np.ndarray:
    """按显式或无歧义协议把体素转为 `(X,Y,Z,C)`。"""
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D voxel, got shape={arr.shape}")
    layout = str(layout).strip().lower()
    if layout not in {"auto", "xyzc", "czxy"}:
        raise ValueError(f"voxel layout 必须是 auto/xyzc/czxy，当前为 {layout}")
    if layout == "xyzc":
        if not 1 <= arr.shape[-1] <= 8:
            raise ValueError(f"xyzc layout 的通道维异常: shape={arr.shape}")
        return arr.astype(np.float32)
    if layout == "czxy":
        if not 1 <= arr.shape[0] <= 8:
            raise ValueError(f"czxy layout 的通道维异常: shape={arr.shape}")
        return np.transpose(arr, (2, 3, 1, 0)).astype(np.float32)
    xyzc_candidate = arr.shape[-1] <= 8
    czxy_candidate = arr.shape[0] <= 8
    if xyzc_candidate and czxy_candidate:
        raise ValueError(
            f"voxel layout 歧义: shape={arr.shape}，请显式指定 xyzc 或 czxy"
        )
    if xyzc_candidate:
        return arr.astype(np.float32)
    if czxy_candidate:
        return np.transpose(arr, (2, 3, 1, 0)).astype(np.float32)
    raise ValueError(f"Unsupported voxel layout: {arr.shape}")


def load_ir_bev(bev_path: str, target_shape_xy) -> np.ndarray:
    bev = np.load(bev_path).astype(np.float32)
    # TODO: 增加红外质量门控(噪声/模糊/失焦检测)并输出融合置信度。
    if bev.ndim == 3:
        bev = bev[..., 0]
    if bev.shape != target_shape_xy:
        raise ValueError(f"Infrared BEV shape mismatch. expected={target_shape_xy}, got={bev.shape}")
    if bev.max() > 1.0:
        bev = bev / 255.0
    return np.clip(bev, 0.0, 1.0)


def find_uncertainty_file(uncertainty_dir: str, voxel_file_name: str) -> str:
    if not uncertainty_dir:
        return ""
    stem = os.path.splitext(voxel_file_name)[0]
    base = stem[:-6] if stem.endswith("_voxel") else stem
    candidates = [
        os.path.join(uncertainty_dir, f"{base}_uncertainty.npy"),
        os.path.join(uncertainty_dir, f"{stem}_uncertainty.npy"),
        os.path.join(uncertainty_dir, f"{base}_voxel_uncertainty.npy"),
    ]
    for path in candidates:
        if os.path.isfile(path) and not os.path.islink(path):
            return path
    return ""


def find_observed_mask_file(mask_dir: str, voxel_file_name: str) -> str:
    """按帧查找显式 observed mask，不把 mask 当成 voxel 输入。"""
    if not mask_dir:
        return ""
    stem = os.path.splitext(voxel_file_name)[0]
    base = stem[:-6] if stem.endswith("_voxel") else stem
    for extension in (".npy", ".npz"):
        candidate = os.path.join(mask_dir, f"{base}_observed_mask{extension}")
        if os.path.isfile(candidate) and not os.path.islink(candidate):
            return candidate
    return ""


def find_target_voxel_file(target_dir: str, voxel_file_name: str) -> str:
    """按 Radar 帧键查找评价 target，拒绝 symlink 文件。"""
    if not target_dir:
        return ""
    frame = _voxel_frame_key(voxel_file_name)
    for extension in (".npz", ".npy"):
        candidate = os.path.join(target_dir, f"{frame}{extension}")
        if os.path.isfile(candidate) and not os.path.islink(candidate):
            return candidate
    return ""


def load_dynamic_evidence_protocol(
    evidence_dir: str,
    voxel_file_names: List[str],
    cfg: GridMapConfig,
) -> Dict[str, object]:
    """严格加载 body-voxel 动态 evidence 目录协议并预检帧覆盖。"""
    metadata_path = os.path.join(evidence_dir, DYNAMIC_EVIDENCE_METADATA)
    if os.path.islink(metadata_path) or not os.path.isfile(metadata_path):
        raise ValueError(f"动态 evidence metadata 必须是普通文件: {metadata_path}")
    try:
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"动态 evidence metadata 无法解析: {metadata_path}") from exc
    if not isinstance(metadata, dict) or set(metadata) != DYNAMIC_EVIDENCE_KEYS:
        raise ValueError(
            "动态 evidence metadata 字段必须精确为: "
            f"{sorted(DYNAMIC_EVIDENCE_KEYS)}"
        )
    expected_literals = {
        "protocol": DYNAMIC_EVIDENCE_PROTOCOL,
        "coordinate_frame": "body_voxel",
        "value_semantics": "dynamic_probability",
        "observed_semantics": "explicit_boolean_mask",
    }
    for name, expected in expected_literals.items():
        if metadata.get(name) != expected:
            raise ValueError(
                f"动态 evidence {name} 必须为 {expected!r}"
            )
    source = metadata.get("source")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("动态 evidence source 必须是非空字符串")
    source_hash = metadata.get("source_artifact_sha256")
    if (
        not isinstance(source_hash, str)
        or len(source_hash) != 64
        or any(char not in "0123456789abcdef" for char in source_hash)
    ):
        raise ValueError("动态 evidence source_artifact_sha256 必须是小写 SHA-256")
    frame_count = metadata.get("frame_count")
    if type(frame_count) is not int or frame_count != len(voxel_file_names):
        raise ValueError(
            "动态 evidence frame_count 与 Radar 帧数不一致: "
            f"metadata={frame_count!r}, radar={len(voxel_file_names)}"
        )
    expected_shape = [int(value) for value in cfg.shape_xyz]
    actual_shape = metadata.get("shape_xyz")
    if (
        not isinstance(actual_shape, list)
        or len(actual_shape) != 3
        or any(type(value) is not int or value <= 0 for value in actual_shape)
        or actual_shape != expected_shape
    ):
        raise ValueError(
            "动态 evidence shape_xyz 与地图不一致: "
            f"metadata={actual_shape!r}, map={expected_shape}"
        )
    expected_range = [
        float(cfg.x_min),
        float(cfg.y_min),
        float(cfg.z_min),
        float(cfg.x_max),
        float(cfg.y_max),
        float(cfg.z_max),
    ]
    range_payload = metadata.get("pc_range")
    if (
        not isinstance(range_payload, list)
        or len(range_payload) != 6
        or any(type(value) not in (int, float) for value in range_payload)
    ):
        raise ValueError("动态 evidence pc_range 必须是 6 个 JSON number")
    actual_range = [float(value) for value in range_payload]
    if (
        not np.all(np.isfinite(actual_range))
        or not np.allclose(actual_range, expected_range, atol=1e-9, rtol=0.0)
    ):
        raise ValueError(
            "动态 evidence pc_range 与地图不一致: "
            f"metadata={actual_range}, map={expected_range}"
        )

    paths: Dict[str, str] = {}
    expected_entries = {DYNAMIC_EVIDENCE_METADATA}
    for voxel_file_name in voxel_file_names:
        frame = _voxel_frame_key(voxel_file_name)
        evidence_name = f"{frame}_dynamic_evidence.npz"
        evidence_path = os.path.join(evidence_dir, evidence_name)
        if os.path.islink(evidence_path) or not os.path.isfile(evidence_path):
            raise ValueError(f"动态 evidence 帧覆盖不完整，缺少: {frame}")
        paths[voxel_file_name] = evidence_path
        expected_entries.add(evidence_name)
    actual_entries = set(os.listdir(evidence_dir))
    if actual_entries != expected_entries:
        extras = sorted(actual_entries - expected_entries)
        missing = sorted(expected_entries - actual_entries)
        raise ValueError(
            f"动态 evidence 目录内容不匹配: extras={extras[:5]}, missing={missing[:5]}"
        )
    file_sha256: Dict[str, str] = {}
    for voxel_file_name, evidence_path in paths.items():
        _probability, _observed, digest = load_dynamic_evidence(
            evidence_path,
            voxel_shape=cfg.shape_xyz,
        )
        file_sha256[voxel_file_name] = digest
    return {
        "metadata": metadata,
        "metadata_path": metadata_path,
        "metadata_sha256": _sha256_file(metadata_path),
        "paths": paths,
        "file_sha256": file_sha256,
    }


def load_dynamic_evidence(
    path: str,
    voxel_shape,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """读取单帧动态概率/观测域，并返回实际文件 SHA-256。"""
    if os.path.islink(path) or not os.path.isfile(path):
        raise ValueError(f"动态 evidence 必须是普通 NPZ 文件: {path}")
    with open(path, "rb") as handle:
        payload = handle.read()
    digest = hashlib.sha256(payload).hexdigest()
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as data:
            if set(data.files) != {"probability", "observed"}:
                raise ValueError("动态 evidence NPZ 必须精确包含 probability/observed")
            probability = np.asarray(data["probability"], dtype=np.float32)
            observed = np.asarray(data["observed"], dtype=np.float32)
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and "必须精确包含" in str(exc):
            raise
        raise ValueError(f"动态 evidence NPZ 无法解析: {path}") from exc
    expected_shape = tuple(int(value) for value in voxel_shape)
    if probability.shape != expected_shape or observed.shape != expected_shape:
        raise ValueError(
            "动态 evidence shape 不匹配: "
            f"probability={probability.shape}, observed={observed.shape}, "
            f"voxel={expected_shape}"
        )
    if not np.all(np.isfinite(probability)) or not np.all(np.isfinite(observed)):
        raise ValueError("动态 evidence 必须全部为有限数")
    if np.any(probability < 0.0) or np.any(probability > 1.0):
        raise ValueError("动态 evidence probability 必须位于 [0,1]")
    if not np.all((observed == 0.0) | (observed == 1.0)):
        raise ValueError("动态 evidence observed 必须是严格 0/1")
    if np.any(probability[observed == 0.0] != 0.0):
        raise ValueError("动态 evidence 未观测位置的 probability 必须为 0")
    return probability, observed, digest


def load_model_uncertainty(path: str) -> np.ndarray:
    if not path:
        return None
    arr = np.load(path).astype(np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 4 and arr.shape[0] <= 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
        arr = np.transpose(arr, (1, 2, 0))
    return arr


def load_observed_mask(
    path: str,
    voxel_shape,
    preserve_height: bool = False,
) -> np.ndarray:
    """读取 observed mask；正式分层地图可保留 `(X,Y,Z)` 高度信息。"""
    if not path:
        raise ValueError("observed mask path 不能为空")
    if path.endswith(".npz"):
        arr = load_sparse_voxel_npz(path)
    else:
        arr = np.load(path).astype(np.float32)
    arr = np.asarray(arr, dtype=np.float32)
    voxel_shape = tuple(int(value) for value in voxel_shape)
    if len(voxel_shape) != 3:
        raise ValueError(f"voxel_shape 必须是 (X,Y,Z)，当前为 {voxel_shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"observed mask 必须全部为有限数: {path}")
    if arr.shape == voxel_shape:
        if preserve_height:
            arr = (arr > 0.5).astype(np.float32)
        else:
            arr = np.any(arr > 0.5, axis=2).astype(np.float32)
    elif arr.shape == voxel_shape[:2]:
        arr = (arr > 0.5).astype(np.float32)
    else:
        raise ValueError(
            f"observed mask shape {arr.shape} 不匹配 voxel shape {voxel_shape}"
        )
    return arr


def _voxel_frame_key(voxel_file_name: str) -> str:
    """把 `<frame>[_voxel].npy/.npz` 统一为 pose CSV 的 frame 键。"""
    stem = os.path.splitext(os.path.basename(voxel_file_name))[0]
    return stem[:-6] if stem.endswith("_voxel") else stem


def _pose_from_quaternion(row: Dict[str, str], row_number: int) -> np.ndarray:
    """解析 body→local 平移和 ROS 顺序 `(qx,qy,qz,qw)` 单位四元数。"""
    names = ("tx", "ty", "tz", "qx", "qy", "qz", "qw")
    try:
        values = {name: float(row[name]) for name in names}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"pose CSV 第 {row_number} 行位姿字段格式错误") from exc
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError(f"pose CSV 第 {row_number} 行位姿字段必须为有限数")
    qx, qy, qz, qw = (values[name] for name in ("qx", "qy", "qz", "qw"))
    norm = float(np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw))
    if not np.isclose(norm, 1.0, atol=1e-4):
        raise ValueError(
            f"pose CSV 第 {row_number} 行四元数必须归一化，当前 norm={norm:.9f}"
        )
    qx, qy, qz, qw = (value / norm for value in (qx, qy, qz, qw))
    rotation = np.asarray(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float32,
    )
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation
    transform[:3, 3] = [values["tx"], values["ty"], values["tz"]]
    return transform


def load_pose_table(
    pose_path: str,
    voxel_file_names: List[str],
) -> Dict[str, Dict[str, object]]:
    """严格加载逐帧 body→local pose，返回与 voxel 顺序一致的映射。"""
    if os.path.islink(pose_path) or not os.path.isfile(pose_path):
        raise ValueError(f"pose_file 必须是普通 CSV 文件: {pose_path}")
    expected_keys = [_voxel_frame_key(name) for name in voxel_file_names]
    if len(expected_keys) != len(set(expected_keys)):
        raise ValueError("voxel 文件映射到重复 frame 键")
    required = {"frame", "timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"}
    parsed: Dict[str, Dict[str, object]] = {}
    with open(pose_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                "pose CSV 必须包含 frame,timestamp,tx,ty,tz,qx,qy,qz,qw"
            )
        for row_number, row in enumerate(reader, start=2):
            frame = str(row.get("frame", "")).strip()
            if not frame:
                raise ValueError(f"pose CSV 第 {row_number} 行 frame 不能为空")
            if frame in parsed:
                raise ValueError(f"pose CSV 存在重复 frame: {frame}")
            try:
                timestamp = float(row["timestamp"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"pose CSV 第 {row_number} 行 timestamp 格式错误") from exc
            if not np.isfinite(timestamp) or timestamp < 0.0:
                raise ValueError(f"pose CSV 第 {row_number} 行 timestamp 必须是有限非负数")
            parsed[frame] = {
                "timestamp": timestamp,
                "T_local_body": _pose_from_quaternion(row, row_number),
            }

    missing = [frame for frame in expected_keys if frame not in parsed]
    if missing:
        raise ValueError(f"pose CSV 帧覆盖不完整，缺少: {missing[:5]}")
    ordered = {frame: parsed[frame] for frame in expected_keys}
    timestamps = [float(record["timestamp"]) for record in ordered.values()]
    if any(after <= before for before, after in zip(timestamps, timestamps[1:])):
        raise ValueError("pose CSV timestamp 必须按 voxel 文件顺序严格递增")
    return ordered


def _sha256_file(path: str) -> str:
    """流式计算输入协议文件哈希，避免把路径名当作 provenance。"""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: str, payload: Dict[str, object]) -> None:
    """在同目录完成 JSON 原子发布。"""
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
        dir=os.path.dirname(path),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


def _snapshot_payload(
    snapshot: Dict[str, np.ndarray],
    pose_mode: str,
) -> Dict[str, np.ndarray]:
    """为每份地图快照附加坐标系和位姿协议标签。"""
    payload = dict(snapshot)
    payload["map_frame"] = np.asarray("local")
    payload["pose_mode"] = np.asarray(pose_mode)
    return payload


def map_occ_to_points(occ_prob: np.ndarray, cfg: GridMapConfig, threshold: float = 0.55) -> np.ndarray:
    """把 BEV 或分高度层 occupancy 转成 local 坐标系体素中心。"""
    occupancy = np.asarray(occ_prob, dtype=np.float32)
    if occupancy.ndim not in (2, 3):
        raise ValueError(f"occupancy 必须是 (X,Y) 或 (X,Y,Z)，当前为 {occupancy.shape}")
    expected_shape = cfg.shape_xy if occupancy.ndim == 2 else cfg.shape_xyz
    if occupancy.shape != expected_shape:
        raise ValueError(
            f"occupancy shape {occupancy.shape} != map shape {expected_shape}"
        )
    idx = np.argwhere(occupancy >= float(threshold))
    if idx.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    x = cfg.x_min + (idx[:, 0].astype(np.float32) + 0.5) * cfg.x_resolution
    y = cfg.y_min + (idx[:, 1].astype(np.float32) + 0.5) * cfg.y_resolution
    if occupancy.ndim == 3:
        z = cfg.z_min + (idx[:, 2].astype(np.float32) + 0.5) * cfg.z_resolution
    else:
        # 旧 BEV 消费者没有高度语义，继续保留历史 z=0 行为。
        z = np.zeros_like(x)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def transform_points(points_body: np.ndarray, T_local_body: np.ndarray) -> np.ndarray:
    """使用 body→local 刚体变换统一逐帧 target 与累计地图坐标系。"""
    points = np.asarray(points_body, dtype=np.float32)
    transform = np.asarray(T_local_body, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_body 必须是 (N,3)，当前为 {points.shape}")
    if transform.shape != (4, 4):
        raise ValueError(f"T_local_body 必须是 (4,4)，当前为 {transform.shape}")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(transform)):
        raise ValueError("points_body 和 T_local_body 必须全部为有限数")
    if points.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return (
        points @ transform[:3, :3].T + transform[:3, 3][np.newaxis, :]
    ).astype(np.float32)


def validate_runtime_args(args: argparse.Namespace) -> None:
    """在任何输出副作用前校验流式入口的数值协议。"""
    finite_values = {
        "dt": args.dt,
        "decay_rate": args.decay_rate,
        "dynamic_decay_rate": args.dynamic_decay_rate,
        "prior_reliability": args.prior_reliability,
        "radar_reliability": args.radar_reliability,
        "infrared_reliability": args.infrared_reliability,
        "speed_m_s": args.speed_m_s,
        "odom_cov_trace": args.odom_cov_trace,
        "calib_confidence": args.calib_confidence,
    }
    for name, value in finite_values.items():
        if not np.isfinite(float(value)):
            raise ValueError(f"{name} 必须是有限数")
    if not args.pose_file and float(args.dt) <= 0.0:
        raise ValueError("identity_legacy 模式的 dt 必须大于 0")
    if int(args.window_size) <= 0 or int(args.frame_limit) < 0:
        raise ValueError("window_size 必须大于 0，frame_limit 必须非负")
    if int(args.save_every) <= 0:
        raise ValueError("save_every 必须大于 0")
    if float(args.decay_rate) < 0.0 or float(args.speed_m_s) <= 0.0:
        raise ValueError("decay_rate 必须非负，speed_m_s 必须大于 0")
    if (
        args.dynamic_evidence_dir
        and float(args.dynamic_decay_rate) <= float(args.decay_rate)
    ):
        raise ValueError("dynamic_decay_rate 必须严格大于静态 decay_rate")
    if float(args.odom_cov_trace) < 0.0:
        raise ValueError("odom_cov_trace 必须非负")
    for name in (
        "prior_reliability",
        "radar_reliability",
        "infrared_reliability",
        "calib_confidence",
    ):
        value = float(getattr(args, name))
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{name} 必须位于 [0,1]")
    pc_range = np.asarray(args.pc_range, dtype=np.float64)
    if pc_range.shape != (6,) or not np.all(np.isfinite(pc_range)):
        raise ValueError("pc_range 必须包含 6 个有限数")
    if not np.all(pc_range[3:] > pc_range[:3]):
        raise ValueError("pc_range 的 max 必须逐轴大于 min")


def build_config(args, first_voxel_xyzc: np.ndarray) -> GridMapConfig:
    nx, ny, nz = first_voxel_xyzc.shape[:3]
    x_min, y_min, z_min, x_max, y_max, z_max = args.pc_range
    x_res = (x_max - x_min) / max(nx, 1)
    y_res = (y_max - y_min) / max(ny, 1)
    z_res = (z_max - z_min) / max(nz, 1)
    return GridMapConfig(
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
        x_resolution=float(x_res),
        y_resolution=float(y_res),
        z_min=z_min,
        z_max=z_max,
        z_resolution=float(z_res),
        window_size=args.window_size,
        decay_rate=args.decay_rate,
        dynamic_decay_rate=args.dynamic_decay_rate,
        prior_reliability=args.prior_reliability,
        radar_reliability=args.radar_reliability,
        infrared_reliability=args.infrared_reliability,
        speed_m_s=args.speed_m_s,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming probabilistic map update")
    parser.add_argument("--radar_voxel_dir", type=str, required=True)
    parser.add_argument(
        "--radar_voxel_layout",
        choices=("auto", "xyzc", "czxy"),
        default="auto",
        help="Radar/prediction voxel layout; auto only accepts unambiguous shapes",
    )
    parser.add_argument(
        "--pose_file",
        type=str,
        default="",
        help=(
            "Optional strict body-to-local CSV: "
            "frame,timestamp,tx,ty,tz,qx,qy,qz,qw"
        ),
    )
    parser.add_argument("--uncertainty_dir", type=str, default="",
                        help="Directory containing *_uncertainty.npy files from multimodal inference")
    parser.add_argument(
        "--observed_mask_dir",
        type=str,
        default="",
        help="Optional per-frame <frame>_observed_mask.npy/.npz; absent cells remain unknown",
    )
    parser.add_argument(
        "--dynamic_evidence_dir",
        type=str,
        default="",
        help=(
            "Optional strict dynamic evidence directory containing "
            "dynamic_evidence.json and per-frame *_dynamic_evidence.npz"
        ),
    )
    parser.add_argument("--infrared_bev_dir", type=str, default="")
    parser.add_argument("--prior_dem", type=str, default="")
    parser.add_argument("--target_voxel_dir", type=str, default="")
    parser.add_argument(
        "--target_voxel_layout",
        choices=("auto", "xyzc", "czxy"),
        default="auto",
        help="Optional target voxel layout",
    )
    parser.add_argument("--output_dir", type=str, default="./streaming_results")
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--window_size", type=int, default=12)
    parser.add_argument("--decay_rate", type=float, default=0.12)
    parser.add_argument("--dynamic_decay_rate", type=float, default=0.60)
    parser.add_argument("--prior_reliability", type=float, default=0.90)
    parser.add_argument("--radar_reliability", type=float, default=0.75)
    parser.add_argument("--infrared_reliability", type=float, default=0.65)
    parser.add_argument("--speed_m_s", type=float, default=50.0)
    parser.add_argument("--odom_cov_trace", type=float, default=0.0)
    parser.add_argument("--calib_confidence", type=float, default=1.0)
    parser.add_argument("--frame_limit", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument(
        "--pc_range",
        type=float,
        nargs=6,
        default=[0, -20, -6, 120, 20, 10],
    )
    args = parser.parse_args()
    validate_runtime_args(args)

    # 在创建输出目录前校验全部输入，避免协议错误留下不可重跑的半成品。
    if os.path.islink(args.radar_voxel_dir) or not os.path.isdir(args.radar_voxel_dir):
        raise ValueError(f"radar_voxel_dir 必须是普通目录: {args.radar_voxel_dir}")
    for argument_name in (
        "observed_mask_dir",
        "dynamic_evidence_dir",
        "uncertainty_dir",
        "infrared_bev_dir",
        "target_voxel_dir",
    ):
        folder = getattr(args, argument_name)
        if folder and (os.path.islink(folder) or not os.path.isdir(folder)):
            raise ValueError(f"{argument_name} 必须是普通目录: {folder}")
    if args.prior_dem and (
        os.path.islink(args.prior_dem) or not os.path.isfile(args.prior_dem)
    ):
        raise ValueError(f"prior_dem 必须是普通文件: {args.prior_dem}")

    all_radar_files = list_voxel_files(args.radar_voxel_dir)
    if not all_radar_files:
        raise RuntimeError(f"No voxel files found under {args.radar_voxel_dir}")
    radar_frame_keys = [_voxel_frame_key(file_name) for file_name in all_radar_files]
    seen_frame_keys = set()
    duplicate_frame_keys = set()
    for frame_key in radar_frame_keys:
        if frame_key in seen_frame_keys:
            duplicate_frame_keys.add(frame_key)
        seen_frame_keys.add(frame_key)
    if duplicate_frame_keys:
        raise ValueError(
            f"Radar voxel 映射到重复 frame 键: "
            f"{sorted(duplicate_frame_keys)[:5]}"
        )
    radar_files = all_radar_files
    if args.frame_limit > 0:
        radar_files = radar_files[: args.frame_limit]
    pose_table = load_pose_table(args.pose_file, radar_files) if args.pose_file else {}
    pose_mode = "body_to_local_csv" if pose_table else "identity_legacy"
    target_paths: Dict[str, str] = {}
    if args.target_voxel_dir:
        target_paths = {
            file_name: find_target_voxel_file(args.target_voxel_dir, file_name)
            for file_name in radar_files
        }
        missing_targets = [
            _voxel_frame_key(file_name)
            for file_name, path in target_paths.items()
            if not path
        ]
        if missing_targets:
            raise ValueError(
                f"target voxel 帧覆盖不完整，缺少: {missing_targets[:5]}"
            )
    first_voxel = load_voxel(
        os.path.join(args.radar_voxel_dir, radar_files[0]),
        layout=args.radar_voxel_layout,
    )
    cfg = build_config(args, first_voxel)
    grid_map = SlidingProbabilisticGridMap(cfg)
    query = LazyLocalMapQuery(cfg)
    dynamic_protocol = (
        load_dynamic_evidence_protocol(
            args.dynamic_evidence_dir,
            all_radar_files,
            cfg,
        )
        if args.dynamic_evidence_dir
        else None
    )
    dynamic_hash_records: List[Tuple[str, str]] = []
    odom_cov = None
    if args.odom_cov_trace > 0.0:
        odom_cov = np.eye(3, dtype=np.float32) * (float(args.odom_cov_trace) / 3.0)

    prior_dem = None
    if args.prior_dem:
        # TODO: 支持先验DEM多来源输入及置信度地图，而非单一栅格文件。
        loaded_prior = np.load(args.prior_dem, allow_pickle=False)
        if not isinstance(loaded_prior, np.ndarray):
            raise ValueError("prior_dem 必须是单一 NumPy 数组文件")
        prior_dem = loaded_prior.astype(np.float32)
        if prior_dem.shape != grid_map.dem_mean.shape:
            raise ValueError(
                f"prior DEM shape {prior_dem.shape} != map shape "
                f"{grid_map.dem_mean.shape}"
            )
        if not np.all(np.isfinite(prior_dem) | np.isnan(prior_dem)):
            raise ValueError("prior DEM 只能包含有限值或 NaN")

    os.makedirs(args.output_dir, exist_ok=True)

    metric_path = os.path.join(args.output_dir, "streaming_metrics.csv")
    with open(metric_path, "w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow([
            "frame",
            "timestamp",
            "update_ms",
            "nearest_dist",
            "nearest_uncertainty",
            "is_risky",
            "speed_m_s",
            "odom_cov_trace",
            "obstacle_precision",
            "obstacle_recall",
            "false_positive_rate",
            "mean_uncertainty",
            "observed_mask_used",
            "observed_mask_cells",
            "unknown_fraction",
            "pose_used",
            "body_local_x",
            "body_local_y",
            "body_local_z",
            "dynamic_evidence_used",
            "dynamic_observed_cells",
            "dynamic_probability_mean_observed",
            "dynamic_unknown_fraction",
        ])

        for i, file_name in enumerate(radar_files):
            frame_start = time.perf_counter()
            frame_key = _voxel_frame_key(file_name)
            if pose_table:
                pose_record = pose_table[frame_key]
                timestamp = float(pose_record["timestamp"])
                T_local_body = np.asarray(
                    pose_record["T_local_body"],
                    dtype=np.float32,
                )
            else:
                timestamp = i * args.dt
                T_local_body = np.eye(4, dtype=np.float32)

            voxel = load_voxel(
                os.path.join(args.radar_voxel_dir, file_name),
                layout=args.radar_voxel_layout,
            )
            unc_path = find_uncertainty_file(args.uncertainty_dir, file_name)
            model_uncertainty = load_model_uncertainty(unc_path) if unc_path else None
            observed_mask_path = find_observed_mask_file(
                args.observed_mask_dir,
                file_name,
            )
            observed_mask = (
                load_observed_mask(
                    observed_mask_path,
                    voxel_shape=voxel.shape[:3],
                    preserve_height=True,
                )
                if observed_mask_path
                else None
            )
            dynamic_probability = None
            dynamic_observed = None
            if dynamic_protocol is not None:
                dynamic_probability, dynamic_observed, evidence_hash = (
                    load_dynamic_evidence(
                        dynamic_protocol["paths"][file_name],
                        voxel_shape=voxel.shape[:3],
                    )
                )
                if evidence_hash != dynamic_protocol["file_sha256"][file_name]:
                    raise ValueError(
                        f"动态 evidence 在预检后发生变化: {frame_key}"
                    )
                dynamic_hash_records.append((frame_key, evidence_hash))
            grid_map.update_from_voxel(
                voxel_xyzc=voxel,
                timestamp=timestamp,
                sensor="radar",
                odom_cov=odom_cov,
                model_uncertainty=model_uncertainty,
                calib_confidence=args.calib_confidence,
                observed_mask=observed_mask,
                T_local_body=T_local_body,
                dynamic_probability=dynamic_probability,
                dynamic_observed_mask=dynamic_observed,
            )

            if args.infrared_bev_dir:
                ir_path = os.path.join(args.infrared_bev_dir, file_name.replace("_voxel", "_bev"))
                if os.path.exists(ir_path):
                    bev = load_ir_bev(ir_path, target_shape_xy=grid_map.occ_prob.shape)
                    grid_map.update_from_ir_bev(
                        bev_xy=bev,
                        timestamp=timestamp,
                        T_local_body=T_local_body,
                    )

            if prior_dem is not None:
                grid_map.fuse_with_prior_dem(prior_dem=prior_dem, prior_confidence=0.6)

            snapshot = grid_map.snapshot()
            query.refresh(snapshot)
            # 查询点是当前机体原点在 local 系中的位置，不再使用历史固定测试坐标。
            prox = query.query_proximity(
                x_m=float(T_local_body[0, 3]),
                y_m=float(T_local_body[1, 3]),
                z_m=float(T_local_body[2, 3]),
                search_radius=30.0,
            )
            obstacle_precision = ""
            obstacle_recall = ""
            false_positive_rate = ""
            if target_paths:
                target_points_body = voxel_to_points(
                    load_voxel(
                        target_paths[file_name],
                        layout=args.target_voxel_layout,
                    ),
                    pc_range=args.pc_range,
                    occ_threshold=0.1,
                )
                target_points = transform_points(
                    target_points_body,
                    T_local_body,
                )
                map_points = map_occ_to_points(
                    snapshot["occ_prob_layers"],
                    cfg,
                    threshold=0.55,
                )
                prf = occupancy_prf(
                    map_points,
                    target_points,
                    pc_range=args.pc_range,
                    cell_size=max(cfg.x_resolution, cfg.y_resolution),
                )
                obstacle_precision = f"{prf['precision']:.6f}"
                obstacle_recall = f"{prf['recall']:.6f}"
                denom = prf["fp"] + prf["tp"]
                false_positive_rate = (
                    f"{(prf['fp'] / denom if denom else 0.0):.6f}"
                )

            frame_ms = (time.perf_counter() - frame_start) * 1000.0
            # TODO: 增加端到端时延分解统计(读取/融合/查询/写盘)与资源监控(CPU/GPU/内存)。
            writer.writerow([
                i,
                f"{timestamp:.3f}",
                f"{frame_ms:.3f}",
                f"{prox['distance']:.3f}",
                f"{prox['uncertainty']:.3f}",
                int(prox["is_risky"] > 0.5),
                f"{args.speed_m_s:.3f}",
                f"{args.odom_cov_trace:.6f}",
                obstacle_precision,
                obstacle_recall,
                false_positive_rate,
                f"{float(np.mean(1.0 - snapshot['belief'])):.6f}",
                int(bool(observed_mask_path)),
                int(np.count_nonzero(observed_mask)) if observed_mask is not None else "",
                f"{float(np.mean(snapshot['unknown_mass'])):.6f}",
                int(bool(pose_table)),
                f"{float(T_local_body[0, 3]):.6f}",
                f"{float(T_local_body[1, 3]):.6f}",
                f"{float(T_local_body[2, 3]):.6f}",
                int(dynamic_probability is not None),
                (
                    int(np.count_nonzero(dynamic_observed))
                    if dynamic_observed is not None
                    else ""
                ),
                (
                    f"{float(np.mean(dynamic_probability[dynamic_observed > 0])):.6f}"
                    if dynamic_observed is not None
                    and np.any(dynamic_observed > 0)
                    else ""
                ),
                (
                    f"{float(np.mean(snapshot['dynamic_unknown_mass_layers'])):.6f}"
                    if "dynamic_unknown_mass_layers" in snapshot
                    else ""
                ),
            ])

            if i % max(1, args.save_every) == 0:
                np.savez_compressed(
                    os.path.join(args.output_dir, f"map_snapshot_{i:06d}.npz"),
                    **_snapshot_payload(snapshot, pose_mode),
                )
                print(f"[frame {i}] update={frame_ms:.2f}ms dist={prox['distance']:.2f}m unc={prox['uncertainty']:.2f}")

    final_snapshot = grid_map.snapshot()
    np.savez_compressed(
        os.path.join(args.output_dir, "map_final.npz"),
        **_snapshot_payload(final_snapshot, pose_mode),
    )
    dynamic_files_sha256 = None
    if dynamic_hash_records:
        digest = hashlib.sha256()
        for frame_key, file_sha256 in dynamic_hash_records:
            digest.update(frame_key.encode("utf-8"))
            digest.update(b"\0")
            digest.update(file_sha256.encode("ascii"))
            digest.update(b"\n")
        dynamic_files_sha256 = digest.hexdigest()
    dynamic_metadata = (
        dynamic_protocol["metadata"] if dynamic_protocol is not None else None
    )
    _write_json_atomic(
        os.path.join(args.output_dir, "map_run.json"),
        {
            "protocol": "pose_aware_layered_map_v2",
            "map_frame": "local",
            "pose_mode": pose_mode,
            "pose_direction": "body_to_local",
            "pose_file": os.path.basename(args.pose_file) if args.pose_file else None,
            "pose_file_sha256": _sha256_file(args.pose_file) if args.pose_file else None,
            "frame_count": len(radar_files),
            "radar_voxel_layout": args.radar_voxel_layout,
            "target_voxel_layout": (
                args.target_voxel_layout if target_paths else None
            ),
            "pc_range": [float(value) for value in args.pc_range],
            "map_shape_xyz": [int(value) for value in cfg.shape_xyz],
            "occupancy_probability": "pignistic_m_occ_plus_half_unknown",
            "map_occupancy_threshold": 0.55,
            "target_occupancy_threshold": 0.1 if target_paths else None,
            "metric_frame": "local",
            "obstacle_metric_space": "bev_xy",
            "proximity_query": "body_origin_local_3d",
            "dynamic_evidence_enabled": dynamic_protocol is not None,
            "decay_rate_base": float(args.decay_rate),
            "decay_rate_effective": float(cfg.decay_rate),
            "dynamic_decay_rate": float(cfg.dynamic_decay_rate),
            "dynamic_decay_rate_base": float(args.dynamic_decay_rate),
            "dynamic_decay_rate_effective": float(cfg.dynamic_decay_rate),
            "dynamic_evidence_reliability": (
                "explicit_observed_times_odometry_confidence"
                if dynamic_protocol is not None
                else None
            ),
            "combined_static_dynamic_semantics": (
                "dynamic_occupied_pignistic_overlay"
                if dynamic_protocol is not None
                else None
            ),
            "dynamic_evidence_metadata_filename": (
                DYNAMIC_EVIDENCE_METADATA if dynamic_protocol is not None else None
            ),
            "dynamic_evidence_metadata_sha256": (
                dynamic_protocol["metadata_sha256"]
                if dynamic_protocol is not None
                else None
            ),
            "dynamic_evidence_source": (
                dynamic_metadata["source"] if dynamic_metadata is not None else None
            ),
            "dynamic_evidence_source_artifact_sha256": (
                dynamic_metadata["source_artifact_sha256"]
                if dynamic_metadata is not None
                else None
            ),
            "dynamic_evidence_source_artifact_hash_status": (
                "declared_by_metadata_unresolved"
                if dynamic_metadata is not None
                else None
            ),
            "dynamic_evidence_consumed_frame_count": len(dynamic_hash_records),
            "dynamic_evidence_files_sha256": dynamic_files_sha256,
            "legacy_bev_keys": ["occ_prob", "belief", "plausibility", "unknown_mass"],
            "layer_keys": [
                "occ_prob_layers",
                "belief_layers",
                "plausibility_layers",
                "unknown_mass_layers",
            ],
            "static_layer_keys": (
                [
                    "static_occ_prob_layers",
                    "static_belief_layers",
                    "static_plausibility_layers",
                    "static_unknown_mass_layers",
                ]
                if dynamic_protocol is not None
                else []
            ),
            "dynamic_layer_keys": (
                [
                    "dynamic_occ_prob_layers",
                    "dynamic_belief_layers",
                    "dynamic_plausibility_layers",
                    "dynamic_unknown_mass_layers",
                ]
                if dynamic_protocol is not None
                else []
            ),
        },
    )
    print(f"Saved final map to: {os.path.join(args.output_dir, 'map_final.npz')}")
    print(f"Saved metrics to: {metric_path}")
    # TODO: 新增ROS1发布模式，将map_final/streaming指标同步为service/action可消费接口。


if __name__ == "__main__":
    main()
