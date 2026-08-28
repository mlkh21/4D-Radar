# -*- coding: utf-8 -*-
"""文件功能：构建并严格校验仅限离线地图的经验 LiDAR→local 位姿合同。"""

import csv
import hashlib
import io
import json
import os
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


EMPIRICAL_POSE_PROTOCOL = "empirical_lidar_pose_contract_v1"
EMPIRICAL_POSE_RECEIPT = "empirical_pose_receipt.json"
DIRECT_POSE_FILE = "lidar_to_local.csv"
SOURCE_CANDIDATE_AUDIT_FILE = "source_candidate_audit.json"
SOURCE_OVERLAP_AUDIT_FILE = "source_overlap_audit.json"
SOURCE_CANDIDATE_POSE_FILE = (
    "source_candidate_body_to_local_gt_as_lidar.diagnostic.csv"
)
SOURCE_CANDIDATE_EXTRINSIC_FILE = (
    "source_candidate_lidar_to_imu_body.diagnostic.txt"
)
SYNC_SNAPSHOT_FILE = "radar_lidar_sync.snapshot.csv"
README_FILE = "README.md"
COMMAND_FILE = "command.txt"

MEMBER_FILES = {
    "lidar_to_local_pose": DIRECT_POSE_FILE,
    "source_candidate_audit": SOURCE_CANDIDATE_AUDIT_FILE,
    "source_overlap_audit": SOURCE_OVERLAP_AUDIT_FILE,
    "source_candidate_pose": SOURCE_CANDIDATE_POSE_FILE,
    "source_candidate_lidar_to_body": SOURCE_CANDIDATE_EXTRINSIC_FILE,
    "radar_lidar_sync_snapshot": SYNC_SNAPSHOT_FILE,
}


def _sha256_file(path: str) -> str:
    """流式计算普通文件 SHA-256。"""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: str, label: str) -> str:
    normalized = os.path.abspath(os.fspath(path))
    if os.path.islink(normalized) or not os.path.isfile(normalized):
        raise ValueError(f"{label} 必须是普通文件: {normalized}")
    return normalized


def _regular_directory(path: str, label: str) -> str:
    normalized = os.path.abspath(os.fspath(path))
    if os.path.islink(normalized) or not os.path.isdir(normalized):
        raise ValueError(f"{label} 必须是普通目录: {normalized}")
    return normalized


def _member_path(root: str, file_name: object, label: str) -> str:
    """只接受当前合同目录内的 basename 成员，拒绝路径穿越和 symlink。"""
    name = str(file_name or "")
    if not name or os.path.basename(name) != name:
        raise ValueError(f"{label} file 必须是 basename")
    return _regular_file(os.path.join(root, name), label)


def _load_json(path: str, label: str) -> Dict[str, object]:
    path = _regular_file(path, label)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} 无法解析: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} 顶层必须是 JSON 对象")
    return payload


def _valid_sha256(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _require_sha256(path: str, expected: object, label: str) -> str:
    if not _valid_sha256(expected):
        raise ValueError(f"{label} SHA-256 声明无效")
    actual = _sha256_file(path)
    if actual != str(expected):
        raise ValueError(f"{label} SHA-256 不匹配")
    return actual


def _validated_transform(value: object, label: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"{label} 必须是有限 4x4 矩阵")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6, rtol=0.0):
        raise ValueError(f"{label} 齐次矩阵末行无效")
    rotation = transform[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=5e-3, rtol=0.0):
        raise ValueError(f"{label} 旋转矩阵不正交")
    determinant = float(np.linalg.det(rotation))
    if abs(determinant - 1.0) > 5e-3:
        raise ValueError(f"{label} 旋转 determinant 必须接近 1")
    return transform


def _load_diagnostic_extrinsic(path: str) -> np.ndarray:
    """解析 audit 已绑定的诊断 R/T；此函数不会把它提升为 formal 外参。"""
    values: Dict[str, List[float]] = {}
    with open(_regular_file(path, "诊断 LiDAR→body 候选"), "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or ":" not in stripped:
                continue
            key, raw = stripped.split(":", 1)
            try:
                values[key.strip()] = [float(item) for item in raw.split()]
            except ValueError as exc:
                raise ValueError("诊断 LiDAR→body 候选含非数值 R/T") from exc
    rotation = values.get("R")
    translation = values.get("T")
    if rotation is None or len(rotation) != 9 or translation is None or len(translation) != 3:
        raise ValueError("诊断 LiDAR→body 候选必须包含 3x3 R 和 3 元 T")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return _validated_transform(transform, "诊断 LiDAR→body 候选")


def _quaternion_to_transform(row: Dict[str, str], row_number: int) -> np.ndarray:
    names = ("tx", "ty", "tz", "qx", "qy", "qz", "qw")
    try:
        values = {name: float(row[name]) for name in names}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"pose CSV 第 {row_number} 行位姿字段格式错误") from exc
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError(f"pose CSV 第 {row_number} 行位姿字段必须为有限数")
    qx, qy, qz, qw = (values[name] for name in ("qx", "qy", "qz", "qw"))
    norm = float(np.linalg.norm([qx, qy, qz, qw]))
    if not np.isclose(norm, 1.0, atol=1e-4):
        raise ValueError(f"pose CSV 第 {row_number} 行四元数必须归一化")
    qx, qy, qz, qw = (value / norm for value in (qx, qy, qz, qw))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = [
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ]
    transform[:3, 3] = [values["tx"], values["ty"], values["tz"]]
    return _validated_transform(transform, f"pose CSV 第 {row_number} 行")


def _transform_to_quaternion(transform: np.ndarray) -> Tuple[float, float, float, float]:
    """把正交旋转矩阵稳定转换为 ROS 顺序 `(qx,qy,qz,qw)`。"""
    rotation = _validated_transform(transform, "待写出 LiDAR pose")[:3, :3]
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (rotation[2, 1] - rotation[1, 2]) / scale
        qy = (rotation[0, 2] - rotation[2, 0]) / scale
        qz = (rotation[1, 0] - rotation[0, 1]) / scale
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        qw = (rotation[2, 1] - rotation[1, 2]) / scale
        qx = 0.25 * scale
        qy = (rotation[0, 1] + rotation[1, 0]) / scale
        qz = (rotation[0, 2] + rotation[2, 0]) / scale
    elif rotation[1, 1] > rotation[2, 2]:
        scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        qw = (rotation[0, 2] - rotation[2, 0]) / scale
        qx = (rotation[0, 1] + rotation[1, 0]) / scale
        qy = 0.25 * scale
        qz = (rotation[1, 2] + rotation[2, 1]) / scale
    else:
        scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        qw = (rotation[1, 0] - rotation[0, 1]) / scale
        qx = (rotation[0, 2] + rotation[2, 0]) / scale
        qy = (rotation[1, 2] + rotation[2, 1]) / scale
        qz = 0.25 * scale
    quaternion = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    # 固定符号以保证同一输入生成稳定字节；q 与 -q 表示同一旋转。
    if quaternion[3] < 0.0:
        quaternion *= -1.0
    return tuple(float(value) for value in quaternion)


def _load_pose_rows(
    path: str,
    *,
    require_diagnostic: bool,
) -> List[Dict[str, object]]:
    required = {"frame", "timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"}
    rows: List[Dict[str, object]] = []
    with open(_regular_file(path, "pose CSV"), "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        if not required.issubset(fields):
            raise ValueError("pose CSV 缺少 frame,timestamp,tx,ty,tz,qx,qy,qz,qw")
        if require_diagnostic and "diagnostic_formal" not in fields:
            raise ValueError("来源 pose CSV 必须显式声明 diagnostic_formal=false")
        if not require_diagnostic and fields != required:
            raise ValueError("经验 direct pose CSV 只允许标准九列")
        seen = set()
        for row_number, row in enumerate(reader, start=2):
            frame = str(row.get("frame", "")).strip()
            if not frame or frame in seen:
                raise ValueError(f"pose CSV frame 为空或重复: {frame}")
            seen.add(frame)
            if require_diagnostic and str(row.get("diagnostic_formal", "")).lower() != "false":
                raise ValueError("来源 pose CSV 必须逐行声明 diagnostic_formal=false")
            try:
                timestamp = float(row["timestamp"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"pose CSV 第 {row_number} 行 timestamp 格式错误") from exc
            if not np.isfinite(timestamp) or timestamp < 0.0:
                raise ValueError(f"pose CSV 第 {row_number} 行 timestamp 必须有限非负")
            rows.append(
                {
                    "frame": frame,
                    "timestamp": timestamp,
                    "transform": _quaternion_to_transform(row, row_number),
                }
            )
    if not rows:
        raise ValueError("pose CSV 不得为空")
    timestamps = [float(row["timestamp"]) for row in rows]
    if any(after <= before for before, after in zip(timestamps, timestamps[1:])):
        raise ValueError("pose CSV timestamp 必须严格递增")
    return rows


def _direct_pose_csv(rows: Sequence[Dict[str, object]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["frame", "timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])
    for row in rows:
        transform = np.asarray(row["transform"], dtype=np.float64)
        qx, qy, qz, qw = _transform_to_quaternion(transform)
        writer.writerow(
            [
                row["frame"],
                f"{float(row['timestamp']):.9f}",
                f"{transform[0, 3]:.12f}",
                f"{transform[1, 3]:.12f}",
                f"{transform[2, 3]:.12f}",
                f"{qx:.12f}",
                f"{qy:.12f}",
                f"{qz:.12f}",
                f"{qw:.12f}",
            ]
        )
    return buffer.getvalue()


def _write_bytes_atomic(path: str, content: bytes) -> None:
    folder = os.path.dirname(os.path.abspath(path))
    with tempfile.NamedTemporaryFile(dir=folder, prefix=".tmp_", delete=False) as handle:
        temporary = handle.name
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_bytes(path: str) -> bytes:
    """读取已通过普通文件检查的合同来源。"""
    with open(path, "rb") as handle:
        return handle.read()


def _write_text_atomic(path: str, content: str) -> None:
    _write_bytes_atomic(path, content.encode("utf-8"))


def _write_json_atomic(path: str, payload: Dict[str, object]) -> None:
    _write_text_atomic(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _validate_candidate_sources(candidate_dir: str) -> Dict[str, object]:
    candidate_dir = _regular_directory(candidate_dir, "candidate_dir")
    audit_path = _regular_file(os.path.join(candidate_dir, "audit.json"), "candidate audit")
    audit = _load_json(audit_path, "candidate audit")
    coverage = audit.get("coverage")
    timing = audit.get("timing")
    pose_candidates = audit.get("pose_candidates")
    lidar_candidate = pose_candidates.get("gt_as_lidar") if isinstance(pose_candidates, dict) else None
    extrinsic = audit.get("candidate_lidar_to_body")
    inputs = audit.get("inputs")
    sync = inputs.get("radar_lidar_sync_snapshot") if isinstance(inputs, dict) else None
    if (
        audit.get("protocol") != "mapping_pose_candidate_diagnostic_v2"
        or audit.get("formal") is not False
        or audit.get("candidate_only") is not True
        or audit.get("assumptions_resolved") is not False
        or not isinstance(coverage, dict)
        or coverage.get("no_extrapolation") is not True
        or type(coverage.get("radar_frame_count")) is not int
        or type(coverage.get("covered_frame_count")) is not int
        or type(coverage.get("uncovered_frame_count")) is not int
        or coverage.get("covered_frame_count", 0) <= 0
        or coverage.get("radar_frame_count")
        != coverage.get("covered_frame_count") + coverage.get("uncovered_frame_count")
        or not isinstance(coverage.get("uncovered_frame_ids"), list)
        or len(coverage.get("uncovered_frame_ids")) != coverage.get("uncovered_frame_count")
        or not isinstance(timing, dict)
        or timing.get("pose_reference_sensor") != "lidar"
        or not isinstance(lidar_candidate, dict)
        or lidar_candidate.get("direction") != "imu_body_to_local"
        or lidar_candidate.get("gt_pose_hypothesis") != "lidar"
        or lidar_candidate.get("frame_count") != coverage.get("covered_frame_count")
        or not isinstance(extrinsic, dict)
        or extrinsic.get("direction") != "lidar_to_imu_body"
        or not isinstance(sync, dict)
    ):
        raise ValueError("candidate audit 不满足 LiDAR-time/no-extrapolation 经验来源协议")

    pose_path = _member_path(candidate_dir, lidar_candidate.get("file"), "candidate pose")
    extrinsic_path = _member_path(candidate_dir, extrinsic.get("file"), "candidate LiDAR→body")
    sync_path = _member_path(candidate_dir, sync.get("file"), "Radar--LiDAR sync snapshot")
    _require_sha256(pose_path, lidar_candidate.get("sha256"), "candidate pose")
    _require_sha256(extrinsic_path, extrinsic.get("sha256"), "candidate LiDAR→body")
    sync_sha256 = _require_sha256(sync_path, sync.get("sha256"), "Radar--LiDAR sync snapshot")
    if sync.get("source_sha256") != sync_sha256:
        raise ValueError("Radar--LiDAR sync source/snapshot SHA-256 不一致")
    T_body_lidar = _load_diagnostic_extrinsic(extrinsic_path)
    audit_matrix = _validated_transform(extrinsic.get("matrix_4x4"), "candidate audit matrix_4x4")
    if not np.allclose(T_body_lidar, audit_matrix, atol=1e-8, rtol=0.0):
        raise ValueError("candidate LiDAR→body 文件与 audit matrix_4x4 不一致")
    source_rows = _load_pose_rows(pose_path, require_diagnostic=True)
    if len(source_rows) != coverage["covered_frame_count"]:
        raise ValueError("candidate pose 实际行数与 audit 不一致")
    direct_rows = [
        {
            "frame": row["frame"],
            "timestamp": row["timestamp"],
            "transform": _validated_transform(
                np.asarray(row["transform"]) @ T_body_lidar,
                "T_local_lidar",
            ),
        }
        for row in source_rows
    ]
    return {
        "audit": audit,
        "audit_path": audit_path,
        "audit_sha256": _sha256_file(audit_path),
        "pose_path": pose_path,
        "extrinsic_path": extrinsic_path,
        "sync_path": sync_path,
        "sync_sha256": sync_sha256,
        "T_body_lidar": T_body_lidar,
        "source_rows": source_rows,
        "direct_rows": direct_rows,
    }


def _validate_overlap_source(overlap_dir: str, candidate: Dict[str, object]) -> Dict[str, object]:
    overlap_dir = _regular_directory(overlap_dir, "overlap_dir")
    audit_path = _regular_file(os.path.join(overlap_dir, "audit.json"), "overlap audit")
    audit = _load_json(audit_path, "overlap audit")
    inputs = audit.get("inputs")
    coordinate = audit.get("coordinate_contract")
    ranking = audit.get("empirical_ranking")
    pairs = audit.get("pair_selection")
    identifiability = audit.get("identifiability")
    if (
        audit.get("protocol") != "mapping_pose_overlap_diagnostic_v1"
        or audit.get("formal") is not False
        or audit.get("diagnostic_only") is not True
        or not isinstance(inputs, dict)
        or inputs.get("candidate_audit_sha256") != candidate["audit_sha256"]
    ):
        if isinstance(inputs, dict) and inputs.get("candidate_audit_sha256") != candidate["audit_sha256"]:
            raise ValueError("overlap candidate audit SHA-256 不匹配")
        raise ValueError("overlap audit 不是受支持的 diagnostic 协议")
    if inputs.get("radar_lidar_sync_snapshot_sha256") != candidate["sync_sha256"]:
        raise ValueError("overlap Radar--LiDAR sync SHA-256 不匹配")
    if (
        not isinstance(coordinate, dict)
        or coordinate.get("voxel_coordinate_frame") != "lidar"
        or coordinate.get("pose_composition")
        != "T_local_lidar = T_local_body @ T_body_lidar"
        or coordinate.get("gt_as_lidar_external_cancels") is not True
        or not isinstance(ranking, dict)
        or ranking.get("preferred_hypothesis_diagnostic_only") != "gt_as_lidar"
        or not isinstance(ranking.get("lower_median_residual_first"), list)
        or ranking.get("lower_median_residual_first", [None])[0] != "gt_as_lidar"
        or not isinstance(pairs, dict)
        or type(pairs.get("selected_pair_count")) is not int
        or pairs.get("selected_pair_count", 0) <= 0
        or not isinstance(identifiability, dict)
        or identifiability.get("can_publish_formal_pose") is not False
    ):
        raise ValueError("overlap audit 不足以发布经验 GT-as-LiDAR 合同")
    return {
        "audit": audit,
        "audit_path": audit_path,
        "audit_sha256": _sha256_file(audit_path),
    }


def build_empirical_lidar_pose_contract(
    candidate_dir: str,
    overlap_dir: str,
    output_dir: str,
    command_line: Optional[str] = None,
) -> Dict[str, object]:
    """从已绑定诊断证据生成自包含经验合同；绝不提升为 airborne formal。"""
    candidate = _validate_candidate_sources(candidate_dir)
    overlap = _validate_overlap_source(overlap_dir, candidate)
    normalized_output = os.path.abspath(os.fspath(output_dir))
    if os.path.lexists(normalized_output):
        raise ValueError(f"输出目录已存在，拒绝覆盖: {normalized_output}")
    parent = os.path.dirname(normalized_output)
    if parent:
        os.makedirs(parent, exist_ok=True)

    direct_csv = _direct_pose_csv(candidate["direct_rows"])
    source_bytes = {
        SOURCE_CANDIDATE_AUDIT_FILE: _read_bytes(candidate["audit_path"]),
        SOURCE_OVERLAP_AUDIT_FILE: _read_bytes(overlap["audit_path"]),
        SOURCE_CANDIDATE_POSE_FILE: _read_bytes(candidate["pose_path"]),
        SOURCE_CANDIDATE_EXTRINSIC_FILE: _read_bytes(candidate["extrinsic_path"]),
        SYNC_SNAPSHOT_FILE: _read_bytes(candidate["sync_path"]),
        DIRECT_POSE_FILE: direct_csv.encode("utf-8"),
    }
    os.makedirs(normalized_output)
    for file_name, content in source_bytes.items():
        _write_bytes_atomic(os.path.join(normalized_output, file_name), content)

    coverage = candidate["audit"]["coverage"]
    overlap_audit = overlap["audit"]
    members = {
        key: {
            "file": file_name,
            "sha256": _sha256_file(os.path.join(normalized_output, file_name)),
        }
        for key, file_name in MEMBER_FILES.items()
    }
    receipt: Dict[str, object] = {
        "protocol": EMPIRICAL_POSE_PROTOCOL,
        "offline_empirical_mapping": True,
        "airborne_formal": False,
        "avoidance_formal": False,
        "evidence_level": "empirical_cross_frame_lidar_overlap",
        "map_frame": "local",
        "voxel_coordinate_frame": "lidar",
        "pose_direction": "lidar_to_local",
        "pose_composition": "direct_T_local_voxel",
        "frame_selection": "receipt_bound_ordered_subset_no_extrapolation",
        "coverage": {
            "available_radar_frame_count": int(coverage["radar_frame_count"]),
            "selected_pose_frame_count": int(coverage["covered_frame_count"]),
            "uncovered_frame_count": int(coverage["uncovered_frame_count"]),
            "uncovered_frame_ids": [str(frame) for frame in coverage["uncovered_frame_ids"]],
            "no_extrapolation": True,
        },
        "empirical_evidence": {
            "preferred_hypothesis": "gt_as_lidar",
            "selected_pair_count": int(overlap_audit["pair_selection"]["selected_pair_count"]),
            "ranking_metric": overlap_audit["empirical_ranking"].get("metric"),
            "hypothesis_summary": overlap_audit.get("hypothesis_summary"),
            "can_publish_airborne_formal_pose": False,
        },
        "formal_blockers": list(candidate["audit"].get("formal_blockers", [])),
        "members": members,
    }
    readme = (
        "# Empirical LiDAR pose contract\n\n"
        "本目录只授权离线经验地图，不能用于 airborne/avoidance formal。\n\n"
        "`lidar_to_local.csv` 由已绑定的 GT-as-LiDAR 候选与诊断外参代数组合得到，"
        "运行时会重新计算并交叉验证。未覆盖帧保持拒绝，不做外推。\n"
    )
    _write_text_atomic(os.path.join(normalized_output, README_FILE), readme)
    _write_text_atomic(
        os.path.join(normalized_output, COMMAND_FILE),
        (command_line or "Python API invocation; see empirical_pose_receipt.json") + "\n",
    )
    _write_json_atomic(os.path.join(normalized_output, EMPIRICAL_POSE_RECEIPT), receipt)
    return receipt


def _validate_receipt_members(
    contract_dir: str,
    receipt: Dict[str, object],
) -> Dict[str, str]:
    members = receipt.get("members")
    if not isinstance(members, dict) or set(members) != set(MEMBER_FILES):
        raise ValueError("经验 pose receipt members 集合不完整")
    paths: Dict[str, str] = {}
    member_names = []
    for key, expected_file in MEMBER_FILES.items():
        record = members.get(key)
        if not isinstance(record, dict) or record.get("file") != expected_file:
            raise ValueError(f"经验 pose receipt 成员名不匹配: {key}")
        path = _member_path(contract_dir, record.get("file"), f"经验 pose 成员 {key}")
        _require_sha256(path, record.get("sha256"), f"经验 pose 成员 {key}")
        paths[key] = path
        member_names.append(expected_file)
    expected_entries = set(member_names) | {EMPIRICAL_POSE_RECEIPT, README_FILE, COMMAND_FILE}
    actual_entries = set(os.listdir(contract_dir))
    if actual_entries != expected_entries:
        raise ValueError("经验 pose 合同目录含缺失或未绑定的额外条目")
    return paths


def load_empirical_lidar_pose_contract(
    receipt_path: str,
    available_voxel_file_names: Sequence[str],
) -> Dict[str, object]:
    """运行时复验经验合同，并返回 receipt 绑定的有序 inference 帧子集。"""
    receipt_path = _regular_file(receipt_path, "empirical_pose_receipt")
    contract_dir = _regular_directory(os.path.dirname(receipt_path), "经验 pose 合同目录")
    if os.path.basename(receipt_path) != EMPIRICAL_POSE_RECEIPT:
        raise ValueError(f"经验 pose receipt 文件名必须是 {EMPIRICAL_POSE_RECEIPT}")
    receipt = _load_json(receipt_path, "empirical_pose_receipt")
    coverage = receipt.get("coverage")
    if (
        receipt.get("protocol") != EMPIRICAL_POSE_PROTOCOL
        or receipt.get("offline_empirical_mapping") is not True
        or receipt.get("airborne_formal") is not False
        or receipt.get("avoidance_formal") is not False
        or receipt.get("evidence_level") != "empirical_cross_frame_lidar_overlap"
        or receipt.get("map_frame") != "local"
        or receipt.get("voxel_coordinate_frame") != "lidar"
        or receipt.get("pose_direction") != "lidar_to_local"
        or receipt.get("pose_composition") != "direct_T_local_voxel"
        or receipt.get("frame_selection")
        != "receipt_bound_ordered_subset_no_extrapolation"
        or not isinstance(coverage, dict)
        or coverage.get("no_extrapolation") is not True
    ):
        raise ValueError("经验 pose receipt 顶层合同无效")
    paths = _validate_receipt_members(contract_dir, receipt)

    # 合同内来源使用固定快照名；按 receipt 成员而不是原始绝对路径交叉校验。
    candidate_audit = _load_json(paths["source_candidate_audit"], "source candidate audit")
    overlap_audit = _load_json(paths["source_overlap_audit"], "source overlap audit")
    candidate_pose_records = candidate_audit.get("pose_candidates")
    candidate_inputs = candidate_audit.get("inputs")
    candidate_coverage = candidate_audit.get("coverage")
    candidate_timing = candidate_audit.get("timing")
    candidate_extrinsic_record = candidate_audit.get("candidate_lidar_to_body")
    overlap_inputs = overlap_audit.get("inputs")
    overlap_ranking = overlap_audit.get("empirical_ranking")
    overlap_coordinate = overlap_audit.get("coordinate_contract")
    overlap_pairs = overlap_audit.get("pair_selection")
    overlap_identifiability = overlap_audit.get("identifiability")
    empirical_evidence = receipt.get("empirical_evidence")
    if not all(
        isinstance(value, dict)
        for value in (
            candidate_pose_records,
            candidate_inputs,
            candidate_coverage,
            candidate_timing,
            candidate_extrinsic_record,
            overlap_inputs,
            overlap_ranking,
            overlap_coordinate,
            overlap_pairs,
            overlap_identifiability,
            empirical_evidence,
        )
    ):
        raise ValueError("经验 pose receipt 或来源快照对象结构不完整")
    candidate_pose_record = candidate_pose_records.get("gt_as_lidar")
    candidate_sync_record = candidate_inputs.get("radar_lidar_sync_snapshot")
    if not isinstance(candidate_pose_record, dict) or not isinstance(
        candidate_sync_record,
        dict,
    ):
        raise ValueError("经验 pose candidate 成员记录不完整")
    if (
        candidate_pose_record.get("sha256") != _sha256_file(paths["source_candidate_pose"])
        or candidate_extrinsic_record.get("sha256")
        != _sha256_file(paths["source_candidate_lidar_to_body"])
        or candidate_sync_record.get("sha256")
        != _sha256_file(paths["radar_lidar_sync_snapshot"])
        or overlap_inputs.get("candidate_audit_sha256")
        != _sha256_file(paths["source_candidate_audit"])
        or overlap_inputs.get("radar_lidar_sync_snapshot_sha256")
        != _sha256_file(paths["radar_lidar_sync_snapshot"])
        or candidate_sync_record.get("source_sha256")
        != _sha256_file(paths["radar_lidar_sync_snapshot"])
    ):
        raise ValueError("经验 pose 来源快照交叉 SHA-256 不一致")
    if (
        candidate_audit.get("protocol") != "mapping_pose_candidate_diagnostic_v2"
        or candidate_audit.get("formal") is not False
        or candidate_audit.get("candidate_only") is not True
        or candidate_audit.get("assumptions_resolved") is not False
        or candidate_timing.get("pose_reference_sensor") != "lidar"
        or candidate_coverage.get("no_extrapolation") is not True
        or candidate_pose_record.get("direction") != "imu_body_to_local"
        or candidate_pose_record.get("gt_pose_hypothesis") != "lidar"
        or candidate_pose_record.get("frame_count")
        != coverage.get("selected_pose_frame_count")
        or candidate_extrinsic_record.get("direction") != "lidar_to_imu_body"
        or overlap_audit.get("protocol") != "mapping_pose_overlap_diagnostic_v1"
        or overlap_audit.get("formal") is not False
        or overlap_audit.get("diagnostic_only") is not True
        or overlap_ranking.get("preferred_hypothesis_diagnostic_only") != "gt_as_lidar"
        or not isinstance(overlap_ranking.get("lower_median_residual_first"), list)
        or overlap_ranking.get("lower_median_residual_first", [None])[0]
        != "gt_as_lidar"
        or overlap_coordinate.get("voxel_coordinate_frame") != "lidar"
        or overlap_coordinate.get("pose_composition")
        != "T_local_lidar = T_local_body @ T_body_lidar"
        or overlap_coordinate.get("gt_as_lidar_external_cancels") is not True
        or type(overlap_pairs.get("selected_pair_count")) is not int
        or overlap_pairs.get("selected_pair_count", 0) <= 0
        or overlap_identifiability.get("can_publish_formal_pose") is not False
        or empirical_evidence.get("preferred_hypothesis") != "gt_as_lidar"
        or empirical_evidence.get("can_publish_airborne_formal_pose") is not False
        or empirical_evidence.get("selected_pair_count")
        != overlap_pairs.get("selected_pair_count")
        or empirical_evidence.get("ranking_metric") != overlap_ranking.get("metric")
        or empirical_evidence.get("hypothesis_summary")
        != overlap_audit.get("hypothesis_summary")
        or receipt.get("formal_blockers") != candidate_audit.get("formal_blockers")
        or candidate_coverage.get("radar_frame_count")
        != coverage.get("available_radar_frame_count")
        or candidate_coverage.get("covered_frame_count")
        != coverage.get("selected_pose_frame_count")
        or candidate_coverage.get("uncovered_frame_count")
        != coverage.get("uncovered_frame_count")
        or candidate_coverage.get("uncovered_frame_ids")
        != coverage.get("uncovered_frame_ids")
    ):
        raise ValueError("经验 pose 来源快照语义不完整")

    source_rows = _load_pose_rows(paths["source_candidate_pose"], require_diagnostic=True)
    T_body_lidar = _load_diagnostic_extrinsic(paths["source_candidate_lidar_to_body"])
    audit_matrix = _validated_transform(
        candidate_extrinsic_record.get("matrix_4x4"),
        "source candidate matrix_4x4",
    )
    if not np.allclose(T_body_lidar, audit_matrix, atol=1e-8, rtol=0.0):
        raise ValueError("经验 pose 来源外参文件与 audit 矩阵不一致")
    direct_rows = _load_pose_rows(paths["lidar_to_local_pose"], require_diagnostic=False)
    if len(source_rows) != len(direct_rows):
        raise ValueError("经验 direct pose 与来源 pose 行数不一致")
    pose_table: Dict[str, Dict[str, object]] = {}
    for source, direct in zip(source_rows, direct_rows):
        expected = _validated_transform(
            np.asarray(source["transform"]) @ T_body_lidar,
            "重算 T_local_lidar",
        )
        if (
            source["frame"] != direct["frame"]
            or not np.isclose(source["timestamp"], direct["timestamp"], atol=5e-7, rtol=0.0)
            or not np.allclose(expected, direct["transform"], atol=5e-6, rtol=0.0)
        ):
            raise ValueError("经验 direct pose 与来源组合不一致")
        pose_table[str(direct["frame"])] = {
            "timestamp": float(direct["timestamp"]),
            "T_local_voxel": np.asarray(direct["transform"], dtype=np.float32),
        }

    def frame_key(file_name: str) -> str:
        stem = os.path.splitext(os.path.basename(str(file_name)))[0]
        return stem[:-6] if stem.endswith("_voxel") else stem

    available = [str(name) for name in available_voxel_file_names]
    available_keys = [frame_key(name) for name in available]
    if len(available_keys) != len(set(available_keys)):
        raise ValueError("可用 inference voxel 映射到重复 frame 键")
    expected_available = coverage.get("available_radar_frame_count")
    expected_selected = coverage.get("selected_pose_frame_count")
    uncovered = [str(frame) for frame in coverage.get("uncovered_frame_ids", [])]
    if (
        type(expected_available) is not int
        or type(expected_selected) is not int
        or type(coverage.get("uncovered_frame_count")) is not int
        or expected_available != len(available)
        or expected_selected != len(pose_table)
        or coverage.get("uncovered_frame_count") != len(uncovered)
        or expected_available != expected_selected + len(uncovered)
    ):
        raise ValueError("经验 pose receipt coverage 与可用 inference 帧数不一致")
    selected_keys = list(pose_table)
    if set(available_keys) != set(selected_keys) | set(uncovered):
        raise ValueError("经验 pose receipt 的 selected/uncovered 帧集合不完整")
    available_index = {frame: index for index, frame in enumerate(available_keys)}
    if any(frame not in available_index for frame in selected_keys):
        raise ValueError("经验 pose 帧不在 inference voxel 中")
    selected_indices = [available_index[frame] for frame in selected_keys]
    if selected_indices != sorted(selected_indices):
        raise ValueError("经验 pose 帧不是 inference 的有序子集")
    selected_files = [available[index] for index in selected_indices]
    return {
        "receipt": receipt,
        "receipt_path": receipt_path,
        "receipt_sha256": _sha256_file(receipt_path),
        "pose_path": paths["lidar_to_local_pose"],
        "pose_sha256": _sha256_file(paths["lidar_to_local_pose"]),
        "pose_table": pose_table,
        "selected_voxel_file_names": selected_files,
        "available_frame_count": len(available),
        "selected_frame_count": len(selected_files),
        "uncovered_frame_count": len(uncovered),
    }
