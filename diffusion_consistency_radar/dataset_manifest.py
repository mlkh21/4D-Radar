#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成并验证严格、可移植的场景级数据集内容 manifest。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from typing import Dict, List, Mapping, Sequence, Tuple


MANIFEST_FILENAME = "dataset_manifest.json"
SCHEMA_VERSION = 1
SCHEMA_VERSION_V2 = 2
SCHEMA_VERSION_DEPLOYMENT = 3
POLICY_FILENAME = "preprocess_policy.json"
DEPLOYMENT_RECEIPT_FILENAME = "deployment_view.json"
SOURCE_TRAINING_MANIFEST_FILENAME = "source_training_manifest.json"
REQUIRED_PROVENANCE = (
    "preprocess_script",
    "calibration",
    "radar_index",
    "lidar_index",
)
MODALITY_PATTERNS = {
    "radar_voxel": re.compile(r"^(\d{6})\.(?:npy|npz)$"),
    "lidar_voxel": re.compile(r"^(\d{6})\.(?:npy|npz)$"),
    "target_voxel": re.compile(r"^(\d{6})\.(?:npy|npz)$"),
    "observed_mask": re.compile(r"^(\d{6})\.npz$"),
    "ir_image": re.compile(r"^(\d{6})_ir\.npy$"),
}
LEGACY_MODALITIES = ("radar_voxel", "lidar_voxel", "target_voxel", "ir_image")
PROFILE_MODALITIES = {
    "training": (
        "radar_voxel",
        "lidar_voxel",
        "target_voxel",
        "observed_mask",
        "ir_image",
    ),
    "deployment": ("radar_voxel", "ir_image"),
}
PROFILE_PROVENANCE = {
    "training": (
        "preprocess_script",
        "radar_to_lidar",
        "radar_to_thermal",
        "lidar_to_thermal",
        "thermal_intrinsics",
        "radar_lidar_sync",
        "radar_ir_sync",
        "target_policy",
    ),
    "deployment": (
        "preprocess_script",
        "radar_to_lidar",
        "radar_to_thermal",
        "lidar_to_thermal",
        "thermal_intrinsics",
        "radar_ir_sync",
    ),
}
DEPLOYMENT_V3_PROVENANCE = PROFILE_PROVENANCE["deployment"] + (
    "source_training_manifest",
    "deployment_view_receipt",
)
DEPLOYMENT_V3_SCENE_ENTRIES = {
    "radar_voxel",
    "ir_image",
    POLICY_FILENAME,
    SOURCE_TRAINING_MANIFEST_FILENAME,
    DEPLOYMENT_RECEIPT_FILENAME,
    "radar_ir_sync.csv",
    MANIFEST_FILENAME,
}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class DatasetManifestError(ValueError):
    """表示数据集内容或 manifest 不满足严格协议。"""


def _canonical_bytes(value: object) -> bytes:
    """把 JSON 值编码为确定性的 UTF-8 字节。"""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha256_bytes(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def sha256_json_value(value: object) -> str:
    """返回规范 JSON 值的内容 SHA-256，供派生 artifact 绑定清单子集。"""
    return _sha256_bytes(value)


def sha256_file(path: str) -> str:
    """以固定分块流式计算普通文件 SHA-256。"""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_directory(path: str, label: str) -> None:
    if not os.path.lexists(path):
        raise DatasetManifestError(f"{label} 目录不存在: {path}")
    if os.path.islink(path):
        raise DatasetManifestError(f"{label} 不允许使用符号链接: {path}")
    if not os.path.isdir(path):
        raise DatasetManifestError(f"{label} 不是目录: {path}")


def _require_file(path: str, label: str) -> None:
    if not os.path.lexists(path):
        raise DatasetManifestError(f"{label} 文件不存在: {path}")
    if os.path.islink(path):
        raise DatasetManifestError(f"{label} 不允许使用符号链接: {path}")
    if not os.path.isfile(path):
        raise DatasetManifestError(f"{label} 不是普通文件: {path}")


def _load_policy(
    scene_dir: str,
    scene: str,
    expected_frame_count: int,
) -> Tuple[Dict[str, object], str]:
    policy_path = os.path.join(scene_dir, POLICY_FILENAME)
    _require_file(policy_path, POLICY_FILENAME)
    try:
        with open(policy_path, "r", encoding="utf-8") as handle:
            policy = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetManifestError(
            f"{POLICY_FILENAME} 无法解析: {policy_path}: {exc}"
        ) from exc
    if not isinstance(policy, dict):
        raise DatasetManifestError(f"{POLICY_FILENAME} 必须是 JSON 对象")
    if policy.get("source_scene") != scene:
        raise DatasetManifestError(
            "preprocess_policy source_scene 与场景不一致: "
            f"expected={scene!r}, actual={policy.get('source_scene')!r}"
        )
    frames_written = policy.get("frames_written")
    if type(frames_written) is not int or frames_written != expected_frame_count:
        raise DatasetManifestError(
            "preprocess_policy frames_written 与预期不一致: "
            f"expected={expected_frame_count}, actual={frames_written!r}"
        )
    return policy, sha256_file(policy_path)


def _scan_modality(
    scene_dir: str,
    modality: str,
) -> Tuple[List[str], List[Dict[str, object]]]:
    directory = os.path.join(scene_dir, modality)
    _require_directory(directory, modality)
    pattern = MODALITY_PATTERNS[modality]
    by_frame: Dict[str, Dict[str, object]] = {}
    with os.scandir(directory) as iterator:
        entries = sorted(iterator, key=lambda entry: entry.name)
    for entry in entries:
        path = entry.path
        if entry.is_symlink():
            raise DatasetManifestError(
                f"{modality} 不允许使用符号链接: {path}"
            )
        if not entry.is_file(follow_symlinks=False):
            raise DatasetManifestError(f"{modality} 包含未知目录项: {path}")
        match = pattern.fullmatch(entry.name)
        if match is None:
            raise DatasetManifestError(f"{modality} 包含未知文件: {path}")
        frame_id = match.group(1)
        if frame_id in by_frame:
            raise DatasetManifestError(
                f"{modality} frame ID 重复: {frame_id}"
            )
        stat_result = entry.stat(follow_symlinks=False)
        by_frame[frame_id] = {
            "frame_id": frame_id,
            "path": f"{modality}/{entry.name}",
            "size": int(stat_result.st_size),
            "sha256": sha256_file(path),
        }
    frame_ids = sorted(by_frame)
    return frame_ids, [by_frame[frame_id] for frame_id in frame_ids]


def _collect_modalities(
    scene_dir: str,
    expected_frame_count: int,
) -> Dict[str, List[Dict[str, object]]]:
    if type(expected_frame_count) is not int or expected_frame_count <= 0:
        raise DatasetManifestError(
            "expected_frame_count 必须是严格正整数: "
            f"{expected_frame_count!r}"
        )
    frame_ids_by_modality: Dict[str, List[str]] = {}
    records_by_modality: Dict[str, List[Dict[str, object]]] = {}
    for modality in LEGACY_MODALITIES:
        frame_ids, records = _scan_modality(scene_dir, modality)
        frame_ids_by_modality[modality] = frame_ids
        records_by_modality[modality] = records

    reference = frame_ids_by_modality["radar_voxel"]
    for modality, frame_ids in frame_ids_by_modality.items():
        if frame_ids != reference:
            raise DatasetManifestError(
                "legacy 四模态 frame ID 集合不一致: "
                f"radar_voxel={reference}, {modality}={frame_ids}"
            )
    expected = [f"{index:06d}" for index in range(expected_frame_count)]
    if reference != expected:
        raise DatasetManifestError(
            "frame ID 必须从 000000 严格连续且匹配 expected_frame_count: "
            f"expected={expected}, actual={reference}"
        )
    return records_by_modality


def _collect_provenance(
    provenance_paths: Mapping[str, str],
) -> Dict[str, Dict[str, str]]:
    if set(provenance_paths) != set(REQUIRED_PROVENANCE):
        raise DatasetManifestError(
            "provenance 字段必须精确包含: "
            f"{list(REQUIRED_PROVENANCE)}"
        )
    provenance: Dict[str, Dict[str, str]] = {}
    for key in REQUIRED_PROVENANCE:
        path = os.fspath(provenance_paths[key])
        _require_file(path, f"provenance {key}")
        provenance[key] = {
            "name": os.path.basename(path),
            "sha256": sha256_file(path),
        }
    return provenance


def _profile_contract(profile: str, schema_version: int = SCHEMA_VERSION_V2):
    profile = str(profile).strip().lower()
    if profile not in PROFILE_MODALITIES:
        raise DatasetManifestError(
            f"manifest profile 必须为 {sorted(PROFILE_MODALITIES)}，实际为 {profile!r}"
        )
    if schema_version == SCHEMA_VERSION_DEPLOYMENT:
        if profile != "deployment":
            raise DatasetManifestError("manifest schema v3 只允许 deployment profile")
        provenance = DEPLOYMENT_V3_PROVENANCE
    elif schema_version == SCHEMA_VERSION_V2:
        provenance = PROFILE_PROVENANCE[profile]
    else:
        raise DatasetManifestError(f"未知 profile manifest schema: {schema_version}")
    return profile, PROFILE_MODALITIES[profile], provenance


def _collect_modalities_v2(
    scene_dir: str,
    expected_frame_count: int,
    required_modalities: Sequence[str],
) -> Dict[str, List[Dict[str, object]]]:
    """按 profile 精确扫描模态；required 集合内禁止缺帧或不连续。"""
    if type(expected_frame_count) is not int or expected_frame_count <= 0:
        raise DatasetManifestError("expected_frame_count 必须是严格正整数")
    records_by_modality: Dict[str, List[Dict[str, object]]] = {}
    frame_ids_by_modality: Dict[str, List[str]] = {}
    for modality in required_modalities:
        frame_ids, records = _scan_modality(scene_dir, modality)
        frame_ids_by_modality[modality] = frame_ids
        records_by_modality[modality] = records
    reference_name = required_modalities[0]
    reference = frame_ids_by_modality[reference_name]
    for modality, frame_ids in frame_ids_by_modality.items():
        if frame_ids != reference:
            raise DatasetManifestError(
                f"profile 模态 frame ID 集合不一致: {reference_name}={reference}, "
                f"{modality}={frame_ids}"
            )
    expected = [f"{index:06d}" for index in range(expected_frame_count)]
    if reference != expected:
        raise DatasetManifestError(
            "frame ID 必须从 000000 严格连续且匹配 expected_frame_count: "
            f"expected={expected}, actual={reference}"
        )
    return records_by_modality


def _collect_provenance_v2(
    provenance_paths: Mapping[str, str],
    required_provenance: Sequence[str],
) -> Dict[str, Dict[str, str]]:
    if set(provenance_paths) != set(required_provenance):
        raise DatasetManifestError(
            "v2 provenance 字段必须精确包含: " + str(list(required_provenance))
        )
    provenance: Dict[str, Dict[str, str]] = {}
    for key in required_provenance:
        path = os.fspath(provenance_paths[key])
        _require_file(path, f"provenance {key}")
        provenance[key] = {
            "name": os.path.basename(path),
            "sha256": sha256_file(path),
        }
    return provenance


def _manifest_payload_v2(
    *,
    scene: str,
    profile: str,
    frame_count: int,
    voxel_coordinate_frame: str,
    policy_sha256: str,
    provenance: Mapping[str, Mapping[str, str]],
    modalities: Mapping[str, Sequence[Mapping[str, object]]],
    schema_version: int = SCHEMA_VERSION_V2,
) -> Dict[str, object]:
    return {
        "schema_version": schema_version,
        "profile": profile,
        "scene": scene,
        "frame_count": frame_count,
        "voxel_coordinate_frame": voxel_coordinate_frame,
        "preprocessing": {
            "policy_path": POLICY_FILENAME,
            "policy_sha256": policy_sha256,
            "provenance": dict(provenance),
        },
        "modalities": dict(modalities),
    }


def _manifest_payload(
    scene: str,
    frame_count: int,
    policy_sha256: str,
    provenance: Mapping[str, Mapping[str, str]],
    modalities: Mapping[str, Sequence[Mapping[str, object]]],
) -> Dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "scene": scene,
        "frame_count": frame_count,
        "preprocessing": {
            "policy_path": POLICY_FILENAME,
            "policy_sha256": policy_sha256,
            "provenance": dict(provenance),
        },
        "modalities": dict(modalities),
    }


def build_scene_manifest(
    scene_dir: str,
    scene: str,
    expected_frame_count: int,
    provenance_paths: Mapping[str, str],
    *,
    profile: str | None = None,
) -> Dict[str, object]:
    """扫描干净场景；未指定 profile 仅用于兼容构建 v1 测试数据。"""
    scene_dir = os.path.abspath(scene_dir)
    _require_directory(scene_dir, "scene")
    if not isinstance(scene, str) or not scene:
        raise DatasetManifestError("scene 必须是非空字符串")
    policy, policy_sha256 = _load_policy(
        scene_dir,
        scene,
        expected_frame_count,
    )
    if profile is None:
        modalities = _collect_modalities(scene_dir, expected_frame_count)
        provenance = _collect_provenance(provenance_paths)
        payload = _manifest_payload(
            scene,
            expected_frame_count,
            policy_sha256,
            provenance,
            modalities,
        )
    else:
        manifest_schema = (
            SCHEMA_VERSION_DEPLOYMENT
            if str(profile).strip().lower() == "deployment"
            else SCHEMA_VERSION_V2
        )
        profile, modalities_required, provenance_required = _profile_contract(
            profile,
            manifest_schema,
        )
        voxel_coordinate_frame = policy.get("voxel_coordinate_frame") or policy.get("align_to")
        if voxel_coordinate_frame not in ("lidar", "radar"):
            raise DatasetManifestError(
                "v2 preprocess_policy 必须声明 voxel_coordinate_frame=lidar|radar"
            )
        modalities = _collect_modalities_v2(
            scene_dir,
            expected_frame_count,
            modalities_required,
        )
        provenance = _collect_provenance_v2(
            provenance_paths,
            provenance_required,
        )
        payload = _manifest_payload_v2(
            scene=scene,
            profile=profile,
            frame_count=expected_frame_count,
            voxel_coordinate_frame=voxel_coordinate_frame,
            policy_sha256=policy_sha256,
            provenance=provenance,
            modalities=modalities,
            schema_version=manifest_schema,
        )
    manifest = dict(payload)
    manifest["content_sha256"] = _sha256_bytes(payload)
    return manifest


def write_scene_manifest_atomic(
    scene_dir: str,
    scene: str,
    expected_frame_count: int,
    provenance_paths: Mapping[str, str],
    *,
    profile: str | None = None,
) -> str:
    """原子发布 manifest；正式文件存在时拒绝覆盖。"""
    scene_dir = os.path.abspath(scene_dir)
    _require_directory(scene_dir, "scene")
    manifest_path = os.path.join(scene_dir, MANIFEST_FILENAME)
    if os.path.lexists(manifest_path):
        raise DatasetManifestError(f"manifest 已存在，拒绝覆盖: {manifest_path}")
    manifest = build_scene_manifest(
        scene_dir,
        scene,
        expected_frame_count,
        provenance_paths,
        profile=profile,
    )

    descriptor, temp_path = tempfile.mkstemp(
        dir=scene_dir,
        prefix=".dataset_manifest.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                manifest,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp_path, manifest_path)
        except FileExistsError as exc:
            raise DatasetManifestError(
                f"manifest 已存在，拒绝覆盖: {manifest_path}"
            ) from exc
        except OSError as exc:
            raise DatasetManifestError(
                f"manifest 原子发布失败: {manifest_path}: {exc}"
            ) from exc
    finally:
        if os.path.lexists(temp_path):
            os.unlink(temp_path)
    return manifest_path


def _validate_provenance_records(value: object) -> Dict[str, Dict[str, str]]:
    if not isinstance(value, dict) or set(value) != set(REQUIRED_PROVENANCE):
        raise DatasetManifestError("manifest provenance 字段不完整")
    validated: Dict[str, Dict[str, str]] = {}
    for key in REQUIRED_PROVENANCE:
        record = value[key]
        if not isinstance(record, dict) or set(record) != {"name", "sha256"}:
            raise DatasetManifestError(
                f"manifest provenance {key} 记录格式无效"
            )
        name = record.get("name")
        digest = record.get("sha256")
        if (
            not isinstance(name, str)
            or not name
            or os.path.basename(name) != name
        ):
            raise DatasetManifestError(
                f"manifest provenance {key} name 无效"
            )
        if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
            raise DatasetManifestError(
                f"manifest provenance {key} sha256 无效"
            )
        validated[key] = {"name": name, "sha256": digest}
    return validated


def _validate_provenance_records_v2(
    value: object,
    required_provenance: Sequence[str],
) -> Dict[str, Dict[str, str]]:
    if not isinstance(value, dict) or set(value) != set(required_provenance):
        raise DatasetManifestError("manifest v2 provenance 字段不完整")
    validated: Dict[str, Dict[str, str]] = {}
    for key in required_provenance:
        record = value[key]
        if not isinstance(record, dict) or set(record) != {"name", "sha256"}:
            raise DatasetManifestError(f"manifest provenance {key} 记录格式无效")
        name = record.get("name")
        digest = record.get("sha256")
        if not isinstance(name, str) or not name or os.path.basename(name) != name:
            raise DatasetManifestError(f"manifest provenance {key} name 无效")
        if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
            raise DatasetManifestError(f"manifest provenance {key} sha256 无效")
        validated[key] = {"name": name, "sha256": digest}
    return validated


def _validate_profile_manifest(
    scene_dir: str,
    expected_scene: str,
    expected_profile: str | None,
    manifest: Mapping[str, object],
    schema_version: int,
) -> Dict[str, object]:
    expected_top_keys = {
        "schema_version",
        "profile",
        "scene",
        "frame_count",
        "voxel_coordinate_frame",
        "preprocessing",
        "modalities",
        "content_sha256",
    }
    if set(manifest) != expected_top_keys:
        raise DatasetManifestError(
            f"dataset_manifest.json 顶层字段不符合 v{schema_version}"
        )
    if manifest.get("schema_version") != schema_version:
        raise DatasetManifestError("manifest schema_version 与验证入口不一致")
    recorded_content_sha256 = manifest.get("content_sha256")
    payload = {key: value for key, value in manifest.items() if key != "content_sha256"}
    if (
        not isinstance(recorded_content_sha256, str)
        or _sha256_bytes(payload) != recorded_content_sha256
    ):
        raise DatasetManifestError("manifest content_sha256 不一致")
    profile, modalities_required, provenance_required = _profile_contract(
        manifest.get("profile"),
        schema_version,
    )
    if expected_profile is not None and profile != expected_profile:
        raise DatasetManifestError(
            f"manifest profile 与入口期望不一致: expected={expected_profile!r}, "
            f"actual={profile!r}"
        )
    if manifest.get("scene") != expected_scene:
        raise DatasetManifestError("manifest scene 与入口期望不一致")
    frame_count = manifest.get("frame_count")
    if type(frame_count) is not int or frame_count <= 0:
        raise DatasetManifestError("manifest frame_count 必须是严格正整数")
    voxel_coordinate_frame = manifest.get("voxel_coordinate_frame")
    if voxel_coordinate_frame not in ("lidar", "radar"):
        raise DatasetManifestError("manifest voxel_coordinate_frame 无效")
    preprocessing = manifest.get("preprocessing")
    if not isinstance(preprocessing, dict) or set(preprocessing) != {
        "policy_path",
        "policy_sha256",
        "provenance",
    }:
        raise DatasetManifestError(
            f"manifest preprocessing 字段不符合 v{schema_version}"
        )
    if preprocessing.get("policy_path") != POLICY_FILENAME:
        raise DatasetManifestError(
            f"manifest policy_path 不符合 v{schema_version}"
        )
    recorded_policy_sha256 = preprocessing.get("policy_sha256")
    if (
        not isinstance(recorded_policy_sha256, str)
        or _SHA256_PATTERN.fullmatch(recorded_policy_sha256) is None
    ):
        raise DatasetManifestError("manifest policy_sha256 无效")
    provenance = _validate_provenance_records_v2(
        preprocessing.get("provenance"),
        provenance_required,
    )
    policy, actual_policy_sha256 = _load_policy(
        scene_dir,
        expected_scene,
        frame_count,
    )
    if actual_policy_sha256 != recorded_policy_sha256:
        raise DatasetManifestError("preprocess policy SHA-256 不一致")
    policy_frame = policy.get("voxel_coordinate_frame") or policy.get("align_to")
    if policy_frame != voxel_coordinate_frame:
        raise DatasetManifestError("manifest voxel frame 与 preprocess policy 不一致")
    if schema_version == SCHEMA_VERSION_DEPLOYMENT:
        with os.scandir(scene_dir) as iterator:
            actual_entries = {entry.name for entry in iterator}
        if actual_entries != DEPLOYMENT_V3_SCENE_ENTRIES:
            extra = sorted(actual_entries - DEPLOYMENT_V3_SCENE_ENTRIES)
            missing = sorted(DEPLOYMENT_V3_SCENE_ENTRIES - actual_entries)
            raise DatasetManifestError(
                "严格 deployment 场景包含未知或缺失目录项: "
                f"extra={extra}, missing={missing}"
            )
    actual_modalities = _collect_modalities_v2(
        scene_dir,
        frame_count,
        modalities_required,
    )
    if manifest.get("modalities") != actual_modalities:
        raise DatasetManifestError("manifest 模态文件内容不一致")
    actual_payload = _manifest_payload_v2(
        scene=expected_scene,
        profile=profile,
        frame_count=frame_count,
        voxel_coordinate_frame=voxel_coordinate_frame,
        policy_sha256=actual_policy_sha256,
        provenance=provenance,
        modalities=actual_modalities,
        schema_version=schema_version,
    )
    if _sha256_bytes(actual_payload) != recorded_content_sha256:
        raise DatasetManifestError("场景内容与 manifest content_sha256 不一致")
    return dict(manifest)


def validate_scene_manifest(
    scene_dir: str,
    expected_scene: str,
    expected_profile: str | None = None,
) -> Dict[str, object]:
    """自动识别 v1/v2/v3；正式 deployment 入口只接受严格 v3。"""
    scene_dir = os.path.abspath(scene_dir)
    _require_directory(scene_dir, "scene")
    manifest_path = os.path.join(scene_dir, MANIFEST_FILENAME)
    _require_file(manifest_path, MANIFEST_FILENAME)
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetManifestError(
            f"{MANIFEST_FILENAME} 无法解析: {manifest_path}: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise DatasetManifestError("dataset_manifest.json 必须是 JSON 对象")
    manifest_schema = manifest.get("schema_version")
    if manifest_schema == SCHEMA_VERSION_V2:
        if expected_profile == "deployment":
            raise DatasetManifestError(
                "严格 deployment 入口要求 schema v3，拒绝旧 v2 deployment"
            )
        return _validate_profile_manifest(
            scene_dir,
            expected_scene,
            expected_profile,
            manifest,
            SCHEMA_VERSION_V2,
        )
    if manifest_schema == SCHEMA_VERSION_DEPLOYMENT:
        return _validate_profile_manifest(
            scene_dir,
            expected_scene,
            expected_profile,
            manifest,
            SCHEMA_VERSION_DEPLOYMENT,
        )
    if expected_profile is not None:
        raise DatasetManifestError(
            f"入口要求 manifest profile={expected_profile!r}，legacy v1 不包含 profile"
        )
    expected_top_keys = {
        "schema_version",
        "scene",
        "frame_count",
        "preprocessing",
        "modalities",
        "content_sha256",
    }
    if set(manifest) != expected_top_keys:
        raise DatasetManifestError("dataset_manifest.json 顶层字段不符合 v1")

    recorded_content_sha256 = manifest.get("content_sha256")
    payload = {
        key: value
        for key, value in manifest.items()
        if key != "content_sha256"
    }
    if (
        not isinstance(recorded_content_sha256, str)
        or _sha256_bytes(payload) != recorded_content_sha256
    ):
        raise DatasetManifestError("manifest content_sha256 不一致")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise DatasetManifestError(
            f"manifest schema_version 不支持: {manifest.get('schema_version')!r}"
        )
    if manifest.get("scene") != expected_scene:
        raise DatasetManifestError(
            "manifest scene 与入口期望不一致: "
            f"expected={expected_scene!r}, actual={manifest.get('scene')!r}"
        )
    frame_count = manifest.get("frame_count")
    if type(frame_count) is not int or frame_count <= 0:
        raise DatasetManifestError("manifest frame_count 必须是严格正整数")

    preprocessing = manifest.get("preprocessing")
    if not isinstance(preprocessing, dict) or set(preprocessing) != {
        "policy_path",
        "policy_sha256",
        "provenance",
    }:
        raise DatasetManifestError("manifest preprocessing 字段不符合 v1")
    if preprocessing.get("policy_path") != POLICY_FILENAME:
        raise DatasetManifestError("manifest policy_path 不符合 v1")
    recorded_policy_sha256 = preprocessing.get("policy_sha256")
    if (
        not isinstance(recorded_policy_sha256, str)
        or _SHA256_PATTERN.fullmatch(recorded_policy_sha256) is None
    ):
        raise DatasetManifestError("manifest policy_sha256 无效")
    provenance = _validate_provenance_records(
        preprocessing.get("provenance")
    )

    _policy, actual_policy_sha256 = _load_policy(
        scene_dir,
        expected_scene,
        frame_count,
    )
    if actual_policy_sha256 != recorded_policy_sha256:
        raise DatasetManifestError("preprocess policy SHA-256 不一致")
    actual_modalities = _collect_modalities(scene_dir, frame_count)
    if manifest.get("modalities") != actual_modalities:
        raise DatasetManifestError("manifest 模态文件内容不一致")

    actual_payload = _manifest_payload(
        expected_scene,
        frame_count,
        actual_policy_sha256,
        provenance,
        actual_modalities,
    )
    if _sha256_bytes(actual_payload) != recorded_content_sha256:
        raise DatasetManifestError("场景内容与 manifest content_sha256 不一致")
    return manifest
