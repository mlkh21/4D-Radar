#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从严格 training manifest 生成仅含 Radar+IR 的部署数据视图。"""

from __future__ import annotations

import json
import os
import shutil
from typing import Dict, Mapping, Sequence, Tuple

from diffusion_consistency_radar.dataset_manifest import (
    DEPLOYMENT_RECEIPT_FILENAME,
    MANIFEST_FILENAME,
    POLICY_FILENAME,
    PROFILE_PROVENANCE,
    SCHEMA_VERSION_V2,
    SOURCE_TRAINING_MANIFEST_FILENAME,
    DatasetManifestError,
    sha256_file,
    sha256_json_value,
    validate_scene_manifest,
    write_scene_manifest_atomic,
)


DEPLOYMENT_VIEW_PROTOCOL = "deployment_view_v1"
DEPLOYMENT_DATASET_PROTOCOL = "deployment_dataset_v1"
DEPLOYMENT_DATASET_FILENAME = "deployment_dataset.json"
DEPLOYMENT_MODALITIES = ("radar_voxel", "ir_image")
CALIBRATION_FILENAMES = {
    "radar_to_lidar": "calib_radar_to_livox.txt",
    "radar_to_thermal": "calib_radar_to_thermal.txt",
    "lidar_to_thermal": "calib_livox_to_thermal.txt",
    "thermal_intrinsics": "calib_cam_thermal.txt",
}


class DeploymentViewError(DatasetManifestError):
    """表示 deployment view 无法安全生成或验证。"""


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _require_regular_file(path: str, label: str) -> str:
    path = os.path.abspath(os.fspath(path))
    if not os.path.lexists(path):
        raise DeploymentViewError(f"{label} 不存在: {path}")
    if os.path.islink(path) or not os.path.isfile(path):
        raise DeploymentViewError(f"{label} 必须是非符号链接普通文件: {path}")
    return path


def _require_directory(path: str, label: str) -> str:
    path = os.path.abspath(os.fspath(path))
    if not os.path.lexists(path):
        raise DeploymentViewError(f"{label} 不存在: {path}")
    if os.path.islink(path) or not os.path.isdir(path):
        raise DeploymentViewError(f"{label} 必须是非符号链接目录: {path}")
    return path


def _normalize_scenes(scenes: Sequence[str]) -> Tuple[str, ...]:
    if isinstance(scenes, (str, bytes)):
        raise DeploymentViewError("scenes 必须是非空字符串序列")
    normalized = []
    for scene in scenes:
        if (
            not isinstance(scene, str)
            or not scene
            or os.path.basename(scene) != scene
            or scene in (".", "..")
        ):
            raise DeploymentViewError(f"scene 名称非法: {scene!r}")
        if scene in normalized:
            raise DeploymentViewError(f"scene 重复: {scene}")
        normalized.append(scene)
    if not normalized:
        raise DeploymentViewError("scenes 不能为空")
    return tuple(sorted(normalized))


def _write_exclusive(path: str, payload: bytes) -> None:
    try:
        with open(path, "xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise DeploymentViewError(f"输出已存在，拒绝覆盖: {path}") from exc


def _copy_exclusive(source: str, destination: str) -> None:
    source = _require_regular_file(source, "源文件")
    try:
        with open(source, "rb") as source_handle, open(destination, "xb") as dest_handle:
            shutil.copyfileobj(source_handle, dest_handle, length=1024 * 1024)
            dest_handle.flush()
            os.fsync(dest_handle.fileno())
    except FileExistsError as exc:
        raise DeploymentViewError(f"输出已存在，拒绝覆盖: {destination}") from exc


def _materialize_file(source: str, destination: str, link_mode: str) -> None:
    source = _require_regular_file(source, "部署模态源文件")
    if link_mode == "hardlink":
        try:
            os.link(source, destination, follow_symlinks=False)
        except FileExistsError as exc:
            raise DeploymentViewError(f"输出已存在，拒绝覆盖: {destination}") from exc
        except OSError as exc:
            raise DeploymentViewError(
                "创建硬链接失败；跨文件系统时请显式使用 --link_mode copy: "
                f"{source} -> {destination}: {exc}"
            ) from exc
    elif link_mode == "copy":
        _copy_exclusive(source, destination)
    else:
        raise DeploymentViewError("link_mode 必须是 hardlink 或 copy")


def _load_json_file(path: str, label: str) -> Dict[str, object]:
    path = _require_regular_file(path, label)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise DeploymentViewError(f"{label} 无法解析: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DeploymentViewError(f"{label} 必须是 JSON 对象")
    return value


def _validate_content_hash(value: Mapping[str, object], label: str) -> None:
    content_sha256 = value.get("content_sha256")
    payload = {key: item for key, item in value.items() if key != "content_sha256"}
    if not isinstance(content_sha256, str) or sha256_json_value(payload) != content_sha256:
        raise DeploymentViewError(f"{label} content_sha256 不一致")


def _actual_deployment_provenance_paths(
    *,
    source_scene_dir: str,
    calibration_dir: str,
    preprocess_script: str,
) -> Dict[str, str]:
    result = {"preprocess_script": _require_regular_file(preprocess_script, "预处理脚本")}
    for key, filename in CALIBRATION_FILENAMES.items():
        result[key] = _require_regular_file(
            os.path.join(calibration_dir, filename),
            f"部署 provenance {key}",
        )
    result["radar_ir_sync"] = _require_regular_file(
        os.path.join(source_scene_dir, "radar_ir_sync.csv"),
        "部署 provenance radar_ir_sync",
    )
    return result


def _provenance_records(paths: Mapping[str, str]) -> Dict[str, Dict[str, str]]:
    return {
        key: {"name": os.path.basename(path), "sha256": sha256_file(path)}
        for key, path in paths.items()
    }


def _preflight_source_scene(
    *,
    training_dataset_dir: str,
    scene: str,
    calibration_dir: str,
    preprocess_script: str,
) -> Dict[str, object]:
    source_scene_dir = os.path.join(training_dataset_dir, scene)
    manifest = validate_scene_manifest(
        source_scene_dir,
        scene,
        expected_profile="training",
    )
    actual_paths = _actual_deployment_provenance_paths(
        source_scene_dir=source_scene_dir,
        calibration_dir=calibration_dir,
        preprocess_script=preprocess_script,
    )
    actual_records = _provenance_records(actual_paths)
    source_provenance = manifest["preprocessing"]["provenance"]
    expected_records = {
        key: source_provenance[key] for key in PROFILE_PROVENANCE["deployment"]
    }
    if actual_records != expected_records:
        raise DeploymentViewError(
            f"scene={scene} 外部 provenance SHA-256 与 training manifest 不一致"
        )
    return {
        "scene": scene,
        "source_scene_dir": source_scene_dir,
        "source_manifest_path": os.path.join(source_scene_dir, MANIFEST_FILENAME),
        "manifest": manifest,
        "provenance_paths": actual_paths,
    }


def _build_scene_view(
    preflight: Mapping[str, object],
    output_dataset_dir: str,
    link_mode: str,
) -> Dict[str, object]:
    scene = preflight["scene"]
    source_scene_dir = preflight["source_scene_dir"]
    source_manifest_path = preflight["source_manifest_path"]
    source_manifest = preflight["manifest"]
    output_scene_dir = os.path.join(output_dataset_dir, scene)
    os.mkdir(output_scene_dir)
    for modality in DEPLOYMENT_MODALITIES:
        os.mkdir(os.path.join(output_scene_dir, modality))
        for record in source_manifest["modalities"][modality]:
            relative_path = record["path"]
            expected_prefix = modality + "/"
            if (
                not isinstance(relative_path, str)
                or not relative_path.startswith(expected_prefix)
                or os.path.basename(relative_path) != relative_path[len(expected_prefix):]
            ):
                raise DeploymentViewError(
                    f"source manifest {modality} path 非法: {relative_path!r}"
                )
            source = os.path.join(source_scene_dir, relative_path)
            destination = os.path.join(output_scene_dir, relative_path)
            _materialize_file(source, destination, link_mode)
            if sha256_file(destination) != record["sha256"]:
                raise DeploymentViewError(f"部署模态物化后 SHA-256 不一致: {relative_path}")

    source_policy_path = os.path.join(source_scene_dir, POLICY_FILENAME)
    output_policy_path = os.path.join(output_scene_dir, POLICY_FILENAME)
    _copy_exclusive(source_policy_path, output_policy_path)
    snapshot_path = os.path.join(output_scene_dir, SOURCE_TRAINING_MANIFEST_FILENAME)
    _copy_exclusive(source_manifest_path, snapshot_path)
    output_radar_ir_sync_path = os.path.join(output_scene_dir, "radar_ir_sync.csv")
    _copy_exclusive(
        preflight["provenance_paths"]["radar_ir_sync"],
        output_radar_ir_sync_path,
    )

    receipt_payload = {
        "protocol": DEPLOYMENT_VIEW_PROTOCOL,
        "scene": scene,
        "frame_count": source_manifest["frame_count"],
        "voxel_coordinate_frame": source_manifest["voxel_coordinate_frame"],
        "materialization_mode_at_creation": link_mode,
        "source_training_manifest": {
            "schema_version": source_manifest["schema_version"],
            "profile": source_manifest["profile"],
            "content_sha256": source_manifest["content_sha256"],
            "file_sha256": sha256_file(source_manifest_path),
        },
        "preprocess_policy_sha256": source_manifest["preprocessing"]["policy_sha256"],
        "source_provenance": {
            key: source_manifest["preprocessing"]["provenance"][key]
            for key in PROFILE_PROVENANCE["deployment"]
        },
        "modality_records_sha256": {
            modality: sha256_json_value(source_manifest["modalities"][modality])
            for modality in DEPLOYMENT_MODALITIES
        },
    }
    receipt = dict(receipt_payload)
    receipt["content_sha256"] = sha256_json_value(receipt_payload)
    receipt_path = os.path.join(output_scene_dir, DEPLOYMENT_RECEIPT_FILENAME)
    _write_exclusive(receipt_path, _canonical_json_bytes(receipt))

    deployment_provenance = dict(preflight["provenance_paths"])
    deployment_provenance["radar_ir_sync"] = output_radar_ir_sync_path
    deployment_provenance.update(
        {
            "source_training_manifest": snapshot_path,
            "deployment_view_receipt": receipt_path,
        }
    )
    write_scene_manifest_atomic(
        output_scene_dir,
        scene,
        int(source_manifest["frame_count"]),
        deployment_provenance,
        profile="deployment",
    )
    return validate_deployment_view(output_scene_dir, scene)


def build_deployment_dataset(
    *,
    training_dataset_dir: str,
    output_dataset_dir: str,
    scenes: Sequence[str],
    calibration_dir: str,
    preprocess_script: str,
    link_mode: str = "hardlink",
) -> Dict[str, object]:
    """预检全部来源后，在 fresh 根中发布严格 deployment 数据集。"""
    scenes = _normalize_scenes(scenes)
    training_dataset_dir = _require_directory(training_dataset_dir, "training dataset")
    calibration_dir = _require_directory(calibration_dir, "calibration")
    preprocess_script = _require_regular_file(preprocess_script, "预处理脚本")
    output_dataset_dir = os.path.abspath(os.fspath(output_dataset_dir))
    if os.path.lexists(output_dataset_dir):
        raise DeploymentViewError(
            f"deployment 输出已存在，要求 fresh 路径且拒绝覆盖: {output_dataset_dir}"
        )
    _require_directory(
        os.path.dirname(output_dataset_dir),
        "deployment 输出父目录",
    )
    if os.path.commonpath((training_dataset_dir, output_dataset_dir)) == training_dataset_dir:
        raise DeploymentViewError("deployment 输出不能位于 training dataset 内部")
    if link_mode not in ("hardlink", "copy"):
        raise DeploymentViewError("link_mode 必须是 hardlink 或 copy")

    preflights = [
        _preflight_source_scene(
            training_dataset_dir=training_dataset_dir,
            scene=scene,
            calibration_dir=calibration_dir,
            preprocess_script=preprocess_script,
        )
        for scene in scenes
    ]
    os.mkdir(output_dataset_dir)
    scene_results = {
        preflight["scene"]: _build_scene_view(
            preflight,
            output_dataset_dir,
            link_mode,
        )
        for preflight in preflights
    }
    dataset_payload = {
        "protocol": DEPLOYMENT_DATASET_PROTOCOL,
        "scenes": list(scenes),
        "scene_manifest_content_sha256": {
            scene: scene_results[scene]["dataset_manifest_content_sha256"]
            for scene in scenes
        },
    }
    dataset_receipt = dict(dataset_payload)
    dataset_receipt["content_sha256"] = sha256_json_value(dataset_payload)
    _write_exclusive(
        os.path.join(output_dataset_dir, DEPLOYMENT_DATASET_FILENAME),
        _canonical_json_bytes(dataset_receipt),
    )
    return dataset_receipt


def validate_deployment_view(scene_dir: str, scene: str) -> Dict[str, object]:
    """自包含验证 deployment manifest、receipt 与源 training manifest 快照。"""
    scene_dir = _require_directory(scene_dir, "deployment scene")
    manifest = validate_scene_manifest(
        scene_dir,
        scene,
        expected_profile="deployment",
    )
    receipt_path = os.path.join(scene_dir, DEPLOYMENT_RECEIPT_FILENAME)
    snapshot_path = os.path.join(scene_dir, SOURCE_TRAINING_MANIFEST_FILENAME)
    bound_provenance = manifest["preprocessing"]["provenance"]
    if bound_provenance["deployment_view_receipt"]["sha256"] != sha256_file(
        _require_regular_file(receipt_path, "deployment receipt")
    ):
        raise DeploymentViewError("deployment manifest 未绑定当前 deployment receipt")
    if bound_provenance["source_training_manifest"]["sha256"] != sha256_file(
        _require_regular_file(snapshot_path, "source training manifest snapshot")
    ):
        raise DeploymentViewError("deployment manifest 未绑定当前源 training manifest 快照")
    sync_path = _require_regular_file(
        os.path.join(scene_dir, "radar_ir_sync.csv"),
        "deployment radar_ir_sync",
    )
    if bound_provenance["radar_ir_sync"]["sha256"] != sha256_file(sync_path):
        raise DeploymentViewError("deployment radar_ir_sync.csv SHA-256 不一致")
    receipt = _load_json_file(receipt_path, "deployment receipt")
    expected_receipt_keys = {
        "protocol",
        "scene",
        "frame_count",
        "voxel_coordinate_frame",
        "materialization_mode_at_creation",
        "source_training_manifest",
        "preprocess_policy_sha256",
        "source_provenance",
        "modality_records_sha256",
        "content_sha256",
    }
    if set(receipt) != expected_receipt_keys:
        raise DeploymentViewError("deployment receipt 字段不符合协议")
    _validate_content_hash(receipt, "deployment receipt")
    if receipt.get("protocol") != DEPLOYMENT_VIEW_PROTOCOL:
        raise DeploymentViewError("deployment receipt protocol 不一致")
    if receipt.get("materialization_mode_at_creation") not in ("hardlink", "copy"):
        raise DeploymentViewError(
            "deployment receipt materialization_mode_at_creation 无效"
        )

    source_manifest = _load_json_file(snapshot_path, "source training manifest snapshot")
    _validate_content_hash(source_manifest, "source training manifest snapshot")
    if source_manifest.get("schema_version") != SCHEMA_VERSION_V2:
        raise DeploymentViewError("source training manifest 必须是 schema v2")
    if source_manifest.get("profile") != "training":
        raise DeploymentViewError("source training manifest 必须是 training profile")
    if set(source_manifest.get("modalities", {})) != {
        "radar_voxel",
        "lidar_voxel",
        "target_voxel",
        "observed_mask",
        "ir_image",
    }:
        raise DeploymentViewError("source training manifest 模态字段不完整")

    source_identity = {
        "schema_version": source_manifest["schema_version"],
        "profile": source_manifest["profile"],
        "content_sha256": source_manifest["content_sha256"],
        "file_sha256": sha256_file(snapshot_path),
    }
    if receipt.get("source_training_manifest") != source_identity:
        raise DeploymentViewError("deployment receipt 与源 training manifest 身份不一致")
    if receipt.get("scene") != scene or source_manifest.get("scene") != scene:
        raise DeploymentViewError("deployment view scene 身份不一致")
    if receipt.get("frame_count") != manifest.get("frame_count") or receipt.get(
        "frame_count"
    ) != source_manifest.get("frame_count"):
        raise DeploymentViewError("deployment view frame_count 不一致")
    if receipt.get("voxel_coordinate_frame") != manifest.get(
        "voxel_coordinate_frame"
    ) or receipt.get("voxel_coordinate_frame") != source_manifest.get(
        "voxel_coordinate_frame"
    ):
        raise DeploymentViewError("deployment view voxel frame 不一致")

    for modality in DEPLOYMENT_MODALITIES:
        if manifest["modalities"][modality] != source_manifest["modalities"][modality]:
            raise DeploymentViewError(f"deployment {modality} 与源 training manifest 不一致")
        expected_records_hash = sha256_json_value(manifest["modalities"][modality])
        if receipt["modality_records_sha256"].get(modality) != expected_records_hash:
            raise DeploymentViewError(f"deployment receipt {modality} 记录 hash 不一致")
    if set(receipt["modality_records_sha256"]) != set(DEPLOYMENT_MODALITIES):
        raise DeploymentViewError("deployment receipt 模态记录字段不完整")

    source_preprocessing = source_manifest["preprocessing"]
    deployment_preprocessing = manifest["preprocessing"]
    policy_sha256 = deployment_preprocessing["policy_sha256"]
    if (
        receipt.get("preprocess_policy_sha256") != policy_sha256
        or source_preprocessing.get("policy_sha256") != policy_sha256
    ):
        raise DeploymentViewError("deployment preprocess policy 身份不一致")
    shared_source_provenance = {
        key: source_preprocessing["provenance"][key]
        for key in PROFILE_PROVENANCE["deployment"]
    }
    shared_deployment_provenance = {
        key: deployment_preprocessing["provenance"][key]
        for key in PROFILE_PROVENANCE["deployment"]
    }
    if (
        receipt.get("source_provenance") != shared_source_provenance
        or shared_deployment_provenance != shared_source_provenance
    ):
        raise DeploymentViewError("deployment provenance 与源 training manifest 不一致")
    return {
        "scene": scene,
        "frame_count": manifest["frame_count"],
        "dataset_manifest_content_sha256": manifest["content_sha256"],
        "source_training_manifest_content_sha256": source_manifest[
            "content_sha256"
        ],
        "deployment_receipt_sha256": sha256_file(receipt_path),
        "radar_ir_sync_sha256": shared_deployment_provenance["radar_ir_sync"][
            "sha256"
        ],
        "calibration_sha256": {
            key: shared_deployment_provenance[key]["sha256"]
            for key in ("lidar_to_thermal", "thermal_intrinsics")
        },
        "voxel_coordinate_frame": manifest["voxel_coordinate_frame"],
        "materialization_mode_at_creation": receipt[
            "materialization_mode_at_creation"
        ],
    }


def validate_deployment_dataset(
    dataset_dir: str,
    *,
    scenes: Sequence[str],
) -> Dict[str, object]:
    """验证数据集根收据、精确场景集合及每个严格 deployment view。"""
    scenes = _normalize_scenes(scenes)
    dataset_dir = _require_directory(dataset_dir, "deployment dataset")
    with os.scandir(dataset_dir) as iterator:
        actual_entries = {entry.name for entry in iterator}
    expected_entries = set(scenes) | {DEPLOYMENT_DATASET_FILENAME}
    if actual_entries != expected_entries:
        raise DeploymentViewError(
            "deployment dataset 场景集合或根目录项不一致: "
            f"extra={sorted(actual_entries - expected_entries)}, "
            f"missing={sorted(expected_entries - actual_entries)}"
        )
    receipt_path = os.path.join(dataset_dir, DEPLOYMENT_DATASET_FILENAME)
    receipt = _load_json_file(receipt_path, "deployment dataset receipt")
    if set(receipt) != {
        "protocol",
        "scenes",
        "scene_manifest_content_sha256",
        "content_sha256",
    }:
        raise DeploymentViewError("deployment dataset receipt 字段不符合协议")
    _validate_content_hash(receipt, "deployment dataset receipt")
    if receipt.get("protocol") != DEPLOYMENT_DATASET_PROTOCOL:
        raise DeploymentViewError("deployment dataset protocol 不一致")
    if receipt.get("scenes") != list(scenes):
        raise DeploymentViewError("deployment dataset scenes 与入口不一致")
    scene_results = {
        scene: validate_deployment_view(os.path.join(dataset_dir, scene), scene)
        for scene in scenes
    }
    expected_manifest_hashes = {
        scene: scene_results[scene]["dataset_manifest_content_sha256"]
        for scene in scenes
    }
    if receipt.get("scene_manifest_content_sha256") != expected_manifest_hashes:
        raise DeploymentViewError("deployment dataset receipt 场景 manifest 身份不一致")
    return {
        "protocol": receipt["protocol"],
        "scenes": list(scenes),
        "content_sha256": receipt["content_sha256"],
        "scene_results": scene_results,
    }
