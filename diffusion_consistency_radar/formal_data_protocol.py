# -*- coding: utf-8 -*-
"""文件功能：从正式数据、切分和监督清单派生 checkpoint 数据身份 artifact。"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from typing import Dict, Tuple

from diffusion_consistency_radar.checkpoint_chain import (
    FORMAL_DATA_PROTOCOL,
    validate_checkpoint_data_protocol,
)
from diffusion_consistency_radar.dataset_manifest import (
    sha256_file,
    sha256_json_value,
    validate_scene_manifest,
)
from diffusion_consistency_radar.observed_mask import OBSERVED_MASK_PROTOCOL
from diffusion_consistency_radar.temporal_split import (
    load_temporal_split_artifact,
)


class FormalDataProtocolError(ValueError):
    """表示 formal data protocol 无法从当前数据唯一重建。"""


def _validate_scenes(scenes: Sequence[str]) -> list[str]:
    if (
        not isinstance(scenes, (list, tuple))
        or not scenes
        or any(
            not isinstance(scene, str)
            or not scene
            or os.path.basename(scene) != scene
            for scene in scenes
        )
    ):
        raise FormalDataProtocolError("scenes 必须是非空普通场景名数组")
    result = list(scenes)
    if len(set(result)) != len(result):
        raise FormalDataProtocolError("scenes 不得重复")
    return result


def _load_policy(scene_dir: str) -> Mapping[str, object]:
    policy_path = os.path.join(scene_dir, "preprocess_policy.json")
    if os.path.islink(policy_path) or not os.path.isfile(policy_path):
        raise FormalDataProtocolError(f"preprocess policy 必须是普通文件: {policy_path}")
    try:
        with open(policy_path, "r", encoding="utf-8") as handle:
            policy = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalDataProtocolError(f"preprocess policy 无法解析: {exc}") from exc
    if not isinstance(policy, Mapping):
        raise FormalDataProtocolError("preprocess policy 必须是 JSON 对象")
    return policy


def build_formal_data_protocol(
    *,
    dataset_dir: str,
    scenes: Sequence[str],
    split_artifact_path: str,
) -> Dict[str, object]:
    """从当前 training manifests 与唯一 split 构建完整 formal_data_v2。"""
    selected_scenes = _validate_scenes(scenes)
    dataset_root = os.path.abspath(os.fspath(dataset_dir))
    if os.path.islink(dataset_root) or not os.path.isdir(dataset_root):
        raise FormalDataProtocolError(f"dataset_dir 必须是普通目录: {dataset_root}")
    _split, split_sha256 = load_temporal_split_artifact(
        split_artifact_path,
        dataset_dir=dataset_root,
        expected_scenes=selected_scenes,
        require_formal=True,
    )

    manifest_hashes: Dict[str, str] = {}
    target_policy_hashes: Dict[str, str] = {}
    observed_mask_hashes: Dict[str, str] = {}
    radar_ir_sync_hashes: Dict[str, str] = {}
    shared_calibration = None
    for scene in selected_scenes:
        scene_dir = os.path.join(dataset_root, scene)
        manifest = validate_scene_manifest(
            scene_dir,
            scene,
            expected_profile="training",
        )
        manifest_hashes[scene] = manifest["content_sha256"]
        preprocessing = manifest["preprocessing"]
        provenance = preprocessing["provenance"]
        target_policy_hashes[scene] = provenance["target_policy"]["sha256"]
        radar_ir_sync_hashes[scene] = provenance["radar_ir_sync"]["sha256"]
        observed_records = manifest["modalities"].get("observed_mask")
        if not isinstance(observed_records, list) or not observed_records:
            raise FormalDataProtocolError(f"场景 {scene!r} 缺少 observed mask records")
        observed_mask_hashes[scene] = sha256_json_value(observed_records)
        policy = _load_policy(scene_dir)
        if policy.get("observed_mask_protocol") != OBSERVED_MASK_PROTOCOL:
            raise FormalDataProtocolError(
                f"场景 {scene!r} observed mask protocol 不匹配"
            )
        calibration = {
            "lidar_to_thermal": provenance["lidar_to_thermal"]["sha256"],
            "thermal_intrinsics": provenance["thermal_intrinsics"]["sha256"],
        }
        if shared_calibration is None:
            shared_calibration = calibration
        elif calibration != shared_calibration:
            raise FormalDataProtocolError("训练场景的 IR 标定 SHA-256 不一致")

    protocol: Dict[str, object] = {
        "protocol": FORMAL_DATA_PROTOCOL,
        "dataset_manifest_sha256": manifest_hashes,
        "split_artifact_sha256": split_sha256,
        "target_policy_sha256": target_policy_hashes,
        "observed_mask_sha256": observed_mask_hashes,
        "observed_mask_protocol": OBSERVED_MASK_PROTOCOL,
        "calibration_sha256": shared_calibration,
        "radar_ir_sync_sha256": radar_ir_sync_hashes,
    }
    # 使用最严格的 LDM schema 验证，使同一对象可供 VAE/LDM/CD 继承。
    validated = validate_checkpoint_data_protocol(protocol, stage="ldm")
    if validated is None:
        raise FormalDataProtocolError("formal data protocol 校验未返回对象")
    return validated


def _write_json_immutable(path: str, value: Mapping[str, object]) -> str:
    output_path = os.path.abspath(os.fspath(path))
    if os.path.lexists(output_path):
        raise FormalDataProtocolError(
            f"formal data protocol 输出已存在，拒绝覆盖: {output_path}"
        )
    parent = os.path.dirname(output_path) or os.curdir
    os.makedirs(parent, exist_ok=True)
    descriptor, temp_path = tempfile.mkstemp(
        dir=parent,
        prefix=".formal_data_protocol.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp_path, output_path)
        except FileExistsError as exc:
            raise FormalDataProtocolError(
                f"formal data protocol 输出已存在，拒绝覆盖: {output_path}"
            ) from exc
    finally:
        if os.path.lexists(temp_path):
            os.unlink(temp_path)
    return output_path


def build_and_write_formal_data_protocol(
    *,
    dataset_dir: str,
    scenes: Sequence[str],
    split_artifact_path: str,
    output_path: str,
) -> str:
    """构建并不可覆盖地发布 formal data protocol。"""
    if os.path.lexists(os.path.abspath(os.fspath(output_path))):
        raise FormalDataProtocolError(
            f"formal data protocol 输出已存在，拒绝覆盖: {output_path}"
        )
    protocol = build_formal_data_protocol(
        dataset_dir=dataset_dir,
        scenes=scenes,
        split_artifact_path=split_artifact_path,
    )
    return _write_json_immutable(output_path, protocol)


def load_formal_data_protocol_artifact(
    path: str,
    *,
    dataset_dir: str,
    scenes: Sequence[str],
    split_artifact_path: str,
    stage: str,
) -> Tuple[Dict[str, object], str]:
    """加载 artifact，并从当前 dataset/split 重建全文进行交叉比对。"""
    artifact_path = os.path.abspath(os.fspath(path))
    if os.path.islink(artifact_path) or not os.path.isfile(artifact_path):
        raise FormalDataProtocolError(
            f"formal data protocol 必须是普通文件: {artifact_path}"
        )
    try:
        with open(artifact_path, "r", encoding="utf-8") as handle:
            protocol = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalDataProtocolError(f"formal data protocol 无法解析: {exc}") from exc
    validated = validate_checkpoint_data_protocol(protocol, stage=stage)
    expected = build_formal_data_protocol(
        dataset_dir=dataset_dir,
        scenes=scenes,
        split_artifact_path=split_artifact_path,
    )
    if validated != expected:
        raise FormalDataProtocolError(
            "formal data protocol 与当前 dataset/split 派生身份不一致"
        )
    return validated, sha256_file(artifact_path)
