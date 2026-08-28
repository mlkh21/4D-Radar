# -*- coding: utf-8 -*-
"""文件功能：生成并验证内容寻址的连续时间切分与 purge artifact。"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from typing import Dict, Tuple

from diffusion_consistency_radar.dataset_manifest import (
    sha256_file,
    validate_scene_manifest,
)


TEMPORAL_SPLIT_PROTOCOL = "temporal_split_v1"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_TOP_LEVEL_KEYS = {
    "protocol",
    "formal",
    "train_fraction",
    "purge_seconds",
    "ordering",
    "scenes",
    "content_sha256",
}
_SCENE_KEYS = {
    "dataset_manifest_sha256",
    "frame_count",
    "train_frame_ids",
    "purged_frame_ids",
    "validation_frame_ids",
    "train_last_timestamp",
    "validation_first_timestamp",
    "actual_gap_seconds",
}


class TemporalSplitError(ValueError):
    """表示 temporal split 输入、artifact 或数据绑定不满足正式协议。"""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _content_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


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
        raise TemporalSplitError("scenes 必须是非空普通场景名数组")
    result = list(scenes)
    if len(set(result)) != len(result):
        raise TemporalSplitError("scenes 不得重复")
    return result


def _validate_split_parameters(
    train_fraction: float,
    purge_seconds: float,
    formal: bool,
) -> Tuple[float, float, bool]:
    if type(formal) is not bool:
        raise TemporalSplitError("formal 必须是 bool")
    if isinstance(train_fraction, bool):
        raise TemporalSplitError("train_fraction 必须位于 (0,1)")
    try:
        fraction = float(train_fraction)
        purge = float(purge_seconds)
    except (TypeError, ValueError) as exc:
        raise TemporalSplitError("train_fraction/purge_seconds 必须是有限数") from exc
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise TemporalSplitError("train_fraction 必须位于 (0,1)")
    if not math.isfinite(purge) or purge < 0.0:
        raise TemporalSplitError("purge_seconds 必须是非负有限数")
    if formal and purge <= 0.0:
        raise TemporalSplitError("formal temporal split 的 purge_seconds 必须大于 0")
    return fraction, purge, formal


def _read_scene_timestamps(
    scene_dir: str,
    manifest: Mapping[str, object],
) -> Tuple[list[str], list[float]]:
    modalities = manifest.get("modalities")
    if not isinstance(modalities, Mapping):
        raise TemporalSplitError("training manifest 缺少 modalities")
    radar_records = modalities.get("radar_voxel")
    if not isinstance(radar_records, list) or not radar_records:
        raise TemporalSplitError("training manifest 缺少 Radar frame records")
    frame_ids = [record.get("frame_id") for record in radar_records]
    if any(not isinstance(frame_id, str) for frame_id in frame_ids):
        raise TemporalSplitError("training manifest Radar frame_id 无效")

    preprocessing = manifest.get("preprocessing")
    provenance = (
        preprocessing.get("provenance")
        if isinstance(preprocessing, Mapping)
        else None
    )
    sync_record = provenance.get("radar_ir_sync") if isinstance(provenance, Mapping) else None
    if not isinstance(sync_record, Mapping):
        raise TemporalSplitError("training manifest 缺少 radar_ir_sync provenance")
    sync_name = sync_record.get("name")
    sync_digest = sync_record.get("sha256")
    if (
        not isinstance(sync_name, str)
        or not sync_name
        or os.path.basename(sync_name) != sync_name
    ):
        raise TemporalSplitError("radar_ir_sync provenance name 无效")
    sync_path = os.path.join(scene_dir, sync_name)
    if os.path.islink(sync_path) or not os.path.isfile(sync_path):
        raise TemporalSplitError(f"radar_ir_sync 必须是普通文件: {sync_path}")
    if sha256_file(sync_path) != sync_digest:
        raise TemporalSplitError("radar_ir_sync 文件与 training manifest SHA-256 不一致")

    timestamps: list[float] = []
    seen_ids: list[str] = []
    try:
        with open(sync_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not {
                "frame_index",
                "radar_timestamp",
            }.issubset(reader.fieldnames):
                raise TemporalSplitError("radar_ir_sync 缺少 frame_index/radar_timestamp")
            for expected_index, row in enumerate(reader):
                try:
                    frame_index = int(row["frame_index"])
                    timestamp = float(row["radar_timestamp"])
                except (TypeError, ValueError) as exc:
                    raise TemporalSplitError("radar_ir_sync 含非法索引或时间戳") from exc
                if frame_index != expected_index:
                    raise TemporalSplitError("radar_ir_sync frame_index 必须从 0 严格连续")
                if not math.isfinite(timestamp):
                    raise TemporalSplitError("radar_ir_sync timestamp 必须有限")
                seen_ids.append(f"{frame_index:06d}")
                timestamps.append(timestamp)
    except OSError as exc:
        raise TemporalSplitError(f"无法读取 radar_ir_sync: {exc}") from exc
    if seen_ids != frame_ids:
        raise TemporalSplitError("radar_ir_sync frame 集合与 training manifest 不一致")
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise TemporalSplitError("radar_timestamp 必须严格递增")
    return frame_ids, timestamps


def _build_scene_partition(
    frame_ids: Sequence[str],
    timestamps: Sequence[float],
    *,
    manifest_sha256: str,
    train_fraction: float,
    purge_seconds: float,
) -> Dict[str, object]:
    if len(frame_ids) != len(timestamps) or len(frame_ids) < 2:
        raise TemporalSplitError("每个场景至少需要两个带时间戳的 frame")
    train_count = int(math.floor(len(frame_ids) * train_fraction))
    train_count = min(max(train_count, 1), len(frame_ids) - 1)
    train_last_timestamp = float(timestamps[train_count - 1])
    validation_start = train_count
    while (
        validation_start < len(frame_ids)
        and float(timestamps[validation_start]) - train_last_timestamp < purge_seconds
    ):
        validation_start += 1
    if validation_start >= len(frame_ids):
        raise TemporalSplitError(
            "purge 后没有 validation frame；请调整 train_fraction 或 purge_seconds"
        )
    validation_first_timestamp = float(timestamps[validation_start])
    return {
        "dataset_manifest_sha256": manifest_sha256,
        "frame_count": len(frame_ids),
        "train_frame_ids": list(frame_ids[:train_count]),
        "purged_frame_ids": list(frame_ids[train_count:validation_start]),
        "validation_frame_ids": list(frame_ids[validation_start:]),
        "train_last_timestamp": train_last_timestamp,
        "validation_first_timestamp": validation_first_timestamp,
        "actual_gap_seconds": validation_first_timestamp - train_last_timestamp,
    }


def build_temporal_split_artifact(
    *,
    dataset_dir: str,
    scenes: Sequence[str],
    train_fraction: float,
    purge_seconds: float,
    formal: bool,
) -> Dict[str, object]:
    """从严格 training manifest 和其 Radar 时间轴构造唯一切分。"""
    selected_scenes = _validate_scenes(scenes)
    fraction, purge, formal = _validate_split_parameters(
        train_fraction,
        purge_seconds,
        formal,
    )
    dataset_root = os.path.abspath(os.fspath(dataset_dir))
    if os.path.islink(dataset_root) or not os.path.isdir(dataset_root):
        raise TemporalSplitError(f"dataset_dir 必须是普通目录: {dataset_root}")
    scene_partitions: Dict[str, Dict[str, object]] = {}
    for scene in selected_scenes:
        scene_dir = os.path.join(dataset_root, scene)
        manifest = validate_scene_manifest(
            scene_dir,
            scene,
            expected_profile="training",
        )
        manifest_sha256 = manifest.get("content_sha256")
        if (
            not isinstance(manifest_sha256, str)
            or _SHA256_PATTERN.fullmatch(manifest_sha256) is None
        ):
            raise TemporalSplitError(f"场景 {scene!r} manifest content SHA-256 无效")
        frame_ids, timestamps = _read_scene_timestamps(scene_dir, manifest)
        scene_partitions[scene] = _build_scene_partition(
            frame_ids,
            timestamps,
            manifest_sha256=manifest_sha256,
            train_fraction=fraction,
            purge_seconds=purge,
        )
    payload: Dict[str, object] = {
        "protocol": TEMPORAL_SPLIT_PROTOCOL,
        "formal": formal,
        "train_fraction": fraction,
        "purge_seconds": purge,
        "ordering": "scene_radar_timestamp_ascending_contiguous_blocks",
        "scenes": scene_partitions,
    }
    artifact = dict(payload)
    artifact["content_sha256"] = _content_sha256(payload)
    return artifact


def _write_json_immutable(path: str, artifact: Mapping[str, object]) -> str:
    output_path = os.path.abspath(os.fspath(path))
    if os.path.lexists(output_path):
        raise TemporalSplitError(f"temporal split 输出已存在，拒绝覆盖: {output_path}")
    parent = os.path.dirname(output_path) or os.curdir
    os.makedirs(parent, exist_ok=True)
    descriptor, temp_path = tempfile.mkstemp(
        dir=parent,
        prefix=".temporal_split.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                artifact,
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
            raise TemporalSplitError(
                f"temporal split 输出已存在，拒绝覆盖: {output_path}"
            ) from exc
    finally:
        if os.path.lexists(temp_path):
            os.unlink(temp_path)
    return output_path


def build_and_write_temporal_split(
    *,
    dataset_dir: str,
    scenes: Sequence[str],
    output_path: str,
    train_fraction: float,
    purge_seconds: float,
    formal: bool = True,
) -> str:
    """构建并不可覆盖地发布 temporal split artifact。"""
    if os.path.lexists(os.path.abspath(os.fspath(output_path))):
        raise TemporalSplitError(f"temporal split 输出已存在，拒绝覆盖: {output_path}")
    artifact = build_temporal_split_artifact(
        dataset_dir=dataset_dir,
        scenes=scenes,
        train_fraction=train_fraction,
        purge_seconds=purge_seconds,
        formal=formal,
    )
    return _write_json_immutable(output_path, artifact)


def load_temporal_split_artifact(
    path: str,
    *,
    dataset_dir: str,
    expected_scenes: Sequence[str],
    require_formal: bool = True,
) -> Tuple[Dict[str, object], str]:
    """加载 split，随后依据当前 dataset 重建并比对完整内容。"""
    artifact_path = os.path.abspath(os.fspath(path))
    if os.path.islink(artifact_path) or not os.path.isfile(artifact_path):
        raise TemporalSplitError(f"temporal split 必须是普通文件: {artifact_path}")
    try:
        with open(artifact_path, "r", encoding="utf-8") as handle:
            artifact = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise TemporalSplitError(f"temporal split 无法解析: {artifact_path}: {exc}") from exc
    if not isinstance(artifact, dict) or set(artifact) != _TOP_LEVEL_KEYS:
        raise TemporalSplitError("temporal split 顶层字段不符合协议")
    if artifact.get("protocol") != TEMPORAL_SPLIT_PROTOCOL:
        raise TemporalSplitError("temporal split protocol 不匹配")
    formal = artifact.get("formal")
    if type(formal) is not bool:
        raise TemporalSplitError("temporal split formal 必须是 bool")
    if require_formal and not formal:
        raise TemporalSplitError("正式入口拒绝 formal=false temporal split")
    scenes = _validate_scenes(expected_scenes)
    scene_records = artifact.get("scenes")
    if not isinstance(scene_records, Mapping) or set(scene_records) != set(scenes):
        raise TemporalSplitError("temporal split 场景集合与入口不一致")
    for scene, record in scene_records.items():
        if not isinstance(record, Mapping) or set(record) != _SCENE_KEYS:
            raise TemporalSplitError(f"场景 {scene!r} split 字段不符合协议")
    payload = {key: value for key, value in artifact.items() if key != "content_sha256"}
    recorded_content = artifact.get("content_sha256")
    if (
        not isinstance(recorded_content, str)
        or _content_sha256(payload) != recorded_content
    ):
        raise TemporalSplitError("temporal split content_sha256 不一致")
    expected = build_temporal_split_artifact(
        dataset_dir=dataset_dir,
        scenes=scenes,
        train_fraction=artifact.get("train_fraction"),
        purge_seconds=artifact.get("purge_seconds"),
        formal=formal,
    )
    if artifact != expected:
        raise TemporalSplitError("temporal split 与当前 dataset/time axis 不一致")
    return artifact, sha256_file(artifact_path)


def split_frame_ids_by_scene(
    artifact: Mapping[str, object],
    split: str,
) -> Dict[str, list[str]]:
    """从已验证 artifact 提取 train/validation/purged frame ID 映射。"""
    field_by_split = {
        "train": "train_frame_ids",
        "validation": "validation_frame_ids",
        "purged": "purged_frame_ids",
    }
    if split not in field_by_split:
        raise TemporalSplitError(f"split 必须为 {sorted(field_by_split)}")
    scenes = artifact.get("scenes")
    if not isinstance(scenes, Mapping) or not scenes:
        raise TemporalSplitError("temporal split 缺少 scenes")
    field = field_by_split[split]
    result: Dict[str, list[str]] = {}
    for scene, record in scenes.items():
        values = record.get(field) if isinstance(record, Mapping) else None
        if not isinstance(values, list) or any(
            not isinstance(value, str) for value in values
        ):
            raise TemporalSplitError(f"场景 {scene!r} 的 {field} 无效")
        result[str(scene)] = list(values)
    return result


def limit_frame_ids_by_scene(
    frame_ids_by_scene: Mapping[str, Sequence[str]],
    frames_per_scene: int,
    *,
    partition: str,
) -> Dict[str, list[str]]:
    """按 split artifact 的既有顺序确定性截取 formal mini 帧。"""
    if (
        not isinstance(partition, str)
        or partition not in {"train", "validation"}
    ):
        raise TemporalSplitError("partition 必须为 train 或 validation")
    if type(frames_per_scene) is not int or frames_per_scene <= 0:
        raise TemporalSplitError("frames_per_scene 必须是正整数")
    if not isinstance(frame_ids_by_scene, Mapping) or not frame_ids_by_scene:
        raise TemporalSplitError("frame_ids_by_scene 必须是非空场景映射")

    limited: Dict[str, list[str]] = {}
    for scene, frame_ids in frame_ids_by_scene.items():
        if (
            not isinstance(scene, str)
            or not scene
            or os.path.basename(scene) != scene
            or not isinstance(frame_ids, (list, tuple))
            or any(not isinstance(frame_id, str) for frame_id in frame_ids)
        ):
            raise TemporalSplitError("frame_ids_by_scene 场景或 frame ID 无效")
        if len(frame_ids) < frames_per_scene:
            raise TemporalSplitError(
                f"{partition} 场景 {scene!r} 只有 {len(frame_ids)} 帧，"
                f"无法选择 {frames_per_scene} 帧"
            )
        limited[scene] = list(frame_ids[:frames_per_scene])
    return limited
