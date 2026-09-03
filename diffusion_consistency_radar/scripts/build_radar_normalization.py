#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从显式训练场景生成冻结的 Radar normalization artifact。"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from typing import Sequence

import numpy as np
import torch


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from diffusion_consistency_radar.cm.dataset_loader import (  # noqa: E402
    crop_voxel_channels_to_pc_range,
    resize_radar_voxel_channels,
)
from diffusion_consistency_radar.dataset_manifest import (  # noqa: E402
    validate_scene_manifest,
)
from diffusion_consistency_radar.radar_normalization import (  # noqa: E402
    RADAR_NORMALIZATION_PROTOCOL,
    RADAR_NORMALIZATION_PROTOCOL_V2,
    RadarNormalizationError,
    validate_radar_normalization_spec,
)
from diffusion_consistency_radar.radar_statistics import (  # noqa: E402
    RADAR_RESIZE_AGGREGATION,
    RADAR_RESIZE_AGGREGATION_V1,
    RADAR_STATISTICS_PROTOCOL,
    load_sparse_radar_voxel,
    load_sparse_radar_voxel_with_statistics,
)
from diffusion_consistency_radar.temporal_split import (  # noqa: E402
    load_temporal_split_artifact,
    split_frame_ids_by_scene,
)


def _positive_scale(value: float) -> float:
    if isinstance(value, bool):
        raise RadarNormalizationError("Doppler scale/量程必须是正有限数")
    try:
        scale = float(value)
    except (TypeError, ValueError) as exc:
        raise RadarNormalizationError("Doppler scale/量程必须是正有限数") from exc
    if not math.isfinite(scale) or scale <= 0.0:
        raise RadarNormalizationError("Doppler scale/量程必须是正有限数")
    return scale


def _validate_scenes(scenes: Sequence[str]) -> list[str]:
    if (
        not isinstance(scenes, (list, tuple))
        or not scenes
        or any(not isinstance(scene, str) or not scene for scene in scenes)
    ):
        raise RadarNormalizationError("--scene 必须至少提供一个非空训练场景")
    result = list(scenes)
    if len(set(result)) != len(result):
        raise RadarNormalizationError("训练场景不得重复")
    return result


def _radar_frame_paths(scene_dir: str) -> list[str]:
    radar_dir = os.path.join(scene_dir, "radar_voxel")
    if not os.path.isdir(radar_dir):
        raise RadarNormalizationError(f"Radar voxel 目录不存在: {radar_dir}")
    paths = [
        os.path.join(radar_dir, name)
        for name in sorted(os.listdir(radar_dir))
        if name.endswith((".npy", ".npz"))
    ]
    if not paths:
        raise RadarNormalizationError(f"场景没有 Radar voxel 帧: {scene_dir}")
    return paths


def _load_radar_tensor(path: str):
    resize_statistics = None
    resize_aggregation = RADAR_RESIZE_AGGREGATION_V1
    if path.endswith(".npz"):
        voxel, summary = load_sparse_radar_voxel(path)
        if summary is not None and summary["protocol"] == RADAR_STATISTICS_PROTOCOL:
            voxel, fields, _summary = load_sparse_radar_voxel_with_statistics(path)
            resize_statistics = {"protocol": fields["protocol"]}
            for name in (
                "point_count",
                "intensity_valid_count",
                "doppler_valid_count",
            ):
                resize_statistics[name] = torch.from_numpy(
                    np.asarray(fields[name])
                ).permute(2, 0, 1)
            resize_aggregation = RADAR_RESIZE_AGGREGATION
    else:
        voxel = np.load(path).astype(np.float32)
    if voxel.ndim != 4 or voxel.shape[-1] != 4:
        raise RadarNormalizationError(
            f"Radar voxel 必须是 (X,Y,Z,4): {path}: shape={voxel.shape}"
        )
    if not np.isfinite(voxel).all():
        raise RadarNormalizationError(f"Radar voxel 包含非有限值: {path}")
    return (
        torch.from_numpy(voxel).permute(3, 2, 0, 1),
        resize_statistics,
        resize_aggregation,
    )


def _preflight_output(output_path: str) -> str:
    path = os.path.abspath(os.fspath(output_path))
    if os.path.lexists(path):
        if os.path.islink(path):
            raise RadarNormalizationError(
                f"normalization 输出拒绝符号链接/symlink: {path}"
            )
        raise RadarNormalizationError(f"normalization 输出已存在，拒绝覆盖: {path}")
    return path


def _write_json_atomic(output_path: str, artifact: dict) -> None:
    parent = os.path.dirname(output_path) or os.curdir
    os.makedirs(parent, exist_ok=True)
    if os.path.lexists(output_path):
        raise RadarNormalizationError(
            f"normalization 输出在发布前已存在，拒绝覆盖: {output_path}"
        )
    descriptor, temp_path = tempfile.mkstemp(
        dir=parent,
        prefix=".radar_normalization.",
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
        if os.path.lexists(output_path):
            raise RadarNormalizationError(
                f"normalization 输出在发布前已存在，拒绝覆盖: {output_path}"
            )
        os.replace(temp_path, output_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def build_and_write_artifact(
    *,
    dataset_dir: str,
    scenes: Sequence[str],
    output_path: str,
    target_size: Sequence[int],
    source_pc_range: Sequence[float],
    model_pc_range: Sequence[float],
    doppler_scale_mps: float,
    intensity_transform: str = "log1p_robust_zscore",
    intensity_quantity: str = "intensity",
    intensity_unit: str = "sensor_native_linear_nonnegative",
    max_frames: int = 0,
    split_artifact_path: str = "",
) -> str:
    """验证显式训练场景，统计 occupied intensity 并原子发布 artifact。"""
    output_path = _preflight_output(output_path)
    scale_mps = _positive_scale(doppler_scale_mps)
    if intensity_transform == "log1p_robust_zscore":
        if (
            intensity_quantity != "intensity"
            or intensity_unit != "sensor_native_linear_nonnegative"
        ):
            raise RadarNormalizationError(
                "log1p intensity 必须显式声明为 sensor-native linear intensity"
            )
        normalization_protocol = RADAR_NORMALIZATION_PROTOCOL
    elif intensity_transform == "identity_robust_zscore":
        if (
            intensity_quantity != "signal_to_noise_ratio"
            or intensity_unit != "dB"
        ):
            raise RadarNormalizationError(
                "identity intensity 必须显式声明为 dB signal_to_noise_ratio"
            )
        normalization_protocol = RADAR_NORMALIZATION_PROTOCOL_V2
    else:
        raise RadarNormalizationError(
            f"intensity_transform 不支持: {intensity_transform!r}"
        )
    selected_scenes = _validate_scenes(scenes)
    if type(max_frames) is not int or max_frames < 0:
        raise RadarNormalizationError("max_frames 必须是非负整数")

    dataset_root = os.path.abspath(os.fspath(dataset_dir))
    if not os.path.isdir(dataset_root):
        raise RadarNormalizationError(f"dataset_dir 不存在或不是目录: {dataset_root}")

    split_artifact_sha256 = ""
    train_frame_ids_by_scene = None
    if split_artifact_path:
        split_artifact, split_artifact_sha256 = load_temporal_split_artifact(
            split_artifact_path,
            dataset_dir=dataset_root,
            expected_scenes=selected_scenes,
            require_formal=True,
        )
        train_frame_ids_by_scene = split_frame_ids_by_scene(
            split_artifact,
            "train",
        )

    manifest_hashes = {}
    intensity_chunks = []
    total_frames = 0
    resize_aggregations = set()
    for scene in selected_scenes:
        scene_dir = os.path.join(dataset_root, scene)
        manifest = validate_scene_manifest(
            scene_dir,
            scene,
            expected_profile="training" if train_frame_ids_by_scene is not None else None,
        )
        if not isinstance(manifest, dict):
            raise RadarNormalizationError(f"场景 {scene!r} manifest 返回值无效")
        manifest_hashes[scene] = manifest.get("content_sha256")
        frame_paths = _radar_frame_paths(scene_dir)
        manifest_frame_count = manifest.get("frame_count")
        if type(manifest_frame_count) is not int or manifest_frame_count != len(frame_paths):
            raise RadarNormalizationError(
                f"场景 {scene!r} Radar 帧数与 manifest 不一致: "
                f"manifest={manifest_frame_count!r}, radar={len(frame_paths)}"
            )
        if train_frame_ids_by_scene is not None:
            paths_by_frame = {
                os.path.splitext(os.path.basename(path))[0]: path
                for path in frame_paths
            }
            requested_ids = train_frame_ids_by_scene[scene]
            missing_ids = [frame_id for frame_id in requested_ids if frame_id not in paths_by_frame]
            if missing_ids:
                raise RadarNormalizationError(
                    f"场景 {scene!r} split train frame 缺失: {missing_ids}"
                )
            selected_paths = [paths_by_frame[frame_id] for frame_id in requested_ids]
        else:
            selected_paths = frame_paths
        if max_frames > 0:
            selected_paths = selected_paths[:max_frames]
        for path in selected_paths:
            radar, resize_statistics, resize_aggregation = _load_radar_tensor(path)
            radar = crop_voxel_channels_to_pc_range(
                radar,
                source_pc_range,
                model_pc_range,
            )
            if resize_statistics is not None:
                cropped_statistics = {"protocol": resize_statistics["protocol"]}
                for name in (
                    "point_count",
                    "intensity_valid_count",
                    "doppler_valid_count",
                ):
                    count = crop_voxel_channels_to_pc_range(
                        resize_statistics[name].unsqueeze(0),
                        source_pc_range,
                        model_pc_range,
                    )
                    cropped_statistics[name] = count.squeeze(0)
                resize_statistics = cropped_statistics
            radar = resize_radar_voxel_channels(
                radar,
                target_size,
                statistics=resize_statistics,
            )
            resize_aggregations.add(resize_aggregation)
            occupied = radar[0] > 0
            if torch.any(occupied):
                values = radar[1][occupied]
                if intensity_transform == "log1p_robust_zscore":
                    values = torch.log1p(values.clamp_min(0.0))
                intensity_chunks.append(values.cpu().numpy().astype(np.float64))
            total_frames += 1

    if total_frames <= 0:
        raise RadarNormalizationError("没有可用于统计的训练帧")
    if not intensity_chunks:
        raise RadarNormalizationError("训练场景没有 occupied Radar voxel")
    if len(resize_aggregations) != 1:
        raise RadarNormalizationError(
            "训练帧混用 Radar resize aggregation，拒绝生成 normalization"
        )
    resize_aggregation = next(iter(resize_aggregations))
    transformed_intensity = np.concatenate(intensity_chunks)
    if transformed_intensity.size == 0 or not np.isfinite(transformed_intensity).all():
        raise RadarNormalizationError("occupied intensity 统计为空或包含非有限值")
    q25, median, q75 = np.quantile(transformed_intensity, [0.25, 0.5, 0.75])
    intensity_iqr = float(q75 - q25)
    if not math.isfinite(intensity_iqr) or intensity_iqr <= 0.0:
        raise RadarNormalizationError("occupied intensity 的 IQR 必须为正有限数")

    is_formal = bool(train_frame_ids_by_scene is not None and max_frames == 0)
    input_provenance = {
        "dataset_manifest_sha256": manifest_hashes,
    }
    if split_artifact_sha256:
        input_provenance["split_artifact_sha256"] = split_artifact_sha256
    artifact = {
        "protocol": normalization_protocol,
        "formal": is_formal,
        "training_scenes": selected_scenes,
        "frame_count": total_frames,
        "target_size": [int(value) for value in target_size],
        "source_pc_range": [float(value) for value in source_pc_range],
        "model_pc_range": [float(value) for value in model_pc_range],
        "intensity": (
            {
                "transform": "log1p_robust_zscore",
                "log_median": float(median),
                "log_iqr": intensity_iqr,
                "clip": [-5.0, 5.0],
            }
            if normalization_protocol == RADAR_NORMALIZATION_PROTOCOL
            else {
                "transform": "identity_robust_zscore",
                "quantity": intensity_quantity,
                "unit": intensity_unit,
                "median": float(median),
                "iqr": intensity_iqr,
                "clip": [-5.0, 5.0],
            }
        ),
        "doppler": {
            "transform": "symmetric_physical_scale",
            "scale_mps": scale_mps,
            "clip": [-1.0, 1.0],
        },
        "variance": {
            "transform": "identity",
            "unit": "m2_s2",
            "aggregation": resize_aggregation,
        },
        "input_provenance": input_provenance,
    }
    artifact = validate_radar_normalization_spec(
        artifact,
        target_size=target_size,
        source_pc_range=source_pc_range,
        model_pc_range=model_pc_range,
        doppler_scale_mps=scale_mps,
        require_formal=is_formal,
        expected_split_artifact_sha256=(
            split_artifact_sha256 if is_formal else None
        ),
    )
    _write_json_atomic(output_path, artifact)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从显式训练场景生成冻结的 Radar normalization artifact"
    )
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--scene", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target_size", nargs=3, type=int, required=True, metavar=("Z", "X", "Y"))
    parser.add_argument("--source_pc_range", nargs=6, type=float, required=True)
    parser.add_argument("--model_pc_range", nargs=6, type=float, required=True)
    parser.add_argument("--doppler_scale_mps", type=float, required=True)
    parser.add_argument(
        "--intensity_transform",
        choices=("log1p_robust_zscore", "identity_robust_zscore"),
        default="log1p_robust_zscore",
        help="正式脚本必须显式选择；默认值仅保留旧诊断入口兼容",
    )
    parser.add_argument("--intensity_quantity", default="intensity")
    parser.add_argument(
        "--intensity_unit",
        default="sensor_native_linear_nonnegative",
    )
    parser.add_argument(
        "--split_artifact",
        default="",
        help="正式 artifact 必须绑定的 temporal split；仅统计其 train frame",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="0 扫描全部训练帧；正数只生成 formal=false 诊断 artifact",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        output = build_and_write_artifact(
            dataset_dir=args.dataset_dir,
            scenes=args.scene,
            output_path=args.output,
            target_size=args.target_size,
            source_pc_range=args.source_pc_range,
            model_pc_range=args.model_pc_range,
            doppler_scale_mps=args.doppler_scale_mps,
            intensity_transform=args.intensity_transform,
            intensity_quantity=args.intensity_quantity,
            intensity_unit=args.intensity_unit,
            max_frames=args.max_frames,
            split_artifact_path=args.split_artifact,
        )
    except RadarNormalizationError as exc:
        parser.error(str(exc))
    print(json.dumps({"artifact_path": output}, ensure_ascii=False))


if __name__ == "__main__":
    main()
