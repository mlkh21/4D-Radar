#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化的推理脚本 - 使用训练好的 VAE/LDM/CD 模型生成雷达数据
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
import time
import random
from datetime import datetime
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from diffusion_consistency_radar.cm.vae_3d import VAE3D, build_vae_from_checkpoint, resolve_checkpoint_grid_config
    from diffusion_consistency_radar.cm.unet_optimized import OptimizedUNetModel
    from diffusion_consistency_radar.cm.multimodal_fusion import (
        CompleteDualModalityPerceptionNet,
        migrate_ir_gate_state_dict,
    )
    from diffusion_consistency_radar.cm.karras_diffusion import KarrasDenoiser
    from diffusion_consistency_radar.cm.dataset_loader import (
        CalibrationProvider,
        NTU4DRadLM_VoxelDataset,
        _resize_or_pad_ir_tensor,
        DEFAULT_THERMAL_K,
        crop_voxel_channels_to_pc_range,
        collate_voxel_samples,
        resize_radar_voxel_channels,
        resize_voxel_channels,
    )
except ImportError:
    from cm.vae_3d import VAE3D, build_vae_from_checkpoint, resolve_checkpoint_grid_config
    from cm.unet_optimized import OptimizedUNetModel
    from cm.multimodal_fusion import CompleteDualModalityPerceptionNet, migrate_ir_gate_state_dict
    from cm.karras_diffusion import KarrasDenoiser
    from cm.dataset_loader import (
        CalibrationProvider,
        NTU4DRadLM_VoxelDataset,
        _resize_or_pad_ir_tensor,
        DEFAULT_THERMAL_K,
        crop_voxel_channels_to_pc_range,
        collate_voxel_samples,
        resize_radar_voxel_channels,
        resize_voxel_channels,
    )

try:
    from diffusion_consistency_radar.radar_normalization import (
        LEGACY_RADAR_NORMALIZATION_PROTOCOL,
        RadarNormalizationError,
        apply_radar_normalization,
        radar_normalization_from_checkpoint,
        validate_radar_normalization_sha256,
    )
except ImportError:
    from radar_normalization import (
        LEGACY_RADAR_NORMALIZATION_PROTOCOL,
        RadarNormalizationError,
        apply_radar_normalization,
        radar_normalization_from_checkpoint,
        validate_radar_normalization_sha256,
    )

try:
    from diffusion_consistency_radar.cm.evaluation_metrics import (
        bev_iou as task_bev_iou,
        filter_points_by_band,
        nearest_neighbor_metrics,
        occupancy_prf,
        parse_range_bins,
        uncertainty_calibration_metrics,
    )
except ImportError:
    from cm.evaluation_metrics import (
        bev_iou as task_bev_iou,
        filter_points_by_band,
        nearest_neighbor_metrics,
        occupancy_prf,
        parse_range_bins,
        uncertainty_calibration_metrics,
    )
from torch.utils.data import DataLoader

try:
    from diffusion_consistency_radar.checkpoint_chain import (
        CheckpointChainError,
        FORMAL_CHECKPOINT_PROTOCOL,
        FORMAL_MINI_CHECKPOINT_PROTOCOL,
        safe_torch_load as safe_checkpoint_load,
        validate_checkpoint_data_protocol,
    )
except ImportError:
    from checkpoint_chain import (
        CheckpointChainError,
        FORMAL_CHECKPOINT_PROTOCOL,
        FORMAL_MINI_CHECKPOINT_PROTOCOL,
        safe_torch_load as safe_checkpoint_load,
        validate_checkpoint_data_protocol,
    )

try:
    from diffusion_consistency_radar.dataset_manifest import (
        sha256_file,
    )
    from diffusion_consistency_radar.deployment_view import validate_deployment_view
except ImportError:
    from dataset_manifest import sha256_file
    from deployment_view import validate_deployment_view

try:
    from diffusion_consistency_radar.geometry_protocol import load_extrinsic_transform
except ImportError:
    from geometry_protocol import load_extrinsic_transform

try:
    from diffusion_consistency_radar.prediction_artifact_protocol import (
        build_prediction_voxel_metadata,
    )
except ImportError:
    from prediction_artifact_protocol import build_prediction_voxel_metadata

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None


TASK_METRIC_FIELDS = [
    "task_near_recall_mean",
    "task_near_precision_mean",
    "task_near_bev_iou_mean",
    "task_near_nn_mean",
    "task_near_match_ratio_2_mean",
    "uncertainty_ece",
    "uncertainty_brier",
    "uncertainty_nll",
    "uncertainty_error_corr",
]

REMOVED_ORACLE_ARGUMENTS = (
    "--adaptive_occ_from_target",
    "--adaptive_target_threshold",
)

RUNTIME_FIELDS = [
    "index",
    "radar_file",
    "radar_point_count",
    "effective_occ_threshold",
    "inference_seconds",
    "pred_point_count",
    "is_empty_frame",
    "used_topk_fallback",
    "train_duration_seconds",
    "total_infer_seconds",
    "avg_infer_seconds",
    "avg_pred_point_count",
    "empty_frame_rate",
]

RADAR_ENDPOINT_RAY_OBSERVED_PROTOCOL = "radar_endpoint_ray_visibility_v1"


def _uses_legacy_evaluation(args) -> bool:
    """判断是否显式启用了兼容的生成时真值评价分支。"""
    return bool(
        args.compare_with_target
        or args.compare_with_lidar
        or args.report_task_metrics
    )


def inference_csv_name(args) -> str:
    """部署运行时与兼容评价使用不同的 CSV 文件名。"""
    return (
        "inference_metrics.csv"
        if _uses_legacy_evaluation(args)
        else "inference_runtime.csv"
    )


def resolve_effective_voxel_size(voxel_size, pc_range, target_size):
    """按实际 C-Z-X-Y 输出网格解析 XYZ 体素尺寸。"""
    if voxel_size is not None:
        return [float(value) for value in voxel_size]
    z_size, x_size, y_size = [int(value) for value in target_size]
    return [
        (float(pc_range[3]) - float(pc_range[0])) / max(x_size, 1),
        (float(pc_range[4]) - float(pc_range[1])) / max(y_size, 1),
        (float(pc_range[5]) - float(pc_range[2])) / max(z_size, 1),
    ]


def _observed_mask_records_digest(records) -> str:
    """按帧顺序绑定 deployment observed mask 的内容身份。"""
    digest = hashlib.sha256()
    for record in records:
        digest.update(str(record["frame_id"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["file"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["sha256"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(int(record["observed_voxels"])).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_observed_mask_metadata(records, identity=None):
    """构造可由地图入口逐帧重算的 Radar 射线 observed 合同。"""
    normalized = []
    seen = set()
    for record in records:
        frame_id = str(record.get("frame_id", ""))
        file_name = str(record.get("file", ""))
        file_sha256 = str(record.get("sha256", ""))
        observed_voxels = int(record.get("observed_voxels", -1))
        if (
            not frame_id
            or frame_id in seen
            or os.path.basename(file_name) != file_name
            or not file_name.endswith("_observed_mask.npy")
            or len(file_sha256) != 64
            or any(character not in "0123456789abcdef" for character in file_sha256)
            or observed_voxels < 0
        ):
            raise ValueError("observed mask 记录格式无效或 frame 重复")
        seen.add(frame_id)
        normalized.append(
            {
                "frame_id": frame_id,
                "file": file_name,
                "sha256": file_sha256,
                "observed_voxels": observed_voxels,
            }
        )
    metadata = {
        "protocol": RADAR_ENDPOINT_RAY_OBSERVED_PROTOCOL,
        "coordinate_frame": "lidar",
        "source": "radar_endpoint_rays",
        "ir_frustum_marks_free_space": False,
        "frame_count": len(normalized),
        "observed_voxels": sum(record["observed_voxels"] for record in normalized),
        "files_sha256": _observed_mask_records_digest(normalized),
        "records": normalized,
    }
    if identity is not None:
        origin = np.asarray(identity.get("radar_origin_lidar_m"), dtype=np.float64)
        calibration_sha256 = str(identity.get("radar_to_lidar_sha256", ""))
        if (
            origin.shape != (3,)
            or not np.all(np.isfinite(origin))
            or len(calibration_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in calibration_sha256
            )
        ):
            raise ValueError("Radar observed identity 的原点或标定 SHA-256 无效")
        metadata["radar_origin_lidar_m"] = origin.astype(float).tolist()
        metadata["radar_to_lidar_sha256"] = calibration_sha256
    return metadata


def build_inference_run_metadata(
    args,
    generator,
    frame_count: int,
    observed_mask_records=None,
    prediction_voxel_records=None,
):
    """记录离线 evaluator 复现坐标和阈值协议所需的实际参数。"""
    observed_metadata = None
    if observed_mask_records is not None:
        observed_metadata = build_observed_mask_metadata(
            observed_mask_records,
            identity=getattr(generator, "radar_observed_identity", None),
        )
        if observed_metadata["frame_count"] != int(frame_count):
            raise ValueError("observed mask frame_count 与推理 frame_count 不一致")
    if bool(args.require_real_ir) and observed_metadata is None:
        raise ValueError("正式 deployment inference 必须发布 observed mask 合同")
    prediction_metadata = None
    if prediction_voxel_records is not None:
        prediction_metadata = build_prediction_voxel_metadata(
            prediction_voxel_records
        )
        if prediction_metadata["frame_count"] != int(frame_count):
            raise ValueError("prediction voxel frame_count 与推理 frame_count 不一致")
    if bool(args.require_real_ir) and prediction_metadata is None:
        raise ValueError("正式 deployment inference 必须发布 prediction voxel 合同")
    if bool(args.require_real_ir):
        deployment_identity = getattr(generator, "deployment_identity", None)
        deployment_calibration = (
            deployment_identity.get("calibration_sha256")
            if isinstance(deployment_identity, dict)
            else None
        )
        if (
            not isinstance(deployment_calibration, dict)
            or observed_metadata.get("radar_to_lidar_sha256")
            != deployment_calibration.get("radar_to_lidar")
        ):
            raise ValueError(
                "正式 observed mask 的 Radar→LiDAR 标定必须与 deployment identity 一致"
            )
    return {
        "stage": "deployment_generation",
        "target_size": [int(value) for value in args.target_size],
        "source_pc_range": [float(value) for value in args.source_pc_range],
        "model_pc_range": [float(value) for value in args.pc_range],
        "voxel_size": resolve_effective_voxel_size(
            args.voxel_size,
            args.pc_range,
            args.target_size,
        ),
        "occ_threshold": float(args.occ_threshold),
        "model_type": str(args.model_type),
        "steps": int(args.steps),
        "sampler": str(args.sampler),
        "model_is_multimodal": bool(
            getattr(generator.model, "is_multimodal", False)
        ),
        "require_real_ir": bool(args.require_real_ir),
        "radar_normalization_protocol": (
            generator.radar_normalization["protocol"]
            if generator.radar_normalization is not None
            else LEGACY_RADAR_NORMALIZATION_PROTOCOL
        ),
        "radar_normalization": generator.radar_normalization,
        "radar_normalization_sha256": generator.radar_normalization_sha256,
        "allow_legacy_radar_units": bool(generator.allow_legacy_radar_units),
        "checkpoint_protocol": getattr(generator, "checkpoint_protocol", None),
        "formal_protocol": (
            getattr(generator, "checkpoint_protocol", None)
            == FORMAL_CHECKPOINT_PROTOCOL
        ),
        "formal_mini_smoke": (
            getattr(generator, "checkpoint_protocol", None)
            == FORMAL_MINI_CHECKPOINT_PROTOCOL
        ),
        "frame_count": int(frame_count),
        "voxel_coordinate_frame": "lidar",
        "observed_mask": observed_metadata,
        "prediction_voxel": prediction_metadata,
        "time_alignment_compensation": "preprocessing_signed_delta_only",
        "deployment_identity": getattr(generator, "deployment_identity", None),
    }


def _write_json_atomic(path: str, payload) -> None:
    """通过同目录临时文件原子发布 JSON 运行记录。"""
    temp_path = f"{path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temp_path, path)


def _save_npy_atomic(path: str, array: np.ndarray) -> None:
    """同目录临时写入 NPY 后原子发布，避免暴露半文件。"""
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
        dir=os.path.dirname(path),
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


def prepare_fresh_output_dir(path: str) -> str:
    """创建或接受空输出目录，拒绝符号链接及任何历史内容。"""
    output_dir = os.path.abspath(os.fspath(path))
    if os.path.islink(output_dir):
        raise ValueError(f"输出目录不得是符号链接: {output_dir}")
    if os.path.exists(output_dir):
        if not os.path.isdir(output_dir):
            raise ValueError(f"输出路径必须是目录: {output_dir}")
        if os.listdir(output_dir):
            raise ValueError(f"输出目录必须为空，拒绝覆盖非空目录: {output_dir}")
    else:
        os.makedirs(output_dir, exist_ok=False)
    return output_dir


def reject_removed_oracle_arguments(argv) -> None:
    """拒绝已从正式推理移除的 target 自适应参数。"""
    for token in argv:
        text = str(token)
        if any(
            text == name or text.startswith(f"{name}=")
            for name in REMOVED_ORACLE_ARGUMENTS
        ):
            raise ValueError(
                f"{text.split('=', 1)[0]} 已从正式推理移除；"
                "请先用固定 --occ_threshold 保存预测体素，再运行 "
                "test/diagnostics/occupancy/diagnose_oracle_target_adaptation.py"
            )


def resolve_formal_inference_data_protocol(
    vae_checkpoint,
    model_checkpoint,
    *,
    model_type: str,
    allow_formal_mini_checkpoint: bool = False,
):
    """验证正式数据身份；mini-v2 仅可经显式授权用于离线 smoke。"""
    vae_protocol = (
        vae_checkpoint.get("checkpoint_protocol")
        if isinstance(vae_checkpoint, dict)
        else None
    )
    model_protocol = (
        model_checkpoint.get("checkpoint_protocol")
        if isinstance(model_checkpoint, dict)
        else None
    )
    formal_protocols = {
        FORMAL_CHECKPOINT_PROTOCOL,
        FORMAL_MINI_CHECKPOINT_PROTOCOL,
    }
    if not formal_protocols.intersection({vae_protocol, model_protocol}):
        return None

    errors = []
    if vae_protocol != model_protocol:
        errors.append("VAE 与生成模型 checkpoint protocol 不一致")
    if vae_protocol not in formal_protocols:
        errors.append("正式生成模型不能搭配 legacy VAE checkpoint")
    if model_protocol not in formal_protocols:
        errors.append("正式 VAE 不能搭配 legacy 生成模型 checkpoint")
    if (
        FORMAL_MINI_CHECKPOINT_PROTOCOL in (vae_protocol, model_protocol)
        and not allow_formal_mini_checkpoint
    ):
        errors.append(
            "formal mini checkpoint 仅允许通过显式开关执行 test smoke"
        )
    if isinstance(vae_checkpoint, dict) and vae_checkpoint.get("stage") != "vae":
        errors.append("VAE checkpoint stage 必须为 'vae'")
    if isinstance(model_checkpoint, dict) and model_checkpoint.get("stage") != model_type:
        errors.append(f"生成模型 checkpoint stage 必须为 {model_type!r}")

    vae_data = validate_checkpoint_data_protocol(
        vae_checkpoint.get("data_protocol") if isinstance(vae_checkpoint, dict) else None,
        stage="vae",
        errors=errors,
    )
    model_data = validate_checkpoint_data_protocol(
        model_checkpoint.get("data_protocol")
        if isinstance(model_checkpoint, dict)
        else None,
        stage=model_type,
        errors=errors,
    )
    if vae_data is not None and model_data is not None and vae_data != model_data:
        errors.append("VAE 与生成模型 data_protocol 不一致")
    if errors:
        raise CheckpointChainError(errors)
    return dict(model_data)


def assert_formal_deployment_identity(
    *,
    scene_dir: str,
    radar_voxel_dir: str,
    calibration_dir: str,
    data_protocol,
):
    """绑定严格 deployment view、输入目录与 checkpoint 训练标定。"""
    scene_dir = os.path.abspath(os.fspath(scene_dir))
    scene = os.path.basename(scene_dir)
    if not scene or scene in (".", ".."):
        raise ValueError("正式 deployment scene 目录名非法")
    view_identity = validate_deployment_view(scene_dir, scene)

    expected_radar_dir = os.path.join(scene_dir, "radar_voxel")
    if os.path.abspath(os.fspath(radar_voxel_dir)) != expected_radar_dir:
        raise ValueError("--radar_voxel_dir 必须属于 --deployment_scene_dir/radar_voxel")
    if view_identity.get("voxel_coordinate_frame") != "lidar":
        raise ValueError("正式 Radar+IR checkpoint 只接受 LiDAR frame deployment 数据")

    calibration_dir = os.path.abspath(os.fspath(calibration_dir))
    calibration_files = {
        "lidar_to_thermal": os.path.join(
            calibration_dir, "calib_livox_to_thermal.txt"
        ),
        "thermal_intrinsics": os.path.join(
            calibration_dir, "calib_cam_thermal.txt"
        ),
    }
    actual_calibration = {}
    for key, path in calibration_files.items():
        if os.path.islink(path) or not os.path.isfile(path):
            raise ValueError(f"正式部署标定缺失或不是普通文件: {path}")
        actual_calibration[key] = sha256_file(path)

    checkpoint_calibration = (data_protocol or {}).get("calibration_sha256")
    if actual_calibration != checkpoint_calibration:
        raise ValueError("当前 deployment 标定 SHA-256 与 checkpoint data_protocol 不一致")
    if view_identity.get("calibration_sha256") != actual_calibration:
        raise ValueError("deployment view 标定 SHA-256 与当前 calibration_dir 不一致")

    identity = dict(view_identity)
    identity["calibration_sha256"] = actual_calibration
    return identity


def safe_torch_load(path, map_location, *, allow_legacy_pickle=False):
    """推理默认只加载 weights-only；历史 pickle 需显式授权。"""
    return safe_checkpoint_load(
        path,
        map_location=map_location,
        allow_legacy_pickle=allow_legacy_pickle,
    )


def is_multimodal_state_dict(state_dict) -> bool:
    keys = list((state_dict or {}).keys())
    return any(
        key.startswith("unet_3d.")
        or key.startswith("ir_extractor.")
        or key.startswith("projection_layer.")
        or key.startswith("radar_encoder.")
        or key.startswith("uncertainty_head.")
        or key.startswith("model_uncertainty_head.")
        or key.startswith("fusion_conv.")
        for key in keys
    )


def _compatible_state_dict(model, state_dict):
    model_state = model.state_dict()
    return {
        key: value
        for key, value in (state_dict or {}).items()
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape)
    }


def resolve_generation_model_config(checkpoint, fallback_latent_dim=None):
    """优先读取生成模型元数据，旧 checkpoint 从首尾卷积 shape 推导。"""
    checkpoint_dict = checkpoint if isinstance(checkpoint, dict) else {}
    state_dict = checkpoint_dict.get("model_state_dict", checkpoint)
    metadata_config = checkpoint_dict.get("model_config") or {}
    latent_dim = checkpoint_dict.get("latent_dim", metadata_config.get("latent_dim"))
    in_channels = metadata_config.get("in_channels")
    prefixes = ("", "unet_3d.")
    if isinstance(state_dict, dict):
        for prefix in prefixes:
            input_weight = state_dict.get(f"{prefix}input_blocks.0.0.weight")
            output_weight = state_dict.get(f"{prefix}out.2.weight")
            if in_channels is None and torch.is_tensor(input_weight) and input_weight.ndim >= 2:
                in_channels = int(input_weight.shape[1])
            if latent_dim is None and torch.is_tensor(output_weight) and output_weight.ndim >= 1:
                latent_dim = int(output_weight.shape[0])
    if latent_dim is None:
        if fallback_latent_dim is None:
            raise ValueError(
                "生成模型 checkpoint 缺少 latent_dim 且无法从 state_dict 推导；"
                "请显式提供 fallback"
            )
        latent_dim = int(fallback_latent_dim)
    resolved = dict(metadata_config)
    resolved["latent_dim"] = int(latent_dim)
    resolved["in_channels"] = int(in_channels or 2 * int(latent_dim))
    resolved.setdefault("out_channels", int(latent_dim))
    return resolved


def build_inference_model(
    state_dict,
    device,
    strict: bool = True,
    fusion_voxel_shape=(32, 128, 128),
    fusion_pc_range=(0, -20, -6, 120, 20, 10),
    latent_dim: int = 4,
    model_config=None,
):
    if strict and not state_dict:
        raise ValueError(
            "build_inference_model strict=True 时 state_dict 不能为空，"
            "拒绝返回随机初始化模型"
        )
    cfg = dict(model_config or {})
    latent_dim = int(cfg.get("latent_dim", latent_dim))
    model_channels = int(cfg.get("model_channels", 32))
    num_res_blocks = int(cfg.get("num_res_blocks", 1))
    channel_mult = tuple(cfg.get("channel_mult", [1, 2, 3]))
    attention_resolutions = tuple(cfg.get("attention_resolutions", []))
    if is_multimodal_state_dict(state_dict):
        backbone_in_channels = int(cfg.get("in_channels", max(16, 2 * latent_dim)))
        base_unet = OptimizedUNetModel(
            image_size=32,
            in_channels=backbone_in_channels,
            model_channels=model_channels,
            out_channels=latent_dim,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            channel_mult=channel_mult,
            use_checkpoint=False,
            attention_type="linear",
        )
        model = CompleteDualModalityPerceptionNet(
            base_unet,
            voxel_shape=tuple(int(v) for v in fusion_voxel_shape),
            pc_range=tuple(float(v) for v in fusion_pc_range),
            fused_channels=backbone_in_channels,
            downsample_to_latent=True,
        ).to(device)
    else:
        backbone_in_channels = int(cfg.get("in_channels", 2 * latent_dim))
        model = OptimizedUNetModel(
            image_size=32,
            in_channels=backbone_in_channels,
            model_channels=model_channels,
            out_channels=latent_dim,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            channel_mult=channel_mult,
            use_checkpoint=False,
            attention_type="linear",
        ).to(device)
    if state_dict:
        state_dict = migrate_ir_gate_state_dict(model, state_dict)
        if strict:
            prefix = "unet_3d." if getattr(model, "is_multimodal", False) else ""
            critical_keys = (
                f"{prefix}input_blocks.0.0.weight",
                f"{prefix}out.2.weight",
            )
            missing_critical = [key for key in critical_keys if key not in state_dict]
            if missing_critical:
                model_kind = "多模态" if prefix else "单模态"
                raise RuntimeError(
                    f"{model_kind}生成模型 checkpoint 缺少关键权重: "
                    + ", ".join(missing_critical)
                )
            try:
                model.load_state_dict(state_dict, strict=True)
            except RuntimeError as exc:
                model_kind = "多模态" if getattr(model, "is_multimodal", False) else "单模态"
                raise RuntimeError(
                    f"{model_kind}生成模型 checkpoint 与当前结构不兼容: {exc}"
                ) from exc
        else:
            model.load_state_dict(_compatible_state_dict(model, state_dict), strict=False)
    return model


def append_task_metric_headers(header):
    for field in TASK_METRIC_FIELDS:
        if field not in header:
            header.append(field)
    return header


def build_task_metric_row(row, header, values):
    while len(row) < len(header):
        row.append("")
    for key, value in (values or {}).items():
        if key in header:
            row[header.index(key)] = f"{float(value):.6f}" if np.isfinite(float(value)) else ""
    return row


def build_task_metric_summary_row(row, header, summary):
    return build_task_metric_row(row, header, summary)


def _mock_multimodal_meta(batch_size: int, device):
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, 480, device=device),
        torch.linspace(-1.0, 1.0, 640, device=device),
        indexing="ij",
    )
    thermal = torch.exp(-((xx * 1.8) ** 2 + (yy * 1.2) ** 2))
    ir_img = torch.stack([thermal, thermal * 0.85, thermal * 0.65], dim=0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    return {
        "ir_img": ir_img,
        "r_mat": torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1),
        "t_vec": torch.zeros(batch_size, 3, device=device),
        "k_mat": torch.from_numpy(DEFAULT_THERMAL_K).to(device).unsqueeze(0).repeat(batch_size, 1, 1),
        "is_mock_ir": torch.ones(batch_size, device=device),
        "is_mock_calib": torch.ones(batch_size, device=device),
        "odom_cov_trace": torch.zeros(batch_size, device=device),
    }


def prepare_multimodal_meta(meta_dict, batch_size: int, device):
    meta = _mock_multimodal_meta(batch_size, device)
    for key in ("ir_img", "r_mat", "t_vec", "k_mat", "is_mock_ir", "is_mock_calib", "odom_cov_trace"):
        value = (meta_dict or {}).get(key) if isinstance(meta_dict, dict) else None
        if torch.is_tensor(value):
            value = value.to(device)
            if value.dim() == meta[key].dim() - 1:
                value = value.unsqueeze(0)
            meta[key] = value
        elif value is not None and key in ("is_mock_ir", "is_mock_calib", "odom_cov_trace"):
            meta[key] = torch.as_tensor(value, device=device, dtype=torch.float32).reshape(-1)
            if meta[key].numel() == 1 and batch_size > 1:
                meta[key] = meta[key].repeat(batch_size)
    return meta


def validate_real_ir_model(model):
    """正式真实 IR 模式只接受实际构建出的多模态生成模型。"""
    if not bool(getattr(model, "is_multimodal", False)):
        raise RuntimeError(
            "--require_real_ir requires a multimodal Radar+IR checkpoint"
        )


def _load_ir_array(ir_path: str, require_real_ir: bool) -> np.ndarray:
    """读取并验证逐帧 IR；严格模式额外拒绝链接和非普通文件。"""
    if require_real_ir and os.path.islink(ir_path):
        raise RuntimeError(f"严格真实 IR 模式拒绝符号链接: {ir_path}")
    if require_real_ir and not os.path.isfile(ir_path):
        raise RuntimeError(f"严格真实 IR 模式缺少普通 IR 文件: {ir_path}")

    array = np.load(ir_path).astype(np.float32)
    if array.ndim not in (2, 3):
        raise RuntimeError(f"IR 数组维度非法 {array.shape}: {ir_path}")
    if not np.isfinite(array).all():
        raise RuntimeError(f"IR 数组含非有限值: {ir_path}")
    return array


def load_multimodal_meta_for_radar(
    radar_path: str,
    device,
    require_real_ir: bool = False,
    calibration_dir: str = "",
):
    """加载逐帧 IR/标定元数据，并可启用正式 fail-closed 协议。"""
    scene_dir = os.path.dirname(os.path.dirname(radar_path))
    dataset_root = os.path.dirname(scene_dir)
    frame_id = os.path.splitext(os.path.basename(radar_path))[0].replace("_voxel", "")
    ir_path = os.path.join(scene_dir, "ir_image", f"{frame_id}_ir.npy")
    meta = {}

    provider = CalibrationProvider(
        dataset_root,
        calibration_dir=calibration_dir or None,
        require_real=require_real_ir,
        voxel_coordinate_frame="lidar",
    )
    r_mat, t_vec, k_mat, calib_meta = provider.load_with_metadata()
    is_mock = bool(calib_meta["is_mock_calib"])
    if require_real_ir and (
        is_mock or not bool(calib_meta["calib_is_thermal"])
    ):
        raise RuntimeError(
            "严格真实 IR 模式需要 calib_livox_to_thermal.txt: "
            f"scene={scene_dir}, source={calib_meta['calib_source']}, "
            f"reason={calib_meta['calib_fallback_reason']}"
        )
    if require_real_ir and not bool(calib_meta.get("has_thermal_intrinsics", False)):
        raise RuntimeError(
            "严格真实 IR 模式需要 calib_cam_thermal.txt（S/K/D）: "
            f"scene={scene_dir}, source={calib_meta.get('thermal_intrinsics_source', 'unknown')}"
        )

    if os.path.lexists(ir_path):
        arr = _load_ir_array(ir_path, require_real_ir=require_real_ir)
        meta["ir_img"] = _resize_or_pad_ir_tensor(
            torch.from_numpy(arr), calib_meta
        ).to(device)
        meta["is_mock_ir"] = torch.zeros(1, device=device)
    elif require_real_ir:
        raise RuntimeError(
            f"严格真实 IR 模式缺少 IR: scene={scene_dir}, frame={frame_id}"
        )
    else:
        meta["is_mock_ir"] = torch.ones(1, device=device)

    meta["r_mat"] = r_mat.to(device)
    meta["t_vec"] = t_vec.to(device)
    meta["k_mat"] = k_mat.to(device)
    meta["is_mock_calib"] = torch.ones(1, device=device) if is_mock else torch.zeros(1, device=device)
    meta["odom_cov_trace"] = torch.zeros(1, device=device)
    meta["time_alignment_compensation"] = "preprocessing_signed_delta_only"
    meta["calib_source"] = calib_meta["calib_source"]
    meta["thermal_intrinsics_source"] = calib_meta["thermal_intrinsics_source"]
    meta["thermal_intrinsics_path"] = calib_meta["thermal_intrinsics_path"]
    meta["thermal_source_size"] = calib_meta["thermal_source_size"]
    meta["thermal_output_size"] = calib_meta["thermal_output_size"]
    meta["thermal_distortion"] = calib_meta["thermal_distortion"]
    meta["has_thermal_intrinsics"] = bool(calib_meta["has_thermal_intrinsics"])
    return meta


def preflight_real_ir_inputs(model, radar_paths, calibration_dir: str):
    """在创建输出目录前验证全部正式帧，禁止留下部分正式结果。"""
    validate_real_ir_model(model)
    for radar_path in radar_paths:
        meta = load_multimodal_meta_for_radar(
            radar_path,
            torch.device("cpu"),
            require_real_ir=True,
            calibration_dir=calibration_dir,
        )
        if (
            float(meta["is_mock_ir"].item()) != 0.0
            or float(meta["is_mock_calib"].item()) != 0.0
        ):
            raise RuntimeError(f"严格真实 IR preflight 得到 mock meta: {radar_path}")


def resolve_inference_grid_config(
    checkpoint_metadata,
    target_size,
    source_pc_range,
    model_pc_range,
):
    """解析推理全流程使用的有效网格。"""
    return resolve_checkpoint_grid_config(
        checkpoint_metadata,
        target_size=target_size,
        source_pc_range=source_pc_range,
        model_pc_range=model_pc_range,
    )


def resolve_inference_radar_normalization(
    checkpoint_metadata,
    *,
    target_size,
    source_pc_range,
    model_pc_range,
    allow_legacy_radar_units=False,
):
    """从生成模型 checkpoint 解析唯一 Radar 量纲；默认 fail-closed。"""
    if type(allow_legacy_radar_units) is not bool:
        raise RadarNormalizationError("allow_legacy_radar_units 必须是 bool")
    checkpoint = checkpoint_metadata if isinstance(checkpoint_metadata, dict) else {}
    has_embedded = (
        "radar_normalization" in checkpoint
        or "radar_normalization_sha256" in checkpoint
    )
    if allow_legacy_radar_units:
        if has_embedded:
            raise RadarNormalizationError(
                "显式 legacy Radar 单位与正式 radar_normalization 不能同时启用"
            )
        return None, ""
    return radar_normalization_from_checkpoint(
        checkpoint,
        target_size=target_size,
        source_pc_range=source_pc_range,
        model_pc_range=model_pc_range,
        context="推理生成模型 checkpoint",
    )


class RadarGenerator:
    """雷达数据生成器"""
    
    def __init__(
        self,
        vae_path,
        model_path,
        model_type='ldm',
        device='cuda',
        target_size=None,
        pc_range=None,
        source_pc_range=None,
        vae_fallback_config_type=None,
        allow_legacy_radar_units=False,
        allow_legacy_checkpoint_pickle=False,
        allow_formal_mini_checkpoint=False,
    ):
        """
        Args:
            vae_path: VAE 模型路径
            model_path: LDM 或 CD 模型路径
            model_type: 'ldm' 或 'cd'
            device: 'cuda' 或 'cpu'
        """
        self.device = torch.device(device)
        self.model_type = model_type
        self.vae_fallback_config_type = vae_fallback_config_type
        self.allow_legacy_radar_units = allow_legacy_radar_units
        self.allow_legacy_checkpoint_pickle = bool(allow_legacy_checkpoint_pickle)
        self.allow_formal_mini_checkpoint = bool(allow_formal_mini_checkpoint)
        self.data_protocol = None
        self.deployment_identity = None
        self.radar_observed_identity = None
        
        # NOTE: 变分自编码器（VAE）负责体素空间与潜空间之间的编码/解码。
        print(f"Loading VAE from {vae_path}...")
        self.vae = self._load_vae(vae_path)
        self.vae.eval()
        (
            self.target_size,
            self.source_pc_range,
            self.pc_range,
        ) = resolve_inference_grid_config(
            self.vae_checkpoint_metadata,
            target_size,
            source_pc_range,
            pc_range,
        )
        
        # NOTE: 生成模型在潜空间进行去噪采样。
        print(f"Loading {model_type.upper()} from {model_path}...")
        self.model = self._load_model(model_path)
        self.model.eval()
        self.last_uncertainty = None
        
        # NOTE: 复用训练时一致的噪声调度参数。
        self.denoiser = KarrasDenoiser(
            sigma_data=0.5,
            sigma_max=80.0,
            sigma_min=0.002,
            loss_norm='l2',
            device=self.device,
        )
        
        print("Models loaded successfully!")

    def _load_vae(self, ckpt_path):
        """加载 VAE 模型"""
        ckpt = safe_torch_load(
            ckpt_path,
            map_location=self.device,
            allow_legacy_pickle=getattr(
                self, "allow_legacy_checkpoint_pickle", False
            ),
        )
        vae, metadata = build_vae_from_checkpoint(
            ckpt,
            fallback_config_type=self.vae_fallback_config_type,
        )
        self.vae_checkpoint_payload = ckpt if isinstance(ckpt, dict) else {}
        self.vae_checkpoint_metadata = metadata
        return vae.to(self.device)
    
    def _load_model(self, ckpt_path):
        """加载 LDM 或 CD 模型"""
        ckpt = safe_torch_load(
            ckpt_path,
            map_location=self.device,
            allow_legacy_pickle=getattr(
                self, "allow_legacy_checkpoint_pickle", False
            ),
        )
        self.model_checkpoint_metadata = ckpt if isinstance(ckpt, dict) else {}
        self.data_protocol = resolve_formal_inference_data_protocol(
            getattr(self, "vae_checkpoint_payload", {}),
            self.model_checkpoint_metadata,
            model_type=getattr(self, "model_type", "ldm"),
            allow_formal_mini_checkpoint=getattr(
                self, "allow_formal_mini_checkpoint", False
            ),
        )
        self.checkpoint_protocol = (
            ckpt.get("checkpoint_protocol") if isinstance(ckpt, dict) else None
        )
        (
            self.radar_normalization,
            self.radar_normalization_sha256,
        ) = resolve_inference_radar_normalization(
            self.model_checkpoint_metadata,
            target_size=self.target_size,
            source_pc_range=self.source_pc_range,
            model_pc_range=self.pc_range,
            allow_legacy_radar_units=self.allow_legacy_radar_units,
        )
        state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        model_config = resolve_generation_model_config(
            ckpt,
            fallback_latent_dim=self.vae.latent_dim,
        )
        return build_inference_model(
            state_dict,
            self.device,
            strict=True,
            fusion_voxel_shape=self.target_size,
            fusion_pc_range=self.pc_range,
            latent_dim=model_config["latent_dim"],
            model_config=model_config,
        )

    def _apply_vae_occupancy_activation(self, decoded):
        """按 checkpoint 协议转换 occupancy，连续通道保持原值。"""
        metadata = getattr(self, "vae_checkpoint_metadata", {}) or {}
        if metadata.get("occupancy_activation", "raw") != "sigmoid":
            return decoded
        activated = decoded.clone()
        activated[:, 0:1] = VAE3D.occupancy_probability(decoded[:, 0:1])
        return activated
    
    @torch.no_grad()
    def generate(self, condition, num_samples=1, steps=40, sampler='heun', show_sampling_progress=False, meta_dict=None):
        """
        生成雷达数据
        
        Args:
            condition: 条件数据 (B, 4, 32, 128, 128) 或 None
            num_samples: 生成样本数
            steps: 采样步数 (LDM: 40, CD: 1-4)
            sampler: 'heun' 或 'euler'
        
        Returns:
            generated: 生成的雷达数据 (B, 4, 32, 128, 128)
        """
        if show_sampling_progress:
            print(f"\nGenerating {num_samples} samples with {steps} steps ({sampler} sampler)...")
        self.last_uncertainty = None
        
        # NOTE: 正式多模态模型直接消费物理 Radar 体素；legacy UNet 才使用条件潜变量。
        condition_voxel = None
        is_multimodal = bool(getattr(self.model, "is_multimodal", False))
        if condition is not None:
            condition = condition.to(self.device)
            condition_voxel = condition
            if is_multimodal:
                latent_spatial = self.vae.latent_spatial_shape(self.target_size)
                z_shape = (
                    condition.shape[0],
                    int(self.vae.latent_dim),
                    *latent_spatial,
                )
                z_cond = None
            else:
                z_cond = self.vae.get_latent(condition)
                z_shape = tuple(z_cond.shape)
            num_samples = z_shape[0]
        else:
            # NOTE: 无条件采样按已加载 VAE 架构推导潜空间，避免固定 z4/default grid。
            latent_spatial = self.vae.latent_spatial_shape(self.target_size)
            z_shape = (num_samples, int(self.vae.latent_dim), *latent_spatial)
            z_cond = None if is_multimodal else torch.zeros(z_shape, device=self.device)
            vae_config = (
                getattr(self, "vae_checkpoint_metadata", {}) or {}
            ).get("vae_config", {})
            input_channels = int(vae_config.get("in_channels", 4))
            condition_voxel = torch.zeros(
                num_samples,
                input_channels,
                *self.target_size,
                device=self.device,
            )
        
        # NOTE: 按 sigma_max 初始化起始噪声，匹配 Karras 采样假设。
        z_T = torch.randn(z_shape, device=self.device) * self.denoiser.sigma_max
        
        # NOTE: 一致性蒸馏（CD）模式支持一步采样，潜扩散模型（LDM）走多步求解器。
        if self.model_type == 'cd' and steps == 1:
            z_0 = self._cd_sample(z_T, z_cond, condition_voxel, meta_dict)
        else:
            z_0 = self._ldm_sample(z_T, z_cond, steps, sampler, show_sampling_progress, condition_voxel, meta_dict)
        
        # NOTE: 将潜变量解码回体素空间输出。
        generated = self._apply_vae_occupancy_activation(self.vae.decode(z_0))
        
        return generated
    
    def _cd_sample(self, z_T, z_cond, condition_voxel=None, meta_dict=None):
        """CD 一步采样"""
        if getattr(self.model, "is_multimodal", False):
            meta = prepare_multimodal_meta(meta_dict, z_T.shape[0], self.device)
            return self._call_multimodal_model(
                condition_voxel,
                meta,
                torch.ones(z_T.shape[0], device=self.device) * self.denoiser.sigma_max,
                z_T,
            )
        model_input = torch.cat([z_T, z_cond], dim=1)
        sigma = torch.ones(z_T.shape[0], device=self.device) * self.denoiser.sigma_max
        z_0 = self.model(model_input, sigma)
        return z_0

    def _call_multimodal_model(self, condition_voxel, meta, sigma_batch, noised_latent):
        kwargs = {
            "noised_latent": noised_latent,
            "odom_cov_trace": meta.get("odom_cov_trace"),
            "is_mock_ir": meta.get("is_mock_ir"),
            "is_mock_calib": meta.get("is_mock_calib"),
            "return_uncertainty": True,
        }
        try:
            out = self.model(
                condition_voxel,
                meta["ir_img"],
                meta["r_mat"],
                meta["t_vec"],
                meta["k_mat"],
                sigma_batch,
                **kwargs,
            )
        except TypeError:
            out = self.model(
                condition_voxel,
                meta["ir_img"],
                meta["r_mat"],
                meta["t_vec"],
                meta["k_mat"],
                sigma_batch,
                noised_latent=noised_latent,
            )
        if isinstance(out, tuple):
            denoised, uncertainty = out
            self.last_uncertainty = uncertainty
            return denoised
        self.last_uncertainty = None
        return out
    
    def _ldm_sample(self, z_T, z_cond, steps, sampler, show_sampling_progress=False, condition_voxel=None, meta_dict=None):
        """LDM 多步采样 (Heun/Euler)"""
        # NOTE: 生成单调递减的 sigma 序列，驱动 ODE 采样。
        sigmas = self._get_sigmas(steps)
        z_t = z_T
        
        iterator = range(len(sigmas) - 1)
        if show_sampling_progress:
            iterator = tqdm(iterator, desc="Sampling")

        for i in iterator:
            sigma_t = sigmas[i]
            sigma_next = sigmas[i + 1]
            
            # NOTE: 拼接条件潜变量后预测去噪结果。
            sigma_batch = torch.ones(z_t.shape[0], device=self.device) * sigma_t
            if getattr(self.model, "is_multimodal", False):
                meta = prepare_multimodal_meta(meta_dict, z_t.shape[0], self.device)
                denoised = self._call_multimodal_model(condition_voxel, meta, sigma_batch, z_t)
            else:
                model_input = torch.cat([z_t, z_cond], dim=1)
                denoised = self.model(model_input, sigma_batch)
            
            # NOTE: 常微分方程（ODE）形式导数 d = (x - denoised) / sigma。
            d = (z_t - denoised) / sigma_t
            
            if sampler == 'heun' and i < len(sigmas) - 2:
                # NOTE: 海恩（Heun）二阶法：先欧拉预测，再做一次校正。
                z_next = z_t + d * (sigma_next - sigma_t)
                
                # NOTE: 在预测点重新估计导数。
                sigma_batch_2 = torch.ones(z_t.shape[0], device=self.device) * sigma_next
                if getattr(self.model, "is_multimodal", False):
                    denoised_2 = self._call_multimodal_model(condition_voxel, meta, sigma_batch_2, z_next)
                else:
                    model_input_2 = torch.cat([z_next, z_cond], dim=1)
                    denoised_2 = self.model(model_input_2, sigma_batch_2)
                d_2 = (z_next - denoised_2) / sigma_next
                
                # NOTE: 两次导数求均值完成校正。
                z_t = z_t + (d + d_2) / 2 * (sigma_next - sigma_t)
            else:
                # NOTE: 欧拉（Euler）一阶法速度快但精度较低。
                z_t = z_t + d * (sigma_next - sigma_t)
        
        return z_t
    
    def _get_sigmas(self, steps):
        """生成噪声水平调度"""
        rho = 7.0
        sigma_min = self.denoiser.sigma_min
        sigma_max = self.denoiser.sigma_max
        
        step_indices = torch.arange(steps + 1, device=self.device)
        t = step_indices / steps
        sigmas = (sigma_max ** (1 / rho) + t * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        
        return sigmas


def set_random_seed(seed: int):
    """设置随机种子以保证推理可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_sparse_voxel(filename):
    # NOTE: 历史数据可能以稀疏格式保存，这里恢复为稠密体素以复用统一推理流程。
    data = np.load(filename)
    voxel_grid = np.zeros(data['shape'], dtype=np.float32)
    coords = data['coords']
    if coords.shape[0] > 0:
        voxel_grid[coords[:, 0], coords[:, 1], coords[:, 2]] = data['features']
    return voxel_grid


def load_radar_endpoint_mask(
    path,
    target_size=(32, 128, 128),
    source_pc_range=(0, -20, -6, 120, 20, 10),
    model_pc_range=None,
):
    """按模型物理范围加载 Radar endpoint occupancy 为 ``(Z,X,Y)``。"""
    if os.path.islink(path) or not os.path.isfile(path):
        raise ValueError(f"Radar voxel 必须是普通文件: {path}")
    radar_voxel = (
        load_sparse_voxel(path)
        if path.endswith(".npz")
        else np.load(path, allow_pickle=False).astype(np.float32)
    )
    if radar_voxel.ndim != 4 or radar_voxel.shape[-1] != 4:
        raise ValueError(f"Radar voxel 必须是 (X,Y,Z,4): {path} -> {radar_voxel.shape}")
    if not np.isfinite(radar_voxel).all():
        raise ValueError(f"Radar voxel 含非有限数: {path}")
    tensor = torch.from_numpy(radar_voxel).permute(3, 2, 0, 1)
    effective_model_range = (
        source_pc_range if model_pc_range is None else model_pc_range
    )
    tensor = crop_voxel_channels_to_pc_range(
        tensor,
        source_pc_range,
        effective_model_range,
    )
    occupancy = F.adaptive_max_pool3d(
        tensor[0:1].unsqueeze(0),
        tuple(int(value) for value in target_size),
    ).squeeze(0).squeeze(0)
    return (occupancy.detach().cpu().numpy() > 0).astype(np.uint8)


def build_radar_ray_observed_mask(
    endpoint_mask_zxy,
    *,
    pc_range,
    radar_origin_lidar_m,
):
    """从 Radar endpoint 到传感器原点离散射线，构造 LiDAR-frame 可见域。

    IR 只参与条件特征，不参与 free-space 标记；无 Radar endpoint 时返回全零，
    从而保持整帧 unknown。
    """
    endpoints = np.asarray(endpoint_mask_zxy)
    if endpoints.ndim != 3 or not np.all(np.isfinite(endpoints)):
        raise ValueError("endpoint_mask_zxy 必须是有限的 (Z,X,Y) 数组")
    bounds = np.asarray(pc_range, dtype=np.float64)
    origin = np.asarray(radar_origin_lidar_m, dtype=np.float64)
    if (
        bounds.shape != (6,)
        or origin.shape != (3,)
        or not np.all(np.isfinite(bounds))
        or not np.all(np.isfinite(origin))
        or not np.all(bounds[3:] > bounds[:3])
    ):
        raise ValueError("pc_range 与 radar_origin_lidar_m 必须是有效有限坐标")

    z_size, x_size, y_size = endpoints.shape
    xyz_size = np.asarray([x_size, y_size, z_size], dtype=np.float64)
    xyz_resolution = (bounds[3:] - bounds[:3]) / xyz_size
    origin_xyz_index = (origin - bounds[:3]) / xyz_resolution - 0.5
    observed = np.zeros_like(endpoints, dtype=np.uint8)
    endpoint_indices_zxy = np.argwhere(endpoints > 0)
    if endpoint_indices_zxy.size == 0:
        return observed
    observed[tuple(endpoint_indices_zxy.T)] = 1
    endpoint_indices_xyz = endpoint_indices_zxy[:, [1, 2, 0]].astype(np.int64)
    origin_cell_xyz = np.floor(
        (origin - bounds[:3]) / xyz_resolution
    ).astype(np.int64)
    directions = endpoint_indices_xyz - origin_cell_xyz[np.newaxis, :]
    gcd = np.gcd.reduce(np.abs(directions), axis=1)
    gcd[gcd == 0] = 1
    normalized_directions = directions // gcd[:, np.newaxis]
    endpoint_centers_xyz = (
        bounds[:3]
        + (endpoint_indices_xyz.astype(np.float64) + 0.5) * xyz_resolution
    )
    distances = np.linalg.norm(endpoint_centers_xyz - origin[np.newaxis, :], axis=1)
    order = np.argsort(distances, kind="stable")
    _, nearest_positions = np.unique(
        normalized_directions[order],
        axis=0,
        return_index=True,
    )
    selected_endpoint_indices = order[nearest_positions]
    for endpoint_xyz_index in endpoint_indices_xyz[selected_endpoint_indices]:
        endpoint_xyz_index = endpoint_xyz_index.astype(np.float64)
        step_count = int(
            np.ceil(np.max(np.abs(endpoint_xyz_index - origin_xyz_index)))
        ) + 1
        samples_xyz = np.linspace(
            origin_xyz_index,
            endpoint_xyz_index,
            num=max(step_count, 1),
        )
        samples_xyz = np.rint(samples_xyz).astype(np.int64)
        valid = (
            (samples_xyz[:, 0] >= 0)
            & (samples_xyz[:, 0] < x_size)
            & (samples_xyz[:, 1] >= 0)
            & (samples_xyz[:, 1] < y_size)
            & (samples_xyz[:, 2] >= 0)
            & (samples_xyz[:, 2] < z_size)
        )
        samples_xyz = samples_xyz[valid]
        observed[
            samples_xyz[:, 2],
            samples_xyz[:, 0],
            samples_xyz[:, 1],
        ] = 1
    return observed


def preflight_radar_observed_inputs(
    radar_paths,
    *,
    calibration_dir,
    target_size,
    source_pc_range,
    model_pc_range,
):
    """在创建输出前验证全部 Radar 文件和 Radar→LiDAR 射线原点。"""
    calibration_path = os.path.join(
        os.path.abspath(os.fspath(calibration_dir)),
        "calib_radar_to_livox.txt",
    )
    T_lidar_radar = load_extrinsic_transform(calibration_path)
    for radar_path in radar_paths:
        load_radar_endpoint_mask(
            radar_path,
            target_size=target_size,
            source_pc_range=source_pc_range,
            model_pc_range=model_pc_range,
        )
    return {
        "radar_origin_lidar_m": T_lidar_radar[:3, 3].astype(float).tolist(),
        "radar_to_lidar_sha256": sha256_file(calibration_path),
    }


def load_radar_voxel_as_tensor(
    path,
    device,
    target_size=(32, 128, 128),
    source_pc_range=(0, -20, -6, 120, 20, 10),
    model_pc_range=None,
    radar_normalization=None,
    radar_normalization_sha256="",
    allow_legacy_radar_units=False,
):
    """按训练一致的物理重采样和冻结 artifact 加载 Radar 条件。"""
    if path.endswith('.npz'):
        radar_voxel = load_sparse_voxel(path)
    else:
        radar_voxel = np.load(path).astype(np.float32)

    radar_tensor = torch.from_numpy(radar_voxel).permute(3, 2, 0, 1)
    if model_pc_range is None:
        model_pc_range = source_pc_range
    radar_tensor = crop_voxel_channels_to_pc_range(radar_tensor, source_pc_range, model_pc_range)
    radar_tensor = resize_radar_voxel_channels(radar_tensor, target_size)
    if radar_normalization is None:
        if not allow_legacy_radar_units:
            raise RadarNormalizationError(
                "正式推理缺少 radar_normalization；历史诊断需显式启用 legacy"
            )
        if radar_normalization_sha256:
            raise RadarNormalizationError(
                "legacy Radar 单位不得携带 normalization SHA-256"
            )
    else:
        if allow_legacy_radar_units:
            raise RadarNormalizationError(
                "legacy Radar 单位与正式 radar_normalization 不能同时启用"
            )
        validate_radar_normalization_sha256(
            radar_normalization_sha256,
            context="逐文件推理 Radar normalization",
        )
        radar_tensor = apply_radar_normalization(radar_tensor, radar_normalization)

    return radar_tensor.to(device)


def count_radar_points(path):
    """统计原始 radar_voxel 中的占据点数量（基于 Occ 通道 > 0）。"""
    if path.endswith('.npz'):
        radar_voxel = load_sparse_voxel(path)
    else:
        radar_voxel = np.load(path).astype(np.float32)

    if radar_voxel.ndim != 4 or radar_voxel.shape[-1] < 1:
        return 0

    return int(np.count_nonzero(radar_voxel[..., 0] > 0))


def count_raw_radar_pcl_points(path):
    """统计原始 radar_pcl 点数（第一维长度）。"""
    pts = np.load(path)
    if pts.ndim == 0:
        return 0
    return int(pts.shape[0])


def load_voxel_as_czxy(
    path,
    device,
    mask_channel=None,
    target_size=(32, 128, 128),
    source_pc_range=(0, -20, -6, 120, 20, 10),
    model_pc_range=None,
):
    """加载 XY-Z-C 体素并按训练/推理规则重采样为 (C, Z, X, Y)。"""
    if path.endswith('.npz'):
        voxel = load_sparse_voxel(path)
    else:
        voxel = np.load(path).astype(np.float32)

    tensor = torch.from_numpy(voxel).permute(3, 2, 0, 1)
    if model_pc_range is None:
        model_pc_range = source_pc_range
    tensor = crop_voxel_channels_to_pc_range(tensor, source_pc_range, model_pc_range)
    tensor = resize_voxel_channels(tensor, target_size, mask_channel=mask_channel)
    return tensor.to(device).cpu().numpy()


def find_matching_voxel_file(folder, frame_id):
    """按帧号在 target/radar voxel 目录中查找 .npz/.npy 文件。"""
    if not folder:
        return ""
    for ext in (".npz", ".npy"):
        path = os.path.join(folder, f"{frame_id}{ext}")
        if os.path.exists(path):
            return path
    return ""


def voxel_to_pointcloud(voxel, voxel_size, pc_range, occ_threshold=0.1, empty_fallback_topk=0):
    # TODO:  occ_threshold 的默认值需要根据训练时的体素化参数进行调整，过高可能导致生成点云过于稀疏，过低则可能引入大量噪声点。
    # NOTE: 输入 voxel 形状为 (C, Z, H, W)。
    occ = voxel[0]
    intensity = voxel[1]

    # NOTE: 若未显式指定体素尺寸，则按当前输出网格分辨率与 pc_range 自动反推。
    # NOTE: 这样可避免“训练/推理重采样后仍使用原始 0.2m”导致的坐标尺度失真。
    if voxel_size is None:
        voxel_size = [
            (pc_range[3] - pc_range[0]) / max(occ.shape[1], 1),
            (pc_range[4] - pc_range[1]) / max(occ.shape[2], 1),
            (pc_range[5] - pc_range[2]) / max(occ.shape[0], 1),
        ]

    occ_mask = occ > occ_threshold
    used_topk_fallback = False
    if not np.any(occ_mask):
        if empty_fallback_topk <= 0:
            return np.zeros((0, 4), dtype=np.float32), used_topk_fallback

        # HACK: 当阈值筛选后为空点云时，回退到 top-k 占据体素避免评估中断。
        used_topk_fallback = True
        flat_occ = occ.reshape(-1)
        k = int(min(max(empty_fallback_topk, 1), flat_occ.shape[0]))
        topk_idx = np.argpartition(flat_occ, -k)[-k:]
        z_idx, x_idx, y_idx = np.unravel_index(topk_idx, occ.shape)
        x = pc_range[0] + (x_idx + 0.5) * voxel_size[0]
        y = pc_range[1] + (y_idx + 0.5) * voxel_size[1]
        z = pc_range[2] + (z_idx + 0.5) * voxel_size[2]
        inten = intensity[z_idx, x_idx, y_idx]
        pcl = np.stack([x, y, z, inten], axis=1).astype(np.float32)
        return pcl, used_topk_fallback

    z_idx, x_idx, y_idx = np.where(occ_mask)
    x = pc_range[0] + (x_idx + 0.5) * voxel_size[0]
    y = pc_range[1] + (y_idx + 0.5) * voxel_size[1]
    z = pc_range[2] + (z_idx + 0.5) * voxel_size[2]
    inten = intensity[z_idx, x_idx, y_idx]

    return np.stack([x, y, z, inten], axis=1).astype(np.float32), used_topk_fallback


def compute_chamfer(pcl_a, pcl_b):
    # NOTE: 倒角距离（Chamfer）用于衡量生成点云与激光雷达（LiDAR）真值点云的几何一致性。
    if cKDTree is None:
        raise RuntimeError("scipy is required for chamfer distance.")
    if pcl_a.shape[0] == 0 or pcl_b.shape[0] == 0:
        return float('inf')

    tree_a = cKDTree(pcl_a[:, :3])
    tree_b = cKDTree(pcl_b[:, :3])
    dists_a, _ = tree_b.query(pcl_a[:, :3], k=1)
    dists_b, _ = tree_a.query(pcl_b[:, :3], k=1)
    return float(dists_a.mean() + dists_b.mean())


def centroid_delta(pcl_a, pcl_b):
    """返回 a 相对 b 的质心偏移；空点云时返回 NaN。"""
    if pcl_a.shape[0] == 0 or pcl_b.shape[0] == 0:
        return float('nan'), float('nan'), float('nan')
    ca = np.mean(pcl_a[:, :3], axis=0)
    cb = np.mean(pcl_b[:, :3], axis=0)
    delta = ca - cb
    return float(delta[0]), float(delta[1]), float(delta[2])


def main():
    parser = argparse.ArgumentParser(description="Radar Data Inference")
    parser.add_argument("--vae_ckpt", type=str, required=True, help="Path to VAE checkpoint")
    parser.add_argument("--model_ckpt", type=str, required=True, help="Path to LDM/CD checkpoint")
    parser.add_argument("--model_type", type=str, default="ldm", choices=["ldm", "cd"], help="Model type")
    parser.add_argument("--dataset_dir", type=str, default="./Data/NTU4DRadLM_Pre", 
                        help="Dataset directory for condition data")
    parser.add_argument(
        "--calibration_dir",
        type=str,
        default="",
        help="正式 LiDAR-to-Thermal 外参和 thermal 内参目录",
    )
    parser.add_argument(
        "--deployment_scene_dir",
        type=str,
        default="",
        help="正式部署场景目录；用于把输入、manifest 与 checkpoint 标定身份交叉绑定",
    )
    parser.add_argument("--radar_voxel_dir", type=str, default="",
                        help="If set, load radar_voxel files from this directory and run per-sample inference")
    parser.add_argument("--max_files", type=int, default=0,
                        help="Max number of radar files to run in per-sample mode (0 means all)")
    parser.add_argument("--raw_livox_dir", type=str, default="",
                        help="Raw livox_lidar directory for comparison")
    parser.add_argument("--lidar_index_file", type=str, default="",
                        help="lidar_index_sequence.txt for mapping preprocessed index to raw LiDAR")
    parser.add_argument("--raw_radar_dir", type=str, default="",
                        help="Raw radar_pcl directory for mapping preprocessed index to raw Radar")
    parser.add_argument("--radar_index_file", type=str, default="",
                        help="radar_index_sequence.txt for mapping preprocessed index to raw Radar")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--steps", type=int, default=40, help="Sampling steps (LDM: 40, CD: 1-4)")
    parser.add_argument("--sampler", type=str, default="heun", choices=["heun", "euler"], help="Sampler type")
    parser.add_argument("--output_dir", type=str, default="./Result/inference_results", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument(
        "--vae_config_type",
        choices=["ultra_lightweight", "lightweight", "standard"],
        default=None,
        help="仅用于缺少 vae_config 的历史 VAE checkpoint",
    )
    parser.add_argument("--use_condition", action="store_true", help="Use condition data from dataset")
    parser.add_argument("--save_pointcloud", action="store_true", help="Save point cloud .npy per sample")
    parser.add_argument("--save_voxel", action="store_true", help="Save voxel .npy per sample")
    parser.add_argument("--save_uncertainty", action="store_true",
                        help="Save multimodal uncertainty/confidence .npy per sample when available")
    parser.add_argument("--compare_with_lidar", action="store_true", help="Compare with raw livox point clouds")
    parser.add_argument("--compare_with_target", action="store_true",
                        help="Compare prediction and radar baseline with processed target_voxel")
    parser.add_argument("--occ_threshold", type=float, default=0.1, help="Occupancy threshold for point cloud")
    parser.add_argument("--empty_fallback_topk", type=int, default=0,
                        help="If threshold yields empty point cloud, fallback to top-k occupancy voxels (0 disables)")
    parser.add_argument("--voxel_size", type=float, nargs=3, default=None,
                        help="Voxel size [x,y,z] for point cloud conversion; default is auto-derived from output grid and pc_range")
    parser.add_argument("--pc_range", type=float, nargs=6, default=None,
                        help="Model/output point cloud range used for voxel-to-point conversion")
    parser.add_argument("--source_pc_range", type=float, nargs=6, default=None,
                        help="Original preprocessing range before optional physical crop; defaults to --pc_range")
    parser.add_argument("--target_size", type=int, nargs=3, default=None,
                        help="Model tensor size as [Z, X, Y] after loading/cropping")
    parser.add_argument("--train_duration_seconds", type=float, default=-1.0,
                        help="Optional training duration in seconds for experiment tracking (negative means unknown)")
    parser.add_argument("--seed", type=int, default=-1,
                        help="Random seed for reproducible inference (negative disables fixed seed)")
    parser.add_argument("--target_voxel_dir", type=str, default="",
                        help="Directory containing target_voxel files for offline comparison")
    parser.add_argument("--use_multimodal_meta", action="store_true",
                        help="Use IR/calibration metadata when the checkpoint is multimodal")
    parser.add_argument(
        "--require_real_ir",
        action="store_true",
        help="正式多模态模式：在写输出前要求全部帧具有真实 IR 和 thermal 外参",
    )
    parser.add_argument(
        "--allow_legacy_radar_units",
        action="store_true",
        help="仅用于历史 checkpoint/诊断；正式入口不得启用",
    )
    parser.add_argument(
        "--allow_legacy_checkpoint_pickle",
        action="store_true",
        help="仅用于可信历史 checkpoint 诊断；正式入口不得启用",
    )
    parser.add_argument(
        "--allow_formal_mini_checkpoint",
        action="store_true",
        help="仅供 test/mini-test 离线 smoke；正式部署入口不得启用",
    )
    parser.add_argument("--report_task_metrics", action="store_true",
                        help="Report shared-visible obstacle metrics in inference_metrics.csv")
    parser.add_argument("--task_range_bins", type=str, default="0-20,20-40,40-80,80-120",
                        help="Range bins for task metrics, formatted as start-end,start-end")
    parser.add_argument("--task_z_mins", type=str, default="-1,0",
                        help="Comma-separated z_min filters for task metrics")

    try:
        reject_removed_oracle_arguments(sys.argv[1:])
    except ValueError as exc:
        parser.error(str(exc))
    args = parser.parse_args()
    if args.require_real_ir and not args.radar_voxel_dir:
        parser.error("--require_real_ir requires --radar_voxel_dir")
    if args.require_real_ir and not args.calibration_dir:
        parser.error("--require_real_ir requires explicit --calibration_dir")
    if args.require_real_ir and not args.deployment_scene_dir:
        parser.error("--require_real_ir requires explicit --deployment_scene_dir")
    if args.require_real_ir and not args.save_voxel:
        parser.error("--require_real_ir formal deployment requires --save_voxel")
    if args.allow_formal_mini_checkpoint and not args.require_real_ir:
        parser.error(
            "--allow_formal_mini_checkpoint requires --require_real_ir strict inputs"
        )
    if args.seed >= 0:
        set_random_seed(args.seed)
    
    # NOTE: 初始化模型与采样器。
    generator = RadarGenerator(
        vae_path=args.vae_ckpt,
        model_path=args.model_ckpt,
        model_type=args.model_type,
        device=args.device,
        target_size=args.target_size,
        pc_range=args.pc_range,
        source_pc_range=args.source_pc_range,
        vae_fallback_config_type=args.vae_config_type,
        allow_legacy_radar_units=args.allow_legacy_radar_units,
        allow_legacy_checkpoint_pickle=args.allow_legacy_checkpoint_pickle,
        allow_formal_mini_checkpoint=args.allow_formal_mini_checkpoint,
    )
    args.target_size = generator.target_size
    args.source_pc_range = generator.source_pc_range
    args.pc_range = generator.pc_range
    
    if args.radar_voxel_dir:
        # NOTE: 逐文件推理模式：每个 radar_voxel 输入生成一个对应输出。
        radar_files = sorted([
            f for f in os.listdir(args.radar_voxel_dir)
            if f.endswith('.npy') or f.endswith('.npz')
        ])

        if args.max_files > 0:
            radar_files = radar_files[:args.max_files]

        if not radar_files:
            raise RuntimeError(f"No radar voxel files found in {args.radar_voxel_dir}")
        needs_target_voxel = args.compare_with_target
        if needs_target_voxel and (not args.target_voxel_dir or not os.path.isdir(args.target_voxel_dir)):
            raise RuntimeError(
                "--compare_with_target requires a valid --target_voxel_dir"
            )

        lidar_files = []
        lidar_indices = []
        raw_radar_files = []
        radar_indices = []
        if args.raw_radar_dir:
            raw_radar_files = sorted([
                f for f in os.listdir(args.raw_radar_dir) if f.endswith('.npy')
            ])
            if args.radar_index_file:
                with open(args.radar_index_file, 'r', encoding='utf-8') as f:
                    radar_indices = [int(line.strip()) for line in f.readlines()]
        if args.compare_with_lidar:
            # NOTE: 当开启对比评估时，按索引映射原始 LiDAR 帧并导出 CSV 指标。
            if not args.raw_livox_dir:
                raise ValueError("--raw_livox_dir is required when --compare_with_lidar is set")
            lidar_files = sorted([
                f for f in os.listdir(args.raw_livox_dir) if f.endswith('.npy')
            ])
            if args.lidar_index_file:
                with open(args.lidar_index_file, 'r', encoding='utf-8') as f:
                    lidar_indices = [int(line.strip()) for line in f.readlines()]

        if args.require_real_ir:
            if generator.data_protocol is None:
                raise RuntimeError(
                    "--require_real_ir 正式部署要求 formal_chain_v2 checkpoint data_protocol"
                )
            generator.deployment_identity = assert_formal_deployment_identity(
                scene_dir=args.deployment_scene_dir,
                radar_voxel_dir=args.radar_voxel_dir,
                calibration_dir=args.calibration_dir,
                data_protocol=generator.data_protocol,
            )
            preflight_real_ir_inputs(
                generator.model,
                [os.path.join(args.radar_voxel_dir, name) for name in radar_files],
                args.calibration_dir,
            )
            generator.radar_observed_identity = preflight_radar_observed_inputs(
                [os.path.join(args.radar_voxel_dir, name) for name in radar_files],
                calibration_dir=args.calibration_dir,
                target_size=args.target_size,
                source_pc_range=args.source_pc_range,
                model_pc_range=args.pc_range,
            )

        args.output_dir = prepare_fresh_output_dir(args.output_dir)

        legacy_evaluation = _uses_legacy_evaluation(args)
        merged_csv_path = os.path.join(args.output_dir, inference_csv_name(args))
        merged_csv_file = open(merged_csv_path, 'w', newline='')
        merged_writer = csv.writer(merged_csv_file)
        if legacy_evaluation:
            metric_header = [
                "index",
                "radar_file",
                "radar_point_count",
                "effective_occ_threshold",
                "target_file",
                "target_point_count",
                "pred_target_chamfer",
                "radar_target_chamfer",
                "pred_target_count_ratio",
                "pred_target_dx",
                "pred_target_dy",
                "pred_target_dz",
                "radar_target_dx",
                "radar_target_dy",
                "radar_target_dz",
                "lidar_file",
                "inference_seconds",
                "pred_point_count",
                "lidar_point_count",
                "raw_lidar_chamfer",
                "is_empty_frame",
                "used_topk_fallback",
                "train_duration_seconds",
                "total_infer_seconds",
                "avg_infer_seconds",
                "avg_pred_point_count",
                "empty_frame_rate",
                "mean_pred_target_chamfer",
                "mean_radar_target_chamfer",
                "mean_raw_lidar_chamfer",
            ]
            if args.report_task_metrics:
                append_task_metric_headers(metric_header)
        else:
            metric_header = list(RUNTIME_FIELDS)
        merged_writer.writerow(metric_header)

        log_path = os.path.join(args.output_dir, "inference_runtime.log")
        log_file = open(log_path, 'w', encoding='utf-8')
        log_file.write("=== Inference Runtime Log ===\n")
        log_file.write(f"time: {datetime.now().isoformat()}\n")
        log_file.write(f"model_type: {args.model_type}\n")
        log_file.write(f"vae_ckpt: {args.vae_ckpt}\n")
        log_file.write(f"model_ckpt: {args.model_ckpt}\n")
        log_file.write(f"device: {args.device}\n")
        log_file.write(f"steps: {args.steps}\n")
        log_file.write(f"sampler: {args.sampler}\n")
        log_file.write(f"seed: {args.seed}\n")
        log_file.write(f"max_files: {args.max_files}\n")
        log_file.write(f"occ_threshold: {args.occ_threshold}\n")
        log_file.write(f"target_size: {list(args.target_size)}\n")
        log_file.write(f"source_pc_range: {args.source_pc_range}\n")
        log_file.write(f"model_pc_range: {args.pc_range}\n")
        log_file.write(f"compare_with_target: {int(args.compare_with_target)}\n")
        log_file.write(f"target_voxel_dir: {args.target_voxel_dir}\n")
        log_file.write(f"empty_fallback_topk: {args.empty_fallback_topk}\n")
        log_file.write(f"train_duration_seconds: {args.train_duration_seconds:.3f}\n")
        log_file.write(f"num_files: {len(radar_files)}\n")
        log_file.write("\n")

        total_infer_sec = 0.0
        fallback_count = 0
        empty_frame_count = 0
        total_pred_points = 0
        target_chamfer_values = []
        radar_target_chamfer_values = []
        raw_lidar_chamfer_values = []
        task_acc = {field: [] for field in TASK_METRIC_FIELDS}
        task_range_bins = parse_range_bins(args.task_range_bins)
        observed_mask_records = []
        prediction_voxel_records = []

        file_pbar = tqdm(
            enumerate(radar_files),
            total=len(radar_files),
            desc="Inferring files",
            unit="file",
        )

        for i, fname in file_pbar:
            radar_path = os.path.join(args.radar_voxel_dir, fname)
            deployment_observed_mask = None
            if args.require_real_ir:
                endpoint_mask = load_radar_endpoint_mask(
                    radar_path,
                    target_size=args.target_size,
                    source_pc_range=args.source_pc_range,
                    model_pc_range=args.pc_range,
                )
                deployment_observed_mask = build_radar_ray_observed_mask(
                    endpoint_mask,
                    pc_range=args.pc_range,
                    radar_origin_lidar_m=generator.radar_observed_identity[
                        "radar_origin_lidar_m"
                    ],
                )
            radar_point_count = ""
            if raw_radar_files:
                radar_raw_file = None
                if radar_indices:
                    if i < len(radar_indices):
                        idx = radar_indices[i]
                        if idx < len(raw_radar_files):
                            radar_raw_file = raw_radar_files[idx]
                elif i < len(raw_radar_files):
                    radar_raw_file = raw_radar_files[i]

                if radar_raw_file:
                    radar_raw_path = os.path.join(args.raw_radar_dir, radar_raw_file)
                    radar_point_count = count_raw_radar_pcl_points(radar_raw_path)
            else:
                # NOTE: 未提供 raw radar_pcl 时，回退为 radar_voxel 占据点计数。
                radar_point_count = count_radar_points(radar_path)
            condition_data = load_radar_voxel_as_tensor(
                radar_path,
                generator.device,
                target_size=args.target_size,
                source_pc_range=args.source_pc_range,
                model_pc_range=args.pc_range,
                radar_normalization=generator.radar_normalization,
                radar_normalization_sha256=generator.radar_normalization_sha256,
                allow_legacy_radar_units=generator.allow_legacy_radar_units,
            )
            condition_data = condition_data.unsqueeze(0)
            meta_dict = (
                load_multimodal_meta_for_radar(
                    radar_path,
                    generator.device,
                    require_real_ir=args.require_real_ir,
                    calibration_dir=args.calibration_dir,
                )
                if args.use_multimodal_meta or args.require_real_ir
                else None
            )

            file_start = time.perf_counter()

            generated = generator.generate(
                condition=condition_data,
                num_samples=1,
                steps=args.steps,
                sampler=args.sampler,
                show_sampling_progress=False,
                meta_dict=meta_dict,
            )

            file_infer_sec = time.perf_counter() - file_start
            total_infer_sec += file_infer_sec
            log_file.write(f"file[{i + 1}/{len(radar_files)}] {fname} infer_sec={file_infer_sec:.6f}\n")
            file_pbar.set_postfix_str(f"{i + 1}/{len(radar_files)} | {file_infer_sec:.3f}s")

            sample = generated[0].detach().cpu().numpy()
            pcl = np.zeros((0, 4), dtype=np.float32)
            used_topk_fallback = False
            effective_occ_threshold = float(args.occ_threshold)
            task_values = {}

            if args.save_voxel:
                frame_id = os.path.splitext(fname)[0].replace("_voxel", "")
                voxel_file = f"{frame_id}_voxel.npy"
                out_voxel = os.path.join(args.output_dir, voxel_file)
                _save_npy_atomic(out_voxel, sample)
                prediction_voxel_records.append(
                    {
                        "frame_id": frame_id,
                        "file": voxel_file,
                        "sha256": sha256_file(out_voxel),
                        "shape_czxy": [int(value) for value in sample.shape],
                        "dtype": str(sample.dtype),
                    }
                )
            if deployment_observed_mask is not None:
                frame_id = os.path.splitext(fname)[0].replace("_voxel", "")
                mask_file = f"{frame_id}_observed_mask.npy"
                mask_path = os.path.join(args.output_dir, mask_file)
                _save_npy_atomic(mask_path, deployment_observed_mask)
                observed_mask_records.append(
                    {
                        "frame_id": frame_id,
                        "file": mask_file,
                        "sha256": sha256_file(mask_path),
                        "observed_voxels": int(
                            np.count_nonzero(deployment_observed_mask)
                        ),
                    }
                )
            if (args.save_voxel or args.save_uncertainty) and generator.last_uncertainty is not None:
                unc = generator.last_uncertainty.get(
                    "variance",
                    generator.last_uncertainty.get("log_var", generator.last_uncertainty.get("confidence")),
                )
                if torch.is_tensor(unc):
                    unc_np = unc[0].detach().cpu().numpy()
                    out_unc = os.path.join(args.output_dir, f"{os.path.splitext(fname)[0]}_uncertainty.npy")
                    _save_npy_atomic(out_unc, unc_np)

            if args.save_pointcloud or args.compare_with_lidar or args.compare_with_target:
                pcl, used_topk_fallback = voxel_to_pointcloud(
                    sample,
                    voxel_size=args.voxel_size,
                    pc_range=args.pc_range,
                    occ_threshold=effective_occ_threshold,
                    empty_fallback_topk=args.empty_fallback_topk,
                )
                if used_topk_fallback:
                    fallback_count += 1
                if args.save_pointcloud:
                    out_pcl = os.path.join(args.output_dir, f"{os.path.splitext(fname)[0]}_pcl.npy")
                    _save_npy_atomic(out_pcl, pcl)

            pred_point_count = int(pcl.shape[0])
            total_pred_points += pred_point_count
            is_empty_frame = int(pred_point_count == 0)
            empty_frame_count += is_empty_frame

            target_file = ""
            target_point_count = ""
            pred_target_chamfer = ""
            radar_target_chamfer = ""
            pred_target_count_ratio = ""
            pred_target_dx = ""
            pred_target_dy = ""
            pred_target_dz = ""
            radar_target_dx = ""
            radar_target_dy = ""
            radar_target_dz = ""

            if args.compare_with_target:
                frame_id = os.path.splitext(fname)[0]
                target_path = find_matching_voxel_file(args.target_voxel_dir, frame_id)
                if target_path:
                    target_file = os.path.basename(target_path)
                    target_voxel = load_voxel_as_czxy(
                        target_path,
                        generator.device,
                        mask_channel=3,
                        target_size=args.target_size,
                        source_pc_range=args.source_pc_range,
                        model_pc_range=args.pc_range,
                    )
                    radar_voxel = load_voxel_as_czxy(
                        radar_path,
                        generator.device,
                        target_size=args.target_size,
                        source_pc_range=args.source_pc_range,
                        model_pc_range=args.pc_range,
                    )
                    target_pcl, _ = voxel_to_pointcloud(
                        target_voxel,
                        voxel_size=args.voxel_size,
                        pc_range=args.pc_range,
                        occ_threshold=args.occ_threshold,
                        empty_fallback_topk=0,
                    )
                    radar_pcl, _ = voxel_to_pointcloud(
                        radar_voxel,
                        voxel_size=args.voxel_size,
                        pc_range=args.pc_range,
                        occ_threshold=args.occ_threshold,
                        empty_fallback_topk=0,
                    )
                    target_point_count = int(target_pcl.shape[0])
                    if target_point_count > 0:
                        pred_target_count_ratio = pred_point_count / max(target_point_count, 1)
                        pred_target_count_ratio = f"{pred_target_count_ratio:.6f}"
                    pred_target_val = compute_chamfer(pcl, target_pcl)
                    radar_target_val = compute_chamfer(radar_pcl, target_pcl)
                    pred_target_chamfer = f"{pred_target_val:.6f}"
                    radar_target_chamfer = f"{radar_target_val:.6f}"
                    if np.isfinite(pred_target_val):
                        target_chamfer_values.append(pred_target_val)
                    if np.isfinite(radar_target_val):
                        radar_target_chamfer_values.append(radar_target_val)

                    pdx, pdy, pdz = centroid_delta(pcl, target_pcl)
                    rdx, rdy, rdz = centroid_delta(radar_pcl, target_pcl)
                    pred_target_dx = f"{pdx:.6f}" if np.isfinite(pdx) else ""
                    pred_target_dy = f"{pdy:.6f}" if np.isfinite(pdy) else ""
                    pred_target_dz = f"{pdz:.6f}" if np.isfinite(pdz) else ""
                    radar_target_dx = f"{rdx:.6f}" if np.isfinite(rdx) else ""
                    radar_target_dy = f"{rdy:.6f}" if np.isfinite(rdy) else ""
                    radar_target_dz = f"{rdz:.6f}" if np.isfinite(rdz) else ""
                    if args.report_task_metrics:
                        near_pred = filter_points_by_band(pcl[:, :3], args.pc_range, x_min=0.0, x_max=20.0, z_min=-1.0)
                        near_target = filter_points_by_band(target_pcl[:, :3], args.pc_range, x_min=0.0, x_max=20.0, z_min=-1.0)
                        prf = occupancy_prf(near_pred, near_target, args.pc_range, cell_size=0.5)
                        iou = task_bev_iou(near_pred, near_target, args.pc_range, cell_size=0.5)
                        nn = nearest_neighbor_metrics(near_pred, near_target, thresholds=(2.0,))
                        task_values = {
                            "task_near_recall_mean": prf["recall"],
                            "task_near_precision_mean": prf["precision"],
                            "task_near_bev_iou_mean": iou["bev_iou"],
                            "task_near_nn_mean": nn["nn_mean"],
                            "task_near_match_ratio_2_mean": nn["match_ratio_2"],
                        }
                        for key, value in task_values.items():
                            if np.isfinite(value):
                                task_acc[key].append(float(value))
                        if generator.last_uncertainty is not None:
                            uncertainty_tensor = generator.last_uncertainty.get("variance")
                            if torch.is_tensor(uncertainty_tensor):
                                calibration = uncertainty_calibration_metrics(
                                    sample[0],
                                    target_voxel[0],
                                    uncertainty_tensor[0].detach().cpu().numpy(),
                                    occ_threshold=effective_occ_threshold,
                                )
                                for key in (
                                    "uncertainty_ece",
                                    "uncertainty_brier",
                                    "uncertainty_nll",
                                    "uncertainty_error_corr",
                                ):
                                    value = calibration[key]
                                    task_values[key] = value
                                    if np.isfinite(value):
                                        task_acc[key].append(float(value))
                else:
                    log_file.write(f"warning: target voxel not found for {fname}, skip processed target metrics\n")

            lidar_point_count = ""
            raw_lidar_chamfer = ""
            lidar_file = ""

            if args.compare_with_lidar:
                lidar_file = None
                if lidar_indices:
                    if i < len(lidar_indices):
                        idx = lidar_indices[i]
                        if idx < len(lidar_files):
                            lidar_file = lidar_files[idx]
                elif i < len(lidar_files):
                    lidar_file = lidar_files[i]

                if lidar_file:
                    lidar_path = os.path.join(args.raw_livox_dir, lidar_file)
                    lidar_pcl = np.load(lidar_path).astype(np.float32)
                    lidar_point_count = int(lidar_pcl.shape[0])
                    chamfer_val = compute_chamfer(pcl, lidar_pcl)
                    raw_lidar_chamfer = f"{chamfer_val:.6f}"
                    if np.isfinite(chamfer_val):
                        raw_lidar_chamfer_values.append(chamfer_val)

            if legacy_evaluation:
                row = [
                    i,
                    fname,
                    radar_point_count,
                    f"{effective_occ_threshold:.8f}",
                    target_file,
                    target_point_count,
                    pred_target_chamfer,
                    radar_target_chamfer,
                    pred_target_count_ratio,
                    pred_target_dx,
                    pred_target_dy,
                    pred_target_dz,
                    radar_target_dx,
                    radar_target_dy,
                    radar_target_dz,
                    lidar_file,
                    f"{file_infer_sec:.6f}",
                    pred_point_count,
                    lidar_point_count,
                    raw_lidar_chamfer,
                    is_empty_frame,
                    int(
                        used_topk_fallback
                        if (
                            args.save_pointcloud
                            or args.compare_with_lidar
                            or args.compare_with_target
                        )
                        else 0
                    ),
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
                if args.report_task_metrics:
                    build_task_metric_row(row, metric_header, task_values)
            else:
                row = [
                    i,
                    fname,
                    radar_point_count,
                    f"{effective_occ_threshold:.8f}",
                    f"{file_infer_sec:.6f}",
                    pred_point_count,
                    is_empty_frame,
                    int(used_topk_fallback if args.save_pointcloud else 0),
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            merged_writer.writerow(row)

        merged_csv_file.flush()
        avg_infer_sec = total_infer_sec / max(len(radar_files), 1)
        avg_pred_point_count = total_pred_points / max(len(radar_files), 1)
        empty_frame_rate = empty_frame_count / max(len(radar_files), 1)
        mean_pred_target_chamfer = float(np.mean(target_chamfer_values)) if len(target_chamfer_values) > 0 else float('nan')
        mean_radar_target_chamfer = float(np.mean(radar_target_chamfer_values)) if len(radar_target_chamfer_values) > 0 else float('nan')
        mean_raw_lidar_chamfer = float(np.mean(raw_lidar_chamfer_values)) if len(raw_lidar_chamfer_values) > 0 else float('nan')

        summary_row = [""] * len(metric_header)
        summary_row[0] = "__summary__"
        if legacy_evaluation:
            summary_row[23] = f"{args.train_duration_seconds:.3f}"
            summary_row[24] = f"{total_infer_sec:.6f}"
            summary_row[25] = f"{avg_infer_sec:.6f}"
            summary_row[26] = f"{avg_pred_point_count:.3f}"
            summary_row[27] = f"{empty_frame_rate:.6f}"
            summary_row[28] = f"{mean_pred_target_chamfer:.6f}" if np.isfinite(mean_pred_target_chamfer) else ""
            summary_row[29] = f"{mean_radar_target_chamfer:.6f}" if np.isfinite(mean_radar_target_chamfer) else ""
            summary_row[30] = f"{mean_raw_lidar_chamfer:.6f}" if np.isfinite(mean_raw_lidar_chamfer) else ""
            if args.report_task_metrics:
                task_summary = {
                    key: float(np.mean(values)) if values else float("nan")
                    for key, values in task_acc.items()
                }
                build_task_metric_summary_row(summary_row, metric_header, task_summary)
        else:
            summary_row[8] = f"{args.train_duration_seconds:.3f}"
            summary_row[9] = f"{total_infer_sec:.6f}"
            summary_row[10] = f"{avg_infer_sec:.6f}"
            summary_row[11] = f"{avg_pred_point_count:.3f}"
            summary_row[12] = f"{empty_frame_rate:.6f}"
        merged_writer.writerow(summary_row)

        _write_json_atomic(
            os.path.join(args.output_dir, "inference_run.json"),
            build_inference_run_metadata(
                args,
                generator,
                len(radar_files),
                observed_mask_records=(
                    observed_mask_records if args.require_real_ir else None
                ),
                prediction_voxel_records=(
                    prediction_voxel_records if args.save_voxel else None
                ),
            ),
        )

        log_file.write("\n")
        log_file.write(f"total_infer_sec: {total_infer_sec:.6f}\n")
        log_file.write(f"avg_infer_sec_per_file: {avg_infer_sec:.6f}\n")
        log_file.write(f"avg_pred_point_count: {avg_pred_point_count:.3f}\n")
        log_file.write(f"empty_frame_rate: {empty_frame_rate:.6f}\n")
        log_file.write(
            f"mean_pred_target_chamfer: {mean_pred_target_chamfer:.6f}\n"
            if np.isfinite(mean_pred_target_chamfer)
            else "mean_pred_target_chamfer: \n"
        )
        log_file.write(
            f"mean_radar_target_chamfer: {mean_radar_target_chamfer:.6f}\n"
            if np.isfinite(mean_radar_target_chamfer)
            else "mean_radar_target_chamfer: \n"
        )
        log_file.write(
            f"mean_raw_lidar_chamfer: {mean_raw_lidar_chamfer:.6f}\n"
            if np.isfinite(mean_raw_lidar_chamfer)
            else "mean_raw_lidar_chamfer: \n"
        )
        log_file.write(f"topk_fallback_frames: {fallback_count}\n")
        log_file.flush()
        file_pbar.close()

        merged_csv_file.close()
        log_file.close()
        print(f"Saved merged metrics csv to {merged_csv_path}")
        print(f"Saved runtime log to {log_path}")

    else:
        args.output_dir = prepare_fresh_output_dir(args.output_dir)
        # NOTE: 简单模式下可从验证集抽取一个条件样本。
        if args.use_condition:
            print(f"Loading dataset from {args.dataset_dir}...")
            dataset = NTU4DRadLM_VoxelDataset(
                args.dataset_dir,
                split='val',
                target_size=args.target_size,
                source_pc_range=args.source_pc_range,
                model_pc_range=args.pc_range,
                radar_normalization=generator.radar_normalization,
                radar_normalization_sha256=generator.radar_normalization_sha256,
                allow_legacy_radar_units=generator.allow_legacy_radar_units,
            )
            dataloader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=True,
                collate_fn=collate_voxel_samples,
            )
            # FIXME: 当验证集为空时这里会触发 StopIteration，后续可改为显式空集检查。
            batch = next(iter(dataloader))
            condition_data = batch[1]
            meta_dict = batch[2] if args.use_multimodal_meta and len(batch) > 2 else None
        else:
            condition_data = None
            meta_dict = None

        # NOTE: 运行生成并保存体素结果。
        generated = generator.generate(
            condition=condition_data,
            num_samples=args.num_samples,
            steps=args.steps,
            sampler=args.sampler,
            meta_dict=meta_dict,
        )

        output_path = os.path.join(args.output_dir, f"{args.model_type}_samples_{args.steps}steps.npy")
        _save_npy_atomic(output_path, generated.cpu().numpy())
        print(f"\nSaved {args.num_samples} samples to {output_path}")
        print(f"Shape: {generated.shape}")
        print(f"Range: [{generated.min():.3f}, {generated.max():.3f}]")


if __name__ == "__main__":
    main()
