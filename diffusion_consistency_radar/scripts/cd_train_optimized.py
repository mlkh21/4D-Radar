# -- coding: utf-8 --
"""
LDM 初始化、边界锚定的 EMA Consistency 训练脚本

改进点：
1. 清晰的训练语义 - LDM 仅初始化 CD/EMA，训练目标来自 CD EMA
2. sigma_min 硬边界与 observed-target 重建锚点排除常数零损失解
3. 显存优化 - 梯度累积、检查点、混合精度
4. checkpoint 显式记录初始化来源、一致性目标和防塌缩诊断
5. 模块化设计 - 易于理解和维护
"""

import sys
import os
import atexit

# 直接执行本文件时同时暴露仓库根和包目录：前者支持
# diffusion_consistency_radar.*，后者兼容既有 cm.* 导入。
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_ROOT = os.path.dirname(_SCRIPT_DIR)
_PROJECT_ROOT = os.path.dirname(_PACKAGE_ROOT)
for _import_root in (_PACKAGE_ROOT, _PROJECT_ROOT):
    if _import_root not in sys.path:
        sys.path.insert(0, _import_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
import argparse
import logging
import csv
import math
import time
from typing import Any, Dict, Optional, Tuple
from tqdm import tqdm
import yaml

from diffusion_consistency_radar.cm.unet_optimized import OptimizedUNetModel
from diffusion_consistency_radar.cm.multimodal_fusion import (
    CompleteDualModalityPerceptionNet,
)
from diffusion_consistency_radar.cm.karras_diffusion import KarrasDenoiser
from diffusion_consistency_radar.cm.vae_3d import (
    VAE3D,
    build_vae_from_checkpoint,
    create_lightweight_vae_config,
    create_standard_vae_config,
    create_ultra_lightweight_vae_config,
)
from diffusion_consistency_radar.cm.dataset_loader import (
    NTU4DRadLM_VoxelDataset,
    collate_voxel_samples,
)
from diffusion_consistency_radar.checkpoint_chain import (
    FORMAL_CHECKPOINT_PROTOCOL,
    FORMAL_MINI_CHECKPOINT_PROTOCOL,
    assert_checkpoint_training_identity,
    build_formal_mini_selection,
    build_formal_stage_training_selection,
    resolve_training_checkpoint_protocol,
    safe_torch_load as safe_checkpoint_load,
    sha256_file,
    validate_checkpoint_data_protocol,
    validate_deployment_validation_receipt,
    validate_formal_stage_training_selection,
)
from diffusion_consistency_radar.radar_normalization import (
    RadarNormalizationError,
    assert_checkpoint_radar_normalization,
    assert_same_radar_normalization,
    load_radar_normalization_artifact,
    radar_normalization_from_checkpoint,
)
from diffusion_consistency_radar.formal_data_protocol import (
    load_formal_data_protocol_artifact,
)
from diffusion_consistency_radar.cd_validation_protocol import (
    CD_DENOISING_DIAGNOSTIC_PROTOCOL,
    CD_VALIDATION_PROTOCOL,
    CD_VALIDATION_SELECTOR,
    CD_VALIDATION_SPLIT,
)
from diffusion_consistency_radar.deployment_validation import (
    build_deployment_validation_selection,
    karras_sigma_schedule,
    resolve_deployment_validation_config,
    validate_deployment_metrics,
)
from diffusion_consistency_radar.cd_training_protocol import (
    CD_DENOISING_PARAMETERIZATION,
    CD_TRAINING_SEMANTICS,
    apply_cd_boundary_parameterization,
    build_cd_collapse_diagnostic,
    build_cd_collapse_diagnostics_receipt,
    resolve_cd_consistency_config,
    validate_cd_consistency_receipt,
)
from diffusion_consistency_radar.occupancy_threshold_artifact import (
    DEFAULT_THRESHOLD_CANDIDATES,
    THRESHOLD_SWEEP_PROTOCOL,
    threshold_sweep_batch_counts,
    threshold_sweep_metrics,
    validate_checkpoint_threshold_sweep,
    validate_threshold_candidates,
    build_threshold_artifact,
    write_threshold_artifact,
)
from diffusion_consistency_radar.temporal_split import (
    limit_frame_ids_by_scene,
    load_temporal_split_artifact,
    split_frame_ids_by_scene,
)
from diffusion_consistency_radar.distributed_training import (
    DistributedContext,
    DistributedEvalSampler,
    WorldBatchPlan,
    assert_distributed_config_compatible,
    assert_resume_distributed_compatible,
    cleanup_distributed,
    distributed_barrier,
    distributed_checkpoint_metadata,
    deterministic_noise_from_sample_ids,
    initialize_distributed,
    prepare_model_for_distributed,
    reduce_named_sums,
    set_loader_epoch,
    unwrap_model,
    wrap_model_for_ddp,
)


CD_EMA_UPDATE_PROTOCOL = "named_parameter_and_buffer_ema_v1"


def _validated_cd_validation_metrics(
    metrics: Dict[str, Any],
    *,
    source: str,
) -> tuple:
    """校验 CD 部署选优所依赖的完整采样 observed-domain 指标。"""
    resolved = validate_deployment_metrics(metrics, source=f"{source} CD")
    return (
        resolved["deployment_latent_loss"],
        resolved["deployment_occupancy_iou"],
    )


def select_cd_deployment_weight_source(
    online_metrics: Dict[str, Any],
    ema_metrics: Dict[str, Any],
) -> str:
    """按 observed IoU 优先、latent loss 次优选择部署权重；完全相同取 EMA。"""
    online_loss, online_iou = _validated_cd_validation_metrics(
        online_metrics,
        source="online",
    )
    ema_loss, ema_iou = _validated_cd_validation_metrics(
        ema_metrics,
        source="EMA",
    )
    online_rank = (online_iou, -online_loss)
    ema_rank = (ema_iou, -ema_loss)
    return (
        "model_state_dict"
        if online_rank > ema_rank
        else "ema_model_state_dict"
    )


def resolve_cd_validation_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """解析决定 CD online/EMA 部署权重的完整一步采样配置。"""
    return resolve_deployment_validation_config(config, stage="cd")


def resolve_cd_denoising_diagnostic_config(
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """解析保留但不得参与部署选优的 target-latent 小噪声诊断。"""
    seed = config.get("validation_seed", 42)
    if type(seed) is not int or seed < 0:
        raise ValueError("cd.validation_seed 必须是非负整数")
    try:
        sigma = float(config.get("validation_sigma", 0.5))
        threshold = float(config.get("validation_occupancy_threshold", 0.5))
    except (TypeError, ValueError) as exc:
        raise ValueError("CD validation sigma/threshold 必须是有限数") from exc
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("cd.validation_sigma 必须为正有限数")
    if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError("cd.validation_occupancy_threshold 必须严格位于 (0,1)")
    return {
        "protocol": CD_DENOISING_DIAGNOSTIC_PROTOCOL,
        "split": CD_VALIDATION_SPLIT,
        "seed": seed,
        "sigma": sigma,
        "occupancy_threshold": threshold,
        "noise_identity": "sha256_sample_id_seed_v1",
        "threshold_candidates": list(
            validate_threshold_candidates(
                config.get(
                    "validation_threshold_candidates",
                    DEFAULT_THRESHOLD_CANDIDATES,
                )
            )
        ),
    }


def assert_cd_validation_checkpoint_protocol(
    checkpoint: Dict[str, Any],
    *,
    current_config: Dict[str, Any],
    require_formal: bool,
) -> Optional[Dict[str, Any]]:
    """验证 CD resume 的 online/EMA 选择协议并返回可恢复状态。"""
    saved = checkpoint.get("cd_validation") if isinstance(checkpoint, dict) else None
    if saved is None:
        if require_formal:
            raise ValueError("formal CD checkpoint 缺少 cd_validation")
        return None
    if not isinstance(saved, dict):
        raise ValueError("CD checkpoint cd_validation 必须是字典")
    for field in (
        "protocol",
        "stage",
        "split",
        "selector",
        "selection_strategy",
        "noise_identity",
        "initial_latent",
        "sampler",
    ):
        if saved.get(field) != current_config.get(field):
            raise ValueError(
                f"CD checkpoint validation {field} 与当前配置不一致"
            )
    for field in ("seed", "steps", "frames_per_scene"):
        if (
            type(saved.get(field)) is not int
            or saved.get(field) != current_config.get(field)
        ):
            raise ValueError(
                f"CD checkpoint validation {field} 与当前配置不一致"
            )
    for field in ("sigma_min", "sigma_max", "rho", "occupancy_threshold"):
        try:
            saved_value = float(saved[field])
            current_value = float(current_config[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"CD checkpoint validation {field} 必须是有限数"
            ) from exc
        if (
            not math.isfinite(saved_value)
            or not math.isclose(
                saved_value,
                current_value,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                f"CD checkpoint validation {field} 与当前配置不一致"
            )
    if saved.get("threshold_candidates") != current_config.get(
        "threshold_candidates"
    ):
        raise ValueError("CD checkpoint validation threshold candidates 不一致")
    if saved.get("threshold_selection_constraints") != current_config.get(
        "threshold_selection_constraints"
    ):
        raise ValueError("CD checkpoint validation threshold selection constraints 不一致")
    if saved.get("selection") != current_config.get("selection"):
        raise ValueError("CD checkpoint deployment validation selection 不一致")
    metrics = saved.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != {
        "model_state_dict",
        "ema_model_state_dict",
    }:
        raise ValueError("CD checkpoint validation metrics 权重源集合不完整")
    selected_source = select_cd_deployment_weight_source(
        metrics["model_state_dict"],
        metrics["ema_model_state_dict"],
    )
    if (
        saved.get("selected_source") != selected_source
        or checkpoint.get("deployment_weight_source") != selected_source
    ):
        raise ValueError("CD checkpoint validation 选择结果与部署权重不一致")
    best_metrics = saved.get("best_selected_metrics")
    _validated_cd_validation_metrics(best_metrics, source="best selected")
    if require_formal:
        validate_checkpoint_threshold_sweep(
            checkpoint,
            expected_stage="cd",
            expected_weight_source=selected_source,
            expected_candidates=current_config["threshold_candidates"],
        )
    return {
        "metrics": {
            key: dict(value)
            for key, value in metrics.items()
        },
        "selected_source": selected_source,
        "best_selected_metrics": dict(best_metrics),
    }


def resolve_cd_observed_masks(
    observed_mask: Optional[torch.Tensor],
    target: torch.Tensor,
    latent: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """校验 persisted mask，并生成 voxel/latent 两级 observed domain。"""
    if observed_mask is None:
        return None, None
    mask = torch.as_tensor(observed_mask, device=target.device)
    if mask.ndim == 4:
        mask = mask.unsqueeze(1)
    if (
        target.ndim != 5
        or latent.ndim != 5
        or mask.ndim != 5
        or mask.shape[1] != 1
        or mask.shape[0] != target.shape[0]
        or tuple(mask.shape[-3:]) != tuple(target.shape[-3:])
    ):
        raise ValueError("CD observed_mask 必须与 target 的 B/Z/X/Y 一致")
    if not torch.isfinite(mask).all():
        raise ValueError("CD observed_mask 必须全部为有限数")
    if not torch.logical_or(mask == 0, mask == 1).all():
        raise ValueError("CD observed_mask 必须是严格 0/1")
    voxel_observed = mask.bool() | (target[:, 0:1] > 0.5)
    latent_observed = F.adaptive_max_pool3d(
        voxel_observed.float(),
        output_size=latent.shape[-3:],
    ).bool()
    if not latent_observed.any():
        raise ValueError("CD observed domain 为空")
    return voxel_observed, latent_observed


def _masked_latent_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    observed_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """只在可信 observed latent 域计算均方误差。"""
    if prediction.shape != target.shape:
        raise ValueError("CD latent prediction/target shape 不一致")
    squared_error = torch.square(prediction.float() - target.float())
    if observed_mask is None:
        return squared_error.mean()
    mask = torch.as_tensor(observed_mask, device=prediction.device).bool()
    if mask.ndim != prediction.ndim or mask.shape[0] != prediction.shape[0]:
        raise ValueError("CD latent observed mask 维度与 prediction 不一致")
    try:
        expanded = mask.expand_as(squared_error)
    except RuntimeError as exc:
        raise ValueError("CD latent observed mask 无法广播到 prediction") from exc
    if not expanded.any():
        raise ValueError("CD latent observed domain 为空")
    return squared_error[expanded].mean()


def compute_cd_training_losses(
    student_denoised: torch.Tensor,
    ema_target: torch.Tensor,
    latent_target: torch.Tensor,
    latent_observed_mask: Optional[torch.Tensor],
    consistency_config: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """组合 EMA consistency 与 observed-target 重建锚点，排除常数零损失。"""
    resolved = validate_cd_consistency_receipt(consistency_config)
    consistency_loss = _masked_latent_mse(
        student_denoised,
        ema_target,
        latent_observed_mask,
    )
    reconstruction_anchor_loss = _masked_latent_mse(
        student_denoised,
        latent_target,
        latent_observed_mask,
    )
    total_loss = (
        resolved["consistency_loss_weight"] * consistency_loss
        + resolved["reconstruction_anchor_weight"]
        * reconstruction_anchor_loss
    )
    return total_loss, {
        "consistency_loss": float(consistency_loss.detach().item()),
        "reconstruction_anchor_loss": float(
            reconstruction_anchor_loss.detach().item()
        ),
    }


def assert_cd_ema_update_protocol(
    checkpoint: Dict[str, Any],
    *,
    require_formal: bool,
) -> None:
    """拒绝以 parameters-only EMA 轨迹恢复正式 CD 训练。"""
    saved_protocol = checkpoint.get("ema_update_protocol")
    if require_formal and saved_protocol != CD_EMA_UPDATE_PROTOCOL:
        raise ValueError(
            "formal CD checkpoint EMA update protocol 不匹配："
            f"checkpoint={saved_protocol!r}, "
            f"current={CD_EMA_UPDATE_PROTOCOL!r}"
        )
    if (
        saved_protocol is not None
        and saved_protocol != CD_EMA_UPDATE_PROTOCOL
    ):
        raise ValueError(
            "CD checkpoint EMA update protocol 不匹配："
            f"checkpoint={saved_protocol!r}, "
            f"current={CD_EMA_UPDATE_PROTOCOL!r}"
        )


def safe_torch_load(path, map_location, *, allow_legacy_pickle=False):
    """CD 正式入口默认只允许 weights-only checkpoint。"""
    return safe_checkpoint_load(
        path,
        map_location=map_location,
        allow_legacy_pickle=allow_legacy_pickle,
    )


def atomic_torch_save(payload: Any, path: str) -> None:
    """在 checkpoint 同目录原子替换，避免中断留下半写文件。"""
    temp_path = f"{path}.tmp-{os.getpid()}-{time.time_ns()}"
    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def checkpoint_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    """Return the actual model state dict from either raw or wrapped checkpoints."""
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt


def load_yaml_config(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_data_grid_config(data_config: Dict[str, Any]):
    """从 data 配置解析原始体素范围、模型裁剪范围和训练张量尺寸。"""
    source_pc_range = data_config.get("source_pc_range", data_config.get("pc_range", [0, -20, -6, 120, 20, 10]))
    model_pc_range = data_config.get("model_pc_range", data_config.get("pc_range", source_pc_range))
    target_size = data_config.get("target_size", data_config.get("voxel_shape", [32, 128, 128]))
    return (
        tuple(int(v) for v in target_size),
        tuple(float(v) for v in source_pc_range),
        tuple(float(v) for v in model_pc_range),
    )


def assert_formal_cd_data_config(data_config: Dict[str, Any]) -> None:
    """让独立 CD 入口与统一训练入口使用相同的正式数据门禁。"""
    scene_names = data_config.get("scene_names")
    if (
        not isinstance(scene_names, list)
        or not scene_names
        or any(not isinstance(scene, str) or not scene.strip() for scene in scene_names)
    ):
        raise ValueError("正式 CD 必须显式配置非空 data.scene_names")
    if data_config.get("require_real_ir") is not True:
        raise ValueError("正式 CD 必须设置 data.require_real_ir=true")
    if data_config.get("require_real_calibration") is not True:
        raise ValueError("正式 CD 必须设置 data.require_real_calibration=true")
    if data_config.get("voxel_coordinate_frame") != "lidar":
        raise ValueError("当前正式 CD 只接受 voxel_coordinate_frame=lidar")
    if data_config.get("require_persisted_observed_mask") is not True:
        raise ValueError(
            "正式 CD 必须设置 data.require_persisted_observed_mask=true"
        )
    if data_config.get("require_radar_statistics") is not True:
        raise ValueError(
            "正式 CD 必须设置 data.require_radar_statistics=true"
        )


def is_formal_cd_training(
    checkpoint_protocol: str,
    allow_legacy_radar_units: bool,
) -> bool:
    """统一识别全量与 mini-v2；legacy 显式开关始终关闭正式门禁。"""
    return (
        checkpoint_protocol
        in {FORMAL_CHECKPOINT_PROTOCOL, FORMAL_MINI_CHECKPOINT_PROTOCOL}
        and not allow_legacy_radar_units
    )


def prepare_cd_data_protocol(
    data_protocol: Dict[str, Any],
    data_config: Dict[str, Any],
    *,
    checkpoint_protocol: str,
) -> Dict[str, Any]:
    """为独立 CD 入口绑定 mini 子集身份，并拒绝全量链隐式截断。"""
    resolved = dict(data_protocol)
    train_limit = data_config.get("mini_train_frames_per_scene")
    validation_limit = data_config.get("mini_validation_frames_per_scene")
    if checkpoint_protocol == FORMAL_MINI_CHECKPOINT_PROTOCOL:
        selection = build_formal_mini_selection(train_limit, validation_limit)
        configured = resolved.get("mini_selection")
        if configured is not None and configured != selection:
            raise ValueError("CD data protocol 的 mini_selection 与当前配置不一致")
        resolved["mini_selection"] = selection
    elif train_limit not in (None, "") or validation_limit not in (None, ""):
        raise ValueError("正式全量 CD 链禁止隐式截断 train/validation 帧")
    return validate_checkpoint_data_protocol(resolved, stage="cd")


def resolve_cd_radar_normalization(
    ldm_checkpoint,
    configured_spec,
    configured_sha256,
    *,
    data_grid_config,
    allow_legacy_radar_units=False,
):
    """在 CD 输出目录创建前校验配置 artifact 与初始化 LDM 完全一致。"""
    if type(allow_legacy_radar_units) is not bool:
        raise RadarNormalizationError("allow_legacy_radar_units 必须是 bool")
    has_embedded = isinstance(ldm_checkpoint, dict) and (
        "radar_normalization" in ldm_checkpoint
        or "radar_normalization_sha256" in ldm_checkpoint
    )
    if allow_legacy_radar_units:
        if configured_spec is not None or configured_sha256 or has_embedded:
            raise RadarNormalizationError(
                "CD legacy 开关与正式 Radar normalization 不能同时启用"
            )
        return None, ""
    if configured_spec is None or not configured_sha256:
        raise RadarNormalizationError("正式 CD 缺少配置 Radar normalization")
    initialization_spec, initialization_sha256 = radar_normalization_from_checkpoint(
        ldm_checkpoint,
        target_size=data_grid_config.get("target_size"),
        source_pc_range=data_grid_config.get("source_pc_range"),
        model_pc_range=data_grid_config.get("model_pc_range"),
        context="CD initialization LDM checkpoint",
    )
    assert_same_radar_normalization(
        configured_spec,
        configured_sha256,
        initialization_spec,
        initialization_sha256,
        context="CD initialization/config",
    )
    return initialization_spec, initialization_sha256


def create_vae_from_config(config: Optional[Dict[str, Any]] = None) -> VAE3D:
    cfg = config or {}
    vae_cfg = cfg.get("vae", {}) if isinstance(cfg.get("vae", {}), dict) else {}
    config_type = vae_cfg.get("config_type", "ultra_lightweight")
    if config_type == "lightweight":
        model_cfg = create_lightweight_vae_config()
    elif config_type == "standard":
        model_cfg = create_standard_vae_config()
    else:
        model_cfg = create_ultra_lightweight_vae_config()
    return VAE3D(**model_cfg)


def build_cd_vae_from_checkpoint(
    checkpoint: Any,
    fallback_config_type: Optional[str] = None,
):
    """按共享 checkpoint 协议构建 CD 训练使用的 VAE。"""
    return build_vae_from_checkpoint(
        checkpoint,
        fallback_config_type=fallback_config_type,
    )


def has_multimodal_state_dict(state_dict: Dict[str, torch.Tensor]) -> bool:
    """Detect checkpoints saved from CompleteDualModalityPerceptionNet."""
    keys = tuple(state_dict.keys())
    prefixes = ("unet_3d.", "ir_extractor.", "projection_layer.", "fusion_conv.")
    return any(key.startswith(prefixes) for key in keys)


def create_legacy_unet(config: Optional[Dict[str, Any]] = None) -> OptimizedUNetModel:
    """Build the legacy latent denoiser used by historical CD/LDM checkpoints."""
    cfg = config or {}
    latent_dim = int(cfg.get("latent_dim", 4))
    return OptimizedUNetModel(
        image_size=32,
        in_channels=int(cfg.get("in_channels", cfg.get("legacy_in_channels", 2 * latent_dim))),
        model_channels=int(cfg.get("model_channels", 32)),
        out_channels=latent_dim,
        num_res_blocks=int(cfg.get("num_res_blocks", 1)),
        attention_resolutions=tuple(cfg.get("attention_resolutions", [])),
        channel_mult=tuple(cfg.get("channel_mult", [1, 2, 3])),
        use_checkpoint=bool(cfg.get("use_checkpoint", True)),
        attention_type="linear",
    )


def create_multimodal_cd_model(config: Optional[Dict[str, Any]] = None) -> CompleteDualModalityPerceptionNet:
    """Build the multimodal CD/LDM denoiser with a 16-channel latent backbone."""
    cfg = config or {}
    latent_dim = int(cfg.get("latent_dim", 4))
    backbone_in_channels = int(cfg.get("in_channels", max(16, 2 * latent_dim)))
    base_unet = OptimizedUNetModel(
        image_size=32,
        in_channels=backbone_in_channels,
        model_channels=int(cfg.get("model_channels", 32)),
        out_channels=latent_dim,
        num_res_blocks=int(cfg.get("num_res_blocks", 1)),
        attention_resolutions=tuple(cfg.get("attention_resolutions", [])),
        channel_mult=tuple(cfg.get("channel_mult", [1, 2, 3])),
        use_checkpoint=bool(cfg.get("use_checkpoint", True)),
        attention_type="linear",
    )
    fusion_voxel_shape = tuple(int(v) for v in cfg.get("fusion_voxel_shape", [32, 128, 128]))
    fusion_latent_shape = tuple(int(v) for v in cfg.get("fusion_latent_shape", fusion_voxel_shape))
    fusion_pc_range = tuple(float(v) for v in cfg.get("fusion_pc_range", [0, -20, -6, 120, 20, 10]))
    return CompleteDualModalityPerceptionNet(
        base_unet,
        voxel_shape=fusion_voxel_shape,
        pc_range=fusion_pc_range,
        downsample_to_latent=True,
        latent_shape=fusion_latent_shape,
        fused_channels=backbone_in_channels,
    )


def create_cd_model(multimodal: bool, config: Optional[Dict[str, Any]] = None) -> nn.Module:
    return create_multimodal_cd_model(config) if multimodal else create_legacy_unet(config)


def resolve_cd_generation_config(checkpoint, fallback_latent_dim: Optional[int]):
    """读取教师/CD checkpoint 模型配置，旧权重从卷积 shape 推导。"""
    checkpoint_dict = checkpoint if isinstance(checkpoint, dict) else {}
    state_dict = checkpoint_state_dict(checkpoint)
    config = dict(checkpoint_dict.get("model_config") or {})
    latent_dim = checkpoint_dict.get("latent_dim", config.get("latent_dim"))
    for prefix in ("", "unet_3d."):
        input_weight = state_dict.get(f"{prefix}input_blocks.0.0.weight")
        output_weight = state_dict.get(f"{prefix}out.2.weight")
        if "in_channels" not in config and torch.is_tensor(input_weight) and input_weight.ndim >= 2:
            config["in_channels"] = int(input_weight.shape[1])
        if latent_dim is None and torch.is_tensor(output_weight) and output_weight.ndim >= 1:
            latent_dim = int(output_weight.shape[0])
    if latent_dim is None:
        if fallback_latent_dim is None:
            raise ValueError("生成模型 checkpoint 无法推导 latent_dim，请显式提供 fallback")
        latent_dim = fallback_latent_dim
    config["latent_dim"] = int(latent_dim)
    config.setdefault("in_channels", 2 * int(latent_dim))
    config.setdefault("out_channels", int(latent_dim))
    return config


def has_multimodal_meta(meta: Optional[Dict[str, Any]]) -> bool:
    required = ("ir_img", "r_mat", "t_vec", "k_mat")
    return all(torch.is_tensor((meta or {}).get(key)) for key in required)


def encode_cd_training_latents(vae, target, condition, meta_dict):
    """正式多模态只编码 target；legacy batch 才构造 condition latent。"""
    z_target = vae.get_latent(target)
    z_cond = None
    if not has_multimodal_meta(meta_dict):
        z_cond = vae.get_latent(condition)
    return z_target, z_cond


def move_meta_to_device(meta: Optional[Dict[str, Any]], device: torch.device) -> Dict[str, Any]:
    moved = {}
    for key, value in (meta or {}).items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def unpack_cd_batch(batch):
    """Support both legacy (target, radar) and metadata-rich dataset batches."""
    if len(batch) == 2:
        target, radar = batch
        return target, radar, {}
    if len(batch) == 3:
        target, radar, meta = batch
        return target, radar, meta
    if len(batch) == 4:
        target, radar, meta, _path = batch
        return target, radar, meta
    raise ValueError(f"Unsupported batch format with {len(batch)} elements")


def pad_latent_input_to_sixteen_channels(model_input: torch.Tensor) -> torch.Tensor:
    if model_input.shape[1] == 16:
        return model_input
    if model_input.shape[1] > 16:
        raise ValueError(f"Expected <=16 latent input channels, got {model_input.shape[1]}")
    pad = torch.zeros(
        model_input.shape[0],
        16 - model_input.shape[1],
        *model_input.shape[2:],
        device=model_input.device,
        dtype=model_input.dtype,
    )
    return torch.cat([model_input, pad], dim=1)


def call_cd_denoiser(
    model: nn.Module,
    x_t: torch.Tensor,
    z_cond: Optional[torch.Tensor],
    timesteps: torch.Tensor,
    radar_voxel: Optional[torch.Tensor] = None,
    meta_dict: Optional[Dict[str, Any]] = None,
    consistency_config: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """Call legacy or multimodal denoisers through one CD training interface."""
    base_model = unwrap_model(model)
    if getattr(base_model, "is_multimodal", False):
        if has_multimodal_meta(meta_dict) and radar_voxel is not None:
            raw_output = model(
                radar_voxel,
                meta_dict["ir_img"],
                meta_dict["r_mat"],
                meta_dict["t_vec"],
                meta_dict["k_mat"],
                timesteps,
                noised_latent=x_t,
            )
        else:
            if z_cond is None:
                raise ValueError("缺少 legacy multimodal CD condition latent")
            model_input = pad_latent_input_to_sixteen_channels(
                torch.cat([x_t, z_cond], dim=1)
            )
            if model is not base_model:
                raise RuntimeError(
                    "DDP 多模态 CD 不支持缺少 IR/标定的 legacy 旁路 batch"
                )
            raw_output = base_model.unet_3d(model_input, timesteps)
    else:
        if z_cond is None:
            raise ValueError("缺少 legacy CD condition latent")
        raw_output = model(torch.cat([x_t, z_cond], dim=1), timesteps)
    if consistency_config is None:
        return raw_output
    return apply_cd_boundary_parameterization(
        raw_output,
        x_t,
        timesteps,
        consistency_config,
    )


def _trainer_distributed_context(trainer) -> DistributedContext:
    """兼容单元测试直接构造的单进程 CD trainer。"""
    context = getattr(trainer, "distributed", None)
    if context is not None:
        return context
    return DistributedContext.single_process(getattr(trainer, "device", "cpu"))


class ConsistencyDistillationTrainer:
    """
    LDM 初始化的 EMA consistency 训练器（类名仅保留历史 API 兼容）
    
    流程：
    1. 加载预训练 LDM 作为一次性初始化来源
    2. 从相同权重创建 CD 模型和 EMA 目标模型
    3. 训练 CD 输出逼近持续更新的 CD EMA 目标
    """
    
    def __init__(
        self,
        ldm_ckpt_path: str,
        vae: nn.Module,
        device: str = "cuda",
        config: dict = None,
    ):
        self.config = config or {}
        self.distributed = self.config.get("distributed_context") or (
            DistributedContext.single_process(device)
        )
        self.device = torch.device(self.distributed.device)
        self.data_grid_config = dict(
            self.config.get("data_grid_config", {}) or {}
        )
        self.allow_legacy_radar_units = self.config.get(
            "allow_legacy_radar_units", False
        )
        self.require_persisted_observed_mask = bool(
            self.config.get("require_persisted_observed_mask", False)
        )
        self.validation_config = resolve_cd_validation_config(self.config)
        self.denoising_diagnostic_config = (
            resolve_cd_denoising_diagnostic_config(self.config)
        )
        self.deployment_validation_selection = self.config.get(
            "deployment_validation_selection"
        )
        if self.deployment_validation_selection is not None:
            self.validation_config["selection"] = dict(
                self.deployment_validation_selection
            )
        self.consistency_config = resolve_cd_consistency_config(self.config)
        self.checkpoint_protocol = resolve_training_checkpoint_protocol(
            self.config.get("checkpoint_protocol", FORMAL_CHECKPOINT_PROTOCOL)
        )
        configured_data_protocol = self.config.get("data_protocol")
        if configured_data_protocol is None and self.allow_legacy_radar_units:
            self.data_protocol = {"protocol": "legacy_data_v0"}
        else:
            self.data_protocol = validate_checkpoint_data_protocol(
                configured_data_protocol,
                stage="cd",
            )
        configured_selection = self.config.get("stage_training_selection")
        self.stage_training_selection = (
            validate_formal_stage_training_selection(
                configured_selection,
                expected_stage="cd",
            )
            if configured_selection is not None
            else None
        )
        initialization_checkpoint = safe_torch_load(
            ldm_ckpt_path,
            map_location="cpu",
        )
        if not self.allow_legacy_radar_units:
            assert_checkpoint_training_identity(
                initialization_checkpoint,
                expected_stage="ldm",
                checkpoint_protocol=self.checkpoint_protocol,
                data_protocol=self.data_protocol,
            )
            validate_deployment_validation_receipt(
                initialization_checkpoint.get("ldm_validation"),
                stage="ldm",
            )
        self.radar_normalization, self.radar_normalization_sha256 = (
            resolve_cd_radar_normalization(
                initialization_checkpoint,
                self.config.get("radar_normalization"),
                self.config.get("radar_normalization_sha256", ""),
                data_grid_config=self.data_grid_config,
                allow_legacy_radar_units=self.allow_legacy_radar_units,
            )
        )
        self._preloaded_ldm_checkpoint = initialization_checkpoint
        resume_path = self.config.get('resume_path')
        if resume_path and os.path.exists(resume_path):
            resume_checkpoint = safe_torch_load(resume_path, map_location="cpu")
            if not self.allow_legacy_radar_units:
                assert_checkpoint_training_identity(
                    resume_checkpoint,
                    expected_stage="cd",
                    checkpoint_protocol=self.checkpoint_protocol,
                    data_protocol=self.data_protocol,
                    stage_training_selection=self.stage_training_selection,
                )
            assert_checkpoint_radar_normalization(
                resume_checkpoint,
                self.radar_normalization,
                self.radar_normalization_sha256,
                target_size=self.data_grid_config.get("target_size"),
                source_pc_range=self.data_grid_config.get("source_pc_range"),
                model_pc_range=self.data_grid_config.get("model_pc_range"),
                allow_legacy_radar_units=self.allow_legacy_radar_units,
                context="CD resume preflight",
            )
            self._preloaded_resume_checkpoint = resume_checkpoint
        
        # 设置保存目录和日志
        self.save_dir = self.config.get('save_dir', './Result/train_results/cd')
        if self.distributed.is_main_process:
            os.makedirs(self.save_dir, exist_ok=True)
        distributed_barrier(self.distributed)
        
        # 初始化训练状态
        self.start_epoch = 1
        self.best_loss = float('inf')
        self.best_val_iou = float('-inf')
        self.best_val_loss = float('inf')
        self.last_validation_metrics = None
        self.last_validation_threshold_sweeps = None
        self.last_denoising_diagnostic_metrics = None
        self.last_collapse_diagnostics = None
        self.last_train_step_loss_components = None
        self.deployment_weight_source = None
        self.is_resumed = False
        self.model_config = dict(self.config.get("ldm", {}) or self.config.get("model", {}) or {})
        self.model_config.setdefault("latent_dim", int(vae.latent_dim))
        self.vae_checkpoint_sha256 = str(self.config.get("vae_checkpoint_sha256", "") or "")
        self.ldm_checkpoint_sha256 = str(self.config.get("ldm_checkpoint_sha256", "") or "")
        self.model_config.setdefault(
            "fusion_voxel_shape",
            list(self.data_grid_config.get("target_size", [32, 128, 128])),
        )
        self.model_config.setdefault(
            "fusion_latent_shape",
            list(self.model_config["fusion_voxel_shape"]),
        )
        self.model_config.setdefault(
            "fusion_pc_range",
            list(self.data_grid_config.get("model_pc_range", [0, -20, -6, 120, 20, 10])),
        )
        self.data_grid_config.setdefault(
            "target_size", list(self.model_config["fusion_voxel_shape"])
        )
        self.data_grid_config.setdefault(
            "source_pc_range", list(self.data_grid_config.get("model_pc_range", [0, -20, -6, 120, 20, 10]))
        )
        self.data_grid_config.setdefault(
            "model_pc_range", list(self.model_config["fusion_pc_range"])
        )
        
        # LDM 仅提供初始权重；后续 consistency target 来自 CD EMA。
        self.ldm_model = self._load_ldm_model(ldm_ckpt_path)
        self.use_multimodal = bool(getattr(self.ldm_model, "is_multimodal", False))
        self.vae = vae.to(self.device)
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
        
        # 创建 CD 学生模型（与 LDM 同结构）
        self.cd_model = create_cd_model(
            self.use_multimodal, self.model_config
        ).to(self.device)
        self.cd_model = prepare_model_for_distributed(
            self.cd_model,
            self.distributed,
        )
        
        # 创建 EMA 目标模型
        self.cd_model_ema = self._create_ema_model(self.cd_model)
        
        # 初始化 CD 模型为 LDM 的拷贝
        self._initialize_from_ldm()
        # LDM 仅用于一次初始化，释放其 GPU 副本以降低 CD 训练常驻显存。
        del self.ldm_model
        self.cd_model = wrap_model_for_ddp(
            self.cd_model,
            self.distributed,
            find_unused_parameters=False,
        )
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.cd_model.parameters(),
            lr=self.config.get('lr', 5e-5),
            weight_decay=1e-4,
        )
        
        # Denoiser
        self.denoiser = KarrasDenoiser(
            # 当前 CD 直接输出 x0；sigma_data 不参与该 train_step 的预条件公式。
            sigma_data=0.5,
            sigma_max=self.consistency_config["sigma_max"],
            sigma_min=self.consistency_config["sigma_min"],
            rho=self.consistency_config["rho"],
            loss_norm='l2',
            device=self.device,
        )
        if not (
            self.denoiser.sigma_min
            <= self.denoising_diagnostic_config["sigma"]
            <= self.denoiser.sigma_max
        ):
            raise ValueError(
                "cd.validation_sigma 必须位于训练 sigma 范围内"
            )
        
        # 禁用混合精度（避免 FP16/FP32 类型不匹配）
        self.use_amp = False
        if self.use_amp:
            self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # 设置日志
        self.log_file = os.path.join(self.save_dir, 'training.log')
        self.csv_file = os.path.join(self.save_dir, 'metrics.csv')
        
        # 检查是否恢复训练
        if resume_path and os.path.exists(resume_path):
            self.is_resumed = True
        
        self._setup_logging()
        
        # 恢复训练
        if self.is_resumed:
            self._resume_from_checkpoint(resume_path)
    
    def _load_ldm_model(self, ckpt_path: str) -> nn.Module:
        """加载 LDM 教师模型"""
        # 静默创建模型（避免重复打印）
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        ckpt = getattr(self, "_preloaded_ldm_checkpoint", None)
        if ckpt is None:
            ckpt = safe_torch_load(ckpt_path, map_location='cpu')
        state_dict = checkpoint_state_dict(ckpt)
        resolved_config = resolve_cd_generation_config(
            ckpt,
            fallback_latent_dim=self.model_config.get("latent_dim"),
        )
        resolved_config.update({
            key: value for key, value in self.model_config.items()
            if key not in resolved_config
        })
        self.model_config = resolved_config
        model = create_cd_model(has_multimodal_state_dict(state_dict), self.model_config).to(self.device)
        
        sys.stdout = old_stdout
        
        model.load_state_dict(state_dict)
        
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        
        if self.distributed.is_main_process:
            print(f"Loaded LDM initialization checkpoint from {ckpt_path}")
        return model
    
    def _create_ema_model(self, model: nn.Module) -> nn.Module:
        """创建 EMA 目标模型"""
        # 静默创建模型
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        ema_model = create_cd_model(
            bool(getattr(model, "is_multimodal", False)),
            self.model_config,
        ).to(self.device)
        ema_model = prepare_model_for_distributed(
            ema_model,
            self.distributed,
        )
        
        sys.stdout = old_stdout
        
        ema_model.load_state_dict(model.state_dict())
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad = False
        
        return ema_model
    
    def _setup_logging(self):
        """设置日志系统"""
        if not self.distributed.is_main_process:
            self.logger = logging.getLogger(
                f"{__name__}.cd.rank{self.distributed.rank}"
            )
            self.logger.handlers.clear()
            self.logger.addHandler(logging.NullHandler())
            self.logger.propagate = False
            return
        # 确定日志文件模式
        log_mode = 'a' if self.is_resumed else 'w'
        
        # 配置文本日志（只写入文件，不输出到终端）
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file, mode=log_mode)
            ],
            force=True
        )
        self.logger = logging.getLogger(__name__ + '_cd')
        
        # 添加训练会话分隔符
        if self.is_resumed:
            self.logger.info("\n" + "="*70)
            self.logger.info("RESUMING TRAINING SESSION")
            self.logger.info("="*70)
        
        # 初始化 CSV 文件
        if not os.path.exists(self.csv_file) or not self.is_resumed:
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'loss', 'lr', 'time_seconds'])
    
    def _resume_from_checkpoint(self, ckpt_path: str):
        """从检查点恢复训练"""
        if self.distributed.is_main_process:
            print(f"Resuming CD from checkpoint: {ckpt_path}")
        ckpt = getattr(self, "_preloaded_resume_checkpoint", None)
        if ckpt is None:
            ckpt = safe_torch_load(ckpt_path, map_location=self.device)
        assert_checkpoint_radar_normalization(
            ckpt,
            self.radar_normalization,
            self.radar_normalization_sha256,
            target_size=self.data_grid_config.get("target_size"),
            source_pc_range=self.data_grid_config.get("source_pc_range"),
            model_pc_range=self.data_grid_config.get("model_pc_range"),
            allow_legacy_radar_units=self.allow_legacy_radar_units,
            context="CD resume checkpoint",
        )
        assert_cd_ema_update_protocol(
            ckpt,
            require_formal=(
                self.use_multimodal and self.radar_normalization is not None
            ),
        )
        saved_consistency_config = ckpt.get("consistency_training_config")
        if saved_consistency_config is None:
            if self.use_multimodal and self.radar_normalization is not None:
                raise ValueError("formal CD resume 缺少 consistency training config")
        else:
            saved_consistency_config = validate_cd_consistency_receipt(
                saved_consistency_config
            )
            if saved_consistency_config != self.consistency_config:
                raise ValueError("CD resume consistency training config 不一致")
        restored_validation = assert_cd_validation_checkpoint_protocol(
            ckpt,
            current_config=self.validation_config,
            require_formal=(
                self.use_multimodal and self.radar_normalization is not None
            ),
        )
        saved_collapse_diagnostics = ckpt.get("cd_collapse_diagnostics")
        if self.use_multimodal and self.radar_normalization is not None:
            collapse_receipt = validate_cd_collapse_diagnostics_receipt(
                saved_collapse_diagnostics
            )
            if collapse_receipt["selected_source"] != restored_validation[
                "selected_source"
            ]:
                raise ValueError("CD resume 防塌缩诊断与部署权重不一致")
        elif saved_collapse_diagnostics is not None:
            validate_cd_collapse_diagnostics_receipt(
                saved_collapse_diagnostics
            )
        expected_effective_batch = self.config.get(
            "expected_effective_global_batch_size"
        )
        if expected_effective_batch is not None:
            assert_resume_distributed_compatible(
                ckpt,
                expected_effective_global_batch_size=int(
                    expected_effective_batch
                ),
            )
        
        # 加载模型
        unwrap_model(self.cd_model).load_state_dict(ckpt['model_state_dict'])
        if 'ema_model_state_dict' in ckpt:
            self.cd_model_ema.load_state_dict(ckpt['ema_model_state_dict'])
        
        # 加载优化器
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        
        # 加载训练状态
        self.start_epoch = ckpt.get('epoch', 0) + 1
        self.best_loss = ckpt.get('best_loss', ckpt.get('loss', float('inf')))
        if restored_validation is not None:
            self.last_validation_metrics = restored_validation["metrics"]
            self.deployment_weight_source = restored_validation[
                "selected_source"
            ]
            best_loss, best_iou = _validated_cd_validation_metrics(
                restored_validation["best_selected_metrics"],
                source="best selected",
            )
            self.best_val_loss = best_loss
            self.best_val_iou = best_iou
        
        if self.distributed.is_main_process:
            print(f"Resumed from epoch {self.start_epoch - 1}, best loss: {self.best_loss:.4f}")
    
    def _log_metrics(self, epoch: int, loss: float, epoch_time: float):
        """记录指标到 CSV 文件"""
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f'{loss:.6f}',
                f'{self.optimizer.param_groups[0]["lr"]:.8f}',
                f'{epoch_time:.2f}'
            ])
    
    def _initialize_from_ldm(self):
        """用 LDM 权重初始化 CD 模型"""
        if not self.is_resumed:
            self.cd_model.load_state_dict(self.ldm_model.state_dict())
            self.cd_model_ema.load_state_dict(self.ldm_model.state_dict())
            if self.distributed.is_main_process:
                print("Initialized CD model and EMA target from LDM checkpoint")
    
    def _update_ema(self, ema_rate: float = 0.999):
        """按具名状态更新 EMA 参数、浮点 buffer 和整数计数器。"""
        if (
            isinstance(ema_rate, bool)
            or not isinstance(ema_rate, (int, float))
            or not 0.0 <= float(ema_rate) <= 1.0
        ):
            raise ValueError(f"ema_rate 必须位于 [0,1]，实际为 {ema_rate!r}")
        source_model = unwrap_model(self.cd_model)
        source_parameters = dict(source_model.named_parameters())
        ema_parameters = dict(self.cd_model_ema.named_parameters())
        source_buffers = dict(source_model.named_buffers())
        ema_buffers = dict(self.cd_model_ema.named_buffers())
        if source_parameters.keys() != ema_parameters.keys():
            raise RuntimeError("CD online/EMA parameter 名称不一致")
        if source_buffers.keys() != ema_buffers.keys():
            raise RuntimeError("CD online/EMA buffer 名称不一致")

        with torch.no_grad():
            for name, source in source_parameters.items():
                target = ema_parameters[name]
                if source.shape != target.shape or source.dtype != target.dtype:
                    raise RuntimeError(f"CD online/EMA parameter {name!r} 结构不一致")
                target.mul_(ema_rate).add_(source.detach(), alpha=1 - ema_rate)
            for name, source in source_buffers.items():
                target = ema_buffers[name]
                if source.shape != target.shape or source.dtype != target.dtype:
                    raise RuntimeError(f"CD online/EMA buffer {name!r} 结构不一致")
                if torch.is_floating_point(target) or torch.is_complex(target):
                    target.mul_(ema_rate).add_(
                        source.detach(),
                        alpha=1 - ema_rate,
                    )
                else:
                    target.copy_(source.detach())
    
    @torch.no_grad()
    def _euler_solver(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        next_t: torch.Tensor,
        cond: torch.Tensor,
        radar_voxel: Optional[torch.Tensor] = None,
        meta_dict: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Euler ODE 求解器
        
        用于从当前时间步推进到下一时间步
        """
        # 模型输出
        denoised = call_cd_denoiser(
            model,
            x_t,
            cond,
            t,
            radar_voxel=radar_voxel,
            meta_dict=meta_dict,
            consistency_config=self.consistency_config,
        )
        
        # NOTE: 这里使用显式 Euler 推进 CD EMA 目标，不存在冻结 LDM 教师。
        # NOTE: 常微分方程（ODE）形式：dx/dt = (x - denoised) / t。
        d = (x_t - denoised) / t.view(-1, 1, 1, 1, 1)
        
        # Euler 步进
        dt = next_t - t
        x_next = x_t + d * dt.view(-1, 1, 1, 1, 1)
        
        return x_next
    
    def train_step(
        self,
        z_target: torch.Tensor,
        z_cond: Optional[torch.Tensor],
        num_scales: Optional[int] = None,
        radar_voxel: Optional[torch.Tensor] = None,
        meta_dict: Optional[Dict[str, Any]] = None,
        latent_observed_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        单个 LDM-initialized EMA consistency 训练步骤
        
        核心思想：
        1. 学生模型：从 x(t_n) 直接一步预测去噪后的结果
        2. CD EMA 目标：从 x(t_n) 用 Euler 推进到 t_{n+1} 后预测
        3. 让 online CD 一步预测接近持续更新的 EMA 目标
        4. 在 persisted observed latent 域锚定真实 target
        """
        if num_scales is None:
            num_scales = self.consistency_config["num_scales"]
        batch_size = z_target.shape[0]
        device = z_target.device
        
        # NOTE: 随机采样相邻时间步对 (t_n, t_{n+1})，覆盖不同噪声区间。
        indices = torch.randint(0, num_scales - 1, (batch_size,), device=device)
        
        # 训练与部署共用同一调度器，确保理论端点不会因浮点舍入越界。
        sigma_schedule = karras_sigma_schedule(
            steps=num_scales - 1,
            sigma_min=self.denoiser.sigma_min,
            sigma_max=self.denoiser.sigma_max,
            rho=self.denoiser.rho,
            device=device,
            dtype=z_target.dtype,
        )
        t_n = sigma_schedule[indices]
        t_next = sigma_schedule[indices + 1]
        
        # 生成带噪数据 x(t_n)
        noise = torch.randn_like(z_target)
        x_t_n = z_target + noise * t_n.view(-1, 1, 1, 1, 1)
        
        # 学生模型：从 x(t_n) 直接预测去噪结果
        student_denoised = call_cd_denoiser(
            self.cd_model,
            x_t_n,
            z_cond,
            t_n,
            radar_voxel=radar_voxel,
            meta_dict=meta_dict,
            consistency_config=self.consistency_config,
        )
        
        # CD EMA 目标模型：从 x(t_n) 推进到 x(t_{n+1})，再预测。
        with torch.no_grad():
            # 该模型由 CD 参数持续 EMA 更新，不是冻结的 LDM 教师。
            x_t_next = self._euler_solver(
                self.cd_model_ema,
                x_t_n,
                t_n,
                t_next,
                z_cond,
                radar_voxel=radar_voxel,
                meta_dict=meta_dict,
            )
            
            # EMA 目标在 t_{n+1} 的预测
            target_denoised = call_cd_denoiser(
                self.cd_model_ema,
                x_t_next,
                z_cond,
                t_next,
                radar_voxel=radar_voxel,
                meta_dict=meta_dict,
                consistency_config=self.consistency_config,
            )
        
        # NOTE: 重建锚点只消费 persisted observed latent，常数输出不再是零损失解。
        loss, components = compute_cd_training_losses(
            student_denoised,
            target_denoised,
            z_target,
            latent_observed_mask,
            self.consistency_config,
        )
        self.last_train_step_loss_components = components
        
        return loss
    
    def train_epoch(
        self,
        epoch: int,
        train_loader: DataLoader,
        num_scales: Optional[int] = None,
        grad_accum_steps: int = 8,
    ) -> float:
        """训练一个 epoch"""
        distributed = _trainer_distributed_context(self)
        self.cd_model.train()
        total_loss = 0
        
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}",
            disable=not distributed.is_main_process,
        )
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            target, cond, meta_dict = unpack_cd_batch(batch)
            target = target.to(self.device)
            cond = cond.to(self.device)
            meta_dict = move_meta_to_device(meta_dict, self.device)
            
            # 编码到潜空间
            with torch.no_grad():
                z_target, z_cond = encode_cd_training_latents(
                    self.vae,
                    target,
                    cond,
                    meta_dict,
                )
                raw_observed_mask = meta_dict.get("occupancy_observed_mask")
                if (
                    raw_observed_mask is None
                    and self.require_persisted_observed_mask
                ):
                    raise RuntimeError(
                        "formal CD training batch 缺少 persisted "
                        "occupancy_observed_mask"
                    )
                _, latent_observed = resolve_cd_observed_masks(
                    raw_observed_mask,
                    target,
                    z_target,
                )
            
            # 计算损失
            loss = self.train_step(
                z_target,
                z_cond,
                num_scales=num_scales,
                radar_voxel=cond,
                meta_dict=meta_dict,
                latent_observed_mask=latent_observed,
            )
            loss = loss / grad_accum_steps
            
            # 反向传播
            loss.backward()
            
            # NOTE: 梯度累积用于控制显存；每 grad_accum_steps 次小步做一次参数更新。
            if (batch_idx + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.cd_model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # 更新 EMA 模型
                self._update_ema(ema_rate=self.consistency_config["ema_rate"])
            
            total_loss += loss.item() * grad_accum_steps
            pbar.set_postfix({'loss': f'{loss.item() * grad_accum_steps:.6f}'})

        if len(train_loader) % grad_accum_steps != 0:
            accumulation_count = len(train_loader) % grad_accum_steps
            tail_scale = float(grad_accum_steps) / float(accumulation_count)
            for parameter in self.cd_model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(tail_scale)
            torch.nn.utils.clip_grad_norm_(self.cd_model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self._update_ema(ema_rate=self.consistency_config["ema_rate"])
        
        totals = reduce_named_sums(
            {"loss": total_loss, "batch_count": len(train_loader)},
            distributed,
        )
        if totals["batch_count"] == 0:
            raise RuntimeError("CD 全局训练 DataLoader 为空")
        return totals["loss"] / totals["batch_count"]

    @torch.no_grad()
    def _validate_denoising_diagnostic_model(
        self,
        model: nn.Module,
        val_loader: DataLoader,
    ) -> Dict[str, float]:
        """计算 target-latent 小噪声诊断；结果不得参与部署选优。"""
        distributed = _trainer_distributed_context(self)
        was_training = model.training
        model.eval()
        latent_squared_error = 0.0
        latent_element_count = 0
        intersection = 0
        union = 0
        try:
            for batch in val_loader:
                target, radar, meta_dict = unpack_cd_batch(batch)
                target = target.to(self.device, non_blocking=True)
                radar = radar.to(self.device, non_blocking=True)
                meta_dict = move_meta_to_device(meta_dict, self.device)
                raw_observed_mask = meta_dict.get("occupancy_observed_mask")
                if (
                    raw_observed_mask is None
                    and self.require_persisted_observed_mask
                ):
                    raise RuntimeError(
                        "formal CD validation batch 缺少 persisted "
                        "occupancy_observed_mask"
                    )
                z_target, z_cond = encode_cd_training_latents(
                    self.vae,
                    target,
                    radar,
                    meta_dict,
                )
                voxel_observed, latent_observed = resolve_cd_observed_masks(
                    raw_observed_mask,
                    target,
                    z_target,
                )
                sigma = torch.full(
                    (z_target.shape[0],),
                    self.denoising_diagnostic_config["sigma"],
                    device=z_target.device,
                    dtype=z_target.dtype,
                )
                sample_ids = meta_dict.get("sample_id")
                if sample_ids is None:
                    generator = torch.Generator(device=z_target.device)
                    generator.manual_seed(
                        self.denoising_diagnostic_config["seed"]
                    )
                    noise = torch.randn(
                        z_target.shape,
                        generator=generator,
                        device=z_target.device,
                        dtype=z_target.dtype,
                    )
                else:
                    noise = deterministic_noise_from_sample_ids(
                        z_target,
                        sample_ids,
                        seed=self.denoising_diagnostic_config["seed"],
                    )
                noised = z_target + noise * sigma.view(-1, 1, 1, 1, 1)
                denoised = call_cd_denoiser(
                    model,
                    noised,
                    z_cond,
                    sigma,
                    radar_voxel=radar,
                    meta_dict=meta_dict,
                    consistency_config=self.consistency_config,
                )
                latent_error = torch.square(
                    denoised.float() - z_target.float()
                )
                if latent_observed is None:
                    latent_squared_error += float(latent_error.sum().item())
                    latent_element_count += latent_error.numel()
                else:
                    expanded = latent_observed.expand_as(latent_error)
                    latent_squared_error += float(
                        latent_error[expanded].sum().item()
                    )
                    latent_element_count += int(expanded.sum().item())

                decoded = self.vae.decode(denoised)
                probability = decoded[:, 0:1]
                occupancy_activation = getattr(
                    self.vae,
                    "occupancy_activation",
                    "raw",
                )
                if occupancy_activation == "sigmoid":
                    probability = torch.sigmoid(probability)
                elif occupancy_activation != "raw":
                    raise ValueError(
                        "CD validation 不支持的 occupancy activation: "
                        f"{occupancy_activation!r}"
                    )
                prediction = (
                    probability
                    >= self.denoising_diagnostic_config[
                        "occupancy_threshold"
                    ]
                )
                truth = target[:, 0:1] >= 0.5
                if voxel_observed is not None:
                    prediction = prediction & voxel_observed
                    truth = truth & voxel_observed
                intersection += int((prediction & truth).sum().item())
                union += int((prediction | truth).sum().item())
        finally:
            model.train(was_training)

        reduced = reduce_named_sums(
            {
                "latent_squared_error": latent_squared_error,
                "latent_element_count": latent_element_count,
                "intersection": intersection,
                "union": union,
            },
            distributed,
        )
        if reduced["latent_element_count"] == 0:
            raise RuntimeError("CD validation 没有可计算的 latent 元素")
        metrics = {
            "denoising_latent_loss": (
                reduced["latent_squared_error"]
                / reduced["latent_element_count"]
            ),
            "denoising_occupancy_iou": (
                reduced["intersection"] / max(reduced["union"], 1)
            ),
        }
        return metrics

    @torch.no_grad()
    def _validate_deployment_model(
        self,
        model: nn.Module,
        val_loader: DataLoader,
    ) -> Tuple[Dict[str, float], list, Dict[str, Any]]:
        """从 sigma_max 纯噪声一步采样，并审计常数输出和条件忽略。"""
        distributed = _trainer_distributed_context(self)
        was_training = model.training
        model.eval()
        latent_squared_error = 0.0
        latent_element_count = 0
        intersection = 0
        union = 0
        output_sum = 0.0
        output_squared_sum = 0.0
        output_element_count = 0
        inter_sample_squared_error = 0.0
        inter_sample_element_count = 0
        inter_sample_pair_count = 0
        condition_squared_error = 0.0
        condition_element_count = 0
        previous_output = None
        threshold_counts = {
            key: 0
            for key in threshold_sweep_batch_counts(
                torch.zeros(1, 1, 1, 1, 1),
                torch.zeros(1, 1, 1, 1, 1),
                torch.ones(1, 1, 1, 1, 1, dtype=torch.bool),
                self.validation_config["threshold_candidates"],
            )
        }
        try:
            for batch in val_loader:
                target, radar, meta_dict = unpack_cd_batch(batch)
                target = target.to(self.device, non_blocking=True)
                radar = radar.to(self.device, non_blocking=True)
                meta_dict = move_meta_to_device(meta_dict, self.device)
                raw_observed_mask = meta_dict.get("occupancy_observed_mask")
                if (
                    raw_observed_mask is None
                    and self.require_persisted_observed_mask
                ):
                    raise RuntimeError(
                        "formal CD deployment validation 缺少 persisted "
                        "occupancy_observed_mask"
                    )
                z_target, z_cond = encode_cd_training_latents(
                    self.vae,
                    target,
                    radar,
                    meta_dict,
                )
                voxel_observed, latent_observed = resolve_cd_observed_masks(
                    raw_observed_mask,
                    target,
                    z_target,
                )
                sample_ids = meta_dict.get("sample_id")
                if sample_ids is None:
                    if self.radar_normalization is not None:
                        raise RuntimeError(
                            "formal CD deployment validation 缺少 sample_id"
                        )
                    generator = torch.Generator(device=z_target.device)
                    generator.manual_seed(self.validation_config["seed"])
                    noise = torch.randn(
                        z_target.shape,
                        generator=generator,
                        device=z_target.device,
                        dtype=z_target.dtype,
                    )
                else:
                    noise = deterministic_noise_from_sample_ids(
                        z_target,
                        sample_ids,
                        seed=self.validation_config["seed"],
                    )
                sigma = torch.full(
                    (z_target.shape[0],),
                    self.validation_config["sigma_max"],
                    device=z_target.device,
                    dtype=z_target.dtype,
                )
                initial_latent = noise * sigma.view(-1, 1, 1, 1, 1)
                generated_latent = call_cd_denoiser(
                    model,
                    initial_latent,
                    z_cond,
                    sigma,
                    radar_voxel=radar,
                    meta_dict=meta_dict,
                    consistency_config=self.consistency_config,
                )
                if has_multimodal_meta(meta_dict):
                    ablated_meta = dict(meta_dict)
                    ablated_meta["ir_img"] = torch.zeros_like(
                        meta_dict["ir_img"]
                    )
                    ablated_latent = call_cd_denoiser(
                        model,
                        initial_latent,
                        z_cond,
                        sigma,
                        radar_voxel=torch.zeros_like(radar),
                        meta_dict=ablated_meta,
                        consistency_config=self.consistency_config,
                    )
                else:
                    if z_cond is None:
                        raise RuntimeError("legacy CD 防塌缩诊断缺少 condition latent")
                    ablated_latent = call_cd_denoiser(
                        model,
                        initial_latent,
                        torch.zeros_like(z_cond),
                        sigma,
                        radar_voxel=radar,
                        meta_dict=meta_dict,
                        consistency_config=self.consistency_config,
                    )

                generated_float = generated_latent.float()
                output_sum += float(generated_float.sum().item())
                output_squared_sum += float(
                    torch.square(generated_float).sum().item()
                )
                output_element_count += generated_float.numel()
                condition_difference = torch.square(
                    generated_float - ablated_latent.float()
                )
                condition_squared_error += float(
                    condition_difference.sum().item()
                )
                condition_element_count += condition_difference.numel()
                for sample_output in generated_float:
                    if previous_output is not None:
                        sample_difference = torch.square(
                            sample_output - previous_output
                        )
                        inter_sample_squared_error += float(
                            sample_difference.sum().item()
                        )
                        inter_sample_element_count += sample_difference.numel()
                        inter_sample_pair_count += 1
                    previous_output = sample_output.detach()
                latent_error = torch.square(
                    generated_latent.float() - z_target.float()
                )
                if latent_observed is None:
                    latent_squared_error += float(latent_error.sum().item())
                    latent_element_count += latent_error.numel()
                else:
                    expanded = latent_observed.expand_as(latent_error)
                    latent_squared_error += float(
                        latent_error[expanded].sum().item()
                    )
                    latent_element_count += int(expanded.sum().item())

                decoded = self.vae.decode(generated_latent)
                probability = decoded[:, 0:1]
                occupancy_activation = getattr(
                    self.vae,
                    "occupancy_activation",
                    "raw",
                )
                if occupancy_activation == "sigmoid":
                    probability = torch.sigmoid(probability)
                elif occupancy_activation != "raw":
                    raise ValueError(
                        "CD deployment validation 不支持 occupancy activation: "
                        f"{occupancy_activation!r}"
                    )
                prediction = (
                    probability >= self.validation_config["occupancy_threshold"]
                )
                truth = target[:, 0:1] >= 0.5
                if voxel_observed is not None:
                    prediction = prediction & voxel_observed
                    truth = truth & voxel_observed
                intersection += int((prediction & truth).sum().item())
                union += int((prediction | truth).sum().item())
                batch_threshold_counts = threshold_sweep_batch_counts(
                    probability,
                    target[:, 0:1],
                    voxel_observed,
                    self.validation_config["threshold_candidates"],
                )
                for name, value in batch_threshold_counts.items():
                    threshold_counts[name] += value
        finally:
            model.train(was_training)

        reduced = reduce_named_sums(
            {
                "latent_squared_error": latent_squared_error,
                "latent_element_count": latent_element_count,
                "intersection": intersection,
                "union": union,
                "output_sum": output_sum,
                "output_squared_sum": output_squared_sum,
                "output_element_count": output_element_count,
                "inter_sample_squared_error": inter_sample_squared_error,
                "inter_sample_element_count": inter_sample_element_count,
                "inter_sample_pair_count": inter_sample_pair_count,
                "condition_squared_error": condition_squared_error,
                "condition_element_count": condition_element_count,
                **threshold_counts,
            },
            distributed,
        )
        if reduced["latent_element_count"] == 0:
            raise RuntimeError(
                "CD deployment validation 没有可计算的 latent 元素"
            )
        metrics = validate_deployment_metrics(
            {
                "deployment_latent_loss": (
                    reduced["latent_squared_error"]
                    / reduced["latent_element_count"]
                ),
                "deployment_occupancy_iou": (
                    reduced["intersection"] / max(reduced["union"], 1)
                ),
            },
            source="CD",
        )
        sweep = threshold_sweep_metrics(
            reduced,
            self.validation_config["threshold_candidates"],
        )
        if reduced["output_element_count"] == 0:
            raise RuntimeError("CD collapse diagnostic 没有输出元素")
        output_mean = reduced["output_sum"] / reduced["output_element_count"]
        output_variance = max(
            reduced["output_squared_sum"] / reduced["output_element_count"]
            - output_mean * output_mean,
            0.0,
        )
        inter_sample_mse = (
            reduced["inter_sample_squared_error"]
            / reduced["inter_sample_element_count"]
            if reduced["inter_sample_element_count"] > 0
            else None
        )
        if reduced["condition_element_count"] == 0:
            raise RuntimeError("CD collapse diagnostic 没有条件消融元素")
        collapse_diagnostic = build_cd_collapse_diagnostic(
            output_variance=output_variance,
            inter_sample_mse=inter_sample_mse,
            inter_sample_pair_count=int(reduced["inter_sample_pair_count"]),
            condition_ablation_mse=(
                reduced["condition_squared_error"]
                / reduced["condition_element_count"]
            ),
            guard_epsilon=self.consistency_config["collapse_guard_epsilon"],
        )
        return metrics, sweep, collapse_diagnostic

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        deployment_val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, Dict[str, float]]:
        """局部去噪仅诊断；固定子集一步采样选择 online/EMA。"""
        if len(val_loader) == 0 and not _trainer_distributed_context(self).initialized:
            raise RuntimeError("CD validation DataLoader 为空")
        if (
            getattr(self, "radar_normalization", None) is not None
            and deployment_val_loader is None
        ):
            raise RuntimeError(
                "formal CD 必须提供固定 deployment validation DataLoader"
            )
        if deployment_val_loader is None:
            deployment_val_loader = val_loader

        online_diagnostic = self._validate_denoising_diagnostic_model(
            self.cd_model,
            val_loader,
        )
        ema_diagnostic = self._validate_denoising_diagnostic_model(
            self.cd_model_ema,
            val_loader,
        )
        self.last_denoising_diagnostic_metrics = {
            "model_state_dict": online_diagnostic,
            "ema_model_state_dict": ema_diagnostic,
        }

        online_metrics, online_sweep, online_collapse = self._validate_deployment_model(
            self.cd_model,
            deployment_val_loader,
        )
        ema_metrics, ema_sweep, ema_collapse = self._validate_deployment_model(
            self.cd_model_ema,
            deployment_val_loader,
        )
        metrics = {
            "model_state_dict": online_metrics,
            "ema_model_state_dict": ema_metrics,
        }
        self.last_validation_threshold_sweeps = {
            "model_state_dict": online_sweep,
            "ema_model_state_dict": ema_sweep,
        }
        self.last_validation_metrics = metrics
        self.last_collapse_diagnostics = {
            "model_state_dict": online_collapse,
            "ema_model_state_dict": ema_collapse,
        }
        self.deployment_weight_source = select_cd_deployment_weight_source(
            metrics["model_state_dict"],
            metrics["ema_model_state_dict"],
        )
        return metrics

    def _checkpoint_payload(self, epoch: int, loss: float, best_loss: float) -> Dict[str, Any]:
        """构造带完整网格和父 checkpoint hash 的正式 CD checkpoint。"""
        formal_checkpoint = bool(
            self.use_multimodal and self.radar_normalization is not None
        )
        validation_metrics = getattr(self, "last_validation_metrics", None)
        deployment_weight_source = getattr(
            self,
            "deployment_weight_source",
            None,
        )
        if formal_checkpoint:
            if not isinstance(validation_metrics, dict):
                raise ValueError("formal CD checkpoint 缺少 online/EMA validation metrics")
            expected_source = select_cd_deployment_weight_source(
                validation_metrics.get("model_state_dict"),
                validation_metrics.get("ema_model_state_dict"),
            )
            if deployment_weight_source != expected_source:
                raise ValueError(
                    "formal CD deployment_weight_source 与 validation 选优结果不一致"
                )
            if not (
                math.isfinite(getattr(self, "best_val_loss", float("inf")))
                and math.isfinite(getattr(self, "best_val_iou", float("-inf")))
            ):
                raise ValueError("formal CD checkpoint 缺少历史最佳 validation 指标")
            threshold_sweeps = getattr(
                self,
                "last_validation_threshold_sweeps",
                None,
            )
            if (
                not isinstance(threshold_sweeps, dict)
                or set(threshold_sweeps) != {
                    "model_state_dict",
                    "ema_model_state_dict",
                }
            ):
                raise ValueError("formal CD checkpoint 缺少 online/EMA threshold sweep")
            if not isinstance(self.deployment_validation_selection, dict):
                raise ValueError(
                    "formal CD checkpoint 缺少 deployment validation selection"
                )
            collapse_receipt = build_cd_collapse_diagnostics_receipt(
                selected_source=deployment_weight_source,
                diagnostics=getattr(self, "last_collapse_diagnostics", None),
            )
        else:
            collapse_receipt = None
        payload = {
            "epoch": epoch,
            "loss": loss,
            "best_loss": best_loss,
            "model_state_dict": unwrap_model(self.cd_model).state_dict(),
            "ema_model_state_dict": self.cd_model_ema.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "latent_dim": int(self.model_config["latent_dim"]),
            "model_config": dict(self.model_config),
            "checkpoint_protocol": (
                getattr(
                    self,
                    "checkpoint_protocol",
                    FORMAL_CHECKPOINT_PROTOCOL,
                )
                if self.use_multimodal and self.radar_normalization is not None
                else "legacy_cd_v0"
            ),
            "stage": "cd",
            "data_grid_config": dict(self.data_grid_config),
            "vae_checkpoint_sha256": self.vae_checkpoint_sha256,
            "ldm_checkpoint_sha256": self.ldm_checkpoint_sha256,
            "model_family": "multimodal" if self.use_multimodal else "legacy",
            "training_semantics": CD_TRAINING_SEMANTICS,
            "ldm_role": "initialization_checkpoint",
            "consistency_target_source": "cd_model_ema",
            "denoising_parameterization": CD_DENOISING_PARAMETERIZATION,
            "consistency_training_config": dict(
                getattr(
                    self,
                    "consistency_config",
                    resolve_cd_consistency_config({}),
                )
            ),
            "ema_update_protocol": CD_EMA_UPDATE_PROTOCOL,
            "data_protocol": dict(self.data_protocol),
        }
        if validation_metrics is not None:
            payload["deployment_weight_source"] = deployment_weight_source
            payload["cd_validation"] = {
                **dict(getattr(self, "validation_config", {}) or {}),
                "protocol": CD_VALIDATION_PROTOCOL,
                "selector": CD_VALIDATION_SELECTOR,
                "selected_source": deployment_weight_source,
                "metrics": {
                    key: dict(value)
                    for key, value in validation_metrics.items()
                },
                "best_selected_metrics": {
                    "deployment_latent_loss": float(self.best_val_loss),
                    "deployment_occupancy_iou": float(self.best_val_iou),
                },
                "denoising_diagnostic": {
                    **dict(self.denoising_diagnostic_config),
                    "role": "diagnostic_only_not_deployment_selector",
                    "metrics": {
                        key: dict(value)
                        for key, value in (
                            self.last_denoising_diagnostic_metrics or {}
                        ).items()
                    },
                },
            }
            if collapse_receipt is not None:
                payload["cd_collapse_diagnostics"] = collapse_receipt
            if formal_checkpoint:
                payload["occupancy_threshold_validation"] = {
                    "protocol": THRESHOLD_SWEEP_PROTOCOL,
                    "split": self.validation_config["split"],
                    "observation_domain": "persisted_observed_mask_v1",
                    "sampling_protocol": CD_VALIDATION_PROTOCOL,
                    "deployment_validation_selection_sha256": (
                        self.deployment_validation_selection[
                            "selection_sha256"
                        ]
                    ),
                    "deployment_weight_source": deployment_weight_source,
                    "candidate_thresholds": list(
                        self.validation_config["threshold_candidates"]
                    ),
                    "selection_constraints": dict(
                        self.validation_config[
                            "threshold_selection_constraints"
                        ]
                    ),
                    "metrics_by_threshold": [
                        dict(record)
                        for record in threshold_sweeps[
                            deployment_weight_source
                        ]
                    ],
                }
        if getattr(self, "stage_training_selection", None) is not None:
            payload["stage_training_selection"] = dict(
                self.stage_training_selection
            )
        if self.radar_normalization is not None:
            payload["radar_normalization"] = dict(self.radar_normalization)
            payload["radar_normalization_sha256"] = self.radar_normalization_sha256
        if hasattr(self, "distributed_training"):
            payload["distributed_training"] = dict(self.distributed_training)
        if formal_checkpoint:
            validate_checkpoint_threshold_sweep(
                payload,
                expected_stage="cd",
                expected_weight_source=deployment_weight_source,
                expected_candidates=self.validation_config[
                    "threshold_candidates"
                ],
            )
        return payload
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        deployment_val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        save_every: int = 10,
        grad_accum_steps: int = 8,
    ):
        """完整训练流程"""
        distributed = _trainer_distributed_context(self)
        batch_size = int(getattr(train_loader, "batch_size", 1))
        try:
            train_dataset_size = len(train_loader.dataset)
        except (AttributeError, TypeError):
            train_dataset_size = len(train_loader) * batch_size * distributed.world_size
        batch_plan = WorldBatchPlan(
            world_size=distributed.world_size,
            per_rank_batch_size=batch_size,
            gradient_accumulation_steps=int(grad_accum_steps),
            effective_global_batch_size=(
                distributed.world_size * batch_size * int(grad_accum_steps)
            ),
        )
        self.distributed_training = distributed_checkpoint_metadata(
            distributed,
            batch_plan,
            train_dataset_size=train_dataset_size,
        )
        estimated_total_steps = num_epochs * len(train_loader)
        formal_training = bool(
            self.use_multimodal and self.radar_normalization is not None
        )
        if formal_training and val_loader is None:
            raise ValueError("formal CD 训练必须提供独立 validation DataLoader")
        if formal_training and deployment_val_loader is None:
            raise ValueError(
                "formal CD 训练必须提供固定 deployment validation DataLoader"
            )
        
        msg = "="*70 + "\n"
        msg += f"Starting CD Training\n"
        msg += f"  Total epochs: {num_epochs}\n"
        msg += f"  Batches per epoch: {len(train_loader)}\n"
        msg += f"  Estimated total steps: {estimated_total_steps:,}\n"
        msg += f"  Start epoch: {self.start_epoch}\n"
        msg += f"  Batch size: {train_loader.batch_size}\n"
        msg += f"  Gradient accumulation: {grad_accum_steps}\n"
        msg += f"  Effective batch size: {batch_plan.effective_global_batch_size}\n"
        msg += f"  Distributed world size: {distributed.world_size}\n"
        msg += f"  Learning rate: {self.optimizer.param_groups[0]['lr']:.2e}\n"
        msg += f"  Save directory: {self.save_dir}\n"
        msg += f"  Log file: {self.log_file}\n"
        msg += f"  CSV file: {self.csv_file}\n"
        msg += "="*70
        if distributed.is_main_process:
            print(msg)
            self.logger.info(msg)

        for epoch in range(self.start_epoch, num_epochs + 1):
            set_loader_epoch(train_loader, epoch)
            epoch_start = time.time()
            loss = self.train_epoch(epoch, train_loader, grad_accum_steps=grad_accum_steps)
            epoch_time = time.time() - epoch_start
            validation_metrics = (
                self.validate(val_loader, deployment_val_loader)
                if val_loader is not None
                else None
            )
            
            # 记录日志
            msg = f"\n[Epoch {epoch}/{num_epochs}] Loss: {loss:.4f} | LR: {self.optimizer.param_groups[0]['lr']:.2e} | Time: {epoch_time:.1f}s"
            if distributed.is_main_process:
                print(msg)
                self.logger.info(msg)
                self._log_metrics(epoch, loss, epoch_time)
            
            # 保存最佳模型
            train_improved = loss < self.best_loss
            improved = train_improved
            if validation_metrics is not None:
                selected_metrics = validation_metrics[
                    self.deployment_weight_source
                ]
                current_loss, current_iou = _validated_cd_validation_metrics(
                    selected_metrics,
                    source="selected",
                )
                improved = (
                    current_iou > self.best_val_iou
                    or (
                        current_iou == self.best_val_iou
                        and current_loss < self.best_val_loss
                    )
                )
                if improved:
                    self.best_val_iou = current_iou
                    self.best_val_loss = current_loss
            if train_improved:
                self.best_loss = loss
            if distributed.is_main_process and improved:
                best_ckpt = os.path.join(self.save_dir, "cd_best.pt")
                atomic_torch_save(
                    self._checkpoint_payload(epoch, loss, self.best_loss),
                    best_ckpt,
                )
                msg = f"  ✓ Saved best model (loss: {loss:.4f})"
                self.logger.info(msg)
            
            # 定期保存检查点
            if distributed.is_main_process and epoch % save_every == 0:
                ckpt_path = os.path.join(self.save_dir, f"cd_epoch{epoch:04d}.pt")
                atomic_torch_save(
                    self._checkpoint_payload(epoch, loss, self.best_loss),
                    ckpt_path,
                )
                self.logger.info(f"  Saved checkpoint: {ckpt_path}")
        
        msg = "\nTraining completed!"
        if distributed.is_main_process:
            print(msg)
            self.logger.info(msg)


def main():
    parser = argparse.ArgumentParser(
        description="LDM-initialized EMA consistency training"
    )
    parser.add_argument(
        "--ldm_ckpt",
        type=str,
        required=True,
        help="LDM initialization checkpoint path",
    )
    parser.add_argument("--vae_ckpt", type=str, required=True, help="VAE checkpoint path")
    parser.add_argument("--config", type=str, default="", help="Optional unified YAML config")
    parser.add_argument("--dataset_dir", type=str, default="./Data/NTU4DRadLM_Pre")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--save_dir", type=str, default="./Result/train_results/cd")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="显式恢复 CD checkpoint；不再按输出目录自动探测",
    )
    parser.add_argument(
        "--allow_legacy_radar_units",
        action="store_true",
        help="仅供 mini-test/旧诊断保留未归一化 Radar；正式 CD 禁止使用",
    )
    
    args = parser.parse_args()
    
    config = load_yaml_config(args.config)
    hardware_config = (
        config.get("hardware", {})
        if isinstance(config.get("hardware", {}), dict)
        else {}
    )
    distributed = initialize_distributed(hardware_config.get("device", "cuda"))
    atexit.register(cleanup_distributed, distributed)
    cd_config = dict(config.get("cd", {}) if isinstance(config.get("cd", {}), dict) else {})
    ldm_config = dict(config.get("ldm", {}) if isinstance(config.get("ldm", {}), dict) else {})
    opt_config = dict(config.get("optimization", {}) if isinstance(config.get("optimization", {}), dict) else {})
    data_config = dict(config.get("data", {}) if isinstance(config.get("data", {}), dict) else {})
    runtime_batch_plan = assert_distributed_config_compatible(
        distributed,
        per_rank_batch_size=int(data_config.get("batch_size", args.batch_size)),
        gradient_accumulation_steps=int(
            opt_config.get("gradient_accumulation_steps", args.grad_accum_steps)
        ),
        configured_protocol=hardware_config.get("distributed_protocol"),
        configured_world_size=hardware_config.get("world_size"),
        configured_effective_global_batch_size=hardware_config.get(
            "effective_global_batch_size"
        ),
    )
    checkpoint_protocol = resolve_training_checkpoint_protocol(
        data_config.get("checkpoint_protocol", FORMAL_CHECKPOINT_PROTOCOL)
    )
    formal_training = is_formal_cd_training(
        checkpoint_protocol,
        args.allow_legacy_radar_units,
    )
    if formal_training:
        assert_formal_cd_data_config(data_config)
    dataset_dir = data_config.get("dataset_dir", args.dataset_dir)
    scene_names = data_config.get("scene_names")
    split_artifact_path = data_config.get("temporal_split_artifact")
    data_protocol_path = data_config.get("data_protocol_path")
    if data_protocol_path:
        data_protocol, _data_protocol_sha256 = load_formal_data_protocol_artifact(
            data_protocol_path,
            dataset_dir=dataset_dir,
            scenes=scene_names,
            split_artifact_path=split_artifact_path,
            stage="cd",
        )
    elif args.allow_legacy_radar_units and data_config.get("data_protocol") is None:
        data_protocol = {"protocol": "legacy_data_v0"}
    else:
        data_protocol = validate_checkpoint_data_protocol(
            data_config.get("data_protocol"),
            stage="cd",
        )
    if formal_training:
        data_protocol = prepare_cd_data_protocol(
            data_protocol,
            data_config,
            checkpoint_protocol=checkpoint_protocol,
        )
    train_frame_ids_by_scene = None
    validation_frame_ids_by_scene = None
    deployment_validation_frame_ids_by_scene = None
    deployment_validation_selection = None
    stage_training_selection = None
    split_artifact_sha256 = None
    if formal_training:
        if not split_artifact_path:
            raise ValueError("正式 CD 必须配置 temporal_split_artifact")
        split_artifact, split_artifact_sha256 = load_temporal_split_artifact(
            split_artifact_path,
            dataset_dir=dataset_dir,
            expected_scenes=scene_names,
            require_formal=True,
        )
        if data_protocol.get("split_artifact_sha256") != split_artifact_sha256:
            raise ValueError("CD split artifact 与 data protocol 不一致")
        train_frame_ids_by_scene = split_frame_ids_by_scene(
            split_artifact,
            "train",
        )
        validation_frame_ids_by_scene = split_frame_ids_by_scene(
            split_artifact,
            "validation",
        )
        if checkpoint_protocol == FORMAL_MINI_CHECKPOINT_PROTOCOL:
            train_frame_ids_by_scene = limit_frame_ids_by_scene(
                train_frame_ids_by_scene,
                data_protocol["mini_selection"]["train_frames_per_scene"],
                partition="train",
            )
            validation_frame_ids_by_scene = limit_frame_ids_by_scene(
                validation_frame_ids_by_scene,
                data_protocol["mini_selection"]["validation_frames_per_scene"],
                partition="validation",
            )
        else:
            train_limit = cd_config.get("train_frames_per_epoch", 0)
            validation_limit = cd_config.get(
                "validation_frames_per_epoch",
                0,
            )
            for name, value in (
                ("cd.train_frames_per_epoch", train_limit),
                ("cd.validation_frames_per_epoch", validation_limit),
            ):
                if type(value) is not int or value < 0:
                    raise ValueError(f"{name} 必须是非负整数")
            if train_limit > 0:
                train_frame_ids_by_scene = limit_frame_ids_by_scene(
                    train_frame_ids_by_scene,
                    train_limit,
                    partition="train",
                )
            if validation_limit > 0:
                validation_frame_ids_by_scene = limit_frame_ids_by_scene(
                    validation_frame_ids_by_scene,
                    validation_limit,
                    partition="validation",
                )
            stage_training_selection = build_formal_stage_training_selection(
                stage="cd",
                train_frame_ids_by_scene=train_frame_ids_by_scene,
                validation_frame_ids_by_scene=validation_frame_ids_by_scene,
                configured_train_frames_per_scene=train_limit,
                configured_validation_frames_per_scene=validation_limit,
            )
        deployment_validation_config = resolve_cd_validation_config(cd_config)
        (
            deployment_validation_frame_ids_by_scene,
            deployment_validation_selection,
        ) = build_deployment_validation_selection(
            validation_frame_ids_by_scene,
            frames_per_scene=deployment_validation_config[
                "frames_per_scene"
            ],
        )
    target_size, source_pc_range, model_pc_range = resolve_data_grid_config(data_config)
    ldm_config.setdefault("fusion_voxel_shape", list(target_size))
    ldm_config.setdefault("fusion_pc_range", list(model_pc_range))
    artifact_path = data_config.get("radar_normalization_path")
    scale_mps = data_config.get("doppler_scale_mps")
    path_configured = isinstance(artifact_path, str) and bool(artifact_path.strip())
    scale_configured = scale_mps not in (None, "")
    if args.allow_legacy_radar_units:
        if path_configured or scale_configured:
            raise RadarNormalizationError(
                "CD legacy 开关与正式 normalization 配置不能同时启用"
            )
        radar_normalization, radar_normalization_sha256 = None, ""
    else:
        if not path_configured or not scale_configured:
            raise RadarNormalizationError(
                "正式 CD 必须配置 radar_normalization_path 和 doppler_scale_mps"
            )
        radar_normalization, radar_normalization_sha256 = (
            load_radar_normalization_artifact(
                artifact_path.strip(),
                target_size=target_size,
                source_pc_range=source_pc_range,
                model_pc_range=model_pc_range,
                doppler_scale_mps=scale_mps,
                require_formal=True,
                expected_split_artifact_sha256=split_artifact_sha256,
            )
        )

    # 加载 VAE
    ckpt = safe_torch_load(args.vae_ckpt, map_location='cpu')
    assert_checkpoint_training_identity(
        ckpt,
        expected_stage="vae",
        checkpoint_protocol=checkpoint_protocol,
        data_protocol=data_protocol,
    )
    vae_config = config.get("vae", {}) if isinstance(config.get("vae", {}), dict) else {}
    vae, _vae_metadata = build_cd_vae_from_checkpoint(
        ckpt,
        fallback_config_type=vae_config.get("config_type"),
    )
    
    # 创建数据加载器
    dataset = NTU4DRadLM_VoxelDataset(
        root_dir=dataset_dir,
        split='train',
        use_augmentation=False,
        target_size=target_size,
        source_pc_range=source_pc_range,
        model_pc_range=model_pc_range,
        radar_normalization=radar_normalization,
        radar_normalization_sha256=radar_normalization_sha256,
        allow_legacy_radar_units=args.allow_legacy_radar_units,
        scene_names=scene_names,
        calibration_dir=data_config.get("calibration_dir"),
        require_real_ir=bool(data_config.get("require_real_ir", False)),
        require_real_calibration=bool(
            data_config.get("require_real_calibration", False)
        ),
        require_persisted_observed_mask=bool(
            data_config.get("require_persisted_observed_mask", False)
        ),
        require_radar_statistics=bool(
            data_config.get("require_radar_statistics", False)
        ),
        frame_ids_by_scene=train_frame_ids_by_scene,
        voxel_coordinate_frame=data_config.get("voxel_coordinate_frame", "lidar"),
    )
    train_sampler = None
    if distributed.initialized:
        train_sampler = DistributedSampler(
            dataset,
            num_replicas=distributed.world_size,
            rank=distributed.rank,
            shuffle=True,
            seed=int(data_config.get("training_seed", 42)),
            drop_last=False,
        )
    train_loader = DataLoader(
        dataset,
        batch_size=int(data_config.get("batch_size", args.batch_size)),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=int(data_config.get("num_workers", 4)),
        pin_memory=False,
        collate_fn=collate_voxel_samples,
    )
    val_loader = None
    deployment_val_loader = None
    if formal_training:
        validation_dataset = NTU4DRadLM_VoxelDataset(
            root_dir=dataset_dir,
            split='train',
            return_path=True,
            use_augmentation=False,
            target_size=target_size,
            source_pc_range=source_pc_range,
            model_pc_range=model_pc_range,
            radar_normalization=radar_normalization,
            radar_normalization_sha256=radar_normalization_sha256,
            allow_legacy_radar_units=args.allow_legacy_radar_units,
            scene_names=scene_names,
            calibration_dir=data_config.get("calibration_dir"),
            require_real_ir=bool(data_config.get("require_real_ir", False)),
            require_real_calibration=bool(
                data_config.get("require_real_calibration", False)
            ),
            require_persisted_observed_mask=bool(
                data_config.get("require_persisted_observed_mask", False)
            ),
            require_radar_statistics=bool(
                data_config.get("require_radar_statistics", False)
            ),
            frame_ids_by_scene=validation_frame_ids_by_scene,
            voxel_coordinate_frame=data_config.get(
                "voxel_coordinate_frame", "lidar"
            ),
        )
        validation_sampler = None
        if distributed.initialized:
            validation_sampler = DistributedEvalSampler(
                validation_dataset,
                num_replicas=distributed.world_size,
                rank=distributed.rank,
            )
        val_loader = DataLoader(
            validation_dataset,
            batch_size=int(data_config.get("batch_size", args.batch_size)),
            shuffle=False,
            sampler=validation_sampler,
            num_workers=int(data_config.get("num_workers", 4)),
            pin_memory=False,
            collate_fn=collate_voxel_samples,
        )
        deployment_validation_dataset = NTU4DRadLM_VoxelDataset(
            root_dir=dataset_dir,
            split='train',
            return_path=True,
            use_augmentation=False,
            target_size=target_size,
            source_pc_range=source_pc_range,
            model_pc_range=model_pc_range,
            radar_normalization=radar_normalization,
            radar_normalization_sha256=radar_normalization_sha256,
            allow_legacy_radar_units=args.allow_legacy_radar_units,
            scene_names=scene_names,
            calibration_dir=data_config.get("calibration_dir"),
            require_real_ir=bool(data_config.get("require_real_ir", False)),
            require_real_calibration=bool(
                data_config.get("require_real_calibration", False)
            ),
            require_persisted_observed_mask=bool(
                data_config.get("require_persisted_observed_mask", False)
            ),
            require_radar_statistics=bool(
                data_config.get("require_radar_statistics", False)
            ),
            frame_ids_by_scene=deployment_validation_frame_ids_by_scene,
            voxel_coordinate_frame=data_config.get(
                "voxel_coordinate_frame", "lidar"
            ),
        )
        deployment_validation_sampler = None
        if distributed.initialized:
            deployment_validation_sampler = DistributedEvalSampler(
                deployment_validation_dataset,
                num_replicas=distributed.world_size,
                rank=distributed.rank,
            )
        deployment_val_loader = DataLoader(
            deployment_validation_dataset,
            batch_size=int(data_config.get("batch_size", args.batch_size)),
            shuffle=False,
            sampler=deployment_validation_sampler,
            num_workers=int(data_config.get("num_workers", 4)),
            pin_memory=False,
            collate_fn=collate_voxel_samples,
        )
    
    cd_save_dir = cd_config.get("save_dir", args.save_dir)

    # 恢复必须由 CLI 明确授权，禁止跨协议目录自动续训。
    resume_path = args.resume or None
    if resume_path and not os.path.isfile(resume_path):
        raise FileNotFoundError(f"显式 CD resume checkpoint 不存在: {resume_path}")
    
    # 创建训练器并训练
    trainer = ConsistencyDistillationTrainer(
        ldm_ckpt_path=args.ldm_ckpt,
        vae=vae,
        config={
            'lr': float(cd_config.get("lr", args.lr)),
            'save_dir': cd_save_dir,
            'resume_path': resume_path,
            'ldm': ldm_config,
            'data_grid_config': {
                'target_size': list(target_size),
                'source_pc_range': list(source_pc_range),
                'model_pc_range': list(model_pc_range),
            },
            'vae_checkpoint_sha256': sha256_file(args.vae_ckpt),
            'ldm_checkpoint_sha256': sha256_file(args.ldm_ckpt),
            'radar_normalization': radar_normalization,
            'radar_normalization_sha256': radar_normalization_sha256,
            'allow_legacy_radar_units': args.allow_legacy_radar_units,
            'checkpoint_protocol': checkpoint_protocol,
            'data_protocol': data_protocol,
            'stage_training_selection': stage_training_selection,
            'deployment_validation_selection': (
                deployment_validation_selection
            ),
            'distributed_context': distributed,
            'expected_effective_global_batch_size': (
                runtime_batch_plan.effective_global_batch_size
            ),
            'require_persisted_observed_mask': bool(
                data_config.get("require_persisted_observed_mask", False)
            ),
            'validation_seed': cd_config.get('validation_seed', 42),
            'validation_sigma': cd_config.get('validation_sigma', 0.5),
            'validation_occupancy_threshold': cd_config.get(
                'validation_occupancy_threshold', 0.5
            ),
            'validation_threshold_candidates': cd_config.get(
                'validation_threshold_candidates',
                DEFAULT_THRESHOLD_CANDIDATES,
            ),
            'deployment_validation_frames_per_scene': cd_config.get(
                'deployment_validation_frames_per_scene', 16
            ),
            'deployment_validation_seed': cd_config.get(
                'deployment_validation_seed',
                cd_config.get('validation_seed', 42),
            ),
            'deployment_validation_steps': cd_config.get(
                'deployment_validation_steps', 1
            ),
            'deployment_validation_sampler': cd_config.get(
                'deployment_validation_sampler', 'one_step'
            ),
            'deployment_validation_occupancy_threshold': cd_config.get(
                'deployment_validation_occupancy_threshold',
                cd_config.get('validation_occupancy_threshold', 0.5),
            ),
            'deployment_validation_threshold_candidates': cd_config.get(
                'deployment_validation_threshold_candidates',
                cd_config.get(
                    'validation_threshold_candidates',
                    DEFAULT_THRESHOLD_CANDIDATES,
                ),
            ),
            'deployment_validation_min_occupied_recall': cd_config.get(
                'deployment_validation_min_occupied_recall'
            ),
            'deployment_validation_recall_constraint_authority': cd_config.get(
                'deployment_validation_recall_constraint_authority', ''
            ),
            'training_semantics': cd_config.get(
                'training_semantics', CD_TRAINING_SEMANTICS
            ),
            'num_scales': cd_config.get('num_scales', 40),
            'ema_rate': cd_config.get('ema_rate', 0.999),
            'sigma_min': cd_config.get('sigma_min', 0.002),
            'sigma_max': cd_config.get('sigma_max', 80.0),
            'rho': cd_config.get('rho', 7.0),
            'consistency_loss_weight': cd_config.get(
                'consistency_loss_weight', 1.0
            ),
            'reconstruction_anchor_weight': cd_config.get(
                'reconstruction_anchor_weight', 0.1
            ),
            'collapse_guard_epsilon': cd_config.get(
                'collapse_guard_epsilon', 0.0
            ),
        },
    )
    
    trainer.train(
        train_loader,
        val_loader,
        deployment_val_loader,
        num_epochs=int(cd_config.get("epochs", args.num_epochs)),
        save_every=int(cd_config.get("save_every", 10)),
        grad_accum_steps=int(opt_config.get("gradient_accumulation_steps", args.grad_accum_steps)),
    )
    if (
        distributed.is_main_process
        and checkpoint_protocol == FORMAL_CHECKPOINT_PROTOCOL
    ):
        best_checkpoint_path = os.path.join(cd_save_dir, "cd_best.pt")
        best_checkpoint = safe_torch_load(
            best_checkpoint_path,
            map_location="cpu",
        )
        threshold_artifact = build_threshold_artifact(
            best_checkpoint,
            checkpoint_path=best_checkpoint_path,
        )
        write_threshold_artifact(
            os.path.join(cd_save_dir, "occupancy_threshold.json"),
            threshold_artifact,
        )
    
    if distributed.is_main_process:
        print("Training completed!")
    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
