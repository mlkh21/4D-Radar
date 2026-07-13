# -- coding: utf-8 --
"""
统一训练脚本 - 整合 VAE、LDM、CD 训练

优化点：
1. 统一配置系统 - 使用 YAML 配置，避免代码中硬编码参数
2. 显存优化 - 梯度累积、检查点、稀疏处理
3. 蒸馏流程优化 - 清晰的 LDM -> CD 蒸馏步骤
4. 模块化架构 - 易于维护和扩展
"""

import argparse
import math
import os
import random
import shutil
import sys
import yaml
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple
from torch.utils.data import DataLoader, Subset
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.optim.adamw import AdamW
import gc
from tqdm import tqdm
import time
import csv
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cm.vae_3d import (
    VAE3D, 
    build_vae_from_checkpoint,
    create_vae_config,
    create_lightweight_vae_config,
    create_ultra_lightweight_vae_config,
    create_standard_vae_config,
)
from cm.unet_optimized import OptimizedUNetModel, create_lightweight_unet_config
from cm.dataset_loader import NTU4DRadLM_VoxelDataset
from cm.karras_diffusion import KarrasDenoiser
from cm.multimodal_fusion import (
    CompleteDualModalityPerceptionNet,
    heteroscedastic_gaussian_nll,
    migrate_ir_gate_state_dict,
)
from scripts.cd_train_optimized import ConsistencyDistillationTrainer


def atomic_torch_save(payload: Any, path: str):
    """同目录写临时文件后原子替换 checkpoint。"""
    temp_path = f"{path}.tmp-{os.getpid()}-{time.time_ns()}"
    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def atomic_copy_file(source: str, destination: str):
    """通过独立临时文件原子更新 checkpoint 兼容别名。"""
    temp_path = f"{destination}.tmp-{os.getpid()}-{time.time_ns()}"
    try:
        shutil.copyfile(source, temp_path)
        os.replace(temp_path, destination)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def unpack_training_batch(batch):
    """Support legacy (target, radar) and multimodal (target, radar, meta[, path]) batches."""
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


def move_meta_to_device(meta: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved = {}
    for key, value in (meta or {}).items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def pad_ldm_input_to_sixteen_channels(model_input: torch.Tensor) -> torch.Tensor:
    if model_input.shape[1] == 16:
        return model_input
    if model_input.shape[1] > 16:
        raise ValueError(f"LDM model input has {model_input.shape[1]} channels, expected <= 16")
    pad_channels = 16 - model_input.shape[1]
    pad = torch.zeros(
        model_input.shape[0],
        pad_channels,
        *model_input.shape[2:],
        device=model_input.device,
        dtype=model_input.dtype,
    )
    return torch.cat([model_input, pad], dim=1)


def has_multimodal_meta(meta: Dict[str, Any]) -> bool:
    required = ("ir_img", "r_mat", "t_vec", "k_mat")
    return all(torch.is_tensor((meta or {}).get(key)) for key in required)


def resolve_cd_teacher_checkpoint(args_ldm_ckpt: str, config: "ConfigManager") -> str:
    """Resolve the CD teacher checkpoint from CLI first, then YAML config."""
    return args_ldm_ckpt or config.get('cd.teacher_model_path', '')


def safe_torch_load(path, map_location):
    """兼容不同 PyTorch 版本的 checkpoint 加载逻辑。"""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # HACK: 低版本 PyTorch 不支持 weights_only，回退到兼容模式。
        return torch.load(path, map_location=map_location)
    except Exception as exc:
        # NOTE: 某些历史权重包含自定义对象，weights_only=True 会拒绝加载。
        msg = str(exc)
        if "Weights only load failed" in msg or "Unsupported global" in msg:
            return torch.load(path, map_location=map_location)
        raise


def apply_vae_config_overrides(vae_config: Dict[str, Any], config: "ConfigManager") -> Dict[str, Any]:
    """将 YAML 中允许的 VAE 架构与损失配置覆盖到 preset。"""
    merged = dict(vae_config)
    yaml_vae = config.get('vae', {}) or {}
    for key in (
        "latent_dim",
        "kl_weight",
        "occupied_weight",
        "empty_weight",
        "channel_weights",
        "false_positive_weight",
        "occupancy_mass_weight",
        "occupancy_loss_type",
        "occupancy_bce_weight",
        "occupancy_dice_weight",
        "occupancy_pos_weight_cap",
        "continuous_recon_weight",
    ):
        if key in yaml_vae:
            value = yaml_vae[key]
            if key == "channel_weights" and value is not None:
                value = tuple(float(v) for v in value)
            merged[key] = value
    return merged


def deterministic_split_indices(
    dataset_size: int,
    train_split: float = 0.8,
    split_seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """按固定随机种子划分非空训练集和验证集。"""
    if dataset_size < 2:
        raise ValueError("训练/验证划分至少需要 2 个样本")
    if not 0.0 < train_split < 1.0:
        raise ValueError("data.train_split 必须严格位于 (0, 1)")
    train_size = int(dataset_size * train_split)
    if train_size <= 0 or train_size >= dataset_size:
        raise ValueError(
            f"train_split={train_split} 导致空划分："
            f"dataset_size={dataset_size}, train_size={train_size}"
        )
    generator = torch.Generator().manual_seed(int(split_seed))
    indices = torch.randperm(dataset_size, generator=generator).tolist()
    return indices[:train_size], indices[train_size:]


def seed_training_run(training_seed: int) -> torch.Generator:
    """统一固定模型初始化、数据增强与 DataLoader shuffle 的随机状态。"""
    seed = int(training_seed)
    if seed < 0 or seed > 2**32 - 1:
        raise ValueError(f"data.training_seed 必须位于 [0, 2^32-1]，当前为 {training_seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return torch.Generator().manual_seed(seed)


def micro_occupancy_metrics(
    probability: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, int]:
    """返回可跨 batch 累加的 occupancy 微平均计数。"""
    prediction = probability >= threshold
    truth = target >= 0.5
    return {
        "intersection": int(torch.logical_and(prediction, truth).sum().item()),
        "union": int(torch.logical_or(prediction, truth).sum().item()),
        "target_positive": int(truth.sum().item()),
        "predicted_positive": int(prediction.sum().item()),
    }


def decoded_occupancy_auxiliary_loss(
    decoded: torch.Tensor,
    target: torch.Tensor,
    occupancy_activation: str,
    reconstruction_weight: float = 1.0,
    false_positive_weight: float = 0.0,
    mass_weight: float = 0.0,
) -> torch.Tensor:
    """按 VAE occupancy 语义计算 LDM 解码辅助损失。"""
    if occupancy_activation not in {"raw", "sigmoid"}:
        raise ValueError(f"不支持 occupancy_activation={occupancy_activation!r}")
    decoded_occ = decoded[:, 0:1]
    if occupancy_activation == "sigmoid":
        decoded_occ = torch.sigmoid(decoded_occ)
    target_occ = target[:, 0:1]
    occ_mask = (target_occ > 0).float()
    reconstruction_loss = (
        (decoded_occ - target_occ) ** 2 * (1.0 + 7.0 * occ_mask)
    ).mean()
    loss = reconstruction_weight * reconstruction_loss
    if false_positive_weight > 0.0:
        empty_mask = 1.0 - occ_mask
        false_positive_loss = (
            torch.relu(decoded_occ - target_occ) ** 2 * empty_mask
        ).mean()
        loss = loss + false_positive_weight * false_positive_loss
    if mass_weight > 0.0:
        mass_loss = torch.abs(
            torch.relu(decoded_occ).mean() - torch.relu(target_occ).mean()
        )
        loss = loss + mass_weight * mass_loss
    return loss


def decoded_vertical_structure_losses(
    decoded: torch.Tensor,
    target: torch.Tensor,
    occupancy_activation: str,
    column_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """计算仅由目标非空 Z 列监督的可微垂直结构损失。"""
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"eps 必须是有限正数，实际为 {eps!r}")
    if decoded.ndim != 5 or target.ndim != 5:
        raise ValueError(
            "decoded 和 target 必须是 5 维 [B,C,Z,X,Y] 张量，"
            f"实际分别为 {decoded.ndim} 维和 {target.ndim} 维"
        )
    if decoded.shape[1] < 1 or target.shape[1] < 1:
        raise ValueError("decoded 和 target 必须至少 1 个通道")
    decoded_grid = tuple(decoded.shape[index] for index in (0, 2, 3, 4))
    target_grid = tuple(target.shape[index] for index in (0, 2, 3, 4))
    if decoded_grid != target_grid:
        raise ValueError(
            "decoded 与 target 的 B/Z/X/Y 必须一致，"
            f"实际分别为 {decoded_grid} 和 {target_grid}"
        )
    if occupancy_activation not in {"sigmoid", "raw"}:
        raise ValueError(f"不支持 occupancy_activation={occupancy_activation!r}")

    decoded_occ_raw = decoded[:, 0].float()
    if occupancy_activation == "sigmoid":
        decoded_occ = torch.sigmoid(decoded_occ_raw)
    else:
        # 兼容历史 raw occupancy/probability 语义，区间外梯度按 clamp 处理。
        decoded_occ = decoded_occ_raw.clamp(0.0, 1.0)
    target_occ = target[:, 0].clamp(0.0, 1.0).float()
    valid_columns = target_occ.sum(dim=1) > 0
    if column_mask is not None:
        if column_mask.ndim == 5:
            column_mask = column_mask[:, 0].bool().any(dim=1)
        elif column_mask.ndim == 4:
            column_mask = column_mask[:, 0].bool() if column_mask.shape[1] == 1 else column_mask.bool().any(dim=1)
        elif column_mask.ndim != 3:
            raise ValueError(f"column_mask 必须能规约为 [B,X,Y]，实际形状为 {tuple(column_mask.shape)}")
        if tuple(column_mask.shape) != tuple(valid_columns.shape):
            raise ValueError(
                "column_mask 与 target 的 B/X/Y 必须一致，"
                f"实际分别为 {tuple(column_mask.shape)} 和 {tuple(valid_columns.shape)}"
            )
        valid_columns = valid_columns & column_mask.to(valid_columns.device)
    graph_zero = decoded_occ.sum() * 0.0

    if not valid_columns.any():
        return {
            "height_distribution_loss": graph_zero,
            "top_height_loss": graph_zero,
            "top_overshoot_loss": graph_zero,
            "vertical_continuity_loss": graph_zero,
        }

    pred_mass = decoded_occ.sum(dim=1, keepdim=True)
    pred_denominator = torch.where(
        pred_mass > eps,
        pred_mass,
        torch.ones_like(pred_mass),
    )
    pred_distribution = decoded_occ / pred_denominator
    target_distribution = target_occ / target_occ.sum(
        dim=1, keepdim=True
    ).clamp_min(eps)
    cdf_difference = torch.abs(
        pred_distribution.cumsum(dim=1) - target_distribution.cumsum(dim=1)
    )
    height_distribution_loss = cdf_difference.mean(dim=1)[valid_columns].mean()

    z_indices = torch.arange(
        decoded_occ.shape[1],
        dtype=torch.long,
        device=decoded_occ.device,
    ).view(1, -1, 1, 1)
    target_top_indices = torch.where(
        target_occ > 0.0,
        z_indices,
        torch.full_like(z_indices, -1),
    ).max(dim=1, keepdim=True).values.clamp_min(0)
    if occupancy_activation == "sigmoid":
        top_logits = torch.gather(decoded_occ_raw, dim=1, index=target_top_indices)
        # 对 target 顶部体素用正类 BCE，避免高处漏检在 sigmoid 饱和区梯度太弱。
        top_height_grid = torch.nn.functional.softplus(-top_logits).squeeze(1)
    else:
        top_probability = torch.gather(decoded_occ, dim=1, index=target_top_indices)
        top_height_grid = (1.0 - top_probability).squeeze(1)
    top_height_loss = top_height_grid[valid_columns].mean()

    above_target_mask = (
        (z_indices > target_top_indices) & valid_columns.unsqueeze(1)
    )
    if not above_target_mask.any():
        top_overshoot_loss = graph_zero
    elif occupancy_activation == "sigmoid":
        # target top 以上均为负类，直接用 logits 形式 BCE 保持极端值数值稳定。
        top_overshoot_grid = torch.nn.functional.softplus(decoded_occ_raw)
        top_overshoot_loss = top_overshoot_grid[above_target_mask].mean()
    else:
        top_overshoot_loss = decoded_occ.square()[above_target_mask].mean()

    if decoded_occ.shape[1] < 2:
        vertical_continuity_loss = graph_zero
    else:
        pred_transitions = torch.abs(
            decoded_occ[:, 1:] - decoded_occ[:, :-1]
        )
        target_transitions = torch.abs(
            target_occ[:, 1:] - target_occ[:, :-1]
        )
        continuity_difference = torch.abs(pred_transitions - target_transitions)
        vertical_continuity_loss = continuity_difference.mean(dim=1)[
            valid_columns
        ].mean()

    return {
        "height_distribution_loss": height_distribution_loss,
        "top_height_loss": top_height_loss,
        "top_overshoot_loss": top_overshoot_loss,
        "vertical_continuity_loss": vertical_continuity_loss,
    }


def decoded_density_precision_loss(
    decoded: torch.Tensor,
    target: torch.Tensor,
    occupancy_activation: str,
) -> torch.Tensor:
    """惩罚背景竖列假阳性，同时尽量保留树干/树冠所在竖列。"""
    if decoded.ndim != 5 or target.ndim != 5:
        raise ValueError(
            "decoded 和 target 必须是 5 维 [B,C,Z,X,Y] 张量，"
            f"实际分别为 {decoded.ndim} 维和 {target.ndim} 维"
        )
    if decoded.shape[1] < 1 or target.shape[1] < 1:
        raise ValueError("decoded 和 target 必须至少 1 个通道")
    decoded_grid = tuple(decoded.shape[index] for index in (0, 2, 3, 4))
    target_grid = tuple(target.shape[index] for index in (0, 2, 3, 4))
    if decoded_grid != target_grid:
        raise ValueError(
            "decoded 与 target 的 B/Z/X/Y 必须一致，"
            f"实际分别为 {decoded_grid} 和 {target_grid}"
        )
    if occupancy_activation not in {"sigmoid", "raw"}:
        raise ValueError(f"不支持 occupancy_activation={occupancy_activation!r}")

    decoded_occ_raw = decoded[:, 0:1].float()
    if occupancy_activation == "sigmoid":
        decoded_occ = torch.sigmoid(decoded_occ_raw)
    else:
        # raw 兼容历史 VAE 概率输出；区间外预测按概率边界裁剪。
        decoded_occ = decoded_occ_raw.clamp(0.0, 1.0)
    target_occ = target[:, 0:1].float().clamp(0.0, 1.0)
    target_column_has_occ = (target_occ.sum(dim=2, keepdim=True) > 0.0).float()
    empty_column_mask = 1.0 - target_column_has_occ

    # 只对 target 完全空的 (X,Y) 竖列施加强假阳性惩罚；已有障碍物的竖列
    # 交给高度分布/连续性损失塑形，避免过早压掉树干的 Z 向补全。
    expanded_empty_column_mask = empty_column_mask.expand_as(decoded_occ)
    empty_column_denominator = expanded_empty_column_mask.sum().clamp_min(1.0)
    if occupancy_activation == "sigmoid":
        # 对 logit 使用空类 BCE，避免高置信背景柱在 sigmoid 饱和区梯度过弱。
        false_positive_values = torch.nn.functional.softplus(decoded_occ_raw)
    else:
        false_positive_values = decoded_occ.square()
    false_positive_loss = (
        false_positive_values * expanded_empty_column_mask
    ).sum() / empty_column_denominator
    return false_positive_loss


def _validate_decoded_occupancy_inputs(
    decoded: torch.Tensor,
    target: torch.Tensor,
    occupancy_activation: str,
) -> None:
    """校验解码 occupancy 损失共享的输入布局与激活语义。"""
    if decoded.ndim != 5 or target.ndim != 5:
        raise ValueError(
            "decoded 和 target 必须是 5 维 [B,C,Z,X,Y] 张量，"
            f"实际分别为 {decoded.ndim} 维和 {target.ndim} 维"
        )
    if decoded.shape[1] < 1 or target.shape[1] < 1:
        raise ValueError("decoded 和 target 必须至少 1 个通道")
    decoded_grid = tuple(decoded.shape[index] for index in (0, 2, 3, 4))
    target_grid = tuple(target.shape[index] for index in (0, 2, 3, 4))
    if decoded_grid != target_grid:
        raise ValueError(
            "decoded 与 target 的 B/Z/X/Y 必须一致，"
            f"实际分别为 {decoded_grid} 和 {target_grid}"
        )
    if occupancy_activation not in {"sigmoid", "raw"}:
        raise ValueError(f"不支持 occupancy_activation={occupancy_activation!r}")


def decoded_column_balanced_losses(
    decoded: torch.Tensor,
    target: torch.Tensor,
    occupancy_activation: str,
    temperature: float = 1.0,
    target_threshold: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """按正负竖列分别平均 column-existence 二分类损失。"""
    _validate_decoded_occupancy_inputs(decoded, target, occupancy_activation)
    if any(size == 0 for size in decoded.shape[2:]):
        raise ValueError(
            f"decoded 和 target 的 Z/X/Y 必须大于 0，实际为 {tuple(decoded.shape[2:])}"
        )
    if decoded.device != target.device:
        raise ValueError(
            "decoded 与 target 的 device 必须一致，"
            f"实际分别为 {decoded.device} 和 {target.device}"
        )
    if not math.isfinite(temperature) or not 1e-3 <= temperature <= 100.0:
        raise ValueError(
            f"temperature 必须是 [1e-3,100.0] 内的有限数，实际为 {temperature!r}"
        )
    if not math.isfinite(target_threshold) or not 0.0 <= target_threshold <= 1.0:
        raise ValueError(
            f"target_threshold 必须是 [0,1] 内的有限数，实际为 {target_threshold!r}"
        )

    # 聚合统一提升到 float32，避免低精度输入放大 logsumexp 数值误差。
    decoded_occ = decoded[:, 0:1].float()
    if occupancy_activation == "sigmoid":
        voxel_logits = decoded_occ
    else:
        voxel_logits = torch.logit(decoded_occ.clamp(1e-6, 1.0 - 1e-6))

    z_size = voxel_logits.shape[2]
    column_logits = temperature * (
        torch.logsumexp(voxel_logits / temperature, dim=2)
        - math.log(z_size)
    )
    positive_columns = (target[:, 0:1] >= target_threshold).any(dim=2)
    negative_columns = ~positive_columns
    graph_zero = column_logits.sum() * 0.0

    positive_loss = (
        torch.nn.functional.softplus(-column_logits)[positive_columns].mean()
        if positive_columns.any()
        else graph_zero
    )
    negative_loss = (
        torch.nn.functional.softplus(column_logits)[negative_columns].mean()
        if negative_columns.any()
        else graph_zero
    )
    return {
        "positive_loss": positive_loss,
        "negative_loss": negative_loss,
    }


def decoded_ir_frustum_occupancy_loss(
    decoded: torch.Tensor,
    target: torch.Tensor,
    occupancy_activation: str,
    frustum_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """只在 IR 视锥内的 target 正样本上强化 occupancy 召回。"""
    _validate_decoded_occupancy_inputs(decoded, target, occupancy_activation)
    decoded_occ_raw = decoded[:, 0:1].float()
    target_occ = target[:, 0:1].float().clamp(0.0, 1.0)
    graph_zero = decoded_occ_raw.sum() * 0.0
    if frustum_mask is None:
        return graph_zero
    if frustum_mask.ndim != 5:
        raise ValueError(f"frustum_mask 必须是 [B,1,Z,X,Y]，实际为 {tuple(frustum_mask.shape)}")
    if frustum_mask.shape[0] != target_occ.shape[0]:
        raise ValueError(
            "frustum_mask 的 batch 必须与 target 一致，"
            f"实际分别为 {frustum_mask.shape[0]} 和 {target_occ.shape[0]}"
        )
    if frustum_mask.shape[1] != 1:
        raise ValueError(
            "frustum_mask 的 channel 必须为 1，"
            f"实际为 {frustum_mask.shape[1]}"
        )
    if tuple(frustum_mask.shape[-3:]) != tuple(target_occ.shape[-3:]):
        frustum_mask = torch.nn.functional.interpolate(
            frustum_mask.float(),
            size=target_occ.shape[-3:],
            mode="nearest",
        ).bool()
    positive_mask = (target_occ > 0.0) & frustum_mask.to(target_occ.device).bool()
    if not positive_mask.any():
        return graph_zero
    if occupancy_activation == "sigmoid":
        positive_loss = torch.nn.functional.softplus(-decoded_occ_raw)
    elif occupancy_activation == "raw":
        decoded_occ = decoded_occ_raw.clamp(0.0, 1.0)
        positive_loss = 1.0 - decoded_occ
    else:
        raise ValueError(f"不支持 occupancy_activation={occupancy_activation!r}")
    return positive_loss[positive_mask].mean()


def decoded_ir_frustum_negative_occupancy_loss(
    decoded: torch.Tensor,
    target: torch.Tensor,
    occupancy_activation: str,
    frustum_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """只在 IR 视锥内的 target 负样本上抑制 occupancy 假阳性。"""
    _validate_decoded_occupancy_inputs(decoded, target, occupancy_activation)
    decoded_occ_raw = decoded[:, 0:1].float()
    target_occ = target[:, 0:1].float().clamp(0.0, 1.0)
    graph_zero = decoded_occ_raw.sum() * 0.0
    if frustum_mask is None:
        return graph_zero
    if frustum_mask.ndim != 5:
        raise ValueError(f"frustum_mask 必须是 [B,1,Z,X,Y]，实际为 {tuple(frustum_mask.shape)}")
    if frustum_mask.shape[0] != target_occ.shape[0]:
        raise ValueError(
            "frustum_mask 的 batch 必须与 target 一致，"
            f"实际分别为 {frustum_mask.shape[0]} 和 {target_occ.shape[0]}"
        )
    if frustum_mask.shape[1] != 1:
        raise ValueError(
            "frustum_mask 的 channel 必须为 1，"
            f"实际为 {frustum_mask.shape[1]}"
        )
    if tuple(frustum_mask.shape[-3:]) != tuple(target_occ.shape[-3:]):
        frustum_mask = torch.nn.functional.interpolate(
            frustum_mask.float(),
            size=target_occ.shape[-3:],
            mode="nearest",
        ).bool()
    negative_mask = (target_occ == 0.0) & frustum_mask.to(target_occ.device).bool()
    if not negative_mask.any():
        return graph_zero
    if occupancy_activation == "sigmoid":
        # 直接使用负类 BCE 的 logits 形式，避免高置信假阳性处梯度饱和。
        negative_loss = torch.nn.functional.softplus(decoded_occ_raw)
    elif occupancy_activation == "raw":
        decoded_occ = decoded_occ_raw.clamp(0.0, 1.0)
        negative_loss = decoded_occ.square()
    else:
        raise ValueError(f"不支持 occupancy_activation={occupancy_activation!r}")
    return negative_loss[negative_mask].mean()


LDM_LOSS_COMPONENT_NAMES = (
    "latent_loss",
    "decoded_occupancy_loss",
    "height_distribution_loss",
    "top_height_loss",
    "top_overshoot_loss",
    "vertical_continuity_loss",
    "decoded_density_loss",
    "ir_frustum_occupancy_loss",
    "ir_frustum_negative_loss",
    "ir_frustum_top_height_loss",
    "column_positive_loss",
    "column_negative_loss",
    "uncertainty_loss",
)
LDM_META_COMPONENT_NAMES = (
    "mock_ir_ratio",
    "mock_calib_ratio",
    "ir_frustum_voxel_ratio",
)
LDM_METRICS_HEADER = (
    "epoch",
    "step",
    "loss",
    *LDM_LOSS_COMPONENT_NAMES,
    *LDM_META_COMPONENT_NAMES,
    "lr",
    "time_seconds",
)


def archive_legacy_metrics_csv(csv_file: str):
    """把旧 metrics.csv 归档到同目录不覆盖的 legacy 文件名。"""
    legacy_path = os.path.join(os.path.dirname(csv_file), "metrics_legacy.csv")
    suffix = 1
    while os.path.exists(legacy_path):
        legacy_path = os.path.join(
            os.path.dirname(csv_file),
            f"metrics_legacy_{suffix}.csv",
        )
        suffix += 1
    os.replace(csv_file, legacy_path)


def prepare_ldm_metrics_csv(csv_file: str, is_resumed: bool):
    """初始化 LDM CSV；恢复旧表头时先归档旧文件再写新表头。"""
    if is_resumed and os.path.exists(csv_file):
        with open(csv_file, newline="") as f:
            header = next(csv.reader(f), [])
        if tuple(header) == LDM_METRICS_HEADER:
            return
        archive_legacy_metrics_csv(csv_file)

    if not os.path.exists(csv_file) or not is_resumed:
        with open(csv_file, "w", newline="") as f:
            csv.writer(f).writerow(LDM_METRICS_HEADER)


def rescale_accumulated_gradients(
    parameters,
    grad_accum_steps: int,
    accumulation_count: int,
):
    """尾部累计不足 grad_accum_steps 时恢复为实际 batch 均值梯度。"""
    if accumulation_count <= 0 or accumulation_count == grad_accum_steps:
        return
    scale = float(grad_accum_steps) / float(accumulation_count)
    for param in parameters:
        if param.grad is not None:
            param.grad.mul_(scale)


def compute_ldm_loss_components(
    denoised: torch.Tensor,
    z_target: torch.Tensor,
    target: torch.Tensor,
    vae: nn.Module,
    occupancy_activation: str,
    decoded_loss_weight: float,
    decoded_false_positive_weight: float,
    decoded_mass_weight: float,
    decoded_height_distribution_weight: float,
    decoded_top_height_weight: float,
    decoded_vertical_continuity_weight: float,
    decoded_density_weight: float,
    decoded_top_overshoot_weight: float = 0.0,
    decoded_ir_frustum_occupancy_weight: float = 0.0,
    decoded_ir_frustum_negative_weight: float = 0.0,
    decoded_ir_frustum_top_weight: float = 0.0,
    ir_frustum_mask: Optional[torch.Tensor] = None,
    uncertainty_loss_weight: float = 0.0,
    uncertainty: Optional[Dict[str, torch.Tensor]] = None,
    decoded_column_positive_weight: float = 0.0,
    decoded_column_negative_weight: float = 0.0,
    decoded_column_temperature: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """汇总 LDM 主损失和解码辅助监督组件，必要时只解码一次。"""
    latent_loss = torch.nn.functional.mse_loss(denoised, z_target)
    graph_zero = denoised.sum() * 0.0
    components = {
        "latent_loss": latent_loss,
        "decoded_occupancy_loss": graph_zero,
        "height_distribution_loss": graph_zero,
        "top_height_loss": graph_zero,
        "top_overshoot_loss": graph_zero,
        "vertical_continuity_loss": graph_zero,
        "decoded_density_loss": graph_zero,
        "ir_frustum_occupancy_loss": graph_zero,
        "ir_frustum_negative_loss": graph_zero,
        "ir_frustum_top_height_loss": graph_zero,
        "column_positive_loss": graph_zero,
        "column_negative_loss": graph_zero,
        "uncertainty_loss": graph_zero,
    }
    loss = latent_loss

    if uncertainty_loss_weight > 0.0 and uncertainty is not None:
        if "variance" not in uncertainty:
            raise ValueError("uncertainty loss requires uncertainty['variance']")
        uncertainty_loss = heteroscedastic_gaussian_nll(
            denoised,
            z_target,
            uncertainty["variance"],
            detach_residual=True,
        )
        components["uncertainty_loss"] = uncertainty_loss
        loss = loss + uncertainty_loss_weight * uncertainty_loss

    needs_decoded = any(
        weight > 0.0
        for weight in (
            decoded_loss_weight,
            decoded_false_positive_weight,
            decoded_mass_weight,
            decoded_height_distribution_weight,
            decoded_top_height_weight,
            decoded_top_overshoot_weight,
            decoded_vertical_continuity_weight,
            decoded_density_weight,
            decoded_ir_frustum_occupancy_weight,
            decoded_ir_frustum_negative_weight,
            decoded_ir_frustum_top_weight,
            decoded_column_positive_weight,
            decoded_column_negative_weight,
        )
    )
    if needs_decoded:
        decoded = vae.decode(denoised)
        if (
            decoded_loss_weight > 0.0
            or decoded_false_positive_weight > 0.0
            or decoded_mass_weight > 0.0
        ):
            decoded_occ_loss = decoded_occupancy_auxiliary_loss(
                decoded,
                target,
                occupancy_activation=occupancy_activation,
                reconstruction_weight=decoded_loss_weight,
                false_positive_weight=decoded_false_positive_weight,
                mass_weight=decoded_mass_weight,
            )
            components["decoded_occupancy_loss"] = decoded_occ_loss
            loss = loss + decoded_occ_loss
        if (
            decoded_height_distribution_weight > 0.0
            or decoded_top_height_weight > 0.0
            or decoded_top_overshoot_weight > 0.0
            or decoded_vertical_continuity_weight > 0.0
        ):
            structure_losses = decoded_vertical_structure_losses(
                decoded,
                target,
                occupancy_activation=occupancy_activation,
            )
            height_loss = structure_losses["height_distribution_loss"]
            top_loss = structure_losses["top_height_loss"]
            top_overshoot_loss = structure_losses["top_overshoot_loss"]
            continuity_loss = structure_losses["vertical_continuity_loss"]
            components["height_distribution_loss"] = height_loss
            components["top_height_loss"] = top_loss
            components["top_overshoot_loss"] = top_overshoot_loss
            components["vertical_continuity_loss"] = continuity_loss
            loss = loss + decoded_height_distribution_weight * height_loss
            loss = loss + decoded_top_height_weight * top_loss
            loss = loss + decoded_top_overshoot_weight * top_overshoot_loss
            loss = loss + decoded_vertical_continuity_weight * continuity_loss
        if decoded_density_weight > 0.0:
            density_loss = decoded_density_precision_loss(
                decoded,
                target,
                occupancy_activation=occupancy_activation,
            )
            components["decoded_density_loss"] = density_loss
            loss = loss + decoded_density_weight * density_loss
        if decoded_ir_frustum_occupancy_weight > 0.0:
            ir_occ_loss = decoded_ir_frustum_occupancy_loss(
                decoded,
                target,
                occupancy_activation=occupancy_activation,
                frustum_mask=ir_frustum_mask,
            )
            components["ir_frustum_occupancy_loss"] = ir_occ_loss
            loss = loss + decoded_ir_frustum_occupancy_weight * ir_occ_loss
        if decoded_ir_frustum_negative_weight > 0.0:
            ir_negative_loss = decoded_ir_frustum_negative_occupancy_loss(
                decoded,
                target,
                occupancy_activation=occupancy_activation,
                frustum_mask=ir_frustum_mask,
            )
            components["ir_frustum_negative_loss"] = ir_negative_loss
            loss = loss + decoded_ir_frustum_negative_weight * ir_negative_loss
        if decoded_ir_frustum_top_weight > 0.0:
            ir_structure_losses = decoded_vertical_structure_losses(
                decoded,
                target,
                occupancy_activation=occupancy_activation,
                column_mask=ir_frustum_mask,
            )
            ir_top_loss = ir_structure_losses["top_height_loss"]
            components["ir_frustum_top_height_loss"] = ir_top_loss
            loss = loss + decoded_ir_frustum_top_weight * ir_top_loss
        if (
            decoded_column_positive_weight > 0.0
            or decoded_column_negative_weight > 0.0
        ):
            column_losses = decoded_column_balanced_losses(
                decoded,
                target,
                occupancy_activation=occupancy_activation,
                temperature=decoded_column_temperature,
            )
            column_positive_loss = column_losses["positive_loss"]
            column_negative_loss = column_losses["negative_loss"]
            components["column_positive_loss"] = column_positive_loss
            components["column_negative_loss"] = column_negative_loss
            loss = loss + decoded_column_positive_weight * column_positive_loss
            loss = loss + decoded_column_negative_weight * column_negative_loss

    return loss, components


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


def align_ldm_grid_config(config: "ConfigManager", target_size, model_pc_range):
    """保证多模态投影网格与 dataset 输出张量使用同一物理范围。"""
    ldm_cfg = config.config.setdefault("ldm", {})
    ldm_cfg.setdefault("fusion_voxel_shape", list(target_size))
    ldm_cfg.setdefault("fusion_pc_range", list(model_pc_range))


class ConfigManager:
    """配置管理器 - 统一加载和管理配置"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path: str) -> Dict:
        """加载 YAML 配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def get(self, key: str, default=None):
        """获取配置值，支持点号分隔的嵌套访问"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value


class MemoryOptimizer:
    """显存优化器 - 统一管理显存优化策略"""
    
    def __init__(self, config: ConfigManager):
        self.use_amp = config.get('optimization.use_amp', True)
        self.use_checkpoint = config.get('optimization.use_checkpoint', True)
        self.grad_accum_steps = config.get('optimization.gradient_accumulation_steps', 1)
        device_cfg = config.get('hardware.device', 'cuda') or 'cuda'
        self.device = torch.device(device_cfg)
        
        self.scaler = GradScaler('cuda') if self.use_amp else None
    
    def clear_cache(self):
        """清理显存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_memory_stats(self) -> Dict:
        """获取显存统计"""
        if not torch.cuda.is_available():
            return {}
        return {
            'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
            'reserved_gb': torch.cuda.memory_reserved() / 1024**3,
            'peak_gb': torch.cuda.max_memory_allocated() / 1024**3,
        }
    
    def print_stats(self, prefix: str = ""):
        """打印显存统计"""
        stats = self.get_memory_stats()
        if stats:
            print(f"{prefix}GPU: {stats['allocated_gb']:.1f}GB allocated, "
                  f"{stats['reserved_gb']:.1f}GB reserved, "
                  f"{stats['peak_gb']:.1f}GB peak")


class OptimizedVAETrainer:
    """优化的 VAE 训练器"""

    METRICS_HEADER = [
        'epoch', 'loss', 'recon_loss', 'kl_loss',
        'occ_bce_loss', 'occ_dice_loss', 'continuous_loss',
        'val_iou', 'val_recall', 'val_precision',
        'lr', 'time_seconds',
    ]
    TASK2_METRICS_HEADER = [
        'epoch', 'loss', 'recon_loss', 'kl_loss',
        'occ_bce_loss', 'occ_dice_loss', 'continuous_loss',
        'lr', 'time_seconds',
    ]
    LEGACY_METRICS_HEADER = [
        'epoch', 'loss', 'recon_loss', 'kl_loss', 'lr', 'time_seconds',
    ]
    
    def __init__(
        self,
        model: nn.Module,
        config: ConfigManager,
        memory_opt: MemoryOptimizer,
        resume_path: Optional[str] = None,
        vae_model_config: Optional[Dict[str, Any]] = None,
        vae_config_type: Optional[str] = None,
    ):
        self.config = config
        self.memory_opt = memory_opt
        self.device = memory_opt.device
        
        # 将模型移到设备
        self.model = model.to(self.device)
        
        # 训练参数
        self.vae_config = config.get('vae', {}) or {}
        self.lr = self.vae_config.get('lr', 1e-4)
        self.epochs = self.vae_config.get('epochs', 100)
        self.save_dir = self.vae_config.get('save_dir', './Result/train_results/vae')
        self.vae_model_config = dict(vae_model_config or {})
        self.enable_validation_metrics = True
        self.vae_config_type = vae_config_type or self.vae_config.get(
            "config_type", "custom"
        )
        target_size, source_range, model_range = resolve_data_grid_config(
            config.get("data", {}) or {}
        )
        self.data_grid_config = {
            "target_size": list(target_size),
            "source_pc_range": list(source_range),
            "model_pc_range": list(model_range),
        }
        self.occupancy_activation = (
            "sigmoid"
            if self.vae_model_config.get("occupancy_loss_type") == "bce_dice"
            else "raw"
        )
        
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 优化器和调度器
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=1e-6
        )
        
        # 初始化训练状态
        self.start_epoch = 1
        self.best_loss = float('inf')
        self.best_iou = float('-inf')
        
        # NOTE: 日志与训练状态在恢复训练时必须共用同一目录。
        self.start_epoch = 1
        self.best_loss = float('inf')
        self.best_iou = float('-inf')
        self.is_resumed = False
        
        # 检查是否恢复训练
        if resume_path and os.path.exists(resume_path):
            self.is_resumed = True
        
        # 设置日志（恢复训练时追加，新训练时清空）
        self.log_file = os.path.join(self.save_dir, 'training.log')
        self.csv_file = os.path.join(self.save_dir, 'metrics.csv')
        self._setup_logging()
        
        # 恢复训练
        if self.is_resumed:
            self._resume_from_checkpoint(resume_path)
    
    def _setup_logging(self):
        """设置日志系统"""
        # 确定日志文件模式：恢复训练时追加，新训练时覆盖
        log_mode = 'a' if self.is_resumed else 'w'
        
        # 配置文本日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file, mode=log_mode),
                logging.StreamHandler()
            ],
            force=True  # 强制重新配置
        )
        self.logger = logging.getLogger(__name__)
        
        # 添加训练会话分隔符
        if self.is_resumed:
            self.logger.info("\n" + "="*70)
            self.logger.info("RESUMING TRAINING SESSION")
            self.logger.info("="*70)
        
        self._prepare_metrics_csv()

    def _prepare_metrics_csv(self):
        """校验恢复训练的 CSV 表头，并无损迁移旧六列指标文件。"""
        target_header = (
            self.METRICS_HEADER
            if getattr(self, "enable_validation_metrics", False)
            else self.TASK2_METRICS_HEADER
        )
        if self.is_resumed and os.path.exists(self.csv_file):
            with open(self.csv_file, newline='') as f:
                header = next(csv.reader(f), [])
            if header == target_header:
                return
            if header not in (self.LEGACY_METRICS_HEADER, self.TASK2_METRICS_HEADER):
                raise RuntimeError(f"无法识别 metrics.csv 表头: {header}")

            legacy_path = os.path.join(self.save_dir, 'metrics_legacy.csv')
            suffix = 1
            while os.path.exists(legacy_path):
                legacy_path = os.path.join(
                    self.save_dir, f'metrics_legacy_{suffix}.csv'
                )
                suffix += 1
            os.replace(self.csv_file, legacy_path)

        with open(self.csv_file, 'w', newline='') as f:
            csv.writer(f).writerow(target_header)
    
    def _resume_from_checkpoint(self, ckpt_path: str):
        """从检查点恢复训练"""
        print(f"Resuming from checkpoint: {ckpt_path}")
        ckpt = safe_torch_load(ckpt_path, map_location=self.device)
        
        # 加载模型
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(ckpt['model_state_dict'])
        else:
            self.model.load_state_dict(ckpt['model_state_dict'])
        
        # 加载优化器
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        else:
            completed_epoch = int(ckpt.get('epoch', 0))
            self.scheduler.last_epoch = completed_epoch
            self.scheduler._step_count = completed_epoch + 1
            self.scheduler._last_lr = [
                group["lr"] for group in self.optimizer.param_groups
            ]
        
        # 加载训练状态
        self.start_epoch = ckpt.get('epoch', 0) + 1
        self.best_loss = ckpt.get('best_loss', ckpt.get('loss', float('inf')))
        self.best_iou = ckpt.get('best_iou', float('-inf'))
        
        print(
            f"Resumed from epoch {self.start_epoch - 1}, "
            f"best loss: {self.best_loss:.4f}, best IoU: {self.best_iou:.4f}"
        )
    
    def _log_metrics(
        self,
        epoch: int,
        loss: float,
        recon_loss: float,
        kl_loss: float,
        epoch_time: float,
        val_metrics: Dict[str, float],
    ):
        """记录指标到 CSV 文件"""
        components = getattr(self, "last_epoch_loss_components", {})
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f'{loss:.6f}',
                f'{recon_loss:.6f}',
                f'{kl_loss:.8f}',
                f'{components.get("occ_bce_loss", 0.0):.6f}',
                f'{components.get("occ_dice_loss", 0.0):.6f}',
                f'{components.get("continuous_loss", 0.0):.6f}',
                f'{val_metrics["iou"]:.6f}',
                f'{val_metrics["recall"]:.6f}',
                f'{val_metrics["precision"]:.6f}',
                f'{self.optimizer.param_groups[0]["lr"]:.8f}',
                f'{epoch_time:.2f}'
            ])
    
    def train_epoch(self, epoch: int, train_loader: DataLoader) -> tuple:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        total_components = {
            "occ_bce_loss": 0.0,
            "occ_dice_loss": 0.0,
            "continuous_loss": 0.0,
        }
        valid_batch_count = 0
        accumulation_count = 0
        
        # 创建进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            target, cond, meta_dict = unpack_training_batch(batch)
            target = target.to(self.device, non_blocking=True)
            
            if not torch.isfinite(target).all():
                print(f"Warning: Batch {batch_idx} target contains NaN or Inf")
                continue
            
            # NOTE: 前向阶段使用 autocast，配合梯度累积降低显存峰值。
            with autocast('cuda', enabled=self.memory_opt.use_amp):
                recon, (mean, logvar) = self.model(target)
                loss, recon_loss, kl_loss = self.model.compute_loss(
                    target, recon, (mean, logvar)
                )

            losses = (loss, recon_loss, kl_loss)
            if not all(torch.isfinite(value).all() for value in losses):
                print(f"Warning: Batch {batch_idx} loss contains NaN or Inf")
                continue

            scaled_loss = loss / self.memory_opt.grad_accum_steps
            
            # 反向传播
            if self.memory_opt.scaler:
                self.memory_opt.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            accumulation_count += 1
            
            # NOTE: 非有限 batch 不计入累计边界，避免污染或提前提交已有梯度。
            if accumulation_count == self.memory_opt.grad_accum_steps:
                if self.memory_opt.scaler:
                    self.memory_opt.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.memory_opt.scaler.step(self.optimizer)
                    self.memory_opt.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                accumulation_count = 0
            
            # 累积损失
            batch_loss = loss.item()
            total_loss += batch_loss
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            valid_batch_count += 1
            component_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            for name, value in component_model.loss_components.items():
                if name in total_components:
                    total_components[name] += value.item()
            
            # 更新进度条显示
            pbar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'kl': f'{kl_loss.item():.6f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # 定期清理显存
            if batch_idx % 50 == 0:
                self.memory_opt.clear_cache()

        if valid_batch_count == 0:
            self.optimizer.zero_grad()
            raise RuntimeError("当前 epoch 没有有限的有效 batch")

        # NOTE: 仅按有效 batch 的实际累计余数提交尾部梯度。
        if accumulation_count:
            tail_gradient_scale = (
                self.memory_opt.grad_accum_steps / accumulation_count
            )
            if self.memory_opt.scaler:
                self.memory_opt.scaler.unscale_(self.optimizer)
            # NOTE: 每批 loss 按完整累计步数缩放，尾部不足时需恢复为有效 batch 均值梯度。
            for parameter in self.model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(tail_gradient_scale)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            if self.memory_opt.scaler:
                self.memory_opt.scaler.step(self.optimizer)
                self.memory_opt.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()
        
        self.scheduler.step()
        self.last_epoch_valid_batch_count = valid_batch_count
        self.last_epoch_loss_components = {
            name: value / valid_batch_count for name, value in total_components.items()
        }
        return (
            total_loss / valid_batch_count,
            total_recon / valid_batch_count,
            total_kl / valid_batch_count,
        )

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """使用确定性编码在完整验证集上计算 threshold=0.5 微平均指标。"""
        self.model.eval()
        counts = {
            "intersection": 0,
            "union": 0,
            "target_positive": 0,
            "predicted_positive": 0,
        }
        batch_count = 0
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        for batch in val_loader:
            target, _condition, _meta = unpack_training_batch(batch)
            target = target.to(self.device, non_blocking=True)
            latent, _posterior = model.encode(target, deterministic=True)
            reconstruction = model.decode(latent)
            probability = (
                model.occupancy_probability(reconstruction[:, 0:1])
                if self.occupancy_activation == "sigmoid"
                else reconstruction[:, 0:1]
            )
            batch_counts = micro_occupancy_metrics(
                probability, target[:, 0:1], threshold=0.5
            )
            for key, value in batch_counts.items():
                counts[key] += value
            batch_count += 1
        if batch_count == 0:
            raise RuntimeError("验证 DataLoader 为空，无法计算 occupancy 指标")
        return {
            "iou": counts["intersection"] / max(counts["union"], 1),
            "recall": counts["intersection"] / max(counts["target_positive"], 1),
            "precision": counts["intersection"] / max(counts["predicted_positive"], 1),
        }

    def _checkpoint_payload(
        self,
        epoch: int,
        loss: float,
        best_loss: float,
        best_iou: float,
    ) -> Dict[str, Any]:
        """构造训练、诊断和推理均可直接消费的自描述 checkpoint。"""
        if not self.vae_model_config:
            raise RuntimeError("保存 VAE checkpoint 前必须提供完整 vae_model_config")
        state_dict = (
            self.model.module.state_dict()
            if isinstance(self.model, nn.DataParallel)
            else self.model.state_dict()
        )
        return {
            "epoch": epoch,
            "model_state_dict": state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "loss": loss,
            "best_loss": best_loss,
            "best_iou": best_iou,
            "vae_config": dict(self.vae_model_config),
            "vae_config_type": self.vae_config_type,
            "data_grid_config": dict(self.data_grid_config),
            "occupancy_activation": self.occupancy_activation,
        }

    def _update_best_metrics(self, loss: float, val_iou: float):
        """先原子更新本 epoch 的全局最佳状态，再允许构建 checkpoint。"""
        improved_loss = loss < self.best_loss
        improved_iou = val_iou > self.best_iou
        if improved_loss:
            self.best_loss = loss
        if improved_iou:
            self.best_iou = val_iou
        return improved_loss, improved_iou
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """完整训练流程"""
        estimated_total_steps = self.epochs * len(train_loader)
        
        msg = "=" * 70 + "\n"
        msg += f"Starting VAE Training\n"
        msg += f"  Total epochs: {self.epochs}\n"
        msg += f"  Batches per epoch: {len(train_loader)}\n"
        msg += f"  Estimated total steps: {estimated_total_steps:,}\n"
        msg += f"  Start epoch: {self.start_epoch}\n"
        msg += f"  Batch size: {train_loader.batch_size}\n"
        msg += f"  Gradient accumulation: {self.memory_opt.grad_accum_steps}\n"
        msg += f"  Effective batch size: {train_loader.batch_size * self.memory_opt.grad_accum_steps}\n"
        msg += f"  Learning rate: {self.lr}\n"
        msg += f"  Device: {self.device}\n"
        msg += f"  Save directory: {self.save_dir}\n"
        msg += f"  Log file: {self.log_file}\n"
        msg += f"  CSV file: {self.csv_file}\n"
        msg += "=" * 70
        print(msg)
        self.logger.info(msg)
        
        start_time = time.time()
        
        for epoch in range(self.start_epoch, self.epochs + 1):
            epoch_start = time.time()
            loss, recon_loss, kl_loss = self.train_epoch(epoch, train_loader)
            val_metrics = self.validate(val_loader)
            epoch_time = time.time() - epoch_start
            
            # 记录到 CSV
            self._log_metrics(
                epoch, loss, recon_loss, kl_loss, epoch_time, val_metrics
            )
            
            # 打印和记录 epoch 总结
            summary = (f"\n[Epoch {epoch}/{self.epochs}] "
                      f"Loss: {loss:.4f} | Recon: {recon_loss:.4f} | KL: {kl_loss:.6f} | "
                      f"Val IoU: {val_metrics['iou']:.4f} | "
                      f"Recall: {val_metrics['recall']:.4f} | "
                      f"Precision: {val_metrics['precision']:.4f} | "
                      f"Time: {epoch_time:.1f}s")
            print(summary)
            self.logger.info(summary)
            
            # 显存统计
            self.memory_opt.print_stats(prefix="  ")
            
            improved_loss, improved_iou = self._update_best_metrics(
                loss, val_metrics["iou"]
            )
            checkpoint_payload = self._checkpoint_payload(
                epoch, loss, self.best_loss, self.best_iou
            )

            # NOTE: 本 epoch 的所有保存路径共享更新完两个 best 后的同一 payload。
            if improved_loss:
                best_ckpt = os.path.join(self.save_dir, "vae_best_loss.pt")
                atomic_torch_save(checkpoint_payload, best_ckpt)
                msg = f"  ✓ Saved best model (loss: {loss:.4f})"
                print(msg)
                self.logger.info(msg)

            if improved_iou:
                best_iou_path = os.path.join(self.save_dir, "vae_best_iou.pt")
                atomic_torch_save(checkpoint_payload, best_iou_path)
                # NOTE: 兼容历史路径，内容始终与 best-IoU checkpoint 一致。
                atomic_copy_file(
                    best_iou_path,
                    os.path.join(self.save_dir, "vae_best.pt"),
                )
                msg = f"  ✓ Saved best-IoU model (IoU: {self.best_iou:.4f})"
                print(msg)
                self.logger.info(msg)
            
            # 定期保存
            if epoch % self.vae_config.get('save_every', 10) == 0:
                ckpt_path = os.path.join(self.save_dir, f"vae_epoch{epoch:04d}.pt")
                atomic_torch_save(checkpoint_payload, ckpt_path)
                msg = f"  ✓ Saved checkpoint: {ckpt_path}"
                print(msg)
                self.logger.info(msg)
        
        total_time = time.time() - start_time
        final_msg = "\n" + "=" * 70 + "\n"
        final_msg += f"Training completed in {total_time/3600:.2f} hours\n"
        final_msg += f"Best loss: {self.best_loss:.4f}\n"
        final_msg += "=" * 70
        print(final_msg)
        self.logger.info(final_msg)


class OptimizedLDMTrainer:
    """优化的 Latent Diffusion 训练器"""
    
    def __init__(
        self,
        vae: nn.Module,
        config: ConfigManager,
        memory_opt: MemoryOptimizer,
        resume_path: Optional[str] = None,
    ):
        self.vae = vae.to(memory_opt.device)
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
        
        self.config = config
        self.memory_opt = memory_opt
        self.device = memory_opt.device
        
        ldm_config: Dict[str, Any] = config.get('ldm', {}) or {}
        self.ldm_config = ldm_config
        self.latent_dim = int(self.vae.latent_dim)
        self.model_config = {
            "latent_dim": self.latent_dim,
            "in_channels": max(16, 2 * self.latent_dim),
            "out_channels": self.latent_dim,
            "model_channels": ldm_config.get('model_channels', 32),
            "num_res_blocks": ldm_config.get('num_res_blocks', 1),
            "attention_resolutions": list(ldm_config.get('attention_resolutions', [])),
            "channel_mult": list(ldm_config.get('channel_mult', [1, 2, 3])),
        }
        self.save_dir = ldm_config.get('save_dir', './Result/train_results/ldm')
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 创建潜空间去噪骨干，再挂到雷达-红外多模态入口。
        base_unet = OptimizedUNetModel(
            image_size=32,  # 潜空间 H/W
            in_channels=self.model_config["in_channels"],
            model_channels=ldm_config.get('model_channels', 32),
            out_channels=self.latent_dim,
            num_res_blocks=ldm_config.get('num_res_blocks', 1),
            attention_resolutions=tuple(ldm_config.get('attention_resolutions', [])),
            channel_mult=tuple(ldm_config.get('channel_mult', [1, 2, 3])),
            use_checkpoint=True,
            attention_type="linear",
        )
        fusion_voxel_shape = tuple(int(v) for v in ldm_config.get('fusion_voxel_shape', [32, 128, 128]))
        fusion_latent_shape = tuple(int(v) for v in ldm_config.get('fusion_latent_shape', fusion_voxel_shape))
        fusion_pc_range = tuple(float(v) for v in ldm_config.get('fusion_pc_range', [0, -20, -6, 120, 20, 10]))
        self.model = CompleteDualModalityPerceptionNet(
            base_unet,
            voxel_shape=fusion_voxel_shape,
            pc_range=fusion_pc_range,
            downsample_to_latent=True,
            latent_shape=fusion_latent_shape,
        ).to(self.device)
        
        # 优化器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=ldm_config.get('lr', 1e-4),
            weight_decay=1e-4,
        )
        
        # Denoiser
        self.denoiser = KarrasDenoiser(
            sigma_data=ldm_config.get('sigma_data', 0.5),
            sigma_max=ldm_config.get('sigma_max', 80.0),
            sigma_min=ldm_config.get('sigma_min', 0.002),
            loss_norm='l2',
        )
        self.decoded_loss_weight = float(ldm_config.get('decoded_loss_weight', 0.0))
        self.decoded_false_positive_weight = float(ldm_config.get('decoded_false_positive_weight', 0.0))
        self.decoded_mass_weight = float(ldm_config.get('decoded_mass_weight', 0.0))
        self.decoded_height_distribution_weight = float(ldm_config.get('decoded_height_distribution_weight', 0.0))
        self.decoded_top_height_weight = float(ldm_config.get('decoded_top_height_weight', 0.0))
        self.decoded_top_overshoot_weight = float(ldm_config.get('decoded_top_overshoot_weight', 0.0))
        self.decoded_vertical_continuity_weight = float(ldm_config.get('decoded_vertical_continuity_weight', 0.0))
        self.decoded_density_weight = float(ldm_config.get('decoded_density_weight', 0.0))
        self.decoded_ir_frustum_occupancy_weight = float(ldm_config.get('decoded_ir_frustum_occupancy_weight', 0.0))
        self.decoded_ir_frustum_negative_weight = float(ldm_config.get('decoded_ir_frustum_negative_weight', 0.0))
        self.decoded_ir_frustum_top_weight = float(ldm_config.get('decoded_ir_frustum_top_weight', 0.0))
        self.decoded_column_positive_weight = float(ldm_config.get('decoded_column_positive_weight', 0.0))
        self.decoded_column_negative_weight = float(ldm_config.get('decoded_column_negative_weight', 0.0))
        self.decoded_column_temperature = float(ldm_config.get('decoded_column_temperature', 1.0))
        for name, weight in (
            ("decoded_column_positive_weight", self.decoded_column_positive_weight),
            ("decoded_column_negative_weight", self.decoded_column_negative_weight),
        ):
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError(f"{name} 必须是有限非负数，实际为 {weight!r}")
        if (
            not math.isfinite(self.decoded_column_temperature)
            or not 1e-3 <= self.decoded_column_temperature <= 100.0
        ):
            raise ValueError(
                "decoded_column_temperature 必须是 [1e-3,100.0] 内的有限数，"
                f"实际为 {self.decoded_column_temperature!r}"
            )
        self.occupancy_activation = getattr(
            self.vae,
            "occupancy_activation",
            (
                "sigmoid"
                if getattr(self.vae, "occupancy_loss_type", "legacy_mse") == "bce_dice"
                else "raw"
            ),
        )
        self.uncertainty_loss_weight = float(ldm_config.get('uncertainty_loss_weight', 0.0))
        
        # 初始化训练状态
        self.start_epoch = 1
        self.global_step = 0
        self.best_loss = float('inf')
        self.is_resumed = False
        
        # 检查是否恢复训练
        if resume_path and os.path.exists(resume_path):
            self.is_resumed = True
        
        # 设置日志（恢复训练时追加，新训练时清空）
        self.log_file = os.path.join(self.save_dir, 'training.log')
        self.csv_file = os.path.join(self.save_dir, 'metrics.csv')
        self._setup_logging()
        
        # 恢复训练
        if self.is_resumed:
            self._resume_from_checkpoint(resume_path)

    def _ldm_loss_config(self) -> Dict[str, float]:
        """返回当前生效的 LDM 损失权重，便于 checkpoint 自描述。"""
        return {
            "decoded_loss_weight": self.decoded_loss_weight,
            "decoded_false_positive_weight": self.decoded_false_positive_weight,
            "decoded_mass_weight": self.decoded_mass_weight,
            "decoded_height_distribution_weight": self.decoded_height_distribution_weight,
            "decoded_top_height_weight": self.decoded_top_height_weight,
            "decoded_top_overshoot_weight": self.decoded_top_overshoot_weight,
            "decoded_vertical_continuity_weight": self.decoded_vertical_continuity_weight,
            "decoded_density_weight": self.decoded_density_weight,
            "decoded_ir_frustum_occupancy_weight": self.decoded_ir_frustum_occupancy_weight,
            "decoded_ir_frustum_negative_weight": self.decoded_ir_frustum_negative_weight,
            "decoded_ir_frustum_top_weight": self.decoded_ir_frustum_top_weight,
            "decoded_column_positive_weight": self.decoded_column_positive_weight,
            "decoded_column_negative_weight": self.decoded_column_negative_weight,
            "decoded_column_temperature": self.decoded_column_temperature,
            "uncertainty_loss_weight": self.uncertainty_loss_weight,
        }
    
    def _setup_logging(self):
        """设置日志系统"""
        # 确定日志文件模式：恢复训练时追加，新训练时覆盖
        log_mode = 'a' if self.is_resumed else 'w'
        
        # 配置文本日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file, mode=log_mode),
                logging.StreamHandler()
            ],
            force=True  # 强制重新配置
        )
        self.logger = logging.getLogger(__name__ + '_ldm')
        
        # 添加训练会话分隔符
        if self.is_resumed:
            self.logger.info("\n" + "="*70)
            self.logger.info("RESUMING TRAINING SESSION")
            self.logger.info("="*70)
        
        # 初始化 CSV 文件；恢复旧 5 列日志时归档，避免追加新行宽。
        prepare_ldm_metrics_csv(self.csv_file, self.is_resumed)
    
    def _resume_from_checkpoint(self, ckpt_path: str):
        """从检查点恢复训练"""
        print(f"Resuming LDM from checkpoint: {ckpt_path}")
        ckpt = safe_torch_load(ckpt_path, map_location=self.device)
        
        self.model.load_state_dict(migrate_ir_gate_state_dict(self.model, ckpt['model_state_dict']))
        if 'optimizer_state_dict' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        
        self.start_epoch = ckpt.get('epoch', 0) + 1
        self.global_step = ckpt.get('step', 0)
        self.best_loss = ckpt.get('best_loss', ckpt.get('loss', float('inf')))
        
        print(f"Resumed from epoch {self.start_epoch - 1}, step {self.global_step}, best loss: {self.best_loss:.4f}")
    
    def _log_metrics(self, epoch: int, step: int, loss: float, epoch_time: float):
        """记录指标到 CSV 文件"""
        components = getattr(self, "last_epoch_loss_components", {})
        meta_components = getattr(self, "last_epoch_meta_components", {})
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                step,
                f'{loss:.6f}',
                *[
                    f'{float(components.get(name, 0.0)):.6f}'
                    for name in LDM_LOSS_COMPONENT_NAMES
                ],
                *[
                    f'{float(meta_components.get(name, 0.0)):.6f}'
                    for name in LDM_META_COMPONENT_NAMES
                ],
                f'{self.optimizer.param_groups[0]["lr"]:.8f}',
                f'{epoch_time:.2f}'
            ])
    
    def train_epoch(self, epoch: int, train_loader: DataLoader) -> float:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        total_components = {name: 0.0 for name in LDM_LOSS_COMPONENT_NAMES}
        total_meta = {name: 0.0 for name in LDM_META_COMPONENT_NAMES}
        accumulation_count = 0
        
        # 创建进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            target, cond, meta_dict = unpack_training_batch(batch)
            target = target.to(self.device)
            cond = cond.to(self.device)
            meta_dict = move_meta_to_device(meta_dict, self.device)
            if "is_mock_ir" in meta_dict:
                total_meta["mock_ir_ratio"] += float(torch.as_tensor(meta_dict["is_mock_ir"]).float().mean().item())
            if "is_mock_calib" in meta_dict:
                total_meta["mock_calib_ratio"] += float(torch.as_tensor(meta_dict["is_mock_calib"]).float().mean().item())
            
            # 编码到潜空间
            with torch.no_grad():
                z_target = self.vae.get_latent(target)
                z_cond = self.vae.get_latent(cond)
            
            # NOTE: 按 Karras 噪声范围采样 sigma，覆盖高噪和低噪训练区间。
            batch_size = z_target.shape[0]
            # NOTE: EDM/Karras 训练常用 log-uniform sigma 采样；同一个随机数决定
            # NOTE: sigma_min 到 sigma_max 的插值位置，避免噪声分布被独立随机数扭曲。
            u = torch.rand(batch_size, device=self.device)
            sigmas = self.denoiser.sigma_max * (
                self.denoiser.sigma_min / self.denoiser.sigma_max
            ) ** u
            
            # 生成噪声并加噪
            noise = torch.randn_like(z_target)
            noised_z = z_target + noise * sigmas.view(-1, 1, 1, 1, 1)
            
            # 前向传播
            with autocast('cuda', enabled=self.memory_opt.use_amp):
                uncertainty = None
                if has_multimodal_meta(meta_dict):
                    model_out = self.model(
                        cond,
                        meta_dict["ir_img"],
                        meta_dict["r_mat"],
                        meta_dict["t_vec"],
                        meta_dict["k_mat"],
                        sigmas,
                        noised_latent=noised_z,
                        odom_cov_trace=meta_dict.get("odom_cov_trace"),
                        is_mock_ir=meta_dict.get("is_mock_ir"),
                        is_mock_calib=meta_dict.get("is_mock_calib"),
                        return_uncertainty=self.uncertainty_loss_weight > 0.0,
                    )
                    if isinstance(model_out, tuple):
                        denoised, uncertainty = model_out
                    else:
                        denoised, uncertainty = model_out, None
                    ir_frustum_mask = getattr(self.model, "last_ir_frustum_mask", None)
                    if ir_frustum_mask is not None:
                        total_meta["ir_frustum_voxel_ratio"] += float(ir_frustum_mask.float().mean().item())
                else:
                    # Legacy batches without IR metadata still train the same 16-channel UNet backbone.
                    model_input = pad_ldm_input_to_sixteen_channels(torch.cat([noised_z, z_cond], dim=1))
                    denoised = self.model.unet_3d(model_input, sigmas)
                    ir_frustum_mask = None
                loss, loss_components = compute_ldm_loss_components(
                    denoised,
                    z_target,
                    target,
                    vae=self.vae,
                    occupancy_activation=self.occupancy_activation,
                    decoded_loss_weight=self.decoded_loss_weight,
                    decoded_false_positive_weight=self.decoded_false_positive_weight,
                    decoded_mass_weight=self.decoded_mass_weight,
                    decoded_height_distribution_weight=self.decoded_height_distribution_weight,
                    decoded_top_height_weight=self.decoded_top_height_weight,
                    decoded_top_overshoot_weight=self.decoded_top_overshoot_weight,
                    decoded_vertical_continuity_weight=self.decoded_vertical_continuity_weight,
                    decoded_density_weight=self.decoded_density_weight,
                    decoded_ir_frustum_occupancy_weight=self.decoded_ir_frustum_occupancy_weight,
                    decoded_ir_frustum_negative_weight=self.decoded_ir_frustum_negative_weight,
                    decoded_ir_frustum_top_weight=self.decoded_ir_frustum_top_weight,
                    ir_frustum_mask=ir_frustum_mask,
                    uncertainty_loss_weight=self.uncertainty_loss_weight,
                    uncertainty=uncertainty,
                    decoded_column_positive_weight=self.decoded_column_positive_weight,
                    decoded_column_negative_weight=self.decoded_column_negative_weight,
                    decoded_column_temperature=self.decoded_column_temperature,
                )
                scaled_loss = loss / self.memory_opt.grad_accum_steps
            
            # 反向传播
            if self.memory_opt.scaler:
                self.memory_opt.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            accumulation_count += 1
            
            # 梯度更新
            if accumulation_count == self.memory_opt.grad_accum_steps:
                if self.memory_opt.scaler:
                    self.memory_opt.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.memory_opt.scaler.step(self.optimizer)
                    self.memory_opt.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                accumulation_count = 0
            
            batch_loss = loss.item()
            total_loss += batch_loss
            for name in LDM_LOSS_COMPONENT_NAMES:
                total_components[name] += float(loss_components[name].detach().item())
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'latent': f'{loss_components["latent_loss"].detach().item():.4f}',
                'height': f'{loss_components["height_distribution_loss"].detach().item():.4f}',
                'top': f'{loss_components["top_height_loss"].detach().item():.4f}',
                'ir_top': f'{loss_components["ir_frustum_top_height_loss"].detach().item():.4f}',
                'ir_neg': f'{loss_components["ir_frustum_negative_loss"].detach().item():.4f}',
                'cont': f'{loss_components["vertical_continuity_loss"].detach().item():.4f}',
                'dens': f'{loss_components["decoded_density_loss"].detach().item():.4f}',
                'col_pos': f'{loss_components["column_positive_loss"].detach().item():.4f}',
                'col_neg': f'{loss_components["column_negative_loss"].detach().item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # NOTE: 当 batch 数不能被梯度累积步数整除时，补上最后一次未提交的更新。
        if accumulation_count:
            if self.memory_opt.scaler:
                self.memory_opt.scaler.unscale_(self.optimizer)
            rescale_accumulated_gradients(
                self.model.parameters(),
                grad_accum_steps=self.memory_opt.grad_accum_steps,
                accumulation_count=accumulation_count,
            )
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            if self.memory_opt.scaler:
                self.memory_opt.scaler.step(self.optimizer)
                self.memory_opt.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()

        self.last_epoch_loss_components = {
            name: total_components[name] / len(train_loader)
            for name in LDM_LOSS_COMPONENT_NAMES
        }
        self.last_epoch_meta_components = {
            name: total_meta[name] / len(train_loader)
            for name in LDM_META_COMPONENT_NAMES
        }
        if self.last_epoch_meta_components["mock_ir_ratio"] > 0.5:
            self.logger.warning(
                "LDM epoch %s mock IR ratio is %.3f; 当前结果不能作为真实红外融合收益。",
                epoch,
                self.last_epoch_meta_components["mock_ir_ratio"],
            )
        if self.last_epoch_meta_components["mock_calib_ratio"] > 0.5:
            self.logger.warning(
                "LDM epoch %s mock calib ratio is %.3f; IR 投影几何可信度较低。",
                epoch,
                self.last_epoch_meta_components["mock_calib_ratio"],
            )
        return total_loss / len(train_loader)
    
    def train(self, train_loader: DataLoader):
        """完整训练流程"""
        epochs = self.ldm_config.get('epochs', 200)
        save_every = self.ldm_config.get('save_every', 5000)
        
        # 计算预估总步数
        estimated_total_steps = epochs * len(train_loader)
        
        msg = "=" * 70 + "\n"
        msg += f"Starting LDM Training\n"
        msg += f"  Total epochs: {epochs}\n"
        msg += f"  Batches per epoch: {len(train_loader)}\n"
        msg += f"  Estimated total steps: {estimated_total_steps:,}\n"
        msg += f"  Start epoch: {self.start_epoch}\n"
        msg += f"  Batch size: {train_loader.batch_size}\n"
        msg += f"  Gradient accumulation: {self.memory_opt.grad_accum_steps}\n"
        msg += f"  Effective batch size: {train_loader.batch_size * self.memory_opt.grad_accum_steps}\n"
        msg += f"  Save directory: {self.save_dir}\n"
        msg += f"  Log file: {self.log_file}\n"
        msg += f"  CSV file: {self.csv_file}\n"
        msg += "=" * 70
        print(msg)
        self.logger.info(msg)
        
        start_time = time.time()
        
        for epoch in range(self.start_epoch, epochs + 1):
            epoch_start = time.time()
            loss = self.train_epoch(epoch, train_loader)
            epoch_time = time.time() - epoch_start
            self.global_step += len(train_loader)
            
            # 记录到 CSV
            self._log_metrics(epoch, self.global_step, loss, epoch_time)
            
            summary = f"\n[Epoch {epoch}/{epochs}] Loss: {loss:.4f} | Step: {self.global_step} | Time: {epoch_time:.1f}s"
            print(summary)
            self.logger.info(summary)
            self.memory_opt.print_stats(prefix="  ")
            
            # 保存最佳模型
            if loss < self.best_loss:
                self.best_loss = loss
                best_ckpt = os.path.join(self.save_dir, "ldm_best.pt")
                torch.save({
                    'epoch': epoch,
                    'step': self.global_step,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss,
                    'best_loss': self.best_loss,
                    'latent_dim': self.latent_dim,
                    'model_config': dict(self.model_config),
                    'ldm_loss_config': self._ldm_loss_config(),
                }, best_ckpt)
                msg = f"  ✓ Saved best model (loss: {loss:.4f})"
                print(msg)
                self.logger.info(msg)
            
            # 定期按epoch保存
            if epoch % save_every == 0:
                ckpt_path = os.path.join(self.save_dir, f"ldm_epoch{epoch:04d}.pt")
                torch.save({
                    'epoch': epoch,
                    'step': self.global_step,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss,
                    'best_loss': self.best_loss,
                    'latent_dim': self.latent_dim,
                    'model_config': dict(self.model_config),
                    'ldm_loss_config': self._ldm_loss_config(),
                }, ckpt_path)
                msg = f"  ✓ Saved checkpoint: {ckpt_path}"
                print(msg)
                self.logger.info(msg)
        
        total_time = time.time() - start_time
        final_msg = "\n" + "=" * 70 + "\n"
        final_msg += f"Training completed in {total_time/3600:.2f} hours\n"
        final_msg += f"Best loss: {self.best_loss:.4f}\n"
        final_msg += "=" * 70
        print(final_msg)
        self.logger.info(final_msg)


def main():
    parser = argparse.ArgumentParser(description="Unified Training Script")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["vae", "ldm", "cd"],
                        help="Training mode: vae, ldm, or cd")
    parser.add_argument("--config", type=str, default="./diffusion_consistency_radar/config/default_config.yaml",
                        help="Config file path")
    parser.add_argument("--vae_ckpt", type=str, default="",
                        help="VAE checkpoint path (for LDM/CD training)")
    parser.add_argument("--ldm_ckpt", type=str, default="",
                        help="LDM checkpoint path (for CD training)")
    parser.add_argument("--resume", type=str, default="",
                        help="Resume training from checkpoint")
    
    args = parser.parse_args()
    
    # 加载配置
    config = ConfigManager(args.config)
    data_config = config.get('data', {})
    training_seed = int(data_config.get("training_seed", data_config.get("split_seed", 42)))
    train_loader_generator = seed_training_run(training_seed)
    memory_opt = MemoryOptimizer(config)

    # 创建数据加载器
    target_size, source_pc_range, model_pc_range = resolve_data_grid_config(data_config)
    align_ldm_grid_config(config, target_size, model_pc_range)
    train_dataset_base = NTU4DRadLM_VoxelDataset(
        root_dir=data_config.get('dataset_dir'),
        split='train',
        use_augmentation=data_config.get('use_augmentation', True),
        target_size=target_size,
        source_pc_range=source_pc_range,
        model_pc_range=model_pc_range,
    )
    val_dataset_base = NTU4DRadLM_VoxelDataset(
        root_dir=data_config.get('dataset_dir'),
        split='train',
        use_augmentation=False,
        target_size=target_size,
        source_pc_range=source_pc_range,
        model_pc_range=model_pc_range,
    )
    if len(train_dataset_base) != len(val_dataset_base):
        raise RuntimeError("训练/验证 dataset 样本索引不一致，无法安全划分")
    train_indices, val_indices = deterministic_split_indices(
        len(train_dataset_base),
        train_split=float(data_config.get("train_split", 0.8)),
        split_seed=int(data_config.get("split_seed", 42)),
    )
    train_dataset = Subset(train_dataset_base, train_indices)
    val_dataset = Subset(val_dataset_base, val_indices)
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config.get('batch_size', 2),
        shuffle=True,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=False,
        generator=train_loader_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config.get('batch_size', 2),
        shuffle=False,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=False,
    )
    
    # VAE 训练
    if args.mode == "vae":
        vae_type = config.get('vae.config_type', 'ultra_lightweight')
        if args.resume:
            resume_checkpoint = safe_torch_load(args.resume, map_location="cpu")
            vae, resume_metadata = build_vae_from_checkpoint(
                resume_checkpoint,
                fallback_config_type=vae_type,
            )
            vae_config = resume_metadata["vae_config"]
            vae_type = resume_metadata["vae_config_type"]
        else:
            vae_config = create_vae_config(vae_type)
            vae_config = apply_vae_config_overrides(vae_config, config)
            vae = VAE3D(**vae_config)
        trainer = OptimizedVAETrainer(
            vae,
            config,
            memory_opt,
            resume_path=args.resume,
            vae_model_config=vae_config,
            vae_config_type=vae_type,
        )
        trainer.train(train_loader, val_loader)
    
    # LDM 训练
    elif args.mode == "ldm":
        if not args.vae_ckpt:
            raise ValueError("Must provide --vae_ckpt for LDM, resume_path=args.resume training")
        
        vae_type = config.get('vae.config_type', 'ultra_lightweight')
        ckpt = safe_torch_load(args.vae_ckpt, map_location='cpu')
        vae, _vae_metadata = build_vae_from_checkpoint(
            ckpt, fallback_config_type=vae_type
        )
        
        trainer = OptimizedLDMTrainer(vae, config, memory_opt, resume_path=args.resume)
        trainer.train(train_loader)

    # CD 训练
    elif args.mode == "cd":
        if not args.vae_ckpt:
            raise ValueError("Must provide --vae_ckpt for CD training")
        ldm_ckpt = resolve_cd_teacher_checkpoint(args.ldm_ckpt, config)
        if not ldm_ckpt:
            raise ValueError("Must provide --ldm_ckpt or set cd.teacher_model_path for CD training")

        vae_type = config.get('vae.config_type', 'ultra_lightweight')
        ckpt = safe_torch_load(args.vae_ckpt, map_location='cpu')
        vae, _vae_metadata = build_vae_from_checkpoint(
            ckpt, fallback_config_type=vae_type
        )

        cd_cfg = config.get('cd', {}) or {}
        opt_cfg = config.get('optimization', {}) or {}
        trainer = ConsistencyDistillationTrainer(
            ldm_ckpt_path=ldm_ckpt,
            vae=vae,
            device=memory_opt.device,
            config={
                'lr': cd_cfg.get('lr', 5e-5),
                'save_dir': cd_cfg.get('save_dir', './Result/train_results/cd'),
                'resume_path': args.resume or None,
                'ldm': config.get('ldm', {}) or {},
            },
        )
        trainer.train(
            train_loader,
            num_epochs=cd_cfg.get('epochs', 100),
            save_every=cd_cfg.get('save_every', 10),
            grad_accum_steps=opt_cfg.get('gradient_accumulation_steps', 8),
        )
    
    print("Training completed!")


if __name__ == "__main__":
    main()
