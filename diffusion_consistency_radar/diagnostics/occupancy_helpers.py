#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""提供独立 occupancy 诊断所需的体素变换，不加载模型或正式评估入口。"""

from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


EPS = 1e-6


def load_sparse_voxel(path: str) -> np.ndarray:
    """将诊断输入中的稀疏 `.npz` 体素恢复为稠密网格。"""
    data = np.load(path)
    voxel = np.zeros(data["shape"], dtype=np.float32)
    coords = data["coords"]
    if coords.shape[0] > 0:
        voxel[coords[:, 0], coords[:, 1], coords[:, 2]] = data["features"]
    return voxel


def crop_voxel_channels_to_pc_range(
    voxel_tensor: torch.Tensor,
    source_pc_range: Sequence[float],
    model_pc_range: Sequence[float],
) -> torch.Tensor:
    """按物理 XYZ 范围裁剪 `(C,Z,X,Y)` 体素，不依赖训练数据集模块。"""
    if voxel_tensor.ndim != 4:
        raise ValueError(f"Expected (C,Z,X,Y), got {tuple(voxel_tensor.shape)}")
    source = tuple(float(value) for value in source_pc_range)
    model = tuple(float(value) for value in model_pc_range)
    if len(source) != 6 or len(model) != 6:
        raise ValueError("source_pc_range and model_pc_range must contain 6 values")
    for axis in range(3):
        if model[axis] < source[axis] or model[axis + 3] > source[axis + 3]:
            raise ValueError(
                f"model_pc_range must lie inside source_pc_range: {model} vs {source}"
            )
        if model[axis] >= model[axis + 3]:
            raise ValueError(f"Invalid model_pc_range: {model}")

    def physical_slice(
        size: int,
        low: float,
        high: float,
        crop_low: float,
        crop_high: float,
    ) -> slice:
        step = (high - low) / float(size)
        centers = low + (torch.arange(size, device=voxel_tensor.device) + 0.5) * step
        indices = torch.where((centers >= crop_low) & (centers < crop_high))[0]
        if indices.numel() == 0:
            raise ValueError(
                f"Physical crop [{crop_low}, {crop_high}) contains no voxel centers"
            )
        return slice(int(indices[0]), int(indices[-1]) + 1)

    z_slice = physical_slice(
        voxel_tensor.shape[1], source[2], source[5], model[2], model[5]
    )
    x_slice = physical_slice(
        voxel_tensor.shape[2], source[0], source[3], model[0], model[3]
    )
    y_slice = physical_slice(
        voxel_tensor.shape[3], source[1], source[4], model[1], model[4]
    )
    return voxel_tensor[:, z_slice, x_slice, y_slice]


def resize_voxel_channels(
    voxel_tensor: torch.Tensor,
    target_size: Sequence[int],
    mask_channel: Optional[int] = None,
) -> torch.Tensor:
    """复现 occupancy 诊断所需的通道感知重采样协议。"""
    if voxel_tensor.ndim != 4:
        raise ValueError(f"Expected (C, Z, H, W), got {tuple(voxel_tensor.shape)}")
    target_size = tuple(int(value) for value in target_size)
    if len(target_size) != 3 or any(value <= 0 for value in target_size):
        raise ValueError(f"target_size 必须包含三个正整数，当前为 {target_size}")

    x = voxel_tensor.unsqueeze(0).float()
    occ = x[:, 0:1]
    resized_occ = F.adaptive_max_pool3d(occ, target_size)
    outputs = [resized_occ]
    occ_density = F.interpolate(occ, size=target_size, mode="trilinear", align_corners=False)
    for channel_index in range(1, x.shape[1]):
        channel = x[:, channel_index : channel_index + 1]
        if mask_channel is not None and channel_index == mask_channel:
            outputs.append(F.adaptive_max_pool3d(channel, target_size))
            continue
        weighted = F.interpolate(
            channel * occ,
            size=target_size,
            mode="trilinear",
            align_corners=False,
        )
        outputs.append(weighted / occ_density.clamp_min(EPS))
    return torch.cat(outputs, dim=1).squeeze(0)


def load_target_occ_resized(
    path: str,
    source_pc_range: Sequence[float],
    model_pc_range: Sequence[float],
    target_size: Sequence[int],
) -> np.ndarray:
    """读取 target voxel 并按物理范围、训练网格协议返回 occupancy 通道。"""
    if path.endswith(".npz"):
        target = load_sparse_voxel(path)
    else:
        target = np.load(path).astype(np.float32)
    tensor = torch.from_numpy(target).permute(3, 2, 0, 1)
    cropped = crop_voxel_channels_to_pc_range(
        tensor,
        source_pc_range=source_pc_range,
        model_pc_range=model_pc_range,
    )
    resized = resize_voxel_channels(cropped, target_size, mask_channel=3)
    return resized[0].cpu().numpy()


def voxel_to_pointcloud(
    voxel: np.ndarray,
    voxel_size: Optional[Sequence[float]],
    pc_range: Sequence[float],
    occ_threshold: float = 0.1,
    empty_fallback_topk: int = 0,
) -> Tuple[np.ndarray, bool]:
    """将 `(C,Z,X,Y)` 体素按阈值转换为 XYZI 点云。"""
    if np.asarray(voxel).ndim != 4 or np.asarray(voxel).shape[0] < 2:
        raise ValueError("voxel 必须为 (C,Z,X,Y)，且至少包含 occupancy/intensity 两个通道")
    pc_range = tuple(float(value) for value in pc_range)
    if len(pc_range) != 6:
        raise ValueError("pc_range 必须包含 6 个数")
    occ = np.asarray(voxel)[0]
    intensity = np.asarray(voxel)[1]
    if voxel_size is None:
        voxel_size = (
            (pc_range[3] - pc_range[0]) / max(occ.shape[1], 1),
            (pc_range[4] - pc_range[1]) / max(occ.shape[2], 1),
            (pc_range[5] - pc_range[2]) / max(occ.shape[0], 1),
        )
    voxel_size = tuple(float(value) for value in voxel_size)
    occ_mask = occ > float(occ_threshold)
    used_topk_fallback = False
    if not np.any(occ_mask):
        if int(empty_fallback_topk) <= 0:
            return np.zeros((0, 4), dtype=np.float32), used_topk_fallback
        used_topk_fallback = True
        flat_occ = occ.reshape(-1)
        k = int(min(max(int(empty_fallback_topk), 1), flat_occ.shape[0]))
        topk_idx = np.argpartition(flat_occ, -k)[-k:]
        z_idx, x_idx, y_idx = np.unravel_index(topk_idx, occ.shape)
    else:
        z_idx, x_idx, y_idx = np.where(occ_mask)

    x = pc_range[0] + (x_idx + 0.5) * voxel_size[0]
    y = pc_range[1] + (y_idx + 0.5) * voxel_size[1]
    z = pc_range[2] + (z_idx + 0.5) * voxel_size[2]
    inten = intensity[z_idx, x_idx, y_idx]
    return np.stack([x, y, z, inten], axis=1).astype(np.float32), used_topk_fallback
