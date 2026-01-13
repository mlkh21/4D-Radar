# -- coding: utf-8 --
"""
数据增强模块

针对 3D 雷达体素数据的数据增强策略:
1. 几何变换 - 翻转、旋转
2. 噪声注入 - 模拟雷达噪声特性
3. 稀疏度变化 - 随机丢弃/添加点
4. 强度/多普勒扰动 - 特征通道增强
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import Tuple, Optional, List


class VoxelAugmentation:
    """
    3D 体素数据增强
    
    输入体素格式: (C, D, H, W) 或 (B, C, D, H, W)
    通道含义: [占用率, 强度, 多普勒均值, 多普勒方差]
    """
    
    def __init__(
        self,
        # 几何变换
        enable_flip: bool = True,
        flip_prob: float = 0.5,
        enable_rotate: bool = True,
        rotate_prob: float = 0.5,
        
        # 噪声注入
        enable_noise: bool = True,
        noise_prob: float = 0.3,
        noise_std: float = 0.05,
        
        # 稀疏度变化
        enable_dropout: bool = True,
        dropout_prob: float = 0.2,
        point_dropout_rate: float = 0.1,
        
        # 强度扰动
        enable_intensity_jitter: bool = True,
        intensity_jitter_prob: float = 0.3,
        intensity_scale_range: Tuple[float, float] = (0.9, 1.1),
        
        # 多普勒扰动
        enable_doppler_jitter: bool = True,
        doppler_jitter_prob: float = 0.3,
        doppler_shift_range: Tuple[float, float] = (-0.1, 0.1),
    ):
        self.enable_flip = enable_flip
        self.flip_prob = flip_prob
        self.enable_rotate = enable_rotate
        self.rotate_prob = rotate_prob
        
        self.enable_noise = enable_noise
        self.noise_prob = noise_prob
        self.noise_std = noise_std
        
        self.enable_dropout = enable_dropout
        self.dropout_prob = dropout_prob
        self.point_dropout_rate = point_dropout_rate
        
        self.enable_intensity_jitter = enable_intensity_jitter
        self.intensity_jitter_prob = intensity_jitter_prob
        self.intensity_scale_range = intensity_scale_range
        
        self.enable_doppler_jitter = enable_doppler_jitter
        self.doppler_jitter_prob = doppler_jitter_prob
        self.doppler_shift_range = doppler_shift_range
    
    def __call__(
        self,
        target: torch.Tensor,
        condition: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对目标和条件同时应用增强 (保持一致性)
        
        Args:
            target: 目标体素 (C, D, H, W) 或 (B, C, D, H, W)
            condition: 条件体素 (雷达输入)
        
        Returns:
            augmented_target, augmented_condition
        """
        # 确保有 batch 维度
        squeeze_batch = False
        if target.ndim == 4:
            target = target.unsqueeze(0)
            condition = condition.unsqueeze(0)
            squeeze_batch = True
        
        # 1. 几何变换 (同时应用于 target 和 condition)
        if self.enable_flip and random.random() < self.flip_prob:
            target, condition = self._random_flip(target, condition)
        
        if self.enable_rotate and random.random() < self.rotate_prob:
            target, condition = self._random_rotate(target, condition)
        
        # 2. 噪声注入 (只对 condition 应用，模拟雷达噪声)
        if self.enable_noise and random.random() < self.noise_prob:
            condition = self._add_noise(condition)
        
        # 3. 稀疏度变化 (只对 condition 应用)
        if self.enable_dropout and random.random() < self.dropout_prob:
            condition = self._random_dropout(condition)
        
        # 4. 强度扰动 (同时应用，但扰动参数相同)
        if self.enable_intensity_jitter and random.random() < self.intensity_jitter_prob:
            target, condition = self._intensity_jitter(target, condition)
        
        # 5. 多普勒扰动 (同时应用)
        if self.enable_doppler_jitter and random.random() < self.doppler_jitter_prob:
            target, condition = self._doppler_jitter(target, condition)
        
        if squeeze_batch:
            target = target.squeeze(0)
            condition = condition.squeeze(0)
        
        return target, condition
    
    def _random_flip(
        self,
        target: torch.Tensor,
        condition: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """随机翻转 (水平/前后)"""
        # 随机选择翻转轴
        if random.random() < 0.5:
            # 水平翻转 (沿 W 轴)
            target = torch.flip(target, dims=[-1])
            condition = torch.flip(condition, dims=[-1])
        else:
            # 前后翻转 (沿 H 轴)
            target = torch.flip(target, dims=[-2])
            condition = torch.flip(condition, dims=[-2])
        
        return target, condition
    
    def _random_rotate(
        self,
        target: torch.Tensor,
        condition: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """随机旋转 (90° 增量，在 H-W 平面)"""
        k = random.randint(1, 3)  # 旋转 90°, 180°, 或 270°
        
        # torch.rot90 沿 dims 指定的平面旋转
        target = torch.rot90(target, k, dims=[-2, -1])
        condition = torch.rot90(condition, k, dims=[-2, -1])
        
        return target, condition
    
    def _add_noise(self, voxel: torch.Tensor) -> torch.Tensor:
        """添加噪声 (只在非空体素上)"""
        # 获取占用掩码
        occupancy_mask = (voxel[:, 0:1] > 0).float()
        
        # 生成噪声
        noise = torch.randn_like(voxel) * self.noise_std
        
        # 只在占用区域添加噪声
        voxel = voxel + noise * occupancy_mask
        
        return voxel
    
    def _random_dropout(self, voxel: torch.Tensor) -> torch.Tensor:
        """随机丢弃点 (模拟雷达漏检)"""
        B, C, D, H, W = voxel.shape
        
        # 获取占用掩码
        occupancy_mask = (voxel[:, 0:1] > 0).float()
        
        # 生成丢弃掩码
        dropout_mask = (torch.rand(B, 1, D, H, W, device=voxel.device) > self.point_dropout_rate).float()
        
        # 应用丢弃
        voxel = voxel * (1 - occupancy_mask + occupancy_mask * dropout_mask)
        
        return voxel
    
    def _intensity_jitter(
        self,
        target: torch.Tensor,
        condition: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """强度扰动 (通道 1)"""
        scale = random.uniform(*self.intensity_scale_range)
        
        # 只扰动强度通道 (索引 1)
        target[:, 1:2] = target[:, 1:2] * scale
        condition[:, 1:2] = condition[:, 1:2] * scale
        
        return target, condition
    
    def _doppler_jitter(
        self,
        target: torch.Tensor,
        condition: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """多普勒扰动 (通道 2)"""
        shift = random.uniform(*self.doppler_shift_range)
        
        # 获取占用掩码
        target_mask = (target[:, 0:1] > 0).float()
        condition_mask = (condition[:, 0:1] > 0).float()
        
        # 只扰动多普勒通道 (索引 2)，且只在占用区域
        target[:, 2:3] = target[:, 2:3] + shift * target_mask
        condition[:, 2:3] = condition[:, 2:3] + shift * condition_mask
        
        return target, condition


class MixupAugmentation:
    """
    Mixup 数据增强
    
    将两个样本混合，增加数据多样性
    """
    
    def __init__(self, alpha: float = 0.2, prob: float = 0.3):
        """
        Args:
            alpha: Beta 分布参数
            prob: 应用 Mixup 的概率
        """
        self.alpha = alpha
        self.prob = prob
    
    def __call__(
        self,
        target1: torch.Tensor,
        condition1: torch.Tensor,
        target2: torch.Tensor,
        condition2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        混合两个样本
        """
        if random.random() > self.prob:
            return target1, condition1
        
        # 从 Beta 分布采样混合比例
        lam = np.random.beta(self.alpha, self.alpha)
        
        # 混合
        target = lam * target1 + (1 - lam) * target2
        condition = lam * condition1 + (1 - lam) * condition2
        
        return target, condition


class CutoutAugmentation:
    """
    Cutout 数据增强
    
    随机遮挡部分区域，增强鲁棒性
    """
    
    def __init__(
        self,
        prob: float = 0.3,
        cutout_size_range: Tuple[float, float] = (0.1, 0.3),
        num_cutouts: int = 1,
    ):
        self.prob = prob
        self.cutout_size_range = cutout_size_range
        self.num_cutouts = num_cutouts
    
    def __call__(
        self,
        target: torch.Tensor,
        condition: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """应用 Cutout"""
        if random.random() > self.prob:
            return target, condition
        
        # 只对 condition 应用 cutout (模拟部分遮挡)
        B, C, D, H, W = condition.shape
        
        for _ in range(self.num_cutouts):
            # 随机大小
            cut_ratio = random.uniform(*self.cutout_size_range)
            cut_d = int(D * cut_ratio)
            cut_h = int(H * cut_ratio)
            cut_w = int(W * cut_ratio)
            
            # 随机位置
            d_start = random.randint(0, D - cut_d)
            h_start = random.randint(0, H - cut_h)
            w_start = random.randint(0, W - cut_w)
            
            # 遮挡
            condition[:, :, d_start:d_start+cut_d, h_start:h_start+cut_h, w_start:w_start+cut_w] = 0
        
        return target, condition


class ComposedAugmentation:
    """组合多种增强策略"""
    
    def __init__(
        self,
        augmentations: List = None,
        training: bool = True,
    ):
        if augmentations is None:
            # 默认增强配置
            self.augmentations = [
                VoxelAugmentation(),
            ]
        else:
            self.augmentations = augmentations
        
        self.training = training
    
    def __call__(
        self,
        target: torch.Tensor,
        condition: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """依次应用所有增强"""
        if not self.training:
            return target, condition
        
        for aug in self.augmentations:
            target, condition = aug(target, condition)
        
        return target, condition


def create_default_augmentation(training: bool = True) -> ComposedAugmentation:
    """创建默认增强配置"""
    return ComposedAugmentation(
        augmentations=[
            VoxelAugmentation(
                enable_flip=True,
                flip_prob=0.5,
                enable_rotate=True,
                rotate_prob=0.3,
                enable_noise=True,
                noise_prob=0.2,
                noise_std=0.03,
                enable_dropout=True,
                dropout_prob=0.15,
                point_dropout_rate=0.05,
                enable_intensity_jitter=True,
                intensity_jitter_prob=0.2,
                enable_doppler_jitter=True,
                doppler_jitter_prob=0.2,
            ),
            CutoutAugmentation(prob=0.1, num_cutouts=1),
        ],
        training=training,
    )
