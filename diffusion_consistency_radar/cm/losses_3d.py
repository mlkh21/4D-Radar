# -- coding: utf-8 --
"""
3D 感知损失模块

替代 2D LPIPS，专为 3D 体素数据设计:
1. 3D 特征提取器 (基于轻量 3D CNN)
2. 多尺度感知损失
3. 占用感知加权
4. 边缘/结构保持损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class LightweightFeatureExtractor3D(nn.Module):
    """
    轻量级 3D 特征提取器
    
    用于感知损失计算，提取多尺度特征
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 32,
        num_layers: int = 4,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList()
        self.in_channels = in_channels
        
        ch = in_channels
        for i in range(num_layers):
            out_ch = base_channels * (2 ** min(i, 2))  # 32, 64, 128, 128
            self.layers.append(nn.Sequential(
                nn.Conv3d(ch, out_ch, 3, stride=2, padding=1),
                nn.InstanceNorm3d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv3d(out_ch, out_ch, 3, padding=1),
                nn.InstanceNorm3d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ))
            ch = out_ch
        
        # 冻结参数 (作为感知损失时不需要训练)
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        提取多尺度特征
        
        Args:
            x: (B, C, D, H, W) 输入体素
        Returns:
            features: 多尺度特征列表
        """
        features = []
        h = x
        
        for layer in self.layers:
            h = layer(h)
            features.append(h)
        
        return features


class Perceptual3DLoss(nn.Module):
    """
    3D 感知损失
    
    基于多尺度特征的感知损失，替代 2D LPIPS
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        feature_layers: List[int] = None,
        weights: List[float] = None,
        use_occupancy_weight: bool = True,
    ):
        """
        Args:
            in_channels: 输入通道数
            feature_layers: 使用的特征层索引
            weights: 各层权重
            use_occupancy_weight: 是否使用占用率加权
        """
        super().__init__()
        
        self.feature_extractor = LightweightFeatureExtractor3D(
            in_channels=in_channels,
            base_channels=32,
            num_layers=4,
        )
        
        self.feature_layers = feature_layers or [0, 1, 2, 3]
        self.weights = weights or [1.0, 1.0, 1.0, 1.0]
        self.use_occupancy_weight = use_occupancy_weight
        
        # 归一化权重
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        occupancy_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算 3D 感知损失
        
        Args:
            pred: (B, C, D, H, W) 预测体素
            target: (B, C, D, H, W) 目标体素
            occupancy_mask: (B, 1, D, H, W) 可选的占用掩码
        
        Returns:
            loss: 感知损失值
        """
        # 提取特征
        pred_features = self.feature_extractor(pred)
        target_features = self.feature_extractor(target)
        
        # 计算各层损失
        total_loss = 0.0
        
        for i, (pf, tf) in enumerate(zip(pred_features, target_features)):
            if i not in self.feature_layers:
                continue
            
            # L1 距离
            diff = torch.abs(pf - tf)
            
            # 可选的占用加权
            if self.use_occupancy_weight and occupancy_mask is not None:
                # 下采样掩码到当前分辨率
                mask = F.interpolate(
                    occupancy_mask.float(),
                    size=diff.shape[2:],
                    mode='trilinear',
                    align_corners=False,
                )
                # 加权
                diff = diff * (mask + 0.1)  # 非占用区域也有小权重
            
            layer_loss = diff.mean()
            total_loss = total_loss + self.weights[i] * layer_loss
        
        return total_loss


class StructurePreservingLoss(nn.Module):
    """
    结构保持损失
    
    强调边缘和结构的保持，对稀疏数据特别重要
    """
    
    def __init__(self, edge_weight: float = 1.0):
        super().__init__()
        self.edge_weight = edge_weight
        
        # 3D Sobel 核
        self.register_buffer('sobel_x', self._create_sobel_kernel('x'))
        self.register_buffer('sobel_y', self._create_sobel_kernel('y'))
        self.register_buffer('sobel_z', self._create_sobel_kernel('z'))
    
    def _create_sobel_kernel(self, direction: str) -> torch.Tensor:
        """创建 3D Sobel 核"""
        if direction == 'x':
            kernel = torch.tensor([
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            ], dtype=torch.float32)
        elif direction == 'y':
            kernel = torch.tensor([
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                [[-2, -4, -2], [0, 0, 0], [2, 4, 2]],
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            ], dtype=torch.float32)
        else:  # z
            kernel = torch.tensor([
                [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
            ], dtype=torch.float32)
        
        return kernel.unsqueeze(0).unsqueeze(0) / 16.0  # (1, 1, 3, 3, 3)
    
    def compute_edges(self, x: torch.Tensor) -> torch.Tensor:
        """计算边缘强度"""
        # 只对占用通道计算边缘
        occ = x[:, 0:1]
        
        edge_x = F.conv3d(occ, self.sobel_x, padding=1)
        edge_y = F.conv3d(occ, self.sobel_y, padding=1)
        edge_z = F.conv3d(occ, self.sobel_z, padding=1)
        
        edges = torch.sqrt(edge_x**2 + edge_y**2 + edge_z**2 + 1e-6)
        return edges
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算结构保持损失
        """
        pred_edges = self.compute_edges(pred)
        target_edges = self.compute_edges(target)
        
        edge_loss = F.l1_loss(pred_edges, target_edges)
        
        return self.edge_weight * edge_loss


class OccupancyAwareLoss(nn.Module):
    """
    占用感知损失
    
    对稀疏体素数据，占用区域和非占用区域应有不同的权重
    """
    
    def __init__(
        self,
        occupied_weight: float = 5.0,
        empty_weight: float = 1.0,
        boundary_weight: float = 2.0,
    ):
        super().__init__()
        self.occupied_weight = occupied_weight
        self.empty_weight = empty_weight
        self.boundary_weight = boundary_weight
    
    def compute_weight_mask(self, target: torch.Tensor) -> torch.Tensor:
        """
        计算加权掩码
        
        Args:
            target: (B, C, D, H, W) 目标体素
        Returns:
            weight_mask: (B, 1, D, H, W)
        """
        # 占用掩码
        occupied = (target[:, 0:1] > 0).float()
        
        # 边界检测 (膨胀 - 原始)
        kernel = torch.ones(1, 1, 3, 3, 3, device=target.device)
        dilated = F.conv3d(occupied, kernel, padding=1)
        boundary = ((dilated > 0) & (dilated < 27)).float()
        
        # 构建权重掩码
        weight = torch.ones_like(occupied) * self.empty_weight
        weight = weight + occupied * (self.occupied_weight - self.empty_weight)
        weight = weight + boundary * self.boundary_weight
        
        return weight
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        base_loss: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        应用占用感知加权
        
        Args:
            pred, target: 预测和目标体素
            base_loss: 可选的基础损失 (如 MSE)，如果提供则加权
        """
        weight_mask = self.compute_weight_mask(target)
        
        if base_loss is None:
            # 计算加权 MSE
            diff = (pred - target) ** 2
            weighted_loss = (diff * weight_mask).mean()
        else:
            # 对提供的损失加权
            weighted_loss = (base_loss * weight_mask).mean()
        
        return weighted_loss


class CompositeLoss3D(nn.Module):
    """
    组合 3D 损失函数
    
    整合多种损失:
    - MSE/L1 基础损失
    - 3D 感知损失
    - 结构保持损失
    - 占用感知加权
    """
    
    def __init__(
        self,
        mse_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        structure_weight: float = 0.05,
        use_occupancy_weighting: bool = True,
        in_channels: int = 4,
    ):
        super().__init__()
        
        self.mse_weight = mse_weight
        self.perceptual_weight = perceptual_weight
        self.structure_weight = structure_weight
        
        # 子损失模块
        self.perceptual_loss = Perceptual3DLoss(in_channels=in_channels)
        self.structure_loss = StructurePreservingLoss()
        self.occupancy_loss = OccupancyAwareLoss() if use_occupancy_weighting else None
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算组合损失
        
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        loss_dict = {}
        
        # 1. 基础 MSE 损失
        mse_loss = F.mse_loss(pred, target, reduction='none')
        
        # 应用占用加权
        if self.occupancy_loss is not None:
            mse_loss = self.occupancy_loss(pred, target, mse_loss)
        else:
            mse_loss = mse_loss.mean()
        
        loss_dict['mse'] = mse_loss
        
        # 2. 3D 感知损失
        if self.perceptual_weight > 0:
            occupancy_mask = (target[:, 0:1] > 0).float()
            perceptual = self.perceptual_loss(pred, target, occupancy_mask)
            loss_dict['perceptual'] = perceptual
        else:
            perceptual = 0.0
        
        # 3. 结构保持损失
        if self.structure_weight > 0:
            structure = self.structure_loss(pred, target)
            loss_dict['structure'] = structure
        else:
            structure = 0.0
        
        # 组合
        total_loss = (
            self.mse_weight * mse_loss +
            self.perceptual_weight * perceptual +
            self.structure_weight * structure
        )
        
        loss_dict['total'] = total_loss
        
        return total_loss, loss_dict


def replace_lpips_with_3d_loss():
    """
    返回用于替换 LPIPS 的 3D 感知损失
    
    使用示例:
        # 在 karras_diffusion.py 中
        # 替换: self.lpips_loss = LPIPS(...)
        # 为: self.lpips_loss = Perceptual3DLoss(...)
    """
    return Perceptual3DLoss(
        in_channels=4,
        feature_layers=[0, 1, 2, 3],
        weights=[0.5, 1.0, 1.0, 0.5],
        use_occupancy_weight=True,
    )
