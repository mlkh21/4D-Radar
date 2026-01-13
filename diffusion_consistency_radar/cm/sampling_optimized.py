# -- coding: utf-8 --

"""
优化的采样模块 - 非对称下采样/上采样
针对3D雷达数据的特点：Z轴（高度/多普勒）通常比XY平面小得多

核心优化：
1. 非对称下采样：在浅层只对X,Y下采样，保留Z轴分辨率
2. 可配置的stride策略
3. 支持多种下采样模式
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union, Optional

from .nn import conv_nd, avg_pool_nd


class AsymmetricDownsample3D(nn.Module):
    """
    非对称3D下采样层
    
    支持在不同维度使用不同的下采样率:
    - Z轴（深度/高度）：通常保持或较少下采样
    - XY平面：正常下采样
    
    这对于雷达数据非常重要，因为Z轴分辨率本身就较低
    
    Args:
        channels: 输入通道数
        use_conv: 是否使用卷积下采样（vs 池化）
        out_channels: 输出通道数
        stride: 下采样步长，可以是:
            - int: 所有维度使用相同步长
            - Tuple[int, int, int]: (D, H, W) 各维度步长
            - str: 预设策略名称 ("xy_only", "z_half", "full")
        kernel_size: 卷积核大小
    """
    
    # 预设的步长策略
    STRIDE_PRESETS = {
        "xy_only": (1, 2, 2),      # 只对XY下采样，保留Z
        "z_half": (1, 2, 2),       # Z轴步长为1（后续可改为2）
        "z_quarter": (1, 2, 2),    # 深层才对Z下采样
        "full": (2, 2, 2),         # 完整3D下采样
        "adaptive": None,          # 根据当前分辨率自动选择
    }
    
    def __init__(
        self,
        channels: int,
        use_conv: bool = True,
        out_channels: Optional[int] = None,
        stride: Union[int, Tuple[int, int, int], str] = "xy_only",
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        current_z_size: Optional[int] = None,  # 用于自适应策略
        min_z_size: int = 4,  # Z轴最小保持的大小
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.min_z_size = min_z_size
        
        # 解析步长
        if isinstance(stride, str):
            if stride == "adaptive" and current_z_size is not None:
                # 自适应策略：如果Z轴太小就不下采样
                self.stride = (1 if current_z_size <= min_z_size else 2, 2, 2)
            elif stride in self.STRIDE_PRESETS:
                self.stride = self.STRIDE_PRESETS[stride]
                if self.stride is None:
                    self.stride = (1, 2, 2)  # 默认
            else:
                raise ValueError(f"Unknown stride preset: {stride}")
        elif isinstance(stride, int):
            self.stride = (stride, stride, stride)
        else:
            self.stride = tuple(stride)
            
        # 解析核大小
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size, kernel_size)
        else:
            self.kernel_size = tuple(kernel_size)
        
        # 创建下采样操作
        if use_conv:
            # 计算padding以保持输出大小 = input_size / stride
            padding = tuple((k - 1) // 2 for k in self.kernel_size)
            self.op = nn.Conv3d(
                channels, 
                self.out_channels, 
                kernel_size=self.kernel_size,
                stride=self.stride, 
                padding=padding
            )
        else:
            assert channels == self.out_channels, \
                "Pooling requires same in/out channels"
            self.op = nn.AvgPool3d(
                kernel_size=self.stride,
                stride=self.stride
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W)
        Returns:
            (B, C', D', H', W') where D' = D/stride[0], etc.
        """
        assert x.shape[1] == self.channels
        return self.op(x)
    
    def get_output_size(self, input_size: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """计算输出空间大小"""
        return tuple(
            (s + self.stride[i] - 1) // self.stride[i] 
            for i, s in enumerate(input_size)
        )


class AsymmetricUpsample3D(nn.Module):
    """
    非对称3D上采样层
    
    与 AsymmetricDownsample3D 配对使用
    
    Args:
        channels: 输入通道数
        use_conv: 是否使用卷积（上采样后）
        out_channels: 输出通道数
        scale_factor: 上采样因子，格式同 stride
    """
    
    SCALE_PRESETS = {
        "xy_only": (1, 2, 2),
        "z_half": (1, 2, 2),
        "full": (2, 2, 2),
    }
    
    def __init__(
        self,
        channels: int,
        use_conv: bool = True,
        out_channels: Optional[int] = None,
        scale_factor: Union[int, Tuple[int, int, int], str] = "xy_only",
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        
        # 解析上采样因子
        if isinstance(scale_factor, str):
            self.scale_factor = self.SCALE_PRESETS.get(scale_factor, (1, 2, 2))
        elif isinstance(scale_factor, int):
            self.scale_factor = (scale_factor, scale_factor, scale_factor)
        else:
            self.scale_factor = tuple(scale_factor)
        
        if use_conv:
            self.conv = nn.Conv3d(channels, self.out_channels, 3, padding=1)
        else:
            self.conv = None
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W)
        Returns:
            (B, C', D*scale[0], H*scale[1], W*scale[2])
        """
        assert x.shape[1] == self.channels
        
        # 三线性插值上采样
        x = F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode='trilinear',
            align_corners=False
        )
        
        if self.conv is not None:
            x = self.conv(x)
            
        return x


class AdaptiveDownsampleScheduler:
    """
    自适应下采样调度器
    
    根据当前分辨率自动决定每一层的下采样策略
    
    策略：
    1. 当Z轴分辨率大于阈值时，允许Z轴下采样
    2. 当Z轴分辨率小于阈值时，只对XY下采样
    3. 确保Z轴永远不会降到min_z以下
    
    Args:
        initial_size: 初始空间大小 (D, H, W)
        num_levels: 下采样层数
        min_z: Z轴最小大小
        z_downsample_threshold: Z轴下采样的阈值
    """
    
    def __init__(
        self,
        initial_size: Tuple[int, int, int],
        num_levels: int,
        min_z: int = 4,
        z_downsample_threshold: int = 8,
    ):
        self.initial_size = initial_size
        self.num_levels = num_levels
        self.min_z = min_z
        self.z_threshold = z_downsample_threshold
        
        # 预计算每层的stride
        self.strides = self._compute_strides()
        
    def _compute_strides(self) -> list:
        """计算每层的stride策略"""
        strides = []
        current_size = list(self.initial_size)
        
        for level in range(self.num_levels):
            d, h, w = current_size
            
            # 决定Z轴是否下采样
            if d > self.z_threshold and d // 2 >= self.min_z:
                stride = (2, 2, 2)  # 完整下采样
            else:
                stride = (1, 2, 2)  # 只对XY下采样
            
            strides.append(stride)
            
            # 更新当前大小
            current_size = [
                s // stride[i] for i, s in enumerate(current_size)
            ]
            
        return strides
    
    def get_stride(self, level: int) -> Tuple[int, int, int]:
        """获取指定层的stride"""
        if level < len(self.strides):
            return self.strides[level]
        return (1, 2, 2)  # 默认
    
    def get_scale_factor(self, level: int) -> Tuple[int, int, int]:
        """获取上采样时的scale factor（stride的逆）"""
        return self.get_stride(level)
    
    def print_schedule(self):
        """打印下采样计划"""
        print("Adaptive Downsample Schedule:")
        print(f"Initial size: {self.initial_size}")
        current_size = list(self.initial_size)
        for i, stride in enumerate(self.strides):
            current_size = [s // stride[j] for j, s in enumerate(current_size)]
            print(f"  Level {i}: stride={stride} -> size={tuple(current_size)}")


class DepthWiseDownsample3D(nn.Module):
    """
    深度可分离3D下采样
    
    使用深度可分离卷积进行下采样，参数量更少
    
    Args:
        channels: 输入通道数
        out_channels: 输出通道数
        stride: 下采样步长
    """
    
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        stride: Tuple[int, int, int] = (1, 2, 2),
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.stride = stride
        
        # 深度卷积（逐通道）
        self.depthwise = nn.Conv3d(
            channels, channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=channels  # 深度可分离
        )
        
        # 逐点卷积（通道混合）
        self.pointwise = nn.Conv3d(
            channels, self.out_channels,
            kernel_size=1
        )
        
        self.norm = nn.GroupNorm(min(32, self.out_channels), self.out_channels)
        self.act = nn.SiLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class MultiScaleDownsample3D(nn.Module):
    """
    多尺度3D下采样
    
    同时捕获不同尺度的特征，类似 Inception 模块
    
    Args:
        channels: 输入通道数
        out_channels: 输出通道数
        stride: 下采样步长
    """
    
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        stride: Tuple[int, int, int] = (1, 2, 2),
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        
        # 分支通道数
        branch_ch = self.out_channels // 3
        remainder = self.out_channels - branch_ch * 2
        
        # 分支1: 1x1 conv -> stride conv
        self.branch1 = nn.Sequential(
            nn.Conv3d(channels, branch_ch, 1),
            nn.Conv3d(branch_ch, branch_ch, 3, stride=stride, padding=1),
        )
        
        # 分支2: 1x1 conv -> 3x3 conv -> stride conv
        self.branch2 = nn.Sequential(
            nn.Conv3d(channels, branch_ch, 1),
            nn.Conv3d(branch_ch, branch_ch, 3, padding=1),
            nn.Conv3d(branch_ch, branch_ch, 3, stride=stride, padding=1),
        )
        
        # 分支3: 池化 -> 1x1 conv
        self.branch3 = nn.Sequential(
            nn.AvgPool3d(kernel_size=stride, stride=stride),
            nn.Conv3d(channels, remainder, 1),
        )
        
        self.norm = nn.GroupNorm(min(32, self.out_channels), self.out_channels)
        self.act = nn.SiLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        
        out = torch.cat([b1, b2, b3], dim=1)
        out = self.norm(out)
        out = self.act(out)
        return out


# ==============================================================================
# 下采样工厂函数
# ==============================================================================

def create_downsample_block(
    channels: int,
    out_channels: Optional[int] = None,
    dims: int = 3,
    use_conv: bool = True,
    downsample_type: str = "asymmetric",
    stride: Union[str, Tuple[int, int, int]] = "xy_only",
    current_z_size: Optional[int] = None,
    **kwargs
) -> nn.Module:
    """
    创建下采样块的工厂函数
    
    Args:
        channels: 输入通道数
        out_channels: 输出通道数
        dims: 维度 (2 或 3)
        use_conv: 是否使用卷积
        downsample_type: 下采样类型
            - "asymmetric": 非对称下采样（推荐用于3D雷达）
            - "depthwise": 深度可分离下采样
            - "multiscale": 多尺度下采样
            - "standard": 标准下采样（原始行为）
        stride: 步长策略
        current_z_size: 当前Z轴大小（用于自适应策略）
    """
    out_channels = out_channels or channels
    
    if dims == 2:
        # 2D 情况，使用标准下采样
        if use_conv:
            return nn.Conv2d(channels, out_channels, 3, stride=2, padding=1)
        else:
            return nn.AvgPool2d(2, 2)
    
    # 3D 情况
    if downsample_type == "asymmetric":
        return AsymmetricDownsample3D(
            channels=channels,
            use_conv=use_conv,
            out_channels=out_channels,
            stride=stride,
            current_z_size=current_z_size,
        )
    elif downsample_type == "depthwise":
        parsed_stride = AsymmetricDownsample3D.STRIDE_PRESETS.get(stride, (1, 2, 2)) \
            if isinstance(stride, str) else stride
        return DepthWiseDownsample3D(
            channels=channels,
            out_channels=out_channels,
            stride=parsed_stride,
        )
    elif downsample_type == "multiscale":
        parsed_stride = AsymmetricDownsample3D.STRIDE_PRESETS.get(stride, (1, 2, 2)) \
            if isinstance(stride, str) else stride
        return MultiScaleDownsample3D(
            channels=channels,
            out_channels=out_channels,
            stride=parsed_stride,
        )
    else:
        # 标准下采样（原始行为）
        standard_stride = (1, 2, 2) if dims == 3 else 2
        if use_conv:
            return nn.Conv3d(
                channels, out_channels, 3, stride=standard_stride, padding=1
            )
        else:
            return nn.AvgPool3d(kernel_size=standard_stride, stride=standard_stride)


def create_upsample_block(
    channels: int,
    out_channels: Optional[int] = None,
    dims: int = 3,
    use_conv: bool = True,
    upsample_type: str = "asymmetric",
    scale_factor: Union[str, Tuple[int, int, int]] = "xy_only",
    **kwargs
) -> nn.Module:
    """
    创建上采样块的工厂函数
    """
    out_channels = out_channels or channels
    
    if dims == 2:
        # 2D 情况
        layers = [nn.Upsample(scale_factor=2, mode='nearest')]
        if use_conv:
            layers.append(nn.Conv2d(channels, out_channels, 3, padding=1))
        return nn.Sequential(*layers)
    
    # 3D 情况
    if upsample_type == "asymmetric":
        return AsymmetricUpsample3D(
            channels=channels,
            use_conv=use_conv,
            out_channels=out_channels,
            scale_factor=scale_factor,
        )
    else:
        # 标准上采样
        parsed_scale = AsymmetricUpsample3D.SCALE_PRESETS.get(scale_factor, (1, 2, 2)) \
            if isinstance(scale_factor, str) else scale_factor
        layers = [nn.Upsample(scale_factor=parsed_scale, mode='trilinear', align_corners=False)]
        if use_conv:
            layers.append(nn.Conv3d(channels, out_channels, 3, padding=1))
        return nn.Sequential(*layers)
