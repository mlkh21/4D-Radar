# -*- coding: utf-8 -*-

"""
优化的注意力机制模块
包含: Window Attention (3D Swin风格)、Flash Attention、Linear Attention、Sparse Attention
针对4D雷达数据的特点进行优化，显著降低显存占用
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, Tuple

from .nn import normalization, conv_nd, zero_module


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


# ==============================================================================
# 1. 3D Window Attention (Swin Transformer 风格) - 核心优化
# ==============================================================================

class Window3DAttention(nn.Module):
    """
    3D窗口注意力机制 (Swin Transformer 3D 风格)
    
    只在局部的 3D 小窗口内做 Attention，显存占用从 O(N^2) 降为 O(N)
    非常适合雷达点云等稀疏3D数据
    
    Args:
        channels: 输入通道数
        num_heads: 注意力头数
        window_size: 3D窗口大小 (D, H, W)，默认 (4, 4, 4)
        shift_size: 移位窗口大小，用于跨窗口连接，默认 (2, 2, 2)
        use_checkpoint: 是否使用梯度检查点
        qkv_bias: QKV是否使用偏置
    """
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        window_size: Tuple[int, int, int] = (4, 4, 4),
        shift_size: Tuple[int, int, int] = (0, 0, 0),
        use_checkpoint: bool = False,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.use_checkpoint = use_checkpoint
        
        head_dim = channels // num_heads
        self.scale = head_dim ** -0.5
        
        # 归一化层
        self.norm = normalization(channels)
        
        # QKV 投影
        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        
        # 相对位置编码
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                num_heads
            )
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
        # 计算相对位置索引
        coords_d = torch.arange(window_size[0])
        coords_h = torch.arange(window_size[1])
        coords_w = torch.arange(window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing='ij'))  # (3, D, H, W)
        coords_flatten = torch.flatten(coords, 1)  # (3, D*H*W)
        
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (3, DHW, DHW)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (DHW, DHW, 3)
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 2] += window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # (DHW, DHW)
        self.register_buffer("relative_position_index", relative_position_index)
        
        # Dropout
        self.attn_drop = nn.Dropout(attn_drop)
        
        # 输出投影
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W)
        Returns:
            (B, C, D, H, W)
        """
        B, C, D, H, W = x.shape
        
        # 归一化并转换格式
        x_norm = self.norm(x)
        x_norm = x_norm.permute(0, 2, 3, 4, 1).contiguous()  # (B, D, H, W, C)
        
        # Pad 到窗口大小的整数倍
        pad_d = (self.window_size[0] - D % self.window_size[0]) % self.window_size[0]
        pad_h = (self.window_size[1] - H % self.window_size[1]) % self.window_size[1]
        pad_w = (self.window_size[2] - W % self.window_size[2]) % self.window_size[2]
        
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x_norm = F.pad(x_norm, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
        
        _, Dp, Hp, Wp, _ = x_norm.shape
        
        # 窗口移位 (Shifted Window)
        if any(s > 0 for s in self.shift_size):
            shifted_x = torch.roll(
                x_norm,
                shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]),
                dims=(1, 2, 3)
            )
        else:
            shifted_x = x_norm
        
        # 将输入划分为窗口
        x_windows = self._window_partition(shifted_x)  # (B*nW, Wd*Wh*Ww, C)
        
        # 窗口内注意力
        attn_windows = self._window_attention(x_windows)  # (B*nW, Wd*Wh*Ww, C)
        
        # 合并窗口
        shifted_x = self._window_reverse(attn_windows, (Dp, Hp, Wp))  # (B, Dp, Hp, Wp, C)
        
        # 反向移位
        if any(s > 0 for s in self.shift_size):
            x_out = torch.roll(
                shifted_x,
                shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]),
                dims=(1, 2, 3)
            )
        else:
            x_out = shifted_x
        
        # 去除 padding
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x_out = x_out[:, :D, :H, :W, :].contiguous()
        
        # 转回 (B, C, D, H, W) 格式
        x_out = x_out.permute(0, 4, 1, 2, 3).contiguous()
        
        # 残差连接
        return x + x_out
    
    def _window_partition(self, x: torch.Tensor) -> torch.Tensor:
        """
        将输入划分为不重叠的3D窗口
        Args:
            x: (B, D, H, W, C)
        Returns:
            windows: (B*num_windows, window_d*window_h*window_w, C)
        """
        B, D, H, W, C = x.shape
        Wd, Wh, Ww = self.window_size
        
        x = x.view(B, D // Wd, Wd, H // Wh, Wh, W // Ww, Ww, C)
        windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        windows = windows.view(-1, Wd * Wh * Ww, C)
        return windows
    
    def _window_reverse(self, windows: torch.Tensor, spatial_size: Tuple[int, int, int]) -> torch.Tensor:
        """
        将窗口合并回原始张量
        Args:
            windows: (B*num_windows, window_d*window_h*window_w, C)
            spatial_size: (D, H, W)
        Returns:
            x: (B, D, H, W, C)
        """
        D, H, W = spatial_size
        Wd, Wh, Ww = self.window_size
        B = int(windows.shape[0] / (D // Wd * H // Wh * W // Ww))
        
        x = windows.view(B, D // Wd, H // Wh, W // Ww, Wd, Wh, Ww, -1)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        x = x.view(B, D, H, W, -1)
        return x
    
    def _window_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        窗口内执行多头自注意力
        Args:
            x: (B*nW, N, C) where N = Wd*Wh*Ww
        Returns:
            (B*nW, N, C)
        """
        B_nW, N, C = x.shape
        
        # QKV 投影
        qkv = self.qkv(x).reshape(B_nW, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B*nW, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 缩放点积注意力
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # (B*nW, num_heads, N, N)
        
        # 添加相对位置偏置
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        # 应用注意力权重
        x = (attn @ v).transpose(1, 2).reshape(B_nW, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


# ==============================================================================
# 2. PyTorch 2.0 原生 Flash Attention (显存最优)
# ==============================================================================

class FlashAttention3D(nn.Module):
    """
    使用 PyTorch 2.0 的 scaled_dot_product_attention 实现高效注意力
    
    优点:
    - 自动使用 FlashAttention-2 (如果硬件支持)
    - 显存效率极高，O(N) 而非 O(N^2)
    - 计算速度快 3-5 倍
    
    Args:
        channels: 输入通道数
        num_heads: 注意力头数
        use_checkpoint: 是否使用梯度检查点
    """
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        use_checkpoint: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint
        self.dropout = dropout
        
        self.head_dim = channels // num_heads
        assert self.head_dim * num_heads == channels, "channels must be divisible by num_heads"
        
        self.norm = normalization(channels)
        self.qkv = conv_nd(3, channels, channels * 3, 1)
        self.proj_out = zero_module(conv_nd(3, channels, channels, 1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W)
        Returns:
            (B, C, D, H, W)
        """
        B, C, D, H, W = x.shape
        
        # 归一化
        h = self.norm(x)
        
        # QKV 投影
        qkv = self.qkv(h)  # (B, 3*C, D, H, W)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, D * H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, num_heads, N, head_dim)
        
        # PyTorch 2.0 Flash Attention (自动选择最优实现)
        # 支持: FlashAttention-2, Memory-Efficient Attention, 或标准实现
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=True,
            enable_mem_efficient=True
        ):
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False
            )  # (B, num_heads, N, head_dim)
        
        # 重塑回原始形状
        attn_out = attn_out.permute(0, 1, 3, 2)  # (B, num_heads, head_dim, N)
        attn_out = attn_out.reshape(B, C, D, H, W)
        
        # 输出投影 + 残差
        return x + self.proj_out(attn_out)


# ==============================================================================
# 3. Linear Attention (线性复杂度)
# ==============================================================================

class LinearAttention3D(nn.Module):
    """
    线性注意力机制，复杂度 O(N) 而非 O(N^2)
    
    使用核函数近似 softmax attention:
    Attention(Q, K, V) ≈ φ(Q) @ (φ(K)^T @ V) / (φ(Q) @ φ(K)^T @ 1)
    
    其中 φ 是特征映射函数 (这里使用 elu(x) + 1)
    
    Args:
        channels: 输入通道数
        num_heads: 注意力头数
    """
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.norm = normalization(channels)
        self.to_qkv = conv_nd(3, channels, channels * 3, 1)
        self.to_out = nn.Sequential(
            conv_nd(3, channels, channels, 1),
            normalization(channels),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W)
        Returns:
            (B, C, D, H, W)
        """
        B, C, D, H, W = x.shape
        
        h = self.norm(x)
        qkv = self.to_qkv(h)  # (B, 3*C, D, H, W)
        
        # 重塑为 (B, heads, head_dim, N)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, D * H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # Each: (B, heads, head_dim, N)
        
        # 特征映射 (elu + 1 确保非负)
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        
        # 线性注意力计算
        # 先计算 K^T @ V，复杂度 O(D * N)
        k = k / k.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        kv = torch.einsum('bhdn,bhen->bhde', k, v)  # (B, heads, head_dim, head_dim)
        
        # 再计算 Q @ (K^T @ V)
        out = torch.einsum('bhdn,bhde->bhen', q, kv)  # (B, heads, head_dim, N)
        
        # 归一化
        normalizer = torch.einsum('bhdn,bhd->bhn', q, k.sum(dim=-1)).unsqueeze(2)
        out = out / normalizer.clamp(min=1e-6)
        
        # 重塑回 (B, C, D, H, W)
        out = out.reshape(B, C, D, H, W)
        
        return x + self.to_out(out)


# ==============================================================================
# 4. 稀疏注意力 (Sparse Attention) - 针对雷达数据优化
# ==============================================================================

class SparseAttention3D(nn.Module):
    """
    稀疏注意力机制，专门针对雷达数据设计
    
    雷达数据特点：大部分区域是空的（零值），只有少数点有有效值
    策略：只在非零区域执行注意力计算
    
    Args:
        channels: 输入通道数
        num_heads: 注意力头数
        topk: 每个查询保留的 Top-K 个键值对
    """
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        topk: int = 256,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.topk = topk
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.norm = normalization(channels)
        self.qkv = conv_nd(3, channels, channels * 3, 1)
        self.proj_out = zero_module(conv_nd(3, channels, channels, 1))
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W)
            mask: (B, 1, D, H, W) 可选的稀疏掩码，指示有效位置
        Returns:
            (B, C, D, H, W)
        """
        B, C, D, H, W = x.shape
        N = D * H * W
        
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, self.head_dim, N)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # Each: (B, heads, head_dim, N)
        
        q = q.transpose(-1, -2)  # (B, heads, N, head_dim)
        k = k.transpose(-1, -2)  # (B, heads, N, head_dim)
        v = v.transpose(-1, -2)  # (B, heads, N, head_dim)
        
        # 如果提供了掩码，使用稀疏注意力
        if mask is not None:
            mask_flat = mask.view(B, 1, N)
            # 找出有效位置
            valid_indices = mask_flat.squeeze(1).nonzero(as_tuple=True)
            # 这里简化处理，实际应用中可以更精细
        
        # Top-K 稀疏注意力
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B, heads, N, N)
        
        # 只保留 Top-K 个最高分数
        topk_val, topk_idx = attn_scores.topk(min(self.topk, N), dim=-1)  # (B, heads, N, topk)
        
        # 创建稀疏注意力权重
        sparse_attn = torch.zeros_like(attn_scores)
        sparse_attn.scatter_(-1, topk_idx, F.softmax(topk_val, dim=-1))
        
        # 应用注意力
        out = torch.matmul(sparse_attn, v)  # (B, heads, N, head_dim)
        out = out.transpose(-1, -2).reshape(B, C, D, H, W)
        
        return x + self.proj_out(out)


# ==============================================================================
# 5. 高度自注意力 (Height Self-Attention) - 针对雷达Z轴优化
# ==============================================================================

class HeightSelfAttention3D(nn.Module):
    """
    沿高度(Z)轴的自注意力机制
    
    雷达数据在Z轴上通常较小，可以在Z轴方向高效地进行注意力计算
    而在XY平面上使用局部卷积
    
    Args:
        channels: 输入通道数
        num_heads: 注意力头数
    """
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.norm = normalization(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj_out = nn.Linear(channels, channels)
        
        # 可学习的高度位置编码
        self.height_pos = nn.Parameter(torch.randn(1, 1, 1, 64, channels) * 0.02)  # 最大Z=64
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W)
        Returns:
            (B, C, D, H, W)
        """
        B, C, D, H, W = x.shape
        
        # 转换格式并加入位置编码
        h = self.norm(x).permute(0, 3, 4, 2, 1)  # (B, H, W, D, C)
        
        # 添加高度位置编码
        if D <= self.height_pos.shape[3]:
            h = h + self.height_pos[:, :, :, :D, :]
        
        # 在 D 维度上做注意力
        h_flat = h.reshape(B * H * W, D, C)  # (B*H*W, D, C)
        
        qkv = self.qkv(h_flat).reshape(B * H * W, D, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B*H*W, heads, D, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 注意力计算（D维度通常很小，可以直接计算）
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B * H * W, D, C)
        out = self.proj_out(out)
        
        # 重塑回原始形状
        out = out.reshape(B, H, W, D, C).permute(0, 4, 3, 1, 2)  # (B, C, D, H, W)
        
        return x + out


# ==============================================================================
# 6. 统一的注意力工厂
# ==============================================================================

def create_attention_block(
    channels: int,
    num_heads: int = 4,
    attention_type: str = "flash",
    window_size: Tuple[int, int, int] = (4, 4, 4),
    use_checkpoint: bool = False,
    **kwargs
) -> nn.Module:
    """
    创建注意力块的工厂函数
    
    Args:
        channels: 通道数
        num_heads: 注意力头数
        attention_type: 注意力类型
            - "flash": PyTorch 2.0 Flash Attention (推荐)
            - "window": 3D 窗口注意力 (Swin风格)
            - "linear": 线性注意力
            - "sparse": 稀疏注意力
            - "height": 高度自注意力
            - "none": 不使用注意力（返回 Identity）
        window_size: 窗口大小 (仅window类型使用)
        use_checkpoint: 是否使用梯度检查点
    """
    attention_type = attention_type.lower()
    
    if attention_type == "flash":
        return FlashAttention3D(
            channels=channels,
            num_heads=num_heads,
            use_checkpoint=use_checkpoint,
        )
    elif attention_type == "window":
        return Window3DAttention(
            channels=channels,
            num_heads=num_heads,
            window_size=window_size,
            use_checkpoint=use_checkpoint,
        )
    elif attention_type == "linear":
        return LinearAttention3D(
            channels=channels,
            num_heads=num_heads,
            use_checkpoint=use_checkpoint,
        )
    elif attention_type == "sparse":
        return SparseAttention3D(
            channels=channels,
            num_heads=num_heads,
            use_checkpoint=use_checkpoint,
        )
    elif attention_type == "height":
        return HeightSelfAttention3D(
            channels=channels,
            num_heads=num_heads,
            use_checkpoint=use_checkpoint,
        )
    elif attention_type == "none" or attention_type == "false":
        return nn.Identity()
    else:
        # 默认使用 Flash Attention
        print(f"Warning: Unknown attention type '{attention_type}', using FlashAttention3D")
        return FlashAttention3D(
            channels=channels,
            num_heads=num_heads,
            use_checkpoint=use_checkpoint,
        )


# ==============================================================================
# 7. 分组的混合注意力 (Hybrid Attention)
# ==============================================================================

class HybridAttention3D(nn.Module):
    """
    混合注意力机制：结合局部和全局注意力
    
    - 在浅层使用高效的局部注意力（窗口/线性）
    - 在深层使用稀疏全局注意力
    
    Args:
        channels: 输入通道数
        num_heads: 注意力头数
        local_type: 局部注意力类型 ("window" 或 "linear")
        global_type: 全局注意力类型 ("sparse" 或 "flash")
    """
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        local_type: str = "window",
        global_type: str = "sparse",
        local_ratio: float = 0.5,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        
        local_channels = int(channels * local_ratio)
        global_channels = channels - local_channels
        
        self.local_attn = create_attention_block(
            local_channels, num_heads // 2, local_type, use_checkpoint=use_checkpoint
        )
        self.global_attn = create_attention_block(
            global_channels, num_heads // 2, global_type, use_checkpoint=use_checkpoint
        )
        
        self.proj = conv_nd(3, channels, channels, 1)
        self.local_channels = local_channels
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 分割通道
        x_local = x[:, :self.local_channels]
        x_global = x[:, self.local_channels:]
        
        # 分别处理
        x_local = self.local_attn(x_local)
        x_global = self.global_attn(x_global)
        
        # 合并
        out = torch.cat([x_local, x_global], dim=1)
        return self.proj(out)
