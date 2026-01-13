# -- coding: utf-8 --
"""
3D VAE 模块 - 用于 Latent Diffusion

Latent Diffusion 的核心思想：
1. 使用 VAE 将高维数据（如 128x128x32 的体素）压缩到低维潜空间（如 32x32x8）
2. 在潜空间中进行扩散训练
3. 收益：显存降低 8-16 倍，训练速度提升 10+ 倍

本模块实现：
- 轻量级 3D VQ-VAE
- 标准 3D VAE（带 KL 散度）
- 针对雷达稀疏数据的优化
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from einops import rearrange

from .nn import normalization, conv_nd


# ==============================================================================
# 基础组件
# ==============================================================================

class ResBlock3D(nn.Module):
    """3D 残差块 - 支持梯度检查点"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        norm_type: str = "group",
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.use_checkpoint = use_checkpoint
        
        self.conv1 = nn.Conv3d(in_channels, self.out_channels, 3, padding=1)
        self.conv2 = nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1)
        self.norm1 = normalization(in_channels, norm_type)
        self.norm2 = normalization(self.out_channels, norm_type)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()
        
        if in_channels != self.out_channels:
            self.skip = nn.Conv3d(in_channels, self.out_channels, 1)
        else:
            self.skip = nn.Identity()
    
    def _forward(self, x):
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        
        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return self.skip(x) + h
    
    def forward(self, x):
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)


class Downsample3D(nn.Module):
    """3D 下采样（非对称，保护 Z 轴）"""
    
    def __init__(self, channels: int, stride: Tuple[int, int, int] = (1, 2, 2)):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, 3, stride=stride, padding=1)
    
    def forward(self, x):
        return self.conv(x)


class Upsample3D(nn.Module):
    """3D 上采样"""
    
    def __init__(self, channels: int, scale_factor: Tuple[int, int, int] = (1, 2, 2)):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv3d(channels, channels, 3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='trilinear', align_corners=False)
        return self.conv(x)


class SelfAttention3D(nn.Module):
    """轻量级 3D 自注意力（用于 VAE）"""
    
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.norm = normalization(channels)
        self.qkv = nn.Conv3d(channels, channels * 3, 1)
        self.proj = nn.Conv3d(channels, channels, 1)
    
    def forward(self, x):
        B, C, D, H, W = x.shape
        
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, self.head_dim, -1)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # Each: (B, heads, head_dim, N)
        
        q = q.transpose(-1, -2)  # (B, heads, N, head_dim)
        k = k.transpose(-1, -2)
        v = v.transpose(-1, -2)
        
        # 使用 PyTorch 2.0 的高效注意力
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            attn_out = F.scaled_dot_product_attention(q, k, v)
        
        attn_out = attn_out.transpose(-1, -2).reshape(B, C, D, H, W)
        return x + self.proj(attn_out)


# ==============================================================================
# VAE 编码器
# ==============================================================================

class VAE3DEncoder(nn.Module):
    """
    3D VAE 编码器
    
    将输入体素压缩到潜空间
    例如: (B, 4, 32, 128, 128) -> (B, latent_dim, 8, 32, 32)
    
    Args:
        in_channels: 输入通道数
        latent_dim: 潜空间维度
        base_channels: 基础通道数
        channel_mult: 通道倍增系数
        num_res_blocks: 每层残差块数
        downsample_strides: 每层下采样步长
        use_attention: 是否使用注意力
        use_checkpoint: 是否使用梯度检查点
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        latent_dim: int = 4,
        base_channels: int = 64,
        channel_mult: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        downsample_strides: List[Tuple[int, int, int]] = None,
        use_attention: bool = True,
        double_z: bool = True,  # 输出 mean 和 logvar
        dropout: float = 0.0,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.double_z = double_z
        self.use_checkpoint = use_checkpoint
        
        # 默认下采样步长：Z 轴保守下采样
        if downsample_strides is None:
            downsample_strides = [(1, 2, 2), (2, 2, 2), (1, 2, 2)]
        
        # 输入卷积
        self.conv_in = nn.Conv3d(in_channels, base_channels, 3, padding=1)
        
        # 下采样块
        self.down_blocks = nn.ModuleList()
        ch = base_channels
        
        for i, mult in enumerate(channel_mult):
            block = nn.ModuleList()
            out_ch = base_channels * mult
            
            for _ in range(num_res_blocks):
                block.append(ResBlock3D(ch, out_ch, dropout, use_checkpoint=use_checkpoint))
                ch = out_ch
            
            # 在最后一层添加注意力
            if use_attention and i == len(channel_mult) - 1:
                block.append(SelfAttention3D(ch))
            
            # 下采样（除了最后一层）
            if i < len(channel_mult) - 1:
                stride = downsample_strides[i] if i < len(downsample_strides) else (1, 2, 2)
                block.append(Downsample3D(ch, stride))
            
            self.down_blocks.append(block)
        
        # 中间块
        self.mid_block = nn.Sequential(
            ResBlock3D(ch, ch, dropout, use_checkpoint=use_checkpoint),
            SelfAttention3D(ch) if use_attention else nn.Identity(),
            ResBlock3D(ch, ch, dropout, use_checkpoint=use_checkpoint),
        )
        
        # 输出卷积
        self.norm_out = normalization(ch)
        self.conv_out = nn.Conv3d(
            ch, 
            latent_dim * 2 if double_z else latent_dim,
            3, padding=1
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, C, D, H, W)
        Returns:
            如果 double_z: (mean, logvar)，各 (B, latent_dim, D', H', W')
            否则: (B, latent_dim, D', H', W')
        """
        h = self.conv_in(x)
        
        for block in self.down_blocks:
            for layer in block:
                h = layer(h)
        
        h = self.mid_block(h)
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        if self.double_z:
            mean, logvar = h.chunk(2, dim=1)
            return mean, logvar
        return h


# ==============================================================================
# VAE 解码器
# ==============================================================================

class VAE3DDecoder(nn.Module):
    """
    3D VAE 解码器
    
    将潜空间重建为原始体素
    例如: (B, latent_dim, 8, 32, 32) -> (B, 4, 32, 128, 128)
    
    Args:
        out_channels: 输出通道数
        latent_dim: 潜空间维度
        base_channels: 基础通道数
        channel_mult: 通道倍增系数（与编码器相反）
        num_res_blocks: 每层残差块数
        upsample_scales: 每层上采样因子
        use_attention: 是否使用注意力
        use_checkpoint: 是否使用梯度检查点
    """
    
    def __init__(
        self,
        out_channels: int = 4,
        latent_dim: int = 4,
        base_channels: int = 64,
        channel_mult: Tuple[int, ...] = (4, 2, 1),
        num_res_blocks: int = 2,
        upsample_scales: List[Tuple[int, int, int]] = None,
        use_attention: bool = True,
        dropout: float = 0.0,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        # 默认上采样因子
        if upsample_scales is None:
            upsample_scales = [(1, 2, 2), (2, 2, 2), (1, 2, 2)]
        
        # 初始通道数
        ch = base_channels * channel_mult[0]
        
        # 输入卷积
        self.conv_in = nn.Conv3d(latent_dim, ch, 3, padding=1)
        
        # 中间块
        self.mid_block = nn.Sequential(
            ResBlock3D(ch, ch, dropout, use_checkpoint=use_checkpoint),
            SelfAttention3D(ch) if use_attention else nn.Identity(),
            ResBlock3D(ch, ch, dropout, use_checkpoint=use_checkpoint),
        )
        
        # 上采样块
        self.up_blocks = nn.ModuleList()
        
        for i, mult in enumerate(channel_mult):
            block = nn.ModuleList()
            out_ch = base_channels * mult
            
            for _ in range(num_res_blocks + 1):
                block.append(ResBlock3D(ch, out_ch, dropout, use_checkpoint=use_checkpoint))
                ch = out_ch
            
            # 在第一层添加注意力
            if use_attention and i == 0:
                block.append(SelfAttention3D(ch))
            
            # 上采样（除了最后一层）
            if i < len(channel_mult) - 1:
                scale = upsample_scales[i] if i < len(upsample_scales) else (1, 2, 2)
                block.append(Upsample3D(ch, scale))
            
            self.up_blocks.append(block)
        
        # 输出卷积
        self.norm_out = normalization(ch)
        self.conv_out = nn.Conv3d(ch, out_channels, 3, padding=1)
    
    def forward(self, z):
        """
        Args:
            z: (B, latent_dim, D', H', W')
        Returns:
            (B, out_channels, D, H, W)
        """
        h = self.conv_in(z)
        h = self.mid_block(h)
        
        for block in self.up_blocks:
            for layer in block:
                h = layer(h)
        
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        return h


# ==============================================================================
# 完整 VAE 模型
# ==============================================================================

class VAE3D(nn.Module):
    """
    完整的 3D VAE 模型
    
    支持：
    - 标准 VAE（KL 散度正则化）
    - β-VAE（加权 KL）
    - 确定性编码（用于推理）
    - 梯度检查点（节省显存）
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        latent_dim: 潜空间维度
        base_channels: 基础通道数
        encoder_channel_mult: 编码器通道倍增
        decoder_channel_mult: 解码器通道倍增
        num_res_blocks: 残差块数
        use_attention: 是否使用注意力
        kl_weight: KL 散度权重 (β)
        use_checkpoint: 是否使用梯度检查点
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        latent_dim: int = 4,
        base_channels: int = 64,
        encoder_channel_mult: Tuple[int, ...] = (1, 2, 4),
        decoder_channel_mult: Tuple[int, ...] = (4, 2, 1),
        num_res_blocks: int = 2,
        use_attention: bool = True,
        kl_weight: float = 1e-6,
        dropout: float = 0.0,
        use_checkpoint: bool = True,  # 默认启用梯度检查点
    ):
        super().__init__()
        self.kl_weight = kl_weight
        self.latent_dim = latent_dim
        self.use_checkpoint = use_checkpoint
        
        self.encoder = VAE3DEncoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            base_channels=base_channels,
            channel_mult=encoder_channel_mult,
            num_res_blocks=num_res_blocks,
            use_attention=use_attention,
            double_z=True,
            dropout=dropout,
            use_checkpoint=use_checkpoint,
        )
        
        self.decoder = VAE3DDecoder(
            out_channels=out_channels,
            latent_dim=latent_dim,
            base_channels=base_channels,
            channel_mult=decoder_channel_mult,
            num_res_blocks=num_res_blocks,
            use_attention=use_attention,
            dropout=dropout,
            use_checkpoint=use_checkpoint,
        )
        
        # 量化层（用于 VQ-VAE）
        self.quant_conv = nn.Conv3d(latent_dim, latent_dim, 1)
        self.post_quant_conv = nn.Conv3d(latent_dim, latent_dim, 1)
    
    def encode(self, x, deterministic: bool = False):
        """
        编码到潜空间
        
        Args:
            x: (B, C, D, H, W) 输入
            deterministic: 是否确定性编码（不采样）
        Returns:
            z: (B, latent_dim, D', H', W') 潜向量
            posterior: 分布参数 (mean, logvar)
        """
        mean, logvar = self.encoder(x)
        
        # [Fix] 数值稳定性保护：截断 logvar 防止爆炸
        logvar = torch.clamp(logvar, -30.0, 20.0)
        
        if deterministic:
            z = mean
        else:
            # 重参数化采样
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mean + eps * std
        
        z = self.quant_conv(z)
        return z, (mean, logvar)
    
    def decode(self, z):
        """
        从潜空间解码
        
        Args:
            z: (B, latent_dim, D', H', W')
        Returns:
            x_recon: (B, C, D, H, W)
        """
        z = self.post_quant_conv(z)
        return self.decoder(z)
    
    def forward(self, x, sample_posterior: bool = True):
        """
        前向传播（编码 -> 解码）
        
        Args:
            x: (B, C, D, H, W) 输入
            sample_posterior: 是否从后验采样
        Returns:
            x_recon: 重建
            posterior: (mean, logvar)
        """
        z, posterior = self.encode(x, deterministic=not sample_posterior)
        x_recon = self.decode(z)
        return x_recon, posterior
    
    def get_latent(self, x, deterministic: bool = True):
        """获取潜向量（用于扩散模型训练）"""
        z, _ = self.encode(x, deterministic=deterministic)
        return z
    
    def compute_loss(self, x, x_recon, posterior, reduction: str = "mean"):
        """
        计算 VAE 损失
        
        Args:
            x: 原始输入
            x_recon: 重建
            posterior: (mean, logvar)
            reduction: 损失聚合方式
        Returns:
            total_loss: 总损失
            recon_loss: 重建损失
            kl_loss: KL 散度损失
        """
        mean, logvar = posterior
        
        # 重建损失 (L1 或 L2)
        recon_loss = F.mse_loss(x_recon, x, reduction=reduction)
        
        # KL 散度损失
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        
        # 总损失
        total_loss = recon_loss + self.kl_weight * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    @staticmethod
    def get_input_size_from_latent(latent_size, downsample_factor=(4, 4, 4)):
        """根据潜空间大小计算输入大小"""
        d, h, w = latent_size
        return (
            d * downsample_factor[0],
            h * downsample_factor[1],
            w * downsample_factor[2],
        )
    
    @staticmethod
    def get_latent_size_from_input(input_size, downsample_factor=(4, 4, 4)):
        """根据输入大小计算潜空间大小"""
        d, h, w = input_size
        return (
            d // downsample_factor[0],
            h // downsample_factor[1],
            w // downsample_factor[2],
        )


# ==============================================================================
# VQ-VAE（向量量化变分自编码器）
# ==============================================================================

class VectorQuantizer(nn.Module):
    """
    向量量化器
    
    将连续潜向量离散化为码本中的向量
    """
    
    def __init__(
        self,
        num_embeddings: int = 8192,
        embedding_dim: int = 4,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        
        # 码本
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
        
        # EMA 更新用
        self.register_buffer('cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('embed_avg', self.embedding.weight.data.clone())
    
    def forward(self, z):
        """
        Args:
            z: (B, C, D, H, W) 连续潜向量
        Returns:
            z_q: (B, C, D, H, W) 量化后的潜向量
            loss: 量化损失
            indices: (B, D, H, W) 码本索引
        """
        # 重塑为 (B*D*H*W, C)
        B, C, D, H, W = z.shape
        z_flat = rearrange(z, 'b c d h w -> (b d h w) c')
        
        # 计算距离
        distances = (
            z_flat.pow(2).sum(1, keepdim=True)
            - 2 * z_flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(1, keepdim=True).t()
        )
        
        # 找到最近的码本向量
        indices = distances.argmin(dim=1)
        z_q_flat = self.embedding(indices)
        
        # 重塑回原始形状
        z_q = rearrange(z_q_flat, '(b d h w) c -> b c d h w', b=B, d=D, h=H, w=W)
        indices = rearrange(indices, '(b d h w) -> b d h w', b=B, d=D, h=H, w=W)
        
        # 计算损失
        commitment_loss = F.mse_loss(z_q.detach(), z)
        codebook_loss = F.mse_loss(z_q, z.detach())
        loss = codebook_loss + self.commitment_cost * commitment_loss
        
        # Straight-through estimator
        z_q = z + (z_q - z).detach()
        
        return z_q, loss, indices


class VQVAE3D(nn.Module):
    """
    3D VQ-VAE
    
    使用向量量化的变分自编码器
    优点：不需要 KL 散度，码本可解释
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        latent_dim: int = 4,
        num_embeddings: int = 8192,
        base_channels: int = 64,
        encoder_channel_mult: Tuple[int, ...] = (1, 2, 4),
        decoder_channel_mult: Tuple[int, ...] = (4, 2, 1),
        num_res_blocks: int = 2,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        
        self.encoder = VAE3DEncoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            base_channels=base_channels,
            channel_mult=encoder_channel_mult,
            num_res_blocks=num_res_blocks,
            double_z=False,  # VQ-VAE 不需要 double_z
        )
        
        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=latent_dim,
            commitment_cost=commitment_cost,
        )
        
        self.decoder = VAE3DDecoder(
            out_channels=out_channels,
            latent_dim=latent_dim,
            base_channels=base_channels,
            channel_mult=decoder_channel_mult,
            num_res_blocks=num_res_blocks,
        )
    
    def encode(self, x):
        z = self.encoder(x)
        z_q, loss, indices = self.quantizer(z)
        return z_q, loss, indices
    
    def decode(self, z_q):
        return self.decoder(z_q)
    
    def forward(self, x):
        z_q, vq_loss, indices = self.encode(x)
        x_recon = self.decode(z_q)
        return x_recon, vq_loss, indices
    
    def get_latent(self, x):
        """获取量化后的潜向量"""
        z = self.encoder(x)
        z_q, _, _ = self.quantizer(z)
        return z_q


# ==============================================================================
# 预设配置
# ==============================================================================

def create_ultra_lightweight_vae_config():
    """超轻量级 VAE 配置 - 极限显存优化 (适用于 <16GB GPU)"""
    return {
        "in_channels": 4,
        "out_channels": 4,
        "latent_dim": 4,
        "base_channels": 16,  # 极小基础通道
        "encoder_channel_mult": (1, 2, 2),  # 减少通道倍增
        "decoder_channel_mult": (2, 2, 1),
        "num_res_blocks": 1,
        "use_attention": False,
        "kl_weight": 1e-6,
        "use_checkpoint": True,
    }


def create_lightweight_vae_config():
    """轻量级 VAE 配置 - 针对显存优化 (适用于 16-24GB GPU)"""
    return {
        "in_channels": 4,
        "out_channels": 4,
        "latent_dim": 4,
        "base_channels": 24,  # 减小基础通道 (32->24)
        "encoder_channel_mult": (1, 2, 3),  # 减少最后一层 (4->3)
        "decoder_channel_mult": (3, 2, 1),
        "num_res_blocks": 1,
        "use_attention": False,
        "kl_weight": 1e-6,
        "use_checkpoint": True,
    }


def create_standard_vae_config():
    """标准 VAE 配置"""
    return {
        "in_channels": 4,
        "out_channels": 4,
        "latent_dim": 4,
        "base_channels": 64,
        "encoder_channel_mult": (1, 2, 4),
        "decoder_channel_mult": (4, 2, 1),
        "num_res_blocks": 2,
        "use_attention": True,
        "kl_weight": 1e-6,
        "use_checkpoint": True,  # 启用梯度检查点
    }
