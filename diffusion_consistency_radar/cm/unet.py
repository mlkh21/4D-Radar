from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)



class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        输入:
            x: 输入张量。
            emb: 时间步嵌入。
        输出:
            输出张量。
        作用: 抽象方法，定义模块的前向传播，接受时间步嵌入作为第二个参数。
        逻辑:
        子类需要实现此方法。
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        """
        输入:
            x: 输入张量。
            emb: 时间步嵌入。
        输出:
            x: 输出张量。
        作用: 顺序执行子模块，如果子模块支持时间步嵌入，则传入。
        逻辑:
        1. 遍历所有层。
        2. 如果层是 TimestepBlock 的实例，传入 x 和 emb。
        3. 否则，只传入 x。
        4. 返回最终结果。
        """
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

   channels: channels in the inputs and outputs.
   use_conv: a bool determining if a convolution is applied.
   dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        """
        输入:
            channels: 输入和输出的通道数。
            use_conv: 是否应用卷积。
            dims: 信号维度 (1D, 2D, 3D)。
            out_channels: 输出通道数 (可选)。
        输出:
            无
        作用: 初始化上采样层。
        逻辑:
        1. 保存参数。
        2. 如果 use_conv 为 True，初始化卷积层。
        """
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        """
        输入:
            x: 输入张量。
        输出:
            x: 上采样后的张量。
        作用: 前向传播。
        逻辑:
        1. 检查输入通道数。
        2. 根据 dims 进行插值上采样。
        3. 如果 use_conv 为 True，应用卷积。
        4. 返回结果。
        """
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

   channels: channels in the inputs and outputs.
   use_conv: a bool determining if a convolution is applied.
   dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        """
        输入:
            channels: 输入和输出的通道数。
            use_conv: 是否应用卷积。
            dims: 信号维度 (1D, 2D, 3D)。
            out_channels: 输出通道数 (可选)。
        输出:
            无
        作用: 初始化下采样层。
        逻辑:
        1. 保存参数。
        2. 计算步长。
        3. 如果 use_conv 为 True，初始化卷积层。
        4. 否则，初始化平均池化层。
        """
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        """
        输入:
            x: 输入张量。
        输出:
            x: 下采样后的张量。
        作用: 前向传播。
        逻辑:
        1. 检查输入通道数。
        2. 应用下采样操作 (卷积或池化)。
        3. 返回结果。
        """
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

   channels: the number of input channels.
   emb_channels: the number of timestep embedding channels.
   dropout: the rate of dropout.
   out_channels: if specified, the number of out channels.
   use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
   dims: determines if the signal is 1D, 2D, or 3D.
   use_checkpoint: if True, use gradient checkpointing on this module.
   up: if True, use this block for upsampling.
   down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        """
        输入:
            channels: 输入通道数。
            emb_channels: 时间步嵌入通道数。
            dropout: dropout 率。
            out_channels: 输出通道数 (可选)。
            use_conv: 是否使用卷积改变通道数。
            use_scale_shift_norm: 是否使用 scale-shift 归一化。
            dims: 信号维度 (1D, 2D, 3D)。
            use_checkpoint: 是否使用梯度检查点。
            up: 是否用于上采样。
            down: 是否用于下采样。
        输出:
            无
        作用: 初始化残差块。
        逻辑:
        1. 保存参数。
        2. 初始化输入层 (归一化, SiLU, 卷积)。
        3. 初始化上/下采样层。
        4. 初始化嵌入层 (SiLU, 线性)。
        5. 初始化输出层 (归一化, SiLU, Dropout, 零卷积)。
        6. 初始化跳跃连接。
        """
        super().__init__()
        self.channels = channels
        # print("channels", channels)
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        输入:
            x: 输入张量，形状为 [N x C x ...]。
            emb: 时间步嵌入张量，形状为 [N x emb_channels]。
        输出:
            输出张量，形状为 [N x C x ...]。
        作用: 前向传播，应用残差块。
        逻辑:
        使用 checkpoint 机制调用 _forward 方法。
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        """
        输入:
            x: 输入张量。
            emb: 时间步嵌入。
        输出:
            输出张量。
        作用: 实际的前向传播逻辑。
        逻辑:
        1. 处理输入层 (可能包含上/下采样)。
        2. 处理时间步嵌入。
        3. 将嵌入加到特征上 (或使用 scale-shift)。
        4. 处理输出层。
        5. 添加跳跃连接。
        """
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        attention_type="false",  ###zrb failed to install flashattention
        encoder_channels=None,
        dims=2,
        channels_last=False,
        use_new_attention_order=False,
    ):
        """
        输入:
            channels: 输入通道数。
            num_heads: 注意力头数。
            num_head_channels: 每个头的通道数。
            use_checkpoint: 是否使用梯度检查点。
            attention_type: 注意力类型 ("flash" 或其他)。
            encoder_channels: 编码器通道数 (可选)。
            dims: 信号维度。
            channels_last: 是否通道在最后。
            use_new_attention_order: 是否使用新的注意力顺序。
        输出:
            无
        作用: 初始化注意力块。
        逻辑:
        1. 计算头数。
        2. 初始化归一化层。
        3. 初始化 QKV 卷积层。
        4. 根据 attention_type 初始化注意力模块 (FlashAttention 或 QKVAttentionLegacy)。
        5. 初始化输出投影层。
        """
        super().__init__()
        self.channels = channels
        # print("num_heads", num_heads)
        # print("num_head_channels", num_head_channels)

        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(dims, channels, channels * 3, 1)
        self.attention_type = attention_type
        if attention_type == "flash":
            self.attention = QKVFlashAttention(channels, self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.use_attention_checkpoint = not (
            self.use_checkpoint or self.attention_type == "flash"
        )
        if encoder_channels is not None:
            assert attention_type != "flash"
            self.encoder_kv = conv_nd(1, encoder_channels, channels * 2, 1)
        self.proj_out = zero_module(conv_nd(dims, channels, channels, 1))

    def forward(self, x, encoder_out=None):
        """
        输入:
            x: 输入张量。
            encoder_out: 编码器输出 (可选)。
        输出:
            输出张量。
        作用: 前向传播。
        逻辑:
        使用 checkpoint 机制调用 _forward 方法。
        """
        if encoder_out is None:
            return checkpoint(
                self._forward, (x,), self.parameters(), self.use_checkpoint
            )
        else:
            return checkpoint(
                self._forward, (x, encoder_out), self.parameters(), self.use_checkpoint
            )

    def _forward(self, x, encoder_out=None):
        """
        输入:
            x: 输入张量。
            encoder_out: 编码器输出 (可选)。
        输出:
            输出张量。
        作用: 实际的前向传播逻辑。
        逻辑:
        1. 计算 QKV。
        2. 如果有 encoder_out，计算 encoder_kv 并进行交叉注意力。
        3. 否则，进行自注意力。
        4. 投影输出。
        5. 添加残差连接。
        """
        b, _, *spatial = x.shape
        qkv = self.qkv(self.norm(x)).view(b, -1, np.prod(spatial))
        if encoder_out is not None:
            encoder_out = self.encoder_kv(encoder_out)
            h = checkpoint(
                self.attention, (qkv, encoder_out), (), self.use_attention_checkpoint
            )
        else:
            h = checkpoint(self.attention, (qkv,), (), self.use_attention_checkpoint)
        h = h.view(b, -1, *spatial)
        h = self.proj_out(h)
        return x + h


class QKVFlashAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        batch_first=True,
        attention_dropout=0.0,
        causal=False,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        """
        输入:
            embed_dim: 嵌入维度。
            num_heads: 注意力头数。
            batch_first: batch 是否在第一维。
            attention_dropout: 注意力 dropout 率。
            causal: 是否使用因果掩码。
            device: 设备。
            dtype: 数据类型。
        输出:
            无
        作用: 初始化 FlashAttention 模块。
        逻辑:
        1. 检查参数。
        2. 初始化 FlashAttention。
        """
        from einops import rearrange
        from flash_attn.flash_attention import FlashAttention

        # print("asdf")
        assert batch_first
        # factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.causal = causal

        # print("self.embed_dim", self.embed_dim)
        # print("num_heads", num_heads)
        assert (
            self.embed_dim % num_heads == 0
        ), "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        print("self.embed_dim", self.embed_dim)
        print("self.num_heads", self.num_heads)
        
        assert self.head_dim in [16, 32, 64], "Only support head_dim == 16, 32, or 64"

        self.inner_attn = FlashAttention(
            attention_dropout=attention_dropout, # **factory_kwargs
        )
        self.rearrange = rearrange

    def forward(self, qkv, attn_mask=None, key_padding_mask=None, need_weights=False):
        """
        输入:
            qkv: QKV 张量。
            attn_mask: 注意力掩码。
            key_padding_mask: 键填充掩码。
            need_weights: 是否需要权重。
        输出:
            输出张量。
        作用: 前向传播。
        逻辑:
        1. 重排 QKV 张量。
        2. 调用 FlashAttention。
        3. 重排输出张量。
        """
        qkv = self.rearrange(
            qkv, "b (three h d) s -> b s three h d", three=3, h=self.num_heads
        )
        #zrb
        # qkv.to(th.float16)
        qkv, _ = self.inner_attn(
            qkv.contiguous().to(th.float16),
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            causal=self.causal,
        )
        qkv = qkv.to(th.float32)
        # print("qkv", qkv.dtype)
        #zrb warning float16_32
        # qkv.to(th.float16)
        return self.rearrange(qkv, "b s h d -> b (h d) s")


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial**2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    执行 QKV 注意力的模块。匹配旧版 QKVAttention + 输入/输出头整形。
    """

    def __init__(self, n_heads):
        """
        输入:
            n_heads: 注意力头数。
        输出:
            无
        作用: 初始化 QKVAttentionLegacy 模块。
        逻辑:
        保存头数。
        """
        super().__init__()
        self.n_heads = n_heads
        from einops import rearrange
        self.rearrange = rearrange


    def forward(self, qkv):
        """
        输入:
            qkv: QKV 张量，形状为 [N x (H * 3 * C) x T]。
        输出:
            输出张量，形状为 [N x (H * C) x T]。
        作用: 应用 QKV 注意力。
        逻辑:
        1. 分割 Q, K, V。
        2. 计算注意力权重。
        3. 应用注意力权重到 V。
        4. 返回结果。
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        qkv = qkv.half()

        qkv =   self.rearrange(
            qkv, "b (three h d) s -> b s three h d", three=3, h=self.n_heads
        ) 
        q, k, v = qkv.transpose(1, 3).transpose(3, 4).split(1, dim=2)
        q = q.reshape(bs*self.n_heads, ch, length)
        k = k.reshape(bs*self.n_heads, ch, length)
        v = v.reshape(bs*self.n_heads, ch, length)

        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight, dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        a = a.float()
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class RadarHeightSelfAttention(nn.Module):
    """
    参考 MetaOcc 的 RHS 模块
    4D 毫米波雷达分支：雷达高度自注意力（RHS）
    """
    def __init__(self, in_channels, out_channels, z_dim, num_heads=4):
        super().__init__()
        self.z_dim = z_dim
        self.out_channels = out_channels
        
        # 1. PointPillars Transformation (Simplified: Conv2d to process BEV input)
        # 注意：这里假设输入已经是 BEV 特征（例如通过 PointPillars 预处理得到，或者输入是 BEV 图像）
        # 如果输入是原始点云，需要先经过 Voxel Feature Encoder (VFE) 和 Scatter 操作
        self.bev_encoder = nn.Sequential(
            conv_nd(2, in_channels, out_channels, 3, padding=1),
            normalization(out_channels),
            nn.SiLU() # 使用 SiLU 替代 ReLU 以保持一致性
        )
        
        # 2. Height Positional Encoding (核心创新)
        # 学习高度分布：引入可学习的高度位置编码 Ph
        # Shape: (1, C, Z, 1, 1) 用于广播
        self.height_pos_enc = nn.Parameter(th.randn(1, out_channels, z_dim, 1, 1) * 0.02)
        
        # 3. Self Attention (along Z axis)
        # 通过计算自注意力权重，自适应地调整垂直方向上的注意力分配
        self.attention = nn.MultiheadAttention(embed_dim=out_channels, num_heads=num_heads, batch_first=True)
        
        # 4. Radar Encoder (3D Conv with Softplus)
        # 带有 Softplus 激活函数的多层 3D 卷积编码器
        self.radar_3d_encoder = nn.Sequential(
            conv_nd(3, out_channels, out_channels, 3, padding=1),
            nn.Softplus(),
            conv_nd(3, out_channels, out_channels, 3, padding=1),
            nn.Softplus()
        )

    def forward(self, x):
        """
        Args:
            x: BEV features (B, C_in, H, W)
        Returns:
            out: 3D Radar Features (B, C_out, Z, H, W)
        """
        # 1. PointPillars / BEV Encoding
        f_bev = self.bev_encoder(x) # (B, C_out, H, W)
        
        # 2. Height Dimension Expansion
        # 将 BEV 特征沿高度轴（Z 轴）重复 Z 次
        # (B, C, H, W) -> (B, C, Z, H, W)
        f_ini = f_bev.unsqueeze(2).repeat(1, 1, self.z_dim, 1, 1)
        
        # 3. Add Height Positional Encoding
        f_ini = f_ini + self.height_pos_enc
        
        # 4. Self Attention along Z
        b, c, z, h, w = f_ini.shape
        # Reshape to (Batch_Size * H * W, Z, C) for Attention
        # Permute: (B, C, Z, H, W) -> (B, H, W, Z, C) -> (B*H*W, Z, C)
        f_flat = f_ini.permute(0, 3, 4, 2, 1).reshape(-1, z, c)
        
        # Self-attention
        # attn_output: (B*H*W, Z, C)
        f_att, _ = self.attention(f_flat, f_flat, f_flat)
        
        # Reshape back to (B, C, Z, H, W)
        f_att = f_att.reshape(b, h, w, z, c).permute(0, 4, 3, 1, 2)
        
        # Residual Connection: 结合初始特征与注意力特征
        f_combined = f_ini + f_att
        
        # 5. 3D Encoder & Activation
        f_out = self.radar_3d_encoder(f_combined)
        
        return f_out


class UNetModel(nn.Module):
    """
        该模型基于 OpenAI 的 Guided Diffusion 代码库中的 UNet 实现，支持多种配置，
        包括 1D/2D/3D 数据处理、残差块、注意力机制、Dropout 以及类条件生成。

        输入：
        image_size: 输入数据的空间尺寸（例如图像的高/宽）。
        model_channels: 模型的基础通道数（即第一层卷积后的特征图通道数）。
        num_res_blocks: 每个下采样阶段（分辨率级别）的残差块数量。
        attention_resolutions: 使用注意力机制的分辨率集合（下采样率）。
            可以是集合、列表或元组。例如，如果包含 4，则在 4 倍下采样（分辨率为 image_size/4）时使用注意力层。
        dropout: Dropout 概率，用于防止过拟合。
        channel_mult: UNet 每个层级的通道倍增系数元组。例如 (1, 2, 4, 8) 表示通道数依次为 model_channels * 1, * 2, * 4, * 8。
        conv_resample: 如果为 True，则使用学习到的卷积层进行上采样和下采样；否则使用插值。
        dims: 数据的空间维度。1 表示 1D 信号，2 表示图像，3 表示体积数据。
        num_classes: 如果指定（作为整数），则此模型将是具有 `num_classes` 个类别的类条件模型，会初始化标签嵌入层。
        use_checkpoint: 如果为 True，使用梯度检查点（gradient checkpointing）以减少显存使用，但会增加计算时间。
        use_fp16: 如果为 True，将模型内部计算转换为 float16 以节省显存。
        num_head_channels: 如果指定（不为 -1），则忽略 `num_heads`，改为强制每个注意力头具有固定的通道宽度。
        num_heads_upsample: 与 `num_heads` 配合使用，专门为上采样阶段设置不同数量的头。如果为 -1，则默认等于 `num_heads`。已弃用。
        use_scale_shift_norm: 如果为 True，在残差块中使用类似 FiLM 的调节机制（Scale-Shift Normalization）。
        resblock_updown: 如果为 True，使用残差块进行上/下采样操作，而不是单独的采样层。
        use_new_attention_order: 如果为 True，使用不同的注意力计算顺序（先投影再拆分头），可能提高效率。
        输出:
            无
        作用: 初始化 UNet 模型。
        逻辑:
        1. 保存参数。
        2. 初始化时间步嵌入层。
        3. 初始化输入块 (卷积, 残差块, 注意力块, 下采样)。
        4. 初始化中间块 (残差块, 注意力块, 残差块)。
        5. 初始化输出块 (残差块, 注意力块, 上采样)。
        6. 初始化输出层 (归一化, SiLU, 卷积)。
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
    ):

        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        print("model_channels", model_channels)
        print("channel_mult", channel_mult)
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers: list = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            # AttentionBlock(
            #     ch,
            #     use_checkpoint=use_checkpoint,
            #     num_heads=num_heads,
            #     num_head_channels=num_head_channels,
            #     use_new_attention_order=use_new_attention_order,
            # ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        输入:
            无
        输出:
            无
        作用: 将模型主体转换为 float16。
        逻辑:
        遍历输入块、中间块和输出块，应用转换函数。
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        输入:
            无
        输出:
            无
        作用: 将模型主体转换为 float32。
        逻辑:
        遍历输入块、中间块和输出块，应用转换函数。
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps, y=None):
        """
        输入:
            x: 输入张量，形状为 [N x C x ...]。
            timesteps: 时间步张量，形状为 [N]。
            y: 标签张量 (可选)，形状为 [N]。
        输出:
            输出张量，形状为 [N x C x ...]。
        作用: 前向传播。
        逻辑:
        1. 如果有 y，拼接到 x。
        2. 计算时间步嵌入。
        3. 通过输入块，保存中间特征。
        4. 通过中间块。
        5. 通过输出块，拼接中间特征。
        6. 通过输出层。
        """
        # print("asdfasfd")
        # assert (y is not None) == (
        #     self.num_classes is not None
        # ), "must specify y if and only if the model is class-conditional"

        # print("y", y.shape)
        # print("x", x.shape)



        # zrb for radar-lidar concate
        if y is not None:
            x = th.cat([x, y], dim=1)
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        # if self.num_classes is not None:
        #     assert y.shape == (x.shape[0],)
        #     emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)
