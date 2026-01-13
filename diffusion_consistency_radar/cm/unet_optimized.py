# -- coding: utf-8 --

"""
优化版 3D UNet 模型
整合所有优化策略：
1. 多种注意力机制选择 (Flash/Window/Linear/Sparse)
2. 非对称下采样 (保护Z轴分辨率)
3. 通道瘦身配置
4. 多种归一化选择
5. 梯度检查点支持
"""

from abc import abstractmethod
import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
    set_norm_type,
)
from .attention_optimized import create_attention_block
from .sampling_optimized import (
    AsymmetricDownsample3D,
    AsymmetricUpsample3D,
    AdaptiveDownsampleScheduler,
    create_downsample_block,
    create_upsample_block,
)


class TimestepBlock(nn.Module):
    """任何接受时间步嵌入作为第二个参数的模块"""
    @abstractmethod
    def forward(self, x, emb):
        pass


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """顺序模块，自动传递时间步嵌入到支持的子模块"""
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class OptimizedResBlock(TimestepBlock):
    """
    优化的残差块
    
    改进:
    - 支持多种归一化
    - 支持非对称上下采样
    - 深度可分离卷积选项（减少参数）
    - Scale-Shift Norm 用于条件注入
    """
    
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float,
        out_channels: Optional[int] = None,
        use_conv: bool = False,
        use_scale_shift_norm: bool = True,
        dims: int = 3,
        use_checkpoint: bool = False,
        up: bool = False,
        down: bool = False,
        norm_type: str = "group",
        downsample_type: str = "asymmetric",
        stride: Union[str, Tuple[int, int, int]] = "xy_only",
        use_depthwise: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.dims = dims
        
        # 输入层
        if use_depthwise and dims == 3:
            # 深度可分离卷积：减少参数量
            self.in_layers = nn.Sequential(
                normalization(channels, norm_type),
                nn.SiLU(),
                nn.Conv3d(channels, channels, 3, padding=1, groups=channels),  # Depthwise
                nn.Conv3d(channels, self.out_channels, 1),  # Pointwise
            )
        else:
            self.in_layers = nn.Sequential(
                normalization(channels, norm_type),
                nn.SiLU(),
                conv_nd(dims, channels, self.out_channels, 3, padding=1),
            )
        
        # 上下采样
        self.updown = up or down
        if up:
            self.h_upd = create_upsample_block(
                channels, channels, dims, use_conv=False,
                upsample_type="asymmetric", scale_factor=stride
            )
            self.x_upd = create_upsample_block(
                channels, channels, dims, use_conv=False,
                upsample_type="asymmetric", scale_factor=stride
            )
        elif down:
            self.h_upd = create_downsample_block(
                channels, channels, dims, use_conv=False,
                downsample_type=downsample_type, stride=stride
            )
            self.x_upd = create_downsample_block(
                channels, channels, dims, use_conv=False,
                downsample_type=downsample_type, stride=stride
            )
        else:
            self.h_upd = self.x_upd = nn.Identity()
        
        # 时间步嵌入层
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        
        # 输出层
        self.out_layers = nn.Sequential(
            normalization(self.out_channels, norm_type),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
        )
        
        # 跳跃连接
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)
    
    def forward(self, x, emb):
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )
    
    def _forward(self, x, emb):
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


class OptimizedUNetModel(nn.Module):
    """
    优化的 3D UNet 模型
    
    主要优化:
    1. 高效注意力机制 (Flash/Window/Linear/Sparse)
    2. 非对称下采样策略 (保护Z轴)
    3. 通道瘦身支持
    4. 多种归一化选项
    5. 深度可分离卷积选项
    
    Args:
        image_size: 输入尺寸
        in_channels: 输入通道数
        model_channels: 基础通道数
        out_channels: 输出通道数
        num_res_blocks: 每层残差块数
        attention_resolutions: 使用注意力的分辨率
        dropout: Dropout率
        channel_mult: 通道倍增系数
        conv_resample: 是否使用卷积重采样
        dims: 维度 (2D/3D)
        num_classes: 类别数（条件生成）
        use_checkpoint: 梯度检查点
        use_fp16: 混合精度
        num_heads: 注意力头数
        num_head_channels: 每头通道数
        attention_type: 注意力类型 ("flash", "window", "linear", "sparse", "none")
        norm_type: 归一化类型 ("group", "layer", "instance", "rms")
        downsample_type: 下采样类型 ("asymmetric", "standard")
        downsample_stride: 下采样步长策略
        use_depthwise: 是否使用深度可分离卷积
        window_size: 窗口注意力的窗口大小
    """
    
    def __init__(
        self,
        image_size: int,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int,
        attention_resolutions: Tuple[int, ...],
        dropout: float = 0.0,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
        conv_resample: bool = True,
        dims: int = 3,
        num_classes: Optional[int] = None,
        use_checkpoint: bool = False,
        use_fp16: bool = False,
        num_heads: int = 4,
        num_head_channels: int = -1,
        num_heads_upsample: int = -1,
        use_scale_shift_norm: bool = True,
        resblock_updown: bool = False,
        use_new_attention_order: bool = False,
        # === 新增优化参数 ===
        attention_type: str = "flash",  # 注意力类型
        norm_type: str = "group",  # 归一化类型
        downsample_type: str = "asymmetric",  # 下采样类型
        downsample_stride: str = "xy_only",  # 下采样步长
        use_depthwise: bool = False,  # 深度可分离卷积
        window_size: Tuple[int, int, int] = (4, 4, 4),  # 窗口大小
        initial_z_size: int = 32,  # 初始Z轴大小（用于自适应下采样）
    ):
        super().__init__()
        
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        
        # 设置全局归一化类型
        set_norm_type(norm_type)
        
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
        self.attention_type = attention_type
        self.norm_type = norm_type
        self.downsample_type = downsample_type
        self.downsample_stride = downsample_stride
        self.window_size = window_size
        
        # 自适应下采样调度
        self.downsample_scheduler = AdaptiveDownsampleScheduler(
            initial_size=(initial_z_size, image_size, image_size),
            num_levels=len(channel_mult) - 1,
            min_z=4,
        )
        
        print(f"[OptimizedUNet] Config:")
        print(f"  - model_channels: {model_channels}")
        print(f"  - channel_mult: {channel_mult}")
        print(f"  - attention_type: {attention_type}")
        print(f"  - norm_type: {norm_type}")
        print(f"  - downsample_type: {downsample_type}")
        print(f"  - downsample_stride: {downsample_stride}")
        self.downsample_scheduler.print_schedule()
        
        # 时间步嵌入
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        
        # 类别嵌入
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)
        
        # 输入块
        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))
        ])
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers: List[nn.Module] = [
                    OptimizedResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        norm_type=norm_type,
                        use_depthwise=use_depthwise,
                    )
                ]
                ch = int(mult * model_channels)
                
                # 添加注意力层（根据分辨率）
                if ds in attention_resolutions:
                    # 计算头数
                    if num_head_channels == -1:
                        n_heads = num_heads
                    else:
                        n_heads = ch // num_head_channels
                    
                    layers.append(
                        create_attention_block(
                            channels=ch,
                            num_heads=n_heads,
                            attention_type=attention_type,
                            window_size=window_size,
                            use_checkpoint=use_checkpoint,
                        )
                    )
                
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            
            # 下采样
            if level != len(channel_mult) - 1:
                out_ch = ch
                stride = self.downsample_scheduler.get_stride(level)
                
                if resblock_updown:
                    self.input_blocks.append(
                        TimestepEmbedSequential(
                            OptimizedResBlock(
                                ch,
                                time_embed_dim,
                                dropout,
                                out_channels=out_ch,
                                dims=dims,
                                use_checkpoint=use_checkpoint,
                                use_scale_shift_norm=use_scale_shift_norm,
                                down=True,
                                norm_type=norm_type,
                                downsample_type=downsample_type,
                                stride=stride,
                            )
                        )
                    )
                else:
                    self.input_blocks.append(
                        TimestepEmbedSequential(
                            create_downsample_block(
                                ch, out_ch, dims, conv_resample,
                                downsample_type=downsample_type,
                                stride=stride,
                            )
                        )
                    )
                
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch
        
        # 中间块
        self.middle_block = TimestepEmbedSequential(
            OptimizedResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                norm_type=norm_type,
            ),
            # 中间块的注意力（可选）
            create_attention_block(
                channels=ch,
                num_heads=num_heads if num_head_channels == -1 else ch // num_head_channels,
                attention_type=attention_type if attention_type != "none" else "linear",  # 中间块至少用linear attention
                window_size=window_size,
                use_checkpoint=use_checkpoint,
            ) if attention_type != "none" else nn.Identity(),
            OptimizedResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                norm_type=norm_type,
            ),
        )
        self._feature_size += ch
        
        # 输出块
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    OptimizedResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        norm_type=norm_type,
                    )
                ]
                ch = int(model_channels * mult)
                
                # 添加注意力层
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        n_heads = num_heads_upsample
                    else:
                        n_heads = ch // num_head_channels
                    
                    layers.append(
                        create_attention_block(
                            channels=ch,
                            num_heads=n_heads,
                            attention_type=attention_type,
                            window_size=window_size,
                            use_checkpoint=use_checkpoint,
                        )
                    )
                
                # 上采样
                if level and i == num_res_blocks:
                    out_ch = ch
                    scale = self.downsample_scheduler.get_scale_factor(level - 1)
                    
                    if resblock_updown:
                        layers.append(
                            OptimizedResBlock(
                                ch,
                                time_embed_dim,
                                dropout,
                                out_channels=out_ch,
                                dims=dims,
                                use_checkpoint=use_checkpoint,
                                use_scale_shift_norm=use_scale_shift_norm,
                                up=True,
                                norm_type=norm_type,
                                stride=scale,
                            )
                        )
                    else:
                        layers.append(
                            create_upsample_block(
                                ch, out_ch, dims, conv_resample,
                                upsample_type="asymmetric",
                                scale_factor=scale,
                            )
                        )
                    ds //= 2
                
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
        
        # 输出层
        self.out = nn.Sequential(
            normalization(ch, norm_type),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )
    
    def convert_to_fp16(self):
        """转换为 FP16"""
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)
    
    def convert_to_fp32(self):
        """转换为 FP32"""
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)
    
    def forward(self, x, timesteps, y=None):
        """
        Args:
            x: (B, C, D, H, W) 输入数据
            timesteps: (B,) 时间步
            y: (B, C_cond, D, H, W) 可选的条件输入
        """
        # 条件拼接
        if y is not None:
            x = th.cat([x, y], dim=1)
        
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        
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


# ==============================================================================
# 轻量级 UNet 配置预设
# ==============================================================================

def create_lightweight_unet_config():
    """
    创建轻量级配置，适合 24GB 显存
    
    相比默认配置:
    - 通道数减半 (128 -> 64)
    - 通道倍增更保守 (1,2,3,4 -> 1,2,2,4)
    - 使用 Flash Attention
    - 非对称下采样
    """
    return {
        "model_channels": 64,
        "channel_mult": (1, 2, 2, 4),
        "num_res_blocks": 2,
        "attention_type": "flash",
        "attention_resolutions": (8, 4),  # 只在低分辨率使用注意力
        "norm_type": "group",
        "downsample_type": "asymmetric",
        "downsample_stride": "xy_only",
        "use_checkpoint": True,
        "use_fp16": True,
        "num_heads": 4,
        "dropout": 0.1,
    }


def create_ultra_lightweight_unet_config():
    """
    超轻量级配置，适合 12GB 显存或更大 batch size
    """
    return {
        "model_channels": 48,
        "channel_mult": (1, 2, 2, 2),
        "num_res_blocks": 1,
        "attention_type": "linear",  # 线性注意力，最省显存
        "attention_resolutions": (4,),  # 只在最低分辨率使用
        "norm_type": "instance",  # InstanceNorm 更适合稀疏数据
        "downsample_type": "asymmetric",
        "downsample_stride": "xy_only",
        "use_checkpoint": True,
        "use_fp16": True,
        "use_depthwise": True,  # 深度可分离卷积
        "num_heads": 2,
        "dropout": 0.05,
    }


def create_balanced_unet_config():
    """
    平衡配置，在效果和效率之间取得平衡
    """
    return {
        "model_channels": 64,
        "channel_mult": (1, 2, 3, 4),
        "num_res_blocks": 2,
        "attention_type": "window",  # 窗口注意力
        "attention_resolutions": (16, 8, 4),
        "norm_type": "group",
        "downsample_type": "asymmetric",
        "downsample_stride": "adaptive",  # 自适应下采样
        "use_checkpoint": True,
        "use_fp16": True,
        "num_heads": 4,
        "window_size": (4, 4, 4),
        "dropout": 0.1,
    }
