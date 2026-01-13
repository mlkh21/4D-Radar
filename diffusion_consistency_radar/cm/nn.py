"""
Various utilities for neural networks.

优化版本 - 添加多种 Normalization 选项
- GroupNorm (默认，适合小 batch size)
- LayerNorm3D (逐样本归一化)
- InstanceNorm3D (适合点云/稀疏数据)
- RMSNorm (更高效的归一化)
"""

import math

import torch as th
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


# ==============================================================================
# 全局配置 - 可通过环境变量或配置文件修改
# ==============================================================================
NORM_TYPE = "group"  # 可选: "group", "layer", "instance", "rms", "batch"
NORM_GROUPS = 32     # GroupNorm 的组数
NORM_EPS = 1e-6      # 归一化的 epsilon


def set_norm_type(norm_type: str):
    """设置全局归一化类型"""
    global NORM_TYPE
    NORM_TYPE = norm_type


def set_norm_groups(num_groups: int):
    """设置 GroupNorm 的组数"""
    global NORM_GROUPS
    NORM_GROUPS = num_groups


# ==============================================================================
# 激活函数
# ==============================================================================

# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    def forward(self, x):
        """
        输入:
            x: 输入张量。
        输出:
            输出张量。
        作用: SiLU 激活函数。
        逻辑:
        x * sigmoid(x)。
        """
        return x * th.sigmoid(x)


# ==============================================================================
# 归一化层
# ==============================================================================

class GroupNorm32(nn.GroupNorm):
    """32组归一化，适合小batch size训练"""
    def forward(self, x):
        """
        输入:
            x: 输入张量。
        输出:
            输出张量。
        作用: 32 组归一化。
        逻辑:
        转换为 float 进行归一化，然后转回原类型。
        """
        return super().forward(x.float()).type(x.dtype)


class LayerNorm3D(nn.Module):
    """
    3D LayerNorm - 对每个样本的所有通道和空间维度进行归一化
    
    适用场景：
    - 极小的 batch size (1-2)
    - 需要样本独立归一化
    
    优点：完全不依赖 batch 统计量
    """
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(th.ones(1, channels, 1, 1, 1))
        self.beta = nn.Parameter(th.zeros(1, channels, 1, 1, 1))
        
    def forward(self, x):
        # x: (B, C, D, H, W)
        # 对 C, D, H, W 维度归一化
        mean = x.mean(dim=[1, 2, 3, 4], keepdim=True)
        var = x.var(dim=[1, 2, 3, 4], keepdim=True, unbiased=False)
        x_norm = (x - mean) / th.sqrt(var + self.eps)
        return x_norm * self.gamma + self.beta


class InstanceNorm3D(nn.Module):
    """
    3D InstanceNorm - 对每个样本的每个通道独立归一化
    
    适用场景：
    - 点云/稀疏数据处理
    - 风格迁移类任务
    - 雷达数据（每个通道可能有不同的物理含义）
    
    优点：
    - 不依赖 batch 统计量
    - 保持通道间的独立性
    """
    def __init__(self, channels: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.affine = affine
        
        if affine:
            self.gamma = nn.Parameter(th.ones(1, channels, 1, 1, 1))
            self.beta = nn.Parameter(th.zeros(1, channels, 1, 1, 1))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)
            
    def forward(self, x):
        # x: (B, C, D, H, W)
        # 对 D, H, W 维度归一化（每个 channel 独立）
        mean = x.mean(dim=[2, 3, 4], keepdim=True)
        var = x.var(dim=[2, 3, 4], keepdim=True, unbiased=False)
        x_norm = (x - mean) / th.sqrt(var + self.eps)
        
        if self.affine:
            return x_norm * self.gamma + self.beta
        return x_norm


class RMSNorm3D(nn.Module):
    """
    3D RMSNorm - Root Mean Square Layer Normalization
    
    优点：
    - 比 LayerNorm 更高效（省略均值计算）
    - 在 LLM 中证明有效（如 LLaMA）
    - 计算更稳定
    
    论文: Root Mean Square Layer Normalization (https://arxiv.org/abs/1910.07467)
    """
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.scale = nn.Parameter(th.ones(1, channels, 1, 1, 1))
        
    def forward(self, x):
        # x: (B, C, D, H, W)
        # 计算 RMS
        rms = th.sqrt(x.pow(2).mean(dim=[1, 2, 3, 4], keepdim=True) + self.eps)
        x_norm = x / rms
        return x_norm * self.scale


class AdaptiveNorm3D(nn.Module):
    """
    自适应归一化 - 根据输入稀疏度自动调整
    
    针对雷达数据的特殊设计：
    - 当输入大部分为零时，使用 InstanceNorm 保持稀疏性
    - 当输入较密集时，使用 GroupNorm 获得更好的统计
    """
    def __init__(self, channels: int, num_groups: int = 32, sparsity_threshold: float = 0.7):
        super().__init__()
        self.channels = channels
        self.sparsity_threshold = sparsity_threshold
        
        self.group_norm = GroupNorm32(num_groups, channels)
        self.instance_norm = InstanceNorm3D(channels)
        
    def forward(self, x):
        # 计算稀疏度（零值比例）
        with th.no_grad():
            sparsity = (x.abs() < 1e-6).float().mean()
        
        # 根据稀疏度选择归一化方式
        if sparsity > self.sparsity_threshold:
            return self.instance_norm(x)
        else:
            return self.group_norm(x)


class ConditionalNorm3D(nn.Module):
    """
    条件归一化 - 用于条件生成（如时间步条件）
    
    类似 AdaIN，但针对 3D 数据优化
    """
    def __init__(self, channels: int, cond_channels: int, num_groups: int = 32):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, channels, affine=False)
        
        # 条件投影
        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_channels, channels * 2),
        )
        
    def forward(self, x, cond):
        """
        Args:
            x: (B, C, D, H, W) 输入特征
            cond: (B, cond_channels) 条件嵌入
        """
        # 归一化
        x = self.norm(x)
        
        # 获取条件 scale 和 shift
        cond_out = self.cond_proj(cond)
        scale, shift = cond_out.chunk(2, dim=1)
        
        # 扩展维度以匹配 x
        scale = scale.view(scale.shape[0], -1, 1, 1, 1)
        shift = shift.view(shift.shape[0], -1, 1, 1, 1)
        
        return x * (1 + scale) + shift


def conv_nd(dims, *args, **kwargs):
    """
    输入:
        dims: 维度 (1, 2, 3)。
        *args: 卷积参数。
        **kwargs: 卷积关键字参数。
    输出:
        卷积模块。
    作用: 创建 N 维卷积模块。
    逻辑:
    根据 dims 返回 Conv1d, Conv2d 或 Conv3d。
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    输入:
        *args: 线性层参数。
        **kwargs: 线性层关键字参数。
    输出:
        线性模块。
    作用: 创建线性模块。
    逻辑:
    返回 nn.Linear。
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    输入:
        dims: 维度 (1, 2, 3)。
        *args: 池化参数。
        **kwargs: 池化关键字参数。
    输出:
        池化模块。
    作用: 创建 N 维平均池化模块。
    逻辑:
    根据 dims 返回 AvgPool1d, AvgPool2d 或 AvgPool3d。
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    """
    输入:
        target_params: 目标参数序列。
        source_params: 源参数序列。
        rate: EMA 速率。
    输出:
        无
    作用: 使用指数移动平均更新目标参数。
    逻辑:
    target = target * rate + source * (1 - rate)。
    """
    # print("aaa")
    # for targ in target_params:
    #     print("targ", targ.shape)
    for targ, src in zip(target_params, source_params):
        # print("targ", targ.detach().shape)
        # print(" targ.detach().mul_(rate)",  targ.detach().mul_(rate).shape)
        # print("src", src.shape)
        # print("rate", rate)
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)
    # print("asdfsdf", src.shape)


def zero_module(module):
    """
    输入:
        module: 模块。
    输出:
        模块。
    作用: 将模块参数置零。
    逻辑:
    遍历参数并置零。
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    输入:
        module: 模块。
        scale: 缩放因子。
    输出:
        模块。
    作用: 缩放模块参数。
    逻辑:
    遍历参数并乘以 scale。
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    输入:
        tensor: 输入张量。
    输出:
        输出张量。
    作用: 对所有非批次维度求平均。
    逻辑:
    对维度 1 到最后求平均。
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def append_dims(x, target_dims):
    """
    输入:
        x: 输入张量。
        target_dims: 目标维度数。
    输出:
        输出张量。
    作用: 在张量末尾添加维度直到达到目标维度数。
    逻辑:
    计算需要添加的维度数，然后使用切片添加。
    """
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


def append_zero(x):
    """
    输入:
        x: 输入张量。
    输出:
        输出张量。
    作用: 在张量末尾添加一个零。
    逻辑:
    拼接一个零张量。
    """
    return th.cat([x, x.new_zeros([1])])


def normalization(channels, norm_type: str = None, num_groups: int = None):
    """
    输入:
        channels: 输入通道数。
        norm_type: 归一化类型，可选 "group", "layer", "instance", "rms", "adaptive"
        num_groups: GroupNorm 的组数
    输出:
        归一化模块。
    作用: 创建标准归一化层。
    逻辑:
    根据 norm_type 返回对应的归一化层。
    """
    # 使用全局配置或参数覆盖
    _norm_type = norm_type or NORM_TYPE
    _num_groups = num_groups or NORM_GROUPS
    
    # 确保 num_groups 不超过通道数
    _num_groups = min(_num_groups, channels)
    
    if _norm_type == "group":
        return GroupNorm32(_num_groups, channels)
    elif _norm_type == "layer":
        return LayerNorm3D(channels)
    elif _norm_type == "instance":
        return InstanceNorm3D(channels)
    elif _norm_type == "rms":
        return RMSNorm3D(channels)
    elif _norm_type == "adaptive":
        return AdaptiveNorm3D(channels, _num_groups)
    else:
        # 默认使用 GroupNorm
        return GroupNorm32(_num_groups, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    输入:
        timesteps: 时间步张量。
        dim: 嵌入维度。
        max_period: 最大周期。
    输出:
        嵌入张量。
    作用: 创建正弦时间步嵌入。
    逻辑:
    1. 计算频率。
    2. 计算正弦和余弦参数。
    3. 拼接正弦和余弦部分。
    """

    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(func, inputs, params, flag):
    """
    输入:
        func: 要评估的函数。
        inputs: 输入参数序列。
        params: 参数序列。
        flag: 是否启用 checkpoint。
    输出:
        函数输出。
    作用: 使用 checkpoint 评估函数。
    逻辑:
    如果 flag 为 True，使用 CheckpointFunction。否则直接调用函数。
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(th.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        """
        输入:
            ctx: 上下文。
            run_function: 运行函数。
            length: 输入张量长度。
            *args: 输入参数。
        输出:
            输出张量。
        作用: 前向传播。
        逻辑:
        1. 保存函数和输入。
        2. 在 no_grad 模式下运行函数。
        """
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with th.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        """
        输入:
            ctx: 上下文。
            *output_grads: 输出梯度。
        输出:
            输入梯度。
        作用: 反向传播。
        逻辑:
        1. 重新计算输出（启用梯度）。
        2. 计算输入梯度。
        """
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with th.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = th.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads
