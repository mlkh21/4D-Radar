"""
Various utilities for neural networks.
"""

import math

import torch as th
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


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


class GroupNorm32(nn.GroupNorm):
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


def normalization(channels):
    """
    输入:
        channels: 输入通道数。
    输出:
        归一化模块。
    作用: 创建标准归一化层。
    逻辑:
    返回 GroupNorm32。
    """
    return GroupNorm32(32, channels)


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
