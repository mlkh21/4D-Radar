"""
Helpers to train with 16-bit precision.
"""

import numpy as np
import torch as th
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from . import logger

INITIAL_LOG_LOSS_SCALE = 20.0


def convert_module_to_f16(l):
    """
    输入:
        l: 模块。
    输出:
        无
    作用: 将基本模块转换为 float16。
    逻辑:
    如果是卷积层，将权重和偏置转换为 half。
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.half()
        if l.bias is not None:
            l.bias.data = l.bias.data.half()


def convert_module_to_f32(l):
    """
    输入:
        l: 模块。
    输出:
        无
    作用: 将基本模块转换为 float32。
    逻辑:
    如果是卷积层，将权重和偏置转换为 float。
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.float()
        if l.bias is not None:
            l.bias.data = l.bias.data.float()


def make_master_params(param_groups_and_shapes):
    """
    输入:
        param_groups_and_shapes: 参数组和形状列表。
    输出:
        master_params: 主参数列表。
    作用: 将模型参数复制到全精度参数列表中。
    逻辑:
    1. 遍历参数组。
    2. 展平并转换为 float。
    3. 创建 Parameter 对象并添加到列表。
    """
    master_params = []
    for param_group, shape in param_groups_and_shapes:
        master_param = nn.Parameter(
            _flatten_dense_tensors(
                [param.detach().float() for (_, param) in param_group]
            ).view(shape)
        )
        master_param.requires_grad = True
        master_params.append(master_param)
    return master_params


def model_grads_to_master_grads(param_groups_and_shapes, master_params):
    """
    输入:
        param_groups_and_shapes: 参数组和形状列表。
        master_params: 主参数列表。
    输出:
        无
    作用: 将模型参数的梯度复制到主参数中。
    逻辑:
    1. 遍历主参数和参数组。
    2. 展平梯度并赋值给主参数的 grad。
    """
    for master_param, (param_group, shape) in zip(
        master_params, param_groups_and_shapes
    ):
        master_param.grad = _flatten_dense_tensors(
            [param_grad_or_zeros(param) for (_, param) in param_group]
        ).view(shape)


def master_params_to_model_params(param_groups_and_shapes, master_params):
    """
    输入:
        param_groups_and_shapes: 参数组和形状列表。
        master_params: 主参数列表。
    输出:
        无
    作用: 将主参数数据复制回模型参数。
    逻辑:
    1. 遍历主参数和参数组。
    2. 解展平主参数。
    3. 将数据复制到模型参数。
    """
    # Without copying to a list, if a generator is passed, this will
    # silently not copy any parameters.
    for master_param, (param_group, _) in zip(master_params, param_groups_and_shapes):
        for (_, param), unflat_master_param in zip(
            param_group, unflatten_master_params(param_group, master_param.view(-1))
        ):
            param.detach().copy_(unflat_master_param)


def unflatten_master_params(param_group, master_param):
    """
    输入:
        param_group: 参数组。
        master_param: 主参数。
    输出:
        解展平后的参数列表。
    作用: 解展平主参数。
    逻辑:
    调用 _unflatten_dense_tensors。
    """
    return _unflatten_dense_tensors(master_param, [param for (_, param) in param_group])


def get_param_groups_and_shapes(named_model_params):
    """
    输入:
        named_model_params: 命名模型参数。
    输出:
        参数组和形状列表。
    作用: 获取参数组和形状。
    逻辑:
    1. 将参数分为标量/向量和矩阵。
    2. 返回分组列表。
    """
    named_model_params = list(named_model_params)
    scalar_vector_named_params = (
        [(n, p) for (n, p) in named_model_params if p.ndim <= 1],
        (-1),
    )
    matrix_named_params = (
        [(n, p) for (n, p) in named_model_params if p.ndim > 1],
        (1, -1),
    )
    return [scalar_vector_named_params, matrix_named_params]


def master_params_to_state_dict(
    model, param_groups_and_shapes, master_params, use_fp16
):
    """
    输入:
        model: 模型。
        param_groups_and_shapes: 参数组和形状。
        master_params: 主参数。
        use_fp16: 是否使用 fp16。
    输出:
        state_dict: 状态字典。
    作用: 将主参数转换为状态字典。
    逻辑:
    1. 如果使用 fp16，解展平主参数并更新状态字典。
    2. 否则，直接使用主参数更新状态字典。
    """
    if use_fp16:
        state_dict = model.state_dict()
        for master_param, (param_group, _) in zip(
            master_params, param_groups_and_shapes
        ):
            for (name, _), unflat_master_param in zip(
                param_group, unflatten_master_params(param_group, master_param.view(-1))
            ):
                assert name in state_dict
                state_dict[name] = unflat_master_param
    else:
        state_dict = model.state_dict()
        for i, (name, _value) in enumerate(model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
    return state_dict


def state_dict_to_master_params(model, state_dict, use_fp16):
    """
    输入:
        model: 模型。
        state_dict: 状态字典。
        use_fp16: 是否使用 fp16。
    输出:
        master_params: 主参数。
    作用: 将状态字典转换为主参数。
    逻辑:
    1. 如果使用 fp16，从状态字典创建主参数。
    2. 否则，直接从状态字典获取参数。
    """
    if use_fp16:
        named_model_params = [
            (name, state_dict[name]) for name, _ in model.named_parameters()
        ]
        param_groups_and_shapes = get_param_groups_and_shapes(named_model_params)
        master_params = make_master_params(param_groups_and_shapes)
    else:
        master_params = [state_dict[name] for name, _ in model.named_parameters()]
    return master_params


def zero_master_grads(master_params):
    """
    输入:
        master_params: 主参数。
    输出:
        无
    作用: 将主参数梯度置零。
    逻辑:
    遍历主参数，将 grad 设为 None。
    """
    for param in master_params:
        param.grad = None


def zero_grad(model_params):
    """
    输入:
        model_params: 模型参数。
    输出:
        无
    作用: 将模型参数梯度置零。
    逻辑:
    遍历模型参数，将 grad 置零。
    """
    for param in model_params:
        # Taken from https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer.add_param_group
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()


def param_grad_or_zeros(param):
    """
    输入:
        param: 参数。
    输出:
        梯度或零张量。
    作用: 获取参数梯度，如果为 None 则返回零。
    逻辑:
    检查 param.grad 是否存在。
    """
    if param.grad is not None:
        return param.grad.data.detach()
    else:
        return th.zeros_like(param)


class MixedPrecisionTrainer:
    def __init__(
        self,
        *,
        model,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        initial_lg_loss_scale=INITIAL_LOG_LOSS_SCALE,
    ):
        """
        输入:
            model: 模型。
            use_fp16: 是否使用 fp16。
            fp16_scale_growth: fp16 缩放增长率。
            initial_lg_loss_scale: 初始对数损失缩放。
        输出:
            无
        作用: 初始化混合精度训练器。
        逻辑:
        1. 初始化参数。
        2. 如果使用 fp16，创建主参数并转换模型。
        """
        self.model = model
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.param_groups_and_shapes = None
        self.lg_loss_scale = initial_lg_loss_scale

        if self.use_fp16:
            self.param_groups_and_shapes = get_param_groups_and_shapes(
                self.model.named_parameters()
            )
            self.master_params = make_master_params(self.param_groups_and_shapes)
            self.model.convert_to_fp16()

    def zero_grad(self):
        """
        输入:
            无
        输出:
            无
        作用: 梯度置零。
        逻辑:
        调用 zero_grad。
        """
        zero_grad(self.model_params)

    def backward(self, loss: th.Tensor):
        """
        输入:
            loss: 损失张量。
        输出:
            无
        作用: 反向传播。
        逻辑:
        1. 如果使用 fp16，缩放损失并反向传播。
        2. 否则，直接反向传播。
        """
        if self.use_fp16:
            loss_scale = 2**self.lg_loss_scale
            (loss * loss_scale).backward()
        else:
            loss.backward()

    def optimize(self, opt: th.optim.Optimizer):
        """
        输入:
            opt: 优化器。
        输出:
            success: 是否成功更新。
        作用: 执行优化步骤。
        逻辑:
        1. 如果使用 fp16，处理梯度缩放和溢出检查。
        2. 否则，直接 step。
        """
        if self.use_fp16:
            return self._optimize_fp16(opt)
        else:
            return self._optimize_normal(opt)

    def _optimize_fp16(self, opt: th.optim.Optimizer):
        """
        输入:
            opt: 优化器。
        输出:
            success: 是否成功更新。
        作用: fp16 优化步骤。
        逻辑:
        1. 记录 loss scale。
        2. 将模型梯度复制到主参数。
        3. 计算梯度范数并检查溢出。
        4. 如果溢出，减小 loss scale 并返回 False。
        5. 如果未溢出，缩放梯度，执行 step，增加 loss scale 并返回 True。
        """
        logger.logkv_mean("lg_loss_scale", self.lg_loss_scale)
        model_grads_to_master_grads(self.param_groups_and_shapes, self.master_params)
        grad_norm, param_norm = self._compute_norms(grad_scale=2**self.lg_loss_scale)
        if check_overflow(grad_norm):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            zero_master_grads(self.master_params)
            return False

        logger.logkv_mean("grad_norm", grad_norm)
        logger.logkv_mean("param_norm", param_norm)

        for p in self.master_params:
            p.grad.mul_(1.0 / (2**self.lg_loss_scale))
        opt.step()
        zero_master_grads(self.master_params)
        master_params_to_model_params(self.param_groups_and_shapes, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth
        return True

    def _optimize_normal(self, opt: th.optim.Optimizer):
        """
        输入:
            opt: 优化器。
        输出:
            success: 是否成功更新。
        作用: 普通优化步骤。
        逻辑:
        1. 计算梯度和参数范数并记录。
        2. 执行 step。
        """
        grad_norm, param_norm = self._compute_norms()
        # print("grad_norm", grad_norm)
        logger.logkv_mean("grad_norm", grad_norm)
        logger.logkv_mean("param_norm", param_norm)
        opt.step()
        return True

    def _compute_norms(self, grad_scale=1.0):
        """
        输入:
            grad_scale: 梯度缩放因子。
        输出:
            grad_norm: 梯度范数。
            param_norm: 参数范数。
        作用: 计算梯度和参数的范数。
        逻辑:
        计算 L2 范数。
        """
        grad_norm = 0.0
        param_norm = 0.0
        for p in self.master_params:
            with th.no_grad():
                param_norm += th.norm(p, p=2, dtype=th.float32).item() ** 2
                if p.grad is not None:
                    grad_norm += th.norm(p.grad, p=2, dtype=th.float32).item() ** 2
        return np.sqrt(grad_norm) / grad_scale, np.sqrt(param_norm)

    def master_params_to_state_dict(self, master_params):
        """
        输入:
            master_params: 主参数。
        输出:
            state_dict: 状态字典。
        作用: 将主参数转换为状态字典。
        逻辑:
        调用 master_params_to_state_dict。
        """
        return master_params_to_state_dict(
            self.model, self.param_groups_and_shapes, master_params, self.use_fp16
        )

    def state_dict_to_master_params(self, state_dict):
        """
        输入:
            state_dict: 状态字典。
        输出:
            master_params: 主参数。
        作用: 将状态字典转换为主参数。
        逻辑:
        调用 state_dict_to_master_params。
        """
        return state_dict_to_master_params(self.model, state_dict, self.use_fp16)


def check_overflow(value):
    return (value == float("inf")) or (value == -float("inf")) or (value != value)