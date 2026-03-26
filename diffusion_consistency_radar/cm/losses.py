"""
概率类损失函数工具集合。

NOTE: 该实现参考 Ho 等人的扩散模型公开代码，并按当前工程接口做了轻量改写。
"""

import numpy as np

import torch as th


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    输入:
        mean1: 分布 1 的均值。
        logvar1: 分布 1 的对数方差。
        mean2: 分布 2 的均值。
        logvar2: 分布 2 的对数方差。
    输出:
        KL 散度。
    作用: 计算两个高斯分布之间的 KL 散度。
    逻辑:
    使用 KL 散度公式计算。
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # NOTE: 方差项必须先统一转为 Tensor；否则 th.exp 在标量分支会报类型错误。
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + th.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    输入:
        x: 输入张量。
    输出:
        CDF 值。
    作用: 标准正态分布累积分布函数的快速近似。
    逻辑:
    使用 tanh 近似公式。
    """
    # NOTE: 采用 GELU 同源近似公式，速度快于直接调用 erf。
    return 0.5 * (1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * th.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    输入:
        x: 目标图像。
        means: 高斯均值。
        log_scales: 高斯对数标准差。
    输出:
        对数概率。
    作用: 计算离散化到给定图像的高斯分布的对数似然。
    逻辑:
    1. 计算中心化 x。
    2. 计算 CDF 的上下界。
    3. 计算对数概率。
    """
    # NOTE: 该函数按逐元素形式计算，输入张量形状必须完全一致。
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = th.where(
        x < -0.999,
        log_cdf_plus,
        th.where(x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs
