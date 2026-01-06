from abc import ABC, abstractmethod

import numpy as np
import torch as th
from scipy.stats import norm
import torch.distributed as dist


def create_named_schedule_sampler(name, diffusion):
    """
    输入:
        name: 采样器名称。
        diffusion: 扩散对象。
    输出:
        ScheduleSampler 对象。
    作用: 从预定义的采样器库中创建一个 ScheduleSampler。
    逻辑:
    根据名称返回相应的采样器实例。
    """
    if name == "uniform":
        return UniformSampler(diffusion)
    elif name == "loss-second-moment":
        return LossSecondMomentResampler(diffusion)
    elif name == "lognormal":
        return LogNormalSampler()
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")


class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """

    @abstractmethod
    def weights(self):
        """
        输入:
            无
        输出:
            权重 numpy 数组。
        作用: 获取每个扩散步骤的权重。
        逻辑:
        抽象方法，由子类实现。
        """

    def sample(self, batch_size, device):
        """
        输入:
            batch_size: 批次大小。
            device: 设备。
        输出:
            timesteps: 时间步索引张量。
            weights: 权重张量。
        作用: 为批次进行重要性采样时间步。
        逻辑:
        1. 获取权重并归一化为概率。
        2. 根据概率随机选择索引。
        3. 计算对应的权重。
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = th.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = th.from_numpy(weights_np).float().to(device)
        return indices, weights



class UniformSampler(ScheduleSampler):
    def __init__(self, diffusion):
        """
        输入:
            diffusion: 扩散对象。
        输出:
            无
        作用: 初始化均匀采样器。
        逻辑:
        初始化权重为全 1。
        """
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self):
        """
        输入:
            无
        输出:
            权重数组。
        作用: 获取权重。
        逻辑:
        返回预先计算的权重。
        """
        return self._weights


class LossAwareSampler(ScheduleSampler):
    def update_with_local_losses(self, local_ts, local_losses):
        """
        输入:
            local_ts: 本地时间步张量。
            local_losses: 本地损失张量。
        输出:
            无
        作用: 使用模型的损失更新重加权。
        逻辑:
        1. 收集所有进程的批次大小。
        2. 收集所有进程的时间步和损失。
        3. 调用 update_with_all_losses 更新权重。
        """
        batch_sizes = [
            th.tensor([0], dtype=th.int32, device=local_ts.device)
            for _ in range(dist.get_world_size())
        ]
        dist.all_gather(
            batch_sizes,
            th.tensor([len(local_ts)], dtype=th.int32, device=local_ts.device),
        )

        # Pad all_gather batches to be the maximum batch size.
        batch_sizes = [x.item() for x in batch_sizes]
        max_bs = max(batch_sizes)

        timestep_batches = [th.zeros(max_bs).to(local_ts) for bs in batch_sizes]
        loss_batches = [th.zeros(max_bs).to(local_losses) for bs in batch_sizes]
        dist.all_gather(timestep_batches, local_ts)
        dist.all_gather(loss_batches, local_losses)
        timesteps = [
            x.item() for y, bs in zip(timestep_batches, batch_sizes) for x in y[:bs]
        ]
        losses = [x.item() for y, bs in zip(loss_batches, batch_sizes) for x in y[:bs]]
        self.update_with_all_losses(timesteps, losses)

    @abstractmethod
    def update_with_all_losses(self, ts, losses):
        """
        输入:
            ts: 时间步列表。
            losses: 损失列表。
        输出:
            无
        作用: 使用所有损失更新重加权。
        逻辑:
        抽象方法，由子类实现。
        """


class LossSecondMomentResampler(LossAwareSampler):
    def __init__(self, diffusion, history_per_term=10, uniform_prob=0.001):
        """
        输入:
            diffusion: 扩散对象。
            history_per_term: 每个时间步的历史记录数。
            uniform_prob: 均匀采样概率。
        输出:
            无
        作用: 初始化损失二阶矩重采样器。
        逻辑:
        初始化损失历史和计数。
        """
        self.diffusion = diffusion
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros(
            [diffusion.num_timesteps, history_per_term], dtype=np.float64
        )
        self._loss_counts = np.zeros([diffusion.num_timesteps], dtype=np.int)

    def weights(self):
        """
        输入:
            无
        输出:
            权重数组。
        作用: 计算权重。
        逻辑:
        1. 如果未预热，返回均匀权重。
        2. 计算损失平方均值的平方根。
        3. 归一化权重。
        4. 混合均匀概率。
        """
        if not self._warmed_up():
            return np.ones([self.diffusion.num_timesteps], dtype=np.float64)
        weights = np.sqrt(np.mean(self._loss_history**2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts, losses):
        """
        输入:
            ts: 时间步列表。
            losses: 损失列表。
        输出:
            无
        作用: 更新损失历史。
        逻辑:
        遍历时间步和损失，更新历史记录。
        """
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                # Shift out the oldest loss term.
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self):
        """
        输入:
            无
        输出:
            布尔值。
        作用: 检查是否预热完成。
        逻辑:
        检查所有时间步的历史记录是否已满。
        """
        return (self._loss_counts == self.history_per_term).all()


class LogNormalSampler:
    def __init__(self, p_mean=-1.2, p_std=1.2, even=False):
        """
        输入:
            p_mean: 对数正态分布均值。
            p_std: 对数正态分布标准差。
            even: 是否均匀采样。
        输出:
            无
        作用: 初始化对数正态采样器。
        逻辑:
        初始化参数。如果 even 为 True，初始化逆累积分布函数。
        """
        self.p_mean = p_mean
        self.p_std = p_std
        self.even = even
        if self.even:
            self.inv_cdf = lambda x: norm.ppf(x, loc=p_mean, scale=p_std)
            self.rank, self.size = dist.get_rank(), dist.get_world_size()

    def sample(self, bs, device):
        """
        输入:
            bs: 批次大小。
            device: 设备。
        输出:
            sigmas: sigma 值张量。
            weights: 权重张量。
        作用: 采样 sigma。
        逻辑:
        1. 如果 even 为 True，使用逆累积分布函数均匀采样。
        2. 否则，随机采样。
        3. 计算 sigma 和权重。
        """
        if self.even:
            # buckets = [1/G]
            start_i, end_i = self.rank * bs, (self.rank + 1) * bs
            global_batch_size = self.size * bs
            locs = (th.arange(start_i, end_i) + th.rand(bs)) / global_batch_size
            log_sigmas = th.tensor(self.inv_cdf(locs), dtype=th.float32, device=device)
        else:
            log_sigmas = self.p_mean + self.p_std * th.randn(bs, device=device)
        sigmas = th.exp(log_sigmas)
        weights = th.ones_like(sigmas)
        return sigmas, weights

    def get_max(self, device, bs = 1):
        """
        输入:
            device: 设备。
            bs: 批次大小。
        输出:
            max_sigma: 最大 sigma 值。
        作用: 获取最大 sigma。
        逻辑:
        随机采样并返回。
        """
        log_sigma = self.p_mean + self.p_std * th.randn(bs, device=device)
        print("log_sigma", log_sigma)
        max_sigma = th.exp(log_sigma)
        return max_sigma
    