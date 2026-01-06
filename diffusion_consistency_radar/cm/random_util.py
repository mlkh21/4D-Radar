import torch as th
import torch.distributed as dist
from . import dist_util


def get_generator(generator, num_samples=0, seed=0):
    """
    输入:
        generator: 生成器类型。
        num_samples: 样本数量。
        seed: 随机种子。
    输出:
        Generator: 随机数生成器。
    作用: 获取随机数生成器。
    逻辑:
    根据 generator 类型返回相应的生成器实例。
    """
    if generator == "dummy":
        return DummyGenerator()
    elif generator == "determ":
        return DeterministicGenerator(num_samples, seed)
    elif generator == "determ-indiv":
        return DeterministicIndividualGenerator(num_samples, seed)
    else:
        raise NotImplementedError


class DummyGenerator:
    def randn(self, *args, **kwargs):
        """
        输入:
            *args: 参数。
            **kwargs: 关键字参数。
        输出:
            Tensor: 随机张量。
        作用: 生成标准正态分布随机数。
        逻辑:
        调用 torch.randn。
        """
        return th.randn(*args, **kwargs)

    def randint(self, *args, **kwargs):
        """
        输入:
            *args: 参数。
            **kwargs: 关键字参数。
        输出:
            Tensor: 随机整数张量。
        作用: 生成随机整数。
        逻辑:
        调用 torch.randint。
        """
        return th.randint(*args, **kwargs)

    def randn_like(self, *args, **kwargs):
        """
        输入:
            *args: 参数。
            **kwargs: 关键字参数。
        输出:
            Tensor: 随机张量。
        作用: 生成与输入形状相同的标准正态分布随机数。
        逻辑:
        调用 torch.randn_like。
        """
        return th.randn_like(*args, **kwargs)


class DeterministicGenerator:
    """
    RNG to deterministically sample num_samples samples that does not depend on batch_size or mpi_machines
    Uses a single rng and samples num_samples sized randomness and subsamples the current indices
    """

    def __init__(self, num_samples, seed=0):
        """
        输入:
            num_samples: 样本数量。
            seed: 随机种子。
        输出:
            无
        作用: 初始化确定性生成器。
        逻辑:
        1. 获取 MPI 排名和世界大小。
        2. 初始化 CPU 和 CUDA 生成器。
        3. 设置种子。
        """
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            print("Warning: Distributed not initialised, using single rank")
            self.rank = 0
            self.world_size = 1
        self.num_samples = num_samples
        self.done_samples = 0
        self.seed = seed
        self.rng_cpu = th.Generator()
        if th.cuda.is_available():
            self.rng_cuda = th.Generator(dist_util.dev())
        self.set_seed(seed)

    def get_global_size_and_indices(self, size):
        """
        输入:
            size: 形状。
        输出:
            tuple: 全局形状和索引。
        作用: 获取全局形状和当前进程的索引。
        逻辑:
        计算全局形状和当前进程需要处理的索引。
        """
        global_size = (self.num_samples, *size[1:])
        indices = th.arange(
            self.done_samples + self.rank,
            self.done_samples + self.world_size * int(size[0]),
            self.world_size,
        )
        indices = th.clamp(indices, 0, self.num_samples - 1)
        assert (
            len(indices) == size[0]
        ), f"rank={self.rank}, ws={self.world_size}, l={len(indices)}, bs={size[0]}"
        return global_size, indices

    def get_generator(self, device):
        """
        输入:
            device: 设备。
        输出:
            Generator: 生成器。
        作用: 获取指定设备的生成器。
        逻辑:
        返回 CPU 或 CUDA 生成器。
        """
        return self.rng_cpu if th.device(device).type == "cpu" else self.rng_cuda

    def randn(self, *size, dtype=th.float, device="cpu"):
        """
        输入:
            *size: 形状。
            dtype: 数据类型。
            device: 设备。
        输出:
            Tensor: 随机张量。
        作用: 生成标准正态分布随机数。
        逻辑:
        1. 获取全局形状和索引。
        2. 生成全局随机数。
        3. 返回当前进程的子集。
        """
        global_size, indices = self.get_global_size_and_indices(size)
        generator = self.get_generator(device)
        return th.randn(*global_size, generator=generator, dtype=dtype, device=device)[
            indices
        ]

    def randint(self, low, high, size, dtype=th.long, device="cpu"):
        """
        输入:
            low: 下界。
            high: 上界。
            size: 形状。
            dtype: 数据类型。
            device: 设备。
        输出:
            Tensor: 随机整数张量。
        作用: 生成随机整数。
        逻辑:
        1. 获取全局形状和索引。
        2. 生成全局随机数。
        3. 返回当前进程的子集。
        """
        global_size, indices = self.get_global_size_and_indices(size)
        generator = self.get_generator(device)
        return th.randint(
            low, high, generator=generator, size=global_size, dtype=dtype, device=device
        )[indices]

    def randn_like(self, tensor):
        """
        输入:
            tensor: 输入张量。
        输出:
            Tensor: 随机张量。
        作用: 生成与输入形状相同的标准正态分布随机数。
        逻辑:
        调用 randn。
        """
        size, dtype, device = tensor.size(), tensor.dtype, tensor.device
        return self.randn(*size, dtype=dtype, device=device)

    def set_done_samples(self, done_samples):
        """
        输入:
            done_samples: 已完成样本数。
        输出:
            无
        作用: 设置已完成样本数。
        逻辑:
        更新 done_samples 并重置种子。
        """
        self.done_samples = done_samples
        self.set_seed(self.seed)

    def get_seed(self):
        """
        输入:
            无
        输出:
            int: 随机种子。
        作用: 获取随机种子。
        逻辑:
        返回 seed。
        """
        return self.seed

    def set_seed(self, seed):
        """
        输入:
            seed: 随机种子。
        输出:
            无
        作用: 设置随机种子。
        逻辑:
        设置 CPU 和 CUDA 生成器的种子。
        """
        self.rng_cpu.manual_seed(seed)
        if th.cuda.is_available():
            self.rng_cuda.manual_seed(seed)


class DeterministicIndividualGenerator:
    """
    RNG to deterministically sample num_samples samples that does not depend on batch_size or mpi_machines
    Uses a separate rng for each sample to reduce memoery usage
    """

    def __init__(self, num_samples, seed=0):
        """
        输入:
            num_samples: 样本数量。
            seed: 随机种子。
        输出:
            无
        作用: 初始化确定性个体生成器。
        逻辑:
        1. 获取 MPI 排名和世界大小。
        2. 初始化 CPU 和 CUDA 生成器列表（每个样本一个）。
        3. 设置种子。
        """
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            print("Warning: Distributed not initialised, using single rank")
            self.rank = 0
            self.world_size = 1
        self.num_samples = num_samples
        self.done_samples = 0
        self.seed = seed
        self.rng_cpu = [th.Generator() for _ in range(num_samples)]
        if th.cuda.is_available():
            self.rng_cuda = [th.Generator(dist_util.dev()) for _ in range(num_samples)]
        self.set_seed(seed)

    def get_size_and_indices(self, size):
        """
        输入:
            size: 形状。
        输出:
            tuple: 形状和索引。
        作用: 获取形状和当前进程的索引。
        逻辑:
        计算当前进程需要处理的索引。
        """
        indices = th.arange(
            self.done_samples + self.rank,
            self.done_samples + self.world_size * int(size[0]),
            self.world_size,
        )
        indices = th.clamp(indices, 0, self.num_samples - 1)
        assert (
            len(indices) == size[0]
        ), f"rank={self.rank}, ws={self.world_size}, l={len(indices)}, bs={size[0]}"
        return (1, *size[1:]), indices

    def get_generator(self, device):
        """
        输入:
            device: 设备。
        输出:
            list: 生成器列表。
        作用: 获取指定设备的生成器列表。
        逻辑:
        返回 CPU 或 CUDA 生成器列表。
        """
        return self.rng_cpu if th.device(device).type == "cpu" else self.rng_cuda

    def randn(self, *size, dtype=th.float, device="cpu"):
        """
        输入:
            *size: 形状。
            dtype: 数据类型。
            device: 设备。
        输出:
            Tensor: 随机张量。
        作用: 生成标准正态分布随机数。
        逻辑:
        1. 获取形状和索引。
        2. 遍历索引，使用对应的生成器生成随机数。
        3. 拼接结果。
        """
        size, indices = self.get_size_and_indices(size)
        generator = self.get_generator(device)
        return th.cat(
            [
                th.randn(*size, generator=generator[i], dtype=dtype, device=device)
                for i in indices
            ],
            dim=0,
        )

    def randint(self, low, high, size, dtype=th.long, device="cpu"):
        """
        输入:
            low: 下界。
            high: 上界。
            size: 形状。
            dtype: 数据类型。
            device: 设备。
        输出:
            Tensor: 随机整数张量。
        作用: 生成随机整数。
        逻辑:
        1. 获取形状和索引。
        2. 遍历索引，使用对应的生成器生成随机数。
        3. 拼接结果。
        """
        size, indices = self.get_size_and_indices(size)
        generator = self.get_generator(device)
        return th.cat(
            [
                th.randint(
                    low,
                    high,
                    generator=generator[i],
                    size=size,
                    dtype=dtype,
                    device=device,
                )
                for i in indices
            ],
            dim=0,
        )

    def randn_like(self, tensor):
        """
        输入:
            tensor: 输入张量。
        输出:
            Tensor: 随机张量。
        作用: 生成与输入形状相同的标准正态分布随机数。
        逻辑:
        调用 randn。
        """
        size, dtype, device = tensor.size(), tensor.dtype, tensor.device
        return self.randn(*size, dtype=dtype, device=device)

    def set_done_samples(self, done_samples):
        """
        输入:
            done_samples: 已完成样本数。
        输出:
            无
        作用: 设置已完成样本数。
        逻辑:
        更新 done_samples。
        """
        self.done_samples = done_samples

    def get_seed(self):
        """
        输入:
            无
        输出:
            int: 随机种子。
        作用: 获取随机种子。
        逻辑:
        返回 seed。
        """
        return self.seed

    def set_seed(self, seed):
        """
        输入:
            seed: 随机种子。
        输出:
            无
        作用: 设置随机种子。
        逻辑:
        为每个生成器设置不同的种子。
        """
        [
            rng_cpu.manual_seed(i + self.num_samples * seed)
            for i, rng_cpu in enumerate(self.rng_cpu)
        ]
        if th.cuda.is_available():
            [
                rng_cuda.manual_seed(i + self.num_samples * seed)
                for i, rng_cuda in enumerate(self.rng_cuda)
            ]
