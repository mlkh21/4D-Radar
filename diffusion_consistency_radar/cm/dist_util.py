"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
from mpi4py import MPI
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3


def setup_dist():
    """
    输入:
        无
    输出:
        无
    作用: 设置分布式进程组。
    逻辑:
    1. 检查是否已初始化。
    2. 获取 MPI 通信器。
    3. 确定后端 (gloo 或 nccl)。
    4. 设置 MASTER_ADDR, RANK, WORLD_SIZE。
    5. 广播端口。
    6. 初始化进程组。
    """
    if dist.is_initialized():
        return

    # os.environ["CUDA_VISIBLE_DEVICES"] = f"{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}"
    # os.environ["CUDA_VISIBLE_DEVICES"] = f"1, 3, 5"
    comm = MPI.COMM_WORLD
    backend = "gloo" if not th.cuda.is_available() else "nccl"
    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())
    os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    os.environ["RANK"] = str(comm.rank)
    os.environ["WORLD_SIZE"] = str(comm.size)

    port = comm.bcast(_find_free_port(), root=0)
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    输入:
        无
    输出:
        设备对象。
    作用: 获取 torch.distributed 使用的设备。
    逻辑:
    如果 cuda 可用，返回 cuda，否则返回 cpu。
    """
    if th.cuda.is_available():
        return th.device("cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    输入:
        path: 文件路径。
        **kwargs: torch.load 参数。
    输出:
        state_dict: 状态字典。
    作用: 加载 PyTorch 文件，避免跨 MPI rank 的冗余获取。
    逻辑:
    1. Rank 0 读取文件并广播数据块。
    2. 其他 Rank 接收数据块并重组。
    3. 使用 torch.load 加载数据。
    """
    chunk_size = 2**30  # MPI has a relatively small size limit
    if MPI.COMM_WORLD.Get_rank() == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
        num_chunks = len(data) // chunk_size
        if len(data) % chunk_size:
            num_chunks += 1
        MPI.COMM_WORLD.bcast(num_chunks)
        for i in range(0, len(data), chunk_size):
            MPI.COMM_WORLD.bcast(data[i : i + chunk_size])
    else:
        num_chunks = MPI.COMM_WORLD.bcast(None)
        data = bytes()
        for _ in range(num_chunks):
            data += MPI.COMM_WORLD.bcast(None)

    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    输入:
        params: 参数列表。
    输出:
        无
    作用: 从 rank 0 同步参数到其他 rank。
    逻辑:
    遍历参数并广播。
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    """
    输入:
        无
    输出:
        端口号。
    作用: 查找空闲端口。
    逻辑:
    绑定到端口 0 并获取分配的端口。
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
