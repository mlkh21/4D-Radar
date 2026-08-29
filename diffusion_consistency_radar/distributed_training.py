# -*- coding: utf-8 -*-
"""正式训练的单机 DDP 初始化、采样、聚合与运行身份辅助函数。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from typing import Dict, Iterator, Mapping, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, Sampler


DDP_PROTOCOL = "single_node_ddp_v1"


@dataclass(frozen=True)
class WorldBatchPlan:
    """描述每个进程批量、梯度累积和实际全局有效 batch。"""

    world_size: int
    per_rank_batch_size: int
    gradient_accumulation_steps: int
    effective_global_batch_size: int


def resolve_world_batch_plan(world_size: int) -> WorldBatchPlan:
    """返回正式 1--4 GPU 批量合同；3 卡需明确接受有效 batch 18。"""
    if type(world_size) is not int:
        raise TypeError(f"world_size 必须是 int，实际为 {world_size!r}")
    plans = {
        1: (2, 8),
        2: (1, 8),
        3: (1, 6),
        4: (1, 4),
    }
    if world_size not in plans:
        raise ValueError(f"正式训练仅支持单机 1--4 个进程，实际为 {world_size}")
    per_rank_batch_size, accumulation = plans[world_size]
    return WorldBatchPlan(
        world_size=world_size,
        per_rank_batch_size=per_rank_batch_size,
        gradient_accumulation_steps=accumulation,
        effective_global_batch_size=(
            world_size * per_rank_batch_size * accumulation
        ),
    )


def assert_distributed_config_compatible(
    context: "DistributedContext",
    *,
    per_rank_batch_size: int,
    gradient_accumulation_steps: int,
    configured_protocol=None,
    configured_world_size=None,
    configured_effective_global_batch_size=None,
) -> WorldBatchPlan:
    """核对配置声明与真实进程拓扑，拒绝静默改变有效全局 batch。"""
    for name, value in (
        ("per_rank_batch_size", per_rank_batch_size),
        ("gradient_accumulation_steps", gradient_accumulation_steps),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} 必须是正整数，实际为 {value!r}")
    if context.world_size > 1 and configured_protocol is None:
        raise ValueError("多进程训练必须显式配置 distributed_protocol")
    if configured_protocol is not None and configured_protocol != DDP_PROTOCOL:
        raise ValueError(
            "配置 distributed_protocol 不一致: "
            f"config={configured_protocol!r}, expected={DDP_PROTOCOL!r}"
        )
    if configured_protocol == DDP_PROTOCOL:
        expected_plan = resolve_world_batch_plan(context.world_size)
        if (
            per_rank_batch_size != expected_plan.per_rank_batch_size
            or gradient_accumulation_steps
            != expected_plan.gradient_accumulation_steps
        ):
            raise ValueError(
                "配置 batch/梯度累积与 DDP 协议不一致: "
                f"runtime=({per_rank_batch_size},"
                f"{gradient_accumulation_steps}), "
                f"expected=({expected_plan.per_rank_batch_size},"
                f"{expected_plan.gradient_accumulation_steps})"
            )
    if configured_world_size is not None:
        if type(configured_world_size) is not int or configured_world_size < 1:
            raise ValueError("配置 world_size 必须是正整数")
        if configured_world_size != context.world_size:
            raise ValueError(
                "配置 world_size 与 torchrun WORLD_SIZE 不一致: "
                f"config={configured_world_size}, runtime={context.world_size}"
            )
    effective_global_batch_size = (
        context.world_size
        * per_rank_batch_size
        * gradient_accumulation_steps
    )
    if configured_effective_global_batch_size is not None:
        if (
            type(configured_effective_global_batch_size) is not int
            or configured_effective_global_batch_size < 1
        ):
            raise ValueError("配置 effective_global_batch_size 必须是正整数")
        if configured_effective_global_batch_size != effective_global_batch_size:
            raise ValueError(
                "配置有效全局 batch 与实际运行不一致: "
                f"config={configured_effective_global_batch_size}, "
                f"runtime={effective_global_batch_size}"
            )
    return WorldBatchPlan(
        world_size=context.world_size,
        per_rank_batch_size=per_rank_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        effective_global_batch_size=effective_global_batch_size,
    )


@dataclass(frozen=True)
class DistributedContext:
    """显式携带当前进程 rank、设备和进程组状态。"""

    rank: int
    local_rank: int
    world_size: int
    device: Union[str, torch.device]
    initialized: bool

    @classmethod
    def single_process(
        cls, device: Union[str, torch.device] = "cpu"
    ) -> "DistributedContext":
        return cls(
            rank=0,
            local_rank=0,
            world_size=1,
            device=torch.device(device),
            initialized=False,
        )

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} 必须是整数，实际为 {raw!r}") from exc
    return value


def initialize_distributed(device_config: str = "cuda") -> DistributedContext:
    """按 torchrun 环境初始化单机 NCCL，并将进程绑定到 LOCAL_RANK。"""
    world_size = _env_int("WORLD_SIZE", 1)
    resolve_world_batch_plan(world_size)
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)
    if not 0 <= rank < world_size:
        raise RuntimeError(f"RANK={rank} 超出 WORLD_SIZE={world_size}")

    configured_device = torch.device(device_config or "cuda")
    if world_size == 1:
        if configured_device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("配置要求 CUDA，但当前进程无法访问 CUDA")
            device_index = configured_device.index or 0
            if device_index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"配置 CUDA device index={device_index} 超出可见 GPU 数量 "
                    f"{torch.cuda.device_count()}"
                )
            torch.cuda.set_device(device_index)
            configured_device = torch.device("cuda", device_index)
        return DistributedContext.single_process(configured_device)

    if configured_device.type != "cuda":
        raise RuntimeError("多进程正式训练只支持 NCCL/CUDA")
    if not torch.cuda.is_available():
        raise RuntimeError("torchrun 已启动多进程，但当前进程无法访问 CUDA")
    visible_device_count = torch.cuda.device_count()
    if not 0 <= local_rank < visible_device_count:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} 超出可见 GPU 数量 {visible_device_count}"
        )
    if dist.is_initialized():
        raise RuntimeError("当前进程已存在 torch.distributed 进程组，拒绝重复初始化")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=torch.device("cuda", local_rank),
        initialized=True,
    )


def cleanup_distributed(context: DistributedContext) -> None:
    """销毁本入口创建的进程组；异常退出时不额外等待其他 rank。"""
    if context.initialized and dist.is_initialized():
        dist.destroy_process_group()


def distributed_barrier(context: DistributedContext) -> None:
    """仅在多进程组已初始化时同步。"""
    if context.initialized:
        dist.barrier()


def wrap_model_for_ddp(
    model: nn.Module,
    context: DistributedContext,
    *,
    find_unused_parameters: bool = False,
) -> nn.Module:
    """多进程时包装模型；单进程保持原对象和历史 checkpoint 键。"""
    if not context.initialized:
        return model
    return DistributedDataParallel(
        model,
        device_ids=[context.local_rank],
        output_device=context.local_rank,
        broadcast_buffers=False,
        find_unused_parameters=find_unused_parameters,
    )


def unwrap_model(model: nn.Module) -> nn.Module:
    """去除 DDP/DataParallel 外壳，保持保存和业务方法接口稳定。"""
    while isinstance(model, (DistributedDataParallel, nn.DataParallel)):
        model = model.module
    return model


class DistributedEvalSampler(Sampler[int]):
    """无补齐地切分验证集，使全局每个样本恰好评估一次。"""

    def __init__(self, dataset: Dataset, *, num_replicas: int, rank: int):
        if type(num_replicas) is not int or num_replicas < 1:
            raise ValueError("num_replicas 必须是正整数")
        if type(rank) is not int or not 0 <= rank < num_replicas:
            raise ValueError("rank 必须位于 [0, num_replicas)")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self) -> int:
        remaining = len(self.dataset) - self.rank
        if remaining <= 0:
            return 0
        return (remaining + self.num_replicas - 1) // self.num_replicas


def reduce_named_sums(
    values: Mapping[str, Union[int, float]],
    context: DistributedContext,
) -> Dict[str, float]:
    """按固定键顺序对标量求全局和；单进程返回 float 副本。"""
    keys = sorted(values)
    tensor = torch.tensor(
        [float(values[key]) for key in keys],
        dtype=torch.float64,
        device=torch.device(context.device),
    )
    if context.initialized:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return {key: float(value) for key, value in zip(keys, tensor.cpu().tolist())}


def all_ranks_true(value: bool, context: DistributedContext) -> bool:
    """所有 rank 都为真时返回真，用于防止分支不一致导致 DDP 挂起。"""
    tensor = torch.tensor(
        1 if value else 0,
        dtype=torch.int32,
        device=torch.device(context.device),
    )
    if context.initialized:
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return bool(tensor.item())


def set_loader_epoch(loader, epoch: int) -> None:
    """通知训练 DistributedSampler 使用本 epoch 的确定性乱序。"""
    sampler = getattr(loader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def distributed_checkpoint_metadata(
    context: DistributedContext,
    batch_plan: WorldBatchPlan,
    *,
    train_dataset_size: int,
) -> Dict[str, Union[int, str]]:
    """构造可审计的进程拓扑、全局 batch 和训练补齐元数据。"""
    if train_dataset_size < 0:
        raise ValueError("train_dataset_size 不能为负数")
    padded_size = math.ceil(train_dataset_size / context.world_size) * context.world_size
    return {
        "protocol": DDP_PROTOCOL,
        "world_size": context.world_size,
        "per_rank_batch_size": batch_plan.per_rank_batch_size,
        "gradient_accumulation_steps": batch_plan.gradient_accumulation_steps,
        "effective_global_batch_size": batch_plan.effective_global_batch_size,
        "train_dataset_size": train_dataset_size,
        "train_sampler_padding": padded_size - train_dataset_size,
        "validation_sampler_padding": 0,
    }


def deterministic_noise_from_sample_ids(
    reference: torch.Tensor,
    sample_ids,
    *,
    seed: int,
) -> torch.Tensor:
    """按稳定样本身份生成噪声，使无补齐验证分片不改变模型输入。"""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed 必须是非负整数")
    if isinstance(sample_ids, str):
        sample_ids = [sample_ids]
    sample_ids = list(sample_ids)
    if len(sample_ids) != reference.shape[0]:
        raise ValueError("sample_ids 数量与 reference batch 不一致")
    parts = []
    for sample_id in sample_ids:
        digest = hashlib.sha256(str(sample_id).encode("utf-8")).digest()
        identity_seed = int.from_bytes(digest[:8], "big")
        generator = torch.Generator(device=reference.device)
        generator.manual_seed((seed + identity_seed) % (2**63 - 1))
        parts.append(
            torch.randn(
                reference.shape[1:],
                generator=generator,
                device=reference.device,
                dtype=reference.dtype,
            )
        )
    return torch.stack(parts, dim=0)


def assert_resume_distributed_compatible(
    checkpoint,
    *,
    expected_effective_global_batch_size: int,
) -> None:
    """允许 1/2/4 卡间恢复，但拒绝已记录的有效全局 batch 漂移。"""
    saved = checkpoint.get("distributed_training") if isinstance(checkpoint, dict) else None
    if saved is None:
        return
    if not isinstance(saved, dict) or saved.get("protocol") != DDP_PROTOCOL:
        raise ValueError("checkpoint distributed_training 协议无效")
    actual = saved.get("effective_global_batch_size")
    if type(actual) is not int or actual < 1:
        raise ValueError("checkpoint effective_global_batch_size 必须是正整数")
    if actual != expected_effective_global_batch_size:
        raise ValueError(
            "checkpoint 有效全局 batch 与当前运行不一致: "
            f"checkpoint={actual}, current={expected_effective_global_batch_size}"
        )
