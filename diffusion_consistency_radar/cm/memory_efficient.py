# -- coding: utf-8 --
"""
显存高效模块 - 整体逻辑层面的显存优化策略

主要优化策略:
1. 动态分辨率调整 - 根据数据稀疏度动态调整处理分辨率
2. 稀疏感知处理 - 只处理非空体素区域
3. 分块处理 - 将大体素分块处理再合并
4. 渐进式训练 - 从低分辨率逐步提升
5. 内存池管理 - 复用张量减少分配

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import gc


class MemoryManager:
    """显存管理器 - 统一管理显存分配和释放"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.tensor_pool = {}
        self.enable_memory_efficient = True
        
    def clear_cache(self):
        """清理缓存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def get_memory_stats(self):
        """获取显存统计"""
        if not torch.cuda.is_available():
            return {}
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**3,
            'reserved': torch.cuda.memory_reserved() / 1024**3,
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**3,
        }
    
    def print_memory_stats(self, prefix=""):
        """打印显存统计"""
        stats = self.get_memory_stats()
        if stats:
            print(f"{prefix}GPU Memory: {stats['allocated']:.2f}GB allocated, "
                  f"{stats['reserved']:.2f}GB reserved, "
                  f"{stats['max_allocated']:.2f}GB peak")


class SparsityAwareProcessor:
    """
    稀疏感知处理器
    
    雷达数据通常非常稀疏（<10% 占用率），可以利用这一特性：
    1. 检测有效区域的边界框
    2. 只处理有效区域
    3. 恢复到原始尺寸
    """
    
    def __init__(self, min_occupancy: float = 0.01, padding: int = 4):
        """
        Args:
            min_occupancy: 最小占用率阈值
            padding: 边界填充
        """
        self.min_occupancy = min_occupancy
        self.padding = padding
    
    def get_valid_bbox(self, voxel: torch.Tensor) -> Tuple[slice, slice, slice]:
        """
        获取非空体素的边界框
        
        Args:
            voxel: (B, C, D, H, W) 体素数据
        Returns:
            d_slice, h_slice, w_slice: 各维度的有效范围
        """
        # 计算占用掩码 (假设第一个通道是占用率)
        occupancy = (voxel[:, 0:1] > 0).float()
        
        # 沿batch维度合并
        occupancy_any = occupancy.any(dim=0).squeeze()  # (D, H, W)
        
        # 找到各维度的有效范围
        d_valid = occupancy_any.any(dim=(1, 2))  # (D,)
        h_valid = occupancy_any.any(dim=(0, 2))  # (H,)
        w_valid = occupancy_any.any(dim=(0, 1))  # (W,)
        
        d_indices = torch.where(d_valid)[0]
        h_indices = torch.where(h_valid)[0]
        w_indices = torch.where(w_valid)[0]
        
        if len(d_indices) == 0 or len(h_indices) == 0 or len(w_indices) == 0:
            # 全空，返回原始尺寸
            return slice(None), slice(None), slice(None)
        
        D, H, W = occupancy_any.shape
        
        # 添加padding并限制范围
        d_start = max(0, d_indices[0].item() - self.padding)
        d_end = min(D, d_indices[-1].item() + self.padding + 1)
        h_start = max(0, h_indices[0].item() - self.padding)
        h_end = min(H, h_indices[-1].item() + self.padding + 1)
        w_start = max(0, w_indices[0].item() - self.padding)
        w_end = min(W, w_indices[-1].item() + self.padding + 1)
        
        return slice(d_start, d_end), slice(h_start, h_end), slice(w_start, w_end)
    
    def crop(self, voxel: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        """
        裁剪到有效区域
        
        Returns:
            cropped_voxel: 裁剪后的体素
            bbox_info: (original_shape, slices) 用于恢复
        """
        original_shape = voxel.shape
        d_slice, h_slice, w_slice = self.get_valid_bbox(voxel)
        
        cropped = voxel[:, :, d_slice, h_slice, w_slice]
        
        return cropped, (original_shape, (d_slice, h_slice, w_slice))
    
    def restore(self, cropped: torch.Tensor, bbox_info: Tuple) -> torch.Tensor:
        """恢复到原始尺寸"""
        original_shape, (d_slice, h_slice, w_slice) = bbox_info
        
        restored = torch.zeros(original_shape, device=cropped.device, dtype=cropped.dtype)
        restored[:, :, d_slice, h_slice, w_slice] = cropped
        
        return restored


class DynamicResolutionProcessor:
    """
    动态分辨率处理器
    
    根据数据复杂度动态调整处理分辨率:
    - 简单场景用低分辨率快速处理
    - 复杂场景用高分辨率精细处理
    """
    
    def __init__(
        self,
        base_resolution: Tuple[int, int, int] = (32, 128, 128),
        min_scale: float = 0.5,
        max_scale: float = 1.0,
        complexity_threshold: float = 0.1,
    ):
        self.base_resolution = base_resolution
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.complexity_threshold = complexity_threshold
    
    def compute_complexity(self, voxel: torch.Tensor) -> float:
        """计算数据复杂度 (基于占用率和空间分布)"""
        occupancy = (voxel[:, 0:1] > 0).float()
        occupancy_rate = occupancy.mean().item()
        
        # 计算空间分布的熵 (简化版)
        spatial_std = occupancy.std(dim=(2, 3, 4)).mean().item()
        
        complexity = occupancy_rate * (1 + spatial_std)
        return complexity
    
    def get_scale(self, voxel: torch.Tensor) -> float:
        """根据复杂度获取缩放因子"""
        complexity = self.compute_complexity(voxel)
        
        # 线性映射复杂度到缩放因子
        if complexity < self.complexity_threshold:
            scale = self.min_scale
        else:
            scale = min(
                self.max_scale,
                self.min_scale + (complexity / self.complexity_threshold) * (self.max_scale - self.min_scale)
            )
        
        return scale
    
    def resize(self, voxel: torch.Tensor, scale: float) -> torch.Tensor:
        """调整分辨率"""
        if scale == 1.0:
            return voxel
        
        return F.interpolate(
            voxel,
            scale_factor=(scale, scale, scale),
            mode='trilinear',
            align_corners=False,
        )
    
    def restore_resolution(self, voxel: torch.Tensor, target_shape: Tuple) -> torch.Tensor:
        """恢复到目标分辨率"""
        if voxel.shape[2:] == target_shape[2:]:
            return voxel
        
        return F.interpolate(
            voxel,
            size=target_shape[2:],
            mode='trilinear',
            align_corners=False,
        )


class ChunkedProcessor:
    """
    分块处理器
    
    将大体素分成小块处理，适用于超大分辨率场景
    """
    
    def __init__(
        self,
        chunk_size: Tuple[int, int, int] = (16, 64, 64),
        overlap: int = 4,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def split_into_chunks(self, voxel: torch.Tensor) -> List[Tuple[torch.Tensor, Tuple]]:
        """
        将体素分割成块
        
        Returns:
            List of (chunk, position_info)
        """
        B, C, D, H, W = voxel.shape
        cd, ch, cw = self.chunk_size
        
        chunks = []
        
        for d_start in range(0, D, cd - self.overlap):
            for h_start in range(0, H, ch - self.overlap):
                for w_start in range(0, W, cw - self.overlap):
                    d_end = min(d_start + cd, D)
                    h_end = min(h_start + ch, H)
                    w_end = min(w_start + cw, W)
                    
                    chunk = voxel[:, :, d_start:d_end, h_start:h_end, w_start:w_end]
                    position = (d_start, h_start, w_start, d_end, h_end, w_end)
                    
                    chunks.append((chunk, position))
        
        return chunks
    
    def merge_chunks(
        self,
        chunks: List[Tuple[torch.Tensor, Tuple]],
        original_shape: Tuple,
    ) -> torch.Tensor:
        """合并处理后的块"""
        B, C, D, H, W = original_shape
        
        result = torch.zeros(original_shape, device=chunks[0][0].device, dtype=chunks[0][0].dtype)
        weight = torch.zeros((1, 1, D, H, W), device=chunks[0][0].device)
        
        for chunk, (d_start, h_start, w_start, d_end, h_end, w_end) in chunks:
            result[:, :, d_start:d_end, h_start:h_end, w_start:w_end] += chunk
            weight[:, :, d_start:d_end, h_start:h_end, w_start:w_end] += 1
        
        # 平均重叠区域
        result = result / weight.clamp(min=1)
        
        return result


class ProgressiveTrainer:
    """
    渐进式训练调度器
    
    从低分辨率开始训练，逐步提升分辨率:
    1. 低分辨率快速收敛
    2. 中等分辨率精化
    3. 高分辨率微调
    """
    
    def __init__(
        self,
        base_resolution: Tuple[int, int, int] = (32, 128, 128),
        stages: List[Tuple[int, float]] = None,  # [(warmup_steps, scale), ...]
    ):
        self.base_resolution = base_resolution
        self.stages = stages or [
            (1000, 0.5),   # 前1000步用0.5x分辨率
            (3000, 0.75),  # 1000-3000步用0.75x分辨率
            (None, 1.0),   # 之后用全分辨率
        ]
        self.current_stage = 0
    
    def get_current_scale(self, step: int) -> float:
        """根据当前步数获取分辨率缩放"""
        cumulative_steps = 0
        
        for warmup_steps, scale in self.stages:
            if warmup_steps is None or step < cumulative_steps + warmup_steps:
                return scale
            cumulative_steps += warmup_steps
        
        return self.stages[-1][1]
    
    def get_current_resolution(self, step: int) -> Tuple[int, int, int]:
        """获取当前分辨率"""
        scale = self.get_current_scale(step)
        D, H, W = self.base_resolution
        return (
            max(8, int(D * scale)),
            max(16, int(H * scale)),
            max(16, int(W * scale)),
        )


class MemoryEfficientTrainingWrapper:
    """
    显存高效训练包装器
    
    整合所有显存优化策略
    """
    
    def __init__(
        self,
        enable_sparse_processing: bool = True,
        enable_dynamic_resolution: bool = True,
        enable_chunked_processing: bool = False,
        enable_progressive_training: bool = True,
        clear_cache_interval: int = 50,
    ):
        self.memory_manager = MemoryManager()
        self.sparse_processor = SparsityAwareProcessor() if enable_sparse_processing else None
        self.dynamic_processor = DynamicResolutionProcessor() if enable_dynamic_resolution else None
        self.chunked_processor = ChunkedProcessor() if enable_chunked_processing else None
        self.progressive_trainer = ProgressiveTrainer() if enable_progressive_training else None
        
        self.clear_cache_interval = clear_cache_interval
        self.step_count = 0
    
    def preprocess(
        self,
        batch: torch.Tensor,
        cond: torch.Tensor,
        step: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        预处理 - 应用显存优化
        
        Returns:
            processed_batch, processed_cond, restore_info
        """
        restore_info = {'original_shape': batch.shape}
        
        # 1. 渐进式分辨率
        if self.progressive_trainer is not None:
            scale = self.progressive_trainer.get_current_scale(step)
            if scale < 1.0:
                batch = F.interpolate(batch, scale_factor=scale, mode='trilinear', align_corners=False)
                cond = F.interpolate(cond, scale_factor=scale, mode='trilinear', align_corners=False)
                restore_info['progressive_scale'] = scale
        
        # 2. 稀疏感知裁剪
        if self.sparse_processor is not None:
            batch, bbox_info_batch = self.sparse_processor.crop(batch)
            cond, bbox_info_cond = self.sparse_processor.crop(cond)
            restore_info['bbox_batch'] = bbox_info_batch
            restore_info['bbox_cond'] = bbox_info_cond
        
        # 3. 动态分辨率
        if self.dynamic_processor is not None:
            scale = self.dynamic_processor.get_scale(batch)
            if scale < 1.0:
                batch = self.dynamic_processor.resize(batch, scale)
                cond = self.dynamic_processor.resize(cond, scale)
                restore_info['dynamic_scale'] = scale
                restore_info['pre_dynamic_shape'] = restore_info.get('pre_dynamic_shape', batch.shape)
        
        return batch, cond, restore_info
    
    def postprocess(
        self,
        output: torch.Tensor,
        restore_info: dict,
    ) -> torch.Tensor:
        """后处理 - 恢复原始尺寸"""
        
        # 按相反顺序恢复
        
        # 3. 恢复动态分辨率
        if 'dynamic_scale' in restore_info:
            output = self.dynamic_processor.restore_resolution(
                output, restore_info['pre_dynamic_shape']
            )
        
        # 2. 恢复稀疏裁剪
        if 'bbox_batch' in restore_info:
            output = self.sparse_processor.restore(output, restore_info['bbox_batch'])
        
        # 1. 恢复渐进式分辨率
        if 'progressive_scale' in restore_info:
            output = F.interpolate(
                output,
                size=restore_info['original_shape'][2:],
                mode='trilinear',
                align_corners=False,
            )
        
        return output
    
    def step(self):
        """训练步骤结束后调用"""
        self.step_count += 1
        
        if self.step_count % self.clear_cache_interval == 0:
            self.memory_manager.clear_cache()


def apply_memory_optimizations():
    """应用全局显存优化设置"""
    if torch.cuda.is_available():
        # 启用 TF32 加速（Ampere 架构）
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # 启用 cuDNN benchmark
        torch.backends.cudnn.benchmark = True
        
        # 设置显存分配策略
        import os
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


# 便捷函数
def create_memory_efficient_wrapper(
    gpu_memory_gb: float = 24.0,
) -> MemoryEfficientTrainingWrapper:
    """
    根据 GPU 显存大小创建优化包装器
    
    Args:
        gpu_memory_gb: GPU 显存大小 (GB)
    """
    if gpu_memory_gb < 12:
        # 小显存: 启用所有优化
        return MemoryEfficientTrainingWrapper(
            enable_sparse_processing=True,
            enable_dynamic_resolution=True,
            enable_chunked_processing=True,
            enable_progressive_training=True,
            clear_cache_interval=20,
        )
    elif gpu_memory_gb < 24:
        # 中等显存: 启用部分优化
        return MemoryEfficientTrainingWrapper(
            enable_sparse_processing=True,
            enable_dynamic_resolution=True,
            enable_chunked_processing=False,
            enable_progressive_training=True,
            clear_cache_interval=50,
        )
    else:
        # 大显存: 基本优化
        return MemoryEfficientTrainingWrapper(
            enable_sparse_processing=True,
            enable_dynamic_resolution=False,
            enable_chunked_processing=False,
            enable_progressive_training=False,
            clear_cache_interval=100,
        )
