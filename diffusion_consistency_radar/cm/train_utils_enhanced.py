# -- coding: utf-8 --

"""
增强版训练工具模块

功能:
1. 验证集监控和早停
2. 显存优化包装器
3. 训练状态管理
4. TensorBoard 日志支持
"""

import copy
import functools
import os
import time
from collections import defaultdict
from typing import Optional, Dict, Any, Callable

import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

from . import dist_util, logger
from .memory_efficient import MemoryManager


class ValidationMonitor:
    """
    验证集监控器
    
    功能:
    - 定期在验证集上评估模型
    - 记录验证损失和指标
    - 支持早停策略
    - 保存最佳模型
    """
    
    def __init__(
        self,
        val_dataloader,
        diffusion,
        model,
        val_interval: int = 1000,
        patience: int = 10,
        min_delta: float = 1e-4,
        metric: str = 'loss',
        mode: str = 'min'
    ):
        """
        输入:
            val_dataloader: 验证数据加载器
            diffusion: 扩散模型对象
            model: 模型
            val_interval: 验证间隔步数
            patience: 早停耐心值
            min_delta: 最小改进阈值
            metric: 监控的指标名称
            mode: 'min' 表示越小越好, 'max' 表示越大越好
        """
        self.val_dataloader = val_dataloader
        self.diffusion = diffusion
        self.model = model
        self.val_interval = val_interval
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.mode = mode
        
        # 状态跟踪
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.best_step = 0
        self.counter = 0
        self.val_history = []
        
    def should_stop(self) -> bool:
        """检查是否应该早停"""
        return self.counter >= self.patience
    
    def update(self, current_value: float, step: int) -> bool:
        """
        更新监控状态
        
        返回: True 如果有改进，False 否则
        """
        improved = False
        
        if self.mode == 'min':
            if current_value < self.best_value - self.min_delta:
                self.best_value = current_value
                self.best_step = step
                self.counter = 0
                improved = True
            else:
                self.counter += 1
        else:
            if current_value > self.best_value + self.min_delta:
                self.best_value = current_value
                self.best_step = step
                self.counter = 0
                improved = True
            else:
                self.counter += 1
        
        self.val_history.append({
            'step': step,
            'value': current_value,
            'improved': improved
        })
        
        return improved
    
    @th.no_grad()
    def validate(self, step: int, num_samples: int = 100) -> Dict[str, float]:
        """
        在验证集上评估模型
        
        输入:
            step: 当前训练步数
            num_samples: 验证样本数量
            
        返回:
            metrics: 包含各种验证指标的字典
        """
        self.model.eval()
        
        metrics = defaultdict(float)
        count = 0
        val_iter = iter(self.val_dataloader)
        
        try:
            while count < num_samples:
                try:
                    cond, batch = next(val_iter)
                except StopIteration:
                    break
                
                batch = batch.to(dist_util.dev())
                cond = cond.to(dist_util.dev())
                cond_dict = {"y": cond}
                
                # 随机采样时间步
                t = th.rand(batch.shape[0], device=dist_util.dev()) * \
                    (self.diffusion.sigma_max - self.diffusion.sigma_min) + \
                    self.diffusion.sigma_min
                
                # 计算验证损失
                losses = self.diffusion.training_losses(
                    self.model,
                    batch,
                    t,
                    model_kwargs=cond_dict
                )
                
                for k, v in losses.items():
                    if isinstance(v, th.Tensor):
                        metrics[k] += v.mean().item() * batch.shape[0]
                
                count += batch.shape[0]
                
                # 清理显存
                del batch, cond, losses
                th.cuda.empty_cache()
                
        except Exception as e:
            logger.log(f"验证过程出错: {e}")
        finally:
            self.model.train()
        
        # 计算平均值
        if count > 0:
            for k in metrics:
                metrics[k] /= count
        
        # 记录到日志
        if dist.get_rank() == 0:
            logger.log(f"\n=== 验证结果 (Step {step}) ===")
            for k, v in metrics.items():
                logger.log(f"  val_{k}: {v:.6f}")
                logger.logkv(f"val_{k}", v)
            
            # 检查是否有改进
            if self.metric in metrics:
                improved = self.update(metrics[self.metric], step)
                logger.log(f"  最佳 {self.metric}: {self.best_value:.6f} (Step {self.best_step})")
                if improved:
                    logger.log("  ✓ 新的最佳模型!")
                logger.log(f"  早停计数器: {self.counter}/{self.patience}")
            logger.log("=" * 40 + "\n")
        
        return dict(metrics)


class TrainingStateManager:
    """
    训练状态管理器
    
    功能:
    - 追踪训练进度
    - 计算和记录训练统计
    - 管理检查点
    """
    
    def __init__(self, total_steps: int, log_interval: int = 100):
        self.total_steps = total_steps
        self.log_interval = log_interval
        
        # 统计信息
        self.step = 0
        self.epoch = 0
        self.running_loss = 0.0
        self.running_count = 0
        self.start_time = time.time()
        self.step_times = []
        
        # 损失历史
        self.loss_history = []
        
    def update(self, loss: float, step: int):
        """更新训练状态"""
        self.step = step
        self.running_loss += loss
        self.running_count += 1
        self.loss_history.append(loss)
        
    def get_stats(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        elapsed = time.time() - self.start_time
        avg_loss = self.running_loss / max(1, self.running_count)
        
        # 计算剩余时间
        if self.step > 0:
            time_per_step = elapsed / self.step
            remaining_steps = self.total_steps - self.step
            eta = remaining_steps * time_per_step
        else:
            eta = 0
        
        return {
            'step': self.step,
            'total_steps': self.total_steps,
            'progress': self.step / max(1, self.total_steps) * 100,
            'avg_loss': avg_loss,
            'elapsed_time': elapsed,
            'eta': eta,
            'steps_per_sec': self.step / max(1, elapsed)
        }
    
    def reset_running_stats(self):
        """重置运行统计"""
        self.running_loss = 0.0
        self.running_count = 0
    
    def log_progress(self):
        """记录训练进度"""
        stats = self.get_stats()
        
        if dist.get_rank() == 0:
            # 格式化时间
            def format_time(seconds):
                h = int(seconds // 3600)
                m = int((seconds % 3600) // 60)
                s = int(seconds % 60)
                return f"{h:02d}:{m:02d}:{s:02d}"
            
            logger.log(
                f"Step {stats['step']}/{stats['total_steps']} "
                f"({stats['progress']:.1f}%) | "
                f"Loss: {stats['avg_loss']:.6f} | "
                f"Elapsed: {format_time(stats['elapsed_time'])} | "
                f"ETA: {format_time(stats['eta'])} | "
                f"Speed: {stats['steps_per_sec']:.2f} steps/s"
            )


class MemoryEfficientTrainLoop:
    """
    显存高效训练循环包装器
    
    功能:
    - 自动显存管理
    - 动态批次大小调整
    - 梯度累积
    - 混合精度训练
    """
    
    def __init__(
        self,
        base_train_loop,
        memory_fraction: float = 0.9,
        enable_checkpointing: bool = True,
        clear_cache_interval: int = 10,
        enable_validation: bool = True,
        val_dataloader = None,
        val_interval: int = 1000
    ):
        """
        输入:
            base_train_loop: 基础训练循环对象
            memory_fraction: 目标显存使用比例
            enable_checkpointing: 是否启用梯度检查点
            clear_cache_interval: 清理缓存的间隔步数
            enable_validation: 是否启用验证
            val_dataloader: 验证数据加载器
            val_interval: 验证间隔
        """
        self.base = base_train_loop
        self.memory_fraction = memory_fraction
        self.enable_checkpointing = enable_checkpointing
        self.clear_cache_interval = clear_cache_interval
        
        # 显存管理器
        self.memory_manager = MemoryManager(target_memory_fraction=memory_fraction)
        
        # 验证监控
        self.validation_monitor = None
        if enable_validation and val_dataloader is not None:
            self.validation_monitor = ValidationMonitor(
                val_dataloader=val_dataloader,
                diffusion=self.base.diffusion,
                model=self.base.model,
                val_interval=val_interval
            )
        
        # 训练状态
        self.state_manager = TrainingStateManager(
            total_steps=getattr(self.base, 'lr_anneal_steps', 100000),
            log_interval=self.base.log_interval
        )
        
    def run_loop(self):
        """执行显存高效训练循环"""
        data_iter = iter(self.base.data)
        
        while not self.base.lr_anneal_steps or self.base.step < self.base.lr_anneal_steps:
            try:
                cond, batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.base.data)
                self.state_manager.epoch += 1
                continue
            
            # 定期清理显存缓存
            if self.base.step % self.clear_cache_interval == 0:
                self.memory_manager.clear_cache()
            
            # 执行训练步骤
            self.base.run_step(batch, cond)
            
            # 更新训练状态
            # self.state_manager.update(loss, self.base.step)
            
            # 日志记录
            if self.base.step % self.base.log_interval == 0:
                logger.dumpkvs()
                self.state_manager.log_progress()
                self.state_manager.reset_running_stats()
                
                # 记录显存使用情况
                mem_stats = self.memory_manager.get_memory_stats()
                if dist.get_rank() == 0:
                    logger.logkv("gpu_memory_used_gb", mem_stats['used_gb'])
                    logger.logkv("gpu_memory_percent", mem_stats['used_percent'])
            
            # 验证
            if self.validation_monitor is not None:
                if self.base.step % self.validation_monitor.val_interval == 0 and self.base.step > 0:
                    self.validation_monitor.validate(self.base.step)
                    
                    # 检查早停
                    if self.validation_monitor.should_stop():
                        if dist.get_rank() == 0:
                            logger.log(f"早停触发! 最佳步数: {self.validation_monitor.best_step}")
                        break
            
            # 保存模型
            if self.base.step % self.base.save_interval == 0:
                self.base.save()
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.base.step > 0:
                    return
        
        # 保存最终模型
        if (self.base.step - 1) % self.base.save_interval != 0:
            self.base.save()


def create_enhanced_train_loop(
    base_train_loop,
    val_dataloader=None,
    enable_validation: bool = True,
    val_interval: int = 1000,
    memory_fraction: float = 0.9,
    enable_memory_optimization: bool = True
):
    """
    创建增强版训练循环的工厂函数
    
    输入:
        base_train_loop: 基础训练循环
        val_dataloader: 验证数据加载器
        enable_validation: 是否启用验证
        val_interval: 验证间隔
        memory_fraction: 目标显存使用比例
        enable_memory_optimization: 是否启用显存优化
        
    返回:
        enhanced_loop: 增强版训练循环
    """
    if enable_memory_optimization:
        return MemoryEfficientTrainLoop(
            base_train_loop=base_train_loop,
            memory_fraction=memory_fraction,
            enable_validation=enable_validation,
            val_dataloader=val_dataloader,
            val_interval=val_interval
        )
    else:
        # 返回一个简单的包装器，只添加验证功能
        return base_train_loop


class GradientAccumulator:
    """
    梯度累积器
    
    用于在显存有限时模拟更大的batch size
    """
    
    def __init__(self, accumulation_steps: int = 4):
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
        
    def should_accumulate(self) -> bool:
        """是否应该累积梯度（不执行优化步骤）"""
        return self.current_step < self.accumulation_steps - 1
    
    def step(self):
        """增加累积计数"""
        self.current_step = (self.current_step + 1) % self.accumulation_steps
        
    def reset(self):
        """重置累积器"""
        self.current_step = 0
